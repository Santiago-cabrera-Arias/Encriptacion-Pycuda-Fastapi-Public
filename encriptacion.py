import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import binascii
from fastapi import HTTPException
import time

SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)


class AESKeyExpansion:
    def __init__(self):

        self.RCON = np.array([
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
        ], dtype=np.uint8)

    def expand_key(self, key):
        num_words = len(key) // 4
        num_round_keys = (num_words + 6) * 4

        round_keys = np.zeros((num_round_keys, 4), dtype=np.uint8)
        round_keys[:num_words] = np.array(key).reshape(num_words, 4)

        for i in range(num_words, num_round_keys):
            temp = round_keys[i - 1].copy()

            if i % num_words == 0:
                temp = np.roll(temp, -1)
                temp[0] = SBOX[temp[0]]
                temp ^= self.RCON[i // num_words]
            elif num_words > 6 and i % num_words == 4:
                temp = SBOX[temp]

            round_keys[i] = round_keys[i - num_words] ^ temp

        return round_keys
    

class AddRoundKey:
    def __init__(self):
        # Código del kernel
        self.mod_add_round_key = SourceModule("""
            __global__ void aes_addroundkey_kernel(uint8_t *block, uint8_t *round_key, int block_size) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                // Convertir tid a índice de bloque
                int block_idx = tid / 16;

                // Calcular el índice de inicio del bloque en el array
                int block_start = block_idx * 16;

                // Calcular el índice local dentro del bloque
                int local_idx = tid % 16;

                // Asegurarse de que estamos dentro de los límites del array
                if (block_start < block_size) {
                    // Realizar la operación de XOR con la clave de ronda
                    block[block_start + local_idx] ^= round_key[local_idx];
                }
            }
        """)

        # Obtener la función del kernel
        self.aes_addroundkey_kernel = self.mod_add_round_key.get_function("aes_addroundkey_kernel")        

    def add_round_key(self, state, round_key):
        """
        Aplica la operación AddRoundKey a un estado utilizando una clave de ronda.
        Ambos deben ser matrices de 4x4 de bytes (np.uint8).
        """
        # Convertir la entrada a un numpy array si no es del tipo correcto
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.uint8)

        # Aplanar la matriz para pasarla al kernel
        flat_state = state.flatten()

        # Copiar la matriz a la GPU
        d_state = cuda.to_device(flat_state)

        # Copiar la clave de ronda a la GPU
        d_round_key = cuda.to_device(np.array(round_key, dtype=np.uint8))

        # Configurar la grilla y el bloque
        block_size = len(flat_state)
        grid_size = 1

        # Llamar al kernel
        self.aes_addroundkey_kernel(d_state, d_round_key, np.int32(block_size), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Copiar la matriz resultante de nuevo a la CPU
        cuda.memcpy_dtoh(flat_state, d_state)

        # Reformar la matriz resultante
        output_state = flat_state.reshape(state.shape)

        return output_state
    

class AESSubBytes:
    def __init__(self):

        # Cargar el kernel CUDA para SubBytes
        mod = SourceModule("""
        __global__ void aes_subbytes_kernel(uint8_t *block, int block_size, uint8_t *SBOX) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < block_size) {
                // Sustituir cada byte utilizando la S-Box
                block[tid] = SBOX[block[tid]];
            }
        }
        """)

        # Obtener el kernel
        self.aes_subbytes_kernel = mod.get_function("aes_subbytes_kernel")

    def sub_bytes(self, data):
        # Tamaño del bloque
        block_size = len(data)

        # Copiar datos al dispositivo
        data_gpu = cuda.to_device(data)
        SBOX_gpu = cuda.to_device(SBOX)

        # Definir la configuración de bloques y malla
        block_size_x = 128
        grid_size_x = (block_size + block_size_x - 1) // block_size_x

        # Llamar al kernel SubBytes
        self.aes_subbytes_kernel(data_gpu, np.int32(block_size), SBOX_gpu, block=(block_size_x, 1, 1), grid=(grid_size_x, 1))

        # Copiar los resultados de vuelta al host
        cuda.memcpy_dtoh(data, data_gpu)

        return data

class AESShiftRows:
    def __init__(self):
        # Definir el kernel para ShiftRows
        mod_shift_rows = SourceModule("""
            __global__ void shiftRows(int *matrix)
            {
                int tid = threadIdx.x + blockDim.x * blockIdx.x;

                if (tid < 4) {
                    int temp[4];

                    // Almacenar la fila actual en un array temporal
                    for (int i = 0; i < 4; i++) {
                        temp[i] = matrix[tid * 4 + i];
                    }

                    // Aplicar el ShiftRows a la fila actual
                    for (int i = 0; i < 4; i++) {
                        matrix[tid * 4 + i] = temp[(i + tid) % 4];
                    }
                }
            }
        """)

        # Obtener la función del kernel
        self.shift_rows_kernel = mod_shift_rows.get_function("shiftRows")

    def shift_rows(self, matrix):
        # Aplanar la matriz para pasarla al kernel
        flat_matrix = matrix.flatten()

        # Copiar la matriz a la GPU
        d_matrix = cuda.to_device(flat_matrix)

        # Configurar la grilla y el bloque
        block_size = len(flat_matrix)
        grid_size = 1

        # Llamar al kernel
        self.shift_rows_kernel(d_matrix, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Copiar la matriz resultante de nuevo a la CPU
        cuda.memcpy_dtoh(flat_matrix, d_matrix)

        # Reformar la matriz resultante
        output_matrix = flat_matrix.reshape(matrix.shape)

        return output_matrix

class AESMixin:
    def __init__(self):
        # Definir el kernel para MixColumns
        mix_columns_kernel = """
        __device__ unsigned char multiply(unsigned char a, unsigned char b) {
            unsigned char result = 0;
            for (int i = 0; i < 8; ++i) {
                if (b & 1) {
                    result ^= a;
                }
                unsigned char carry = (a & 0x80) != 0;
                a <<= 1;
                if (carry) {
                    a ^= 0x1B; // AES irreducible polynomial
                }
                b >>= 1;
            }
            return result;
        }

        __global__ void mix_columns_kernel(unsigned char *state) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;

            // Cada columna se procesa por separado
            for (int col = 0; col < 4; ++col) {
                int base_idx = col * 4;

                unsigned char a0 = state[base_idx];
                unsigned char a1 = state[base_idx + 1];
                unsigned char a2 = state[base_idx + 2];
                unsigned char a3 = state[base_idx + 3];

                state[base_idx] = multiply(a0, 2) ^ multiply(a1, 3) ^ a2 ^ a3;
                state[base_idx + 1] = a0 ^ multiply(a1, 2) ^ multiply(a2, 3) ^ a3;
                state[base_idx + 2] = a0 ^ a1 ^ multiply(a2, 2) ^ multiply(a3, 3);
                state[base_idx + 3] = multiply(a0, 3) ^ a1 ^ a2 ^ multiply(a3, 2);
            }
        }
        """

        # Compilar el kernel
        self.mod = SourceModule(mix_columns_kernel)

        # Obtener la función del kernel
        self.mix_columns_func = self.mod.get_function("mix_columns_kernel")

    def mix_columns(self, data):
        data_gpu = cuda.to_device(data)

        # Configurar la ejecución del kernel
        block_size = 4  # Tamaño del bloque
        grid_size = 1   # Tamaño de la cuadrícula

        # Llamar al kernel
        self.mix_columns_func(data_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Obtener los resultados de la GPU
        cuda.memcpy_dtoh(data, data_gpu)
        
        return data.tolist()
    
def pkcs7_padding(text, block_size):
    padding_size = block_size - len(text) % block_size
    padding = bytes([padding_size] * padding_size)
    return text + padding


class AESCipher:
    def __init__(self):
        self.key_expander = AESKeyExpansion()
        self.sub_bytes = AESSubBytes()
        self.shift_rows = AESShiftRows()
        self.mix_columns = AESMixin()
        self.add_round_key = AddRoundKey()

    def encrypt_block(self, data_block, key):

        round_keys = self.key_expander.expand_key(key)
        print("Initial Round Key:", round_keys[0])
        data_block = self.add_round_key.add_round_key(data_block, round_keys[0])

        for i in range(0, 10):
            print(f"Round {i + 1} Key:", round_keys[i])
            data_block = self.sub_bytes.sub_bytes(data_block)
            data_block = self.shift_rows.shift_rows(data_block)
            data_block = self.mix_columns.mix_columns(data_block)
            data_block = self.add_round_key.add_round_key(data_block, round_keys[i])

        data_block = self.sub_bytes.sub_bytes(data_block)
        data_block = self.shift_rows.shift_rows(data_block)
        data_block = self.add_round_key.add_round_key(data_block, round_keys[10])

        encrypted_block = data_block


        return encrypted_block


    def encrypt_text(self, plain_text, key):


        padded_text = pkcs7_padding(plain_text.encode('utf-8'), 16)
        data_blocks = [np.frombuffer(padded_text[i:i+16], dtype=np.uint8).reshape(4, 4) for i in range(0, len(padded_text), 16)]

        encrypted_blocks = []

        start_time = time.time()

        for block in data_blocks:
            encrypted_block = self.encrypt_block(block, key)
            encrypted_blocks.append(encrypted_block.tolist())  # Convertir NumPy array a lista

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total encryption time: {total_time} seconds")

        return encrypted_blocks
