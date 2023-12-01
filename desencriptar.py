import pycuda.autoinit
from encriptacion import AESCipher
from encriptacion import AESKeyExpansion
from encriptacion import AddRoundKey
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import binascii
import time

#Inversa de SubBytes
class AESInvSubBytes:
    def __init__(self):
        # Definir la S-Box inversa
        self.INV_SBOX = np.array([
            0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
            0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
            0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
            0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
            0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
            0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
            0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
            0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
            0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
            0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
            0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
            0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
            0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
            0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
            0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
        ], dtype=np.uint8)

        # Cargar el kernel CUDA para InvSubBytes
        mod = SourceModule("""
        __global__ void aes_invsubbytes_kernel(uint8_t *block, int block_size, uint8_t *INV_SBOX) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (tid < block_size) {
                // Sustituir cada byte utilizando la S-Box inversa
                block[tid] = INV_SBOX[block[tid]];
            }
        }
        """)

        # Obtener el kernel
        self.aes_invsubbytes_kernel = mod.get_function("aes_invsubbytes_kernel")

    def inv_sub_bytes(self, data):
        # Tamaño del bloque
        block_size = len(data)

        # Copiar datos al dispositivo
        data_gpu = cuda.to_device(data)
        INV_SBOX_gpu = cuda.to_device(self.INV_SBOX)

        # Definir la configuración de bloques y malla
        block_size_x = 128
        grid_size_x = (block_size + block_size_x - 1) // block_size_x

        # Llamar al kernel InvSubBytes
        self.aes_invsubbytes_kernel(data_gpu, np.int32(block_size), INV_SBOX_gpu, block=(block_size_x, 1, 1), grid=(grid_size_x, 1))

        # Copiar los resultados de vuelta al host
        cuda.memcpy_dtoh(data, data_gpu)

        return data

# Inversa de ShiftRows


class AESInvShiftRows:
    def __init__(self):
        # Definir el kernel para InvShiftRows
        mod_inv_shift_rows = SourceModule("""
            __global__ void invShiftRows(int *matrix)
            {
                int tid = threadIdx.x + blockDim.x * blockIdx.x;

                if (tid < 4) {
                    int temp[4];

                    // Almacenar la fila actual en un array temporal
                    for (int i = 0; i < 4; i++) {
                        temp[i] = matrix[tid * 4 + i];
                    }

                    // Aplicar el InvShiftRows a la fila actual
                    for (int i = 0; i < 4; i++) {
                        matrix[tid * 4 + i] = temp[(i - tid + 4) % 4];
                    }
                }
            }
        """)

        # Obtener la función del kernel
        self.inv_shift_rows_kernel = mod_inv_shift_rows.get_function("invShiftRows")

    def inv_shift_rows(self, matrix):
        # Convertir la entrada a un numpy array si no es del tipo correcto
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=np.uint8)

        # Aplanar la matriz para pasarla al kernel
        flat_matrix = matrix.flatten()

        # Copiar la matriz a la GPU
        d_matrix = cuda.to_device(flat_matrix)

        # Configurar la grilla y el bloque
        block_size = len(flat_matrix)
        grid_size = 1

        # Llamar al kernel
        self.inv_shift_rows_kernel(d_matrix, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Copiar la matriz resultante de nuevo a la CPU
        cuda.memcpy_dtoh(flat_matrix, d_matrix)

        # Reformar la matriz resultante
        output_matrix = flat_matrix.reshape(matrix.shape)

        return output_matrix

# Inversa de Mix Columns


class AESInvMixin:
    def __init__(self):
        # Definir el kernel para InvMixColumns
        inv_mix_columns_kernel = """
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

        __global__ void inv_mix_columns_kernel(unsigned char *state) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;

            // Cada columna se procesa por separado
            for (int col = 0; col < 4; ++col) {
                int base_idx = col * 4;

                unsigned char a0 = state[base_idx];
                unsigned char a1 = state[base_idx + 1];
                unsigned char a2 = state[base_idx + 2];
                unsigned char a3 = state[base_idx + 3];

                state[base_idx] = multiply(a0, 0x0E) ^ multiply(a1, 0x0B) ^ multiply(a2, 0x0D) ^ multiply(a3, 0x09);
                state[base_idx + 1] = multiply(a0, 0x09) ^ multiply(a1, 0x0E) ^ multiply(a2, 0x0B) ^ multiply(a3, 0x0D);
                state[base_idx + 2] = multiply(a0, 0x0D) ^ multiply(a1, 0x09) ^ multiply(a2, 0x0E) ^ multiply(a3, 0x0B);
                state[base_idx + 3] = multiply(a0, 0x0B) ^ multiply(a1, 0x0D) ^ multiply(a2, 0x09) ^ multiply(a3, 0x0E);
            }
        }
        """

        # Compilar el kernel
        self.mod = SourceModule(inv_mix_columns_kernel)

        # Obtener la función del kernel
        self.inv_mix_columns_func = self.mod.get_function("inv_mix_columns_kernel")

    def inv_mix_columns(self, data):
        data_gpu = cuda.to_device(data)

        # Configurar la ejecución del kernel
        block_size = 4  # Tamaño del bloque
        grid_size = 1   # Tamaño de la cuadrícula

        # Llamar al kernel
        self.inv_mix_columns_func(data_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))

        # Obtener los resultados de la GPU
        cuda.memcpy_dtoh(data, data_gpu)
        
        return data.tolist()

class AESDecipher:

    def __init__(self):
        self.key_expander = AESKeyExpansion()
        self.inv_sub_bytes = AESInvSubBytes()
        self.inv_shift_rows = AESInvShiftRows()
        self.inv_mix_columns = AESInvMixin()
        self.add_round_key = AddRoundKey()

    def decrypt_block(self, encrypted_block, key):

        start_time = time.time()

        round_keys = self.key_expander.expand_key(key)
        encrypted_block = self.add_round_key.add_round_key(encrypted_block, round_keys[9])

        for i in range(9, -1, -1):
            encrypted_block = self.add_round_key.add_round_key(encrypted_block, round_keys[i])
            encrypted_block = self.inv_mix_columns.inv_mix_columns(encrypted_block)
            encrypted_block = self.inv_shift_rows.inv_shift_rows(encrypted_block)
            encrypted_block = self.inv_sub_bytes.inv_sub_bytes(encrypted_block)

        encrypted_block = self.inv_sub_bytes.inv_sub_bytes(encrypted_block)
        encrypted_block = self.inv_shift_rows.inv_shift_rows(encrypted_block)
        encrypted_block = self.add_round_key.add_round_key(encrypted_block, round_keys[0])


        decrypted_block = encrypted_block

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total encryption time: {total_time} seconds")

        return decrypted_block

    def decrypt_text(self, encrypted_text, key):
        decrypted_blocks = []

        for block in encrypted_text:
            decrypted_block = self.decrypt_block(block, key)
            decrypted_blocks.append(decrypted_block.flatten())

        print("dasda",decrypted_blocks)
        return decrypted_blocks
    
    def remove_pkcs7_padding(self, block):
        padding_size = int(block[-1])
        return block[:-padding_size]
    