import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

class RSADecryptor:
    def __init__(self, d, N):
        # Definir el kernel de desencriptación
        mod_decrypt = SourceModule("""
            __device__ unsigned long long int mod_pow(unsigned long long int base, unsigned long long int exponent, unsigned long long int modulus) {
                unsigned long long int result = 1;
                base = base % modulus;

                while (exponent > 0) {
                    if (exponent % 2 == 1) {
                        result = (result * base) % modulus;
                    }

                    exponent = exponent >> 1;
                    base = (base * base) % modulus;
                }

                return result;
            }

            __global__ void rsa_decrypt_kernel(int *cipher, int d, int N, int *result, int length) {
                int i = threadIdx.x;
                if (i < length) {
                    result[i] = mod_pow(cipher[i], d, N);
                }
            }
        """)

        # Obtener la función del kernel de desencriptación
        self.rsa_decrypt_kernel = mod_decrypt.get_function("rsa_decrypt_kernel")

        # Parámetros d y N
        self.d = d
        self.N = N

    def decrypt_cipher(self, cipher_text):
        # Crear arrays de GPU y copiar datos desde el host
        cipher_gpu = cuda.to_device(cipher_text)
        decrypted_gpu = cuda.mem_alloc(cipher_text.nbytes)

        # Llamar al kernel de desencriptación
        block_size = len(cipher_text)
        self.rsa_decrypt_kernel(cipher_gpu, np.int32(self.d), np.int32(self.N), decrypted_gpu, np.int32(block_size), block=(block_size, 1, 1), grid=(1, 1))

        # Copiar el resultado de vuelta al host
        decrypted_host = np.empty_like(cipher_text)
        cuda.memcpy_dtoh(decrypted_host, decrypted_gpu)

        return decrypted_host

# # Mensaje cifrado obtenido después de la encriptación
# result_host = np.array([3000, 2185, 745, 1632, 678, 1992, 1230, 2185, 487, 1992, 2923, 2185, 690, 2160, 2825], dtype=np.int32)

# # Crear una instancia de RSADecryptor con los parámetros d y N
# rsa_decryptor = RSADecryptor(d=2753, N=3233)

# # Obtener el mensaje descifrado
# decrypted_message = rsa_decryptor.decrypt_cipher(result_host)

# # Mostrar resultados
# print(f"\nMensaje cifrado: {result_host}")
# print(f"\nMensaje descifrado: {decrypted_message}")

# # Convertir los valores ASCII a caracteres
# decrypted_text = ''.join(chr(char) for char in decrypted_message)

# # Mostrar el mensaje original descifrado
# print(f"\nMensaje original descifrado: {decrypted_text}")
