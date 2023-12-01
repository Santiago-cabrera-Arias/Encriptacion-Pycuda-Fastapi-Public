import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

class RSAEncryptor:
    def __init__(self, e, N):
        # Definir el kernel
        self.mod = SourceModule("""
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

            __global__ void rsa_encrypt_kernel(int *message, int e, int N, int *result, int length) {
                int i = threadIdx.x;
                if (i < length) {
                    result[i] = mod_pow(message[i], e, N);
                }
            }
        """)

        # Obtener la función del kernel
        self.rsa_encrypt_kernel = self.mod.get_function("rsa_encrypt_kernel")

        # Parámetros e y N
        self.e = e
        self.N = N

    def encrypt_message(self, message_to_encrypt):
        # Convertir el mensaje a una lista de enteros
        message_int = [ord(char) for char in message_to_encrypt]

        # Crear arrays de GPU y copiar datos desde el host
        message_host = np.array(message_int, dtype=np.int32)
        message_gpu = cuda.to_device(message_host)
        result_gpu = cuda.mem_alloc(message_host.nbytes)

        # Llamar al kernel
        block_size = len(message_int)
        self.rsa_encrypt_kernel(message_gpu, np.int32(self.e), np.int32(self.N), result_gpu, np.int32(block_size), block=(block_size, 1, 1), grid=(1, 1))

        # Copiar el resultado de vuelta al host
        result_host = np.empty_like(message_host)
        cuda.memcpy_dtoh(result_host, result_gpu)

        return result_host
