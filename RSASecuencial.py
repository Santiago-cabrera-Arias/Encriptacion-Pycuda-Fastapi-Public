from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import time

def generate_key_pair():
    start_time = time.time()
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    end_time = time.time()
    print(f'Tiempo de generación de claves: {(end_time - start_time) * 1000:.4f} milisegundos')

    public_key = private_key.public_key()
    return private_key, public_key

def encrypt_message(message, public_key):
    start_time = time.time()
    ciphertext = public_key.encrypt(
        message.encode('utf-8'),
        padding.PKCS1v15()
    )
    end_time = time.time()
    print(f'Tiempo de cifrado: {(end_time - start_time) * 1000:.4f} milisegundos')

    return ciphertext

def decrypt_message(ciphertext, private_key):
    start_time = time.time()
    plaintext = private_key.decrypt(
        ciphertext,
        padding.PKCS1v15()
    )
    end_time = time.time()
    print(f'Tiempo de descifrado: {(end_time - start_time) * 1000:.4f} milisegundos')

    return plaintext.decode('utf-8')

# Ejemplo de uso:
private_key, public_key = generate_key_pair()

message_to_encrypt = "Hola, este es un ejemplo de RSA."

ciphertext = encrypt_message(message_to_encrypt, public_key)
print(f'Texto cifrado: {ciphertext}')

decrypted_text = decrypt_message(ciphertext, private_key)
print(f'Texto descifrado: {decrypted_text}')
