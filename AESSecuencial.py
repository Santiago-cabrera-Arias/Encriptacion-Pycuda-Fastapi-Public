from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import time

def generate_key():
    return get_random_bytes(16)  # Clave de 128 bits (16 bytes)

def encrypt(plain_text, key):
    cipher = AES.new(key, AES.MODE_CBC)
    start_time = time.time()
    cipher_text = cipher.encrypt(pad(plain_text.encode('utf-8'), AES.block_size))
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convertir a milisegundos
    print(f'Tiempo de cifrado: {elapsed_time:.4f} milisegundos')
    return cipher.iv + cipher_text

def decrypt(cipher_text, key):
    iv = cipher_text[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    start_time = time.time()
    decrypted_text = unpad(cipher.decrypt(cipher_text[AES.block_size:]), AES.block_size)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convertir a milisegundos
    print(f'Tiempo de descifrado: {elapsed_time:.4f} milisegundos')
    return decrypted_text.decode('utf-8')

# Ejemplo de uso:
key = generate_key()
text_to_encrypt = "Hola, este es un ejemplo de AES."

encrypted_text = encrypt(text_to_encrypt, key)
print(f'Texto cifrado: {encrypted_text}')

decrypted_text = decrypt(encrypted_text, key)
print(f'Texto descifrado: {decrypted_text}')
