import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from fastapi import FastAPI
from encriptacion import AESCipher
from typing import List
from fastapi import FastAPI, HTTPException
from desencriptar import AESDecipher
from pydantic import BaseModel
import numpy as np
from encriptacion import AESCipher
from encriptacion import AESKeyExpansion
from encriptacion import AddRoundKey
from RSADesencriptado import RSADecryptor
from RSAEncriptado import RSAEncryptor
from fastapi.middleware.cors import CORSMiddleware



aes_cipher = AESCipher()
aes_decipher = AESDecipher()
# rsa_encryptor = RSAEncryptor()
# rsa_desencriptado = RSADecryptor()
app = FastAPI()

# Configura CORS para permitir solicitudes desde todos los orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Reemplaza con la lista de orígenes permitidos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/encrypt")
async def encrypt_text(plain_text: str, key_str: str):

    try:
        # Convertir la clave de texto a una lista de enteros
        key_bytes = [ord(char) for char in key_str]

        # Cifrar la frase
        aes_cipher = AESCipher()  # Asegúrate de tener una instancia de AESCipher
        encrypted_text = aes_cipher.encrypt_text(plain_text, key_bytes)
        print("Texto cifrado:", encrypted_text)

        # Descifrar la frase
        aes_decipher = AESDecipher()  # Asegúrate de tener una instancia de AESDecipher
        decrypted_text = aes_decipher.decrypt_text(encrypted_text, key_bytes)

        # Convertir los bloques descifrados a texto
        decoded_text = "".join([chr(b) for block in decrypted_text for b in block])
        print("Frase descifrada:", decoded_text)

        # Devolver resultados
        return {"encrypted_text": encrypted_text, "decoded_text": decoded_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class EncryptedData(BaseModel):
    encrypted_text: List[List[List[int]]]

@app.post("/decrypt")
async def decrypt_data(key_str: str , encrypted_data: EncryptedData):
    try:
   
        encrypted_text = encrypted_data.encrypted_text
        # encrypted_text = [[[19,219,193,169],[12,12,12,12],[12,12,12,12],[12,12,12,12]]]

        # Convertir la clave de texto a una lista de enteros
        key_bytes = [ord(char) for char in key_str]

        aes_decipher = AESDecipher()

        # Descifrar la frase
        decrypted_text = aes_decipher.decrypt_text(encrypted_text, key_bytes)

        # Mostrar resultados
        print("Frase descifrada:")
        decoded_text = "".join([chr(b) for block in decrypted_text for b in block])
        print(decoded_text)

        return {"encrypted_text": decoded_text}


    except Exception as e:
        # Maneja cualquier excepción y devuelve una respuesta HTTP con el error
        raise HTTPException(status_code=500, detail=f"Error during decryption: {str(e)}")
    

# Instancia de RSAEncryptor con parámetros e y N
rsa_encryptor = RSAEncryptor(e=65537, N=3233)

@app.post("/encryptRSA")
async def encrypt_message(message: str):
    # Obtener el mensaje cifrado
    encrypted_message = rsa_encryptor.encrypt_message(message)

    # Devolver el resultado como JSON
    return {"original_message": message, "encrypted_message": encrypted_message.tolist()}

class DecryptRequest(BaseModel):
    result_host: List[int]
    d: int
    N: int

@app.post("/decryptRSA")
async def decrypt_rsa(request: DecryptRequest):
    try:
        # Crear una instancia de RSADecryptor con los parámetros proporcionados
        decryptor = RSADecryptor(d=request.d, N=request.N)

        # Desencriptar el texto cifrado
        decrypted_result = decryptor.decrypt_cipher(np.array(request.result_host, dtype=np.int32))

        # Convertir los valores decifrados a una cadena de texto
        decrypted_text = ''.join([chr(int(char)) for char in decrypted_result])

        return {"decrypted_message": decrypted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RSA decryption: {str(e)}")