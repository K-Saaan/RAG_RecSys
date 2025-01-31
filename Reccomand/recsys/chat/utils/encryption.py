# from cryptography.fernet import Fernet
# from typing import Union
# from config.settings import Settings

# class Encryptor:
#     def __init__(self):
#         self.cipher_suite = Fernet(Settings.ENCRYPTION_KEY)
    
#     def encrypt(self, data: str) -> bytes:
#         return self.cipher_suite.encrypt(data.encode())
    
#     def decrypt(self, encrypted_data: bytes) -> str:
#         return self.cipher_suite.decrypt(encrypted_data).decode()