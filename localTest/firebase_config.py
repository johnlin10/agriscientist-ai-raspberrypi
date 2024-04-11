import firebase_admin
from firebase_admin import credentials, firestore, storage


# 初始化 Firebase 应用程序的函数
def initialize_firebase():
    cred = credentials.Certificate("../serviceAccountKey.json")
    firebase_admin.initialize_app(
        cred, {"storageBucket": "agriscientist-ai.appspot.com"}
    )


# 获取 Firestore 客户端的函数
def get_firestore_client():
    return firestore.client()


# 获取 Storage 客户端的函数
def get_storage_client():
    return storage.bucket()
