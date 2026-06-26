import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="google")

import firebase_admin
from firebase_admin import credentials, auth, firestore

_app = None
_db = None


def init_firebase():
    global _app, _db
    if _app is not None:
        return

    from config import (
        FIREBASE_PROJECT_ID, FIREBASE_PRIVATE_KEY,
        FIREBASE_PRIVATE_KEY_ID, FIREBASE_CLIENT_EMAIL, FIREBASE_CLIENT_ID,
    )

    if FIREBASE_PRIVATE_KEY and FIREBASE_CLIENT_EMAIL:
        cred_dict = {
            "type": "service_account",
            "project_id": FIREBASE_PROJECT_ID,
            "private_key_id": FIREBASE_PRIVATE_KEY_ID,
            "private_key": FIREBASE_PRIVATE_KEY,
            "client_email": FIREBASE_CLIENT_EMAIL,
            "client_id": FIREBASE_CLIENT_ID,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{FIREBASE_CLIENT_EMAIL.replace('@', '%40')}",
            "universe_domain": "googleapis.com",
        }
        cred = credentials.Certificate(cred_dict)
    elif os.path.exists("firebase-key.json"):
        cred = credentials.Certificate("firebase-key.json")
    else:
        raise RuntimeError("No Firebase credentials found")

    _app = firebase_admin.initialize_app(cred)
    _db = firestore.client(database_id="default")


def get_db():
    if _db is None:
        init_firebase()
    return _db


async def verify_token(authorization: str) -> dict:
    if not authorization.startswith("Bearer "):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid or expired token")
