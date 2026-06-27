import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Arko007/toxipredict-gnn-models")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SPACE_TOKEN = os.getenv("HF_SPACE_TOKEN", "")
MODEL_CACHE_DIR = os.getenv("MODEL_DIR", "/model")

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
FIREBASE_PRIVATE_KEY_ID = os.getenv("FIREBASE_PRIVATE_KEY_ID", "")
FIREBASE_CLIENT_EMAIL = os.getenv("FIREBASE_CLIENT_EMAIL", "")
FIREBASE_CLIENT_ID = os.getenv("FIREBASE_CLIENT_ID", "")

FIREBASE_API_KEY = "AIzaSyDU4EEHT3HEvKNPOrpglLdF3y5Tfs6qy4E"
FIREBASE_AUTH_DOMAIN = "plant-cloud-cd461.firebaseapp.com"
FIREBASE_PROJECT_ID_WEB = "plant-cloud-cd461"

_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://*.vercel.app")
CORS_ORIGINS = [o for o in _cors_raw.split(",") if "*" not in o]
CORS_ORIGIN_REGEX = "|".join(
    o.strip().replace(".", "\\.").replace("*", ".*") + "$"
    for o in _cors_raw.split(",")
    if "*" in o
) or None

NUM_TASKS = 10
TASK_NAMES = [
    "NR-AR", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
    "SR-MMP", "SR-p53",
]

TASK_CLASSES = {
    "NR-AR": "Nuclear Receptor", "NR-AhR": "Nuclear Receptor",
    "NR-Aromatase": "Nuclear Receptor", "NR-ER": "Nuclear Receptor",
    "NR-PPAR-gamma": "Nuclear Receptor", "SR-ARE": "Stress Response",
    "SR-ATAD5": "Stress Response", "SR-HSE": "Stress Response",
    "SR-MMP": "Stress Response", "SR-p53": "Stress Response",
}

EDGE_DIM = 11
NODE_DIM = 45
HIDDEN_DIM = 128
DROPOUT = 0.15
