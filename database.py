# database.py
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Global DB Client
db = None

def initialize_firebase():
    global db
    if not firebase_admin._apps:
        cred = None
        # Check for Render Env Var
        if os.environ.get("FIREBASE_CREDENTIALS"):
            import json
            service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
            cred = credentials.Certificate(service_account_info)
        # Check for Local File
        elif os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        
        if cred:
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firebase Connected")
        else:
            print("❌ Warning: No Credentials Found")
    else:
        db = firestore.client()
    return db

# Initialize immediately when imported
db = initialize_firebase()
