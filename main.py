import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
from pydantic import BaseModel

# 1. Initialize Firebase
cred = None
if os.environ.get("FIREBASE_CREDENTIALS"):
    import json
    service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(service_account_info)
elif os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 2. Setup App
app = FastAPI()

# --- NEW: CORS CONFIGURATION (The Fix) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows ALL origins (Mobile App, Localhost, Website)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- Data Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str
    subject: str
    chapter: str

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI Backend is Running (CORS Enabled)"}

@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        # A. RAG STEP
        doc_id = f"{request.class_level}_{request.subject}_{request.chapter}"
        doc_ref = db.collection("book_content").document(doc_id)
        doc = doc_ref.get()

        book_context = ""
        if doc.exists:
            book_context = doc.to_dict().get("text_content", "")

        # B. Construct Prompt
        system_instruction = "You are a friendly Bangladeshi Tutor. Use the provided book context."
        full_prompt = f"""
        System: {system_instruction}
        Context: {book_context[:15000]}
        
        Student: {request.message}
        """

        # C. Call Gemini
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        
        if response.status_code != 200:
            return {"reply": f"Error from Google: {response.text}"}
            
        data = response.json()
        
        try:
            ai_reply = data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            ai_reply = "Sorry, I couldn't understand that."

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
