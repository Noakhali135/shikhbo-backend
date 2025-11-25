import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

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
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

class ChatRequest(BaseModel):
    user_id: str      # The Firebase UID (e.g., "abc12345")
    message: str
    class_level: str
    subject: str
    chapter: str

@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        # --- STEP 1: Define Database Paths ---
        # We store chats specifically for this chapter
        session_id = f"{request.subject}_{request.chapter}"
        history_ref = db.collection("users").document(request.user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")

        # --- STEP 2: Fetch Recent History (Context) ---
        # Get last 5 messages so AI remembers context
        recent_docs = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
        history_text = ""
        # We fetch in reverse order (newest first), so we need to flip it back
        msgs = reversed(list(recent_docs)) 
        
        for msg in msgs:
            data = msg.to_dict()
            role = "Student" if data['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {data['text']}\n"

        # --- STEP 3: RAG (Book Content) ---
        doc_id = f"{request.class_level}_{request.subject}_{request.chapter}"
        book_doc = db.collection("book_content").document(doc_id).get()
        book_context = book_doc.to_dict().get("text_content", "") if book_doc.exists else ""

        # --- STEP 4: Construct Prompt ---
        full_prompt = f"""
        System: You are a friendly Bangladeshi Tutor. 
        Book Context: {book_context[:10000]}
        
        Conversation History:
        {history_text}
        
        Current Student Question: {request.message}
        """

        # --- STEP 5: Call Gemini ---
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        headers = {"Content-Type": "application/json"}
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        
        if response.status_code != 200:
            return {"reply": "Error connecting to AI teacher."}
            
        data = response.json()
        try:
            ai_reply = data['candidates'][0]['content']['parts'][0]['text']
        except:
            ai_reply = "Sorry, I didn't understand."

        # --- STEP 6: Save to History (The Memory) ---
        timestamp = int(time.time() * 1000)
        
        # Save User Message
        history_ref.add({
            "text": request.message,
            "sender": "user",
            "timestamp": timestamp
        })
        
        # Save AI Message
        history_ref.add({
            "text": ai_reply,
            "sender": "ai",
            "timestamp": timestamp + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
