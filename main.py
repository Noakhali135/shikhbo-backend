import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# --- 1. Initialize Firebase ---
cred = None
# Check for Render Environment Variable first
if os.environ.get("FIREBASE_CREDENTIALS"):
    import json
    service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(service_account_info)
# Fallback to local file for testing
elif os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- 2. Setup FastAPI ---
app = FastAPI()

# Allow ALL origins (Crucial for Mobile/Web Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Gemini Configuration (Raw HTTP) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- 4. Data Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str
    subject: str
    chapter: str

class ImageRequest(BaseModel):
    image_base64: str
    user_id: str

# --- 5. Helper Functions ---
def call_gemini_raw(prompt: str):
    payload = { "contents": [{ "parts": [{ "text": prompt }] }] }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"AI Error: {response.text}"
    except Exception as e:
        return "Sorry, I am having trouble connecting."

# --- 6. Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo Backend Live (Fixed History Sorting)"}

# GET HISTORY (The Fix for your issue)
@app.get("/history")
def get_history(user_id: str, subject: str, chapter: str):
    try:
        # 1. Construct Session ID
        session_id = f"{subject}_{chapter}"
        
        # 2. Reference
        history_ref = db.collection("users").document(user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")
        
        # 3. CRITICAL FIX: Fetch WITHOUT .order_by()
        # Firestore throws 500 error if you sort without an index.
        # We fetch the last 50 docs and sort them in Python instead.
        docs = history_ref.limit(50).stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                "id": doc.id,
                "text": data.get("text", ""),
                "isUser": data.get("sender") == "user",
                "time": data.get("timestamp", 0)
            })
        
        # 4. Sort in Python (Oldest -> Newest)
        messages.sort(key=lambda x: x["time"])
        
        return {"messages": messages}

    except Exception as e:
        print(f"History Error: {e}")
        # Return empty list so app doesn't crash
        return {"messages": []}

# CHAT ENDPOINT
@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        session_id = f"{request.subject}_{request.chapter}"
        
        # 1. Save User Message Immediately
        history_ref = db.collection("users").document(request.user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")
        
        current_ts = int(time.time() * 1000)
        history_ref.add({
            "text": request.message, 
            "sender": "user", 
            "timestamp": current_ts
        })

        # 2. Get Context (Last 3 messages) - manual sort
        docs = history_ref.limit(5).stream()
        msgs_list = []
        for d in docs:
            msgs_list.append(d.to_dict())
        msgs_list.sort(key=lambda x: x['timestamp']) # Sort oldest to newest
        
        history_text = ""
        for m in msgs_list:
            role = "Student" if m['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {m['text']}\n"

        # 3. Get Book Content
        doc_id = f"{request.class_level}_{request.subject}_{request.chapter}"
        book_doc = db.collection("book_content").document(doc_id).get()
        book_context = book_doc.to_dict().get("text_content", "") if book_doc.exists else ""

        # 4. Ask Gemini
        full_prompt = f"""
        System: You are a friendly Bangladeshi Tutor.
        Book Context: {book_context[:15000]}
        History: {history_text}
        Student Question: {request.message}
        """
        
        ai_reply = call_gemini_raw(full_prompt)

        # 5. Save AI Message
        history_ref.add({
            "text": ai_reply, 
            "sender": "ai", 
            "timestamp": current_ts + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# VISION ENDPOINT
@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        # Simplified Vision Call
        prompt = "Analyze this math problem. Return the solution in steps."
        
        # Clean Base64
        b64 = request.image_base64
        if "," in b64: b64 = b64.split(",")[1]

        payload = {
          "contents": [{
            "parts": [
              {"text": prompt},
              {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
            ]
          }]
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        
        if response.status_code != 200:
            return {"raw_text": "Error analyzing image."}
            
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return {"raw_text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
