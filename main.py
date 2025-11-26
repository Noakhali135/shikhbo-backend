import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from typing import List, Optional

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

# --- Data Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str
    subject: str
    chapter: str

class ImageRequest(BaseModel):
    image_base64: str  # The image data
    user_id: str

# --- Helper: Call Gemini (Text) ---
def call_gemini_text(prompt: str):
    payload = {"contents": [{"parts": [{"text prompt": prompt}]}]} # Fixed key
    # Raw HTTP payload adjustment for Gemini API
    payload = { "contents": [{ "parts": [{ "text": prompt }] }] }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        return f"Error: {response.text}"
    
    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return "I am having trouble thinking right now."

# --- Helper: Call Gemini (Vision) ---
def call_gemini_vision(image_base64: str):
    # Gemini requires the base64 string without the "data:image/jpeg;base64," prefix
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    payload = {
      "contents": [{
        "parts": [
          {"text": "Analyze this math problem. Return a JSON object with a list called 'steps'. Each step has 'math' (latex) and 'explanation' (string)."},
          {
            "inline_data": {
              "mime_type": "image/jpeg",
              "data": image_base64
            }
          }
        ]
      }]
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(response.text)
        return None
        
    try:
        # Gemini returns text, we need to hope it's JSON or parse it
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return None

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo Backend V2 (Vision Enabled)"}

# 1. GET HISTORY
# Replace your existing get_history function with this:

@app.get("/history")
def get_history(user_id: str, subject: str, chapter: str):
    try:
        # 1. Construct Session ID
        session_id = f"{subject}_{chapter}"
        print(f"DEBUG: Fetching history for path: users/{user_id}/chat_sessions/{session_id}/messages")

        history_ref = db.collection("users").document(user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")
        
        # 2. Fetch WITHOUT ordering first (Fixes 'Missing Index' error)
        # We fetch all (limit 50) and sort in Python
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
        
        # 3. Sort in Python (Ascending Order: Oldest -> Newest)
        messages.sort(key=lambda x: x["time"])
        
        print(f"DEBUG: Found {len(messages)} messages")
        return {"messages": messages}

    except Exception as e:
        print(f"ERROR in get_history: {e}")
        # Return empty list instead of crashing
        return {"messages": []}
# 2. CHAT (With RAG + Memory)
@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        session_id = f"{request.subject}_{request.chapter}"
        history_ref = db.collection("users").document(request.user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")

        # 1. Get recent context
        recent_docs = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(3).stream()
        history_text = ""
        for doc in reversed(list(recent_docs)):
            data = doc.to_dict()
            role = "Student" if data['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {data['text']}\n"

        # 2. Get Book Content
        doc_id = f"{request.class_level}_{request.subject}_{request.chapter}"
        book_doc = db.collection("book_content").document(doc_id).get()
        book_context = book_doc.to_dict().get("text_content", "") if book_doc.exists else "No specific book content found."

        # 3. Prompt
        full_prompt = f"""
        System: You are a friendly Bangladeshi Tutor. 
        Context: {book_context[:8000]}
        History: {history_text}
        Student: {request.message}
        """

        # 4. Generate
        ai_reply = call_gemini_text(full_prompt)

        # 5. Save
        ts = int(time.time() * 1000)
        history_ref.add({"text": request.message, "sender": "user", "timestamp": ts})
        history_ref.add({"text": ai_reply, "sender": "ai", "timestamp": ts + 1})

        return {"reply": ai_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. ANALYZE IMAGE (Vision)
@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        # Call Gemini Vision
        json_result = call_gemini_vision(request.image_base64)
        
        if not json_result:
             # Fallback mock if vision fails (common in free tier limits)
             return {
                 "solution": [
                     {"id": 1, "math": "Error", "explanation": "Could not analyze image. Try again."}
                 ]
             }
        
        # In a real app, we would parse the JSON string here. 
        # For now, we return the raw text, and frontend can try to parse or display it.
        return {"raw_text": json_result}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
