# main.py
import os
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

# 1. Import DB and Routes
from database import db
from admin_routes import admin_router 

app = FastAPI()

# 2. Register Admin Router
app.include_router(admin_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GEMINI SETUP ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025" 
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str
    group: str
    subject: str
    chapter_id: str
    session_id: Optional[str] = None
    medium: Optional[str] = "Bangla Medium"

class AvailabilityRequest(BaseModel):
    email: str
    mobile: str

class UserProfileRequest(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    mobile: Optional[str] = None
    class_level: Optional[str] = None
    group: Optional[str] = None
    medium: Optional[str] = None
    language: Optional[str] = None

class RenameSessionRequest(BaseModel):
    user_id: str
    new_title: str

# --- HELPER FUNCTIONS ---
def call_gemini_raw(system_instruction: str, prompt: str):
    payload = { 
        "contents": [{ "parts": [{ "text": prompt }] }],
        "system_instruction": { "parts": [{ "text": system_instruction }] },
        "generationConfig": { "maxOutputTokens": 2000, "temperature": 0.3 }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Thinking error. Please try again."
    except Exception:
        return "Connection error."

# --- STUDENT API ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI System Operational"}

@app.post("/auth/check-availability")
def check_availability(req: AvailabilityRequest):
    try:
        if db.collection("users").where("mobile", "==", req.mobile).limit(1).get():
            raise HTTPException(status_code=409, detail="Mobile used")
        if db.collection("users").where("email", "==", req.email).limit(1).get():
            raise HTTPException(status_code=409, detail="Email used")
        return {"available": True}
    except HTTPException as he: raise he
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/profile")
def update_user_profile(profile: UserProfileRequest):
    try:
        doc_ref = db.collection("users").document(profile.user_id)
        update_data = {k: v for k, v in profile.dict().items() if v is not None and k != "user_id"}
        if profile.first_name:
            update_data["name"] = f"{profile.first_name} {profile.middle_name or ''} {profile.last_name or ''}".replace("  ", " ").strip()
        update_data["last_active"] = int(time.time() * 1000)
        doc_ref.set(update_data, merge=True)
        return {"status": "success"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}")
def get_user_profile(user_id: str):
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists: return doc.to_dict()
        raise HTTPException(status_code=404, detail="Not found")
    except Exception as e: 
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/curriculum")
def get_curriculum(class_level: str, group: str):
    try:
        doc_id = f"{class_level}_{group}".replace(" ", "_")
        doc = db.collection("curriculum_metadata").document(doc_id).get()
        return doc.to_dict() if doc.exists else {}
    except: return {}

@app.get("/sessions")
def get_sessions(user_id: str):
    try:
        docs = db.collection("users").document(user_id).collection("chat_sessions").stream()
        sessions = [{**doc.to_dict(), "id": doc.id} for doc in docs]
        sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        return {"sessions": sessions}
    except: return {"sessions": []}

@app.patch("/session/{session_id}/rename")
def rename_session(session_id: str, request: RenameSessionRequest):
    try:
        doc_ref = db.collection("users").document(request.user_id).collection("chat_sessions").document(session_id)
        if not doc_ref.get().exists: raise HTTPException(404)
        doc_ref.update({"custom_title": request.new_title})
        return {"status": "success"}
    except: raise HTTPException(500)

@app.delete("/session/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    try:
        db.collection("users").document(user_id).collection("chat_sessions").document(session_id).delete()
        return {"status": "success"}
    except: raise HTTPException(500)

@app.get("/history")
def get_history(user_id: str, session_id: Optional[str] = Query(None), subject: Optional[str] = None, chapter: Optional[str] = None):
    try:
        target_id = session_id if session_id else f"{subject}_{chapter}"
        docs = db.collection("users").document(user_id).collection("chat_sessions").document(target_id).collection("messages").order_by("timestamp").limit(50).stream()
        return {"messages": [{"id": d.id, "text": d.get("text"), "isUser": d.get("sender") == "user", "time": d.get("timestamp")} for d in docs]}
    except: return {"messages": []}

@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        session_id = request.session_id or f"{request.subject}_{request.chapter_id}"
        user_ref = db.collection("users").document(request.user_id)
        session_ref = user_ref.collection("chat_sessions").document(session_id)
        messages_ref = session_ref.collection("messages")
        current_ts = int(time.time() * 1000)

        session_ref.set({
            "subject": request.subject,
            "chapter": request.chapter_id,
            "class_level": request.class_level,
            "group": request.group,
            "updated_at": current_ts,
            "last_message": request.message[:50]
        }, merge=True)

        messages_ref.add({"text": request.message, "sender": "user", "timestamp": current_ts})

        rag_id = f"{request.class_level}_{request.subject}_{request.chapter_id}".replace(" ", "_")
        book_doc = db.collection("book_content").document(rag_id).get()
        context = book_doc.to_dict().get("text_content", "") if book_doc.exists else "No context found."

        history_docs = messages_ref.order_by("timestamp").limit(10).stream()
        history_text = "\n".join([f"{'Student' if d.get('sender')=='user' else 'Tutor'}: {d.get('text')}" for d in history_docs])

        lang_inst = "You are a Tutor for English Version." if request.medium and "English" in request.medium else "Speak in Tanglish (Bangla/English)."
        
        system = f"You are a friendly BD Tutor for {request.class_level}. {lang_inst} Strictly use Book Context."
        prompt = f"BOOK CONTEXT: {context}\nHISTORY: {history_text}\nQUESTION: {request.message}"

        reply = call_gemini_raw(system, prompt)
        messages_ref.add({"text": reply, "sender": "ai", "timestamp": current_ts + 1})
        
        return {"reply": reply}
    except Exception as e: raise HTTPException(500, detail=str(e))
