import os
import time
import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shikhbo-backend")

# Initialize Firebase
cred = None
firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_creds:
    cred_dict = json.loads(firebase_creds)
    cred = credentials.Certificate(cred_dict)
elif os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    if cred:
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Initialized")
    else:
        logger.warning("No Firebase Credentials found! Database operations will fail.")

db = firestore.client()

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Admin Secret (Set this in Render Env Vars)
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "123qwe")

# --- Models ---

class AvailabilityRequest(BaseModel):
    email: str
    mobile: str

class UserProfile(BaseModel):
    user_id: str
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    class_level: Optional[str] = None
    group: Optional[str] = None
    medium: Optional[str] = None
    language: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str
    subject: str
    chapter_id: str
    class_level: str
    group: str
    medium: str

class RenameSessionRequest(BaseModel):
    user_id: str
    new_title: str

# Admin Models
class Chapter(BaseModel):
    id: str
    title: str
    titleBn: str

class UpdateChapterRequest(BaseModel):
    class_level: str
    group: str
    subject: str
    chapters: List[Chapter]

class UpdateContextRequest(BaseModel):
    class_level: str
    subject: str
    chapter_id: str
    text_content: str

# --- App Setup ---
app = FastAPI(title="Shikhbo AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Dependency ---
async def verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid Admin Key")
    return True

# --- Helper Functions ---

def get_system_instruction(class_level: str, group: str, medium: str, subject: str) -> str:
    lang_instruction = "Reply primarily in English." if medium == "English Version" else "Reply in a mix of Bangla (Tanglish) and English naturally."
    return (
        f"You are a friendly and encouraging AI Tutor for a student in {class_level} ({group}), {medium}. "
        f"The subject is {subject}. "
        f"{lang_instruction} "
        "Keep answers concise, engaging, and easy to understand. "
        "Use emojis occasionally. "
        "If the student asks a math or physics problem, solve it step-by-step using LaTeX formatting for equations (e.g., $$x^2$$)."
    )

def fetch_book_context(class_level: str, subject: str, chapter_id: str) -> str:
    try:
        doc_id = f"{class_level}_{subject}_{chapter_id}".replace(" ", "_")
        doc = db.collection("book_content").document(doc_id).get()
        if doc.exists:
            return doc.to_dict().get("text_content", "")
    except Exception as e:
        logger.error(f"Error fetching context: {e}")
    return ""

# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "active", "service": "Shikhbo AI"}

# --- ADMIN ENDPOINTS (NEW) ---

@app.get("/admin/stats", dependencies=[Depends(verify_admin)])
def get_admin_stats():
    """Get basic usage statistics."""
    try:
        # Note: Counting documents is expensive in Firestore. 
        # For MVP (<1000 users) this is fine. For scale, maintain a separate counter document.
        users_count = len(list(db.collection("users").stream()))
        
        # We can't easily count all subcollections in Firestore without recursive functions.
        # Returning user count is a good start.
        return {
            "total_users": users_count,
            "status": "online"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users", dependencies=[Depends(verify_admin)])
def get_all_users(limit: int = 50):
    """List recent users."""
    try:
        docs = db.collection("users").order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
        users = []
        for doc in docs:
            d = doc.to_dict()
            users.append({
                "id": doc.id,
                "name": d.get("name", "Unknown"),
                "class": d.get("class_level", "-"),
                "last_active": d.get("updated_at")
            })
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/chapter", dependencies=[Depends(verify_admin)])
def update_curriculum(req: UpdateChapterRequest):
    """Add or Remove chapters for a subject."""
    try:
        doc_id = f"{req.class_level}_{req.group}".replace(" ", "_")
        doc_ref = db.collection("curriculum_metadata").document(doc_id)
        
        # Serialize chapters list
        chapters_data = [c.dict() for c in req.chapters]
        
        # Update specific subject field (merge=True preserves other subjects)
        doc_ref.set({req.subject: chapters_data}, merge=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/context", dependencies=[Depends(verify_admin)])
def update_book_context(req: UpdateContextRequest):
    """Upload text content for a chapter."""
    try:
        doc_id = f"{req.class_level}_{req.subject}_{req.chapter_id}".replace(" ", "_")
        doc_ref = db.collection("book_content").document(doc_id)
        
        doc_ref.set({
            "text_content": req.text_content,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/context", dependencies=[Depends(verify_admin)])
def get_book_context(class_level: str, subject: str, chapter_id: str):
    """Retrieve existing context."""
    text = fetch_book_context(class_level, subject, chapter_id)
    return {"text_content": text}


# --- EXISTING ENDPOINTS ---

@app.post("/auth/check-availability")
def check_availability(req: AvailabilityRequest):
    try:
        phone_query = db.collection("users").where("mobile", "==", req.mobile).limit(1).stream()
        if any(phone_query):
            raise HTTPException(status_code=409, detail="Mobile number already registered")
        email_query = db.collection("users").where("email", "==", req.email).limit(1).stream()
        if any(email_query):
            raise HTTPException(status_code=409, detail="Email already registered")
        return {"available": True}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Availability check error: {e}")
        return {"available": True}

@app.post("/user/profile")
def save_profile(profile: UserProfile):
    try:
        doc_ref = db.collection("users").document(profile.user_id)
        update_data = {k: v for k, v in profile.dict().items() if v is not None}
        fname = profile.first_name or ""
        lname = profile.last_name or ""
        update_data["name"] = f"{fname} {lname}".strip()
        update_data["updated_at"] = firestore.SERVER_TIMESTAMP
        doc_ref.set(update_data, merge=True)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Profile save error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save profile")

@app.get("/user/{user_id}")
def get_profile(user_id: str):
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
        raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/curriculum")
def get_curriculum(class_level: str = Query(...), group: str = Query(...)):
    try:
        doc_id = f"{class_level}_{group}".replace(" ", "_")
        doc = db.collection("curriculum_metadata").document(doc_id).get()
        if doc.exists:
            return doc.to_dict() 
        return {}
    except Exception as e:
        logger.error(f"Curriculum fetch error: {e}")
        return {}

@app.post("/chat")
async def chat_tutor(req: ChatRequest):
    try:
        user_ref = db.collection("users").document(req.user_id)
        session_ref = user_ref.collection("chat_sessions").document(req.session_id)
        msgs_ref = session_ref.collection("messages")
        
        timestamp = int(time.time() * 1000)
        msgs_ref.add({
            "text": req.message,
            "sender": "user",
            "timestamp": timestamp
        })

        session_data = {
            "subject": req.subject,
            "chapter": req.chapter_id,
            "class_level": req.class_level,
            "group": req.group,
            "updated_at": timestamp,
            "last_message": req.message[:60]
        }
        session_ref.set(session_data, merge=True)

        system_instruction = get_system_instruction(req.class_level, req.group, req.medium, req.subject)
        book_context = fetch_book_context(req.class_level, req.subject, req.chapter_id)
        
        history_docs = msgs_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
        history_list = sorted([d.to_dict() for d in history_docs], key=lambda x: x['timestamp'])
        
        history_text = ""
        for msg in history_list:
            role = "Student" if msg['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {msg['text']}\n"

        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            f"SYSTEM INSTRUCTION: {system_instruction}\n"
            f"REFERENCE MATERIAL: {book_context[:10000]}\n"
            f"CONVERSATION HISTORY:\n{history_text}\n"
            f"STUDENT: {req.message}\n"
            f"TUTOR:"
        )

        response = model.generate_content(prompt)
        ai_reply = response.text

        msgs_ref.add({
            "text": ai_reply,
            "sender": "ai",
            "timestamp": int(time.time() * 1000) + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": "I'm having a little trouble connecting right now. Could you ask that again?"}

@app.get("/sessions")
def get_user_sessions(user_id: str):
    try:
        sessions_ref = db.collection("users").document(user_id).collection("chat_sessions")
        docs = sessions_ref.order_by("updated_at", direction=firestore.Query.DESCENDING).stream()
        sessions = []
        for doc in docs:
            d = doc.to_dict()
            sessions.append({
                "id": doc.id,
                "subject": d.get("subject"),
                "chapter": d.get("chapter"),
                "custom_title": d.get("custom_title"),
                "title_bn": d.get("title_bn"),
                "class_level": d.get("class_level"),
                "group": d.get("group"),
                "updated_at": d.get("updated_at")
            })
        return {"sessions": sessions}
    except Exception as e:
        return {"sessions": []}

@app.get("/history")
def get_chat_history(user_id: str, session_id: str):
    try:
        msgs_ref = db.collection("users").document(user_id)\
                     .collection("chat_sessions").document(session_id)\
                     .collection("messages")
        docs = msgs_ref.order_by("timestamp").limit(100).stream()
        messages = []
        for doc in docs:
            d = doc.to_dict()
            messages.append({
                "id": doc.id,
                "text": d.get("text"),
                "isUser": d.get("sender") == "user",
                "time": d.get("timestamp")
            })
        return {"messages": messages}
    except Exception as e:
        return {"messages": []}

@app.patch("/session/{session_id}/rename")
def rename_session(session_id: str, req: RenameSessionRequest):
    try:
        doc_ref = db.collection("users").document(req.user_id)\
                    .collection("chat_sessions").document(session_id)
        if not doc_ref.get().exists:
            raise HTTPException(status_code=404, detail="Session not found")
        doc_ref.update({"custom_title": req.new_title})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rename")

@app.delete("/session/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    try:
        doc_ref = db.collection("users").document(user_id)\
                    .collection("chat_sessions").document(session_id)
        doc_ref.delete()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
