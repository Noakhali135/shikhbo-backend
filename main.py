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
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shikhbo-backend")

# Initialize Firebase
cred = None
firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")

if firebase_creds:
    try:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
    except Exception as e:
        logger.error(f"Firebase Creds Error: {e}")
elif os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    if cred:
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Initialized")
    else:
        logger.warning("No Firebase Credentials found!")

db = firestore.client()

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.error("GEMINI_API_KEY Not Found!")

# --- ADMIN SECURITY ---
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "change_this_in_render_env_vars")

async def verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

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

# --- SYSTEM INSTRUCTION (HUMAN PERSONA) ---
def get_system_instruction(class_level: str, group: str, medium: str, subject: str) -> str:
    
    # 1. Output Language Rule
    if medium == "English Version":
        lang_rule = (
            "LANGUAGE RULE: Speak strictly in fluent, natural English. "
            "Do not switch to Bangla unless explaining a specific cultural term."
        )
    else:
        lang_rule = (
            "LANGUAGE RULE: Speak primarily in Standard Bangla (Bangla Script). "
            "You CANNOT use 'Banglish' or Romanized Bangla under any circumstances. "
            "Write exactly like a text book or a native speaker writing in Bangla. "
            "You can use English terms in brackets for technical words, e.g., 'মহাকর্ষ (Gravity)'."
        )

    return (
        f"ROLE: You are an expert, caring, and enthusiastic human private tutor. Your name is 'Shikhbo Tutor'. "
        f"You are teaching a student of {class_level} ({group}), {medium}. "
        f"The current subject is {subject}.\n\n"
        
        f"{lang_rule}\n\n"

        "PERSONA GUIDELINES (STRICTLY FOLLOW):\n"
        "1. BE HUMAN: Never say 'As an AI', 'I am a model', or 'I cannot feel'. If asked how you are, say you're great and happy to help.\n"
        "2. BE NATURAL: Speak conversationally. Don't be robotic or overly formal. Use a warm, encouraging tone like an older sibling or a favorite teacher.\n"
        "3. NO LECTURES: Do not dump giant walls of text. Keep your answers concise (2-3 paragraphs max). Ask follow-up questions to check understanding.\n"
        "4. BE ENCOURAGING: Use emojis occasionally (✨, 👍, 📚). If the student makes a mistake, be gentle. If they do well, celebrate it!\n"
        "5. SHOW YOUR WORK: For Math/Physics, solve problems step-by-step using LaTeX formatting (e.g., $$F = ma$$).\n"
        "6. ADAPTABILITY: If the student chats in Banglish, English, or Bangla, YOU must understand them, but your reply MUST follow the 'LANGUAGE RULE' above."
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

# ==========================================
#       ENDPOINTS
# ==========================================

@app.get("/")
def health_check():
    return {"status": "Shikhbo Backend Live"}

@app.post("/chat")
async def chat_tutor(req: ChatRequest):
    try:
        if not GEMINI_API_KEY:
             return {"reply": "System Error: Gemini API Key is missing."}

        # 1. DB Operations
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

        # 2. Prepare Context
        system_instruction = get_system_instruction(req.class_level, req.group, req.medium, req.subject)
        book_context = fetch_book_context(req.class_level, req.subject, req.chapter_id)
        
        history_docs = msgs_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
        history_list = sorted([d.to_dict() for d in history_docs], key=lambda x: x['timestamp'])
        
        history_text = ""
        for msg in history_list:
            role = "Student" if msg['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {msg['text']}\n"

        # 3. Call AI (Updated Model)
        # Using Gemini 2.0 Flash Lite Preview (Latest fast model)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        prompt = (
            f"SYSTEM INSTRUCTION: {system_instruction}\n"
            f"REFERENCE MATERIAL: {book_context[:10000]}\n"
            f"CONVERSATION HISTORY:\n{history_text}\n"
            f"STUDENT: {req.message}\n"
            f"TUTOR:"
        )

        # Allow Math/Science content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        ai_reply = response.text

        # 4. Stats
        try:
            if hasattr(response, 'usage_metadata'):
                t_in = response.usage_metadata.prompt_token_count
                t_out = response.usage_metadata.candidates_token_count
                total_tokens = t_in + t_out
                db.collection("admin").document("global_stats").set({
                    "total_tokens": firestore.Increment(total_tokens)
                }, merge=True)
        except Exception:
            pass

        msgs_ref.add({
            "text": ai_reply,
            "sender": "ai",
            "timestamp": int(time.time() * 1000) + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        return {"reply": f"System Error: {str(e)}"}

# --- Other Endpoints ---

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
        raise HTTPException(status_code=500, detail="Failed to save profile")

@app.get("/user/{user_id}")
def get_profile(user_id: str):
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            return doc.to_dict()
        raise HTTPException(status_code=404, detail="User not found")
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
        return {}

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
                "id": doc.id, "text": d.get("text"), "isUser": d.get("sender") == "user", "time": d.get("timestamp")
            })
        return {"messages": messages}
    except Exception as e:
        return {"messages": []}

@app.patch("/session/{session_id}/rename")
def rename_session(session_id: str, req: RenameSessionRequest):
    try:
        doc_ref = db.collection("users").document(req.user_id).collection("chat_sessions").document(session_id)
        if not doc_ref.get().exists: raise HTTPException(status_code=404, detail="Session not found")
        doc_ref.update({"custom_title": req.new_title})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to rename")

@app.delete("/session/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    try:
        doc_ref = db.collection("users").document(user_id).collection("chat_sessions").document(session_id)
        doc_ref.delete()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete")

# Admin Endpoints
@app.get("/admin/stats", dependencies=[Depends(verify_admin)])
def get_admin_stats():
    try:
        users_count = len(list(db.collection("users").stream()))
        stats_doc = db.collection("admin").document("global_stats").get()
        total_tokens = stats_doc.to_dict().get("total_tokens", 0) if stats_doc.exists else 0
        return {"total_users": users_count, "total_tokens": total_tokens, "status": "online"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users", dependencies=[Depends(verify_admin)])
def get_all_users(limit: int = 50):
    try:
        docs = db.collection("users").order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
        users = []
        for doc in docs:
            d = doc.to_dict()
            la = d.get("updated_at"); 
            if hasattr(la, 'isoformat'): la = la.isoformat()
            users.append({"id": doc.id, "name": d.get("name"), "class": d.get("class_level"), "email": d.get("email"), "mobile": d.get("mobile"), "last_active": la})
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/chapter", dependencies=[Depends(verify_admin)])
def update_curriculum(req: UpdateChapterRequest):
    try:
        doc_id = f"{req.class_level}_{req.group}".replace(" ", "_")
        doc_ref = db.collection("curriculum_metadata").document(doc_id)
        chapters_data = [c.dict() for c in req.chapters]
        doc_ref.set({req.subject: chapters_data}, merge=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/context", dependencies=[Depends(verify_admin)])
def update_book_context(req: UpdateContextRequest):
    try:
        doc_id = f"{req.class_level}_{req.subject}_{req.chapter_id}".replace(" ", "_")
        doc_ref = db.collection("book_content").document(doc_id)
        doc_ref.set({"text_content": req.text_content, "updated_at": firestore.SERVER_TIMESTAMP})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/context", dependencies=[Depends(verify_admin)])
def get_book_context(class_level: str, subject: str, chapter_id: str, verified=Depends(verify_admin)):
    text = fetch_book_context(class_level, subject, chapter_id)
    return {"text_content": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
