import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

# --- 1. Initialize Firebase ---
cred = None
if os.environ.get("FIREBASE_CREDENTIALS"):
    import json
    service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(service_account_info)
elif os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")

if not firebase_admin._apps:
    if cred:
        firebase_admin.initialize_app(cred)
    else:
        print("Warning: No Firebase Credentials found. Database features will fail.")

db = firestore.client()

# --- 2. Setup FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ADMIN_SECRET_KEY = os.environ.get("ADMIN_SECRET_KEY", "my-secret-admin-password") # Set this in Render!
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025" 
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- 4. Data Models ---

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

# NEW: Admin Models
class AdminContentUpload(BaseModel):
    class_level: str
    group: str
    subject: str
    chapter_id: str
    chapter_title: str
    chapter_title_bn: str
    text_content: str

# --- 5. Security ---
def verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized Admin Access")

# --- 6. Helper Functions ---
def call_gemini_raw(system_instruction: str, prompt: str):
    payload = { 
        "contents": [{ "parts": [{ "text": prompt }] }],
        "system_instruction": { "parts": [{ "text": system_instruction }] },
        "generationConfig": {
            "maxOutputTokens": 2000,
            "temperature": 0.3
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(GEMINI_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I am having trouble thinking right now. Please try again."
    except Exception as e:
        return "Sorry, I am having trouble connecting to the internet."

# --- 7. Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI Tutor Backend Live"}

# ==========================================
# ADMIN PANEL ENDPOINTS
# ==========================================

@app.get("/admin/users", dependencies=[Depends(verify_admin)])
def admin_get_users():
    """List all users with their stats."""
    try:
        docs = db.collection("users").stream()
        users = []
        for doc in docs:
            d = doc.to_dict()
            users.append({
                "id": doc.id,
                "name": d.get("name", "Unknown"),
                "mobile": d.get("mobile", "N/A"),
                "class_level": d.get("class_level", "N/A"),
                "group": d.get("group", "N/A"),
                "total_usage": d.get("total_usage", 0), # Token usage proxy
                "last_active": d.get("last_active", 0)
            })
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/user/{user_id}/history", dependencies=[Depends(verify_admin)])
def admin_get_user_history(user_id: str):
    """See sessions for a specific user."""
    try:
        sessions = db.collection("users").document(user_id).collection("chat_sessions").order_by("updated_at", direction=firestore.Query.DESCENDING).limit(20).stream()
        data = []
        for s in sessions:
            sd = s.to_dict()
            data.append(sd)
        return {"history": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/upload-content", dependencies=[Depends(verify_admin)])
def admin_upload_content(data: AdminContentUpload):
    """
    Simultaneously updates the 'Book Content' (Text) AND 'Curriculum Metadata' (List).
    This automates what you were doing with Python scripts.
    """
    try:
        # 1. Upload Book Text (RAG)
        rag_doc_id = f"{data.class_level}_{data.subject}_{data.chapter_id}".replace(" ", "_")
        db.collection("book_content").document(rag_doc_id).set({
            "text_content": data.text_content,
            "updated_at": int(time.time() * 1000)
        })

        # 2. Update Curriculum List (Table of Contents)
        curr_doc_id = f"{data.class_level}_{data.group}".replace(" ", "_")
        curr_ref = db.collection("curriculum_metadata").document(curr_doc_id)
        
        # We need to add the chapter to the specific subject array
        # This requires reading, appending, and writing back (Transaction safest, but simple read/write okay for MVP admin)
        doc = curr_ref.get()
        if doc.exists:
            curr_data = doc.to_dict()
        else:
            curr_data = {}

        subject_list = curr_data.get(data.subject, [])
        
        # Check if chapter exists to avoid duplicates
        existing_index = next((index for (index, d) in enumerate(subject_list) if d["id"] == data.chapter_id), None)
        
        new_chapter_meta = {
            "id": data.chapter_id,
            "title": data.chapter_title,
            "titleBn": data.chapter_title_bn
        }

        if existing_index is not None:
            subject_list[existing_index] = new_chapter_meta # Update
        else:
            subject_list.append(new_chapter_meta) # Add
        
        curr_ref.set({data.subject: subject_list}, merge=True)

        return {"status": "success", "message": f"Uploaded {data.chapter_title} to {data.subject}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# APP ENDPOINTS (Existing)
# ==========================================

@app.post("/auth/check-availability")
def check_availability(req: AvailabilityRequest):
    try:
        phone_query = db.collection("users").where("mobile", "==", req.mobile).limit(1).stream()
        for doc in phone_query:
            raise HTTPException(status_code=409, detail="Mobile number already registered")
        email_query = db.collection("users").where("email", "==", req.email).limit(1).stream()
        for doc in email_query:
            raise HTTPException(status_code=409, detail="Email already registered")
        return {"available": True}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/profile")
def update_user_profile(profile: UserProfileRequest):
    try:
        doc_ref = db.collection("users").document(profile.user_id)
        update_data = {k: v for k, v in profile.dict().items() if v is not None and k != "user_id"}
        
        if profile.first_name:
            fname = profile.first_name or ""
            mname = profile.middle_name or ""
            lname = profile.last_name or ""
            update_data["name"] = f"{fname} {mname} {lname}".replace("  ", " ").strip()

        update_data["last_active"] = int(time.time() * 1000)
        doc_ref.set(update_data, merge=True)
        return {"status": "success", "message": "Profile updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}")
def get_user_profile(user_id: str):
    try:
        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/curriculum")
def get_curriculum(class_level: str, group: str):
    try:
        doc_id = f"{class_level}_{group}".replace(" ", "_")
        doc_ref = db.collection("curriculum_metadata").document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {}
    except Exception as e:
        return {}

@app.get("/sessions")
def get_sessions(user_id: str):
    try:
        sessions_ref = db.collection("users").document(user_id).collection("chat_sessions")
        docs = sessions_ref.stream()
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            sessions.append({
                "id": doc.id,
                "subject": data.get("subject", "Unknown"),
                "chapter": data.get("chapter", "Unknown"),
                "custom_title": data.get("custom_title", None),
                "class_level": data.get("class_level", ""),
                "group": data.get("group", ""),
                "updated_at": data.get("updated_at", 0),
            })
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"sessions": sessions}
    except Exception as e:
        return {"sessions": []}

@app.patch("/session/{session_id}/rename")
def rename_session(session_id: str, request: RenameSessionRequest):
    try:
        doc_ref = db.collection("users").document(request.user_id)\
                    .collection("chat_sessions").document(session_id)
        if not doc_ref.get().exists:
            raise HTTPException(status_code=404, detail="Session not found")
        doc_ref.update({"custom_title": request.new_title})
        return {"status": "success", "message": "Session renamed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
def delete_session(session_id: str, user_id: str = Query(...)):
    try:
        doc_ref = db.collection("users").document(user_id)\
                    .collection("chat_sessions").document(session_id)
        doc_ref.delete()
        return {"status": "success", "message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(user_id: str, session_id: Optional[str] = Query(None), subject: Optional[str] = None, chapter: Optional[str] = None):
    try:
        if session_id:
            target_id = session_id
        elif subject and chapter:
            target_id = f"{subject}_{chapter}"
        else:
            return {"messages": []}

        history_ref = db.collection("users").document(user_id)\
                        .collection("chat_sessions").document(target_id)\
                        .collection("messages")
        
        docs = history_ref.order_by("timestamp").limit(50).stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                "id": doc.id,
                "text": data.get("text", ""),
                "isUser": data.get("sender") == "user",
                "time": data.get("timestamp", 0)
            })
        return {"messages": messages}
    except Exception as e:
        return {"messages": []}

@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = f"{request.subject}_{request.chapter_id}"
        
        user_doc_ref = db.collection("users").document(request.user_id)
        session_ref = user_doc_ref.collection("chat_sessions").document(session_id)
        messages_ref = session_ref.collection("messages")
        current_ts = int(time.time() * 1000)

        # Update Session Metadata
        session_ref.set({
            "subject": request.subject,
            "chapter": request.chapter_id, 
            "class_level": request.class_level,
            "group": request.group,
            "updated_at": current_ts,
            "last_message": request.message[:50]
        }, merge=True)

        # Save User Message
        messages_ref.add({"text": request.message, "sender": "user", "timestamp": current_ts})

        # Calculate estimated tokens (Approximation: 1 word ~ 1.3 tokens)
        # We track this on the User Profile
        user_doc_ref.update({
            "total_usage": firestore.Increment(len(request.message) // 4),
            "last_active": current_ts
        })

        # RAG Logic
        rag_doc_id = f"{request.class_level}_{request.subject}_{request.chapter_id}".replace(" ", "_")
        book_doc = db.collection("book_content").document(rag_doc_id).get()
        book_context = book_doc.to_dict().get("text_content", "") if book_doc.exists else "No specific book content found. Use general knowledge."

        docs = messages_ref.order_by("timestamp").limit(10).stream()
        msgs_list = [d.to_dict() for d in docs]
        history_text = "\n".join([f"{'Student' if m['sender'] == 'user' else 'Tutor'}: {m['text']}" for m in msgs_list])

        lang_instruction = "Speak in a friendly mix of Bangla and English (Tanglish). Act like a Bangladeshi older brother/sister (Bhaiya/Apu)."
        if request.medium and "English" in request.medium:
            lang_instruction = "You are a Tutor for English Version students. Explain primarily in English. You may use Bangla text for very difficult terms only if necessary."

        system_instruction = f"""
        You are a private tutor for a Bangladeshi student in {request.class_level} ({request.group}).
        {lang_instruction}
        STRICTLY use the provided 'Book Context' to answer.
        Use LaTeX for Math.
        """

        full_prompt = f"BOOK CONTEXT:\n{book_context}\nCHAT HISTORY:\n{history_text}\nSTUDENT QUESTION:\n{request.message}"
        
        ai_reply = call_gemini_raw(system_instruction, full_prompt)
        messages_ref.add({"text": ai_reply, "sender": "ai", "timestamp": current_ts + 1})

        # Track output usage
        user_doc_ref.update({"total_usage": firestore.Increment(len(ai_reply) // 4)})

        return {"reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
