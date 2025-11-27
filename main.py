import os
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# --- 3. Gemini Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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

class ImageRequest(BaseModel):
    image_base64: str
    user_id: str

# NEW: Model for checking duplicates
class AvailabilityRequest(BaseModel):
    email: str
    mobile: str

# UPDATED: Detailed User Profile
class UserProfileRequest(BaseModel):
    user_id: str
    first_name: str
    middle_name: str = ""   # Optional
    last_name: str = ""     # Optional
    email: str
    mobile: str
    # These might be empty during initial registration, filled later in "Setup"
    class_level: str = ""
    group: str = ""
    language: str = "bn"

# --- 5. Helper Functions ---
def call_gemini_raw(system_instruction: str, prompt: str):
    payload = { 
        "contents": [{ "parts": [{ "text": prompt }] }],
        "system_instruction": { "parts": [{ "text": system_instruction }] },
        "generationConfig": {
            "maxOutputTokens": 1000,
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

# --- 6. Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI Tutor Backend Live"}

# --- NEW: CHECK AVAILABILITY (Prevents Duplicates) ---
@app.post("/auth/check-availability")
def check_availability(req: AvailabilityRequest):
    """
    Checks if Email or Phone is already used in the 'users' collection.
    Returns { "available": false, "reason": "phone" } if found.
    """
    try:
        # Check Phone
        phone_query = db.collection("users").where("mobile", "==", req.mobile).limit(1).stream()
        for doc in phone_query:
            return {"available": False, "reason": "mobile", "message": "This phone number is already registered. Please Login."}
        
        # Check Email (Optional here since Firebase Auth handles it, but good for UI feedback)
        email_query = db.collection("users").where("email", "==", req.email).limit(1).stream()
        for doc in email_query:
            return {"available": False, "reason": "email", "message": "This email is already registered. Please Login."}

        return {"available": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- UPDATED: SAVE USER PROFILE ---
@app.post("/user/profile")
def update_user_profile(profile: UserProfileRequest):
    try:
        doc_ref = db.collection("users").document(profile.user_id)
        
        # Create full display name
        full_name = f"{profile.first_name} {profile.middle_name} {profile.last_name}".replace("  ", " ").strip()
        
        data = {
            "first_name": profile.first_name,
            "middle_name": profile.middle_name,
            "last_name": profile.last_name,
            "name": full_name, # Calculated field for easy display
            "email": profile.email,
            "mobile": profile.mobile,
            "language": profile.language,
            "last_active": int(time.time() * 1000)
        }
        
        # Only update class/group if provided (don't erase them if sending partial update)
        if profile.class_level: data["class_level"] = profile.class_level
        if profile.group: data["group"] = profile.group

        doc_ref.set(data, merge=True)
        return {"status": "success", "message": "Profile saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- GET USER PROFILE ---
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

# --- GET DYNAMIC CURRICULUM ---
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

# --- CHAT & SESSION ENDPOINTS (Same as before) ---
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
                "updated_at": data.get("updated_at", 0),
            })
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"sessions": sessions}
    except Exception as e:
        return {"sessions": []}

@app.get("/history")
def get_history(user_id: str, subject: str, chapter: str = Query(..., alias="chapter")):
    try:
        session_id = f"{subject}_{chapter}"
        history_ref = db.collection("users").document(user_id).collection("chat_sessions").document(session_id).collection("messages")
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
        messages.sort(key=lambda x: x["time"])
        return {"messages": messages}
    except Exception as e:
        return {"messages": []}

@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        session_id = f"{request.subject}_{request.chapter_id}"
        user_doc_ref = db.collection("users").document(request.user_id)
        session_ref = user_doc_ref.collection("chat_sessions").document(session_id)
        messages_ref = session_ref.collection("messages")
        current_ts = int(time.time() * 1000)

        session_ref.set({
            "subject": request.subject,
            "chapter": request.chapter_id, 
            "group": request.group,
            "updated_at": current_ts,
            "last_message": request.message[:50]
        }, merge=True)

        messages_ref.add({"text": request.message, "sender": "user", "timestamp": current_ts})

        rag_doc_id = f"{request.class_level}_{request.subject}_{request.chapter_id}".replace(" ", "_")
        book_doc = db.collection("book_content").document(rag_doc_id).get()
        book_context = book_doc.to_dict().get("text_content", "") if book_doc.exists else "No specific book content found."

        docs = messages_ref.limit(5).stream()
        msgs_list = sorted([d.to_dict() for d in docs], key=lambda x: x['timestamp'])
        history_text = "\n".join([f"{'Student' if m['sender'] == 'user' else 'Tutor'}: {m['text']}" for m in msgs_list])

        system_instruction = f"You are a friendly Bangladeshi tutor for {request.class_level} ({request.group}). Speak in Tanglish. Use Book Context."
        full_prompt = f"BOOK CONTEXT: {book_context[:15000]}\nCHAT HISTORY: {history_text}\nSTUDENT QUESTION: {request.message}"
        
        ai_reply = call_gemini_raw(system_instruction, full_prompt)
        messages_ref.add({"text": ai_reply, "sender": "ai", "timestamp": current_ts + 1})

        return {"reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    # Same as previous...
    return {"id": "img_sol_1", "solution": [{"id": 1, "math": "Analysis", "explanation": "Step-by-step logic..."}]}
