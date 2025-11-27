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

# Model for Chatting
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str
    group: str
    subject: str
    chapter_id: str

# Model for Image Analysis
class ImageRequest(BaseModel):
    image_base64: str
    user_id: str

# NEW: Model for User Profile Saving
class UserProfileRequest(BaseModel):
    user_id: str
    name: str = "Student"
    email: str = ""
    mobile: str = ""
    class_level: str
    group: str
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
            print(f"Gemini API Error: {response.text}")
            return "I am having trouble thinking right now. Please try again."
    except Exception as e:
        print(f"Connection Error: {e}")
        return "Sorry, I am having trouble connecting to the internet."

# --- 6. Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI Tutor Backend Live"}

# --- NEW: SAVE USER PROFILE ---
@app.post("/user/profile")
def update_user_profile(profile: UserProfileRequest):
    """
    Saves the user's class, group, and personal info to Firestore.
    Called after 'Profile Screen' or 'Registration'.
    """
    try:
        doc_ref = db.collection("users").document(profile.user_id)
        # We merge=True so we don't overwrite existing fields like 'subscription_status' if they exist
        doc_ref.set({
            "name": profile.name,
            "email": profile.email,
            "mobile": profile.mobile,
            "class_level": profile.class_level,
            "group": profile.group,
            "language": profile.language,
            "last_active": int(time.time() * 1000)
        }, merge=True)
        return {"status": "success", "message": "Profile updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: GET USER PROFILE ---
@app.get("/user/{user_id}")
def get_user_profile(user_id: str):
    """
    Fetches user details. Called immediately after Login to restore state.
    """
    try:
        doc_ref = db.collection("users").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            # Return 404 so frontend knows to show the Setup Screen
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
        print(f"Curriculum Error: {e}")
        return {}

# --- GET SESSIONS ---
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
        print(f"Sessions Error: {e}")
        return {"sessions": []}

# --- GET HISTORY ---
@app.get("/history")
def get_history(user_id: str, subject: str, chapter: str = Query(..., alias="chapter")):
    try:
        session_id = f"{subject}_{chapter}"
        history_ref = db.collection("users").document(user_id)\
                        .collection("chat_sessions").document(session_id)\
                        .collection("messages")
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
        print(f"History Error: {e}")
        return {"messages": []}

# --- POST CHAT ---
@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        session_id = f"{request.subject}_{request.chapter_id}"
        
        user_doc_ref = db.collection("users").document(request.user_id)
        session_ref = user_doc_ref.collection("chat_sessions").document(session_id)
        messages_ref = session_ref.collection("messages")
        current_ts = int(time.time() * 1000)

        # Update Session Metadata
        session_ref.set({
            "subject": request.subject,
            "chapter": request.chapter_id, 
            "group": request.group,
            "updated_at": current_ts,
            "last_message": request.message[:50]
        }, merge=True)

        # Save User Message
        messages_ref.add({
            "text": request.message, 
            "sender": "user", 
            "timestamp": current_ts
        })

        # RAG Logic
        rag_doc_id = f"{request.class_level}_{request.subject}_{request.chapter_id}".replace(" ", "_")
        book_doc = db.collection("book_content").document(rag_doc_id).get()
        
        if book_doc.exists:
            book_context = book_doc.to_dict().get("text_content", "")
        else:
            book_context = "No specific book content found for this chapter. Answer based on general knowledge."

        # Fetch History
        docs = messages_ref.limit(5).stream()
        msgs_list = []
        for d in docs: msgs_list.append(d.to_dict())
        msgs_list.sort(key=lambda x: x['timestamp'])
        
        history_text = ""
        for m in msgs_list:
            role = "Student" if m['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {m['text']}\n"

        system_instruction = f"""
        You are a friendly Bangladeshi tutor for {request.class_level} ({request.group}).
        Speak in Tanglish. Use the Book Context strictly.
        """

        full_prompt = f"""
        BOOK CONTEXT: {book_context[:15000]} 
        CHAT HISTORY: {history_text}
        STUDENT QUESTION: {request.message}
        """
        
        ai_reply = call_gemini_raw(system_instruction, full_prompt)

        messages_ref.add({
            "text": ai_reply, 
            "sender": "ai", 
            "timestamp": current_ts + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- POST VISION ---
@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        prompt = "Analyze this image (math/diagram). Solve it step-by-step."
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
        
        return {
            "id": "img_sol_1",
            "solution": [
                {"id": 1, "math": "Analysis", "explanation": text[:200] + "..."}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
