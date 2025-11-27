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
# Check for Render Environment Variable first (Production)
if os.environ.get("FIREBASE_CREDENTIALS"):
    import json
    service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(service_account_info)
# Fallback to local file (Development)
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

# Allow ALL origins (Crucial for Mobile/Web Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Gemini Configuration (Raw HTTP) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Using Flash Lite as requested
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

# --- 5. Helper Functions ---
def call_gemini_raw(system_instruction: str, prompt: str):
    # Constructing payload with System Instruction for persona
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

# --- NEW: GET DYNAMIC CURRICULUM ---
@app.get("/curriculum")
def get_curriculum(class_level: str, group: str):
    """
    Fetches the dynamic chapter list from Firestore.
    The Frontend uses this to update its local cache silently.
    
    Database Structure Expected:
    Collection: 'curriculum_metadata'
    Document ID: 'Class_10_Science' (spaces replaced by underscores)
    Fields: { 'Physics': [...], 'Chemistry': [...] }
    """
    try:
        # Standardize ID: "Class 10" -> "Class_10"
        doc_id = f"{class_level}_{group}".replace(" ", "_")
        
        doc_ref = db.collection("curriculum_metadata").document(doc_id)
        doc = doc_ref.get()
        
        if doc.exists:
            # Returns the full map of subjects and chapters
            return doc.to_dict()
        else:
            # If not found, return empty dict (Frontend handles fallback to local defaults)
            return {}

    except Exception as e:
        print(f"Curriculum Error: {e}")
        # Don't crash the app, just return empty so it uses local default
        return {}

# --- GET SESSIONS (History List) ---
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
                "chapter": data.get("chapter_title", data.get("chapter", "Unknown")),
                "updated_at": data.get("updated_at", 0),
            })
            
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"sessions": sessions}

    except Exception as e:
        print(f"Sessions Error: {e}")
        return {"sessions": []}

# --- GET HISTORY (Messages inside a chapter) ---
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

# --- POST CHAT (The Main Logic) ---
@app.post("/chat")
def chat_tutor(request: ChatRequest):
    try:
        # 1. Logic Layer: Pacific Chapter Selection
        session_id = f"{request.subject}_{request.chapter_id}"
        
        user_doc_ref = db.collection("users").document(request.user_id)
        session_ref = user_doc_ref.collection("chat_sessions").document(session_id)
        messages_ref = session_ref.collection("messages")
        
        current_ts = int(time.time() * 1000)

        # 2. Update Session Metadata
        session_ref.set({
            "subject": request.subject,
            "chapter": request.chapter_id, 
            "group": request.group,
            "updated_at": current_ts,
            "last_message": request.message[:50]
        }, merge=True)

        # 3. Save User Message
        messages_ref.add({
            "text": request.message, 
            "sender": "user", 
            "timestamp": current_ts
        })

        # 4. RAG: Fetch "Pacific Chapter" Content
        rag_doc_id = f"{request.class_level}_{request.subject}_{request.chapter_id}"
        rag_doc_id = rag_doc_id.replace(" ", "_")
        
        book_doc = db.collection("book_content").document(rag_doc_id).get()
        
        if book_doc.exists:
            book_context = book_doc.to_dict().get("text_content", "")
            print(f"Loaded Context: {len(book_context)} chars")
        else:
            book_context = "No specific book content found for this chapter. Answer based on general knowledge of the curriculum."
            print(f"Context Missing for: {rag_doc_id}")

        # 5. Fetch Recent History (Memory)
        docs = messages_ref.limit(5).stream()
        msgs_list = []
        for d in docs: msgs_list.append(d.to_dict())
        msgs_list.sort(key=lambda x: x['timestamp'])
        
        history_text = ""
        for m in msgs_list:
            role = "Student" if m['sender'] == 'user' else "Tutor"
            history_text += f"{role}: {m['text']}\n"

        # 6. Construct Prompt
        system_instruction = f"""
        You are a friendly, encouraging private tutor for a Bangladeshi student in {request.class_level} ({request.group}).
        - Speak in a mix of Bangla and English (Tanglish) naturally.
        - Your goal is to explain concepts clearly.
        - STRICTLY use the provided 'Book Context' to answer. If the answer is not in the context, politely say so.
        - Be concise.
        """

        full_prompt = f"""
        BOOK CONTEXT:
        {book_context[:15000]} 

        CHAT HISTORY:
        {history_text}

        STUDENT QUESTION:
        {request.message}
        """
        
        # 7. Call Gemini
        ai_reply = call_gemini_raw(system_instruction, full_prompt)

        # 8. Save AI Reply
        messages_ref.add({
            "text": ai_reply, 
            "sender": "ai", 
            "timestamp": current_ts + 1
        })

        return {"reply": ai_reply}

    except Exception as e:
        print(f"Chat Logic Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- POST VISION ---
@app.post("/analyze-image")
def analyze_image(request: ImageRequest):
    try:
        prompt = "Analyze this image (likely a math problem or diagram from BD syllabus). Solve it step-by-step."
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
        
        # Mocking the JSON structure expected by frontend
        return {
            "id": "img_sol_1",
            "solution": [
                {"id": 1, "math": "Analysis", "explanation": text[:200] + "..."}
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
