import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Request
import google.generative_ai as genai
from pydantic import BaseModel

# 1. Initialize Firebase (Database)
# On Render, we will use an Environment Variable. Locally, use the file.
cred = None
if os.path.exists("serviceAccountKey.json"):
    cred = credentials.Certificate("serviceAccountKey.json")
else:
    # This logic handles Render's environment variable approach
    import json
    service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
    cred = credentials.Certificate(service_account_info)

firebase_admin.initialize_app(cred)
db = firestore.client()

# 2. Initialize Gemini (AI)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE_FOR_TESTING"))
model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-09-2025')

# 3. Setup App
app = FastAPI()

# --- Data Models ---
class ChatRequest(BaseModel):
    user_id: str
    message: str
    class_level: str  # e.g., "class_10"
    subject: str      # e.g., "physics"
    chapter: str      # e.g., "ch3"

# --- Endpoints ---

@app.get("/")
def home():
    return {"status": "Shikhbo AI Backend is Running"}

@app.post("/chat")
async def chat_tutor(request: ChatRequest):
    try:
        # A. RAG STEP: Fetch the Book Content from Firestore
        # We construct the ID: "class_10_physics_ch3"
        doc_id = f"{request.class_level}_{request.subject}_{request.chapter}"
        doc_ref = db.collection("book_content").document(doc_id)
        doc = doc_ref.get()

        book_context = ""
        if doc.exists:
            book_context = doc.to_dict().get("text_content", "")
        else:
            print(f"Warning: No book found for {doc_id}")
            # Fallback: AI uses general knowledge if book is missing

        # B. Construct the Prompt
        prompt = f"""
        System: You are a Bangladeshi Tutor. Use the provided book context to answer.
        Context: {book_context[:20000]}  # Limit context to avoid errors
        
        Student: {request.message}
        """

        # C. Call Gemini
        response = model.generate_content(prompt)
        
        # D. Save Chat History to Firestore (Optional - for History tab)
        # db.collection("users").document(request.user_id).collection("history").add({...})

        return {"reply": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
