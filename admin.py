import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import datetime

# --- 1. Initialize Shared Database ---
# We check if the app is already initialized to avoid conflicts if run in same process
if not firebase_admin._apps:
    cred = None
    if os.environ.get("FIREBASE_CREDENTIALS"):
        import json
        service_account_info = json.loads(os.environ.get("FIREBASE_CREDENTIALS"))
        cred = credentials.Certificate(service_account_info)
    elif os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
    
    if cred:
        firebase_admin.initialize_app(cred)

db = firestore.client()

# --- 2. Setup Admin App ---
app = FastAPI(title="Shikhbo AI Admin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Security ---
# Simple hardcoded key for the MVP. In production, use Environment Variable.
ADMIN_SECRET = "shikhbo_admin_secret_2025"

def verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized Admin Access")

# --- 4. Data Models ---

class ChapterItem(BaseModel):
    id: str
    title: str
    titleBn: str

class CurriculumUpdate(BaseModel):
    class_level: str  # "Class_10"
    group: str        # "Science"
    subject: str      # "Physics"
    chapters: List[ChapterItem]

class BookContentUpload(BaseModel):
    class_level: str
    subject: str
    chapter_id: str
    text_content: str

# --- 5. Statistics Endpoints ---

@app.get("/admin/stats", dependencies=[Depends(verify_admin)])
def get_dashboard_stats():
    try:
        # Count Users (This can be slow if users > 1000, but fine for MVP)
        users = db.collection("users").count().get()
        total_users = users[0][0].value
        
        # Estimate Sessions (Just checking a sample or metadata if available)
        # For MVP, we just return user count
        return {
            "total_users": total_users,
            "status": "Healthy",
            "database": "Connected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. User Management Endpoints ---

@app.get("/admin/users", dependencies=[Depends(verify_admin)])
def list_users(limit: int = 50):
    try:
        users_ref = db.collection("users").order_by("last_active", direction=firestore.Query.DESCENDING).limit(limit)
        docs = users_ref.stream()
        
        user_list = []
        for doc in docs:
            d = doc.to_dict()
            user_list.append({
                "id": doc.id,
                "name": d.get("name", "Unknown"),
                "mobile": d.get("mobile", "N/A"),
                "email": d.get("email", "N/A"),
                "class_group": f"{d.get('class_level', '')} {d.get('group', '')}",
                "last_active": d.get("last_active", 0)
            })
        return user_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/users/{user_id}/history", dependencies=[Depends(verify_admin)])
def get_user_full_history(user_id: str):
    try:
        # 1. Get all sessions
        sessions_ref = db.collection("users").document(user_id).collection("chat_sessions")
        session_docs = sessions_ref.stream()
        
        full_history = []
        total_messages = 0
        
        for sess in session_docs:
            s_data = sess.to_dict()
            session_id = sess.id
            
            # Get messages for this session
            msgs = sessions_ref.document(session_id).collection("messages").order_by("timestamp").stream()
            msg_list = []
            for m in msgs:
                m_data = m.to_dict()
                msg_list.append({
                    "role": m_data.get("sender"),
                    "text": m_data.get("text"),
                    "time": m_data.get("timestamp")
                })
                total_messages += 1
                
            full_history.append({
                "session_id": session_id,
                "subject": s_data.get("subject"),
                "chapter": s_data.get("chapter"),
                "messages": msg_list
            })
            
        # Basic Token Estimation (1 word ~= 1.3 tokens)
        # We calculate total text length and estimate cost
        estimated_words = sum([len(m['text'].split()) for s in full_history for m in s['messages']])
        estimated_tokens = int(estimated_words * 1.3)
            
        return {
            "user_id": user_id,
            "total_sessions": len(full_history),
            "total_messages": total_messages,
            "estimated_tokens_used": estimated_tokens,
            "sessions": full_history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. Curriculum & Content Management ---

@app.post("/admin/curriculum/update", dependencies=[Depends(verify_admin)])
def update_curriculum(data: CurriculumUpdate):
    """
    Adds chapters to the curriculum_metadata collection.
    Example: Adds 5 chapters to Class 10 -> Science -> Physics
    """
    try:
        doc_id = f"{data.class_level}_{data.group}".replace(" ", "_")
        doc_ref = db.collection("curriculum_metadata").document(doc_id)
        
        # We use set with merge=True to update specific subject without wiping others
        doc_ref.set({
            data.subject: [c.dict() for c in data.chapters]
        }, merge=True)
        
        return {"status": "success", "message": f"Updated {data.subject} for {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/book/upload", dependencies=[Depends(verify_admin)])
def upload_book_content(data: BookContentUpload):
    """
    Uploads the raw text for RAG.
    Example: Class 10 Physics Chapter 1 Text.
    """
    try:
        # Generate ID exactly like main.py expects
        rag_doc_id = f"{data.class_level}_{data.subject}_{data.chapter_id}".replace(" ", "_")
        
        db.collection("book_content").document(rag_doc_id).set({
            "text_content": data.text_content,
            "updated_at": datetime.datetime.now().isoformat()
        })
        
        return {"status": "success", "message": f"Uploaded content to {rag_doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
