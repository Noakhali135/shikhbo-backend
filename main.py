import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

# --- Config ---
# Ensure these env vars are set in your deployment (e.g. Render/Railway)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Supabase credentials missing")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    subject: str
    chapter_id: str
    class_level: str
    group: Optional[str] = "Science"
    medium: Optional[str] = "Bangla Medium"

class UploadRequest(BaseModel):
    text: str
    class_level: str
    subject: str
    chapter_id: str

# --- Middleware ---
async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid header")
    token = authorization.split(" ")[1]
    user = supabase.auth.get_user(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    return user.user

async def verify_admin(user=Depends(verify_token)):
    # Check profile role
    res = supabase.table("profiles").select("role").eq("id", user.id).single().execute()
    if not res.data or res.data.get("role") != 'admin':
        raise HTTPException(403, "Admins only")
    return user

# --- Logic ---
def get_rag_context(query: str, class_level: str, subject: str) -> str:
    try:
        # 1. Embed Query
        embed_res = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        vector = embed_res.embeddings[0].values

        # 2. Search DB (RPC Call)
        rpc_res = supabase.rpc("match_textbook_content", {
            "query_embedding": vector,
            "match_threshold": 0.5,
            "match_count": 3,
            "filter_class": class_level,
            "filter_subject": subject
        }).execute()

        chunks = [item['chunk_text'] for item in rpc_res.data]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"RAG Error: {e}")
        return ""

# --- Routes ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest, user=Depends(verify_token)):
    # 1. RAG
    context = get_rag_context(req.message, req.class_level, req.subject)
    
    # 2. System Prompt
    system_instruction = f"""
    You are Shikhbo AI, a friendly tutor for Bangladeshi students.
    
    Context from textbook ({req.subject}):
    {context}
    
    Instructions:
    - Answer based on the context provided.
    - If the answer isn't in the context, use your general knowledge but mention it.
    - Explain simply in a mix of Bangla and English (Banglish) or pure English as preferred.
    - Use LaTeX for math ($...$).
    """

    # 3. Generate
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5
            )
        )
        reply = resp.text
    except Exception as e:
        raise HTTPException(500, f"Gemini Error: {e}")

    # 4. Save History (Async in production)
    # Upsert Session
    supabase.table("sessions").upsert({
        "id": req.session_id,
        "user_id": user.id,
        "subject": req.subject,
        "chapter_id": req.chapter_id,
        "last_active": "now()"
    }).execute()

    # Insert Messages
    supabase.table("messages").insert([
        {"session_id": req.session_id, "role": "user", "content": req.message},
        {"session_id": req.session_id, "role": "ai", "content": reply}
    ]).execute()

    return {"reply": reply}

@app.get("/history")
async def get_history(session_id: str, user=Depends(verify_token)):
    res = supabase.table("messages").select("*").eq("session_id", session_id).order("created_at").execute()
    return {"messages": res.data}

@app.post("/admin/upload")
async def admin_upload(req: UploadRequest, user=Depends(verify_admin)):
    # Simple chunker (every 800 chars)
    chunk_size = 800
    text = req.text
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    count = 0
    for chunk in chunks:
        # Embed
        emb = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        ).embeddings[0].values
        
        # Save
        supabase.table("textbook_content").insert({
            "class_level": req.class_level,
            "subject": req.subject,
            "chapter_id": req.chapter_id,
            "chunk_text": chunk,
            "embedding": emb
        }).execute()
        count += 1
        
    return {"status": "success", "chunks": count}
