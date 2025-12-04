import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
from google.genai import types

# --- Configuration ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # Use Service Role for Admin tasks
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing Environment Variables")

# --- Initialize Clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    subject: str
    chapter_id: str
    class_level: str
    group: Optional[str] = "Science"
    medium: Optional[str] = "Bangla Medium"

class UploadTextRequest(BaseModel):
    text: str
    class_level: str
    subject: str
    chapter_id: str

# --- Auth Middleware ---
async def verify_token(authorization: str = Header(...)):
    """Verifies the Supabase JWT sent from Frontend."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid header format")
    
    token = authorization.split(" ")[1]
    user = supabase.auth.get_user(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Token")
    
    return user.user

async def verify_admin(user=Depends(verify_token)):
    """Checks if the authenticated user is an admin."""
    # Fetch profile to check role
    response = supabase.table("profiles").select("role").eq("id", user.id).single().execute()
    if not response.data or response.data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return user

# --- Vector Search Service ---
def query_vectors(query_text: str, class_level: str, subject: str) -> str:
    """
    1. Embeds the user query using Gemini.
    2. Searches Supabase vectors.
    3. Returns combined text context.
    """
    try:
        # Embed query
        embed_resp = client.models.embed_content(
            model="text-embedding-004",
            contents=query_text
        )
        query_vector = embed_resp.embeddings[0].values

        # Call Supabase RPC function 'match_textbook_content'
        # You must define this function in SQL first (see SQL snippet below)
        response = supabase.rpc(
            "match_textbook_content",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.5,
                "match_count": 3,
                "filter_class": class_level,
                "filter_subject": subject
            }
        ).execute()

        chunks = [item['chunk_text'] for item in response.data]
        return "\n\n".join(chunks)

    except Exception as e:
        print(f"Vector Search Error: {e}")
        return ""

# --- Chat Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, user=Depends(verify_token)):
    
    # 1. Retrieve Context (RAG)
    context_text = query_vectors(request.message, request.class_level, request.subject)
    
    # 2. Construct Prompt
    system_instruction = f"""
    You are a helpful AI Tutor for Bangladeshi students.
    Context from textbook:
    {context_text}
    
    Answer the student's question based on the context above.
    If the answer is not in the context, use your general knowledge but mention that it's outside the provided text.
    Use Markdown for formatting.
    """

    # 3. Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7
            )
        )
        reply_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

    # 4. Save History asynchronously (Fire & Forget logic or await)
    # Ensure Session Exists
    session_check = supabase.table("sessions").select("id").eq("id", request.session_id).execute()
    if not session_check.data:
        supabase.table("sessions").insert({
            "id": request.session_id,
            "user_id": user.id,
            "subject": request.subject,
            "chapter_id": request.chapter_id
        }).execute()
    
    # Insert Messages
    supabase.table("messages").insert([
        {"session_id": request.session_id, "role": "user", "content": request.message},
        {"session_id": request.session_id, "role": "ai", "content": reply_text}
    ]).execute()

    # Update Session Last Active
    supabase.table("sessions").update({"last_active": "now()"}).eq("id", request.session_id).execute()

    return {"reply": reply_text}

# --- History Endpoint ---
@app.get("/history")
async def get_history(session_id: str, user=Depends(verify_token)):
    response = supabase.table("messages")\
        .select("*")\
        .eq("session_id", session_id)\
        .order("created_at")\
        .execute()
    return {"messages": response.data}

# --- Admin: Upload Text & Vectorize ---
@app.post("/admin/upload")
async def upload_content(request: UploadTextRequest, user=Depends(verify_admin)):
    """
    Splits text into chunks, embeds them, and saves to DB.
    """
    # Simple chunking by 1000 characters (Improve this with a proper splitter if needed)
    text_chunks = [request.text[i:i+1000] for i in range(0, len(request.text), 1000)]
    
    inserted_count = 0
    
    for chunk in text_chunks:
        # Embed
        embed_resp = client.models.embed_content(
            model="text-embedding-004",
            contents=chunk
        )
        embedding = embed_resp.embeddings[0].values
        
        # Save
        supabase.table("textbook_content").insert({
            "class_level": request.class_level,
            "subject": request.subject,
            "chapter_id": request.chapter_id,
            "chunk_text": chunk,
            "embedding": embedding
        }).execute()
        
        inserted_count += 1
        
    return {"status": "success", "chunks_processed": inserted_count}
