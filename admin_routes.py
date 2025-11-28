# admin_routes.py
from fastapi import APIRouter, HTTPException, Query
from database import db  # Import shared DB connection

admin_router = APIRouter(prefix="/admin", tags=["Admin Panel"])

# Helper: Verify Admin Access (Database Check)
def verify_admin_access(user_id: str) -> bool:
    try:
        # 1. Get User Phone
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists: return False
        user_mobile = user_doc.to_dict().get("mobile")

        # 2. Check Allowed List
        admin_doc = db.collection("system_metadata").document("admin_access").get()
        if not admin_doc.exists: return False
        
        allowed_list = admin_doc.to_dict().get("allowed_phones", [])
        return user_mobile in allowed_list
    except Exception as e:
        print(f"Admin Verification Failed: {e}")
        return False

# Endpoint 1: Check Login Status
@admin_router.get("/status")
def check_admin_status(user_id: str):
    is_admin = verify_admin_access(user_id)
    return {"is_admin": is_admin}

# Endpoint 2: Get Total Stats
@admin_router.get("/stats")
def get_dashboard_stats(user_id: str):
    if not verify_admin_access(user_id):
        raise HTTPException(status_code=403, detail="Access Denied")

    try:
        # User Count
        users_count = db.collection("users").count().get()[0][0].value
        # Total Chat Messages (Estimate)
        # Note: Counting subcollections is hard, so we might just return user count for now
        
        return {
            "total_students": int(users_count),
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
