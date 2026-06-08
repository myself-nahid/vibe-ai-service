from fastapi import FastAPI
from app.api.v1 import feed, onboarding, test_utils
from app.db.milvus_client import milvus_db

app = FastAPI(title="VIBE AI Microservice", version="1.0")

@app.on_event("startup")
async def startup_event():
    milvus_db.connect()

app.include_router(onboarding.router, prefix="/api/v1/user", tags=["Onboarding"])
app.include_router(feed.router, prefix="/api/v1/feed", tags=["Recommendation"])
app.include_router(test_utils.router, prefix="/api/v1/test")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/v1/test/db-status")
def get_db_status():
    from app.db.milvus_client import milvus_db
    from app.db.redis_client import redis_db
    
    # 1. Check Redis
    user_vec = redis_db.get_user_vector("nahid_01")
    
    # 2. Check Milvus
    try:
        # Force a refresh of the count
        milvus_db.client.flush("vibe_videos") 
        stats = milvus_db.client.get_collection_stats(collection_name="vibe_videos")
        count = int(stats.get("row_count", 0))
    except Exception as e:
        count = f"Error: {str(e)}"

    return {
        "videos_in_database": count,
        "user_profile_ready": user_vec is not None
    }