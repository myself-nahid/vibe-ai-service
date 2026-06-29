from fastapi import FastAPI
# 1. Make sure all these are imported correctly
from app.api.v1 import feed, onboarding, test_utils, generation 
from app.db.milvus_client import milvus_db
import time

# 2. Update the Title and Description for your VIBE project
app = FastAPI(
    title="VIBE AI Microservice",
    description="Brain service handling Video AI Categorization, Recommendations, and AI Content Generation.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    # Attempt to connect to Milvus with retries
    for i in range(10):
        try:
            milvus_db.connect()
            print("Successfully connected to Milvus!")
            break
        except Exception as e:
            print(f"Waiting for Milvus... {e}")
            time.sleep(5)

# 3. THIS IS THE CRITICAL PART: You must include all routers here
app.include_router(onboarding.router, prefix="/api/v1/user", tags=["Onboarding"])
app.include_router(feed.router, prefix="/api/v1/feed", tags=["Recommendation"])
app.include_router(generation.router, prefix="/api/v1/ai", tags=["Generation"])
app.include_router(test_utils.router, prefix="/api/v1/test", tags=["Testing Utilities"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "VIBE-AI"}