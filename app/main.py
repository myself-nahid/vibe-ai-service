from fastapi import FastAPI
from app.api.v1 import feed, onboarding
from app.db.milvus_client import milvus_db

app = FastAPI(title="VIBE AI Microservice", version="1.0")

@app.on_event("startup")
async def startup_event():
    milvus_db.connect()

app.include_router(onboarding.router, prefix="/api/v1/user", tags=["Onboarding"])
app.include_router(feed.router, prefix="/api/v1/feed", tags=["Recommendation"])

@app.get("/health")
def health_check():
    return {"status": "ok"}