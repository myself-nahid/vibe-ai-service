from fastapi import APIRouter
from app.models.schemas import FeedResponse
from app.services.recommendation import RecommendationService

router = APIRouter()
rec_service = RecommendationService()

@router.get("/recommend", response_model=FeedResponse)
async def get_for_you_feed(user_id: str, limit: int = 20):
    """Returns AI-recommended video IDs for the feed."""
    video_ids = rec_service.generate_feed(user_id, limit)
    return FeedResponse(user_id=user_id, feed=video_ids)