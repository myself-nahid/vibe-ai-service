from fastapi import APIRouter
from app.models.schemas import UserOnboardingRequest
from app.services.recommendation import RecommendationService

router = APIRouter()
rec_service = RecommendationService()

@router.post("/preferences")
async def save_user_preferences(data: UserOnboardingRequest):
    """Saves onboarding choices and converts them to a vector."""
    rec_service.initialize_user(data.user_id, data.categories, data.interests)
    return {"status": "success", "message": "AI profile initialized"}