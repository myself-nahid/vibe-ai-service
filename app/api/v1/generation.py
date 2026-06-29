from fastapi import APIRouter
from app.models.schemas import CaptionRequest, CaptionResponse
from app.ai_models.writer import OpenAIAIModel

router = APIRouter()
ai_model = OpenAIAIModel()

@router.post("/generate-caption", response_model=CaptionResponse)
async def generate_caption(data: CaptionRequest):
    """Endpoint for the 'AI Caption Generator' screen in Figma."""
    caption, hashtags = ai_model.generate_social_content(data.prompt, data.style)
    
    return CaptionResponse(
        caption=caption,
        hashtags=hashtags
    )