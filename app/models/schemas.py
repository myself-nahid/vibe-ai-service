from pydantic import BaseModel
from typing import List

class UserOnboardingRequest(BaseModel):
    user_id: str
    categories: List[str]
    interests: List[str]

class FeedResponse(BaseModel):
    user_id: str
    feed: List[str]

class CaptionRequest(BaseModel):
    prompt: str
    style: str = "Casual"  # Default: Casual, Professional, Funny, Inspiring

class CaptionResponse(BaseModel):
    caption: str
    hashtags: List[str]