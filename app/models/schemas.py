from pydantic import BaseModel
from typing import List

class UserOnboardingRequest(BaseModel):
    user_id: str
    categories: List[str]
    interests: List[str]

class FeedResponse(BaseModel):
    user_id: str
    feed: List[str]