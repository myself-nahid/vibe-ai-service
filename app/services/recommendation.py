from app.db.milvus_client import milvus_db
from app.db.redis_client import redis_db
from app.ai_models.categorizer import OpenAIAIModel

class RecommendationService:
    def __init__(self):
        self.ai = OpenAIAIModel()

    def initialize_user(self, user_id: str, categories: list, interests: list):
        """Called during Onboarding to generate initial profile vector."""
        pref_string = f"User likes: {', '.join(categories)} and {', '.join(interests)}."
        user_vector = self.ai.embed_text(pref_string)
        redis_db.set_user_vector(user_id, user_vector)

    def generate_feed(self, user_id: str, limit: int = 20) -> list:
        user_vector = redis_db.get_user_vector(user_id)
        
        if not user_vector:
            # Fallback if user skipped onboarding: Neutral vector or random
            user_vector = [0.0] * 1536 

        candidates = milvus_db.search_videos(user_vector, top_k=limit * 2)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return [c["video_id"] for c in candidates[:limit]]