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
            print(f"DEBUG: No vector found for user {user_id}")
            return []

        # 1. Search Milvus
        candidates = milvus_db.search_videos(user_vector, top_k=50)
        
        # 2. Filter by Threshold (Lower it to 0.1 to see everything)
        filtered_ids = []
        for c in candidates:
            # Add this print to see the actual scores in your terminal!
            print(f"DEBUG: Video {c['video_id']} score: {c['score']}")
            
            if c["score"] > 0.1:  # Lowered from 0.7 to 0.1
                filtered_ids.append(c["video_id"])

        filtered_ids = [
            c["video_id"] for c in candidates 
            if c["score"] > 0.35 
        ]
        
        return filtered_ids[:limit]