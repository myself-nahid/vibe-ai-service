import redis
import json
from app.core.config import settings

class RedisDB:
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    def set_user_vector(self, user_id: str, vector: list):
        # Expiry set to 7 days. It updates every time they use the app.
        self.client.setex(f"user_vector:{user_id}", 604800, json.dumps(vector))

    def get_user_vector(self, user_id: str) -> list:
        data = self.client.get(f"user_vector:{user_id}")
        return json.loads(data) if data else None

redis_db = RedisDB()