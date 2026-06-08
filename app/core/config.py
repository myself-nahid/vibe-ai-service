from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VIBE AI Service (OpenAI)"
    OPENAI_API_KEY: str
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_UPLOADED: str = "video_uploaded"
    KAFKA_TOPIC_CATEGORIZED: str = "video_categorized"
    
    # DBs
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()