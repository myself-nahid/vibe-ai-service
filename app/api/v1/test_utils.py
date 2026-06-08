from fastapi import APIRouter
import json
from kafka import KafkaProducer
from app.core.config import settings

router = APIRouter()

@router.post("/simulate-upload")
async def simulate_upload(video_id: str, video_url: str):
    producer = KafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        api_version=(2, 0, 2) 
    )
    payload = {"video_id": video_id, "video_url": video_url}
    
    # Send the message
    future = producer.send(settings.KAFKA_TOPIC_UPLOADED, payload)
    
    # FORCE the message to be sent immediately
    producer.flush() 
    
    return {"status": "Message sent to Kafka", "payload": payload}