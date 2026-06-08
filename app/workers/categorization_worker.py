import time
import json
import os
from kafka import KafkaConsumer, KafkaProducer
from app.ai_models.categorizer import OpenAIAIModel
from app.services.video_processor import download_video_temp
from app.db.milvus_client import milvus_db
from app.core.config import settings

def start_worker():
    ai_model = OpenAIAIModel()

    # --- MILVUS RETRY LOGIC ---
    milvus_retries = 10
    for i in range(milvus_retries):
        try:
            print(f"Connecting to Milvus (Attempt {i+1}/{milvus_retries})...")
            milvus_db.connect()
            print("Successfully connected to Milvus!")
            break
        except Exception as e:
            print(f"Milvus not ready: {e}. Waiting 5s...")
            time.sleep(5)
    
    # --- KAFKA RETRY LOGIC ---
    consumer = None
    producer = None
    kafka_retries = 10
    for i in range(kafka_retries):
        try:
            print(f"Connecting to Kafka (Attempt {i+1}/{kafka_retries})...")
            consumer = KafkaConsumer(
            settings.KAFKA_TOPIC_UPLOADED,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            api_version=(2, 0, 2),
            group_id='vibe-worker-group',
            auto_offset_reset='earliest'  
)
            producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                api_version=(2, 0, 2)   # <--- ADD THIS
            )
            print("Successfully connected to Kafka!")
            break
        except Exception as e:
            print(f"Kafka not ready: {e}. Waiting 5s...")
            time.sleep(5)

    if not consumer or not producer:
        print("Fatal error: Services unavailable.")
        return

    print("Worker Started: Listening for new videos...")
    
    for message in consumer:
        video_data = message.value
        video_id = video_data.get('video_id')
        video_url = video_data.get('video_url')
        
        temp_path = None
        try:
            print(f"Processing Video: {video_id}")
            # Try to download the video
            temp_path = download_video_temp(video_url)
            
            category, tags, embedding = ai_model.analyze_video(temp_path)
            milvus_db.insert_video(video_id, category, embedding)

            producer.send(settings.KAFKA_TOPIC_CATEGORIZED, {
                "video_id": video_id,
                "ai_category": category,
                "tags": tags,
                "status": "success"
            })
            print(f"Success! {video_id} processed.")

        except Exception as e:
            print(f"Error on {video_id}: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    start_worker()