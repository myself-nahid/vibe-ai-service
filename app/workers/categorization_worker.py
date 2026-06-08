import json
import os
from kafka import KafkaConsumer, KafkaProducer
from app.ai_models.categorizer import OpenAIAIModel
from app.services.video_processor import download_video_temp
from app.db.milvus_client import milvus_db
from app.core.config import settings

def start_worker():
    ai_model = OpenAIAIModel()
    milvus_db.connect()

    consumer = KafkaConsumer(
        settings.KAFKA_TOPIC_UPLOADED,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    producer = KafkaProducer(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print("Worker Started: Listening for new videos...")
    
    for message in consumer:
        video_data = message.value
        video_id = video_data.get('video_id')
        video_url = video_data.get('video_url')
        
        temp_path = None
        try:
            print(f"Processing Video: {video_id}")
            temp_path = download_video_temp(video_url)
            
            category, tags, embedding = ai_model.analyze_video(temp_path)
            milvus_db.insert_video(video_id, category, embedding)

            producer.send(settings.KAFKA_TOPIC_CATEGORIZED, {
                "video_id": video_id,
                "ai_category": category,
                "tags": tags,
                "status": "success"
            })
            print(f"Success! {video_id} is '{category}'.")

        except Exception as e:
            print(f"Error on {video_id}: {e}")
            producer.send(settings.KAFKA_TOPIC_CATEGORIZED, {
                "video_id": video_id, "status": "failed", "error": str(e)
            })
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    start_worker()