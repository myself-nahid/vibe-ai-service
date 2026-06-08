from pymilvus import MilvusClient, DataType
from app.core.config import settings

class MilvusDB:
    def __init__(self):
        self.collection_name = "vibe_videos"
        self.dim = 1536 # Output dimension for OpenAI 'text-embedding-3-small'
        self.client = None

    def connect(self):
        # Use the modern MilvusClient API
        uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        self.client = MilvusClient(uri=uri)
        self._init_collection()

    def _init_collection(self):
        if not self.client.has_collection(collection_name=self.collection_name):
            # Define schema
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=50)
            schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

            # Define index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", 
                metric_type="COSINE", 
                index_type="IVF_FLAT", 
                params={"nlist": 128}
            )

            # Create Collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
        
        # Load into memory
        self.client.load_collection(collection_name=self.collection_name)

    def insert_video(self, video_id: str, category: str, embedding: list):
        data = [{"video_id": video_id, "category": category, "embedding": embedding}]
        self.client.insert(collection_name=self.collection_name, data=data)
        # Add this line only for testing/debugging to update the count instantly:
        self.client.flush(collection_name=self.collection_name)

    def search_videos(self, user_embedding: list, top_k: int = 50) -> list:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.client.search(
            collection_name=self.collection_name,
            data=[user_embedding],
            limit=top_k,
            search_params=search_params,
            output_fields=["video_id", "category"]
        )
        
        # Parse modern API results
        parsed_results = []
        for hit in results[0]:
            parsed_results.append({
                "video_id": hit["entity"].get("video_id"),
                "category": hit["entity"].get("category"),
                "score": hit["distance"]
            })
        return parsed_results

milvus_db = MilvusDB()