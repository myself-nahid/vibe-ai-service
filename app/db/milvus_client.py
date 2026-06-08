from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.core.config import settings

class MilvusDB:
    def __init__(self):
        self.collection_name = "vibe_videos"
        self.dim = 1536 # Output dimension for OpenAI 'text-embedding-3-small'

    def connect(self):
        connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self._init_collection()

    def _init_collection(self):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="VIBE Video Embeddings")
            self.collection = Collection(self.collection_name, schema)
            
            index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            self.collection.create_index(field_name="embedding", index_params=index_params)
        else:
            self.collection = Collection(self.collection_name)
        self.collection.load()

    def insert_video(self, video_id: str, category: str, embedding: list):
        data = [[video_id], [category], [embedding]]
        self.collection.insert(data)

    def search_videos(self, user_embedding: list, top_k: int = 50) -> list:
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[user_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["video_id", "category"]
        )
        return [{"video_id": hit.entity.get("video_id"), "score": hit.distance} for hit in results[0]]

milvus_db = MilvusDB()