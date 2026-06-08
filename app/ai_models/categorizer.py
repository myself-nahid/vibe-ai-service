import cv2
import base64
import json
from openai import OpenAI
from app.core.config import settings

class OpenAIAIModel:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.categories = [
            "Accountant", "Actor", "Architect", "Banker", "Chef", "Cricketer", 
            "Driver", "Electrician", "Engineer", "Journalist", "Lawyer", "Mechanic", 
            "Politician", "Student", "Welder", "Art", "Auto", "Comedy", "DIY", 
            "Dance", "Food", "Fun", "Gaming", "Life Hacks", "Music", 
            "Beauty & Style", "oddly Satisfying", "Science & Education"
        ]

    def extract_base64_frames(self, video_path: str, num_frames: int = 4) -> list:
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // num_frames, 1)
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (512, 512))
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        return frames

    def analyze_video(self, video_path: str):
        base64_frames = self.extract_base64_frames(video_path)
        
        # 1. GPT-4o Vision Prompt
        messages = [
            {
                "role": "system",
                "content": f"You are a video categorization AI. Return exactly ONE JSON object with 'category' (must be exactly from this list: {self.categories}) and 'tags' (list of descriptive strings)."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What category does this video belong to?"}]
            }
        ]
        for b64 in base64_frames:
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})

        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, response_format={"type": "json_object"}
        )
        
        result_json = json.loads(response.choices[0].message.content)
        category = result_json.get("category", "Fun")
        tags = result_json.get("tags", [])

        # 2. Text Embedding for Vector DB
        embed_response = self.client.embeddings.create(
            input=f"Category: {category}. Tags: {', '.join(tags)}.",
            model="text-embedding-3-small"
        )
        return category, tags, embed_response.data[0].embedding

    def embed_text(self, text: str) -> list:
        """Helper to embed user onboarding profiles."""
        response = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding