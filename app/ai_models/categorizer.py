import cv2
import base64
import json
import re
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
        
        # 1. GPT-4o Vision Prompt (improved: strict JSON schema and decision rules)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an exact, literal classifier for a short video app. "
                    "Choose ONE category from the following exact list (match exactly): "
                    f"{self.categories}. "
                    "Output ONLY a single JSON object that exactly matches this schema: "
                    "{"
                    "\"category\": \"<one of the list>\", "
                    "\"tags\": [list of short tags], "
                    "\"confidence\": <number between 0.0 and 1.0>"
                    "}. \n"
                    "Rules (apply in order): \n"
                    "1) If the video clearly shows food, cooking, street food stalls, or people eating, set \"category\" to \"Food\" (Food has highest priority). \n"
                    "2) If the video is a scripted skit, prank, or stand-up performance, set \"category\" to \"Comedy\". \n"
                    "3) If the video prominently shows professional tools, engineering, construction, or coding screens, set \"category\" to \"Engineer\" or \"Mechanic\" (choose the best fit). \n"
                    "4) Never use \"Science & Education\" for cooking or obvious comedy clips. \n"
                    "5) If multiple categories could apply, pick the most visually dominant or the primary intent (e.g., staged comedy vs. casual eating). \n"
                    "6) Confidence must reflect how visually certain you are (1.0 certain, 0.0 unknown). \n"
                    "7) Tags should be short lowercase phrases describing visible objects or actions (e.g., [\"pizza\", \"street stall\"]). \n"
                    "8) DO NOT include any explanatory text or commentary—ONLY the JSON object. \n"
                    "Examples: \n"
                    "{'category': 'Food', 'tags': ['street food', 'grill'], 'confidence': 0.95} \n"
                    "{'category': 'Comedy', 'tags': ['skit', 'prank'], 'confidence': 0.88}"
                )
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
        
        raw = response.choices[0].message.content
        try:
            result_json = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                try:
                    result_json = json.loads(m.group(0))
                except Exception:
                    result_json = {}
            else:
                result_json = {}

        category = result_json.get("category")
        # Validate category strictness
        if not category or category not in self.categories:
            synonyms = {
                'Chef': 'Food', 'Cooking': 'Food', 'Food & Cooking': 'Food',
                'Science': 'Science & Education', 'Science & Education': 'Science & Education'
            }
            if isinstance(category, str) and category in synonyms:
                category = synonyms[category]
            else:
                category = "Fun"

        tags = result_json.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        # Ensure tags are strings
        tags = [str(t) for t in tags]

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