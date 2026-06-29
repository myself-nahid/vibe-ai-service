import json
from openai import OpenAI
from app.core.config import settings

class OpenAIAIModel:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_social_content(self, user_prompt: str, style: str):
        """Generates a creative caption and hashtags based on user input and style."""
        
        system_message = (
            f"You are a creative social media manager for the VIBE video platform. "
            f"Generate a post in a '{style}' tone. "
            "Return a JSON object with 'caption' (string) and 'hashtags' (list of strings)."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", # Use mini for fast/cheap text generation
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Describe this content: {user_prompt}"}
            ],
            response_format={ "type": "json_object" }
        )

        content = json.loads(response.choices[0].message.content)
        return content.get("caption"), content.get("hashtags")