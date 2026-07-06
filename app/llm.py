import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class LLMService:
    def __init__(self):
       api_key=os.getenv("GEMINI_API_KEY")
       self.model=os.getenv("GEMINI_MODEL")

       if not api_key:
           raise ValueError("Gemini API key is missing in .env")
       if not self.model:
           raise ValueError("GEMINI_MODEL is missing in .env")
       self.client=genai.Client(api_key=api_key)
    
    def generate_answer(self,prompt:str)->str:
        response=self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text