import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Request(BaseModel):
    query: str
    assets: list = []

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/v1/answer")
def answer(req: Request):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are an answer extraction assistant. 
Rules you MUST follow:
- Reply with ONLY the exact answer, nothing else
- NO sentences, NO explanation, NO punctuation
- NO words like 'The answer is' or 'Based on'
- If answer is a name, return ONLY the name e.g. 'Bob'
- If answer is a number, return ONLY the number e.g. '10'
- If answer is yes/no, return ONLY 'YES' or 'NO'
- If answer is a date, return ONLY the date e.g. '12 March 2024'
- Maximum 3 words in response"""
                },
                {
                    "role": "user",
                    "content": req.query
                }
            ],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content.strip()
        
        # Remove quotes if model adds them
        result = result.replace('"', '').replace("'", '')
        
        # Remove common prefixes model might add
        prefixes = ["Answer:", "Output:", "Result:", "The answer is", "Answer is"]
        for prefix in prefixes:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()

        return {"output": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
