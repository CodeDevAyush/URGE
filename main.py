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
                    "content": "You are a precise assistant. Answer with ONLY the name or number asked. No explanation, no punctuation, no extra text. Just the single word or number answer. If multiple people have highest score then give all highest scorer names."
                },
                {"role": "user", "content": req.query}
            ]
        )
        return {"output": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
