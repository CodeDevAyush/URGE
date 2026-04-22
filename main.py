import os
import re
import math
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

def floor_if_float(text: str) -> str:
    """If output is a float, floor it and return as integer string"""
    try:
        value = float(text)
        if '.' in text:
            return str(math.floor(value))
        return str(int(value))
    except ValueError:
        return text

@app.post("/v1/answer")
def answer(req: Request):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise assistant. Follow these rules strictly:
1. Answer with ONLY the name or number asked.
2. No explanation, no punctuation, no extra text.
3. Just the single word or number answer.
4. If multiple people have the highest score give all their names.
5. If the answer is a floating point number, floor it to nearest integer.
For example: 9.9 becomes 9, 3.1 becomes 3, 7.8 becomes 7."""
                },
                {"role": "user", "content": req.query}
            ],
            temperature=0.0,
            max_tokens=20
        )

        raw = response.choices[0].message.content.strip()

        # Floor if float
        result = floor_if_float(raw)

        return {"output": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
