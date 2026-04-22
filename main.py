import os
import re
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
                    "content": "You are an extremely concise assistant. Reply with ONLY the single word or number that directly answers the question. No sentences. No explanation. No punctuation. Just one word or number. For example if asked 'Who scored highest?' reply only 'Bob'."
                },
                {"role": "user", "content": req.query}
            ],
            temperature=0.0,
            max_tokens=20
        )

        raw = response.choices[0].message.content

        raw = raw.strip()

        raw = re.sub(r'[^\w\s]', '', raw)

        words = raw.split()
        if words:
            result = words[0]
        else:
            result = raw

        return {"output": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
