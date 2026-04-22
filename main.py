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

def deterministic_solver(query: str):
    matches = re.findall(r'(\w+)\s+scored\s+(\d+)', query, re.IGNORECASE)
    if matches:
        parsed = [(name, int(score)) for name, score in matches]
        return max(parsed, key=lambda x: x[1])[0]
    return None

def clean_output(text: str):
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.strip().split()
    return tokens[-1] if tokens else ""

@app.post("/v1/answer")
async def answer(req: Request):
    try:
        query = req.query.strip()
        deterministic_result = deterministic_solver(query)
        if deterministic_result:
            return {"output": deterministic_result}

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY the final answer word No explanation No punctuation Single token output"
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=5
        )

        raw = response.choices[0].message.content.strip()
        final_output = clean_output(raw)

        return {"output": final_output}

    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
