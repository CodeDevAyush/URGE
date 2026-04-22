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
    q = query.lower()

    matches = re.findall(r'([a-zA-Z]+)\s+scored\s+(\d+)', query, re.IGNORECASE)
    if matches:
        parsed = [(name, int(score)) for name, score in matches]
        if "lowest" in q or "least" in q or "minimum" in q or "worst" in q:
            return min(parsed, key=lambda x: x[1])[0]
        if "highest" in q or "most" in q or "maximum" in q or "best" in q or "won" in q:
            return max(parsed, key=lambda x: x[1])[0]

    matches = re.findall(r'([a-zA-Z]+)\s+has\s+(\d+)', query, re.IGNORECASE)
    if matches:
        parsed = [(name, int(score)) for name, score in matches]
        if "lowest" in q or "least" in q or "minimum" in q or "worst" in q:
            return min(parsed, key=lambda x: x[1])[0]
        if "highest" in q or "most" in q or "maximum" in q or "best" in q:
            return max(parsed, key=lambda x: x[1])[0]

    matches = re.findall(r'([a-zA-Z]+)\s*:\s*(\d+)', query, re.IGNORECASE)
    if matches:
        parsed = [(name, int(score)) for name, score in matches]
        if "lowest" in q or "least" in q or "minimum" in q or "worst" in q:
            return min(parsed, key=lambda x: x[1])[0]
        if "highest" in q or "most" in q or "maximum" in q or "best" in q:
            return max(parsed, key=lambda x: x[1])[0]

    matches = re.findall(r'([a-zA-Z]+)\s+got\s+(\d+)', query, re.IGNORECASE)
    if matches:
        parsed = [(name, int(score)) for name, score in matches]
        if "lowest" in q or "least" in q or "minimum" in q or "worst" in q:
            return min(parsed, key=lambda x: x[1])[0]
        if "highest" in q or "most" in q or "maximum" in q or "best" in q:
            return max(parsed, key=lambda x: x[1])[0]

    return None

def clean_output(text: str):
    if not text:
        return ""
    text = text.strip()

    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.strip().split()

    return tokens[0] if tokens else ""

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
                    "content": "You are an extremely concise assistant. Reply with ONLY the single word or number that directly answers the question. No sentences. No explanation. No punctuation. Just one word or number."
                },
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=10
        )

        raw = response.choices[0].message.content.strip()
        final_output = clean_output(raw)
        return {"output": final_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
