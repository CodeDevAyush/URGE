import os
import re
import math
from collections import defaultdict
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

def highest_score_solver(query: str):
    """Extract name-marks pairs and compute highest floor average"""
    # Match patterns like "Alice 80", "Alice: 80", "Alice scored 80"
    pairs = re.findall(r'([A-Za-z]+)[:\s]+scored\s+(\d+)|([A-Za-z]+)[:\s]+(\d+)', query)

    entries = []
    for match in pairs:
        if match[0] and match[1]:
            entries.append((match[0], int(match[1])))
        elif match[2] and match[3]:
            entries.append((match[2], int(match[3])))

    if not entries:
        # Try simpler pattern: word followed by number
        entries = re.findall(r'([A-Za-z]+)\s+(\d+)', query)
        entries = [(name, int(score)) for name, score in entries
                   if name.lower() not in ["scored", "got", "has", "with", "numbers", "sum", "level"]]

    if not entries:
        return None

    # Aggregate per person
    mp = defaultdict(lambda: [0, 0])
    for name, marks in entries:
        mp[name][0] += marks
        mp[name][1] += 1

    # Compute max floor average
    max_avg = 0
    for total, count in mp.values():
        avg = total // count
        if avg > max_avg:
            max_avg = avg

    return str(max_avg)

@app.post("/v1/answer")
def answer(req: Request):
    try:
        query = req.query.strip()

        # Try deterministic solver first
        result = highest_score_solver(query)
        if result:
            return {"output": result}

        # Fallback to LLM
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise math assistant. Follow these rules:
1. Answer with ONLY a single number or word.
2. No explanation, no punctuation, no extra text.
3. If computing averages, use floor division (round down).
4. If answer is float, floor it to nearest integer.
5. Return only the final number or name."""
                },
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=10
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'[^\w\s]', '', raw)
        tokens = raw.split()
        return {"output": tokens[0] if tokens else raw}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
