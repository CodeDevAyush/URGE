import os
import re
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
    q = query.lower()

    entries = re.findall(r'([A-Za-z]+)\s+scored\s+(\d+)', query, re.IGNORECASE)
    if not entries:
        entries = re.findall(r'([A-Za-z]+)\s+(\d+)', query, re.IGNORECASE)
        skip = {"scored", "got", "has", "with", "numbers", "sum", "level", "who", "and"}
        entries = [(name, score) for name, score in entries if name.lower() not in skip]

    if not entries:
        return None

    # Use MAX score per person
    mp = defaultdict(float)
    for name, score in entries:
        val = float(score)
        if val > mp[name]:
            mp[name] = val

    # Floor the values
    scores = {name: int(val) for name, val in mp.items()}

    is_lowest = any(w in q for w in ["lowest", "least", "minimum", "worst", "fewest"])
    is_highest = any(w in q for w in ["highest", "most", "maximum", "best", "won", "winner", "top", "greater", "higher"])

    if is_lowest:
        target = min(scores.values())
        winners = [name for name, s in scores.items() if s == target]
        if len(winners) > 1:
            return "Equal"
        return winners[0]

    if is_highest:
        target = max(scores.values())
        winners = [name for name, s in scores.items() if s == target]
        if len(winners) > 1:
            return "Equal"
        return winners[0]

    return None

@app.post("/v1/answer")
def answer(req: Request):
    try:
        query = req.query.strip()

        result = highest_score_solver(query)
        if result:
            return {"output": result}

        # Fallback to LLM - FIXED MODEL
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """You are a precise assistant. Rules:
1. Return ONLY a single word or number.
2. No explanation, no punctuation.
3. Use MAX score per person to compare.
4. Return the NAME of the person with highest/lowest score.
5. If equal scores, return both names separated by space."""
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
