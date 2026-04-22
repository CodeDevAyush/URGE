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

    # Match "Alice scored 80" or "Alice 80"
    entries = re.findall(r'([A-Za-z]+)\s+scored\s+(\d+)', query, re.IGNORECASE)

    if not entries:
        entries = re.findall(r'([A-Za-z]+)\s+(\d+)', query, re.IGNORECASE)
        # Remove common words
        skip = {"scored", "got", "has", "with", "numbers", "sum", "level", "who", "and"}
        entries = [(name, score) for name, score in entries if name.lower() not in skip]

    if not entries:
        return None

    # Aggregate per person - floor average
    mp = defaultdict(lambda: [0, 0])
    for name, marks in entries:
        mp[name][0] += int(marks)
        mp[name][1] += 1

    # Compute floor average per person
    averages = {name: total // count for name, (total, count) in mp.items()}

    is_lowest = any(w in q for w in ["lowest", "least", "minimum", "worst", "fewest"])
    is_highest = any(w in q for w in ["highest", "most", "maximum", "best", "won", "winner", "top", "greater", "higher"])

    if is_lowest:
        target = min(averages.values())
        winners = [name for name, avg in averages.items() if avg == target]
        if len(winners) > 1:
            return "Both"
        return winners[0]

    if is_highest:
        target = max(averages.values())
        winners = [name for name, avg in averages.items() if avg == target]
        if len(winners) > 1:
            return "Both"
        return winners[0]

    return None

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
                    "content": """You are a precise assistant. Rules:
1. Return ONLY a single word or number.
2. No explanation, no punctuation.
3. If multiple scores per person, compute floor average then compare.
4. Return the NAME of the person with highest/lowest average.
5. If equal averages, return Both."""
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
