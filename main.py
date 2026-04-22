import os
import re
import httpx
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
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

@app.get("/health")
def health():
    return {"status": "Service is active"}

async def fetch_assets(assets: list) -> str:
    """Fetch content from asset URLs"""
    contents = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in assets:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    contents.append(f"URL: {url}\nContent: {resp.text[:5000]}")
            except Exception:
                continue
    return "\n\n".join(contents)

def get_rank(q: str):
    match = re.search(r'(\d+)\s*(?:st|nd|rd|th)', q)
    if match:
        return int(match.group(1))
    ordinals = {
        "first": 1, "second": 2, "third": 3, "fourth": 4,
        "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
        "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
        "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
        "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
        "nineteenth": 19, "twentieth": 20
    }
    for word, rank in ordinals.items():
        if word in q:
            return rank
    return 1

def extract_entries(query: str):
    entries = re.findall(r'([A-Za-z]+)\s+scored\s+(\d+\.?\d*)', query, re.IGNORECASE)
    if not entries:
        entries = re.findall(r'([A-Za-z]+)\s+got\s+(\d+\.?\d*)', query, re.IGNORECASE)
    if not entries:
        entries = re.findall(r'([A-Za-z]+)\s+has\s+(\d+\.?\d*)', query, re.IGNORECASE)
    if not entries:
        entries = re.findall(r'([A-Za-z]+)\s*:\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if not entries:
        all_pairs = re.findall(r'([A-Za-z]+)\s+(\d+\.?\d*)', query, re.IGNORECASE)
        skip = {
            "scored", "got", "has", "with", "numbers", "sum",
            "who", "and", "the", "is", "are", "was", "what",
            "level", "extract", "find", "calculate", "highest",
            "lowest", "second", "third", "fourth", "fifth",
            "sixth", "seventh", "eighth", "ninth", "tenth",
            "first", "last", "rank", "ranked", "position"
        }
        entries = [(n, s) for n, s in all_pairs if n.lower() not in skip]
    return entries

def deterministic_solver(query: str):
    q = query.lower()
    entries = extract_entries(query)
    if not entries:
        return None

    # Max score per person
    mp = defaultdict(float)
    for name, score in entries:
        val = float(score)
        if val > mp[name]:
            mp[name] = val

    # Floor values
    scores = {name: int(val) for name, val in mp.items()}
    rank = get_rank(q)

    is_lowest = any(w in q for w in [
        "lowest", "least", "minimum", "worst", "fewest", "bottom", "last", "weakest"
    ])
    is_highest = any(w in q for w in [
        "highest", "most", "maximum", "best", "won", "winner",
        "top", "greatest", "higher", "larger", "strongest"
    ])

    if is_lowest:
        sorted_scores = sorted(set(scores.values()))
        if rank > len(sorted_scores):
            return None
        target = sorted_scores[rank - 1]
        winners = [name for name, s in scores.items() if s == target]
        return "Equal" if len(winners) > 1 else winners[0]

    if is_highest:
        sorted_scores = sorted(set(scores.values()), reverse=True)
        if rank > len(sorted_scores):
            return None
        target = sorted_scores[rank - 1]
        winners = [name for name, s in scores.items() if s == target]
        return "Equal" if len(winners) > 1 else winners[0]

    return None

@app.post("/v1/answer")
async def answer(req: Request):
    try:
        query = req.query.strip()

        # Try deterministic first
        result = deterministic_solver(query)
        if result:
            return {"output": result}

        # Fetch asset content if available
        asset_context = ""
        if req.assets:
            asset_context = await fetch_assets(req.assets)

        # Build prompt with asset context
        user_message = query
        if asset_context:
            user_message = f"Context from documents:\n{asset_context}\n\nQuestion: {query}"

        # LLM fallback
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an extremely precise assistant.\n"
                        "Rules:\n"
                        "- Return ONLY the direct answer: single word or number\n"
                        "- No explanation, no punctuation, no sentences\n"
                        "- For ranking: use MAX score per person\n"
                        "- If scores tie at that rank return Equal\n"
                        "- If float answer floor it (9.9 → 9)\n"
                        "- If assets/documents provided use them to answer\n"
                        "- Never write more than one word unless names"
                    )
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=50
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'[^\w\s]', '', raw).strip()
        tokens = raw.split()
        return {"output": tokens[0] if tokens else raw}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
