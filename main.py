import os
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Ì¥• STRONG SYSTEM PROMPT
SYSTEM_PROMPT = """You are a strict rule engine.

Follow ALL rules step by step EXACTLY.

Return ONLY:
- A number (e.g., 21)
- OR a single word (e.g., FIZZ)

DO NOT include:
- explanation
- punctuation
- spaces
- newlines
- extra text

FINAL ANSWER ONLY.
"""

# Ì¥í STRICT OUTPUT CLEANER
def clean_output(text: str) -> str:
    text = text.strip()

    # Extract ONLY valid answer
    match = re.search(r'\b(FIZZ|\-?\d+)\b', text, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    return "0"


@app.post("/v1/answer")
async def answer(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").strip()

        if not query:
            return JSONResponse(status_code=400, content={"output": "0"})

        # ‚ùå NO CONTEXT (important)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=20
        )

        raw = response.choices[0].message.content

        result = clean_output(raw)

        return {"output": result}

    except Exception:
        return {"output": "0"}


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
