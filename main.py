import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

SYSTEM_PROMPT = "You are an extremely precise assistant. Return ONLY the direct answer as a single word or number. No explanation, no punctuation, no extra text."

def security_scrub(query: str) -> str:
    return query

def format_output(text: str) -> str:
    return text.strip()

async def fetch_context(assets: list) -> str:
    contents = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in assets:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    contents.append(resp.text[:5000])
            except Exception:
                continue
    return "\n\n".join(contents)

@app.post("/v1/answer")
async def answer(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        assets = body.get("assets", [])

        if not query:
            return JSONResponse(status_code=400, content={"output": "Query required."})

        # 1. Scrub the input
        clean_query = security_scrub(query)

        # 2. Gather context
        context = await fetch_context(assets)

        # 3. Build message with context
        user_message = clean_query
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {clean_query}"

        # 4. Call Claude API
        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 100,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_message}
                    ]
                }
            )
            data = response.json()
            result = data["content"][0]["text"]

        return {"output": format_output(result)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"output": "Internal server error."})

@app.get("/health")
async def health():
    return {"status": "Service is active"}

@app.get("/")
async def root():
    return {"status": "running"}
