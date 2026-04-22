import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

SYSTEM_PROMPT = """You are a precise rule-following assistant.
When given a number and a set of rules, apply EVERY rule in order step by step.
Return ONLY the final output value — a number or a word like FIZZ.
No explanation. No working. Just the final answer."""

def security_scrub(query: str) -> str:
    return query

def format_output(text: str) -> str:
    return text.strip()

async def fetch_context(assets: list) -> str:
    contents = []
    async with httpx.AsyncClient(timeout=10.0) as http:
        for url in assets:
            try:
                resp = await http.get(url)
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

        # 3. Build message
        user_message = clean_query
        if context:
            user_message = f"Context:\n{context}\n\nQuestion: {clean_query}"

        # 4. Call Groq LLM
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=50
        )

        result = response.choices[0].message.content

        return {"output": format_output(result)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"output": str(e)})

@app.get("/health")
async def health():
    return {"status": "Service is active"}

@app.get("/")
async def root():
    return {"status": "running"}
