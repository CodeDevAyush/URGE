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
async def answer(req: Request):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a logic engine. Your output MUST be restricted to ONLY "
                        "the specific name or number that answers the query. "
                        "Strict constraints: No punctuation, No sentences, No prefixes, No suffixes. "
                        "Example: If 'Alice' is the answer, output only 'Alice' without a period."
                    )
                },
                {"role": "user", "content": req.query}
            ],
            temperature=0.0,  
            max_tokens=15    
        )

        raw_result = response.choices[0].message.content.strip()

        clean_result = re.sub(r'[^\w\s]', '', raw_result)

        parts = clean_result.split()
        final_output = parts[-1] if parts else clean_result

        return {"output": final_output}

    except Exception as e:
        print(f"Log Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
