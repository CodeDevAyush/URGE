import os
import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

GROQ_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a precise mathematical rule-following engine with perfect accuracy.

INSTRUCTIONS:
- Read all rules carefully before applying any.
- Apply EVERY rule in EXACT order, one by one.
- At each step, track the current value.
- Only output the FINAL result — a plain number (like 21) or a word (like FIZZ).
- DO NOT include explanation, steps, punctuation, or extra words.
- Your ENTIRE response must be ONLY the final answer.

EXAMPLES:
Input: "Apply rules to 6: Rule1: if even double it, else add 10. Rule2: if >20 subtract 5, else add 3. Rule3: if divisible by 3 output FIZZ else output the number."
Correct output: FIZZ

Input: "Apply rules to 7: Rule1: if even double it, else add 10. Rule2: if >20 subtract 5, else add 3. Rule3: if divisible by 3 output FIZZ else output the number."
Correct output: 20
"""

FILLER_WORDS = {"THE", "IT", "AN", "A", "BY", "TO", "OF", "IS", "IN", "AT"}


# ─── Number Extraction ───────────────────────────────────────────────────────

def extract_number(query: str):
    patterns = [
        r"input\s+number\s+(\-?\d+(?:\.\d+)?)",
        r"number\s+(\-?\d+(?:\.\d+)?)",
        r"to\s+(\-?\d+(?:\.\d+)?)\s*Rule",
        r"to\s+(\-?\d+(?:\.\d+)?)\s*[:\.,]",
        r"input\s*[:=]\s*(\-?\d+(?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            val = m.group(1)
            return float(val) if '.' in val else int(val)
    return None


# ─── Condition Evaluator ─────────────────────────────────────────────────────

def apply_condition(value, condition_text: str) -> bool:
    ct = condition_text.lower().strip()
    if "even" in ct:
        return int(value) % 2 == 0
    if "odd" in ct:
        return int(value) % 2 != 0
    m = re.search(r"divisible\s+by\s+(\d+)", ct)
    if m:
        return int(value) % int(m.group(1)) == 0
    m = re.search(r"([><=!]+)\s*(\-?\d+(?:\.\d+)?)", ct)
    if m:
        op, num = m.group(1), float(m.group(2))
        if op == ">":   return value > num
        if op == "<":   return value < num
        if op == ">=":  return value >= num
        if op == "<=":  return value <= num
        if op in ("=", "=="): return value == num
        if op in ("!=", "<>"): return value != num
    return False


# ─── Action Executor ─────────────────────────────────────────────────────────

def apply_action(value, action_text: str):
    at = action_text.lower().strip()

    # Priority 1: output the number/value/result
    if re.search(r"output\s+the\s+(number|value|result)", at):
        return value

    # Priority 2: output a quoted word like "FIZZ"
    m_quoted = re.search(r'output\s+"([^"]+)"', action_text, re.IGNORECASE)
    if m_quoted:
        return m_quoted.group(1).strip().upper()

    # Priority 3: output a bare ALL-CAPS word (FIZZ, BUZZ…)
    m_caps = re.search(r'\boutput\s+([A-Z]{2,})\b', action_text)
    if m_caps and m_caps.group(1) not in FILLER_WORDS:
        return m_caps.group(1)

    if "double" in at:
        return value * 2
    if "triple" in at:
        return value * 3
    if "halve" in at or re.search(r"divide\s+by\s+2\b", at):
        return value / 2
    if "square" in at:
        return value ** 2

    m = re.search(r"add\s+(\-?\d+(?:\.\d+)?)", at)
    if m:
        n = m.group(1)
        return value + (float(n) if '.' in n else int(n))

    m = re.search(r"subtract\s+(\-?\d+(?:\.\d+)?)", at)
    if m:
        n = m.group(1)
        return value - (float(n) if '.' in n else int(n))

    m = re.search(r"multiply\s+by\s+(\-?\d+(?:\.\d+)?)", at)
    if m:
        n = m.group(1)
        return value * (float(n) if '.' in n else int(n))

    m = re.search(r"divide\s+by\s+(\-?\d+(?:\.\d+)?)", at)
    if m:
        n = m.group(1)
        d = float(n) if '.' in n else int(n)
        return value / d if d != 0 else value

    m = re.search(r"(?:set\s+to|becomes?)\s+(\-?\d+(?:\.\d+)?)", at)
    if m:
        n = m.group(1)
        return float(n) if '.' in n else int(n)

    return value  # no-op fallback


# ─── Rule Block Processor ────────────────────────────────────────────────────

def process_rule_block(value, rule_text: str):
    """
    A rule block may contain multiple if-clauses and an optional otherwise/else.
    Strategy:
      1. Normalise arrows.
      2. Find ALL "if <cond> -> <action>" clauses.
      3. Apply the FIRST one whose condition is true (then stop).
      4. If NONE matched, apply the "otherwise/else -> <action>" if present.
    """
    norm = re.sub(r'[→⇒]', '->', rule_text).strip()

    # Extract all "if <cond> -> <action>" clauses
    # The action ends at a period that is followed by 'if'/'otherwise'/'else', or end of string
    if_clauses = re.findall(
        r'(?<!\w)if\s+(.+?)\s*->\s*(.+?)(?=\.\s*(?:if|otherwise|else\b)|\.?\s*$)',
        norm,
        re.IGNORECASE | re.DOTALL
    )

    # Extract otherwise / else action
    m_else = re.search(
        r'(?:otherwise|else)\s*->\s*(.+?)(?=\.\s*(?:if\b)|\.?\s*$)',
        norm,
        re.IGNORECASE | re.DOTALL
    )

    applied = False
    for cond_text, action_text in if_clauses:
        if apply_condition(value, cond_text.strip()):
            value = apply_action(value, action_text.strip())
            applied = True
            break  # Only the first matching branch fires

    if not applied and m_else:
        value = apply_action(value, m_else.group(1).strip())

    # If the block had NO if-clauses at all, treat whole block as a plain action
    if not if_clauses and not m_else:
        value = apply_action(value, norm)

    return value


# ─── Main Parser ─────────────────────────────────────────────────────────────

def parse_and_execute_rules(query: str):
    """
    Parse rules deterministically.
    Returns the final result as a string, or None if parsing fails.
    """
    try:
        number = extract_number(query)
        if number is None:
            return None

        rule_blocks = re.split(r'Rule\s*\d+\s*:', query, flags=re.IGNORECASE)
        rules_raw = rule_blocks[1:]
        if not rules_raw:
            return None

        value = number
        for rule_text in rules_raw:
            value = process_rule_block(value, rule_text)

        # Normalise whole float
        if isinstance(value, float) and value == int(value):
            value = int(value)

        return str(value)

    except Exception:
        return None


# ─── LLM Fallback ────────────────────────────────────────────────────────────

async def llm_answer(query: str, context: str = "") -> str:
    user_message = query
    if context:
        user_message = f"Context:\n{context}\n\nQuestion:\n{query}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


# ─── Asset Fetcher ───────────────────────────────────────────────────────────

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


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.post("/v1/answer")
async def answer(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        assets = body.get("assets", [])

        if not query:
            return JSONResponse(status_code=400, content={"output": "Query required."})

        result = parse_and_execute_rules(query)

        if result is None:
            context = await fetch_context(assets)
            result = await llm_answer(query, context)

        return {"output": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"output": str(e)})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"status": "running", "model": GROQ_MODEL}
