import re
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# NORMALIZE TEXT
# ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = re.sub(r'[→⇒⟹➜➞]', '->', text)
    text = re.sub(r'\bthen\b', '->', text, flags=re.IGNORECASE)
    text = re.sub(r'\botherwise\b', 'else', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# EXTRACT NUMBER
# ─────────────────────────────────────────────────────────────
def extract_number(query: str):
    patterns = [
        r'input\s+number\s+(-?\d+)',
        r'number\s+(-?\d+)',
        r'to\s+(-?\d+)',
        r'(-?\d+)\s*[:.,]?\s*Rule',
    ]
    for p in patterns:
        m = re.search(p, query, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────
# CONDITIONS
# ─────────────────────────────────────────────────────────────
def check_condition(value, cond: str):
    cond = cond.lower()

    if "even" in cond:
        return value % 2 == 0
    if "odd" in cond:
        return value % 2 != 0

    m = re.search(r'divisible by (\d+)', cond)
    if m:
        return value % int(m.group(1)) == 0

    if ">" in cond:
        return value > int(re.findall(r'-?\d+', cond)[0])
    if "<" in cond:
        return value < int(re.findall(r'-?\d+', cond)[0])

    return False


# ─────────────────────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────────────────────
def apply_action(value, action: str):
    action = action.lower().strip()

    # OUTPUT
    if "output" in action:
        if "fizz" in action:
            return "FIZZ"
        if "number" in action or "value" in action:
            return value

    # MATH
    if "double" in action:
        return value * 2
    if "add" in action:
        n = int(re.findall(r'-?\d+', action)[0])
        return value + n
    if "subtract" in action:
        n = int(re.findall(r'-?\d+', action)[0])
        return value - n

    # SYMBOLIC
    if re.match(r'\+\s*\d+', action):
        return value + int(re.findall(r'\d+', action)[0])
    if re.match(r'-\s*\d+', action):
        return value - int(re.findall(r'\d+', action)[0])
    if re.match(r'\*\s*\d+', action):
        return value * int(re.findall(r'\d+', action)[0])
    if re.match(r'/\s*\d+', action):
        d = int(re.findall(r'\d+', action)[0])
        return value // d if d != 0 else value

    return value


# ─────────────────────────────────────────────────────────────
# PROCESS RULE
# ─────────────────────────────────────────────────────────────
def process_rule(value, rule: str):
    rule = normalize(rule)

    # if-else format
    if "if" in rule:
        parts = re.split(r'else', rule, flags=re.IGNORECASE)

        # IF PART
        m = re.search(r'if (.*?) -> (.*)', parts[0], re.IGNORECASE)
        if m:
            cond = m.group(1)
            act = m.group(2)
            if check_condition(value, cond):
                return apply_action(value, act)

        # ELSE PART
        if len(parts) > 1:
            act = parts[1].replace("->", "").strip()
            return apply_action(value, act)

    # fallback direct action
    return apply_action(value, rule)


# ─────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────
def solve(query: str):
    query = normalize(query)

    value = extract_number(query)
    if value is None:
        return "0"

    # robust splitting
    rules = re.split(
        r'(?:Rule\s*\d+\s*:|\d+\)|Step\s*\d+\s*:)',
        query,
        flags=re.IGNORECASE
    )[1:]

    for rule in rules:
        value = process_rule(value, rule.strip())

    return str(value).strip().upper()


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────
@app.post("/v1/answer")
async def answer(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(status_code=400, content={"output": "0"})

        result = solve(query)

        return {"output": result}

    except Exception:
        return {"output": "0"}


@app.get("/")
def home():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}
