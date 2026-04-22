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

GROQ_MODEL = "llama3-70b-8192"

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

FILLER_WORDS = {"THE", "IT", "AN", "A", "BY", "TO", "OF", "IS", "IN", "AT", "AND", "OR", "NOT"}


# ─── Normalise Query ──────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Replace unicode arrows and tidy whitespace."""
    text = re.sub(r'[→⇒⟹➜➞]', '->', text)
    text = re.sub(r'\bthen\b', '->', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── Number Extraction ───────────────────────────────────────────────────────

def extract_number(query: str):
    """Try many common phrasings to pull the starting number."""
    patterns = [
        # "Apply rules to 6:" / "Apply rules in order to 6:"
        r"rules?\s+(?:in\s+order\s+)?to\s+(?:input\s+number\s+)?(\-?\d+(?:\.\d+)?)\s*[:\.,]",
        # "input number 6" / "input: 6"
        r"input\s+number\s+(\-?\d+(?:\.\d+)?)",
        r"input\s*[:=]\s*(\-?\d+(?:\.\d+)?)",
        # "number 6" / "number: 6"
        r"\bnumber\s*[:=]?\s*(\-?\d+(?:\.\d+)?)",
        # "starting value 6"
        r"starting\s+(?:value|number)\s+(?:of\s+)?(\-?\d+(?:\.\d+)?)",
        # "value 6"
        r"\bvalue\s+(?:of\s+)?(\-?\d+(?:\.\d+)?)",
        # "x = 6" / "n = 6"
        r"\b[xnN]\s*=\s*(\-?\d+(?:\.\d+)?)",
        # Last resort: first standalone integer before "Rule"
        r"(\-?\d+(?:\.\d+)?)\s*[,.]?\s*Rule",
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

    # AND / OR compound conditions
    if ' and ' in ct:
        parts = re.split(r'\s+and\s+', ct)
        return all(apply_condition(value, p) for p in parts)
    if ' or ' in ct:
        parts = re.split(r'\s+or\s+', ct)
        return any(apply_condition(value, p) for p in parts)

    if "even" in ct:
        return int(value) % 2 == 0
    if "odd" in ct:
        return int(value) % 2 != 0
    if "positive" in ct:
        return value > 0
    if "negative" in ct:
        return value < 0
    if "zero" in ct:
        return value == 0

    m = re.search(r"divisible\s+by\s+(\d+)", ct)
    if m:
        return int(m.group(1)) != 0 and int(value) % int(m.group(1)) == 0

    m = re.search(r"multiple\s+of\s+(\d+)", ct)
    if m:
        return int(m.group(1)) != 0 and int(value) % int(m.group(1)) == 0

    # Comparison operators — handle >= before > etc.
    for op_str, op_fn in [
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("!=", lambda a, b: a != b),
        ("<>", lambda a, b: a != b),
        (">",  lambda a, b: a > b),
        ("<",  lambda a, b: a < b),
        ("==", lambda a, b: a == b),
        ("=",  lambda a, b: a == b),
    ]:
        m = re.search(re.escape(op_str) + r'\s*(\-?\d+(?:\.\d+)?)', ct)
        if m:
            num = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
            return op_fn(value, num)

    # "greater than X", "less than X", "equal to X"
    m = re.search(r"greater\s+than\s+(?:or\s+equal\s+to\s+)?(\-?\d+(?:\.\d+)?)", ct)
    if m:
        num = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
        return value >= num if "equal" in ct else value > num

    m = re.search(r"less\s+than\s+(?:or\s+equal\s+to\s+)?(\-?\d+(?:\.\d+)?)", ct)
    if m:
        num = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
        return value <= num if "equal" in ct else value < num

    m = re.search(r"equal(?:\s+to)?\s+(\-?\d+(?:\.\d+)?)", ct)
    if m:
        num = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
        return value == num

    return False


# ─── Action Executor ─────────────────────────────────────────────────────────

def apply_action(value, action_text: str):
    at = action_text.strip()
    at_lower = at.lower()

    # Priority 1: output the number/value/result/it
    if re.search(r"output\s+(?:the\s+)?(?:number|value|result|it)\b", at_lower):
        return value

    # Priority 2: output a quoted word like "FIZZ" or 'FIZZ'
    m_quoted = re.search(r'output\s+["\']([^"\']+)["\']', at, re.IGNORECASE)
    if m_quoted:
        return m_quoted.group(1).strip().upper()

    # Priority 3: output a bare WORD (ALL-CAPS or mixed) — not a filler
    m_caps = re.search(r'\boutput\s+([A-Za-z]{2,})\b', at)
    if m_caps:
        word = m_caps.group(1).upper()
        if word not in FILLER_WORDS:
            return word

    # Arithmetic
    if "double" in at_lower:
        return value * 2
    if "triple" in at_lower:
        return value * 3
    if "quadruple" in at_lower:
        return value * 4
    if re.search(r"halve|divide\s+by\s+2\b", at_lower):
        return value / 2
    if re.search(r"square\s+it|square\s+the", at_lower):
        return value ** 2
    if re.search(r"\bsquare\b", at_lower) and "root" not in at_lower:
        return value ** 2
    if "square root" in at_lower or "sqrt" in at_lower:
        return value ** 0.5
    if "negate" in at_lower or "flip sign" in at_lower:
        return -value
    if "absolute" in at_lower or "abs(" in at_lower:
        return abs(value)

    m = re.search(r"add\s+(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        return value + (float(n) if '.' in n else int(n))

    m = re.search(r"subtract\s+(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        return value - (float(n) if '.' in n else int(n))

    m = re.search(r"multiply\s+(?:it\s+)?by\s+(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        return value * (float(n) if '.' in n else int(n))

    m = re.search(r"divide\s+(?:it\s+)?by\s+(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        d = float(n) if '.' in n else int(n)
        return value / d if d != 0 else value

    m = re.search(r"(?:set\s+(?:it\s+)?to|becomes?|replace\s+with)\s+(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        return float(n) if '.' in n else int(n)

    m = re.search(r"raise\s+(?:it\s+)?to\s+(?:the\s+)?(?:power\s+of\s+)?(\-?\d+(?:\.\d+)?)", at_lower)
    if m:
        n = m.group(1)
        return value ** (float(n) if '.' in n else int(n))

    m = re.search(r"(?:mod|modulo|modulus)\s+(\d+)", at_lower)
    if m:
        return int(value) % int(m.group(1))

    # "+ N" or "- N" as a standalone action (e.g., "+5" or "-3")
    m = re.search(r'^\s*([+\-])\s*(\d+(?:\.\d+)?)\s*$', at)
    if m:
        op, n = m.group(1), m.group(2)
        num = float(n) if '.' in n else int(n)
        return value + num if op == '+' else value - num

    return value  # no-op fallback


# ─── Rule Block Processor ────────────────────────────────────────────────────

def process_rule_block(value, rule_text: str):
    """
    Handle if/else-if/else chains within a rule block.
    Supports '-> action', 'else action' (no arrow), and plain action blocks.
    """
    norm = normalise(rule_text)
    # Normalise "otherwise" -> "else"
    norm = re.sub(r'\botherwise\b', 'else', norm, flags=re.IGNORECASE)

    branches = []  # list of (condition_text, action_text)

    # ── Strategy: insert '->' before action if missing ───────────────────────
    # Handles: "if even double it, else add 10"
    # Step 1: ensure "if <cond>" is followed by '->'
    # Step 2: ensure "else" is followed by '->'
    def ensure_arrows(text):
        # Insert '->' after 'if <cond>' when missing an arrow
        # Pattern: 'if <cond> <action_word>' where action_word is a known verb
        action_verbs = (
            r'double|triple|quadruple|halve|square|negate|add|subtract|multiply|divide|'
            r'set|output|mod|modulo|raise|replace|becomes?|abs'
        )
        # Insert -> after condition if missing
        text = re.sub(
            r'((?:else\s+)?if\s+[^->]+?)\s+(' + action_verbs + r')',
            lambda m: m.group(1) + ' -> ' + m.group(2),
            text,
            flags=re.IGNORECASE
        )
        # Insert -> after 'else' if missing
        text = re.sub(
            r'\belse\s+(?!->|if\b)(' + action_verbs + r')',
            lambda m: 'else -> ' + m.group(1),
            text,
            flags=re.IGNORECASE
        )
        return text

    norm = ensure_arrows(norm)

    # Extract all "if <cond> -> <action>" including "else if"
    clause_pattern = re.compile(
        r'(?:else\s+)?if\s+(.+?)\s*->\s*(.+?)(?=\s*\.?\s*(?:else\b|$))',
        re.IGNORECASE | re.DOTALL
    )
    # Match "else -> <action>" (not preceded by 'if')
    else_pattern = re.compile(
        r'\belse\s*->\s*(.+?)(?=\.?\s*$)',
        re.IGNORECASE | re.DOTALL
    )

    for m in clause_pattern.finditer(norm):
        branches.append((m.group(1).strip(), m.group(2).strip()))

    m_else = else_pattern.search(norm)
    # Make sure the else match isn't part of an "else if"
    if m_else:
        # Check there's no 'if' right after 'else ->'
        else_action_text = m_else.group(1).strip()
        if re.match(r'if\b', else_action_text, re.IGNORECASE):
            else_action = None
        else:
            else_action = else_action_text
    else:
        else_action = None

    # No if-branches found: treat block as plain action or else-only
    if not branches:
        if else_action:
            value = apply_action(value, else_action)
        else:
            value = apply_action(value, norm)
        return value

    # Evaluate branches — first match wins
    applied = False
    for cond_text, action_text in branches:
        if apply_condition(value, cond_text):
            value = apply_action(value, action_text)
            applied = True
            break

    if not applied and else_action:
        value = apply_action(value, else_action)

    return value


# ─── Main Parser ─────────────────────────────────────────────────────────────

def parse_and_execute_rules(query: str):
    """
    Deterministic rule parser. Returns final result string or None on failure.
    """
    try:
        query = normalise(query)
        number = extract_number(query)
        if number is None:
            return None

        # Split on "Rule N:" with optional space between Rule and number
        rule_blocks = re.split(r'Rule\s*\d+\s*:', query, flags=re.IGNORECASE)
        rules_raw = rule_blocks[1:]
        if not rules_raw:
            return None

        value = number
        for rule_text in rules_raw:
            rule_text = rule_text.strip().rstrip('.')
            value = process_rule_block(value, rule_text)

        # Normalise whole float -> int
        if isinstance(value, float) and value == int(value):
            value = int(value)

        # Round floats to reasonable precision
        if isinstance(value, float):
            value = round(value, 6)
            if value == int(value):
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
