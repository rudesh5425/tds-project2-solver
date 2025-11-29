import os
import re
import json
import base64
import asyncio
import httpx
import time
import gzip
import io
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from bs4 import BeautifulSoup

# Optional fallbacks
try:
    import pdfplumber
except:
    pdfplumber = None

try:
    from PIL import Image
    import pytesseract
except:
    Image = None
    pytesseract = None


# =====================================================
# CONFIG
# =====================================================
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
OPENROUTER_URL = "https://aipipe.org/openrouter/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-lite-001"

SECRET_KEY = os.getenv("SECRET_KEY", "tds_24ds")
DEADLINE = 180
MAX_RETRIES = 3

app = FastAPI()


# =====================================================
# HELPERS
# =====================================================

def log(*a):
    print(*a, flush=True)


def get_origin(url: str):
    from urllib.parse import urlparse
    u = urlparse(url)
    return f"{u.scheme}://{u.netloc}"


def abs_url(origin: str, u: str):
    """Ensure u is a full URL."""
    if not u:
        return origin
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        return origin + u
    return origin + "/" + u


def extract_text(html: str):
    """Extract all visible text + atob decoded content."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    for s in soup.find_all("script"):
        code = s.string or ""
        for m in re.finditer(r'atob\(`([^`]+)`\)', code):
            b = m.group(1)
            pad = "=" * ((4 - len(b) % 4) % 4)
            try:
                text += "\n" + base64.b64decode(b + pad).decode()
            except:
                pass

        for m in re.finditer(r'atob\("([^"]+)"\)', code):
            b = m.group(1)
            pad = "=" * ((4 - len(b) % 4) % 4)
            try:
                text += "\n" + base64.b64decode(b + pad).decode()
            except:
                pass

    return text


def safe_json(s: str):
    """Extract first valid JSON object."""
    s = s.strip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i+1])
                except:
                    return None
    return None


# =====================================================
# FALLBACK SOLVERS
# =====================================================

def solve_pdf(pdf_bytes: bytes):
    if not pdfplumber:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) < 2:
                return None
            t = pdf.pages[1].extract_text() or ""
            nums = re.findall(r"-?\d+(\.\d+)?", t)
            if nums:
                return sum([float(x[0] if isinstance(x, tuple) else x) for x in nums])
    except:
        return None
    finally:
        os.remove(path)
    return None


def solve_table(html: str):
    soup = BeautifulSoup(html, "html.parser")
    t = soup.find("table")
    if not t:
        return None
    rows = t.find_all("tr")
    if not rows:
        return None

    headers = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]

    if "value" not in headers:
        return None

    idx = headers.index("value")
    total = 0
    found = False

    for r in rows[1:]:
        cells = r.find_all(["td", "th"])
        if len(cells) <= idx:
            continue
        v = re.sub(r"[^0-9.\-]", "", cells[idx].get_text())
        try:
            total += float(v)
            found = True
        except:
            pass

    return total if found else None


def solve_ocr(img_bytes: bytes):
    if not Image or not pytesseract:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        t = pytesseract.image_to_string(img)
        nums = re.findall(r"-?\d+(\.\d+)?", t)
        if nums:
            return float(nums[0][0] if isinstance(nums[0], tuple) else nums[0])
    except:
        return None
    return None


def solve_puzzle(text: str):
    """Look for base64 → gzip puzzle."""
    m = re.search(r'"payload_gz_b64"\s*:\s*"([^"]+)"', text)
    if not m:
        return None

    b64 = m.group(1)
    try:
        raw = base64.b64decode(b64)
        gun = gzip.decompress(raw)
        js = json.loads(gun)
        return js.get("secret_sum")
    except:
        return None


# =====================================================
# LLM SOLVER (GEMINI via OpenRouter)
# =====================================================

def call_llm(text: str, html: str, url: str, email: str) -> Dict[str, Any]:
    if not AIPIPE_TOKEN:
        return {"error": "NO_API_KEY"}

    origin = get_origin(url)
    default_submit = f"{origin}/submit"

    prompt = f"""
You must return ONLY one JSON object:

{{
  "answer": <value>,
  "submit_url": "<submit>",
  "submit_payload": {{
     "email": "{email}",
     "secret": "{SECRET_KEY}",
     "url": "{url}",
     "answer": <value>
  }}
}}

Rules:
- Read PAGE_TEXT and PAGE_HTML
- If unsure, answer: "anything you want"
- submit_url must be absolute; default = {default_submit}

PAGE_TEXT:
{text}

PAGE_HTML:
{html}
"""

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "Return only JSON. No explanation."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        r = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=40)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        js = safe_json(content)
        if js:
            return {"ok": True, "json": js}
        return {"error": "NO_JSON"}
    except Exception as e:
        return {"error": str(e)}


# =====================================================
# SOLVE PAGE
# =====================================================

async def solve_page(url: str, email: str, client):
    origin = get_origin(url)

    # Fetch page
    try:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text
        text = extract_text(html)
    except:
        return {"answer": "anything you want",
                "submit_url": f"{origin}/submit",
                "submit_payload": {
                    "email": email,
                    "secret": SECRET_KEY,
                    "url": url,
                    "answer": "anything you want"
                }}

    # Fallback: table
    t = solve_table(html)
    if t is not None:
        return {
            "answer": t,
            "submit_url": f"{origin}/submit",
            "submit_payload": {
                "email": email,
                "secret": SECRET_KEY,
                "url": url,
                "answer": t
            }
        }

    # Fallback: puzzle
    p = solve_puzzle(text)
    if p is not None:
        return {
            "answer": p,
            "submit_url": f"{origin}/submit",
            "submit_payload": {
                "email": email,
                "secret": SECRET_KEY,
                "url": url,
                "answer": p
            }
        }

    # LLM solver
    out = call_llm(text, html, url, email)
    if out.get("ok"):
        js = out["json"]
        ans = js.get("answer", "anything you want")
        submit_url = abs_url(origin, js.get("submit_url"))
        payload = js.get("submit_payload", {})

        # *** CRITICAL PATCH — ALWAYS FIX URL ***
        payload["url"] = url          # override wrong /demo from LLM
        payload["secret"] = SECRET_KEY

        return {
            "answer": ans,
            "submit_url": submit_url,
            "submit_payload": payload
        }

    # emergency fallback
    return {
        "answer": "anything you want",
        "submit_url": f"{origin}/submit",
        "submit_payload": {
            "email": email,
            "secret": SECRET_KEY,
            "url": url,
            "answer": "anything you want"
        }
    }


# =====================================================
# ORCHESTRATOR
# =====================================================

async def orchestrator(email: str, url: str):
    client = httpx.AsyncClient()
    deadline = time.time() + DEADLINE
    current = url

    while current and time.time() < deadline:
        log("[TASK]", current)

        solved = await solve_page(current, email, client)
        submit_url = solved["submit_url"]
        payload = solved["submit_payload"]

        # *** CRITICAL PATCH — ensure absolute URL ***
        payload["url"] = current

        log("[SUBMIT]", submit_url)
        log("[PAYLOAD]", payload)

        last = {}

        for _ in range(MAX_RETRIES):
            try:
                r = await client.post(submit_url, json=payload, timeout=30)
                try:
                    jr = r.json()
                except:
                    jr = {}
            except:
                jr = {}

            log("[RESPONSE]", jr)

            last = jr

            if jr.get("correct") and jr.get("url"):
                current = jr["url"]
                break

            if jr.get("correct") and not jr.get("url"):
                log("[END] Finished.")
                return

            if jr.get("url"):
                current = jr["url"]
                break

        else:
            log("[END] No next URL → stopping.")
            return


# =====================================================
# API ENDPOINT
# =====================================================

@app.post("/api/quiz")
async def api_quiz(req: Request, bg: BackgroundTasks):
    data = await req.json()
    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if secret != SECRET_KEY:
        raise HTTPException(403, "Invalid secret")

    if not email or not url:
        raise HTTPException(400, "Missing parameters")

    bg.add_task(orchestrator, email, url)
    return {"status": "accepted"}


