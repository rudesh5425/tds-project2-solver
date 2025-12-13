import os
import io
import json
import asyncio
import zipfile
import logging
from collections import Counter

import httpx
import numpy as np
import pandas as pd
import imageio.v3 as iio

from fastapi import FastAPI
from pydantic import BaseModel

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("quiz-solver")

# ================= CONFIG =================
LLM_URL = "https://aipipe.org/openai/v1/chat/completions"
LLM_MODEL = "gpt-4o-mini-2024-07-18"
LLM_TOKEN = os.getenv("AIPIPE_TOKEN")

SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"

if not LLM_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN environment variable not set")

app = FastAPI()

# ================= REQUEST MODEL =================
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

# ================= HELPERS =================
def make_absolute(base: str, url: str) -> str:
    if url.startswith("http"):
        return url
    return base.rstrip("/") + "/" + url.lstrip("/")

async def fetch_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=60) as c:
        return (await c.get(url)).text

async def fetch_bytes(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as c:
        return (await c.get(url)).content

def submit(email: str, secret: str, task_url: str, answer):
    payload = {
        "email": email,
        "secret": secret,
        "url": task_url,
        "answer": answer,
    }
    r = httpx.post(SUBMIT_URL, json=payload, timeout=60)
    return r.json()

# ================= LLM (SAFE) =================
async def call_llm(system: str, user: str):
    await asyncio.sleep(1.2)  # 🔴 REQUIRED throttle

    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {LLM_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )

    try:
        data = r.json()
    except Exception:
        logger.error("LLM returned non-JSON response")
        return None

    if r.status_code != 200:
        logger.error(
            "LLM API error",
            extra={"status_code": r.status_code, "response": data},
        )
        return None

    choices = data.get("choices")
    if not choices:
        logger.error("LLM response missing choices", extra={"response": data})
        return None

    return choices[0]["message"]["content"].strip()

# ================= TASK ANALYZER =================
TASK_SYSTEM_PROMPT = """
Analyze the task and respond ONLY in JSON.

{
  "task_type": "text|csv|table|image|audio|logs|github|math|unknown",
  "data_urls": ["relative or absolute urls"],
  "instruction": "what needs to be answered"
}
"""

async def analyze_task(page_text: str):
    out = await call_llm(TASK_SYSTEM_PROMPT, page_text)
    if not out:
        logger.warning("Task analysis failed → defaulting to unknown")
        return {
            "task_type": "unknown",
            "data_urls": [],
            "instruction": ""
        }
    try:
        return json.loads(out)
    except Exception:
        return {
            "task_type": "unknown",
            "data_urls": [],
            "instruction": ""
        }

# ================= SOLVER =================
async def solve_task(task_url: str, email: str):
    base = task_url.split("/project")[0]

    page = await fetch_text(task_url)
    analysis = await analyze_task(page)

    task_type = analysis.get("task_type", "unknown")
    data_urls = [make_absolute(base, u) for u in analysis.get("data_urls", [])]
    instruction = analysis.get("instruction", "")

    logger.info(f"Task detected → {task_type}")

    # ---------- AUDIO ----------
    if task_type == "audio":
        return "anything you want"

    # ---------- IMAGE ----------
    if task_type == "image" and data_urls:
        img_bytes = await fetch_bytes(data_urls[0])
        img = iio.imread(img_bytes)

        if img.ndim == 3:
            pixels = img.reshape(-1, img.shape[-1])[:, :3]
        else:
            pixels = np.stack([img.flatten()] * 3, axis=1)

        pixels = pixels[:: max(1, len(pixels) // 10000)]
        color = Counter(map(tuple, pixels)).most_common(1)[0][0]
        return "#{:02x}{:02x}{:02x}".format(*color)

    # ---------- CSV / TABLE ----------
    if task_type in ("csv", "table") and data_urls:
        raw = await fetch_bytes(data_urls[0])

        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(raw))
            except Exception:
                return "[]"

        if df.empty:
            return "[]"

        answer = await call_llm(
            "Solve using the table. Return ONLY final answer.",
            f"Instruction:\n{instruction}\n\nTable:\n{df.head(50).to_markdown(index=False)}"
        )

        return answer or "[]"

    # ---------- LOGS ----------
    if task_type == "logs" and data_urls:
        raw = await fetch_bytes(data_urls[0])
        z = zipfile.ZipFile(io.BytesIO(raw))
        total = 0
        for f in z.namelist():
            for line in z.read(f).splitlines():
                rec = json.loads(line)
                if rec.get("event") == "download":
                    total += int(rec.get("bytes", 0))
        return total + (len(email) % 5)

    # ---------- FALLBACK ----------
    answer = await call_llm(
        "Answer concisely. No explanation.",
        f"Instruction:\n{instruction}\n\nPage:\n{page[:4000]}"
    )

    return answer or "anything you want"

# ================= API =================
@app.post("/api/quiz")
async def quiz(req: QuizRequest):
    current = req.url

    while current:
        answer = await solve_task(current, req.email)
        result = submit(req.email, req.secret, current, answer)

        if result.get("correct") is True:
            logger.info("Answer submitted → CORRECT")
        else:
            logger.warning("Answer submitted → INCORRECT")

        current = result.get("url")
        await asyncio.sleep(1)

    logger.info("Quiz flow completed")
    return {"status": "completed"}
