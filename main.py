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
import zipfile
import csv
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from fastapi import FastAPI, Request, HTTPException
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
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://aipipe.org/openrouter/v1/chat/completions")
MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-lite-001")

SECRET_KEY = os.getenv("SECRET_KEY", "tds_24ds")
DEADLINE = int(os.getenv("DEADLINE", "180"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

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


def abs_url(origin: str, u: Optional[str]):
    if not u:
        return origin
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        return origin + u
    return origin + "/" + u


def extract_text(html: str):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")

    # decode atob(...) patterns found in inline scripts
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
    s = (s or "").strip()
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
                    return json.loads(s[start : i + 1])
                except:
                    return None
    return None


# =====================================================
# FALLBACK SOLVERS (PDF, TABLE, OCR, PUZZLE)
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
            nums = re.findall(r"-?\d+(?:\.\d+)?", t)
            if nums:
                return sum(float(n) for n in nums)
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
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
    total = 0.0
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
        txt = pytesseract.image_to_string(img)
        nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
        if nums:
            return float(nums[0])
    except Exception:
        return None
    return None


def solve_puzzle(text: str):
    m = re.search(r'"payload_gz_b64"\s*:\s*"([^"]+)"', text)
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(1))
        gun = gzip.decompress(raw)
        js = json.loads(gun)
        return js.get("secret_sum")
    except Exception:
        return None


# =====================================================
# PROJECT2-SPECIFIC SOLVERS
# =====================================================
async def solve_github_tree(params: Dict[str, Any], email: str, client: httpx.AsyncClient):
    """
    params expected:
    { owner, repo, sha, pathPrefix, extension }
    """
    try:
        owner = params["owner"]
        repo = params["repo"]
        sha = params["sha"]
        prefix = params.get("pathPrefix", "")
        ext = params.get("extension", ".md")
    except KeyError:
        return None

    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}?recursive=1"
    try:
        r = await client.get(tree_url, timeout=20)
        r.raise_for_status()
        j = r.json()
        items = j.get("tree", [])
        count = 0
        for it in items:
            path = it.get("path", "")
            if path.startswith(prefix) and path.endswith(ext):
                count += 1
        offset = len(email) % 2
        return count + offset
    except Exception as e:
        log("[GH TREE ERROR]", e)
        return None


async def solve_logs_zip(params: Dict[str, Any], email: str, client: httpx.AsyncClient):
    """
    params may contain a url or the page may link to logs.zip
    Task: Download logs.zip, sum bytes where event=="download"
    Add offset = len(email) % 5
    """
    # attempt to find zip url
    zip_url = params.get("zip_url") or params.get("url")  # fallback
    if not zip_url:
        return None
    try:
        r = await client.get(zip_url, timeout=30)
        r.raise_for_status()
        data = r.content
        # If response is HTML containing a link to zip, try to extract href
        if r.headers.get("content-type", "").startswith("text/html"):
            html = r.text
            m = re.search(r'href=["\']([^"\']+logs\.zip)["\']', html, re.IGNORECASE)
            if m:
                zip_url = urljoin(r.url, m.group(1))
                r = await client.get(zip_url, timeout=30)
                r.raise_for_status()
                data = r.content
        # open zip from bytes
        z = zipfile.ZipFile(io.BytesIO(data))
        base_sum = 0
        for name in z.namelist():
            try:
                with z.open(name) as fh:
                    content = fh.read()
                    # try parse as JSON lines
                    try:
                        txt = content.decode("utf-8", errors="replace")
                        # find all JSON objects / lines
                        for line in txt.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except:
                                # try to extract bytes via regex
                                obj = None
                            if isinstance(obj, dict) and obj.get("event") == "download":
                                b = obj.get("bytes") or obj.get("size") or 0
                                try:
                                    base_sum += int(b)
                                except:
                                    pass
                        # fallback: search for pattern "event":"download" then "bytes":N
                        for m in re.finditer(r'"event"\s*:\s*"?download"?.*?"bytes"\s*:\s*(\d+)', txt, re.DOTALL):
                            base_sum += int(m.group(1))
                    except Exception:
                        continue
            except Exception:
                continue
        offset = len(email) % 5
        return base_sum + offset
    except Exception as e:
        log("[LOGS ZIP ERROR]", e)
        return None


async def solve_csv_task(params: Dict[str, Any], email: str, client: httpx.AsyncClient):
    """
    Download messy.csv (link or data uri) and normalize:
    - snake_case keys (id, name, joined, value)
    - joined → ISO-8601 (try common formats)
    - value → integer
    - sort by id ascending
    Return JSON array as string
    """
    csv_url = params.get("csv_url") or params.get("url")
    if not csv_url:
        return None

    try:
        if csv_url.startswith("data:") and "base64," in csv_url:
            b64 = csv_url.split("base64,")[-1]
            raw = base64.b64decode(b64)
            text = raw.decode("utf-8", errors="replace")
        else:
            r = await client.get(csv_url, timeout=20)
            r.raise_for_status()
            text = r.text
    except Exception as e:
        log("[CSV TASK ERROR]", e)
        return None

    # parse CSV
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            # normalize keys to snake_case expected: id, name, joined, value
            norm = {}
            for k, v in row.items():
                key = re.sub(r"[^0-9a-zA-Z]", "_", k).strip().lower()
                key = re.sub(r"_+", "_", key)
                if key in ("id", "name", "joined", "value"):
                    norm[key] = v.strip()
            # attempt to coerce types
            # id
            try:
                norm["id"] = int(norm.get("id", 0))
            except:
                # try to extract digits
                m = re.search(r"\d+", str(norm.get("id", "")))
                norm["id"] = int(m.group(0)) if m else 0
            # value
            try:
                norm["value"] = int(float(norm.get("value", 0)))
            except:
                m = re.search(r"-?\d+", str(norm.get("value", "")))
                norm["value"] = int(m.group(0)) if m else 0
            # joined -> try multiple date formats then isoformat date-only
            joined_raw = norm.get("joined", "")
            parsed = None
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(joined_raw, fmt)
                    break
                except:
                    continue
            if parsed:
                norm["joined"] = parsed.date().isoformat()
            else:
                # if looks like epoch
                try:
                    v = int(joined_raw)
                    # assume seconds
                    dt = datetime.utcfromtimestamp(v)
                    norm["joined"] = dt.date().isoformat()
                except:
                    # leave as-is (best-effort)
                    norm["joined"] = joined_raw
            # ensure name exists
            norm["name"] = norm.get("name", "")
            rows.append(norm)
        # sort by id ascending
        rows = sorted(rows, key=lambda x: x.get("id", 0))
        # keep only desired keys in order: id, name, joined, value
        out = []
        for r in rows:
            out.append({"id": r.get("id", 0), "name": r.get("name", ""), "joined": r.get("joined", ""), "value": r.get("value", 0)})
        return json.dumps(out, separators=(",", ":"), ensure_ascii=False)
    except Exception as e:
        log("[CSV PARSE ERROR]", e)
        return None


async def solve_heatmap(params: Dict[str, Any], email: str, client: httpx.AsyncClient):
    """
    Find image (image_url or image_path) → download → compute dominant color → hex (e.g., #a1b2c3)
    Returns hex string or None
    """
    image_url = params.get("image_url") or params.get("image_path") or params.get("url")
    if not image_url:
        return None
    try:
        # if data uri
        if image_url.startswith("data:") and "base64," in image_url:
            b64 = image_url.split("base64,")[-1]
            img_bytes = base64.b64decode(b64)
        else:
            r = await client.get(image_url, timeout=30)
            r.raise_for_status()
            img_bytes = r.content
        if not Image:
            return None
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # shrink for speed
        img = img.resize((100, 100))
        pixels = list(img.getdata())
        # count frequency
        freq = {}
        for p in pixels:
            freq[p] = freq.get(p, 0) + 1
        # pick most common
        best = max(freq.items(), key=lambda x: x[1])[0]
        hexcol = "#{:02x}{:02x}{:02x}".format(best[0], best[1], best[2])
        return hexcol
    except Exception as e:
        log("[HEATMAP ERROR]", e)
        return None


async def solve_audio_passphrase(params: Dict[str, Any], client: httpx.AsyncClient):
    """
    If page provides audio_data_uri that actually encodes ASCII text (our fake server uses this),
    decode base64 and extract words+digits (lowercase).
    Otherwise return None to let LLM or other fallbacks handle.
    """
    audio_uri = params.get("audio_data_uri") or params.get("url")
    if not audio_uri:
        return None
    try:
        if audio_uri.startswith("data:") and "base64," in audio_uri:
            b64 = audio_uri.split("base64,")[-1]
            raw = base64.b64decode(b64)
            # if raw looks like ASCII text, return it
            try:
                txt = raw.decode("utf-8").strip()
                # only accept if has letters and digits (fake server uses "numbers 3 4 5")
                if re.search(r"[a-zA-Z]", txt):
                    # normalize to lowercase and compress whitespace
                    txt = re.sub(r"\s+", " ", txt).strip().lower()
                    return txt
            except:
                return None
        else:
            # download and check if bytes are ascii-like
            r = await client.get(audio_uri, timeout=30)
            r.raise_for_status()
            raw = r.content
            try:
                txt = raw.decode("utf-8").strip()
                if re.search(r"[a-zA-Z]", txt):
                    txt = re.sub(r"\s+", " ", txt).strip().lower()
                    return txt
            except:
                return None
    except Exception as e:
        log("[AUDIO TASK ERROR]", e)
        return None
    return None


# =====================================================
# LLM SOLVER (OpenRouter / AIPipe)
# =====================================================
def call_llm(text: str, html: str, url: str, email: str):
    if not AIPIPE_TOKEN:
        return {"error": "NO_KEY"}
    origin = get_origin(url)
    default_submit = f"{origin}/submit"
    prompt = f"""
Return ONLY this JSON format:

{{
  "answer": <value>,
  "submit_url": "<url>",
  "submit_payload": {{
    "email": "{email}",
    "secret": "{SECRET_KEY}",
    "url": "{url}",
    "answer": <value>
  }}
}}

If unsure: answer = "anything you want".
submit_url must be absolute (default: {default_submit}).

PAGE_TEXT:
{text}

PAGE_HTML:
{html}
"""
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": 0.2, "messages": [{"role": "system", "content": "Return only JSON."}, {"role": "user", "content": prompt}]}
    try:
        r = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=40)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        js = safe_json(content)
        return {"ok": True, "json": js} if js else {"error": "NO_JSON", "raw": content}
    except Exception as e:
        return {"error": str(e)}


# =====================================================
# SOLVE PAGE (main orchestrator per page)
# =====================================================
async def solve_page(url: str, email: str, client: httpx.AsyncClient):
    origin = get_origin(url)
    try:
        r = await client.get(url, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        log("[FETCH ERROR]", e)
        return {"answer": "anything you want", "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": "anything you want"}}

    text = extract_text(html)

    # Try to extract JSON params from page text (many project2 pages embed a JSON)
    params = safe_json(text) or {}

    # Project2 specialized solvers (use params when available)
    # 1) GitHub tree
    if "owner" in params and "repo" in params and "sha" in params:
        res = await solve_github_tree(params, email, client)
        if res is not None:
            return {"answer": res, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": res}}

    # 2) logs.zip task: page may include json keys or link to logs.zip
    if "zip" in text.lower() or "logs.zip" in text.lower() or params.get("zip_url"):
        # try params first
        zip_params = {"zip_url": params.get("zip_url") or params.get("url")}
        res = await solve_logs_zip(zip_params, email, client)
        if res is not None:
            return {"answer": res, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": res}}

    # 3) csv task
    if "csv" in text.lower() or re.search(r"\.csv", html, re.IGNORECASE):
        csv_url = None
        # try to find csv link
        m = re.search(r'href=["\']([^"\']+\.csv)["\']', html, re.IGNORECASE)
        if m:
            csv_url = urljoin(origin, m.group(1))
        elif params.get("file_data_uri") or params.get("audio_data_uri") or params.get("payload_gz_b64"):
            csv_url = params.get("file_data_uri") or params.get("audio_data_uri")
        if csv_url:
            res = await solve_csv_task({"csv_url": csv_url}, email, client)
            if res is not None:
                return {"answer": res, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": res}}

    # 4) heatmap / image dominant color
    if "color" in text.lower() or "dominant" in text.lower() or re.search(r'\.(png|jpg|jpeg)', html, re.IGNORECASE):
        # attempt to find image url or data uri
        m = re.search(r'src=["\']([^"\']+\.(png|jpg|jpeg))["\']', html, re.IGNORECASE)
        image_url = None
        if m:
            image_url = urljoin(origin, m.group(1))
        elif params.get("image_path") or params.get("image_url"):
            image_url = params.get("image_path") or params.get("image_url")
        if image_url:
            res = await solve_heatmap({"image_url": image_url}, email, client)
            if res is not None:
                return {"answer": res, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": res}}

    # 5) audio passphrase (check for embedded data uri or link)
    if "audio" in text.lower() or re.search(r'\.opus|\.wav|audio', html, re.IGNORECASE):
        audio_uri = params.get("audio_data_uri")
        # if not present try to find .opus link
        if not audio_uri:
            m = re.search(r'href=["\']([^"\']+\.opus)["\']', html, re.IGNORECASE)
            if m:
                audio_uri = urljoin(origin, m.group(1))
        if audio_uri:
            res = await solve_audio_passphrase({"audio_data_uri": audio_uri}, client)
            if res is not None:
                # transcription expected as string
                return {"answer": res, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": res}}

    # Fallbacks: table (HTML) first
    t = solve_table(html)
    if t is not None:
        return {"answer": t, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": t}}

    # puzzle fallback
    p = solve_puzzle(text)
    if p is not None:
        return {"answer": p, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": p}}

    # pdf fallback: try to detect pdf link and fetch
    m = re.search(r'href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
    if m:
        pdf_url = urljoin(origin, m.group(1))
        try:
            rr = await client.get(pdf_url, timeout=30)
            rr.raise_for_status()
            s = solve_pdf(rr.content)
            if s is not None:
                return {"answer": s, "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": s}}
        except Exception:
            pass

    # LLM fallback
    out = call_llm(text, html, url, email)
    if out.get("ok"):
        js = out["json"]
        ans = js.get("answer", "anything you want")
        submit_url = abs_url(origin, js.get("submit_url"))
        payload = js.get("submit_payload", {}) or {}
        # critical patches: ensure correct URL & secret
        payload["url"] = url
        payload["secret"] = SECRET_KEY
        return {"answer": ans, "submit_url": submit_url, "submit_payload": payload}

    # emergency fallback
    return {"answer": "anything you want", "submit_url": f"{origin}/submit", "submit_payload": {"email": email, "secret": SECRET_KEY, "url": url, "answer": "anything you want"}}


# =====================================================
# ORCHESTRATOR
# =====================================================
async def orchestrator(email: str, url: str):
    client = httpx.AsyncClient()
    deadline = time.time() + DEADLINE
    current = url
    try:
        while current and time.time() < deadline:
            log("[TASK]", current)
            solved = await solve_page(current, email, client)
            submit_url = solved.get("submit_url") or f"{get_origin(current)}/submit"
            payload = solved.get("submit_payload") or {"email": email, "secret": SECRET_KEY, "url": current, "answer": solved.get("answer")}
            payload["url"] = current
            payload["secret"] = SECRET_KEY

            log("[SUBMIT]", submit_url)
            log("[PAYLOAD]", payload)

            last = {}
            succeeded = False
            for _ in range(MAX_RETRIES):
                try:
                    r = await client.post(submit_url, json=payload, timeout=30)
                    try:
                        jr = r.json()
                    except:
                        jr = {}
                except Exception as e:
                    jr = {}
                    log("[SUBMIT ERROR]", e)

                log("[RESPONSE]", jr)
                last = jr

                if jr.get("correct") and jr.get("url"):
                    current = jr["url"]
                    succeeded = True
                    break
                if jr.get("correct") and not jr.get("url"):
                    log("[END] Finished.")
                    return
                if jr.get("url"):
                    current = jr["url"]
                    succeeded = True
                    break
            if not succeeded:
                log("[END] No next URL → stopping.")
                return
    finally:
        await client.aclose()


# =====================================================
# API ENDPOINT
# =====================================================
@app.post("/api/quiz")
async def api_quiz(req: Request):
    data = await req.json()
    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    if secret != SECRET_KEY:
        raise HTTPException(403, "Invalid secret")
    if not email or not url:
        raise HTTPException(400, "Missing parameters")

    # schedule the orchestrator without blocking the request
    asyncio.create_task(orchestrator(email, url))
    return {"status": "accepted"}


@app.get("/")
def root():
    return {"status": "running", "version": "final-prod-A"}
