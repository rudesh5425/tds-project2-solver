# send_request.py - test deployed solver
import httpx

ENDPOINT = "http://127.0.0.1:8000/api/quiz"

payload = {
    "email": "24ds2000104@ds.study.iitm.ac.in",
    "secret": "tds_24ds",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

r = httpx.post(ENDPOINT, json=payload, timeout=50)
print("status:", r.status_code)
print("response:", r.text)
