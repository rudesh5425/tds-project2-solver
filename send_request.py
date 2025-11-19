# send_request.py - test your endpoint locally
import httpx

ENDPOINT = "http://localhost:8000/api/quiz"
payload = {
    "email": "24ds2000104@ds.study.iitm.ac.in",
    "secret": "tds_24ds",
    "url": "https://tds-llm-analysis.s-anand.net/demo"  # demo test URL
}

r = httpx.post(ENDPOINT, json=payload, timeout=30)
print("status:", r.status_code)
print("response:", r.text)
