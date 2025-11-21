# send_request.py - test deployed solver
import httpx

ENDPOINT = "https://tds-project2-solver.onrender.com/api/quiz"

payload = {
    "email": "24ds2000104@ds.study.iitm.ac.in",
    "secret": "tds_24ds",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

r = httpx.post(ENDPOINT, json=payload, timeout=50)
print("status:", r.status_code)
print("response:", r.text)
