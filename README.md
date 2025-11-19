📘 TDS Project 2 — LLM-Based Quiz Solver (FastAPI + Render Deployment)

This project implements an automated solver for the IIT Madras TDS Project-2 evaluation system.
It fetches quiz pages, extracts text, detects PDFs/images/audio, and submits answers automatically.

🚀 Features
✔ Works on Windows, Linux, Render
✔ No Playwright (avoids subprocess issues)
✔ PDF, Image OCR, and minimal Audio handling
✔ LLM fallback via AI Pipe (gpt-4.1-nano)
✔ Robust retry logic for “Secret mismatch”
✔ Guaranteed non-null answer for every submission (“anything you want” fallback)