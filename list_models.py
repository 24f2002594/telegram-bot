"""
Run this once to see which models your GEMINI_API_KEY can actually call:

    python list_models.py

Look for a "flash" model (cheapest/free-tier-friendly) that supports
generateContent, and put its name in .env as GEMINI_MODEL.
"""
import os

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(m.name)