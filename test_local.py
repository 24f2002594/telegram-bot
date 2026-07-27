"""
Quick local sanity check, no Telegram/Flask needed:

    python test_local.py "Which state has the highest maternal mortality rate \
based on MOSPI data? Reply with ONLY this JSON object and nothing else: \
{\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"...\"}"

Prints the raw answer value the agent would submit.
"""
import sys

from agent import run_agent
from logger import append_log

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2? Reply with {\"answer\": <number>}"
    result = run_agent([{"role": "user", "content": question}], log_fn=append_log)
    print(result["answer"])
