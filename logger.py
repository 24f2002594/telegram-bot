import json
import os
import threading

_LOCK = threading.Lock()
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(os.path.dirname(__file__), "logs", "run.jsonl"))


def append_log(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
