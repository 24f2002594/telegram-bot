import json
import os
import threading

import requests
from flask import Flask, request, jsonify, Response

from agent import run_agent
from logger import append_log, LOG_PATH

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
PUBLIC_BASE_URL = os.environ["PUBLIC_BASE_URL"].rstrip("/")  # e.g. https://your-app.onrender.com
LOG_URL = f"{PUBLIC_BASE_URL}/logs/run.jsonl"
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

# chat_id -> list[{"role": "user"|"assistant", "content": str}]
_conversations: dict[int, list] = {}
_conv_lock = threading.Lock()


def send_message(chat_id: int, text: str) -> None:
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=20,
    )


@app.route("/webhook/<token>", methods=["POST"])
def webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False}), 404

    if WEBHOOK_SECRET:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
            return jsonify({"ok": False}), 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")
    if not message or "text" not in message:
        return jsonify({"ok": True})  # ignore non-text updates

    chat_id = message["chat"]["id"]
    user_text = message["text"]

    if user_text.strip().lower() in ("/start", "/help"):
        send_message(chat_id, "Hi! Send me a data-analysis question and I'll work out the answer.")
        return jsonify({"ok": True})

    with _conv_lock:
        history = _conversations.setdefault(chat_id, [])
        history.append({"role": "user", "content": user_text})
        conversation_snapshot = list(history)

    result = run_agent(conversation_snapshot, log_fn=append_log)
    answer_value = result.get("answer")

    reply_obj = {"answer": answer_value, "log_url": LOG_URL}
    reply_text = json.dumps(reply_obj, ensure_ascii=False)

    with _conv_lock:
        _conversations[chat_id].append({"role": "assistant", "content": reply_text})

    send_message(chat_id, reply_text)
    return jsonify({"ok": True})


@app.route("/logs/run.jsonl", methods=["GET"])
def serve_log():
    if not os.path.exists(LOG_PATH):
        return Response("", mimetype="text/plain")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="application/x-ndjson")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True})


@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    """Visit this once (from a browser or curl) after deploying to register the webhook."""
    url = f"{PUBLIC_BASE_URL}/webhook/{TELEGRAM_TOKEN}"
    payload = {"url": url}
    if WEBHOOK_SECRET:
        payload["secret_token"] = WEBHOOK_SECRET
    resp = requests.post(f"{TELEGRAM_API}/setWebhook", json=payload, timeout=20)
    return jsonify(resp.json())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
