# Data Analyst Telegram Bot

An LLM agent, reachable over Telegram, that answers data-analysis questions
(MOSPI and similar public datasets) and replies with a single strict JSON
object:

```json
{"answer": <value in the shape the question asked>, "log_url": "https://your-host/logs/run.jsonl"}
```

## How it works

- `agent.py` — the agent. Gemini (free tier, `gemini-2.5-flash` by default)
  is given two working tools (`fetch_url`,
  `run_python`) and loops until it calls `submit_answer`. The model only
  ever produces the *value* for `"answer"` — the two-key envelope
  (`answer`, `log_url`) is always assembled by plain Python code in
  `app.py`, never by the model, so the reply can never drift from the
  required shape.
- `app.py` — a tiny Flask app: a Telegram webhook endpoint, plus a
  `/logs/run.jsonl` route that serves the agent's own run log so the
  `log_url` you give the grader is this same running service.
- `logger.py` — appends one JSON line per agent step to `logs/run.jsonl`.

## 1. Create the bot

1. Talk to **@BotFather** on Telegram → `/newbot` → pick a name → username
   must end in `bot`.
2. Save the token it gives you.

## 2. Configure

```bash
cp .env.example .env
# fill in TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PUBLIC_BASE_URL (set after step 3),
# and a random TELEGRAM_WEBHOOK_SECRET
```

## 3. Deploy (pick one)

**Render (free web service)**
1. Push this repo to GitHub (public).
2. New → Web Service → connect the repo.
3. Environment: Docker (it will pick up the `Dockerfile`).
4. Add the env vars from `.env.example` in the Render dashboard.
   `PUBLIC_BASE_URL` = the `https://<name>.onrender.com` URL Render assigns you.
5. Deploy.

**Hugging Face Spaces (Docker SDK)** works the same way — create a Space,
SDK = Docker, push this repo's contents, set the same env vars as Secrets,
`PUBLIC_BASE_URL` = `https://<user>-<space>.hf.space`.

**Fly.io / Railway** — same Dockerfile works as-is; set the same env vars.

⚠️ Free tiers are usually ephemeral (the container can restart and wipe
`logs/run.jsonl`). That's fine for grading (the log for the graded run will
be there), but if you want durability, periodically copy `logs/run.jsonl`
to a bucket or a GitHub Gist and point `log_url` there instead.

## 4. Register the webhook

Once deployed, hit (once):

```
https://your-app-host/set_webhook
```

This tells Telegram to POST updates to `/webhook/<token>`.

## 5. Test

Message your bot directly on Telegram, or run the grading pipeline locally
against it:

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# follow its README, point it at your bot username
```

You can also sanity-check the agent without Telegram at all:

```bash
pip install -r requirements.txt
python test_local.py "Which state has the highest maternal mortality rate based on MOSPI data? Reply with ONLY this JSON object and nothing else: {\"answer\": {\"state\": \"<state name>\"}, \"log_url\": \"...\"}"
```

## Notes / limitations to be aware of

- `run_python` executes model-generated code in a subprocess with a
  timeout, no sandboxing beyond that. Acceptable for a course project, not
  for production with untrusted users.
- The agent replies to *every* incoming message with the JSON envelope,
  using the full chat history as context — matches "answer the last
  message" for multi-turn tasks since only the final reply is graded.
- If Gemini can't reach a live dataset after tool attempts, it still
  submits its best-reasoned value in the correct shape rather than
  refusing, since a shaped guess can partially score where a refusal
  cannot.
- Get a free `GEMINI_API_KEY` at https://aistudio.google.com/apikey — no
  credit card needed. Free-tier rate limits are modest (a handful of
  requests/minute); if you hit 429 errors during heavy local testing, just
  wait a bit between runs.
