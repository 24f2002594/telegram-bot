"""
Data-analyst agent: given a conversation (list of user/assistant text turns)
ending in a data-analysis question, uses Gemini (google-genai SDK) with
tools (fetch a URL, run a python snippet) to work out the answer, then
returns the answer value in whatever shape the question asked for.

Design choice: we NEVER let the model free-form the final JSON envelope.
The model only ever produces the *value* that goes into the "answer" key
(via the submit_answer tool, as a JSON-encoded string). The calling code
(app.py) wraps that value together with the log_url into the exact two-key
envelope the grader wants. This avoids the single most common failure mode:
the model almost-but-not-quite reproducing the requested JSON shape.
"""

import json
import os
import subprocess
import tempfile
import textwrap
import time
import uuid

import requests
from google import genai
from google.genai import types

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
MAX_TOOL_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "10"))
PY_TIMEOUT_SECONDS = int(os.environ.get("AGENT_PY_TIMEOUT", "30"))
FETCH_MAX_CHARS = 20000

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a meticulous data analyst agent operating headless (no human in
    the loop). You will be shown a short conversation. The final user
    message contains a data-analysis question. It usually spells out the
    exact JSON shape the answer should take (for example it might ask you
    to eventually reply with {"answer": {"state": "<state name>"}, ...}).
    You do NOT need to produce that whole envelope. You only need to work
    out the correct VALUE for the "answer" key, in exactly the shape asked
    for (same keys, same nesting, same types: string/number/array/object as
    requested), and hand it to the submit_answer tool as a JSON-encoded
    string (e.g. "42", "\\"Assam\\"", or "{\\"state\\": \\"Assam\\"}").

    You have two working tools:
    - fetch_url: fetch the raw contents of a public URL (web page, CSV,
      JSON, etc). Use this to locate and download real public datasets
      (MOSPI, data.gov.in, RBI, census, etc.) rather than guessing from
      memory.
    - run_python: execute a short Python snippet (pandas/numpy/requests
      available) in a fresh subprocess and see its stdout. Use this to
      parse/clean/aggregate data you fetched. print() whatever you need to
      see. The snippet has its own network access, so it can also do
      requests.get(...) / pd.read_csv(url) directly if that's easier than
      fetch_url.

    Work step by step:
    1. Understand exactly what's being asked and the exact answer shape.
    2. Locate the real data (recall the right MOSPI/public-dataset URL if
       the question doesn't give one directly; try fetch_url on candidate
       URLs).
    3. Compute the answer with run_python, don't eyeball numbers.
    4. Call submit_answer with the final value as a JSON-encoded string,
       matching the requested shape exactly (correct key names, correct
       types).

    If, after reasonable effort, the exact live dataset cannot be reached,
    give your best-reasoned answer based on the most reliable numbers you
    could gather or recall, still in the exact requested shape - never
    refuse and never submit prose instead of the structured value.
    """
).strip()

FETCH_URL_DECL = types.FunctionDeclaration(
    name="fetch_url",
    description="Fetch the raw text contents of a public URL (HTML, CSV, JSON, etc). Returns truncated text.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"url": types.Schema(type=types.Type.STRING, description="The URL to fetch.")},
        required=["url"],
    ),
)

RUN_PYTHON_DECL = types.FunctionDeclaration(
    name="run_python",
    description=(
        "Execute a standalone Python 3 snippet in a fresh subprocess "
        "(pandas, numpy, requests, json available) and return its stdout/stderr. "
        "print() anything you want to inspect. No state is kept between calls, "
        "so re-include any setup code each time."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"code": types.Schema(type=types.Type.STRING, description="Python source to execute.")},
        required=["code"],
    ),
)

SUBMIT_ANSWER_DECL = types.FunctionDeclaration(
    name="submit_answer",
    description="Submit the final answer, as a JSON-encoded string, in exactly the shape the question requested. Ends the run.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "answer_json": types.Schema(
                type=types.Type.STRING,
                description='The final answer value, JSON-encoded as a string (e.g. "42", "\\"Assam\\"", "{\\"state\\": \\"Assam\\"}").',
            )
        },
        required=["answer_json"],
    ),
)

TOOLS = [types.Tool(function_declarations=[FETCH_URL_DECL, RUN_PYTHON_DECL, SUBMIT_ANSWER_DECL])]

GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=TOOLS,
)


def _fetch_url(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (data-analyst-agent)"},
        )
        text = resp.text
        if len(text) > FETCH_MAX_CHARS:
            text = text[:FETCH_MAX_CHARS] + f"\n...[truncated, {len(resp.text)} chars total]"
        return f"status={resp.status_code}\n{text}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR fetching {url}: {e}"


def _run_python(code: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "snippet.py")
        with open(script_path, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                ["python3", script_path],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=PY_TIMEOUT_SECONDS,
            )
            out = proc.stdout
            if proc.stderr:
                out += f"\n[stderr]\n{proc.stderr}"
            if len(out) > FETCH_MAX_CHARS:
                out = out[:FETCH_MAX_CHARS] + "\n...[truncated]"
            return out or "(no output — did you print()?)"
        except subprocess.TimeoutExpired:
            return f"ERROR: python snippet timed out after {PY_TIMEOUT_SECONDS}s"
        except Exception as e:  # noqa: BLE001
            return f"ERROR running snippet: {e}"


def _genai_role(role: str) -> str:
    return "model" if role == "assistant" else "user"


def _send_with_retry(chat, message, log, max_retries: int = 4):
    """Send a chat message, retrying with backoff if we hit a rate limit (429)."""
    delay = 20
    for attempt in range(max_retries + 1):
        try:
            return chat.send_message(message)
        except Exception as e:  # noqa: BLE001
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries:
                log({"event": "rate_limited", "attempt": attempt, "sleeping_seconds": delay})
                time.sleep(delay)
                delay = min(delay * 2, 90)
                continue
            raise


def run_agent(conversation: list[dict], log_fn=None) -> dict:
    """
    conversation: list of {"role": "user"|"assistant", "content": str}
    log_fn: optional callable(record: dict) called for every step, for JSONL logging.

    Returns: {"answer": <value>, "trace": [...]} on success, or
             {"answer": None, "error": "..."} on hard failure.
    """
    run_id = str(uuid.uuid4())
    trace = []

    def log(record: dict):
        record = {"run_id": run_id, "ts": time.time(), **record}
        trace.append(record)
        if log_fn:
            try:
                log_fn(record)
            except Exception:  # noqa: BLE001
                pass

    log({"event": "start", "conversation": conversation})

    history = [
        types.Content(role=_genai_role(m["role"]), parts=[types.Part(text=m["content"])])
        for m in conversation[:-1]
    ]
    last_message = conversation[-1]["content"]

    try:
        chat = client.chats.create(model=MODEL_NAME, config=GEN_CONFIG, history=history)
        response = _send_with_retry(chat, last_message, log)
    except Exception as e:  # noqa: BLE001
        log({"event": "llm_error", "error": str(e)})
        return {"answer": None, "error": f"llm_error: {e}", "trace": trace}

    for round_num in range(MAX_TOOL_ROUNDS):
        function_calls = response.function_calls or []
        text_parts = []
        for part in (response.candidates[0].content.parts or []):
            if getattr(part, "text", None):
                text_parts.append(part.text)

        log(
            {
                "event": "assistant_turn",
                "round": round_num,
                "text": " ".join(text_parts),
                "tool_uses": [{"name": fc.name, "input": dict(fc.args or {})} for fc in function_calls],
            }
        )

        if not function_calls:
            try:
                response = _send_with_retry(
                    chat, "Please call the submit_answer tool with your final answer_json now.", log
                )
            except Exception as e:  # noqa: BLE001
                log({"event": "llm_error", "error": str(e)})
                return {"answer": None, "error": f"llm_error: {e}", "trace": trace}
            continue

        function_response_parts = []
        submitted = None
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args or {})
            if name == "fetch_url":
                result = _fetch_url(args.get("url", ""))
            elif name == "run_python":
                result = _run_python(args.get("code", ""))
            elif name == "submit_answer":
                raw = args.get("answer_json", "")
                try:
                    submitted = json.loads(raw)
                except Exception:  # noqa: BLE001
                    submitted = raw  # fall back to raw string if it wasn't valid JSON
                result = "recorded"
            else:
                result = f"unknown tool {name}"

            log(
                {
                    "event": "tool_result",
                    "round": round_num,
                    "tool": name,
                    "input": args,
                    "output_preview": str(result)[:2000],
                }
            )
            function_response_parts.append(
                types.Part.from_function_response(name=name, response={"result": str(result)})
            )

        if submitted is not None:
            log({"event": "final_answer", "answer": submitted})
            return {"answer": submitted, "trace": trace}

        try:
            response = _send_with_retry(chat, function_response_parts, log)
        except Exception as e:  # noqa: BLE001
            log({"event": "llm_error", "error": str(e)})
            return {"answer": None, "error": f"llm_error: {e}", "trace": trace}

    log({"event": "max_rounds_exhausted"})
    return {"answer": None, "error": "max_tool_rounds_exhausted", "trace": trace}