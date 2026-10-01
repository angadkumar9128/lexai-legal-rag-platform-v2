"""Persistent local Qwen service client using llama.cpp OpenAI-compatible API."""
from __future__ import annotations
import json
import time
from typing import Any
import requests
from config import LLAMA_BASE_URL, LLAMA_MODEL, LLAMA_TIMEOUT, LLAMA_MAX_TOKENS

SESSION = requests.Session()

def health() -> dict:
    try:
        r = SESSION.get(LLAMA_BASE_URL.rsplit("/v1", 1)[0] + "/health", timeout=5)
        return {"ready": r.ok, "status": r.status_code, "error": "" if r.ok else r.text[:300]}
    except Exception as e:
        return {"ready": False, "status": 0, "error": str(e)}

def chat(messages: list[dict[str,str]], temperature: float = 0.15,
         max_tokens: int | None = None, json_schema: dict | None = None,
         timeout: float | None = None) -> tuple[str, dict]:
    payload: dict[str, Any] = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.08,
        "max_tokens": max_tokens or LLAMA_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "schema": json_schema,
        }
    t0 = time.perf_counter()
    try:
        r = SESSION.post(f"{LLAMA_BASE_URL}/chat/completions", json=payload, timeout=timeout or LLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        text = str(data["choices"][0]["message"].get("content", "")).strip()
        return text, {"ok": True, "latency_ms": round((time.perf_counter()-t0)*1000,2)}
    except Exception as e:
        return "", {"ok": False, "error": str(e), "latency_ms": round((time.perf_counter()-t0)*1000,2)}

def json_chat(messages: list[dict[str,str]], schema: dict, max_tokens: int = 500) -> tuple[dict, dict]:
    text, meta = chat(messages, temperature=0.0, max_tokens=max_tokens, json_schema=schema)
    if not meta.get("ok"):
        return {}, meta
    try:
        return json.loads(text), meta
    except Exception:
        return {}, {**meta, "ok": False, "error": "Model returned invalid JSON"}

def answer(messages: list[dict[str,str]], max_tokens: int = 900, timeout: float | None = None) -> tuple[str, dict]:
    return chat(messages, temperature=0.12, max_tokens=max_tokens, timeout=timeout)
