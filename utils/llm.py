from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import Any

import anthropic
import streamlit as st


ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


def _config(name: str, default: str = "") -> str:
    session_value = st.session_state.get(name, "")
    if session_value:
        value = str(session_value).strip()
        if not _is_placeholder(value):
            return value
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    value = str(secret_value or os.getenv(name, default)).strip()
    return "" if _is_placeholder(value) else value


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().strip('"').strip("'").lower()
    return lowered.startswith("your_") and lowered.endswith("_api_key_here")


def active_provider() -> str:
    openai_base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    local_openai = openai_base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if _config("OPENAI_API_KEY") or local_openai:
        return "chatgpt"
    if _config("GEMINI_API_KEY"):
        return "gemini"
    if _config("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def get_client() -> anthropic.Anthropic:
    api_key = _config("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("No Anthropic key found.")
    return anthropic.Anthropic(api_key=api_key)


def _openai_payload(messages: list[dict[str, str]], system: str, stream: bool) -> bytes:
    return json.dumps(
        {
            "model": _config("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.7,
            "max_tokens": 1200,
            "stream": stream,
        }
    ).encode("utf-8")


def _openai_request(messages: list[dict[str, str]], system: str, stream: bool = False) -> urllib.request.Request:
    base_url = _config("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = _config("OPENAI_API_KEY") or "local"
    return urllib.request.Request(
        f"{base_url}/chat/completions",
        data=_openai_payload(messages, system, stream),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )


def _chatgpt_complete(messages: list[dict[str, str]], system: str) -> str:
    try:
        with urllib.request.urlopen(_openai_request(messages, system), timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("ChatGPT/OpenAI", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach ChatGPT/OpenAI endpoint: {exc.reason}") from exc


def _chatgpt_stream(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    try:
        with urllib.request.urlopen(_openai_request(messages, system, stream=True), timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                delta = data["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("ChatGPT/OpenAI", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach ChatGPT/OpenAI endpoint: {exc.reason}") from exc


def _gemini_prompt(messages: list[dict[str, str]], system: str) -> str:
    turns = [f"System: {system}"]
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        turns.append(f"{role}: {message['content']}")
    return "\n\n".join(turns)


def _gemini_complete(messages: list[dict[str, str]], system: str) -> str:
    api_key = _config("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No Gemini key found.")
    model = _config("GEMINI_MODEL", "gemini-2.5-flash")
    payload = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": _gemini_prompt(messages, system)}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(_friendly_api_error("Gemini", exc.code, details or exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc


def _anthropic_complete(messages: list[dict[str, str]], system: str) -> str:
    response = get_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def _anthropic_stream(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    with get_client().messages.stream(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def chat_complete(messages: list[dict[str, str]], system: str, stream: bool = False) -> str | Any:
    provider = active_provider()
    if stream:
        return stream_complete(messages, system)
    if provider == "chatgpt":
        return _chatgpt_complete(messages, system)
    if provider == "gemini":
        return _gemini_complete(messages, system)
    if provider == "anthropic":
        return _anthropic_complete(messages, system)
    raise RuntimeError("Add OPENAI_API_KEY or GEMINI_API_KEY in Streamlit secrets or environment.")


def stream_complete(messages: list[dict[str, str]], system: str) -> Generator[str, None, None]:
    provider = active_provider()
    if provider == "chatgpt":
        yield from _chatgpt_stream(messages, system)
    elif provider == "gemini":
        yield _gemini_complete(messages, system)
    elif provider == "anthropic":
        yield from _anthropic_stream(messages, system)
    else:
        raise RuntimeError("Add OPENAI_API_KEY or GEMINI_API_KEY in Streamlit secrets or environment.")


def _friendly_api_error(provider: str, code: int, details: str) -> str:
    lowered = str(details).lower()
    if "reported as leaked" in lowered or "api key was reported as leaked" in lowered:
        return (
            f"{provider} rejected this API key because it has been reported as leaked. "
            "Revoke it, create a new key, put the new key in .streamlit/secrets.toml, then restart Streamlit."
        )
    if code in {401, 403}:
        return (
            f"{provider} rejected the configured key with HTTP {code}. "
            "Check that the key is active, unrestricted for this API, and saved in .streamlit/secrets.toml."
        )
    return f"{provider} API error {code}: {details}"
