"""Thin, dependency-free adapters for tool-calling LLM APIs.

The agent speaks one neutral message format; each adapter converts it to the
provider's wire format and back. This keeps the agent loop provider-agnostic.

Neutral messages:
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "results": [{"id", "name", "content", "is_error"}]}
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    extra: dict = field(default_factory=dict)  # provider fields to echo back (e.g. Gemini thought signatures)


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class QuotaError(LLMError):
    """Rate limit that will not clear soon (e.g. a daily free-tier cap). Retrying is pointless."""


def _retry_after_seconds(text: str) -> float | None:
    """Read the wait time a provider asks for: 'retry in 1h42m20.7s', 'retryDelay': '36s', 'try again in 20s'."""
    m = re.search(r"(?:retry|try again) in\s+((?:\d+(?:\.\d+)?[hms]\s*)+)", text, re.I) or \
        re.search(r'"retryDelay"\s*:\s*"((?:\d+(?:\.\d+)?[hms])+)"', text)
    if not m:
        return None
    units = {"h": 3600, "m": 60, "s": 1}
    return sum(float(n) * units[u.lower()] for n, u in re.findall(r"(\d+(?:\.\d+)?)([hms])", m.group(1), re.I))


def _post_with_retry(url: str, headers: dict, payload: dict, retries: int = 5, timeout: int = 180) -> dict:
    delay = 2.0
    last = None
    for attempt in range(retries):
        wait = delay
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:  # network blip
            last = str(e)
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise LLMError(last)
            if r.status_code == 429:
                asked = _retry_after_seconds(r.text) or float(r.headers.get("retry-after", 0) or 0)
                if asked > 90 or re.search(r"per ?day|daily|PerDay", r.text):
                    mins = f" (resets in about {asked / 60:.0f} min)" if asked else ""
                    raise QuotaError(f"Rate limit / quota exhausted for this model{mins}. Switch model or provider, "
                                     f"or enable billing. Provider said: {r.text[:300]}")
                if asked:
                    wait = asked + 1
        if attempt < retries - 1:
            time.sleep(wait)
            delay = min(delay * 2, 30)
    raise LLMError(f"LLM request failed after {retries} attempts: {last}")


class AnthropicLLM:
    """Claude via the Messages API."""

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 4096):
        self.model = model or os.getenv("REPRO_MODEL") or "claude-sonnet-5-5"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.max_tokens = max_tokens
        self.url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages"

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        out: list[dict] = []

        def push(role: str, blocks: list[dict]):
            if out and out[-1]["role"] == role:  # API needs alternating roles
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})

        for m in messages:
            if m["role"] == "user":
                push("user", [{"type": "text", "text": m["text"]}])
            elif m["role"] == "assistant":
                blocks = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                if not blocks:
                    blocks = [{"type": "text", "text": "(no output)"}]
                push("assistant", blocks)
            elif m["role"] == "tool":
                push("user", [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"],
                     "is_error": bool(r.get("is_error"))}
                    for r in m["results"]
                ])
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self.to_wire(messages),
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_with_retry(self.url, headers, payload)
        turn = AssistantTurn(usage=data.get("usage", {}))
        texts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block["text"])
            elif block.get("type") == "tool_use":
                turn.tool_calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        turn.text = "\n".join(texts).strip()
        return turn


# Presets for OpenAI-compatible endpoints. Model names change often; override with --model.
OPENAI_COMPAT_PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "anthropic/claude-sonnet-4"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "qwen2.5:14b"),
}


class OpenAICompatLLM:
    """Any /chat/completions endpoint with function calling (OpenAI, Gemini, Groq, OpenRouter, Ollama)."""

    def __init__(self, preset: str = "openai", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 4096):
        if preset not in OPENAI_COMPAT_PRESETS:
            raise LLMError(f"Unknown provider '{preset}'. Choose from: anthropic, {', '.join(OPENAI_COMPAT_PRESETS)}")
        default_url, key_env, default_model = OPENAI_COMPAT_PRESETS[preset]
        self.base_url = (base_url or os.getenv("REPRO_BASE_URL") or default_url).rstrip("/")
        self.model = model or os.getenv("REPRO_MODEL") or default_model
        self.api_key = api_key or os.getenv(key_env) or ("ollama" if preset == "ollama" else None)
        if not self.api_key:
            raise LLMError(f"{key_env} is not set")
        self.max_tokens = max_tokens

    @staticmethod
    def to_wire(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("text") or None}
                msg.update(m.get("extra") or {})
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {**(tc.get("extra") or {}), "id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m["tool_calls"]
                    ]
                elif not msg["content"]:
                    msg["content"] = "(no output)"
                out.append(msg)
            elif m["role"] == "tool":
                for r in m["results"]:
                    content = r["content"]
                    if r.get("is_error"):
                        content = "ERROR: " + content
                    out.append({"role": "tool", "tool_call_id": r["id"], "content": content})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.to_wire(system, messages),
            "tools": [{"type": "function", "function": t} for t in tools],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = _post_with_retry(self.base_url + "/chat/completions", headers, payload)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response: {str(data)[:500]}") from e
        # Keep provider-specific fields (Gemini puts thought signatures in extra_content) to send back verbatim.
        keep = {k: v for k, v in msg.items() if k in ("extra_content", "provider_specific_fields") and v}
        turn = AssistantTurn(text=(msg.get("content") or "").strip(), usage=data.get("usage", {}), extra=keep)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            tc_extra = {k: v for k, v in tc.items() if k not in ("id", "type", "function", "index") and v}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"__unparsed__": raw}
            turn.tool_calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args, tc_extra))
        return turn


def make_llm(provider: str = "anthropic", model: str | None = None, api_key: str | None = None,
             base_url: str | None = None):
    provider = (provider or "anthropic").lower()
    if provider in ("anthropic", "claude"):
        return AnthropicLLM(model=model, api_key=api_key)
    return OpenAICompatLLM(preset=provider, model=model, api_key=api_key, base_url=base_url)
