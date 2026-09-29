"""The single LLM interface (step 1.7). Every LLM call in the project goes through here.

    client = make_llm_client(settings)            # None when PRISM_LLM_PROVIDER=none
    resp = await client.generate(prompt, schema=MyModel, trace=trace)
    resp.text, resp.parsed, resp.tokens_in, resp.tokens_out

Providers:
* `ollama`             POST {base_url}/api/generate               (local Ollama server)
* `openai_compatible`  POST {base_url}/chat/completions           (any OpenAI-style endpoint)
* `mock`               scripted responses, for tests only

The two HTTP providers were written against their documented request/response shapes and
are exercised in tests against a local stub HTTP server only. **Neither has been run
against a real Ollama server or hosted endpoint in Phase 1.**

JSON-constrained output: pass `schema` as a pydantic model class or a JSON-schema dict.
The provider is asked for JSON; the reply is parsed (code fences tolerated) and validated.
On a parse/validation failure the call is retried once with a corrective instruction,
then `LLMOutputError` is raised.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.config import Settings, get_settings
from src.telemetry.events import LLM_CALL, estimate_cost_usd
from src.telemetry.logger import RequestTrace

Schema = type[BaseModel] | dict[str, Any] | None

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
JSON_RETRY_NOTE = "\n\nYour previous reply was not valid JSON for the required schema. Reply with JSON only."


class LLMError(RuntimeError):
    pass


class LLMOutputError(LLMError):
    pass


@dataclass
class RawCompletion:
    text: str
    tokens_in: int | None
    tokens_out: int | None


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    provider: str
    model: str
    parsed: Any = None
    tokens_estimated: bool = False
    attempts: int = 1


def _estimate_tokens(text: str) -> int:
    """Rough fallback when a provider reports no usage: ~0.75 words per token."""
    return max(1, round(len(text.split()) / 0.75)) if text else 0


def _json_schema(schema: Schema) -> dict[str, Any] | None:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


def parse_json_output(text: str, schema: Schema) -> Any:
    """Parse a model reply as JSON and validate it against `schema`."""
    m = _FENCE_RE.match(text)
    body = m.group(1) if m else text.strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise LLMOutputError("reply is not JSON") from None
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"reply is not JSON: {exc}") from None
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputError(f"reply does not match schema: {exc.error_count()} errors") from None
    if isinstance(schema, dict):
        if schema.get("type") == "object" and not isinstance(data, dict):
            raise LLMOutputError("reply is not a JSON object")
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            raise LLMOutputError(f"reply is missing required keys {missing}")
    return data


class LLMClient(ABC):
    provider = "abstract"

    def __init__(self, model: str, settings: Settings | None = None):
        self.model = model
        self.settings = settings or get_settings()

    @abstractmethod
    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion: ...

    async def generate(
        self,
        prompt: str,
        schema: Schema = None,
        system: str | None = None,
        trace: RequestTrace | None = None,
    ) -> LLMResponse:
        js = _json_schema(schema)
        t0 = time.perf_counter()
        tokens_in = tokens_out = 0
        estimated = False
        parsed: Any = None
        attempts = 0
        current_prompt = prompt
        error: LLMOutputError | None = None
        raw = RawCompletion("", 0, 0)
        for attempts in (1, 2):
            raw = await self._complete(current_prompt, system, js)
            if raw.tokens_in is None or raw.tokens_out is None:
                estimated = True
            tokens_in += raw.tokens_in if raw.tokens_in is not None else _estimate_tokens((system or "") + current_prompt)
            tokens_out += raw.tokens_out if raw.tokens_out is not None else _estimate_tokens(raw.text)
            if js is None:
                break
            try:
                parsed = parse_json_output(raw.text, schema)
                error = None
                break
            except LLMOutputError as exc:
                error = exc
                current_prompt = prompt + JSON_RETRY_NOTE
        latency_ms = (time.perf_counter() - t0) * 1000

        if trace is not None:
            trace.emit(
                LLM_CALL,
                stage_latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                est_cost_usd=estimate_cost_usd(
                    tokens_in, tokens_out, self.settings.cost_per_1k_input, self.settings.cost_per_1k_output
                ),
                provider=self.provider,
                model=self.model,
                json_mode=js is not None,
                attempts=attempts,
                tokens_estimated=estimated,
                ok=error is None,
            )
        if error is not None:
            raise error
        return LLMResponse(
            text=raw.text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            provider=self.provider,
            model=self.model,
            parsed=parsed,
            tokens_estimated=estimated,
            attempts=attempts,
        )


# ------------------------------------------------------------------ HTTP helpers


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise LLMError(f"HTTP {exc.code} from {url}: {body}") from None
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {url}: {exc.reason}") from None


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str, settings: Settings | None = None):
        super().__init__(model, settings)
        self.base_url = base_url.rstrip("/")

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.settings.llm_max_tokens},
        }
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema
        data = await asyncio.to_thread(
            _post_json, f"{self.base_url}/api/generate", payload, {}, self.settings.llm_timeout_s
        )
        return RawCompletion(data.get("response", ""), data.get("prompt_eval_count"), data.get("eval_count"))


class OpenAICompatibleClient(LLMClient):
    provider = "openai_compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "", settings: Settings | None = None):
        super().__init__(model, settings)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.settings.llm_max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = await asyncio.to_thread(
            _post_json, f"{self.base_url}/chat/completions", payload, headers, self.settings.llm_timeout_s
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return RawCompletion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"))


class MockLLMClient(LLMClient):
    """Returns scripted replies. For tests only; never selected by default."""

    provider = "mock"

    def __init__(self, replies: list[str] | Callable[[str], str], settings: Settings | None = None):
        super().__init__("mock", settings)
        self._replies = replies
        self.prompts: list[str] = []

    async def _complete(self, prompt: str, system: str | None, json_schema: dict | None) -> RawCompletion:
        self.prompts.append(prompt)
        if callable(self._replies):
            return RawCompletion(self._replies(prompt), None, None)
        if not self._replies:
            raise LLMError("MockLLMClient ran out of scripted replies")
        return RawCompletion(self._replies.pop(0), None, None)


def make_llm_client(settings: Settings | None = None) -> LLMClient | None:
    """Build the configured client, or None when PRISM_LLM_PROVIDER=none."""
    s = settings or get_settings()
    provider = s.llm_provider.lower()
    if provider == "none":
        return None
    if not s.llm_model:
        raise LLMError(f"PRISM_LLM_PROVIDER={provider} requires PRISM_LLM_MODEL")
    if provider == "ollama":
        return OllamaClient(s.llm_model, s.llm_base_url, settings=s)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(s.llm_model, s.llm_base_url, s.llm_api_key, settings=s)
    raise LLMError(f"unknown PRISM_LLM_PROVIDER {provider!r} (expected none, ollama or openai_compatible)")
