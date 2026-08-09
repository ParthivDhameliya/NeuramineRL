"""Thin httpx client for the Gemini API (generateContent).

Structured output uses ``responseMimeType: application/json`` with
``responseSchema``; Gemini's schema dialect is an OpenAPI subset, so
JSON-Schema-only keys are stripped before sending.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..exceptions import LLMError
from .base import LLMResponse, Message

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_UNSUPPORTED_SCHEMA_KEYS = ("additionalProperties", "$schema")


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            out[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            out[key] = [_sanitize_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        # GEMINI_API_KEY/GOOGLE_API_KEY authenticate to Google and nowhere
        # else. A custom base_url is a different operator's server, so the
        # provider key is never forwarded to it: pass api_key=, or set
        # NEURAMINERL_API_KEY. It may also be a local keyless proxy.
        self._api_key = api_key or os.environ.get("NEURAMINERL_API_KEY", "")
        if not self._api_key and self._base_url == DEFAULT_BASE_URL:
            self._api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            if not self._api_key:
                raise LLMError("No Gemini API key (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        contents = [
            {
                "role": "model" if message.get("role") == "assistant" else "user",
                "parts": [{"text": str(message.get("content", ""))}],
            }
            for message in messages
        ]
        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if json_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _sanitize_schema(json_schema)
        body: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        try:
            response = httpx.post(
                f"{self._base_url}/models/{self.model}:generateContent",
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Gemini API error {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini API request failed: {exc}") from exc

        payload = response.json()
        candidates = payload.get("candidates") or []
        text_parts: list[str] = []
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []) or []:
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        text = "".join(text_parts)
        data: dict[str, Any] | None = None
        if json_schema is not None and text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"Gemini returned invalid JSON for structured output: {exc}"
                ) from exc
        usage_meta = payload.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            data=data,
            usage={
                "input_tokens": int(usage_meta.get("promptTokenCount", 0)),
                # 2.5-series models report reasoning separately and bill it as
                # output; omitting it under-reports spend to on_usage.
                "output_tokens": int(usage_meta.get("candidatesTokenCount", 0))
                + int(usage_meta.get("thoughtsTokenCount", 0)),
            },
        )
