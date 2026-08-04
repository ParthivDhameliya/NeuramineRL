"""Two agents for the demo, plus a scripted reflection LLM for offline mode.

- ``LiveAgent``: a real LLM (from env keys) reads the API docs + the injected
  lessons block, emits order JSON, and reacts to error responses. The honest
  experiment.
- ``ScriptedAgent``: deterministic stand-in that behaves naively unless the
  injected lessons tell it better. Exercises the full neuramine pipeline
  (capture -> reflect -> dedup -> inject -> credit) without API keys or
  nondeterminism — this is what CI runs.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from mock_api import DOCS, ApiResult, OrdersAPI, OrderTask

from neuramine import Run
from neuramine.llm.base import LLMClient, LLMResponse, Message


@dataclass
class EpisodeResult:
    ok: bool
    error: str | None


# --------------------------------------------------------------------------
# Scripted (offline) mode
# --------------------------------------------------------------------------


class ScriptedAgent:
    """Naive by default; applies a fix only when the lessons block mentions it."""

    MAX_ATTEMPTS = 3

    def attempt(self, task: OrderTask, api: OrdersAPI, lessons: str, run: Run) -> EpisodeResult:
        text = lessons.lower()
        payload: dict[str, Any] = {
            "item": task.item,
            "quantity": task.quantity,
            "amount": task.amount_cents / 100.0,  # naive: float dollars
            "country": task.country_name.lower(),  # naive: lowercase name
            "ship_date": task.ship_date_text,  # naive: prose date
        }
        # Deliberately narrow markers, one per quirk, so one lesson can never
        # accidentally cover for another — a retired lesson shows up as a
        # regression, which keeps the demo honest.
        if "yyyy-mm-dd" in text:
            payload["ship_date"] = task.ship_date_iso
        if "cents" in text:
            payload["amount"] = task.amount_cents
        if "iso-2" in text:
            payload["country"] = task.country_code
        headers: dict[str, str] = {}
        if "idempotency-key header" in text:
            headers["Idempotency-Key"] = uuid.uuid4().hex
        retry_409_with_new_key = "new idempotency-key" in text

        result = ApiResult(0, "not attempted")
        for _attempt in range(self.MAX_ATTEMPTS):
            run.log_tool_call(
                "place_order",
                {"payload": payload, "headers": list(headers)},
            )
            result = api.place_order(payload, headers, flaky_id=task.id if task.flaky else None)
            run.note(f"API response {result.status}: {result.body}")
            if result.ok:
                return EpisodeResult(True, None)
            if result.status == 409 and retry_409_with_new_key and headers:
                headers["Idempotency-Key"] = uuid.uuid4().hex
                continue
            break  # scripted agent can't discover fixes within an episode
        return EpisodeResult(False, f"{result.status}: {result.body}")


class ScriptedReflectionLLM:
    """Stands in for the reflection LLM offline. Maps the error evidence in
    the transcript to the lesson a competent reflector would write. (The live
    mode does this with a real model — this class is for determinism, not
    proof.)"""

    model = "scripted-reflection"

    _RULES: ClassVar[list[tuple[str, str, str]]] = [
        (
            "missing required header 'idempotency-key'",
            "When POSTing to the orders API",
            "Include an Idempotency-Key header with a unique value on every request.",
        ),
        (
            "do not resubmit the same key",
            "When the orders API returns a 409 conflict",
            "Retry once with a fresh, new Idempotency-Key; never resubmit the same key.",
        ),
        (
            "invalid value for field 'ship_date'",
            "When setting ship_date for the orders API",
            "Use ISO format YYYY-MM-DD.",
        ),
        (
            "field 'amount' has invalid type",
            "When setting the amount for the orders API",
            "Send an integer number of cents, not a dollar float.",
        ),
        (
            "invalid value for field 'country'",
            "When setting the country for the orders API",
            "Use the uppercase ISO-2 country code (e.g. DE), not the country name.",
        ),
    ]

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        properties = (json_schema or {}).get("properties", {})
        if "decision" in properties:  # dedup merge question
            return LLMResponse(data={"decision": "duplicate", "condition": "", "advice": ""})
        transcript = " ".join(str(m.get("content", "")) for m in messages).lower()
        for marker, condition, advice in self._RULES:
            if marker in transcript:
                return LLMResponse(
                    data={
                        "lessons": [
                            {
                                "condition": condition,
                                "advice": advice,
                                "rationale": f"API said: {marker}",
                            }
                        ]
                    }
                )
        return LLMResponse(data={"lessons": []})


# --------------------------------------------------------------------------
# Live mode
# --------------------------------------------------------------------------

_LIVE_SYSTEM = """You are an ordering agent. Place the order via the API described below.
{docs}

{lessons}

Respond with ONLY a JSON object, no prose, of the shape:
{{"payload": {{...the order fields...}}, "headers": {{...HTTP headers, if any...}}}}"""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in model output: {text[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model output is not a JSON object")
    return parsed


class LiveAgent:
    MAX_ATTEMPTS = 4

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def attempt(self, task: OrderTask, api: OrdersAPI, lessons: str, run: Run) -> EpisodeResult:
        system = _LIVE_SYSTEM.format(docs=DOCS, lessons=lessons or "(no prior lessons)")
        messages: list[Message] = [{"role": "user", "content": task.describe()}]
        run.log(messages)
        result = ApiResult(0, "not attempted")
        for _ in range(self.MAX_ATTEMPTS):
            response = self._llm.complete(messages, system=system, max_tokens=800)
            run.log({"role": "assistant", "content": response.text})
            try:
                emitted = _extract_json(response.text)
            except (ValueError, json.JSONDecodeError) as exc:
                messages.append({"role": "assistant", "content": response.text})
                messages.append({"role": "user", "content": f"Bad output ({exc}). JSON only."})
                continue
            payload = emitted.get("payload", {})
            headers = {str(k): str(v) for k, v in (emitted.get("headers") or {}).items()}
            run.log_tool_call("place_order", {"payload": payload, "headers": list(headers)})
            result = api.place_order(payload, headers, flaky_id=task.id if task.flaky else None)
            run.note(f"API response {result.status}: {result.body}")
            if result.ok:
                return EpisodeResult(True, None)
            messages.append({"role": "assistant", "content": response.text})
            messages.append(
                {
                    "role": "user",
                    "content": f"API error {result.status}: {result.body}. Correct and retry.",
                }
            )
        return EpisodeResult(False, f"{result.status}: {result.body}")
