from __future__ import annotations

import warnings

from ..config import LearnerConfig
from ..llm.base import LLMClient
from ..models import MAX_ADVICE_CHARS, MAX_CONDITION_CHARS, LessonDraft, Outcome, Trajectory
from .prompts import REFLECTION_SCHEMA, REFLECTION_SYSTEM

_MAX_DETAIL_CHARS = 2000  # outcome.detail is caller-supplied and unbounded


def render_transcript(trajectory: Trajectory, char_budget: int = 8000) -> str:
    """Render steps into a compact transcript. When over budget, keep the
    head and tail — failures usually live at the end, setup at the start."""
    lines = [f"TASK: {trajectory.task}"]
    for step in trajectory.steps:
        prefix = f"[{step.kind}]"
        body = step.content
        if step.error:
            body = f"{body}\nERROR: {step.error}" if body else f"ERROR: {step.error}"
        lines.append(f"{prefix} {body}")
    text = "\n".join(lines)
    if char_budget <= 0:
        return ""
    if len(text) <= char_budget:
        return text
    head_len = char_budget // 3
    tail_len = max(0, char_budget - head_len)
    head = text[:head_len]
    # Slice from the front: text[-0:] is text[0:], which would return the whole
    # transcript for a zero-length tail — the opposite of truncating it.
    tail = text[len(text) - tail_len :] if tail_len else ""
    return f"{head}\n[... transcript truncated ...]\n{tail}"


class LLMReflector:
    def __init__(self, llm: LLMClient, config: LearnerConfig | None = None) -> None:
        self._llm = llm
        self._config = config or LearnerConfig()

    def reflect(self, trajectory: Trajectory, outcome: Outcome) -> list[LessonDraft]:
        transcript = render_transcript(trajectory, self._config.transcript_char_budget)
        # detail carries whatever the caller passed to end(error=...) — often a
        # whole HTTP error body. Bound it too, or the budget above buys nothing.
        detail = (outcome.detail or "(none)")[-_MAX_DETAIL_CHARS:]
        user = (
            f"{transcript}\n\n"
            f"OUTCOME: {outcome.status} (source: {outcome.source})\n"
            f"DETAIL: {detail}"
        )
        response = self._llm.complete(
            [{"role": "user", "content": user}],
            system=REFLECTION_SYSTEM.format(max_lessons=self._config.max_lessons_per_reflection),
            json_schema=REFLECTION_SCHEMA,
            max_tokens=1024,
        )
        if response.data is None:
            # A refusal, a content filter, or a max_tokens truncation returns
            # no structured object. Staying silent here is indistinguishable
            # from "this failure taught nothing", which is a very different
            # thing to tell a user who just paid for the call.
            warnings.warn(
                "Reflection returned no structured output (refusal, truncation, or filter); "
                "no lessons recorded for this failure.",
                stacklevel=2,
            )
        drafts: list[LessonDraft] = []
        for item in (response.data or {}).get("lessons", []):
            condition = str(item.get("condition", "")).strip()[:MAX_CONDITION_CHARS]
            advice = str(item.get("advice", "")).strip()[:MAX_ADVICE_CHARS]
            if not condition or not advice:
                continue
            drafts.append(
                LessonDraft(
                    condition=condition,
                    advice=advice,
                    rationale=str(item.get("rationale", "")).strip(),
                )
            )
        return drafts[: self._config.max_lessons_per_reflection]


class FallbackReflector:
    """No-LLM degraded mode: store the raw failure as one low-quality
    observation so nothing is silently lost. The lifecycle will retire it if
    it never helps."""

    def reflect(self, trajectory: Trajectory, outcome: Outcome) -> list[LessonDraft]:
        detail = (outcome.detail or "unknown failure").strip().splitlines()
        summary = detail[-1][:200] if detail else "unknown failure"
        return [
            LessonDraft(
                condition=f"When attempting a task like: {trajectory.task[:120]}",
                advice=f"A previous attempt failed with: {summary}",
                rationale="Raw observation (no reflection LLM configured).",
            )
        ]
