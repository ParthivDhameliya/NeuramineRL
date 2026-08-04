from __future__ import annotations

from ..config import LearnerConfig
from ..llm.base import LLMClient
from ..models import LessonDraft, Outcome, Trajectory
from .prompts import REFLECTION_SCHEMA, REFLECTION_SYSTEM

_MAX_ADVICE_CHARS = 240  # hard cap even if the model ignores the prompt


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
    if len(text) <= char_budget:
        return text
    head = text[: char_budget // 3]
    tail = text[-(char_budget - char_budget // 3) :]
    return f"{head}\n[... transcript truncated ...]\n{tail}"


class LLMReflector:
    def __init__(self, llm: LLMClient, config: LearnerConfig | None = None) -> None:
        self._llm = llm
        self._config = config or LearnerConfig()

    def reflect(self, trajectory: Trajectory, outcome: Outcome) -> list[LessonDraft]:
        transcript = render_transcript(trajectory, self._config.transcript_char_budget)
        user = (
            f"{transcript}\n\n"
            f"OUTCOME: {outcome.status} (source: {outcome.source})\n"
            f"DETAIL: {outcome.detail or '(none)'}"
        )
        response = self._llm.complete(
            [{"role": "user", "content": user}],
            system=REFLECTION_SYSTEM.format(max_lessons=self._config.max_lessons_per_reflection),
            json_schema=REFLECTION_SCHEMA,
            max_tokens=1024,
        )
        drafts: list[LessonDraft] = []
        for item in (response.data or {}).get("lessons", []):
            condition = str(item.get("condition", "")).strip()
            advice = str(item.get("advice", "")).strip()[:_MAX_ADVICE_CHARS]
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
