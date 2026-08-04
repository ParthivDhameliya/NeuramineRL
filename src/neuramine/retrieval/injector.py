from __future__ import annotations

from ..config import LearnerConfig
from ..models import Lesson

PREAMBLE = (
    "Lessons from previous attempts at similar tasks. Apply them unless clearly\n"
    "inapplicable to the current situation."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Injector:
    """Renders lessons into a delimited, numbered prompt block under a hard
    token budget. Lessons never truncate mid-text: a lesson that doesn't fit
    is dropped."""

    def __init__(self, config: LearnerConfig) -> None:
        self._config = config

    def render(self, lessons: list[Lesson]) -> tuple[str, list[Lesson]]:
        """Returns (block, included_lessons). Empty string when nothing fits."""
        if not lessons:
            return "", []
        budget = self._config.token_budget
        frame = f"<learned_lessons>\n{PREAMBLE}\n</learned_lessons>"
        used = estimate_tokens(frame)
        lines: list[str] = []
        included: list[Lesson] = []
        for lesson in lessons:
            line = f"{len(lines) + 1}. {lesson.text}"
            cost = estimate_tokens(line)
            if used + cost > budget:
                continue
            used += cost
            lines.append(line)
            included.append(lesson)
        if not lines:
            return "", []
        body = "\n".join(lines)
        return f"<learned_lessons>\n{PREAMBLE}\n{body}\n</learned_lessons>", included
