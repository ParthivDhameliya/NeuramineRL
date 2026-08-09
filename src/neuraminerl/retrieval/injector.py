from __future__ import annotations

from ..config import LearnerConfig
from ..models import Lesson

PREAMBLE = (
    "Lessons from previous attempts at similar tasks. Apply them unless clearly\n"
    "inapplicable to the current situation."
)


OPEN_TAG = "<learned_lessons>"
CLOSE_TAG = "</learned_lessons>"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _neutralize(text: str) -> str:
    """Strip the block's own delimiters out of lesson text.

    Lesson text is model-written from an untrusted transcript, so it can
    contain the closing tag. Rendered verbatim it would end the data block
    early and the remainder would read as top-level prompt instructions.
    """
    return text.replace(CLOSE_TAG, "").replace(OPEN_TAG, "")


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
        lines: list[str] = []
        included: list[Lesson] = []
        for lesson in lessons:
            line = f"{len(lines) + 1}. {_neutralize(lesson.text)}"
            # Cost the assembled block, not the pieces: estimate_tokens floors,
            # so summing per-part costs drops each part's remainder and every
            # joining newline, letting the rendered block exceed the budget the
            # README calls a hard cap.
            if estimate_tokens(self._assemble([*lines, line])) > budget:
                continue
            lines.append(line)
            included.append(lesson)
        if not lines:
            return "", []
        return self._assemble(lines), included

    @staticmethod
    def _assemble(lines: list[str]) -> str:
        body = "\n".join(lines)
        return f"{OPEN_TAG}\n{PREAMBLE}\n{body}\n{CLOSE_TAG}"
