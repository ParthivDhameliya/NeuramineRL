from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ReflectMode = Literal["sync", "off"]


def _default_home() -> Path:
    env = os.environ.get("NEURAMINE_HOME")
    return Path(env) if env else Path.cwd() / ".neuramine"


@dataclass
class LearnerConfig:
    """All tunables in one place. Every threshold the lifecycle uses lives
    here because the right values are an open empirical question."""

    home: Path = field(default_factory=_default_home)

    # Retrieval / injection
    k: int = 5
    token_budget: int = 800
    min_confidence: float = 0.15  # lower-bound gate; fresh candidates (~0.21) pass
    candidate_quota: int = 2  # max candidate lessons per injection block

    # Lifecycle
    promote_min_injections: int = 3
    retire_candidate_min_injections: int = 5
    retire_active_min_injections: int = 8
    retire_margin: float = 0.10  # retire when LB < baseline - margin
    disuse_days: float = 60.0
    max_lessons: int = 200  # hard cap on non-retired lessons per scope
    decay_lambda: float = 0.98  # per-day evidence decay toward the prior
    z: float = 1.0  # lower-bound pessimism (std devs below the Beta mean)
    baseline_window: int = 100  # rolling outcomes for the scope baseline

    # Reflection
    reflect: ReflectMode = "sync"
    max_lessons_per_reflection: int = 3
    dedup_duplicate_threshold: float = 0.90
    dedup_merge_threshold: float = 0.75

    # Capture
    max_step_chars: int = 8192
    transcript_char_budget: int = 8000
