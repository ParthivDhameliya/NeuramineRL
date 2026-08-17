"""NeuramineRL: self-improvement for AI agents — learn from past mistakes,
inject the lessons into future runs, keep only the lessons that help."""

from .aio import AsyncLearner, AsyncRun
from .config import LearnerConfig
from .learner import Learner, LearnerStats
from .llm.base import UsageEvent
from .models import Injection, Lesson, LessonDraft, Outcome, Step, Trajectory
from .run import RecallResult, Run

__version__ = "0.2.2"

__all__ = [
    "AsyncLearner",
    "AsyncRun",
    "Injection",
    "Learner",
    "LearnerConfig",
    "LearnerStats",
    "Lesson",
    "LessonDraft",
    "Outcome",
    "RecallResult",
    "Run",
    "Step",
    "Trajectory",
    "UsageEvent",
    "__version__",
]
