"""Neuramine: self-improvement for AI agents — learn from past mistakes,
inject the lessons into future runs, keep only the lessons that help."""

from .config import LearnerConfig
from .learner import Learner, LearnerStats
from .models import Injection, Lesson, LessonDraft, Outcome, Step, Trajectory
from .run import RecallResult, Run

__version__ = "0.1.0.dev0"

__all__ = [
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
    "__version__",
]
