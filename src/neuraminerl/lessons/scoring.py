"""Lesson health math: Beta-Bernoulli evidence with time decay.

A lesson's health is a pessimistic (lower-bound) estimate of
P(success | lesson injected), compared against the agent's rolling baseline
success rate. Decay pulls old evidence back toward the uniform prior so a
lesson can't coast forever on ancient wins.
"""

from __future__ import annotations

import math
from datetime import datetime


def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def evidence_trials(alpha: float, beta: float) -> float:
    """Credited outcomes behind this evidence, decayed.

    Every credited outcome adds exactly 1.0 across alpha+beta (success 1/0,
    failure 0/1, partial score/1-score), starting from the Beta(1,1) prior, and
    ``decay`` shrinks the excess over that prior. So this is the trial count
    lifecycle decisions should use. ``times_injected`` counts *exposure* and is
    incremented at retrieval time, before any outcome exists — gating on it
    retires lessons for runs that were abandoned or ended without a verdict.
    """
    return max(0.0, alpha + beta - 2.0)


def beta_lower_bound(alpha: float, beta: float, z: float = 1.0) -> float:
    """Normal approximation of a lower quantile of Beta(alpha, beta):
    mean minus z standard deviations. z=1.0 is roughly the 16th percentile."""
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / (total * total * (total + 1.0))
    return max(0.0, mean - z * math.sqrt(variance))


def decay(alpha: float, beta: float, days: float, lam: float = 0.98) -> tuple[float, float]:
    """Shrink evidence toward the Beta(1,1) prior by lam**days."""
    if days <= 0:
        return alpha, beta
    factor = lam**days
    return 1.0 + (alpha - 1.0) * factor, 1.0 + (beta - 1.0) * factor


def days_between(earlier_iso: str, later_iso: str) -> float:
    earlier = datetime.fromisoformat(earlier_iso)
    later = datetime.fromisoformat(later_iso)
    return max(0.0, (later - earlier).total_seconds() / 86400.0)
