"""Lesson lifecycle: credit assignment, decay, and state transitions.

State machine::

    (incident) → CANDIDATE → ACTIVE → RETIRED
                     └──────────────→ RETIRED   (had its chance)
    dedup can mark either non-retired state as MERGED.

Candidates are injected (capped, ranked below actives) so they can earn
evidence — built-in exploration without an explicit bandit.
"""

from __future__ import annotations

from ..config import LearnerConfig
from ..models import Lesson, LessonState, Outcome, utcnow
from ..store.base import Store
from .scoring import beta_lower_bound, beta_mean, days_between, decay


class Lifecycle:
    def __init__(self, store: Store, config: LearnerConfig | None = None) -> None:
        self._store = store
        self._config = config or LearnerConfig()

    # -- credit assignment -------------------------------------------------

    def credit(self, outcome: Outcome) -> None:
        """Apply an outcome to every lesson injected into its trajectory.

        Each injection is one full Bernoulli trial for that lesson: the
        evidence estimates P(success | lesson injected), and the baseline
        comparison in ``maintain`` normalizes away "the run would have
        succeeded anyway". (Splitting credit across co-injected lessons was
        tried and rejected: the weight varies with how many lessons exist at
        the time, which biases early lessons downward.) Re-crediting
        (delayed feedback) first reverses exactly what each injection
        previously added.
        """
        injections = self._store.injections_for(outcome.trajectory_id)
        if not injections:
            return
        if outcome.status == "success":
            add_alpha, add_beta = 1.0, 0.0
        elif outcome.status == "failure":
            add_alpha, add_beta = 0.0, 1.0
        elif outcome.status == "partial":
            score = outcome.score if outcome.score is not None else 0.5
            add_alpha, add_beta = score, 1.0 - score
        else:  # unknown teaches nothing
            return

        for injection in injections:
            lesson = self._store.get_lesson(injection.lesson_id)
            # Reverse any previously applied credit (idempotent re-crediting).
            lesson.alpha -= injection.credited_alpha
            lesson.beta -= injection.credited_beta
            lesson.alpha += add_alpha
            lesson.beta += add_beta
            self._store.upsert_lesson(lesson)
            injection.credited = True
            injection.credited_alpha = add_alpha
            injection.credited_beta = add_beta
            self._store.update_injection(injection)

    # -- maintenance: decay + transitions + cap ------------------------------

    def maintain(self, scope: str) -> None:
        baseline = self._store.baseline_success_rate(scope, self._config.baseline_window)
        now = utcnow()
        cfg = self._config
        lessons = self._store.get_lessons(scope, states=("candidate", "active"))

        for lesson in lessons:
            days = days_between(lesson.last_decay_at, now)
            if days >= 1.0:
                lesson.alpha, lesson.beta = decay(lesson.alpha, lesson.beta, days, cfg.decay_lambda)
                lesson.last_decay_at = now
                self._store.upsert_lesson(lesson)

            lower = beta_lower_bound(lesson.alpha, lesson.beta, cfg.z)
            mean = beta_mean(lesson.alpha, lesson.beta)

            # Asymmetry, on purpose: PROMOTE on the pessimistic lower bound,
            # RETIRE only when even the central estimate (mean) is clearly
            # below baseline. Retiring on the lower bound punishes exactly
            # the lessons that raised the baseline in the first place.

            # Below the injection gate a lesson can never be injected again,
            # so it can never earn the evidence to recover: retire it rather
            # than leave a zombie candidate. The gate is baseline-relative
            # (mirroring the retriever): when the agent fails most runs
            # anyway, blame is not evidence a lesson is harmful.
            gate = min(cfg.min_confidence, max(0.0, baseline - cfg.retire_margin))
            if mean < gate and lesson.times_injected >= cfg.retire_candidate_min_injections:
                self._transition(
                    lesson, "retired", f"mean {mean:.2f} below injection gate {gate:.2f}"
                )
                continue

            if lesson.state == "candidate":
                if lesson.times_injected >= cfg.promote_min_injections and lower >= baseline:
                    self._transition(lesson, "active", f"LB {lower:.2f} >= baseline {baseline:.2f}")
                elif (
                    lesson.times_injected >= cfg.retire_candidate_min_injections
                    and mean < baseline - cfg.retire_margin
                ):
                    self._transition(
                        lesson, "retired", f"mean {mean:.2f} < baseline-{cfg.retire_margin}"
                    )
            elif lesson.state == "active":
                stale = (
                    lesson.last_injected_at is not None
                    and days_between(lesson.last_injected_at, now) > cfg.disuse_days
                    and mean < baseline
                )
                underperforming = (
                    lesson.times_injected >= cfg.retire_active_min_injections
                    and mean < baseline - cfg.retire_margin
                )
                if underperforming or stale:
                    reason = "disuse" if stale else f"mean {mean:.2f} below baseline margin"
                    self._transition(lesson, "retired", reason)

        self._enforce_cap(scope, baseline)

    def _transition(self, lesson: Lesson, state: LessonState, reason: str) -> None:
        old = lesson.state
        lesson.state = state
        self._store.upsert_lesson(lesson)
        self._store.log_event(lesson.id, f"{old}->{state}", reason)

    def _enforce_cap(self, scope: str, baseline: float) -> None:
        """Hard cap on non-retired lessons per scope; retire the weakest."""
        cfg = self._config
        lessons = self._store.get_lessons(scope, states=("candidate", "active"))
        overflow = len(lessons) - cfg.max_lessons
        if overflow <= 0:
            return
        lessons.sort(key=lambda item: beta_lower_bound(item.alpha, item.beta, cfg.z))
        for lesson in lessons[:overflow]:
            self._transition(lesson, "retired", "pruned: scope at max_lessons cap")
