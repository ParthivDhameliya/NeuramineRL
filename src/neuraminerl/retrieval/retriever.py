from __future__ import annotations

from ..config import LearnerConfig
from ..embeddings.base import Embedder
from ..lessons.scoring import beta_lower_bound, beta_mean, evidence_trials
from ..models import Lesson
from ..store.base import Store


class Retriever:
    """Ranks active lessons by cosine-times-health, then fills remaining slots
    with up to ``candidate_quota`` candidates (exploration). At least one
    slot is held for candidates when any exist, so new lessons can always
    earn evidence."""

    def __init__(self, store: Store, embedder: Embedder, config: LearnerConfig) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config

    def retrieve(self, task: str, scope: str) -> list[tuple[Lesson, float]]:
        cfg = self._config
        query = self._embedder.embed([task])[0]
        pool = self._store.search_lessons(
            query, scope, states=("candidate", "active"), k=max(cfg.k * 4, 20)
        )

        # The confidence gate is relative to the scope's baseline: when the
        # agent fails most runs anyway, a low lower-bound is not evidence the
        # lesson is harmful (it shares blame for failures other lessons
        # haven't been learned for yet).
        baseline = self._store.baseline_success_rate(scope, cfg.baseline_window)
        gate = min(cfg.min_confidence, max(0.0, baseline - cfg.retire_margin))

        actives: list[tuple[Lesson, float]] = []
        candidates: list[tuple[Lesson, float]] = []
        for lesson, similarity in pool:
            lower = beta_lower_bound(lesson.alpha, lesson.beta, cfg.z)
            mean = beta_mean(lesson.alpha, lesson.beta)
            # The gate uses the mean and only applies once a lesson has had
            # its exploration chances — the same condition the lifecycle uses
            # to retire, so nothing gets stuck un-injectable but un-retirable.
            # Like the lifecycle, it counts credited evidence rather than
            # exposure, so unfinished runs cannot gate a lesson out.
            if evidence_trials(
                lesson.alpha, lesson.beta
            ) >= cfg.retire_candidate_min_injections and (mean < gate):
                continue
            if lesson.state == "active":
                actives.append((lesson, similarity * (0.5 + lower)))
            else:
                candidates.append((lesson, similarity))

        actives.sort(key=lambda pair: -pair[1])
        candidates.sort(key=lambda pair: -pair[1])

        # Hold a slot for exploration only when it can actually be filled and
        # is not the only slot there is. Reserving unconditionally silently
        # returned k-1 lessons whenever candidate_quota was 0, and at k=1 gave
        # the single slot to a candidate forever, starving every active lesson.
        reserved = 1 if (candidates and cfg.candidate_quota > 0 and cfg.k > 1) else 0
        chosen = actives[: max(cfg.k - reserved, 0)]
        remaining = cfg.k - len(chosen)
        chosen += candidates[: min(cfg.candidate_quota, max(remaining, 0))]
        return chosen
