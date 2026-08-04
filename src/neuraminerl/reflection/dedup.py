"""Write-time dedup/merge: keeps the lesson set small, general, and
evidence-bearing instead of an ever-growing pile of one-off incident notes.

- cosine >= duplicate_threshold: recurrence of a known failure. Append
  provenance only — alpha/beta move exclusively via injection credit.
- merge_threshold <= cosine < duplicate_threshold: ask the LLM whether to
  generalize the existing lesson (evidence transfers, version bumps).
- below merge_threshold: genuinely new; insert as a candidate.
"""

from __future__ import annotations

from ..config import LearnerConfig
from ..embeddings.base import Embedder, Vector
from ..llm.base import LLMClient
from ..models import Lesson, LessonDraft, utcnow
from ..store.base import Store
from .prompts import MERGE_SCHEMA, MERGE_SYSTEM


class Deduplicator:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        llm: LLMClient | None,
        config: LearnerConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._config = config or LearnerConfig()

    def resolve(self, draft: LessonDraft, scope: str, trajectory_id: str) -> Lesson:
        """Insert, reinforce, or generalize. Returns the surviving lesson."""
        draft_text = f"{draft.condition.rstrip('.')}: {draft.advice}"
        vector = self._embedder.embed([draft_text])[0]
        neighbors = self._store.search_lessons(vector, scope, states=("candidate", "active"), k=1)
        if neighbors:
            existing, similarity = neighbors[0]
            if similarity >= self._config.dedup_duplicate_threshold:
                return self._reinforce(existing, trajectory_id)
            if similarity >= self._config.dedup_merge_threshold and self._llm is not None:
                return self._merge(draft, existing, trajectory_id)
        return self._insert(draft, vector, scope, trajectory_id)

    def _insert(self, draft: LessonDraft, vector: Vector, scope: str, trajectory_id: str) -> Lesson:
        lesson = Lesson(
            scope=scope,
            condition=draft.condition,
            advice=draft.advice,
            rationale=draft.rationale,
            source_trajectory_ids=[trajectory_id],
        )
        self._store.upsert_lesson(lesson, embedding=vector)
        self._store.log_event(lesson.id, "created", f"from trajectory {trajectory_id}")
        return lesson

    def _reinforce(self, existing: Lesson, trajectory_id: str) -> Lesson:
        # The failure recurred even though the lesson exists — evidence it is
        # *needed*, not that it works, so only provenance is updated.
        if trajectory_id not in existing.source_trajectory_ids:
            existing.source_trajectory_ids.append(trajectory_id)
        existing.last_reinforced_at = utcnow()
        self._store.upsert_lesson(existing)
        self._store.log_event(existing.id, "reinforced", f"recurred in {trajectory_id}")
        return existing

    def _merge(self, draft: LessonDraft, existing: Lesson, trajectory_id: str) -> Lesson:
        assert self._llm is not None
        user = (
            f"EXISTING lesson:\ncondition: {existing.condition}\nadvice: {existing.advice}\n\n"
            f"NEW draft:\ncondition: {draft.condition}\nadvice: {draft.advice}"
        )
        response = self._llm.complete(
            [{"role": "user", "content": user}],
            system=MERGE_SYSTEM,
            json_schema=MERGE_SCHEMA,
            max_tokens=512,
        )
        data = response.data or {}
        decision = data.get("decision", "distinct")
        if decision == "duplicate":
            return self._reinforce(existing, trajectory_id)
        if decision == "generalize" and data.get("condition") and data.get("advice"):
            existing.condition = str(data["condition"]).strip()
            existing.advice = str(data["advice"]).strip()
            existing.version += 1
            if trajectory_id not in existing.source_trajectory_ids:
                existing.source_trajectory_ids.append(trajectory_id)
            existing.last_reinforced_at = utcnow()
            new_vector = self._embedder.embed([existing.text])[0]
            self._store.upsert_lesson(existing, embedding=new_vector)
            self._store.log_event(
                existing.id, "generalized", f"v{existing.version} absorbing {trajectory_id}"
            )
            return existing
        vector = self._embedder.embed([f"{draft.condition.rstrip('.')}: {draft.advice}"])[0]
        return self._insert(draft, vector, existing.scope, trajectory_id)
