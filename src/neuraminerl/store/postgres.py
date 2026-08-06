"""Postgres store: the multi-process/multi-worker counterpart to SQLite.

Same 16-method Store protocol, same semantics. Differences from SqliteStore:
JSON columns are JSONB, embeddings are BYTEA, and there is deliberately NO
in-memory matrix cache — another process may insert lessons at any time, so
every search re-reads the scope's embeddings (capped at ~200 lessons by the
lifecycle, this stays cheap). Vector search remains numpy brute force;
pgvector is an optimization for far larger stores, not a requirement.

Requires the ``postgres`` extra: ``pip install neuraminerl[postgres]``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..embeddings.base import Vector
from ..exceptions import ConfigError, NeuramineRLError
from ..models import (
    Injection,
    Lesson,
    LessonState,
    Outcome,
    Step,
    Trajectory,
    new_id,
    utcnow,
)

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_traj_scope ON trajectories(scope, started_at);

CREATE TABLE IF NOT EXISTS steps (
  id TEXT PRIMARY KEY,
  trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
  idx INTEGER NOT NULL,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_steps_traj ON steps(trajectory_id, idx);

CREATE TABLE IF NOT EXISTS outcomes (
  id TEXT PRIMARY KEY,
  seq BIGSERIAL,
  trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
  status TEXT NOT NULL,
  score DOUBLE PRECISION,
  source TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outcomes_traj ON outcomes(trajectory_id, created_at);

CREATE TABLE IF NOT EXISTS lessons (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  condition TEXT NOT NULL,
  advice TEXT NOT NULL,
  rationale TEXT NOT NULL DEFAULT '',
  embedding BYTEA,
  state TEXT NOT NULL DEFAULT 'candidate',
  alpha DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  beta DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  times_injected INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  merged_into TEXT,
  source_trajectory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
  last_injected_at TEXT,
  last_reinforced_at TEXT,
  last_decay_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lessons_scope_state ON lessons(scope, state);

CREATE TABLE IF NOT EXISTS injections (
  id TEXT PRIMARY KEY,
  trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
  lesson_id TEXT NOT NULL REFERENCES lessons(id),
  lesson_version INTEGER NOT NULL,
  retrieval_score DOUBLE PRECISION NOT NULL,
  rank INTEGER NOT NULL,
  credited BOOLEAN NOT NULL DEFAULT FALSE,
  credited_alpha DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  credited_beta DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inj_traj ON injections(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_inj_lesson ON injections(lesson_id);

CREATE TABLE IF NOT EXISTS lesson_events (
  id TEXT PRIMARY KEY,
  lesson_id TEXT NOT NULL,
  event TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_lesson ON lesson_events(lesson_id, created_at);

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def _as_json(value: Any) -> Any:
    """JSONB values come back as parsed objects; tolerate TEXT-ish drivers."""
    if isinstance(value, str):
        return json.loads(value)
    return value


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ConfigError(
                "PostgresStore requires psycopg: pip install neuraminerl[postgres]"
            ) from exc
        self._jsonb = Jsonb
        self._lock = threading.RLock()
        self._conn: Any = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
        with self._lock, self._conn.transaction():
            self._conn.execute(_SCHEMA)
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (str(_SCHEMA_VERSION),),
            )

    # -- trajectories ------------------------------------------------------

    def save_trajectory(self, trajectory: Trajectory) -> None:
        with self._lock, self._conn.transaction():
            self._conn.execute(
                """INSERT INTO trajectories
                     (id, scope, task, status, started_at, ended_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                     status=EXCLUDED.status, ended_at=EXCLUDED.ended_at,
                     metadata=EXCLUDED.metadata""",
                (
                    trajectory.id,
                    trajectory.scope,
                    trajectory.task,
                    trajectory.status,
                    trajectory.started_at,
                    trajectory.ended_at,
                    self._jsonb(trajectory.metadata),
                ),
            )

    def get_trajectory(self, trajectory_id: str, *, with_steps: bool = True) -> Trajectory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trajectories WHERE id = %s", (trajectory_id,)
            ).fetchone()
            if row is None:
                raise NeuramineRLError(f"Unknown trajectory: {trajectory_id}")
            trajectory = Trajectory(
                id=row["id"],
                scope=row["scope"],
                task=row["task"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                metadata=_as_json(row["metadata"]),
            )
            if with_steps:
                step_rows = self._conn.execute(
                    "SELECT * FROM steps WHERE trajectory_id = %s ORDER BY idx",
                    (trajectory_id,),
                ).fetchall()
                trajectory.steps = [
                    Step(
                        id=s["id"],
                        trajectory_id=s["trajectory_id"],
                        index=s["idx"],
                        kind=s["kind"],
                        content=s["content"],
                        error=s["error"],
                        created_at=s["created_at"],
                        metadata=_as_json(s["metadata"]),
                    )
                    for s in step_rows
                ]
            return trajectory

    def add_steps(self, steps: Sequence[Step]) -> None:
        if not steps:
            return
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO steps
                     (id, trajectory_id, idx, kind, content, error, created_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        s.id,
                        s.trajectory_id,
                        s.index,
                        s.kind,
                        s.content,
                        s.error,
                        s.created_at,
                        self._jsonb(s.metadata),
                    )
                    for s in steps
                ],
            )

    # -- outcomes ----------------------------------------------------------

    def save_outcome(self, outcome: Outcome) -> None:
        with self._lock, self._conn.transaction():
            self._conn.execute(
                """INSERT INTO outcomes
                     (id, trajectory_id, status, score, source, detail, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    outcome.id,
                    outcome.trajectory_id,
                    outcome.status,
                    outcome.score,
                    outcome.source,
                    outcome.detail,
                    outcome.created_at,
                ),
            )

    def latest_outcome(self, trajectory_id: str) -> Outcome | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM outcomes WHERE trajectory_id = %s
                   ORDER BY created_at DESC, seq DESC LIMIT 1""",
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._outcome_from_row(row)

    @staticmethod
    def _outcome_from_row(row: dict[str, Any]) -> Outcome:
        return Outcome(
            id=row["id"],
            trajectory_id=row["trajectory_id"],
            status=row["status"],
            score=row["score"],
            source=row["source"],
            detail=row["detail"],
            created_at=row["created_at"],
        )

    def baseline_success_rate(self, scope: str, window: int = 100) -> float:
        """Rolling success rate over the scope's most recent primary outcomes.
        Returns 0.5 until there are at least 5 outcomes."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT status, score FROM (
                     SELECT DISTINCT ON (o.trajectory_id)
                       o.status, o.score, o.created_at
                     FROM outcomes o
                     JOIN trajectories t ON t.id = o.trajectory_id
                     WHERE t.scope = %s
                     ORDER BY o.trajectory_id, o.created_at DESC, o.seq DESC
                   ) latest
                   WHERE status != 'unknown'
                   ORDER BY created_at DESC LIMIT %s""",
                (scope, window),
            ).fetchall()
        values: list[float] = []
        for row in rows:
            if row["status"] == "success":
                values.append(1.0)
            elif row["status"] == "failure":
                values.append(0.0)
            else:  # partial
                values.append(row["score"] if row["score"] is not None else 0.5)
        if len(values) < 5:
            return 0.5
        return float(sum(values) / len(values))

    # -- lessons -----------------------------------------------------------

    def upsert_lesson(self, lesson: Lesson, embedding: Vector | None = None) -> None:
        with self._lock, self._conn.transaction():
            blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
            existing = self._conn.execute(
                "SELECT id FROM lessons WHERE id = %s", (lesson.id,)
            ).fetchone()
            lesson.updated_at = utcnow()
            if existing is None:
                if blob is None:
                    raise ConfigError("New lessons must be saved with an embedding")
                self._check_dim(embedding)
                self._conn.execute(
                    """INSERT INTO lessons (id, scope, condition, advice, rationale, embedding,
                         state, alpha, beta, times_injected, version, merged_into,
                         source_trajectory_ids, last_injected_at, last_reinforced_at,
                         last_decay_at, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s)""",
                    (
                        lesson.id,
                        lesson.scope,
                        lesson.condition,
                        lesson.advice,
                        lesson.rationale,
                        blob,
                        lesson.state,
                        lesson.alpha,
                        lesson.beta,
                        lesson.times_injected,
                        lesson.version,
                        lesson.merged_into,
                        self._jsonb(lesson.source_trajectory_ids),
                        lesson.last_injected_at,
                        lesson.last_reinforced_at,
                        lesson.last_decay_at,
                        lesson.created_at,
                        lesson.updated_at,
                    ),
                )
            else:
                sets = """scope=%s, condition=%s, advice=%s, rationale=%s, state=%s, alpha=%s,
                          beta=%s, times_injected=%s, version=%s, merged_into=%s,
                          source_trajectory_ids=%s, last_injected_at=%s, last_reinforced_at=%s,
                          last_decay_at=%s, updated_at=%s"""
                params: list[object] = [
                    lesson.scope,
                    lesson.condition,
                    lesson.advice,
                    lesson.rationale,
                    lesson.state,
                    lesson.alpha,
                    lesson.beta,
                    lesson.times_injected,
                    lesson.version,
                    lesson.merged_into,
                    self._jsonb(lesson.source_trajectory_ids),
                    lesson.last_injected_at,
                    lesson.last_reinforced_at,
                    lesson.last_decay_at,
                    lesson.updated_at,
                ]
                if blob is not None:
                    self._check_dim(embedding)
                    sets += ", embedding=%s"
                    params.append(blob)
                params.append(lesson.id)
                self._conn.execute(f"UPDATE lessons SET {sets} WHERE id=%s", params)

    def _check_dim(self, embedding: Vector | None) -> None:
        assert embedding is not None
        dim = int(embedding.shape[-1])
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('embedding_dim', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (str(dim),),
            )
        elif int(row["value"]) != dim:
            raise ConfigError(
                f"This database stores {row['value']}-dim lesson embeddings but the current "
                f"embedder produces {dim}-dim vectors. Use the original embedder, or start a "
                f"new database to re-learn."
            )

    @staticmethod
    def _lesson_from_row(row: dict[str, Any]) -> Lesson:
        return Lesson(
            id=row["id"],
            scope=row["scope"],
            condition=row["condition"],
            advice=row["advice"],
            rationale=row["rationale"],
            state=row["state"],
            alpha=row["alpha"],
            beta=row["beta"],
            times_injected=row["times_injected"],
            version=row["version"],
            merged_into=row["merged_into"],
            source_trajectory_ids=_as_json(row["source_trajectory_ids"]),
            last_injected_at=row["last_injected_at"],
            last_reinforced_at=row["last_reinforced_at"],
            last_decay_at=row["last_decay_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_lesson(self, lesson_id: str) -> Lesson:
        with self._lock:
            row = self._conn.execute("SELECT * FROM lessons WHERE id = %s", (lesson_id,)).fetchone()
        if row is None:
            raise NeuramineRLError(f"Unknown lesson: {lesson_id}")
        return self._lesson_from_row(row)

    def get_lessons(self, scope: str, states: Sequence[LessonState]) -> list[Lesson]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM lessons WHERE scope = %s AND state = ANY(%s) ORDER BY created_at",
                (scope, list(states)),
            ).fetchall()
        return [self._lesson_from_row(r) for r in rows]

    def count_lessons(self, scope: str, states: Sequence[LessonState]) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM lessons WHERE scope = %s AND state = ANY(%s)",
                (scope, list(states)),
            ).fetchone()
        return int(row["n"])

    def search_lessons(
        self, query_vec: Vector, scope: str, states: Sequence[LessonState], k: int
    ) -> list[tuple[Lesson, float]]:
        # No cache: other processes insert lessons concurrently, so re-read
        # the scope's embeddings every time (bounded by max_lessons).
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, state, embedding FROM lessons WHERE scope = %s "
                "AND state IN ('candidate', 'active') AND embedding IS NOT NULL",
                (scope,),
            ).fetchall()
        wanted = set(states)
        ids = [r["id"] for r in rows if r["state"] in wanted]
        if not ids:
            return []
        matrix = np.vstack(
            [
                np.frombuffer(bytes(r["embedding"]), dtype=np.float32)
                for r in rows
                if r["state"] in wanted
            ]
        )
        sims = matrix @ query_vec.reshape(-1)
        order = np.argsort(-sims)[:k]
        return [(self.get_lesson(ids[int(pos)]), float(sims[int(pos)])) for pos in order]

    # -- injections --------------------------------------------------------

    def log_injections(self, injections: Sequence[Injection]) -> None:
        if not injections:
            return
        with self._lock, self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO injections (id, trajectory_id, lesson_id, lesson_version,
                     retrieval_score, rank, credited, credited_alpha, credited_beta, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (
                        i.id,
                        i.trajectory_id,
                        i.lesson_id,
                        i.lesson_version,
                        i.retrieval_score,
                        i.rank,
                        i.credited,
                        i.credited_alpha,
                        i.credited_beta,
                        i.created_at,
                    )
                    for i in injections
                ],
            )

    def injections_for(self, trajectory_id: str) -> list[Injection]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM injections WHERE trajectory_id = %s ORDER BY rank",
                (trajectory_id,),
            ).fetchall()
        return [
            Injection(
                id=r["id"],
                trajectory_id=r["trajectory_id"],
                lesson_id=r["lesson_id"],
                lesson_version=r["lesson_version"],
                retrieval_score=r["retrieval_score"],
                rank=r["rank"],
                credited=bool(r["credited"]),
                credited_alpha=r["credited_alpha"],
                credited_beta=r["credited_beta"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def update_injection(self, injection: Injection) -> None:
        with self._lock, self._conn.transaction():
            self._conn.execute(
                """UPDATE injections SET credited=%s, credited_alpha=%s, credited_beta=%s
                   WHERE id=%s""",
                (
                    injection.credited,
                    injection.credited_alpha,
                    injection.credited_beta,
                    injection.id,
                ),
            )

    # -- audit -------------------------------------------------------------

    def log_event(self, lesson_id: str, event: str, detail: str = "") -> None:
        with self._lock, self._conn.transaction():
            self._conn.execute(
                "INSERT INTO lesson_events (id, lesson_id, event, detail, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (new_id(), lesson_id, event, detail, utcnow()),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
