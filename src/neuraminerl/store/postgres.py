"""Postgres store: the multi-process/multi-worker counterpart to SQLite.

Same 16-method Store protocol, same semantics. Differences from SqliteStore:
JSON columns are JSONB, embeddings are BYTEA, and there is deliberately NO
in-memory matrix cache — another process may insert lessons at any time, so
every search re-reads the scope's embeddings (capped at ~200 lessons by the
lifecycle, this stays cheap). Vector search remains numpy brute force;
pgvector is an optimization for far larger stores, not a requirement.

Connections come from a psycopg pool rather than one long-lived connection:
a service running for days will eventually see its connection dropped by a
Postgres restart or a network blip, and the pool reconnects instead of
failing every subsequent call. It is also thread-safe, so concurrent callers
(including AsyncLearner's worker threads) do not serialize behind one lock.

Requires the ``postgres`` extra: ``pip install neuraminerl[postgres]``.
"""

from __future__ import annotations

import json
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
from .base import check_query_dim

_SCHEMA_VERSION = 1
# Arbitrary constant, shared by every opener so they serialize on the same lock.
_SCHEMA_LOCK_ID = 7318451234567890

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
  credited_trials DOUBLE PRECISION NOT NULL DEFAULT 0.0,
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
    """``Learner(store="postgresql://...")`` builds one of these.

    ``max_size`` bounds connections held per process; the default suits one
    Celery worker. Each ``pool.connection()`` block is a transaction: it
    commits on clean exit and rolls back if the body raises.
    """

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4) -> None:
        try:
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ConfigError(
                "PostgresStore requires psycopg: pip install neuraminerl[postgres]"
            ) from exc
        self._jsonb = Jsonb
        self._pool: Any = ConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        with self._pool.connection() as conn:
            # CREATE TABLE IF NOT EXISTS checks the catalog before taking the
            # creation lock, so simultaneous opens (many workers starting at
            # once) race and raise a raw UniqueViolation on the system catalog.
            # This transaction-scoped lock serializes setup; it releases on
            # commit, and everyone after the winner sees a committed schema.
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_ID,))
            conn.execute(_SCHEMA)
            conn.execute(
                "ALTER TABLE lessons ADD COLUMN IF NOT EXISTS "
                "credited_trials DOUBLE PRECISION NOT NULL DEFAULT 0.0"
            )
            conn.execute(
                "UPDATE lessons SET credited_trials = GREATEST(0.0, alpha + beta - 2.0) "
                "WHERE credited_trials = 0.0 AND alpha + beta > 2.0"
            )
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('version', %s) "
                "ON CONFLICT (key) DO NOTHING",
                (str(_SCHEMA_VERSION),),
            )
            # Databases written before embedding width was tracked per scope
            # carry a single global row; hand it to every scope that already
            # has lessons so the guard keeps holding across the upgrade.
            conn.execute(
                """INSERT INTO schema_meta (key, value)
                   SELECT 'embedding_dim:' || l.scope, m.value
                     FROM (SELECT DISTINCT scope FROM lessons WHERE embedding IS NOT NULL) l
                     JOIN schema_meta m ON m.key = 'embedding_dim'
                   ON CONFLICT (key) DO NOTHING"""
            )

    # -- trajectories ------------------------------------------------------

    def save_trajectory(self, trajectory: Trajectory) -> None:
        with self._pool.connection() as conn:
            conn.execute(
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
        with self._pool.connection() as conn:
            row = conn.execute(
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
                step_rows = conn.execute(
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
        with self._pool.connection() as conn, conn.cursor() as cur:
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
        with self._pool.connection() as conn:
            conn.execute(
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
        with self._pool.connection() as conn:
            row = conn.execute(
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
        with self._pool.connection() as conn:
            rows = conn.execute(
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
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self._pool.connection() as conn:
            existing = conn.execute("SELECT id FROM lessons WHERE id = %s", (lesson.id,)).fetchone()
            lesson.updated_at = utcnow()
            if existing is None:
                if blob is None:
                    raise ConfigError("New lessons must be saved with an embedding")
                self._check_dim(conn, embedding, lesson.scope)
                conn.execute(
                    """INSERT INTO lessons (id, scope, condition, advice, rationale, embedding,
                         state, alpha, beta, times_injected, credited_trials, version,
                         merged_into, source_trajectory_ids, last_injected_at,
                         last_reinforced_at, last_decay_at, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s)""",
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
                        lesson.credited_trials,
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
                          beta=%s, times_injected=%s, credited_trials=%s, version=%s,
                          merged_into=%s, source_trajectory_ids=%s, last_injected_at=%s,
                          last_reinforced_at=%s, last_decay_at=%s, updated_at=%s"""
                params: list[object] = [
                    lesson.scope,
                    lesson.condition,
                    lesson.advice,
                    lesson.rationale,
                    lesson.state,
                    lesson.alpha,
                    lesson.beta,
                    lesson.times_injected,
                    lesson.credited_trials,
                    lesson.version,
                    lesson.merged_into,
                    self._jsonb(lesson.source_trajectory_ids),
                    lesson.last_injected_at,
                    lesson.last_reinforced_at,
                    lesson.last_decay_at,
                    lesson.updated_at,
                ]
                if blob is not None:
                    self._check_dim(conn, embedding, lesson.scope)
                    sets += ", embedding=%s"
                    params.append(blob)
                params.append(lesson.id)
                conn.execute(f"UPDATE lessons SET {sets} WHERE id=%s", params)

    def _check_dim(self, conn: Any, embedding: Vector | None, scope: str) -> None:
        """Embedding width is enforced per scope, not per database: a search
        only ever stacks one scope's vectors into a matrix, so unrelated
        agents sharing this database may use different embedders."""
        assert embedding is not None
        dim = int(embedding.shape[-1])
        key = f"embedding_dim:{scope}"
        # Claim-or-read in one statement. A plain SELECT-then-INSERT lets two
        # concurrent first-writers with different widths both pass the guard,
        # after which every search of that scope dies inside np.vstack. The
        # no-op DO UPDATE takes a row lock and RETURNING hands back the
        # committed winner, so the loser sees the other width and raises.
        stored = conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = schema_meta.value RETURNING value",
            (key, str(dim)),
        ).fetchone()["value"]
        if int(stored) != dim:
            raise ConfigError(
                f"Scope '{scope}' stores {stored}-dim lesson embeddings but the current "
                f"embedder produces {dim}-dim vectors. Use the original embedder, or start a "
                f"new scope (or database) to re-learn."
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
            credited_trials=row["credited_trials"],
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
        with self._pool.connection() as conn:
            row = conn.execute("SELECT * FROM lessons WHERE id = %s", (lesson_id,)).fetchone()
        if row is None:
            raise NeuramineRLError(f"Unknown lesson: {lesson_id}")
        return self._lesson_from_row(row)

    def get_lessons(self, scope: str, states: Sequence[LessonState]) -> list[Lesson]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM lessons WHERE scope = %s AND state = ANY(%s) ORDER BY created_at",
                (scope, list(states)),
            ).fetchall()
        return [self._lesson_from_row(r) for r in rows]

    def count_lessons(self, scope: str, states: Sequence[LessonState]) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM lessons WHERE scope = %s AND state = ANY(%s)",
                (scope, list(states)),
            ).fetchone()
        return int(row["n"])

    def search_lessons(
        self, query_vec: Vector, scope: str, states: Sequence[LessonState], k: int
    ) -> list[tuple[Lesson, float]]:
        # Full rows in one query: no cache (other processes insert lessons
        # concurrently) and no per-hit re-fetch. Bounded by max_lessons.
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM lessons WHERE scope = %s AND state = ANY(%s) "
                "AND embedding IS NOT NULL",
                (scope, list(states)),
            ).fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(bytes(r["embedding"]), dtype=np.float32) for r in rows])
        check_query_dim(matrix, query_vec, scope)
        sims = matrix @ query_vec.reshape(-1)
        order = np.argsort(-sims)[:k]
        return [(self._lesson_from_row(rows[int(pos)]), float(sims[int(pos)])) for pos in order]

    # -- injections --------------------------------------------------------

    def log_injections(self, injections: Sequence[Injection]) -> None:
        if not injections:
            return
        with self._pool.connection() as conn, conn.cursor() as cur:
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
        with self._pool.connection() as conn:
            rows = conn.execute(
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
        with self._pool.connection() as conn:
            conn.execute(
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
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO lesson_events (id, lesson_id, event, detail, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (new_id(), lesson_id, event, detail, utcnow()),
            )

    def close(self) -> None:
        self._pool.close()
