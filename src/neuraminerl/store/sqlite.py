"""Default store: single-file SQLite in WAL mode.

Vector search is numpy brute force over an in-memory matrix of non-retired
lessons per scope (capped at ~200 by the lifecycle, so a normalized
``matrix @ query`` is microseconds). Multi-process writers are out of scope;
use one Learner per process.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

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
  metadata TEXT NOT NULL DEFAULT '{}'
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
  metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_steps_traj ON steps(trajectory_id, idx);

CREATE TABLE IF NOT EXISTS outcomes (
  id TEXT PRIMARY KEY,
  trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
  status TEXT NOT NULL,
  score REAL,
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
  embedding BLOB,
  state TEXT NOT NULL DEFAULT 'candidate',
  alpha REAL NOT NULL DEFAULT 1.0,
  beta REAL NOT NULL DEFAULT 1.0,
  times_injected INTEGER NOT NULL DEFAULT 0,
  version INTEGER NOT NULL DEFAULT 1,
  merged_into TEXT,
  source_trajectory_ids TEXT NOT NULL DEFAULT '[]',
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
  retrieval_score REAL NOT NULL,
  rank INTEGER NOT NULL,
  credited INTEGER NOT NULL DEFAULT 0,
  credited_alpha REAL NOT NULL DEFAULT 0.0,
  credited_beta REAL NOT NULL DEFAULT 0.0,
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


class SqliteStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._conn.commit()
        # scope -> (lesson_ids, states, matrix) over non-retired lessons
        self._matrix_cache: dict[str, tuple[list[str], list[str], Vector]] = {}

    # -- trajectories ------------------------------------------------------

    def save_trajectory(self, trajectory: Trajectory) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO trajectories
                     (id, scope, task, status, started_at, ended_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     status=excluded.status, ended_at=excluded.ended_at,
                     metadata=excluded.metadata""",
                (
                    trajectory.id,
                    trajectory.scope,
                    trajectory.task,
                    trajectory.status,
                    trajectory.started_at,
                    trajectory.ended_at,
                    json.dumps(trajectory.metadata),
                ),
            )
            self._conn.commit()

    def get_trajectory(self, trajectory_id: str, *, with_steps: bool = True) -> Trajectory:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trajectories WHERE id = ?", (trajectory_id,)
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
                metadata=json.loads(row["metadata"]),
            )
            if with_steps:
                step_rows = self._conn.execute(
                    "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY idx", (trajectory_id,)
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
                        metadata=json.loads(s["metadata"]),
                    )
                    for s in step_rows
                ]
            return trajectory

    def add_steps(self, steps: Sequence[Step]) -> None:
        if not steps:
            return
        with self._lock:
            self._conn.executemany(
                """INSERT INTO steps
                     (id, trajectory_id, idx, kind, content, error, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        s.id,
                        s.trajectory_id,
                        s.index,
                        s.kind,
                        s.content,
                        s.error,
                        s.created_at,
                        json.dumps(s.metadata),
                    )
                    for s in steps
                ],
            )
            self._conn.commit()

    # -- outcomes ----------------------------------------------------------

    def save_outcome(self, outcome: Outcome) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO outcomes
                     (id, trajectory_id, status, score, source, detail, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
            self._conn.commit()

    def latest_outcome(self, trajectory_id: str) -> Outcome | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM outcomes WHERE trajectory_id = ?
                   ORDER BY created_at DESC, rowid DESC LIMIT 1""",
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._outcome_from_row(row)

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row) -> Outcome:
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
                """SELECT o.status, o.score FROM outcomes o
                   JOIN trajectories t ON t.id = o.trajectory_id
                   WHERE t.scope = ? AND o.status != 'unknown'
                     AND o.id IN (
                       SELECT id FROM outcomes o2 WHERE o2.trajectory_id = o.trajectory_id
                       ORDER BY o2.created_at DESC, o2.rowid DESC LIMIT 1
                     )
                   ORDER BY o.created_at DESC LIMIT ?""",
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
        with self._lock:
            blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
            existing = self._conn.execute(
                "SELECT id FROM lessons WHERE id = ?", (lesson.id,)
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
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        json.dumps(lesson.source_trajectory_ids),
                        lesson.last_injected_at,
                        lesson.last_reinforced_at,
                        lesson.last_decay_at,
                        lesson.created_at,
                        lesson.updated_at,
                    ),
                )
            else:
                sets = """scope=?, condition=?, advice=?, rationale=?, state=?, alpha=?, beta=?,
                          times_injected=?, version=?, merged_into=?, source_trajectory_ids=?,
                          last_injected_at=?, last_reinforced_at=?, last_decay_at=?, updated_at=?"""
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
                    json.dumps(lesson.source_trajectory_ids),
                    lesson.last_injected_at,
                    lesson.last_reinforced_at,
                    lesson.last_decay_at,
                    lesson.updated_at,
                ]
                if blob is not None:
                    self._check_dim(embedding)
                    sets += ", embedding=?"
                    params.append(blob)
                params.append(lesson.id)
                self._conn.execute(f"UPDATE lessons SET {sets} WHERE id=?", params)
            self._conn.commit()
            self._matrix_cache.pop(lesson.scope, None)

    def _check_dim(self, embedding: Vector | None) -> None:
        assert embedding is not None
        dim = int(embedding.shape[-1])
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'embedding_dim'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('embedding_dim', ?)", (str(dim),)
            )
        elif int(row["value"]) != dim:
            raise ConfigError(
                f"This database stores {row['value']}-dim lesson embeddings but the current "
                f"embedder produces {dim}-dim vectors. Use the original embedder, or start a "
                f"new database (delete the .neuraminerl directory) to re-learn."
            )

    @staticmethod
    def _lesson_from_row(row: sqlite3.Row) -> Lesson:
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
            source_trajectory_ids=json.loads(row["source_trajectory_ids"]),
            last_injected_at=row["last_injected_at"],
            last_reinforced_at=row["last_reinforced_at"],
            last_decay_at=row["last_decay_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_lesson(self, lesson_id: str) -> Lesson:
        with self._lock:
            row = self._conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        if row is None:
            raise NeuramineRLError(f"Unknown lesson: {lesson_id}")
        return self._lesson_from_row(row)

    def get_lessons(self, scope: str, states: Sequence[LessonState]) -> list[Lesson]:
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM lessons WHERE scope = ? AND state IN ({placeholders}) "
                "ORDER BY created_at",
                (scope, *states),
            ).fetchall()
        return [self._lesson_from_row(r) for r in rows]

    def count_lessons(self, scope: str, states: Sequence[LessonState]) -> int:
        placeholders = ",".join("?" for _ in states)
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM lessons WHERE scope = ? AND state IN ({placeholders})",
                (scope, *states),
            ).fetchone()
        return int(row["n"])

    def search_lessons(
        self, query_vec: Vector, scope: str, states: Sequence[LessonState], k: int
    ) -> list[tuple[Lesson, float]]:
        with self._lock:
            cached = self._matrix_cache.get(scope)
            if cached is None:
                rows = self._conn.execute(
                    "SELECT id, state, embedding FROM lessons WHERE scope = ? "
                    "AND state IN ('candidate', 'active') AND embedding IS NOT NULL",
                    (scope,),
                ).fetchall()
                ids = [r["id"] for r in rows]
                row_states = [r["state"] for r in rows]
                if rows:
                    matrix = np.vstack(
                        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                    )
                else:
                    matrix = np.zeros((0, query_vec.shape[-1]), dtype=np.float32)
                cached = (ids, row_states, matrix)
                self._matrix_cache[scope] = cached
        ids, row_states, matrix = cached
        wanted = set(states)
        mask = [i for i, s in enumerate(row_states) if s in wanted]
        if not mask:
            return []
        sims = matrix[mask] @ query_vec.reshape(-1)
        order = np.argsort(-sims)[:k]
        results: list[tuple[Lesson, float]] = []
        for pos in order:
            lesson_id = ids[mask[int(pos)]]
            results.append((self.get_lesson(lesson_id), float(sims[int(pos)])))
        return results

    # -- injections --------------------------------------------------------

    def log_injections(self, injections: Sequence[Injection]) -> None:
        if not injections:
            return
        with self._lock:
            self._conn.executemany(
                """INSERT INTO injections (id, trajectory_id, lesson_id, lesson_version,
                     retrieval_score, rank, credited, credited_alpha, credited_beta, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        i.id,
                        i.trajectory_id,
                        i.lesson_id,
                        i.lesson_version,
                        i.retrieval_score,
                        i.rank,
                        int(i.credited),
                        i.credited_alpha,
                        i.credited_beta,
                        i.created_at,
                    )
                    for i in injections
                ],
            )
            self._conn.commit()

    def injections_for(self, trajectory_id: str) -> list[Injection]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM injections WHERE trajectory_id = ? ORDER BY rank", (trajectory_id,)
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
        with self._lock:
            self._conn.execute(
                """UPDATE injections SET credited=?, credited_alpha=?, credited_beta=?
                   WHERE id=?""",
                (
                    int(injection.credited),
                    injection.credited_alpha,
                    injection.credited_beta,
                    injection.id,
                ),
            )
            self._conn.commit()

    # -- audit -------------------------------------------------------------

    def log_event(self, lesson_id: str, event: str, detail: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO lesson_events (id, lesson_id, event, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (new_id(), lesson_id, event, detail, utcnow()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
