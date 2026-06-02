from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class CheckpointRecord:
    row_id: str
    branch: str
    stage: str
    task_name: str
    attempt: int
    status: str
    request_hash: str
    response_path: str | None
    error_type: str | None
    error_message: str | None
    started_at: str
    finished_at: str | None


class CheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("PRAGMA busy_timeout = 60000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    row_id TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    task_name TEXT NOT NULL DEFAULT '',
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_path TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    PRIMARY KEY (row_id, branch, stage, task_name, attempt)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_checkpoints_status_started
                ON checkpoints(status, started_at)
                """
            )

    def has_completed(self, row_id: str, branch: str, stage: str, task_name: str = "") -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM checkpoints
                WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ? AND status = 'completed'
                ORDER BY attempt DESC
                LIMIT 1
                """,
                (row_id, branch, stage, task_name),
            ).fetchone()
        return row is not None

    def latest_response_path(
        self, row_id: str, branch: str, stage: str, task_name: str = ""
    ) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_path
                FROM checkpoints
                WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ? AND status = 'completed'
                ORDER BY attempt DESC
                LIMIT 1
                """,
                (row_id, branch, stage, task_name),
            ).fetchone()
        return None if row is None else row["response_path"]

    def next_attempt(self, row_id: str, branch: str, stage: str, task_name: str = "") -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS max_attempt
                FROM checkpoints
                WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ?
                """,
                (row_id, branch, stage, task_name),
            ).fetchone()
        return int(row["max_attempt"]) + 1

    def start(
        self,
        row_id: str,
        branch: str,
        stage: str,
        task_name: str,
        attempt: int,
        request_hash: str,
        started_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    row_id, branch, stage, task_name, attempt, status, request_hash, response_path,
                    error_type, error_message, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, NULL, NULL, NULL, ?, NULL)
                """,
                (row_id, branch, stage, task_name, attempt, request_hash, started_at),
            )

    def complete(
        self,
        row_id: str,
        branch: str,
        stage: str,
        task_name: str,
        attempt: int,
        response_path: str,
        finished_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE checkpoints
                SET status = 'completed', response_path = ?, finished_at = ?
                WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ? AND attempt = ?
                """,
                (response_path, finished_at, row_id, branch, stage, task_name, attempt),
            )

    def fail(
        self,
        row_id: str,
        branch: str,
        stage: str,
        task_name: str,
        attempt: int,
        response_path: str | None,
        error_type: str,
        error_message: str,
        finished_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE checkpoints
                SET status = 'failed',
                    response_path = ?,
                    error_type = ?,
                    error_message = ?,
                    finished_at = ?
                WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ? AND attempt = ?
                """,
                (
                    response_path,
                    error_type,
                    error_message,
                    finished_at,
                    row_id,
                    branch,
                    stage,
                    task_name,
                    attempt,
                ),
            )

    def row_ids_with_status(self, status: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT row_id FROM checkpoints WHERE status = ?",
                (status,),
            ).fetchall()
        return {str(row["row_id"]) for row in rows}

    def row_ids_with_latest_status(self, status: str) -> set[str]:
        query = (
            "WITH ranked AS ("
            " SELECT row_id, status,"
            " ROW_NUMBER() OVER ("
            "   PARTITION BY row_id, branch, stage, task_name"
            "   ORDER BY attempt DESC, started_at DESC"
            " ) AS rn"
            " FROM checkpoints"
            ") "
            "SELECT DISTINCT row_id FROM ranked WHERE rn = 1 AND status = ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, (status,)).fetchall()
        return {str(row["row_id"]) for row in rows}

    def row_ids_with_step_status(
        self,
        *,
        status: str,
        steps: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...],
    ) -> set[str]:
        if not steps:
            return set()
        clauses: list[str] = []
        params: list[str] = [status]
        for branch, stage, task_name in steps:
            clauses.append("(branch = ? AND stage = ? AND task_name = ?)")
            params.extend([branch, stage, task_name])
        query = (
            "SELECT DISTINCT row_id FROM checkpoints "
            "WHERE status = ? AND (" + " OR ".join(clauses) + ")"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return {str(row["row_id"]) for row in rows}

    def row_ids_with_latest_step_status(
        self,
        *,
        status: str,
        steps: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...],
    ) -> set[str]:
        if not steps:
            return set()
        clauses: list[str] = []
        step_params: list[str] = []
        for branch, stage, task_name in steps:
            clauses.append("(branch = ? AND stage = ? AND task_name = ?)")
            step_params.extend([branch, stage, task_name])
        query = (
            "WITH ranked AS ("
            " SELECT row_id, branch, stage, task_name, status,"
            " ROW_NUMBER() OVER ("
            "   PARTITION BY row_id, branch, stage, task_name"
            "   ORDER BY attempt DESC, started_at DESC"
            " ) AS rn"
            " FROM checkpoints"
            " WHERE " + " OR ".join(clauses) +
            ") "
            "SELECT DISTINCT row_id FROM ranked WHERE rn = 1 AND status = ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, [*step_params, status]).fetchall()
        return {str(row["row_id"]) for row in rows}

    def mark_stale_running_failed(self, older_than_minutes: int = 180) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        cutoff_text = cutoff.isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT row_id, branch, stage, task_name, attempt
                FROM checkpoints
                WHERE status = 'running' AND started_at < ?
                """,
                (cutoff_text,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    """
                    UPDATE checkpoints
                    SET status = 'failed',
                        error_type = 'StaleRunningCheckpoint',
                        error_message = ?,
                        finished_at = ?
                    WHERE row_id = ? AND branch = ? AND stage = ? AND task_name = ? AND attempt = ?
                    """,
                    (
                        f"Marked stale after {older_than_minutes} minutes without completion.",
                        datetime.now(timezone.utc).isoformat(),
                        row["row_id"],
                        row["branch"],
                        row["stage"],
                        row["task_name"],
                        row["attempt"],
                    ),
                )
        return len(rows)
