#!/usr/bin/env python3
"""
Quick progress monitor for attacker_nondeceptive reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path


_REQUIRED_FIELDS = (
    "neutral_control_option_id",
    "attacker_nondeceptive_advocated_option_id",
    "attacker_nondeceptive_advocated_option_text",
    "attacker_nondeceptive_advocated_option_source",
    "attacker_nondeceptive_parse_status",
    "attacker_nondeceptive_option_id",
    "attacker_nondeceptive_attacker_response_hash",
    "attacker_nondeceptive_defender_response_hash",
    "attacker_nondeceptive_target_binary",
    "attacker_nondeceptive_shift_to_target",
)

_OUTCOME_FIELDS = (
    "deception_success_unconditional",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor attacker_nondeceptive rerun progress.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--num-shards", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    db_path = run_dir / "checkpoints.sqlite"
    rows_dir = run_dir / "rows"
    workspace_root = run_dir.parents[2] if run_dir.parent.name == "runs" else run_dir.parents[1]
    pid_manifest = workspace_root / "logs" / "rerun_attacker_nondeceptive.pids"

    print(f"=== PID Manifest: {pid_manifest} ===")
    if pid_manifest.exists():
        lines = [line.strip() for line in pid_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Registered workers: {len(lines)}")
        for line in lines:
            pid, shard, log, script = line.split(" ", 3)
            print(f"  worker={shard} pid={pid} alive={_pid_alive(int(pid))} log={log}")
    else:
        print("  No PID manifest found.")

    print("\n=== Active rerun_attacker_nondeceptive.py processes ===")
    active_processes = _active_rerun_processes()
    if not active_processes:
        print("  None.")
    else:
        for process in active_processes:
            print(
                f"  pid={process['pid']} ppid={process['ppid']} stat={process['stat']} "
                f"elapsed={process['elapsed']} cmd={process['command']}"
            )

    print(f"\n=== Checkpoint DB: {db_path} ===")
    if not db_path.exists():
        print("  DB not found yet.")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for branch, stage, task_name, label in (
        ("attacker_nondeceptive", "attacker", "", "A_i attacker"),
        ("attacker_nondeceptive", "defender", "", "A_i defender"),
        ("meta", "task_consistency", "task_consistency", "task consistency"),
    ):
        print(f"{label} checkpoints (latest attempt by row):")
        for status, count in _latest_status_counts(conn, branch, stage, task_name).items():
            print(f"  {status}: {count}")

    print("\n=== Row JSON branch completeness by shard ===")
    shard_counts: dict[int, dict[str, int]] = {}
    missing_outcome_rows = 0
    for path in rows_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {"row_id": path.stem}
        row_id = payload.get("row_id") or _unsafe_name(path.stem)
        shard = _row_shard(str(row_id), args.num_shards)
        counts = shard_counts.setdefault(shard, {"total": 0, "ai_complete": 0})
        counts["total"] += 1
        if _has_required_fields(payload):
            counts["ai_complete"] += 1
        if not _has_fields(payload, _OUTCOME_FIELDS):
            missing_outcome_rows += 1
    for shard in sorted(shard_counts):
        total = shard_counts[shard]["total"]
        ai_complete = shard_counts[shard]["ai_complete"]
        pct = ai_complete / total * 100 if total else 0
        print(f"  shard {shard}: {ai_complete}/{total} ({pct:.1f}%)")
    total_rows = sum(item["total"] for item in shard_counts.values())
    total_complete = sum(item["ai_complete"] for item in shard_counts.values())
    total_pct = total_complete / total_rows * 100 if total_rows else 0
    print(f"  TOTAL: {total_complete}/{total_rows} ({total_pct:.1f}%)")
    if missing_outcome_rows:
        print(f"  Note: {missing_outcome_rows} rows are missing optional outcome fields.")

    print("\n=== Currently running relevant checkpoints ===")
    running = conn.execute(
        """
        SELECT row_id, branch, stage, task_name, started_at
        FROM checkpoints
        WHERE status = 'running'
          AND (
            (branch = 'attacker_nondeceptive' AND stage IN ('attacker', 'defender'))
            OR (branch = 'meta' AND stage = 'task_consistency' AND task_name = 'task_consistency')
          )
        ORDER BY started_at DESC
        LIMIT 10
        """
    ).fetchall()
    if not running:
        print("  None.")
    else:
        for row in running:
            print(
                f"  {row['row_id']} | {row['branch']}:{row['stage']}:{row['task_name']} "
                f"| started={row['started_at']}"
            )

    print("\n=== Recent relevant failures (last 10 latest rows) ===")
    failures = conn.execute(
        """
        WITH ranked AS (
          SELECT row_id, branch, stage, task_name, status, error_type, finished_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY row_id, branch, stage, task_name
                   ORDER BY attempt DESC, started_at DESC
                 ) AS rn
          FROM checkpoints
          WHERE
            (branch = 'attacker_nondeceptive' AND stage IN ('attacker', 'defender'))
            OR (branch = 'meta' AND stage = 'task_consistency' AND task_name = 'task_consistency')
        )
        SELECT row_id, branch, stage, task_name, error_type, finished_at
        FROM ranked
        WHERE rn = 1 AND status = 'failed'
        ORDER BY finished_at DESC
        LIMIT 10
        """
    ).fetchall()
    if not failures:
        print("  None.")
    else:
        for row in failures:
            print(
                f"  {row['row_id']} | {row['branch']}:{row['stage']}:{row['task_name']} "
                f"| {row['error_type']} | {row['finished_at']}"
            )

    conn.close()
    return 0


def _latest_status_counts(
    conn: sqlite3.Connection,
    branch: str,
    stage: str,
    task_name: str,
) -> dict[str, int]:
    rows = conn.execute(
        """
        WITH ranked AS (
          SELECT row_id, status,
                 ROW_NUMBER() OVER (
                   PARTITION BY row_id, branch, stage, task_name
                   ORDER BY attempt DESC, started_at DESC
                 ) AS rn
          FROM checkpoints
          WHERE branch = ? AND stage = ? AND task_name = ?
        )
        SELECT status, COUNT(*) AS cnt
        FROM ranked
        WHERE rn = 1
        GROUP BY status
        ORDER BY status
        """,
        (branch, stage, task_name),
    ).fetchall()
    return {str(row["status"]): int(row["cnt"]) for row in rows}


def _row_shard(row_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def _has_required_fields(payload: dict[str, object]) -> bool:
    return _has_fields(payload, _REQUIRED_FIELDS)


def _has_fields(payload: dict[str, object], fields: tuple[str, ...]) -> bool:
    for field in fields:
        value = payload.get(field)
        if value is None or value == "":
            return False
    return True


def _unsafe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _active_rerun_processes() -> list[dict[str, str]]:
    commands = (
        ("ps", "-axww", "-o", "pid=,ppid=,stat=,etime=,command="),
        ("ps", "-eww", "-o", "pid=,ppid=,stat=,etime=,command="),
    )
    output = ""
    for command in commands:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        output = completed.stdout
        break
    if not output:
        return []

    processes: list[dict[str, str]] = []
    for line in output.splitlines():
        if "scripts/rerun_attacker_nondeceptive.py" not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, elapsed, command = parts
        processes.append(
            {
                "pid": pid,
                "ppid": ppid,
                "stat": stat,
                "elapsed": elapsed,
                "command": command,
            }
        )
    return processes


if __name__ == "__main__":
    raise SystemExit(main())
