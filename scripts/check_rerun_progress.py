#!/usr/bin/env python3
"""
Quick progress monitor for the IV re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor rerun_deception_with_iv progress.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def _row_shard(row_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def main() -> int:
    args = parse_args()
    ws = args.workspace_root
    db_path = ws / "outputs" / "checkpoints.sqlite"
    rows_dir = ws / "outputs" / "rows"
    pid_manifest = ws / "logs" / "rerun_deception_with_iv.pids"

    print(f"=== PID Manifest: {pid_manifest} ===")
    if pid_manifest.exists():
        with pid_manifest.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Registered workers: {len(lines)}")
        for line in lines:
            pid, shard, log, script = line.split(" ", 3)
            print(f"  worker={shard} pid={pid} log={log}")
    else:
        print("  No PID manifest found.")

    print(f"\n=== Checkpoint DB: {db_path} ===")
    if not db_path.exists():
        print("  DB not found yet.")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Overall deception attacker stats for this session
    cur = conn.execute(
        """
        SELECT status, COUNT(DISTINCT row_id) as cnt
        FROM checkpoints
        WHERE branch = 'deception' AND stage = 'attacker' AND task_name = ''
        GROUP BY status
        """
    )
    print("Deception attacker checkpoints (all time):")
    for row in cur.fetchall():
        print(f"  {row['status']}: {row['cnt']} rows")

    # Per-shard progress: rows whose *latest* deception:attacker checkpoint is completed
    # and whose row_result JSON contains safety_prompt_strength
    print(f"\n=== Row results with safety_prompt_strength (by shard) ===")
    shard_counts: dict[int, dict[str, int]] = {}
    for path in rows_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rid = payload.get("row_id", "")
        if not rid:
            continue
        shard = _row_shard(rid, args.num_shards)
        if shard not in shard_counts:
            shard_counts[shard] = {"total": 0, "has_iv": 0}
        shard_counts[shard]["total"] += 1
        if "safety_prompt_strength" in payload:
            shard_counts[shard]["has_iv"] += 1

    for shard in sorted(shard_counts):
        total = shard_counts[shard]["total"]
        has_iv = shard_counts[shard]["has_iv"]
        pct = has_iv / total * 100 if total else 0
        print(f"  shard {shard}: {has_iv}/{total} ({pct:.1f}%)")

    total_has_iv = sum(s["has_iv"] for s in shard_counts.values())
    total_rows = sum(s["total"] for s in shard_counts.values())
    print(f"  TOTAL: {total_has_iv}/{total_rows} ({total_has_iv/total_rows*100:.1f}%)")

    # Recent activity in the last 5 minutes
    print("\n=== Recent checkpoint activity (last 5 min) ===")
    since = datetime.now(timezone.utc).isoformat()[:19]
    # SQLite datetime math
    cur = conn.execute(
        """
        SELECT status, COUNT(*) as cnt
        FROM checkpoints
        WHERE branch = 'deception'
          AND datetime(started_at) >= datetime('now', '-5 minutes')
        GROUP BY status
        """
    )
    recent = {row["status"]: row["cnt"] for row in cur.fetchall()}
    if recent:
        for st, cnt in sorted(recent.items()):
            print(f"  {st}: {cnt}")
    else:
        print("  No checkpoint updates in the last 5 minutes.")

    # Running checkpoints
    cur = conn.execute(
        """
        SELECT row_id, stage, task_name, started_at
        FROM checkpoints
        WHERE status = 'running' AND branch = 'deception'
        ORDER BY started_at DESC
        LIMIT 10
        """
    )
    running = cur.fetchall()
    print(f"\n=== Currently running deception checkpoints: {len(running)} ===")
    for row in running:
        print(f"  {row['row_id']} | {row['stage']}:{row['task_name']} | started={row['started_at']}")

    # Recent failures
    cur = conn.execute(
        """
        SELECT row_id, stage, task_name, error_type, finished_at
        FROM checkpoints
        WHERE status = 'failed' AND branch = 'deception'
        ORDER BY datetime(finished_at) DESC
        LIMIT 5
        """
    )
    fails = cur.fetchall()
    print(f"\n=== Recent deception failures (last 5) ===")
    for row in fails:
        print(f"  {row['row_id']} | {row['stage']}:{row['task_name']} | {row['error_type']} | {row['finished_at']}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
