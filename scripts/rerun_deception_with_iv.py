#!/usr/bin/env python3
"""
Incremental re-run of the deception branch with the safety-prompt IV.

This script forces a re-run of:
  - deception:attacker
  - deception:refusal_engagement
  - deception:defender
  - deception:tactic_tagging
  - meta:task_consistency

Baseline and control branches are read from cache.  Supports multi-worker
sharding, checkpoint-based filtering (--failed-only, --incomplete-only), and
dry-run mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from deception_benchmark.checkpoints import CheckpointStore
from deception_benchmark.config import (
    ANALYSIS_ROWS_JSONL,
    CHECKPOINT_DB_PATH,
    DATA_DIR,
    DERIVED_DIR,
    OUTPUT_DIR,
    ROW_RESULTS_DIR,
)
from deception_benchmark.runner import ExperimentRunner, load_joined_rows, load_scenarios

_FORCE_STEPS = {
    "deception:attacker",
    "deception:refusal_engagement",
    "deception:refusal_engagement:refusal_engagement",
    "deception:defender",
    "deception:tactic_tagging",
    "deception:tactic_tagging:tactic_tagging",
    "meta:task_consistency",
    "meta:task_consistency:task_consistency",
}

_FAILED_STEPS = (
    ("deception", "attacker", ""),
    ("deception", "attacker", "standard_generation"),
    ("deception", "refusal_engagement", "refusal_engagement"),
    ("deception", "defender", ""),
    ("deception", "tactic_tagging", "tactic_tagging"),
    ("meta", "task_consistency", "task_consistency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run deception branch with safety-prompt IV."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N joined rows.")
    parser.add_argument("--row-id", action="append", default=[], help="Specific row_id values to run.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock responses instead of API calls.")
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only rerun rows with at least one failed checkpointed step.",
    )
    parser.add_argument(
        "--incomplete-only",
        action="store_true",
        help="Only run rows without a completed row result JSON.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic row shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this worker.")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up analysis_rows.jsonl.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Project root directory. Defaults to the repository root relative to this script.",
    )
    parser.add_argument(
        "--joined-rows",
        type=Path,
        default=None,
        help="Path to joined_rows.csv. Defaults to <workspace-root>/data/derived/joined_rows.csv.",
    )
    parser.add_argument(
        "--scenario-registry",
        type=Path,
        default=None,
        help="Path to scenario_registry.jsonl. Defaults to <workspace-root>/data/derived/scenario_registry.jsonl.",
    )
    return parser.parse_args()


def _backup_analysis_rows(analysis_rows_jsonl: Path, output_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = output_dir / f"analysis_rows_pre_iv_{ts}.jsonl"
    if analysis_rows_jsonl.exists():
        shutil.copy2(analysis_rows_jsonl, backup_path)
        print(f"Backed up {analysis_rows_jsonl} -> {backup_path}")
    return backup_path


def _row_shard(row_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def main() -> int:
    args = parse_args()
    ws = args.workspace_root
    joined_rows_path = args.joined_rows or ws / "data" / "derived" / "joined_rows.csv"
    scenario_registry_path = args.scenario_registry or ws / "data" / "derived" / "scenario_registry.jsonl"
    analysis_rows_jsonl = ws / "outputs" / "analysis_rows.jsonl"
    output_dir = ws / "outputs"
    row_results_dir = ws / "outputs" / "rows"
    checkpoint_db_path = ws / "outputs" / "checkpoints.sqlite"

    if not joined_rows_path.exists():
        raise FileNotFoundError(f"Joined rows not found: {joined_rows_path}")
    if not scenario_registry_path.exists():
        raise FileNotFoundError(f"Scenario registry not found: {scenario_registry_path}")

    joined_rows = load_joined_rows(joined_rows_path)
    checkpoint_store = CheckpointStore(checkpoint_db_path)

    if args.row_id:
        wanted = set(args.row_id)
        joined_rows = [row for row in joined_rows if row["row_id"] in wanted]

    if args.failed_only:
        failed_ids = checkpoint_store.row_ids_with_latest_step_status(
            status="failed",
            steps=_FAILED_STEPS,
        )
        joined_rows = [row for row in joined_rows if row["row_id"] in failed_ids]

    if args.incomplete_only:
        completed_row_ids = set()
        for path in row_results_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("row_id"):
                completed_row_ids.add(payload["row_id"])
        joined_rows = [row for row in joined_rows if row["row_id"] not in completed_row_ids]

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    if args.num_shards > 1:
        joined_rows = [
            row for row in joined_rows if _row_shard(row["row_id"], args.num_shards) == args.shard_index
        ]

    if args.limit is not None:
        joined_rows = joined_rows[: args.limit]

    if not joined_rows:
        print("No rows to process.")
        return 0

    scenarios = load_scenarios(scenario_registry_path)

    if not args.no_backup and args.shard_index == 0:
        _backup_analysis_rows(analysis_rows_jsonl, output_dir)

    print(
        f"Running {len(joined_rows)} rows "
        f"(shard={args.shard_index}/{args.num_shards}, dry_run={args.dry_run})."
    )
    print(f"Forced steps: {_FORCE_STEPS}")

    runner = ExperimentRunner(
        dry_run=args.dry_run,
        force_steps=_FORCE_STEPS,
        iv_design="legacy_safety_prompt",
    )
    runner.run_rows(joined_rows, scenarios, consolidate=args.num_shards == 1)

    # Consolidate to workspace-specific output paths
    from deception_benchmark.runner import ANALYSIS_ROWS_JSONL as _DEFAULT_ANALYSIS_ROWS_JSONL
    from deception_benchmark.config import ROW_RESULTS_DIR as _DEFAULT_ROW_RESULTS_DIR
    # The runner uses config-level constants; for cloud portability we rely on the
    # workspace-root being the working directory or the config constants being
    # overridden via env.  Here we just emit a reminder.
    print(
        f"completed_rows={len(joined_rows)} dry_run={args.dry_run} "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    print(f"Reminder: consolidate results from {row_results_dir} -> {analysis_rows_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
