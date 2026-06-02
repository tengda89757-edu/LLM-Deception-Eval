#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deception benchmark experiment.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N joined rows.")
    parser.add_argument("--row-id", action="append", default=[], help="Specific row_id values to run.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock responses instead of API calls.")
    parser.add_argument("--run-id", default="", help="Write outputs under outputs/runs/<run-id>.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit run output directory.")
    parser.add_argument(
        "--iv-design",
        choices=("gate_only", "legacy_safety_prompt", "safety_prompt"),
        default="gate_only",
        help="Instrument design. gate_only is the TACL revision default.",
    )
    parser.add_argument("--failed-only", action="store_true", help="Only rerun rows with at least one failed checkpointed step.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Alias for --incomplete-only: skip row JSONs already completed in this run directory.",
    )
    parser.add_argument("--incomplete-only", action="store_true", help="Only run rows without a completed row result JSON.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic row shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this worker.")
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Merge row JSONs into analysis_rows after this worker finishes. Use only for single-worker runs or after all shards finish.",
    )
    parser.add_argument(
        "--recover-stale-minutes",
        type=int,
        default=None,
        help="Mark running checkpoints older than this many minutes as failed before selecting rows.",
    )
    parser.add_argument(
        "--force-step",
        action="append",
        default=[],
        help=(
            "Bypass completed checkpoints for a step key such as branch:stage or "
            "branch:stage:task_name. May be repeated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id:
        os.environ["DECEPTION_RUN_ID"] = args.run_id
    if args.output_dir is not None:
        os.environ["DECEPTION_OUTPUT_DIR"] = str(args.output_dir)
    os.environ["DECEPTION_IV_DESIGN"] = args.iv_design

    from deception_benchmark.checkpoints import CheckpointStore
    from deception_benchmark.config import (
        ANALYSIS_ROWS_JSONL,
        CHECKPOINT_DB_PATH,
        JOINED_ROWS_CSV,
        OUTPUT_DIR,
        ROW_RESULTS_DIR,
        SCENARIO_REGISTRY_JSONL,
    )
    from deception_benchmark.runner import ExperimentRunner, load_joined_rows, load_scenarios

    joined_rows = load_joined_rows(JOINED_ROWS_CSV)
    checkpoint_store = CheckpointStore(CHECKPOINT_DB_PATH)
    if args.recover_stale_minutes is not None:
        recovered = checkpoint_store.mark_stale_running_failed(args.recover_stale_minutes)
        print(f"stale_running_recovered={recovered}")
    if args.row_id:
        wanted = set(args.row_id)
        joined_rows = [row for row in joined_rows if row["row_id"] in wanted]
    if args.failed_only:
        failed_ids = checkpoint_store.row_ids_with_latest_status("failed")
        joined_rows = [row for row in joined_rows if row["row_id"] in failed_ids]
    if args.incomplete_only or args.resume:
        completed_row_ids = set()
        for path in ROW_RESULTS_DIR.glob("*.json"):
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
            row
            for row in joined_rows
            if _row_shard(row["row_id"], args.num_shards) == args.shard_index
        ]
    if args.limit is not None:
        joined_rows = joined_rows[: args.limit]
    scenarios = load_scenarios(SCENARIO_REGISTRY_JSONL)
    runner = ExperimentRunner(
        dry_run=args.dry_run,
        force_steps=set(args.force_step),
        iv_design=args.iv_design,
    )
    consolidate = bool(args.consolidate or args.num_shards == 1)
    runner.run_rows(joined_rows, scenarios, consolidate=consolidate)
    print(
        f"completed_rows={len(joined_rows)} dry_run={args.dry_run} "
        f"shard={args.shard_index}/{args.num_shards} output_dir={OUTPUT_DIR} "
        f"analysis_rows={ANALYSIS_ROWS_JSONL if consolidate else 'not_consolidated'}"
    )


def _row_shard(row_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


if __name__ == "__main__":
    main()
