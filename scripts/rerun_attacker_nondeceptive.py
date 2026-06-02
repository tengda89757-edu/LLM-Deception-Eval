#!/usr/bin/env python3
"""
Incremental re-run of the attacker-authored non-deceptive control branch.

This script forces a re-run of:
  - attacker_nondeceptive:attacker
  - attacker_nondeceptive:defender
  - meta:task_consistency

Baseline, neutral-control, and deception branches are read from cache. Supports
multi-worker sharding, branch-scoped failed/incomplete selection, and dry-run
mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


_FORCE_STEPS = {
    "attacker_nondeceptive:attacker",
    "attacker_nondeceptive:defender",
    "meta:task_consistency",
    "meta:task_consistency:task_consistency",
}

_FAILED_STEPS = (
    ("attacker_nondeceptive", "attacker", ""),
    ("attacker_nondeceptive", "defender", ""),
    ("meta", "task_consistency", "task_consistency"),
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the attacker-authored non-deceptive control branch."
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N joined rows.")
    parser.add_argument("--row-id", action="append", default=[], help="Specific row_id values to run.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock responses instead of API calls.")
    parser.add_argument("--run-id", default="", help="Write outputs under outputs/runs/<run-id>.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Explicit run output directory.")
    parser.add_argument(
        "--iv-design",
        choices=("gate_only", "legacy_safety_prompt", "safety_prompt"),
        default="gate_only",
        help="Instrument design to use while reloading cached non-A_i branches.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only rerun rows whose latest A_i-related checkpoint is failed.",
    )
    parser.add_argument(
        "--incomplete-only",
        action="store_true",
        help="Only rerun rows whose row JSON is missing A_i/schema-v2 fields in this run directory.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic row shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this worker.")
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Merge row JSONs into analysis_rows after this worker finishes. Use only after all shards finish.",
    )
    parser.add_argument(
        "--recover-stale-minutes",
        type=int,
        default=None,
        help="Mark running checkpoints older than this many minutes as failed before selecting rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_id:
        os.environ["DECEPTION_RUN_ID"] = args.run_id
    if args.output_dir is not None:
        os.environ["DECEPTION_OUTPUT_DIR"] = str(args.output_dir.expanduser().resolve())
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
        _log(f"stale_running_recovered={recovered}")

    if args.row_id:
        wanted = set(args.row_id)
        joined_rows = [row for row in joined_rows if row["row_id"] in wanted]

    selected_ids = {row["row_id"] for row in joined_rows}
    _log(f"candidate_rows={len(selected_ids)} output_dir={OUTPUT_DIR}")

    if args.failed_only:
        failed_ids = checkpoint_store.row_ids_with_latest_step_status(
            status="failed",
            steps=_FAILED_STEPS,
        )
        _log(f"latest_failed_rows={len(failed_ids)}")
        selected_ids &= failed_ids

    if args.incomplete_only:
        incomplete_ids = _row_ids_with_incomplete_branch(joined_rows, ROW_RESULTS_DIR)
        _log(f"incomplete_rows={len(incomplete_ids)}")
        selected_ids &= incomplete_ids

    joined_rows = [row for row in joined_rows if row["row_id"] in selected_ids]

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

    if not joined_rows:
        _log("No rows to process.")
        return 0

    scenarios = load_scenarios(SCENARIO_REGISTRY_JSONL)
    _log(
        f"running_rows={len(joined_rows)} shard={args.shard_index}/{args.num_shards} "
        f"dry_run={args.dry_run} force_steps={sorted(_FORCE_STEPS)}"
    )

    runner = ExperimentRunner(
        dry_run=args.dry_run,
        force_steps=_FORCE_STEPS,
        iv_design=args.iv_design,
    )
    consolidate = bool(args.consolidate or args.num_shards == 1)
    runner.run_rows(joined_rows, scenarios, consolidate=consolidate)
    _log(
        f"completed_rows={len(joined_rows)} dry_run={args.dry_run} "
        f"shard={args.shard_index}/{args.num_shards} output_dir={OUTPUT_DIR} "
        f"analysis_rows={ANALYSIS_ROWS_JSONL if consolidate else 'not_consolidated'}"
    )
    return 0


def _row_shard(row_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def _row_ids_with_incomplete_branch(
    joined_rows: list[dict[str, str]],
    rows_dir: Path,
) -> set[str]:
    incomplete_ids: set[str] = set()
    for row in joined_rows:
        row_id = row["row_id"]
        path = rows_dir / f"{_safe_name(row_id)}.json"
        if not path.exists():
            incomplete_ids.add(row_id)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            incomplete_ids.add(row_id)
            continue
        if not _has_required_fields(payload):
            incomplete_ids.add(row_id)
    return incomplete_ids


def _has_required_fields(payload: dict[str, object]) -> bool:
    for field in _REQUIRED_FIELDS:
        value = payload.get(field)
        if value is None or value == "":
            return False
    return True


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
