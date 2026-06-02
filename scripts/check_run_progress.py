#!/usr/bin/env python3
"""Run-scoped progress and coverage checks for cloud workers."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_FIELDS = (
    "row_id",
    "scenario_id",
    "gate_iv_arm",
    "gate_decision",
    "refusal_label_5way",
    "engagement_binary",
    "baseline_option_id",
    "neutral_control_option_id",
    "control_option_id",
    "attacker_nondeceptive_option_id",
    "attacker_response_hash",
    "baseline_defender_response_hash",
    "control_defender_response_hash",
    "attacker_nondeceptive_attacker_response_hash",
    "attacker_nondeceptive_defender_response_hash",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check row and checkpoint progress for one run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--expected-rows", type=int, default=2968, help="Expected row JSON count.")
    parser.add_argument(
        "--required-field",
        action="append",
        default=[],
        help="Additional required field to count. Can be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    rows_dir = run_dir / "rows"
    rows = _load_rows(rows_dir)
    required = tuple(dict.fromkeys((*DEFAULT_REQUIRED_FIELDS, *args.required_field)))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "expected_rows": args.expected_rows,
        "row_json_count": len(rows),
        "unique_row_ids": len({row.get("row_id") for row in rows if row.get("row_id")}),
        "field_coverage": _field_coverage(rows, required),
        "iv_arm_counts": _value_counts(rows, "gate_iv_arm"),
        "engagement_counts": _value_counts(rows, "engagement_binary"),
        "refusal_label_counts": _value_counts(rows, "refusal_label_5way"),
        "checkpoint_status_counts": _checkpoint_status_counts(run_dir / "checkpoints.sqlite"),
        "latest_checkpoint_start": _latest_checkpoint_time(run_dir / "checkpoints.sqlite", "started_at"),
        "latest_checkpoint_finish": _latest_checkpoint_time(run_dir / "checkpoints.sqlite", "finished_at"),
        "complete": len(rows) == args.expected_rows and all(item["missing_count"] == 0 for item in _field_coverage(rows, required).values()),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"run_dir={report['run_dir']}")
        print(f"rows={report['row_json_count']} unique={report['unique_row_ids']} expected={report['expected_rows']}")
        print(f"checkpoint_status_counts={report['checkpoint_status_counts']}")
        print(f"iv_arm_counts={report['iv_arm_counts']}")
        print(f"engagement_counts={report['engagement_counts']}")
        for field, item in report["field_coverage"].items():
            print(f"field={field} present={item['present_count']} missing={item['missing_count']}")
        print(f"complete={report['complete']}")
    return 0


def _load_rows(rows_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not rows_dir.exists():
        return rows
    for path in sorted(rows_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            rows.append({"row_id": path.stem, "_json_decode_error": 1})
    return rows


def _field_coverage(rows: list[dict[str, Any]], required: tuple[str, ...]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for field in required:
        missing = 0
        for row in rows:
            value = row.get(field)
            if value is None or value == "":
                missing += 1
        result[field] = {"present_count": len(rows) - missing, "missing_count": missing}
    return result


def _value_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        key = "<missing>" if value is None or value == "" else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _checkpoint_status_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM checkpoints GROUP BY status").fetchall()
    return {str(status): int(count) for status, count in rows}


def _latest_checkpoint_time(db_path: Path, column: str) -> str | None:
    if not db_path.exists():
        return None
    if column not in {"started_at", "finished_at"}:
        raise ValueError(f"Invalid checkpoint timestamp column: {column}")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"SELECT MAX({column}) FROM checkpoints").fetchone()
    return None if row is None else row[0]


if __name__ == "__main__":
    raise SystemExit(main())
