#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from deception_benchmark.analysis import build_summary_tables, now_iso, read_records, write_csv, write_jsonl


DEFAULT_ANALYSIS_ROWS = WORKSPACE_ROOT / "outputs" / "analysis_rows.jsonl"
DEFAULT_SCENARIO_REGISTRY = WORKSPACE_ROOT / "data" / "derived" / "scenario_registry.csv"
DEFAULT_SUBSET_MANIFEST = WORKSPACE_ROOT / "outputs" / "subsets" / "subset_manifest.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary tables from analysis rows and scenario registry.")
    parser.add_argument("--analysis-rows", type=Path, default=DEFAULT_ANALYSIS_ROWS, help="Path to analysis rows CSV/JSONL.")
    parser.add_argument("--scenario-registry", type=Path, default=DEFAULT_SCENARIO_REGISTRY, help="Path to the scenario registry CSV/JSONL.")
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST, help="Optional subset manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for summary tables.")
    return parser.parse_args()


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> None:
    args = parse_args()
    _ensure_exists(args.analysis_rows, "analysis rows")
    _ensure_exists(args.scenario_registry, "scenario registry")
    analysis_rows = read_records(args.analysis_rows)
    scenario_rows = read_records(args.scenario_registry)
    subset_manifest = None
    if args.subset_manifest.exists():
        subset_manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))

    tables = build_summary_tables(analysis_rows, scenario_rows, subset_manifest=subset_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_files: dict[str, dict[str, str]] = {}
    for table_name, table in tables.items():
        rows = table["rows"]
        csv_path = args.output_dir / f"{table_name}.csv"
        jsonl_path = args.output_dir / f"{table_name}.jsonl"
        write_csv(csv_path, rows)
        write_jsonl(jsonl_path, rows)
        table_files[table_name] = {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "status": table["status"],
            "row_count": table["row_count"],
        }

    manifest_path = args.output_dir / "summary_manifest.json"
    manifest_payload = {
        "generated_at": now_iso(),
        "analysis_rows": str(args.analysis_rows),
        "scenario_registry": str(args.scenario_registry),
        "subset_manifest": str(args.subset_manifest) if args.subset_manifest.exists() else "",
        "output_dir": str(args.output_dir),
        "tables": table_files,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({key: value["row_count"] for key, value in table_files.items()}, ensure_ascii=False))
    print(f"summary_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
