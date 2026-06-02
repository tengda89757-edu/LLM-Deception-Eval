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

from deception_benchmark.analysis import now_iso, read_records, write_csv, write_jsonl
from deception_benchmark.subsets import build_all_subsets


DEFAULT_ANALYSIS_ROWS = WORKSPACE_ROOT / "outputs" / "analysis_rows.jsonl"
DEFAULT_SCENARIO_REGISTRY = WORKSPACE_ROOT / "data" / "derived" / "scenario_registry.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "subsets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export H1/H2/H3/H4 annotation subsets from analysis rows.")
    parser.add_argument("--analysis-rows", type=Path, default=DEFAULT_ANALYSIS_ROWS, help="Path to analysis rows CSV/JSONL.")
    parser.add_argument("--scenario-registry", type=Path, default=DEFAULT_SCENARIO_REGISTRY, help="Path to the scenario registry CSV/JSONL.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported subset files.")
    parser.add_argument("--seed", type=int, default=20260410, help="Deterministic sampling seed.")
    parser.add_argument("--h4-threshold", type=float, default=0.85, help="Judge-confidence threshold for H4 escalation.")
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

    subsets, manifest = build_all_subsets(
        analysis_rows,
        scenario_rows,
        seed=args.seed,
        h4_threshold=args.h4_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_files: dict[str, dict[str, str]] = {}
    for subset_name, rows in subsets.items():
        csv_path = args.output_dir / f"{subset_name}.csv"
        jsonl_path = args.output_dir / f"{subset_name}.jsonl"
        status = manifest["subsets"][subset_name]["status"]
        if rows:
            write_csv(csv_path, rows)
            write_jsonl(jsonl_path, rows)
        else:
            csv_path.write_text("", encoding="utf-8")
            jsonl_path.write_text("", encoding="utf-8")
        if status != "ok":
            status_path = args.output_dir / f"{subset_name}.status.json"
            status_payload = {
                "subset_name": subset_name,
                "status": status,
                **manifest["subsets"][subset_name],
            }
            status_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            subset_files[subset_name] = {
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "status_json": str(status_path),
                "rows": len(rows),
            }
        else:
            subset_files[subset_name] = {"csv": str(csv_path), "jsonl": str(jsonl_path), "rows": len(rows)}

    manifest_path = args.output_dir / "subset_manifest.json"
    manifest_payload = {
        "generated_at": now_iso(),
        "analysis_rows": str(args.analysis_rows),
        "scenario_registry": str(args.scenario_registry),
        "output_dir": str(args.output_dir),
        "subset_files": subset_files,
        "subset_status": manifest["subsets"],
        "seed": manifest["seed"],
        "h4_threshold": manifest["h4_threshold"],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({key: value["rows"] for key, value in subset_files.items()}, ensure_ascii=False))
    print(f"subset_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
