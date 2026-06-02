#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    required = [
        "README.md",
        "DATA_CARD.md",
        "RESPONSIBLE_RELEASE.md",
        "requirements.txt",
        "deception_benchmark/__init__.py",
        "scripts/run_experiment.py",
        "prompts/attacker.yaml",
        "data/derived/scenario_registry.csv",
        "results/canonical_run/analysis_rows_public.csv",
        "results/final_results_20260427/core_metrics.csv",
        "validation/human_validation_v2/rerun_annotation_summary.json",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    scenarios = list((ROOT / "data" / "normalized-scenarios").rglob("*.json"))
    if len(scenarios) != 53:
        raise SystemExit(f"Expected 53 normalized scenarios, found {len(scenarios)}")
    rows = count_csv_rows(ROOT / "results" / "canonical_run" / "analysis_rows_public.csv")
    if rows != 2968:
        raise SystemExit(f"Expected 2968 public analysis rows, found {rows}")
    text_paths = [
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".yaml", ".yml", ".tex", ".cff"}
    ]
    local_home = "/Users/" + "apple"
    api_env_names = ["SILICONFLOW_API_KEY", "OPENROUTER_API_KEY"]
    forbidden = re.compile(
        re.escape(local_home)
        + r"|sk-[A-Za-z0-9_-]{20,}"
        + "|"
        + "|".join(re.escape(name + "=") + r"(?!replace_me)" for name in api_env_names)
    )
    hits = []
    for path in text_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if forbidden.search(text):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        raise SystemExit(f"Public-release scan found sensitive strings: {hits[:20]}")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise SystemExit(f"Manifest file missing: {item['path']}")
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"Manifest hash mismatch: {item['path']}")
    print("Public release verification passed.")
    print(f"Files: {len(manifest['files'])}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Public analysis rows: {rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
