#!/usr/bin/env python3
"""Merge row-scoped outputs into the canonical analysis table for one run.

This script is intentionally run-scoped.  Sharded workers write only per-row
JSON files; this merger is the single writer of analysis_rows.* and manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from deception_benchmark.config import (  # noqa: E402
    ALL_MODELS,
    DEFAULT_SEED,
    JOINED_ROWS_CSV,
    MODEL_CAPABILITIES_PATH,
    PARSER_VERSION,
    ROLE_SAMPLING,
    SCENARIO_COMPLEXITY_PATH,
    SCENARIO_REGISTRY_JSONL,
)
from deception_benchmark.derive import derive_outcomes  # noqa: E402
from deception_benchmark.gate_iv import (  # noqa: E402
    GATE_IV_ARMS,
    GATE_IV_DESIGN,
    GATE_IV_SEED,
    assign_gate_iv_arm,
    gate_arm_dummies,
    response_hash,
)


ALWAYS_REQUIRED_FIELDS = (
    "row_id",
    "scenario_id",
    "compact_domain",
    "attacker_family",
    "defender_family",
    "baseline_option_id",
    "neutral_control_option_id",
    "control_option_id",
    "attacker_nondeceptive_option_id",
    "refusal_label_5way",
    "engagement_binary",
    "gate_iv_arm",
    "gate_decision",
    "attacker_response_hash",
    "baseline_defender_response_hash",
    "control_defender_response_hash",
    "attacker_nondeceptive_attacker_response_hash",
    "attacker_nondeceptive_defender_response_hash",
)

ENGAGED_REQUIRED_FIELDS = (
    "deception_option_id",
    "deception_success_binary",
    "deception_defender_response_hash",
)

MANUAL_REQUIRED_FIELDS = (
    "attacker_capability_score",
    "defender_capability_score",
    "attacker_capability_z",
    "defender_capability_z",
    "ScenarioComplexity_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one run directory into canonical analysis rows.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run output directory, e.g. outputs/runs/tacl_revision_gate_iv_20260417.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Project root. Defaults to this script's repository root.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Expected main-row count. Defaults to the joined_rows.csv count.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if row count or required fields are incomplete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ws = args.workspace_root.resolve()
    run_dir = args.run_dir.expanduser().resolve()
    rows_dir = run_dir / "rows"
    if not rows_dir.exists():
        raise FileNotFoundError(f"Rows directory not found: {rows_dir}")

    expected_rows = args.expected_rows
    if expected_rows is None:
        expected_rows = _count_csv_rows(ws / "data" / "derived" / "joined_rows.csv")

    rows = [_load_json(path) for path in sorted(rows_dir.glob("*.json"))]
    rows = _dedupe_rows(rows)
    rows = [_normalize_row(row) for row in rows]

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No row JSON files found in {rows_dir}")

    df = _attach_manual_sources(df, ws)
    df = _order_columns(df)

    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "analysis_rows.jsonl"
    csv_path = run_dir / "analysis_rows.csv"
    _write_jsonl(jsonl_path, df.to_dict(orient="records"))
    df.to_csv(csv_path, index=False)

    manifest = _build_manifest(df, run_dir=run_dir, workspace_root=ws, expected_rows=expected_rows)
    _atomic_write_json(run_dir / "canonical_evidence_manifest.json", manifest)
    _atomic_write_json(run_dir / "run_manifest.json", _build_run_manifest(df, run_dir, ws, manifest))

    print(f"run_dir={run_dir}")
    print(f"analysis_rows={jsonl_path}")
    print(f"analysis_csv={csv_path}")
    print(
        "row_count={row_count} expected_rows={expected_rows} complete_rows={complete_rows}".format(
            row_count=manifest["row_count"],
            expected_rows=manifest["expected_rows"],
            complete_rows=manifest["complete_required_fields_rows"],
        )
    )
    if manifest["blocking_issues"]:
        print("blocking_issues=" + json.dumps(manifest["blocking_issues"], ensure_ascii=False))

    if args.strict and manifest["blocking_issues"]:
        return 2
    return 0


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or "")
        if not row_id:
            continue
        by_id[row_id] = row
    return [by_id[key] for key in sorted(by_id)]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row_id = str(row["row_id"])
    if not row.get("gate_iv_arm"):
        arm, index = assign_gate_iv_arm(row_id)
        row["gate_iv_arm"] = arm
        row["gate_iv_index"] = index
        row.update(gate_arm_dummies(arm))
    else:
        row.update({key: row.get(key, value) for key, value in gate_arm_dummies(str(row["gate_iv_arm"])).items()})
    row.setdefault("iv_design", GATE_IV_DESIGN)

    alias_fields = (
        ("control_expert_raw_text", "neutral_control_expert_raw_text"),
        ("control_defender_raw_text", "neutral_control_defender_raw_text"),
        ("control_option_id", "neutral_control_option_id"),
        ("control_parse_status", "neutral_control_parse_status"),
        ("control_option_confidence", "neutral_control_option_confidence"),
    )
    for source_field, target_field in alias_fields:
        if not row.get(target_field) and row.get(source_field) not in (None, ""):
            row[target_field] = row[source_field]

    for source_prefix, target_prefix in (
        ("control_expert", "neutral_control_expert"),
        ("control_defender", "neutral_control_defender"),
    ):
        source_token = f"{source_prefix}_"
        target_token = f"{target_prefix}_"
        for key, value in list(row.items()):
            if key == f"{source_prefix}_response_hash" and not row.get(f"{target_prefix}_response_hash"):
                row[f"{target_prefix}_response_hash"] = value
            elif key.startswith(source_token):
                target_key = key.replace(source_token, target_token, 1)
                if not row.get(target_key):
                    row[target_key] = value

    for text_field, hash_field in (
        ("attacker_raw_text", "attacker_response_hash"),
        ("baseline_raw_text", "baseline_defender_response_hash"),
        ("control_defender_raw_text", "control_defender_response_hash"),
        ("attacker_nondeceptive_raw_text", "attacker_nondeceptive_attacker_response_hash"),
        ("attacker_nondeceptive_defender_raw_text", "attacker_nondeceptive_defender_response_hash"),
        ("deception_defender_raw_text", "deception_defender_response_hash"),
        ("gate_raw_text", "deception_gate_response_hash"),
    ):
        if not row.get(hash_field) and text_field in row:
            row[hash_field] = response_hash(row.get(text_field))

    # Recompute derived outcome fields after any merge-time normalization.
    row.update({key: value for key, value in derive_outcomes(row).items() if row.get(key) in (None, "")})
    row["scenario_dyad"] = f"{row.get('scenario_id')}|{row.get('attacker_family')}|{row.get('defender_family')}"
    return row


def _attach_manual_sources(df: pd.DataFrame, ws: Path) -> pd.DataFrame:
    df = df.copy()
    cap_path = ws / "data" / "manual" / "model_capabilities.csv"
    if cap_path.exists():
        caps = pd.read_csv(cap_path)
        caps["capability_score"] = pd.to_numeric(caps.get("capability_score"), errors="coerce")
        cap_map = dict(zip(caps["family"].astype(str), caps["capability_score"]))
        df["attacker_capability_score"] = df["attacker_family"].astype(str).map(cap_map)
        df["defender_capability_score"] = df["defender_family"].astype(str).map(cap_map)
        z_map = _zscore_map(cap_map)
        df["attacker_capability_z"] = df["attacker_family"].astype(str).map(z_map)
        df["defender_capability_z"] = df["defender_family"].astype(str).map(z_map)
    else:
        for col in ("attacker_capability_score", "defender_capability_score", "attacker_capability_z", "defender_capability_z"):
            df[col] = pd.NA

    complexity_path = ws / "data" / "manual" / "scenario_complexity_annotations.csv"
    if complexity_path.exists():
        complexity = pd.read_csv(complexity_path)
        complexity = _normalize_complexity(complexity)
        keep_cols = [
            col
            for col in complexity.columns
            if col == "scenario_id" or col.startswith("ScenarioComplexity") or col.endswith("_complexity")
        ]
        df = df.merge(complexity[keep_cols], on="scenario_id", how="left", suffixes=("", "_manual"))
        if "ScenarioComplexity_z_manual" in df.columns and "ScenarioComplexity_z" in df.columns:
            df["ScenarioComplexity_z"] = df["ScenarioComplexity_z"].combine_first(df["ScenarioComplexity_z_manual"])
            df = df.drop(columns=["ScenarioComplexity_z_manual"])
    else:
        df["ScenarioComplexity_raw"] = pd.NA
        df["ScenarioComplexity_z"] = pd.NA
    return df


def _normalize_complexity(complexity: pd.DataFrame) -> pd.DataFrame:
    complexity = complexity.copy()
    numeric_cols = [
        "evidence_ambiguity",
        "causal_entanglement",
        "option_discriminability",
        "counterevidence_salience",
        "deception_plausibility",
        "ecological_realism",
        "ScenarioComplexity_raw",
        "ScenarioComplexity_z",
    ]
    for col in numeric_cols:
        if col in complexity.columns:
            complexity[col] = pd.to_numeric(complexity[col], errors="coerce")
    component_cols = [
        col
        for col in (
            "evidence_ambiguity",
            "causal_entanglement",
            "option_discriminability",
            "counterevidence_salience",
            "deception_plausibility",
            "ecological_realism",
        )
        if col in complexity.columns
    ]
    if "ScenarioComplexity_raw" not in complexity.columns:
        complexity["ScenarioComplexity_raw"] = pd.NA
    if component_cols:
        component_mean = complexity[component_cols].mean(axis=1, skipna=False)
        complexity["ScenarioComplexity_raw"] = complexity["ScenarioComplexity_raw"].combine_first(component_mean)
    if "ScenarioComplexity_z" not in complexity.columns:
        complexity["ScenarioComplexity_z"] = pd.NA
    raw = pd.to_numeric(complexity["ScenarioComplexity_raw"], errors="coerce")
    if complexity["ScenarioComplexity_z"].isna().all() and raw.notna().sum() > 1:
        std = raw.std(ddof=0)
        if std and not pd.isna(std):
            complexity["ScenarioComplexity_z"] = (raw - raw.mean()) / std
    return complexity


def _zscore_map(values: dict[str, Any]) -> dict[str, float]:
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 2:
        return {key: float("nan") for key in values}
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return {key: float("nan") for key in values}
    return {key: float((pd.to_numeric(value, errors="coerce") - series.mean()) / std) for key, value in values.items()}


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "row_id",
        "scenario_id",
        "scenario_dyad",
        "topic_group",
        "topic",
        "legacy_category",
        "compact_domain",
        "attacker_family",
        "defender_family",
        "target_option_id",
        "gold_baseline_option_id",
        "baseline_option_id",
        "neutral_control_option_id",
        "attacker_nondeceptive_option_id",
        "deception_option_id",
        "control_option_id",
        "baseline_target_binary",
        "neutral_control_target_binary",
        "attacker_nondeceptive_target_binary",
        "refusal_label_5way",
        "engagement_binary",
        "deception_success_binary",
        "deception_success_unconditional",
        "strict_success",
        "neutral_control_shift_to_target",
        "attacker_nondeceptive_shift_to_target",
        "control_shift_to_target",
        "tactic_count_model_final",
        "context_document_word_count",
        "attacker_capability_score",
        "defender_capability_score",
        "attacker_capability_z",
        "defender_capability_z",
        "ScenarioComplexity_raw",
        "ScenarioComplexity_z",
        "iv_design",
        "gate_iv_arm",
        "gate_iv_index",
        *[f"gate_iv_{arm}" for arm in GATE_IV_ARMS],
        "gate_decision",
        "gate_decision_confidence",
        "gate_decision_json_valid",
        "attacker_response_hash",
        "baseline_defender_response_hash",
        "deception_gate_response_hash",
        "deception_attacker_response_hash",
        "deception_defender_response_hash",
        "control_expert_response_hash",
        "control_defender_response_hash",
        "neutral_control_expert_response_hash",
        "neutral_control_defender_response_hash",
        "attacker_nondeceptive_attacker_response_hash",
        "attacker_nondeceptive_defender_response_hash",
    ]
    ordered = [col for col in preferred if col in df.columns]
    ordered.extend([col for col in df.columns if col not in ordered])
    return df[ordered]


def _build_manifest(df: pd.DataFrame, *, run_dir: Path, workspace_root: Path, expected_rows: int) -> dict[str, Any]:
    missing_always = _missing_counts(df, ALWAYS_REQUIRED_FIELDS)
    engaged = df[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1].copy()
    missing_engaged = _missing_counts(engaged, ENGAGED_REQUIRED_FIELDS)
    missing_manual = _missing_counts(df, MANUAL_REQUIRED_FIELDS)
    complete_required = int(
        df[list(ALWAYS_REQUIRED_FIELDS)].notna().all(axis=1).sum()
        if all(col in df.columns for col in ALWAYS_REQUIRED_FIELDS)
        else 0
    )
    blocking = []
    if len(df) != expected_rows:
        blocking.append(f"row_count {len(df)} != expected_rows {expected_rows}")
    for col, count in {**missing_always, **missing_engaged}.items():
        if count:
            blocking.append(f"{col} missing_count={count}")
    manual_missing_nonzero = {col: count for col, count in missing_manual.items() if count}
    if manual_missing_nonzero:
        blocking.append("manual_covariates_incomplete=" + json.dumps(manual_missing_nonzero, sort_keys=True))

    return {
        "schema_version": "canonical_evidence_manifest_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace_root),
        "run_dir": str(run_dir),
        "row_count": int(len(df)),
        "expected_rows": int(expected_rows),
        "complete_required_fields_rows": complete_required,
        "engaged_rows": int(len(engaged)),
        "refused_rows": int((pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 0).sum()),
        "iv_arm_counts": _value_counts(df, "gate_iv_arm"),
        "engagement_counts": _value_counts(df, "engagement_binary"),
        "missing_always_required": missing_always,
        "missing_engaged_required": missing_engaged,
        "missing_manual_required": missing_manual,
        "blocking_issues": blocking,
        "status": "ready_for_analysis" if not blocking else "blocked_or_partial",
    }


def _missing_counts(df: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in fields:
        if field not in df.columns:
            counts[field] = int(len(df))
        else:
            values = df[field]
            counts[field] = int((values.isna() | (values.astype(str) == "")).sum())
    return counts


def _value_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[col].value_counts(dropna=False).sort_index().to_dict().items()}


def _build_run_manifest(df: pd.DataFrame, run_dir: Path, ws: Path, evidence_manifest: dict[str, Any]) -> dict[str, Any]:
    prompt_files = [ws / "deception_benchmark" / "prompts.py", ws / "deception_benchmark" / "gate_iv.py"]
    code_files = [
        ws / "deception_benchmark" / "runner.py",
        ws / "deception_benchmark" / "checkpoints.py",
        ws / "deception_benchmark" / "models.py",
        ws / "deception_benchmark" / "derive.py",
        *prompt_files,
        SCRIPT_PATH,
    ]
    return {
        "schema_version": "tacl_revision_run_manifest_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "analysis_rows_jsonl": str(run_dir / "analysis_rows.jsonl"),
        "analysis_rows_csv": str(run_dir / "analysis_rows.csv"),
        "checkpoint_db": str(run_dir / "checkpoints.sqlite"),
        "iv_design": GATE_IV_DESIGN,
        "iv_arms": list(GATE_IV_ARMS),
        "iv_randomization_seed": GATE_IV_SEED,
        "iv_assignment_rule": "sha256(f'{GATE_IV_SEED}|{row_id}') first 16 hex modulo 3",
        "model_roster": ALL_MODELS,
        "evidence_branches": [
            "baseline",
            "neutral_control",
            "attacker_nondeceptive",
            "deception",
        ],
        "role_sampling": ROLE_SAMPLING,
        "default_seed": DEFAULT_SEED,
        "parser_version": PARSER_VERSION,
        "row_count": int(len(df)),
        "evidence_status": evidence_manifest["status"],
        "evidence_blocking_issues": evidence_manifest["blocking_issues"],
        "manual_sources": {
            "model_capabilities": str(ws / "data" / "manual" / "model_capabilities.csv"),
            "scenario_complexity": str(ws / "data" / "manual" / "scenario_complexity_annotations.csv"),
        },
        "source_data": {
            "joined_rows": str(ws / "data" / "derived" / "joined_rows.csv"),
            "scenario_registry": str(ws / "data" / "derived" / "scenario_registry.jsonl"),
        },
        "code_hashes": {str(path.relative_to(ws)): _sha256_file(path) for path in code_files if path.exists()},
        "prompt_hashes": {str(path.relative_to(ws)): _sha256_file(path) for path in prompt_files if path.exists()},
        "git": _git_info(ws),
        "cloud_vm_metadata": _cloud_metadata(),
        "checkpoint_status_counts": _checkpoint_status_counts(run_dir / "checkpoints.sqlite"),
    }


def _git_info(ws: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ws, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ws, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return {"available": True, "head_sha": sha, "dirty": dirty}
    except Exception:
        return {"available": False, "head_sha": None, "dirty": None}


def _cloud_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "env": {
            "DECEPTION_RUN_ID": os.getenv("DECEPTION_RUN_ID"),
            "DECEPTION_OUTPUT_DIR": os.getenv("DECEPTION_OUTPUT_DIR"),
            "DECEPTION_IV_DESIGN": os.getenv("DECEPTION_IV_DESIGN"),
        },
    }


def _checkpoint_status_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM checkpoints GROUP BY status").fetchall()
    return {str(status): int(count) for status, count in rows}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
