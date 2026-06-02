#!/usr/bin/env python3
"""Combine two scenario-complexity annotation files into the canonical source."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIMENSION_COLUMNS = (
    "evidence_ambiguity",
    "causal_entanglement",
    "option_discriminability",
    "counterevidence_salience",
    "deception_plausibility",
    "ecological_realism",
)
OUTPUT_COLUMNS = (
    "scenario_id",
    *DIMENSION_COLUMNS,
    "gold_baseline_option_valid",
    "ScenarioComplexity_raw",
    "ScenarioComplexity_z",
    "gold_baseline_option_human_vote",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate scenario complexity annotations.")
    parser.add_argument(
        "--annotator-a",
        type=Path,
        default=ROOT / "data" / "manual" / "scenario_complexity_annotations_depo.csv",
    )
    parser.add_argument(
        "--annotator-b",
        type=Path,
        default=ROOT / "data" / "manual" / "scenario_complexity_annotations_nemo.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manual" / "scenario_complexity_annotations.csv",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "data" / "manual" / "scenario_complexity_adjudication",
    )
    parser.add_argument(
        "--adjudication-threshold",
        type=float,
        default=2.0,
        help="Flag scenario-dimension pairs with absolute annotator difference at or above this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a = load_annotation(args.annotator_a, "depo")
    b = load_annotation(args.annotator_b, "nemo")
    validate_inputs(a, b)

    merged = a.merge(b, on="scenario_id", suffixes=("_depo", "_nemo"), validate="one_to_one")
    canonical = build_canonical(merged)
    reliability = reliability_report(merged, canonical)
    disagreements = disagreement_report(merged, args.adjudication_threshold)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(args.output, index=False)
    disagreements.to_csv(args.report_dir / "scenario_complexity_disagreements.csv", index=False)
    _write_json(
        args.report_dir / "scenario_complexity_reliability.json",
        {
            "schema_version": "scenario_complexity_reliability_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "annotator_a": str(args.annotator_a),
            "annotator_b": str(args.annotator_b),
            "output": str(args.output),
            "n_scenarios": int(len(canonical)),
            "dimension_columns": list(DIMENSION_COLUMNS),
            "adjudication_rule": (
                "Canonical scores are the mean of the two annotators. "
                f"Scenario-dimension pairs with absolute difference >= {args.adjudication_threshold} "
                "are flagged for optional manual adjudication."
            ),
            "reliability": reliability,
            "flagged_disagreement_count": int(len(disagreements)),
        },
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report_dir / 'scenario_complexity_reliability.json'}")
    print(f"flagged_disagreement_count={len(disagreements)}")
    return 0


def load_annotation(path: Path, annotator: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df[list(OUTPUT_COLUMNS)].copy()
    for col in DIMENSION_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(DIMENSION_COLUMNS)].isna().any().any():
        bad = df.loc[df[list(DIMENSION_COLUMNS)].isna().any(axis=1), "scenario_id"].tolist()
        raise ValueError(f"{path} has missing/non-numeric dimension scores for: {bad}")
    df["_annotator"] = annotator
    return df


def validate_inputs(a: pd.DataFrame, b: pd.DataFrame) -> None:
    if a["scenario_id"].duplicated().any():
        raise ValueError("annotator A has duplicate scenario_id values")
    if b["scenario_id"].duplicated().any():
        raise ValueError("annotator B has duplicate scenario_id values")
    ids_a = set(a["scenario_id"])
    ids_b = set(b["scenario_id"])
    if ids_a != ids_b:
        raise ValueError(
            "scenario_id coverage differs: "
            f"only_a={sorted(ids_a - ids_b)} only_b={sorted(ids_b - ids_a)}"
        )


def build_canonical(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"scenario_id": merged["scenario_id"]})
    for col in DIMENSION_COLUMNS:
        out[col] = (merged[f"{col}_depo"] + merged[f"{col}_nemo"]) / 2.0
    out["gold_baseline_option_valid"] = combine_matching_optional(
        merged.get("gold_baseline_option_valid_depo"),
        merged.get("gold_baseline_option_valid_nemo"),
    )
    out["ScenarioComplexity_raw"] = out[list(DIMENSION_COLUMNS)].mean(axis=1)
    raw = out["ScenarioComplexity_raw"]
    std = raw.std(ddof=0)
    if not std or pd.isna(std):
        out["ScenarioComplexity_z"] = np.nan
    else:
        out["ScenarioComplexity_z"] = (raw - raw.mean()) / std
    out["gold_baseline_option_human_vote"] = combine_matching_optional(
        merged.get("gold_baseline_option_human_vote_depo"),
        merged.get("gold_baseline_option_human_vote_nemo"),
    )
    for col in DIMENSION_COLUMNS:
        out[col] = out[col].round(4)
    out["ScenarioComplexity_raw"] = out["ScenarioComplexity_raw"].round(4)
    out["ScenarioComplexity_z"] = out["ScenarioComplexity_z"].round(4)
    return out[list(OUTPUT_COLUMNS)].sort_values("scenario_id")


def combine_matching_optional(a: pd.Series | None, b: pd.Series | None) -> pd.Series:
    if a is None or b is None:
        return pd.Series([""] * 0)
    a_text = a.fillna("").astype(str)
    b_text = b.fillna("").astype(str)
    return a_text.where(a_text == b_text, "")


def reliability_report(merged: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in (*DIMENSION_COLUMNS, "ScenarioComplexity_raw"):
        if col == "ScenarioComplexity_raw":
            depo = merged[[f"{dim}_depo" for dim in DIMENSION_COLUMNS]].mean(axis=1)
            nemo = merged[[f"{dim}_nemo" for dim in DIMENSION_COLUMNS]].mean(axis=1)
        else:
            depo = merged[f"{col}_depo"]
            nemo = merged[f"{col}_nemo"]
        diff = depo - nemo
        out[col] = {
            "pearson_r": safe_float(depo.corr(nemo, method="pearson")),
            "spearman_r": safe_float(depo.corr(nemo, method="spearman")),
            "mean_abs_diff": safe_float(diff.abs().mean()),
            "max_abs_diff": safe_float(diff.abs().max()),
            "exact_agreement_rate": safe_float((diff == 0).mean()),
            "within_1_rate": safe_float((diff.abs() <= 1).mean()),
            "icc_2_1": safe_float(icc_2_1(depo, nemo)),
            "icc_2_k": safe_float(icc_2_k(depo, nemo)),
        }
    out["canonical_raw_summary"] = {
        "mean": safe_float(canonical["ScenarioComplexity_raw"].mean()),
        "sd_population": safe_float(canonical["ScenarioComplexity_raw"].std(ddof=0)),
        "min": safe_float(canonical["ScenarioComplexity_raw"].min()),
        "max": safe_float(canonical["ScenarioComplexity_raw"].max()),
    }
    return out


def disagreement_report(merged: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for col in DIMENSION_COLUMNS:
            depo = float(row[f"{col}_depo"])
            nemo = float(row[f"{col}_nemo"])
            diff = abs(depo - nemo)
            if diff >= threshold:
                rows.append(
                    {
                        "scenario_id": row["scenario_id"],
                        "dimension": col,
                        "depo_score": depo,
                        "nemo_score": nemo,
                        "abs_diff": diff,
                    }
                )
    return pd.DataFrame(rows, columns=["scenario_id", "dimension", "depo_score", "nemo_score", "abs_diff"])


def icc_2_1(a: pd.Series, b: pd.Series) -> float:
    msr, msc, mse, n, k = icc_anova_terms(a, b)
    denominator = msr + (k - 1) * mse + (k * (msc - mse) / n)
    return float((msr - mse) / denominator) if denominator else float("nan")


def icc_2_k(a: pd.Series, b: pd.Series) -> float:
    msr, msc, mse, n, _ = icc_anova_terms(a, b)
    denominator = msr + ((msc - mse) / n)
    return float((msr - mse) / denominator) if denominator else float("nan")


def icc_anova_terms(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int, int]:
    ratings = np.column_stack([a.to_numpy(dtype=float), b.to_numpy(dtype=float)])
    n, k = ratings.shape
    grand_mean = ratings.mean()
    row_means = ratings.mean(axis=1, keepdims=True)
    col_means = ratings.mean(axis=0, keepdims=True)
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    residual = ratings - row_means - col_means + grand_mean
    ss_error = (residual**2).sum()
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    return float(msr), float(msc), float(mse), int(n), int(k)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return round(out, 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
