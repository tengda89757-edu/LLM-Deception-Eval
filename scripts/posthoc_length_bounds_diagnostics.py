#!/usr/bin/env python3
"""Post-hoc length and partial-identification diagnostics for the TACL draft."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROWS = ROOT / "outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/analysis_rows.csv"
RESPONSES_DIR = ROOT / "outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/responses"
OUT_DIR = ROOT / "outputs/final_results_20260427"


def whitespace_words(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return float("nan")
    return float(len(re.findall(r"\S+", value)))


def response_dir_for_row(row_id: str) -> Path:
    return RESPONSES_DIR / row_id.replace("|", "_")


def backfill_neutral_memos(df: pd.DataFrame) -> int:
    backfilled = 0
    missing = df["neutral_control_expert_raw_text"].map(lambda value: not isinstance(value, str) or not value.strip())
    for idx, row in df.loc[missing].iterrows():
        path = response_dir_for_row(str(row["row_id"])) / "control__expert.json"
        if not path.exists():
            continue
        raw_text = json.loads(path.read_text(encoding="utf-8")).get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            df.at[idx, "neutral_control_expert_raw_text"] = raw_text
            backfilled += 1
    return backfilled


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half


def tertile_table(
    df: pd.DataFrame,
    *,
    length_col: str,
    outcome_col: str,
    branch: str,
    outcome_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = df.dropna(subset=[length_col]).copy()
    work["length_tertile"] = pd.qcut(
        work[length_col],
        q=3,
        labels=["short", "medium", "long"],
        duplicates="drop",
    )
    rows = []
    for tertile, group in work.groupby("length_tertile", observed=True):
        n = len(group)
        k = int(pd.to_numeric(group[outcome_col]).fillna(0).sum())
        lo, hi = wilson(k, n)
        rows.append(
            {
                "branch": branch,
                "outcome": outcome_label,
                "length_tertile": str(tertile),
                "n": n,
                "events": k,
                "rate": k / n if n else np.nan,
                "wilson_low": lo,
                "wilson_high": hi,
                "min_words": int(group[length_col].min()),
                "max_words": int(group[length_col].max()),
                "median_words": float(group[length_col].median()),
            }
        )

    crosstab = pd.crosstab(work["length_tertile"], work[outcome_col])
    chi2_p = float(chi2_contingency(crosstab)[1]) if crosstab.shape == (3, 2) else float("nan")
    rho, rho_p = spearmanr(work[length_col], work[outcome_col], nan_policy="omit")
    stats = {
        "branch": branch,
        "outcome": outcome_label,
        "length_col": length_col,
        "n": int(len(work)),
        "cutpoints": [float(x) for x in np.quantile(work[length_col], [0, 1 / 3, 2 / 3, 1])],
        "chi_square_p": chi2_p,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }
    return pd.DataFrame(rows), stats


def bounds_for_group(group: pd.DataFrame) -> dict[str, float | int]:
    n = len(group)
    generated = int((group["engagement_binary"] == 1).sum())
    shift = int(((group["engagement_binary"] == 1) & (group["deception_success_binary"] == 1)).sum())
    refused_eligible = int(((group["engagement_binary"] == 0) & (group["baseline_target_binary"] == 0)).sum())
    eligible = int((group["baseline_target_binary"] == 0).sum())
    return {
        "n": n,
        "generated": generated,
        "observed_shift": shift,
        "lower_bound": shift / n if n else np.nan,
        "refused_eligible_added": refused_eligible,
        "upper_bound": (shift + refused_eligible) / n if n else np.nan,
        "bound_width": refused_eligible / n if n else np.nan,
        "baseline_non_target_eligible": eligible,
    }


def stratified_bounds(df: pd.DataFrame, grouping: list[str], label: str) -> pd.DataFrame:
    records = []
    for keys, group in df.groupby(grouping, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {"stratification": label}
        for name, value in zip(grouping, keys):
            record[name] = value
        record.update(bounds_for_group(group))
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ANALYSIS_ROWS)
    backfilled_neutral_memos = backfill_neutral_memos(df)
    df["neutral_memo_words_calc"] = df["neutral_control_expert_raw_text"].map(whitespace_words)
    df["attacker_words_calc"] = df["attacker_raw_text"].map(whitespace_words)

    generated = df[df["engagement_binary"] == 1].copy()
    length_tables = []
    length_stats = []

    table, stats = tertile_table(
        generated,
        length_col="attacker_words_calc",
        outcome_col="deception_success_binary",
        branch="generated_target_aware",
        outcome_label="conditional_target_shift",
    )
    length_tables.append(table)
    length_stats.append(stats)

    table, stats = tertile_table(
        df,
        length_col="neutral_memo_words_calc",
        outcome_col="neutral_control_shift_to_target",
        branch="target_unaware_advice",
        outcome_label="target_unaware_shift_to_target",
    )
    length_tables.append(table)
    length_stats.append(stats)

    length_df = pd.concat(length_tables, ignore_index=True)
    length_df.to_csv(OUT_DIR / "posthoc_length_diagnostics.csv", index=False)

    bounds_df = pd.concat(
        [
            stratified_bounds(df, ["compact_domain"], "domain"),
            stratified_bounds(df, ["attacker_family"], "attacker_family"),
            stratified_bounds(df, ["defender_family"], "defender_family"),
            stratified_bounds(df, ["gate_iv_arm"], "gate_arm"),
            stratified_bounds(df, ["gate_iv_arm", "compact_domain"], "gate_arm_by_domain"),
        ],
        ignore_index=True,
    )
    bounds_df.to_csv(OUT_DIR / "posthoc_partial_bounds_by_covariate.csv", index=False)

    summary = {
        "analysis_rows": str(ANALYSIS_ROWS.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "generated_interventions": int((df["engagement_binary"] == 1).sum()),
        "backfilled_neutral_memos": backfilled_neutral_memos,
        "attacker_generated_words": {
            "mean": float(generated["attacker_words_calc"].mean()),
            "median": float(generated["attacker_words_calc"].median()),
        },
        "neutral_memo_words": {
            "mean": float(df["neutral_memo_words_calc"].mean()),
            "median": float(df["neutral_memo_words_calc"].median()),
        },
        "length_stats": length_stats,
        "domain_bound_range": {
            "lower_min": float(bounds_df.loc[bounds_df["stratification"] == "domain", "lower_bound"].min()),
            "lower_max": float(bounds_df.loc[bounds_df["stratification"] == "domain", "lower_bound"].max()),
            "upper_min": float(bounds_df.loc[bounds_df["stratification"] == "domain", "upper_bound"].min()),
            "upper_max": float(bounds_df.loc[bounds_df["stratification"] == "domain", "upper_bound"].max()),
        },
        "outputs": [
            str((OUT_DIR / "posthoc_length_diagnostics.csv").relative_to(ROOT)),
            str((OUT_DIR / "posthoc_partial_bounds_by_covariate.csv").relative_to(ROOT)),
        ],
    }
    (OUT_DIR / "posthoc_length_bounds_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
