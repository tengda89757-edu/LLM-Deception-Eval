#!/usr/bin/env python3
"""Human-validation representativeness analysis for the TACL revision.

The active default is the two-layer human-validation package exported by
scripts/export_human_validation_two_layer.py. Representativeness diagnostics
apply to the full-denominator core validation sample. The generated tactic
audit is a census of generated interventions, so it is documented in the
manifest rather than treated as a full-denominator random sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scipy.stats as stats


def _resolve_workspace() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-validation representativeness analysis.")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=_resolve_workspace(),
        help="Project root directory. Defaults to the repository root.",
    )
    parser.add_argument(
        "--analysis-rows",
        type=Path,
        default=None,
        help=(
            "Path to the full analysis rows JSONL. Defaults to the canonical "
            "gate-only IV run under <workspace-root>/outputs/runs/."
        ),
    )
    parser.add_argument(
        "--subset-path",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Additional validation subset to compare (can be repeated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSV/JSON. Defaults to <workspace-root>/revision_supplementary/results.",
    )
    return parser.parse_args()


def get_paths(args: argparse.Namespace) -> tuple[Path, dict[str, Path], Path]:
    ws: Path = args.workspace_root
    full_sample = (
        args.analysis_rows
        or ws / "outputs" / "runs" / "tacl_revision_gate_iv_20260417_ai_completed" / "analysis_rows.jsonl"
    )
    active_validation = (
        ws
        / "revision_supplementary"
        / "human_validation_v2"
        / "human_validation_V2_N360_random_machine_key.csv"
    )
    subsets = {}
    if active_validation.exists():
        subsets["HUMAN_VALIDATION_V2_N360_RANDOM"] = active_validation
    if args.subset_path:
        for name, path_str in args.subset_path:
            subsets[name] = Path(path_str)
    results_dir = args.output_dir or ws / "revision_supplementary" / "human_validation_v2" / "results"
    return full_sample, subsets, results_dir


def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def cohens_d(x1: pd.Series, x2: pd.Series) -> float:
    """Cohen's d for two independent samples (uses pooled SD)."""
    x1 = x1.dropna().astype(float)
    x2 = x2.dropna().astype(float)
    if len(x1) == 0 or len(x2) == 0:
        return float("nan")
    pooled_std = ((len(x1) - 1) * x1.var(ddof=1) + (len(x2) - 1) * x2.var(ddof=1)) / (len(x1) + len(x2) - 2)
    pooled_std = float(pooled_std) ** 0.5
    if pooled_std == 0:
        return 0.0
    return (x1.mean() - x2.mean()) / pooled_std


def categorical_comparison(full: pd.DataFrame, subset: pd.DataFrame, column: str, subset_name: str) -> dict:
    full = full[full[column].notna() & (full[column] != "")].copy()
    subset = subset[subset[column].notna() & (subset[column] != "")].copy()
    full[column] = full[column].astype(str)
    subset[column] = subset[column].astype(str)
    full_counts = full[column].value_counts()
    sub_counts = subset[column].value_counts()
    all_cats = sorted(set(full_counts.index).union(set(sub_counts.index)))

    obs = [sub_counts.get(c, 0) for c in all_cats]
    exp_props = [full_counts.get(c, 0) / len(full) for c in all_cats]
    n = len(subset)
    exp = [n * p for p in exp_props]

    # Chi-square goodness-of-fit against full-sample proportions
    if all(e > 0 for e in exp):
        chi2, pvalue = stats.chisquare(f_obs=obs, f_exp=exp)
    else:
        chi2, pvalue = float("nan"), float("nan")

    # Cramér's V (effect size for goodness-of-fit)
    if sum(obs) > 0 and all(e > 0 for e in exp):
        cramers_v = (chi2 / sum(obs)) ** 0.5
    else:
        cramers_v = float("nan")

    return {
        "subset": subset_name,
        "variable": column,
        "variable_type": "categorical",
        "full_n": len(full),
        "subset_n": len(subset),
        "chi2": round(chi2, 3),
        "pvalue": round(pvalue, 4),
        "cramers_v": round(cramers_v, 3),
        "full_distribution": {str(k): int(v) for k, v in full_counts.to_dict().items()},
        "subset_distribution": {str(k): int(v) for k, v in sub_counts.to_dict().items()},
    }


def continuous_comparison(full: pd.DataFrame, subset: pd.DataFrame, column: str, subset_name: str) -> dict:
    full_vals = pd.to_numeric(full[column], errors="coerce").dropna()
    sub_vals = pd.to_numeric(subset[column], errors="coerce").dropna()

    if len(full_vals) == 0 or len(sub_vals) == 0:
        return {
            "subset": subset_name,
            "variable": column,
            "variable_type": "continuous",
            "full_n": len(full_vals),
            "subset_n": len(sub_vals),
            "full_mean": None,
            "subset_mean": None,
            "t_stat": None,
            "pvalue": None,
            "cohens_d": None,
        }

    t_stat, pvalue = stats.ttest_ind(sub_vals, full_vals, equal_var=False)
    d = cohens_d(sub_vals, full_vals)

    return {
        "subset": subset_name,
        "variable": column,
        "variable_type": "continuous",
        "full_n": len(full_vals),
        "subset_n": len(sub_vals),
        "full_mean": round(float(full_vals.mean()), 4),
        "subset_mean": round(float(sub_vals.mean()), 4),
        "t_stat": round(float(t_stat), 3),
        "pvalue": round(float(pvalue), 4),
        "cohens_d": round(float(d), 3),
    }


def main() -> int:
    args = parse_args()
    full_sample_path, subset_paths, results_dir = get_paths(args)

    print(f"Loading full sample from {full_sample_path}")
    full_df = load_df(full_sample_path)
    print(f"Full sample rows: {len(full_df)}")

    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    categorical_vars = [
        "gate_iv_arm",
        "compact_domain",
        "attacker_family",
        "defender_family",
        "engagement_binary",
        "deception_success_binary",
        "refusal_label_5way",
    ]
    continuous_vars = [
        "tactic_count_model_final",
        "context_document_word_count",
        "attacker_response_len_tokens",
        "attacker_capability_z",
        "defender_capability_z",
        "ScenarioComplexity_z",
        "refusal_confidence",
    ]

    for subset_name, subset_path in subset_paths.items():
        if not subset_path.exists():
            print(f"Skipping {subset_name}: file not found {subset_path}")
            continue
        sub_df = load_df(subset_path)
        print(f"\nAnalyzing {subset_name}: n={len(sub_df)}")

        for var in categorical_vars:
            if var not in full_df.columns or var not in sub_df.columns:
                continue
            result = categorical_comparison(full_df, sub_df, var, subset_name)
            all_results.append(result)
            sig = "*" if isinstance(result["pvalue"], float) and result["pvalue"] < 0.05 else ""
            print(f"  {var}: chi2={result['chi2']}, p={result['pvalue']}{sig}, Cramér's V={result['cramers_v']}")

        for var in continuous_vars:
            if var not in full_df.columns or var not in sub_df.columns:
                continue
            result = continuous_comparison(full_df, sub_df, var, subset_name)
            all_results.append(result)
            sig = "*" if isinstance(result["pvalue"], float) and result["pvalue"] < 0.05 else ""
            print(f"  {var}: mean_diff d={result['cohens_d']}, t={result['t_stat']}, p={result['pvalue']}{sig}")

    # Save structured results
    summary_csv = results_dir / "human_validation_representativeness.csv"
    summary_json = results_dir / "human_validation_representativeness.json"

    flat_rows = []
    for r in all_results:
        flat = {
            "subset": r["subset"],
            "variable": r["variable"],
            "variable_type": r["variable_type"],
            "full_n": r["full_n"],
            "subset_n": r["subset_n"],
            "test_stat": r.get("chi2") or r.get("t_stat"),
            "pvalue": r["pvalue"],
            "effect_size": r.get("cramers_v") or r.get("cohens_d"),
        }
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {summary_csv} and {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
