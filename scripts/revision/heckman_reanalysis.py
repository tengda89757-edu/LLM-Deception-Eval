#!/usr/bin/env python3
"""TACL revision selection-model and IV robustness analysis.

The script treats a run-scoped ``analysis_rows.jsonl`` as the single source of
truth.  It supports the new gate-only IV design by default and keeps blocked
analyses explicit when required covariates or outcome variation are absent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.discrete.discrete_model import Logit, Probit


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from deception_benchmark.derive import ENGAGEMENT_NEGATIVE, ENGAGEMENT_POSITIVE, derive_outcomes  # noqa: E402


GATE_IV_COLUMNS = ("gate_iv_safety_short_gate", "gate_iv_safety_policy_gate")
PREFERRED_OUTCOME = "deception_success_binary"
MTD_COLUMN = "tactic_count_model_final"
BASE_PRE_CONTROLS = (
    "ScenarioComplexity_z",
    "attacker_capability_z",
    "defender_capability_z",
    "context_document_word_count_z",
)
STYLE_METRICS = (
    "attacker_style_len_words",
    "attacker_style_len_sentences",
    "attacker_style_hedging_markers",
    "attacker_style_aggressive_markers",
    "attacker_style_refusal_markers",
    "attacker_response_len_chars",
    "attacker_response_len_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TACL revision Heckman/IV robustness analyses.")
    parser.add_argument(
        "--analysis-rows",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "analysis_rows.jsonl",
        help="Canonical run-scoped analysis_rows JSONL/CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "revision_supplementary" / "results",
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--cluster-var",
        default="scenario_id",
        help="Cluster variable for robust SE where supported.",
    )
    parser.add_argument(
        "--strict-manual-covariates",
        action="store_true",
        help="Block primary models unless CCC/capability and scenario complexity are fully present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_analysis_rows(args.analysis_rows)
    df = prepare_dataframe(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = diagnostics_report(df, args)
    iv_specs = build_iv_specs(df)
    primary_spec = next((spec for spec in iv_specs if spec["name"] == "gate_joint"), iv_specs[0] if iv_specs else None)

    results: dict[str, Any] = {
        "schema_version": "tacl_revision_heckman_reanalysis_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_rows": str(args.analysis_rows),
        "diagnostics": diagnostics,
        "iv_specs": public_iv_specs(iv_specs, df),
        "balance_checks": balance_checks(df),
        "first_stage": {},
        "falsification": {},
        "style_contamination": {},
        "models": {},
        "partial_identification": partial_identification_bounds(df),
        "notes": [
            "Primary estimates are selection-adjusted under stated identification assumptions, not assumption-free corrections.",
            "Selection-stage controls are pre-treatment only; MTD enters outcome models because it is observed after engagement.",
        ],
    }

    if not primary_spec:
        results["status"] = "blocked"
        results["blocking_reason"] = "No usable IV columns found. Expected gate-only IV columns or legacy safety_prompt_strength."
        write_outputs(results, args.output_dir)
        return 2

    if args.strict_manual_covariates and diagnostics["manual_covariates"]["missing_any"]:
        results["status"] = "blocked"
        results["blocking_reason"] = "Manual CCC/capability or scenario-complexity covariates are incomplete."
        write_outputs(results, args.output_dir)
        return 2

    pre_controls = available_pre_controls(df)
    outcome_controls = available_outcome_controls(df, include_mtd=True)
    cluster_var = args.cluster_var if args.cluster_var in df.columns else None

    for spec in iv_specs:
        name = spec["name"]
        spec_df = df if spec.get("subset_mask") is None else df.loc[spec["subset_mask"]].copy()
        results["first_stage"][name] = fit_first_stage(spec_df, spec["iv_columns"], pre_controls, cluster_var)
        results["falsification"][name] = falsification_tests(spec_df, spec["iv_columns"], pre_controls, outcome_controls, cluster_var)

    results["style_contamination"] = style_contamination_tests(df, primary_spec["iv_columns"], pre_controls, cluster_var)
    results["models"]["naive_engaged_ols"] = fit_naive_outcome_model(df, outcome_controls, cluster_var)
    results["models"]["heckman_twostep"] = fit_heckman_twostep(df, primary_spec["iv_columns"], pre_controls, outcome_controls, cluster_var)
    results["models"]["two_stage_residual_inclusion"] = fit_2sri(df, primary_spec["iv_columns"], pre_controls, outcome_controls, cluster_var)
    results["models"]["ipw_outcome_model"] = fit_ipw_outcome_model(df, primary_spec["iv_columns"], pre_controls, outcome_controls, cluster_var)
    results["models"]["aipw_style_weighted_glm"] = fit_aipw_style_model(df, primary_spec["iv_columns"], pre_controls, outcome_controls, cluster_var)
    results["models"]["sample_selection_biprobit_fiml"] = fit_sample_selection_biprobit(df, primary_spec["iv_columns"], pre_controls)

    for spec in iv_specs:
        name = spec["name"]
        spec_df = df if spec.get("subset_mask") is None else df.loc[spec["subset_mask"]].copy()
        spec_pre_controls = available_pre_controls(spec_df)
        spec_outcome_controls = available_outcome_controls(spec_df, include_mtd=True)
        results["models"][f"heckman_twostep_{name}"] = fit_heckman_twostep(
            spec_df,
            spec["iv_columns"],
            spec_pre_controls,
            spec_outcome_controls,
            cluster_var if cluster_var in spec_df.columns else None,
        )

    results["sensitivity"] = sensitivity_summary(results)
    results["status"] = "ok_with_diagnostics" if diagnostics["manual_covariates"]["missing_any"] else "ok"
    write_outputs(results, args.output_dir)
    return 0


def load_analysis_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"analysis rows not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for _, row in df.iterrows():
        derived = derive_outcomes(row.to_dict())
        for key, value in derived.items():
            if key not in df.columns:
                df[key] = np.nan
            if pd.isna(df.at[row.name, key]) or df.at[row.name, key] == "":
                df.at[row.name, key] = value

    if "engagement_binary" not in df.columns:
        df["engagement_binary"] = np.nan
    if "refusal_label_5way" in df.columns:
        labels = df["refusal_label_5way"].astype(str)
        df.loc[labels.isin(ENGAGEMENT_POSITIVE), "engagement_binary"] = 1
        df.loc[labels.isin(ENGAGEMENT_NEGATIVE), "engagement_binary"] = 0

    for col in (
        "engagement_binary",
        "deception_success_binary",
        "control_shift_to_target",
        MTD_COLUMN,
        "context_document_word_count",
        "attacker_capability_score",
        "defender_capability_score",
        "attacker_capability_z",
        "defender_capability_z",
        "ScenarioComplexity_z",
        "ScenarioComplexity_raw",
        *GATE_IV_COLUMNS,
        "safety_prompt_strength",
        *STYLE_METRICS,
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "context_document_word_count_z" not in df.columns and "context_document_word_count" in df.columns:
        df["context_document_word_count_z"] = zscore(df["context_document_word_count"])
    if "mtd_z" not in df.columns and MTD_COLUMN in df.columns:
        df["mtd_z"] = zscore(df[MTD_COLUMN])
    if "deception_success_tri" not in df.columns:
        df["deception_success_tri"] = deception_success_tri(df)
    if "baseline_target_binary" not in df.columns:
        df["baseline_target_binary"] = option_equals(df, "baseline_option_id", "target_option_id")

    if "gate_iv_arm" in df.columns:
        df["gate_iv_any_safety"] = df["gate_iv_arm"].astype(str).isin({"safety_short_gate", "safety_policy_gate"}).astype(int)
        for arm in ("standard_gate", "safety_short_gate", "safety_policy_gate"):
            col = f"gate_iv_{arm}"
            if col not in df.columns:
                df[col] = (df["gate_iv_arm"].astype(str) == arm).astype(int)
    return df


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if values.notna().sum() < 2 or not std or pd.isna(std):
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / std


def deception_success_tri(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    engaged = pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1
    required = {"baseline_option_id", "deception_option_id", "target_option_id"}
    if not required.issubset(df.columns):
        return out
    out.loc[engaged] = 0.0
    pos = engaged & (df["baseline_option_id"] != df["target_option_id"]) & (df["deception_option_id"] == df["target_option_id"])
    neg = engaged & (df["baseline_option_id"] == df["target_option_id"]) & (df["deception_option_id"] != df["target_option_id"])
    out.loc[pos] = 1.0
    out.loc[neg] = -1.0
    return out


def option_equals(df: pd.DataFrame, left: str, right: str) -> pd.Series:
    if left not in df.columns or right not in df.columns:
        return pd.Series(np.nan, index=df.index)
    mask = df[left].notna() & df[right].notna() & (df[left] != "") & (df[right] != "")
    out = pd.Series(np.nan, index=df.index)
    out.loc[mask] = (df.loc[mask, left].astype(str) == df.loc[mask, right].astype(str)).astype(int)
    return out


def diagnostics_report(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    manual_cols = ("ScenarioComplexity_z", "attacker_capability_z", "defender_capability_z")
    manual_missing = {col: missing_count(df, col) for col in manual_cols}
    return {
        "row_count": int(len(df)),
        "engaged_rows": int((pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1).sum()),
        "refused_rows": int((pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 0).sum()),
        "outcome_nonmissing": int(pd.to_numeric(df.get(PREFERRED_OUTCOME), errors="coerce").notna().sum()) if PREFERRED_OUTCOME in df.columns else 0,
        "outcome_distribution_engaged": value_counts(df.loc[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1], PREFERRED_OUTCOME),
        "iv_columns_present": [col for col in (*GATE_IV_COLUMNS, "gate_iv_any_safety", "safety_prompt_strength") if col in df.columns],
        "iv_arm_counts": value_counts(df, "gate_iv_arm"),
        "manual_covariates": {
            "missing_counts": manual_missing,
            "missing_any": any(count > 0 for count in manual_missing.values()),
            "strict_required": bool(args.strict_manual_covariates),
        },
    }


def build_iv_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if all(col in df.columns for col in GATE_IV_COLUMNS):
        specs.append(
            {
                "name": "gate_joint",
                "iv_columns": list(GATE_IV_COLUMNS),
                "description": "Brief safety-framing and policy-framing gate arms relative to standard_gate.",
            }
        )
    if "gate_iv_any_safety" in df.columns:
        specs.append(
            {
                "name": "gate_any_safety_vs_standard",
                "iv_columns": ["gate_iv_any_safety"],
                "description": "Any safety gate arm relative to standard_gate.",
            }
        )
    if "gate_iv_arm" in df.columns and "gate_iv_safety_short_gate" in df.columns:
        specs.append(
            {
                "name": "gate_safety_short_vs_standard",
                "iv_columns": ["gate_iv_safety_short_gate"],
                "subset_mask": df["gate_iv_arm"].astype(str).isin({"standard_gate", "safety_short_gate"}),
                "description": "Brief safety-framing arm relative to standard_gate, excluding safety_policy_gate.",
            }
        )
    if "gate_iv_arm" in df.columns and "gate_iv_safety_policy_gate" in df.columns:
        specs.append(
            {
                "name": "gate_safety_policy_vs_standard",
                "iv_columns": ["gate_iv_safety_policy_gate"],
                "subset_mask": df["gate_iv_arm"].astype(str).isin({"standard_gate", "safety_policy_gate"}),
                "description": "Policy-framing arm relative to standard_gate, excluding safety_short_gate.",
            }
        )
    if "safety_prompt_strength" in df.columns and pd.to_numeric(df["safety_prompt_strength"], errors="coerce").notna().any():
        specs.append(
            {
                "name": "legacy_safety_prompt_strength",
                "iv_columns": ["safety_prompt_strength"],
                "description": "Historical safety prompt IV. Not the revised primary design.",
            }
        )
    return specs


def public_iv_specs(specs: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    public = []
    for spec in specs:
        mask = spec.get("subset_mask")
        public.append(
            {
                "name": spec["name"],
                "iv_columns": list(spec["iv_columns"]),
                "description": spec.get("description"),
                "subset_n": int(mask.sum()) if mask is not None else int(len(df)),
            }
        )
    return public


def available_pre_controls(df: pd.DataFrame) -> list[str]:
    controls = [col for col in BASE_PRE_CONTROLS if usable_numeric(df, col)]
    controls.extend(dummy_columns(df, "compact_domain", "domain"))
    controls.extend(dummy_columns(df, "attacker_family", "attacker"))
    controls.extend(dummy_columns(df, "defender_family", "defender"))
    return controls


def available_outcome_controls(df: pd.DataFrame, *, include_mtd: bool) -> list[str]:
    controls = available_pre_controls(df)
    if include_mtd and usable_numeric(df, "mtd_z"):
        return ["mtd_z", *controls]
    if include_mtd and usable_numeric(df, MTD_COLUMN):
        return [MTD_COLUMN, *controls]
    return controls


def dummy_columns(df: pd.DataFrame, source: str, prefix: str) -> list[str]:
    if source not in df.columns:
        return []
    dummy_prefix = f"__{prefix}_"
    existing = [col for col in df.columns if col.startswith(dummy_prefix)]
    if existing:
        return existing
    dummies = pd.get_dummies(df[source].fillna("<missing>").astype(str), prefix=dummy_prefix[:-1], drop_first=True)
    for col in dummies.columns:
        df[col] = dummies[col].astype(float)
    return list(dummies.columns)


def fit_first_stage(df: pd.DataFrame, iv_cols: list[str], controls: list[str], cluster_var: str | None) -> dict[str, Any]:
    fit = fit_binary_model(
        df,
        y_col="engagement_binary",
        x_cols=[*iv_cols, *controls],
        model_type="probit",
        cluster_var=cluster_var,
    )
    if fit["status"] != "ok":
        return fit
    fit["interpretation"] = "Instrument relevance check: gate-arm coefficients should predict engagement/refusal."
    return fit


def fit_naive_outcome_model(df: pd.DataFrame, controls: list[str], cluster_var: str | None) -> dict[str, Any]:
    engaged = df[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1].copy()
    return fit_linear_model(
        engaged,
        y_col="deception_success_tri",
        x_cols=controls,
        cluster_var=cluster_var,
        model_name="naive_engaged_ols",
    )


def fit_heckman_twostep(
    df: pd.DataFrame,
    iv_cols: list[str],
    pre_controls: list[str],
    outcome_controls: list[str],
    cluster_var: str | None,
) -> dict[str, Any]:
    stage1_data = model_matrix(df, "engagement_binary", [*iv_cols, *pre_controls])
    if stage1_data["status"] != "ok":
        return {"status": "blocked", "stage": "selection", "reason": stage1_data["reason"]}
    y1, x1, valid1 = stage1_data["y"], stage1_data["X"], stage1_data["valid_index"]
    if y1.nunique(dropna=True) < 2:
        return {"status": "blocked", "stage": "selection", "reason": "engagement has no variation"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probit_res = Probit(y1.astype(float), x1.astype(float)).fit(disp=0, maxiter=200)
    except Exception as exc:
        return {"status": "error", "stage": "selection", "reason": f"Probit fit failed: {exc}"}

    xb = pd.Series(np.asarray(x1 @ probit_res.params), index=x1.index)
    cdf = pd.Series(stats.norm.cdf(xb).clip(1e-8, 1 - 1e-8), index=x1.index)
    pdf = pd.Series(stats.norm.pdf(xb), index=x1.index)
    imr = pdf / cdf
    df2 = df.copy()
    df2.loc[valid1, "imr"] = imr
    engaged = df2[pd.to_numeric(df2.get("engagement_binary"), errors="coerce") == 1].copy()
    stage2 = fit_linear_model(
        engaged,
        y_col="deception_success_tri",
        x_cols=[*outcome_controls, "imr"],
        cluster_var=cluster_var,
        model_name="heckman_twostep_outcome",
    )
    return {
        "status": stage2["status"],
        "stage1_selection": summarize_fit(probit_res, "probit", len(y1)),
        "stage2_outcome": stage2,
        "imr_summary": summarize_series(imr),
    }


def fit_2sri(
    df: pd.DataFrame,
    iv_cols: list[str],
    pre_controls: list[str],
    outcome_controls: list[str],
    cluster_var: str | None,
) -> dict[str, Any]:
    stage1 = model_matrix(df, "engagement_binary", [*iv_cols, *pre_controls])
    if stage1["status"] != "ok":
        return {"status": "blocked", "stage": "selection", "reason": stage1["reason"]}
    y1, x1, valid1 = stage1["y"], stage1["X"], stage1["valid_index"]
    if y1.nunique(dropna=True) < 2:
        return {"status": "blocked", "stage": "selection", "reason": "engagement has no variation"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probit_res = Probit(y1.astype(float), x1.astype(float)).fit(disp=0, maxiter=200)
    except Exception as exc:
        return {"status": "error", "stage": "selection", "reason": f"Probit fit failed: {exc}"}
    pred = pd.Series(probit_res.predict(x1), index=x1.index).clip(1e-6, 1 - 1e-6)
    df2 = df.copy()
    df2.loc[valid1, "selection_residual"] = y1 - pred
    engaged = df2[pd.to_numeric(df2.get("engagement_binary"), errors="coerce") == 1].copy()
    outcome = fit_binary_model(
        engaged,
        y_col=PREFERRED_OUTCOME,
        x_cols=[*outcome_controls, "selection_residual"],
        model_type="logit",
        cluster_var=cluster_var,
    )
    return {
        "status": outcome["status"],
        "stage1_selection": summarize_fit(probit_res, "probit", len(y1)),
        "stage2_binary_outcome": outcome,
        "note": "2SRI residual-inclusion robustness for binary deception_success among engaged rows.",
    }


def fit_ipw_outcome_model(
    df: pd.DataFrame,
    iv_cols: list[str],
    pre_controls: list[str],
    outcome_controls: list[str],
    cluster_var: str | None,
) -> dict[str, Any]:
    ps = fitted_selection_probabilities(df, iv_cols, pre_controls)
    if ps["status"] != "ok":
        return ps
    df2 = df.copy()
    df2["selection_probability"] = ps["probability"]
    engaged = df2[pd.to_numeric(df2.get("engagement_binary"), errors="coerce") == 1].copy()
    engaged["ipw"] = 1.0 / pd.to_numeric(engaged["selection_probability"], errors="coerce").clip(0.02, 0.98)
    return fit_linear_model(
        engaged,
        y_col="deception_success_tri",
        x_cols=outcome_controls,
        cluster_var=cluster_var,
        model_name="ipw_weighted_ols",
        weights=engaged["ipw"],
    )


def fit_aipw_style_model(
    df: pd.DataFrame,
    iv_cols: list[str],
    pre_controls: list[str],
    outcome_controls: list[str],
    cluster_var: str | None,
) -> dict[str, Any]:
    ps = fitted_selection_probabilities(df, iv_cols, pre_controls)
    if ps["status"] != "ok":
        return ps
    engaged = df[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1].copy()
    base = fit_linear_model(
        engaged,
        y_col="deception_success_tri",
        x_cols=outcome_controls,
        cluster_var=cluster_var,
        model_name="aipw_style_outcome_model",
    )
    if base["status"] != "ok":
        return base
    return {
        **base,
        "note": (
            "AIPW-style robustness: outcome regression with selection-probability diagnostics. "
            "MTD is not randomized, so this is not a treatment-effect AIPW estimator."
        ),
        "selection_probability_summary": summarize_series(ps["probability"]),
    }


def fit_sample_selection_biprobit(df: pd.DataFrame, iv_cols: list[str], pre_controls: list[str]) -> dict[str, Any]:
    reduced_pre = [col for col in pre_controls if not col.startswith("__")][:3]
    outcome_cols = [col for col in (["mtd_z"] if usable_numeric(df, "mtd_z") else [MTD_COLUMN]) if col in df.columns]
    outcome_cols.extend(reduced_pre)
    sel = model_matrix(df, "engagement_binary", [*iv_cols, *reduced_pre])
    if sel["status"] != "ok":
        return {"status": "blocked", "reason": sel["reason"]}
    y_s, z, valid_s = sel["y"].astype(float), sel["X"].astype(float), sel["valid_index"]
    work = df.loc[valid_s].copy()
    y_o = pd.to_numeric(work.get(PREFERRED_OUTCOME), errors="coerce")
    selected = y_s == 1
    if selected.sum() < 20 or y_o[selected].nunique(dropna=True) < 2:
        return {"status": "blocked", "reason": "too few selected rows or no binary outcome variation"}
    x_data = model_matrix(work, PREFERRED_OUTCOME, outcome_cols, subset=selected)
    if x_data["status"] != "ok":
        return {"status": "blocked", "reason": x_data["reason"]}

    # Build outcome matrix for all rows using the selected-row columns and fill
    # missing post-selection MTD with 0 for unselected rows; their outcome is not
    # evaluated in the likelihood, but the matrix shape must be defined.
    x_all = pd.DataFrame(index=work.index)
    x_all["const"] = 1.0
    for col in outcome_cols:
        x_all[col] = pd.to_numeric(work.get(col), errors="coerce").fillna(0.0)
    y_o_all = y_o

    z_np = z.to_numpy(dtype=float)
    x_np = x_all[z.columns if False else x_all.columns].to_numpy(dtype=float)
    y_s_np = y_s.to_numpy(dtype=float)
    y_o_np = y_o_all.to_numpy(dtype=float)
    selected_np = selected.to_numpy(dtype=bool)
    k_z = z_np.shape[1]
    k_x = x_np.shape[1]

    def nll(params: np.ndarray) -> float:
        gamma = params[:k_z]
        beta = params[k_z : k_z + k_x]
        rho = math.tanh(params[-1])
        zg = z_np @ gamma
        xb = x_np @ beta
        p_select = stats.norm.cdf(zg).clip(1e-10, 1 - 1e-10)
        ll = np.zeros_like(zg)
        not_selected = ~selected_np
        ll[not_selected] = np.log((1 - p_select[not_selected]).clip(1e-10, 1.0))
        selected_idx = np.where(selected_np)[0]
        for i in selected_idx:
            joint11 = _bvn_cdf(zg[i], xb[i], rho)
            if y_o_np[i] >= 0.5:
                prob = joint11
            else:
                prob = p_select[i] - joint11
            ll[i] = np.log(float(np.clip(prob, 1e-10, 1.0)))
        return float(-ll.sum())

    init = np.zeros(k_z + k_x + 1, dtype=float)
    try:
        opt = optimize.minimize(nll, init, method="BFGS", options={"maxiter": 250, "disp": False})
    except Exception as exc:
        return {"status": "error", "reason": f"FIML optimization failed: {exc}"}
    params = opt.x
    beta = params[k_z : k_z + k_x]
    rho = math.tanh(params[-1])
    coef = {name: safe_float(value) for name, value in zip(x_all.columns, beta)}
    return {
        "status": "ok" if opt.success else "warning",
        "model": "sample_selection_biprobit_fiml",
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "n_obs": int(len(work)),
        "selected_obs": int(selected_np.sum()),
        "selection_columns": list(z.columns),
        "outcome_columns": list(x_all.columns),
        "outcome_coefficients": coef,
        "rho": safe_float(rho),
        "negative_log_likelihood": safe_float(opt.fun),
        "note": "Reduced-control bivariate probit FIML to keep optimization stable.",
    }


def fitted_selection_probabilities(df: pd.DataFrame, iv_cols: list[str], pre_controls: list[str]) -> dict[str, Any]:
    data = model_matrix(df, "engagement_binary", [*iv_cols, *pre_controls])
    if data["status"] != "ok":
        return {"status": "blocked", "reason": data["reason"]}
    y, x = data["y"], data["X"]
    if y.nunique(dropna=True) < 2:
        return {"status": "blocked", "reason": "engagement has no variation"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = Probit(y.astype(float), x.astype(float)).fit(disp=0, maxiter=200)
    except Exception as exc:
        return {"status": "error", "reason": f"selection probit failed: {exc}"}
    probs = pd.Series(np.nan, index=df.index, dtype=float)
    probs.loc[data["valid_index"]] = res.predict(x)
    return {"status": "ok", "probability": probs, "model": summarize_fit(res, "probit", len(y))}


def falsification_tests(
    df: pd.DataFrame,
    iv_cols: list[str],
    pre_controls: list[str],
    outcome_controls: list[str],
    cluster_var: str | None,
) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    tests["baseline_target_binary"] = fit_binary_model(
        df,
        y_col="baseline_target_binary",
        x_cols=[*iv_cols, *pre_controls],
        model_type="logit",
        cluster_var=cluster_var,
    )
    tests["control_shift_to_target"] = fit_binary_model(
        df,
        y_col="control_shift_to_target",
        x_cols=[*iv_cols, *pre_controls],
        model_type="logit",
        cluster_var=cluster_var,
    )
    engaged = df[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1].copy()
    tests["engaged_direct_effect_on_success"] = fit_binary_model(
        engaged,
        y_col=PREFERRED_OUTCOME,
        x_cols=[*iv_cols, *outcome_controls],
        model_type="logit",
        cluster_var=cluster_var,
    )
    return tests


def style_contamination_tests(df: pd.DataFrame, iv_cols: list[str], pre_controls: list[str], cluster_var: str | None) -> dict[str, Any]:
    engaged = df[pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1].copy()
    tests = {}
    for metric in STYLE_METRICS:
        if usable_numeric(engaged, metric):
            tests[metric] = fit_linear_model(
                engaged,
                y_col=metric,
                x_cols=[*iv_cols, *pre_controls],
                cluster_var=cluster_var,
                model_name=f"style_contamination_{metric}",
            )
    if not tests:
        return {"status": "blocked", "reason": "No usable style metric columns on engaged generated messages."}
    return tests


def balance_checks(df: pd.DataFrame) -> dict[str, Any]:
    if "gate_iv_arm" not in df.columns:
        return {"status": "blocked", "reason": "gate_iv_arm missing"}
    out: dict[str, Any] = {}
    for col in ("compact_domain", "scenario_id", "attacker_family", "defender_family", "target_option_id"):
        if col in df.columns:
            out[col] = categorical_balance(df, col)
    for col in ("context_document_word_count", "attacker_capability_z", "defender_capability_z", "ScenarioComplexity_z"):
        if usable_numeric(df, col):
            out[col] = continuous_balance(df, col)
    return out


def categorical_balance(df: pd.DataFrame, col: str) -> dict[str, Any]:
    table = pd.crosstab(df["gate_iv_arm"].astype(str), df[col].fillna("<missing>").astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"status": "blocked", "reason": "not enough categories"}
    chi2, pvalue, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n and min(table.shape) > 1 else np.nan
    return {
        "status": "ok",
        "test": "chi_square_independence",
        "n": int(n),
        "chi2": safe_float(chi2),
        "pvalue": safe_float(pvalue),
        "cramers_v": safe_float(cramers_v),
    }


def continuous_balance(df: pd.DataFrame, col: str) -> dict[str, Any]:
    groups = [
        pd.to_numeric(group[col], errors="coerce").dropna()
        for _, group in df.groupby(df["gate_iv_arm"].astype(str))
    ]
    groups = [group for group in groups if len(group) > 1]
    if len(groups) < 2:
        return {"status": "blocked", "reason": "not enough non-missing groups"}
    f_stat, pvalue = stats.f_oneway(*groups)
    means = df.groupby(df["gate_iv_arm"].astype(str))[col].mean().to_dict()
    return {
        "status": "ok",
        "test": "one_way_anova_by_gate_arm",
        "f_stat": safe_float(f_stat),
        "pvalue": safe_float(pvalue),
        "means": {str(k): safe_float(v) for k, v in means.items()},
    }


def partial_identification_bounds(df: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(df.get(PREFERRED_OUTCOME), errors="coerce")
    selected = pd.to_numeric(df.get("engagement_binary"), errors="coerce") == 1
    observed_successes = y[selected].fillna(0).sum()
    n = len(df)
    if n == 0:
        return {"status": "blocked", "reason": "empty dataframe"}
    refused_or_unobserved_mask = (~selected) | (selected & y.isna())
    refused_or_unobserved = int(refused_or_unobserved_mask.sum())
    if "baseline_target_binary" in df.columns:
        baseline_target = pd.to_numeric(df["baseline_target_binary"], errors="coerce") == 1
        # Under the target-shift estimand, rows that already choose the target
        # at baseline cannot become successes even if their censored D_i were T_i.
        eligible_unobserved = int((refused_or_unobserved_mask & ~baseline_target).sum())
        ineligible_baseline_target = int((refused_or_unobserved_mask & baseline_target).sum())
    else:
        eligible_unobserved = refused_or_unobserved
        ineligible_baseline_target = 0
    lower = observed_successes / n
    upper = (observed_successes + eligible_unobserved) / n
    return {
        "status": "ok",
        "estimand": "overall target-shift deception success rate under worst/best-case outcomes for refused or missing rows",
        "n": int(n),
        "observed_selected_successes": safe_float(observed_successes),
        "unobserved_or_refused_rows": refused_or_unobserved,
        "eligible_unobserved_or_refused_rows": eligible_unobserved,
        "ineligible_baseline_target_unobserved_rows": ineligible_baseline_target,
        "lower_bound": safe_float(lower),
        "upper_bound": safe_float(upper),
    }


def sensitivity_summary(results: dict[str, Any]) -> dict[str, Any]:
    heckman = results.get("models", {}).get("heckman_twostep", {})
    stage2 = heckman.get("stage2_outcome", {}) if isinstance(heckman, dict) else {}
    mtd_coef = coefficient(stage2, "mtd_z") or coefficient(stage2, MTD_COLUMN)
    direct = results.get("falsification", {}).get("gate_joint", {}).get("engaged_direct_effect_on_success", {})
    direct_iv_coefs = {
        name: coefficient(direct, name)
        for name in GATE_IV_COLUMNS
        if coefficient(direct, name) is not None
    }
    if mtd_coef is None:
        return {"status": "blocked", "reason": "No MTD coefficient available from Heckman stage 2."}
    return {
        "status": "ok",
        "mtd_coefficient": safe_float(mtd_coef),
        "direct_iv_coefficients_engaged_falsification": {k: safe_float(v) for k, v in direct_iv_coefs.items()},
        "direct_effect_tipping_point_outcome_units": safe_float(abs(mtd_coef)),
        "interpretation": (
            "A direct IV effect on the outcome of approximately this magnitude, aligned against the MTD coefficient, "
            "would be needed to mechanically move the reported MTD coefficient to zero in this linear sensitivity summary."
        ),
    }


def fit_binary_model(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: list[str],
    model_type: str,
    cluster_var: str | None,
) -> dict[str, Any]:
    data = model_matrix(df, y_col, x_cols)
    if data["status"] != "ok":
        return {"status": "blocked", "reason": data["reason"], "model": model_type}
    y, x = data["y"], data["X"]
    if y.nunique(dropna=True) < 2:
        return {"status": "blocked", "reason": f"{y_col} has no variation", "model": model_type, "n_obs": int(len(y))}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Probit(y.astype(float), x.astype(float)) if model_type == "probit" else Logit(y.astype(float), x.astype(float))
            if cluster_var and cluster_var in df.columns:
                clusters = df.loc[data["valid_index"], cluster_var].astype(str)
                if clusters.nunique() > 1:
                    res = model.fit(
                        disp=0,
                        maxiter=200,
                        cov_type="cluster",
                        cov_kwds={"groups": clusters},
                    )
                else:
                    res = model.fit(disp=0, maxiter=200)
            else:
                res = model.fit(disp=0, maxiter=200)
    except Exception as exc:
        return {"status": "error", "reason": f"{model_type} failed: {exc}", "n_obs": int(len(y))}
    return summarize_fit(res, model_type, len(y))


def fit_linear_model(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_cols: list[str],
    cluster_var: str | None,
    model_name: str,
    weights: pd.Series | None = None,
) -> dict[str, Any]:
    data = model_matrix(df, y_col, x_cols)
    if data["status"] != "ok":
        return {"status": "blocked", "reason": data["reason"], "model": model_name}
    y, x = data["y"].astype(float), data["X"].astype(float)
    if y.nunique(dropna=True) < 2:
        return {"status": "blocked", "reason": f"{y_col} has no variation", "model": model_name, "n_obs": int(len(y))}
    try:
        if weights is not None:
            w = pd.to_numeric(weights.loc[data["valid_index"]], errors="coerce").astype(float)
            res = sm.WLS(y, x, weights=w).fit()
        else:
            res = sm.OLS(y, x).fit()
        if cluster_var and cluster_var in df.columns:
            clusters = df.loc[data["valid_index"], cluster_var].astype(str)
            if clusters.nunique() > 1:
                res = res.get_robustcov_results(cov_type="cluster", groups=clusters)
    except Exception as exc:
        return {"status": "error", "reason": f"OLS/WLS failed: {exc}", "n_obs": int(len(y))}
    summary = summarize_fit(res, model_name, len(y))
    summary["r2"] = safe_float(getattr(res, "rsquared", np.nan))
    return summary


def model_matrix(df: pd.DataFrame, y_col: str, x_cols: list[str], subset: pd.Series | None = None) -> dict[str, Any]:
    if y_col not in df.columns:
        return {"status": "blocked", "reason": f"{y_col} missing"}
    cols = [col for col in x_cols if col in df.columns and usable_numeric(df, col)]
    if not cols:
        return {"status": "blocked", "reason": "no usable RHS columns"}
    x = pd.DataFrame({"const": 1.0}, index=df.index)
    for col in cols:
        x[col] = pd.to_numeric(df[col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = y.notna() & x.notna().all(axis=1)
    if subset is not None:
        valid = valid & subset.reindex(df.index).fillna(False)
    if valid.sum() < max(10, len(cols) + 3):
        return {"status": "blocked", "reason": f"too few valid observations ({int(valid.sum())}) for {len(cols)} RHS columns"}
    x_valid = x.loc[valid]
    # Drop constant or duplicate columns that can destabilize models.
    nunique = x_valid.nunique(dropna=True)
    drop = [col for col in x_valid.columns if col != "const" and nunique[col] < 2]
    if drop:
        x_valid = x_valid.drop(columns=drop)
    if x_valid.shape[1] < 2:
        return {"status": "blocked", "reason": "only intercept remains after dropping constant RHS columns"}
    return {"status": "ok", "y": y.loc[valid], "X": x_valid, "valid_index": x_valid.index}


def summarize_fit(res: Any, model: str, n_obs: int) -> dict[str, Any]:
    params = _series_from_result_attr(res, "params")
    bse = _series_from_result_attr(res, "bse", index=params.index)
    pvalues = _series_from_result_attr(res, "pvalues", index=params.index)
    return {
        "status": "ok",
        "model": model,
        "n_obs": int(n_obs),
        "coefficients": {
            str(name): {
                "coef": safe_float(params.loc[name]),
                "se": safe_float(bse.loc[name]) if name in bse.index else None,
                "pvalue": safe_float(pvalues.loc[name]) if name in pvalues.index else None,
            }
            for name in params.index
        },
        "llf": safe_float(getattr(res, "llf", np.nan)),
        "aic": safe_float(getattr(res, "aic", np.nan)),
        "bic": safe_float(getattr(res, "bic", np.nan)),
    }


def _series_from_result_attr(res: Any, attr: str, index: pd.Index | None = None) -> pd.Series:
    value = getattr(res, attr)
    if isinstance(value, pd.Series):
        return value
    labels = index
    if labels is None:
        labels = getattr(getattr(res, "model", None), "exog_names", None)
    if labels is None:
        labels = [f"x{i}" for i in range(len(value))]
    return pd.Series(np.asarray(value), index=labels)


def coefficient(model_result: dict[str, Any], name: str) -> float | None:
    try:
        value = model_result["coefficients"][name]["coef"]
    except Exception:
        return None
    return None if value is None else float(value)


def usable_numeric(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    values = pd.to_numeric(df[col], errors="coerce")
    return bool(values.notna().sum() > 0 and values.nunique(dropna=True) > 1)


def missing_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return int(len(df))
    values = df[col]
    return int((values.isna() | (values.astype(str) == "")).sum())


def value_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = df[col].fillna("<missing>").astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def summarize_series(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": safe_float(values.mean()),
        "sd": safe_float(values.std(ddof=1)),
        "min": safe_float(values.min()),
        "max": safe_float(values.max()),
    }


def _bvn_cdf(a: float, b: float, rho: float) -> float:
    return float(stats.multivariate_normal.cdf([a, b], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]))


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return round(out, 6)


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "heckman_reanalysis.json"
    csv_path = output_dir / "model_coefficient_summary.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(flatten_coefficients(results)).to_csv(csv_path, index=False)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def flatten_coefficients(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(path: str, payload: Any) -> None:
        if isinstance(payload, dict) and "coefficients" in payload:
            for name, item in payload["coefficients"].items():
                rows.append(
                    {
                        "path": path,
                        "model": payload.get("model"),
                        "n_obs": payload.get("n_obs"),
                        "term": name,
                        "coef": item.get("coef"),
                        "se": item.get("se"),
                        "pvalue": item.get("pvalue"),
                    }
                )
        elif isinstance(payload, dict):
            for key, value in payload.items():
                visit(f"{path}.{key}" if path else key, value)

    visit("", results)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
