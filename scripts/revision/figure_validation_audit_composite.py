#!/usr/bin/env python3
"""Validation-audit composite: balance, audit rates, and coder agreement."""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_style import (  # noqa: E402
    FULL_COL_IN,
    OKABE_ITO,
    apply_style,
    edge_for,
    pastel,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CSV = ROOT / "revision_supplementary/human_validation_v2/rerun_annotation_adjudicated.csv"
BALANCE_CSV = ROOT / "outputs/final_results_20260427/human_validation_representativeness.csv"
OUT_DIR = ROOT / "revision_supplementary/figures"
PAPER_FIG_DIR = ROOT / "paper/figures"
TACL_FIG_DIR = ROOT / "TACL_revised_package/figures"
for path in (OUT_DIR, PAPER_FIG_DIR, TACL_FIG_DIR):
    path.mkdir(parents=True, exist_ok=True)

AUDIT_ROWS = [
    ("All generated interventions:\ntarget-aware shift", 368, 506, "conditional_shift", "machine-parsed outcome"),
    ("Validation generated subset:\ntarget shift", 42, 62, "validation", "human-validation sample"),
    ("Validation generated subset:\nstrict-deception positive", 12, 62, "realized", "adjudicated strict codebook"),
    ("Strict-positive validation rows:\ntarget shift", 10, 12, "standard_gate", "validation-sample decomposition"),
    ("Validation generated subset:\nbroad misleading positive", 20, 62, "aggregate", "adjudicated broad codebook"),
    ("Broad-positive validation rows:\ntarget shift", 14, 20, "validation", "validation-sample decomposition"),
]

SUMMARY = {
    "strict_agreement": 53,
    "strict_n": 62,
    "strict_kappa": 0.4026,
    "broad_agreement": 25,
    "broad_n": 62,
    "broad_kappa": 0.0402,
}

VARIABLE_LABELS = {
    "gate_iv_arm": "Gate arm",
    "compact_domain": "Domain",
    "attacker_family": "Attacker family",
    "defender_family": "Defender family",
    "engagement_binary": "Generation",
    "deception_success_binary": "Target shift",
    "refusal_label_5way": "Refusal label",
    "tactic_count_model_final": "Tactic count",
    "context_document_word_count": "Context length",
    "attacker_response_len_tokens": "Attacker length",
    "attacker_capability_z": "Attacker capability",
    "defender_capability_z": "Defender capability",
    "ScenarioComplexity_z": "Scenario complexity",
    "refusal_confidence": "Refusal confidence",
}


def wilson_ci(num: int, den: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if den <= 0:
        return float("nan"), float("nan")
    p = num / den
    denom = 1 + z * z / den
    centre = (p + z * z / (2 * den)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * den)) / den) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def verify_validation(df: pd.DataFrame) -> None:
    assert len(df) == 62
    assert int(df["adjudicated_strict"].sum()) == 12
    assert int(df["adjudicated_broad"].sum()) == 20
    assert int(df["strict_agreement"].sum()) == 53
    assert int(df["broad_agreement"].sum()) == 25


def draw_audit_forest(ax: plt.Axes) -> None:
    ax.set_title("(a) Decision-shift and tactic-audit quantities", loc="left", pad=5)
    y = np.arange(len(AUDIT_ROWS))[::-1]
    for y_i, (label, num, den, role, note) in zip(y, AUDIT_ROWS):
        rate = num / den
        lo, hi = wilson_ci(num, den)
        color = edge_for(role)
        ax.errorbar(
            [rate],
            [y_i],
            xerr=[[rate - lo], [hi - rate]],
            fmt="o",
            ms=4.8,
            lw=1.1,
            capsize=2.0,
            color=color,
            markerfacecolor=pastel(color, 0.62),
            markeredgecolor=color,
            zorder=4,
        )
        ax.text(min(1.02, hi + 0.025), y_i, f"{num}/{den} = {100 * rate:.1f}%", ha="left", va="center", fontsize=5.8)
        ax.text(0.01, y_i - 0.31, note, ha="left", va="center", fontsize=5.4, color="#666666", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in AUDIT_ROWS])
    ax.set_xlim(0, 1.14)
    ax.set_ylim(-0.65, len(AUDIT_ROWS) - 0.35)
    ax.set_xlabel("Estimated proportion with Wilson 95% interval")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.axvline(1.0, color="#BBBBBB", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    style_axes(ax, grid_axis="x")


def draw_confusion_pair(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("(b) Independent pre-adjudication label agreement", loc="left", pad=5)
    ax.set_axis_off()
    cmap = LinearSegmentedColormap.from_list("agreement_blue", ["#F7FBFF", "#9ECAE1", OKABE_ITO["blue"]])
    configs = [
        ("Strict", "claude_strict", "gemini_strict", SUMMARY["strict_agreement"], SUMMARY["strict_n"], SUMMARY["strict_kappa"]),
        ("Broad", "claude_broad", "gemini_broad", SUMMARY["broad_agreement"], SUMMARY["broad_n"], SUMMARY["broad_kappa"]),
    ]
    for i, (title, c_col, g_col, agree, n, kappa) in enumerate(configs):
        left = 0.08 + i * 0.47
        sub_ax = ax.inset_axes([left, 0.22, 0.36, 0.58])
        tab = pd.crosstab(df[c_col], df[g_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        values = tab.to_numpy()
        sub_ax.imshow(values, cmap=cmap, vmin=0, vmax=max(1, values.max()))
        for r in range(2):
            for c in range(2):
                sub_ax.text(c, r, str(int(values[r, c])), ha="center", va="center", fontsize=7.0, fontweight="bold")
        sub_ax.set_xticks([0, 1])
        sub_ax.set_yticks([0, 1])
        sub_ax.set_xticklabels(["G=0", "G=1"], fontsize=5.5)
        sub_ax.set_yticklabels(["C=0", "C=1"], fontsize=5.5)
        sub_ax.tick_params(length=0)
        for spine in sub_ax.spines.values():
            spine.set_visible(False)
        sub_ax.set_title(f"{title}\nagree {agree}/{n}; kappa={kappa:.3f}", fontsize=6.2, pad=3)
    ax.text(0.08, 0.08, "Cells show Claude-code by Gemini-code binary counts before adjudication.", transform=ax.transAxes, ha="left", va="center", fontsize=5.5, color="#555555")


def draw_balance(ax: plt.Axes) -> None:
    ax.set_title("(c) N=360 validation-sample balance", loc="left", pad=5)
    balance = pd.read_csv(BALANCE_CSV).copy()
    balance["abs_effect"] = pd.to_numeric(balance["effect_size"], errors="coerce").abs()
    balance["label"] = balance["variable"].map(VARIABLE_LABELS).fillna(balance["variable"])
    balance = balance.sort_values("abs_effect", ascending=True)
    y = np.arange(len(balance))
    colors = [edge_for("validation") if t == "categorical" else edge_for("generation") for t in balance["variable_type"]]
    ax.barh(y, balance["abs_effect"], color=[pastel(c, 0.55) for c in colors], edgecolor=colors, linewidth=0.7)
    ax.axvline(0.10, color="#777777", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(0.102, len(balance) - 0.25, "0.10", ha="left", va="top", fontsize=5.4, color="#666666")
    ax.set_yticks(y)
    ax.set_yticklabels(balance["label"], fontsize=5.5)
    ax.set_xlim(0, max(0.14, float(balance["abs_effect"].max()) * 1.18))
    ax.set_xlabel("Absolute effect size: Cramer's V or Cohen's d")
    style_axes(ax, grid_axis="x")


def make_figure() -> list[Path]:
    apply_style()
    validation = pd.read_csv(VALIDATION_CSV)
    verify_validation(validation)

    fig = plt.figure(figsize=(FULL_COL_IN, 5.10))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.28, 1.0],
        height_ratios=[0.86, 1.14],
        left=0.12,
        right=0.985,
        top=0.94,
        bottom=0.11,
        hspace=0.42,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_audit_forest(ax_a)
    draw_confusion_pair(ax_b, validation)
    draw_balance(ax_c)

    fig.text(
        0.12,
        0.025,
        "Strict and broad tactic labels are validation-sample audit findings over 62 sampled generated interventions, not prevalence estimates over all generated rows.",
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )

    pdf_path = OUT_DIR / "figure_validation_audit_composite.pdf"
    png_path = OUT_DIR / "figure_validation_audit_composite.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    public_paths = [
        TACL_FIG_DIR / "figure_validation_audit_forest.png",
        PAPER_FIG_DIR / "figure_validation_audit_forest.png",
    ]
    for path in public_paths:
        shutil.copy2(png_path, path)
    return [pdf_path, png_path, *public_paths]


if __name__ == "__main__":
    for out in make_figure():
        print(f"wrote: {out}")
