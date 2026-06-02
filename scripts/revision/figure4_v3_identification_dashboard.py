#!/usr/bin/env python3
"""Figure 4 v3: partial-identification dashboard with stratified bounds."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_style import (  # noqa: E402
    FULL_COL_IN,
    OKABE_ITO,
    apply_style,
    edge_for,
    fill_for,
    pastel,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_CSV = ROOT / "outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/analysis_rows.csv"
STRATIFIED_CSV = ROOT / "outputs/final_results_20260427/posthoc_partial_bounds_by_covariate.csv"
OUT_DIR = ROOT / "revision_supplementary/figures"
PAPER_FIG_DIR = ROOT / "paper/figures"
TACL_FIG_DIR = ROOT / "TACL_revised_package/figures"
for path in (OUT_DIR, PAPER_FIG_DIR, TACL_FIG_DIR):
    path.mkdir(parents=True, exist_ok=True)

GATES = [
    {
        "key": "standard_gate",
        "label": "Standard gate",
        "role": "standard_gate",
        "lower_num": 328,
        "eligible_added": 539,
        "upper_num": 867,
        "den": 993,
    },
    {
        "key": "safety_short_gate",
        "label": "Brief safety-framing",
        "role": "safety_short_gate",
        "lower_num": 40,
        "eligible_added": 898,
        "upper_num": 938,
        "den": 962,
    },
    {
        "key": "safety_policy_gate",
        "label": "Policy-framing",
        "role": "safety_policy_gate",
        "lower_num": 0,
        "eligible_added": 1000,
        "upper_num": 1000,
        "den": 1013,
    },
]
GENERATED_ONLY = {"num": 368, "den": 506}
AGGREGATE = {"lower_num": 368, "upper_num": 2805, "den": 2968, "eligible_added": 2437}

DOMAIN_LABELS = {
    "business_organizations": "Business",
    "governance_law_history": "Gov/law/history",
    "physical_infrastructure_environment": "Infrastructure/env.",
    "finance_markets": "Finance",
    "health_biomedicine": "Health/biomed.",
    "digital_information_systems": "Digital/info",
}


def verify_counts() -> None:
    df = pd.read_csv(ANALYSIS_CSV, usecols=["gate_iv_arm"], low_memory=False)
    counts = df["gate_iv_arm"].value_counts().to_dict()
    assert len(df) == 2968
    for gate in GATES:
        assert counts[gate["key"]] == gate["den"]


def draw_bounds_panel(ax: plt.Axes) -> None:
    ax.set_title("(a) Gate-arm identified sets", loc="left", pad=5)
    y_positions = np.arange(len(GATES))[::-1]
    for y_i, gate in zip(y_positions, GATES):
        lower = gate["lower_num"] / gate["den"]
        upper = gate["upper_num"] / gate["den"]
        edge = edge_for(gate["role"])
        ax.hlines(y_i, lower, upper, color=pastel(edge, 0.48), linewidth=10, zorder=1)
        ax.hlines(y_i, lower, upper, color=edge, linewidth=1.1, zorder=2)
        ax.scatter([lower], [y_i], s=34, color=fill_for(gate["role"]), edgecolor=edge, zorder=4)
        ax.vlines([upper], y_i - 0.16, y_i + 0.16, color=edge, linewidth=1.3, zorder=4)
        ax.text(lower, y_i + 0.20, f"{lower:.3f}", ha="center", va="bottom", fontsize=5.8)
        ax.text(upper, y_i + 0.20, f"{upper:.3f}", ha="center", va="bottom", fontsize=5.8)
        ax.text(
            (lower + upper) / 2,
            y_i - 0.22,
            f"+ {gate['eligible_added']:,} refused-eligible",
            ha="center",
            va="top",
            fontsize=5.4,
            color="#666666",
            style="italic",
        )
    generated_rate = GENERATED_ONLY["num"] / GENERATED_ONLY["den"]
    ax.axhline(-0.48, color="#BBBBBB", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    ax.scatter(
        [generated_rate],
        [-0.83],
        marker="D",
        s=44,
        facecolor="white",
        edgecolor=OKABE_ITO["grey"],
        linewidth=1.2,
        zorder=5,
    )
    ax.text(generated_rate, -0.62, "0.727\n(368/506)", ha="center", va="bottom", fontsize=5.6, color="#555555")
    ax.text(
        0.02,
        -0.83,
        "selected generated subset only",
        ha="left",
        va="center",
        fontsize=5.8,
        color="#555555",
    )
    ax.set_yticks(list(y_positions) + [-0.83])
    ax.set_yticklabels([gate["label"] for gate in GATES] + [""])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.25, 2.48)
    ax.set_xlabel("Full-denominator target-shift rate")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    style_axes(ax, grid_axis="x")


def draw_sensitivity_panel(ax: plt.Axes) -> None:
    ax.set_title("(b) Tipping-point sensitivity", loc="left", pad=5)
    pi = np.linspace(0, 1, 201)
    for gate in GATES:
        lower = gate["lower_num"] / gate["den"]
        slope = gate["eligible_added"] / gate["den"]
        y = lower + slope * pi
        color = edge_for(gate["role"])
        ax.plot(pi, y, color=color, linewidth=1.7, label=f"{gate['label']}: {lower:.3f}+{slope:.3f}pi")
        ax.scatter([0, 1], [lower, lower + slope], color=color, s=16, zorder=4)
    agg_lower = AGGREGATE["lower_num"] / AGGREGATE["den"]
    agg_slope = AGGREGATE["eligible_added"] / AGGREGATE["den"]
    ax.plot(
        pi,
        agg_lower + agg_slope * pi,
        color="#555555",
        linewidth=1.2,
        linestyle=(0, (3, 2)),
        label=f"Aggregate: {agg_lower:.3f}+{agg_slope:.3f}pi",
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Assumed shift probability among refused eligible rows")
    ax.set_ylabel("Implied rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="both")
    ax.legend(loc="lower right", fontsize=5.3, handlelength=1.1)


def draw_domain_bounds(ax: plt.Axes) -> None:
    ax.set_title("(c) Domain-stratified no-assumption bounds remain wide", loc="left", pad=5)
    bounds = pd.read_csv(STRATIFIED_CSV)
    dom = bounds[bounds["stratification"] == "domain"].copy()
    dom["label"] = dom["compact_domain"].map(DOMAIN_LABELS)
    dom = dom.sort_values("lower_bound", ascending=True)
    y = np.arange(len(dom))
    cmap = plt.get_cmap("Blues")
    widths = dom["bound_width"].to_numpy()
    width_norm = (widths - widths.min()) / (widths.max() - widths.min() + 1e-12)
    colors = [cmap(0.35 + 0.45 * val) for val in width_norm]
    for y_i, row, color in zip(y, dom.itertuples(index=False), colors):
        ax.hlines(y_i, row.lower_bound, row.upper_bound, color=color, linewidth=8, zorder=1)
        ax.hlines(y_i, row.lower_bound, row.upper_bound, color="#2B5B84", linewidth=0.8, zorder=2)
        ax.scatter([row.lower_bound], [y_i], s=24, color="#FFFFFF", edgecolor="#2B5B84", linewidth=1.0, zorder=3)
        ax.vlines([row.upper_bound], y_i - 0.15, y_i + 0.15, color="#2B5B84", linewidth=1.1, zorder=3)
        ax.text(
            0.985,
            y_i,
            f"n={int(row.n):,}; lower={row.lower_bound:.3f}; width={row.bound_width:.3f}",
            ha="right",
            va="center",
            fontsize=5.2,
            color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(dom["label"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Full-denominator target-shift rate")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="x")
    ax.text(
        0.02,
        0.98,
        "Darker strips indicate wider identified sets.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color="#555555",
    )


def make_figure() -> list[Path]:
    apply_style()
    verify_counts()
    fig = plt.figure(figsize=(FULL_COL_IN, 5.55))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        left=0.11,
        right=0.98,
        top=0.94,
        bottom=0.11,
        hspace=0.50,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    draw_bounds_panel(ax_a)
    draw_sensitivity_panel(ax_b)
    draw_domain_bounds(ax_c)

    handles = [
        mlines.Line2D([], [], color="#333333", marker="o", linewidth=1.0, markersize=4, label="circle = observed lower bound"),
        mlines.Line2D([], [], color="#333333", marker="|", linewidth=0, markersize=9, label="cap = logical upper bound"),
        mlines.Line2D([], [], color=OKABE_ITO["grey"], marker="D", markerfacecolor="white", linewidth=0, markersize=4, label="diamond = selected generated subset"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.015), ncol=3, fontsize=5.7, handletextpad=0.45)

    pdf_path = OUT_DIR / "figure4_v3_identification_dashboard.pdf"
    png_path = OUT_DIR / "figure4_v3_identification_dashboard.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    public_paths = [
        TACL_FIG_DIR / "figure4_identification_band.png",
        PAPER_FIG_DIR / "figure4_identification_band.png",
    ]
    for path in public_paths:
        shutil.copy2(png_path, path)
    return [pdf_path, png_path, *public_paths]


if __name__ == "__main__":
    for out in make_figure():
        print(f"wrote: {out}")
