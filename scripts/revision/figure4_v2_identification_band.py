#!/usr/bin/env python3
"""Figure 4: partial-identification bounds and sensitivity curves."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

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
OUT_DIR = ROOT / "revision_supplementary/figures"
PAPER_FIG_DIR = ROOT / "paper/figures"
TACL_FIG_DIR = ROOT / "TACL-major revision/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
TACL_FIG_DIR.mkdir(parents=True, exist_ok=True)

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
        "label": "Brief safety-framing gate",
        "role": "safety_short_gate",
        "lower_num": 40,
        "eligible_added": 898,
        "upper_num": 938,
        "den": 962,
    },
    {
        "key": "safety_policy_gate",
        "label": "Policy-framing gate",
        "role": "safety_policy_gate",
        "lower_num": 0,
        "eligible_added": 1000,
        "upper_num": 1000,
        "den": 1013,
    },
]
GENERATED_ONLY = {"num": 368, "den": 506}


def verify_counts() -> None:
    df = pd.read_csv(ANALYSIS_CSV, low_memory=False)
    counts = df["gate_iv_arm"].value_counts().to_dict()
    assert len(df) == 2968
    for gate in GATES:
        assert counts[gate["key"]] == gate["den"]


def draw_bounds_panel(ax: plt.Axes) -> None:
    ax.set_title(
        "(a) Partial-identification bounds,\nnot statistical confidence intervals",
        loc="left",
        pad=6,
    )
    y_positions = np.array([2.0, 1.0, 0.0])
    for y, gate in zip(y_positions, GATES):
        lower = gate["lower_num"] / gate["den"]
        upper = gate["upper_num"] / gate["den"]
        edge = edge_for(gate["role"])
        ax.hlines(y, lower, upper, color=pastel(edge, 0.48), linewidth=12, zorder=1)
        ax.hlines(y, lower, upper, color=edge, linewidth=1.2, zorder=2)
        ax.scatter([lower], [y], s=42, color=fill_for(gate["role"]), edgecolor=edge, zorder=4)
        ax.vlines([upper], y - 0.18, y + 0.18, color=edge, linewidth=1.4, zorder=4)
        ax.text(
            lower,
            y + 0.23,
            f"{lower:.3f}\n({gate['lower_num']:,}/{gate['den']:,})",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color="#222222",
        )
        ax.text(
            upper,
            y + 0.23,
            f"{upper:.3f}\n({gate['upper_num']:,}/{gate['den']:,})",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color="#222222",
        )
        ax.text(
            (lower + upper) / 2,
            y - 0.23,
            f"refused eligible added = {gate['eligible_added']:,}",
            ha="center",
            va="top",
            fontsize=5.8,
            color="#666666",
            style="italic",
        )

    # Deliberately separated selected generated-subset marker.
    generated_rate = GENERATED_ONLY["num"] / GENERATED_ONLY["den"]
    generated_y = -0.90
    ax.axhline(-0.45, color="#BBBBBB", linewidth=0.7, linestyle=(0, (3, 3)), zorder=0)
    ax.scatter(
        [generated_rate],
        [generated_y],
        marker="D",
        s=56,
        facecolor="white",
        edgecolor=OKABE_ITO["grey"],
        linewidth=1.3,
        zorder=5,
    )
    ax.text(
        generated_rate,
        generated_y + 0.22,
        f"{generated_rate:.3f}\n({GENERATED_ONLY['num']:,}/{GENERATED_ONLY['den']:,})",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#555555",
    )
    ax.text(
        0.02,
        generated_y,
        "Selected generated subset only\n"
        "denominator = 506; not a full-sample bound",
        ha="left",
        va="center",
        fontsize=6.2,
        color="#555555",
    )

    ax.set_yticks(list(y_positions) + [generated_y])
    ax.set_yticklabels([gate["label"] for gate in GATES] + [""])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.35, 2.58)
    ax.set_xlabel("Full-denominator target-shift rate")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    style_axes(ax, grid_axis="x")


def draw_sensitivity_panel(ax: plt.Axes) -> None:
    ax.set_title("(b) Sensitivity to refused-row shift probability", loc="left", pad=6)
    pi = np.linspace(0, 1, 201)
    for gate in GATES:
        lower = gate["lower_num"] / gate["den"]
        slope = gate["eligible_added"] / gate["den"]
        y = lower + slope * pi
        color = edge_for(gate["role"])
        ax.plot(
            pi,
            y,
            color=color,
            linewidth=1.8,
            label=f"{gate['label']}: {lower:.3f} + {slope:.3f}$\\pi$",
        )
        ax.scatter([0, 1], [lower, lower + slope], color=color, s=18, zorder=4)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("$\\pi$: assumed shift probability among refused eligible rows")
    ax.set_ylabel("Implied target-shift rate")
    style_axes(ax, grid_axis="both")
    ax.legend(loc="lower right", fontsize=6.1, handlelength=1.2)
    ax.text(
        0.03,
        0.97,
        "The width is the point: refused-censored rows keep\n"
        "the forced-generation estimand weakly identified.",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#555555",
    )


def make_figure() -> Path:
    apply_style()
    verify_counts()
    fig = plt.figure(figsize=(FULL_COL_IN, 4.25))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        left=0.12,
        right=0.985,
        top=0.87,
        bottom=0.18,
        wspace=0.30,
    )
    ax_bounds = fig.add_subplot(grid[0, 0])
    ax_sens = fig.add_subplot(grid[0, 1])
    draw_bounds_panel(ax_bounds)
    draw_sensitivity_panel(ax_sens)

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="#333333",
            marker="o",
            linewidth=1.1,
            markersize=5,
            label="circle = observed lower bound",
        ),
        mlines.Line2D(
            [],
            [],
            color="#333333",
            marker="|",
            linewidth=0,
            markersize=10,
            label="cap = no-assumption upper bound",
        ),
        mlines.Line2D(
            [],
            [],
            color=OKABE_ITO["grey"],
            marker="D",
            markerfacecolor="white",
            linewidth=0,
            markersize=5,
            label="grey diamond = selected generated subset only",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.50, 0.02),
        fontsize=6.2,
        handletextpad=0.5,
    )

    paths = [
        OUT_DIR / "figure4_v2_identification_band.pdf",
        OUT_DIR / "figure4_v2_identification_band.png",
        PAPER_FIG_DIR / "figure4_identification_band.png",
        TACL_FIG_DIR / "figure4_identification_band.png",
    ]
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.12)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for out in make_figure():
        print(f"wrote: {out}")
