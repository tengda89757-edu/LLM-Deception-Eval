#!/usr/bin/env python3
"""Validation audit forest plot with scenario-cluster bootstrap intervals."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
OUT_DIR = ROOT / "revision_supplementary/figures"
PAPER_FIG_DIR = ROOT / "paper/figures"
TACL_FIG_DIR = ROOT / "TACL-major revision/figures"
TACL_FINAL_FIG_DIR = ROOT / "TACL_revised_package_final (1)/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
TACL_FIG_DIR.mkdir(parents=True, exist_ok=True)
TACL_FINAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

ROWS = [
    {
        "label": "All generated interventions:\ntarget-aware shift",
        "num": 368,
        "den": 506,
        "ci": (0.667334087905017, 0.7785981321351675),
        "role": "conditional_shift",
        "note": "machine-parsed outcome",
    },
    {
        "label": "Validation generated subset:\ntarget shift",
        "num": 42,
        "den": 62,
        "ci": (0.5536785443578454, 0.7804567135945737),
        "role": "validation",
        "note": "human-validation sample",
    },
    {
        "label": "Validation generated subset:\nstrict-deception positive",
        "num": 12,
        "den": 62,
        "ci": (0.11433696420146605, 0.3085203175170827),
        "role": "realized",
        "note": "adjudicated strict codebook; kappa=0.403",
    },
    {
        "label": "Strict-positive validation rows:\ntarget shift",
        "num": 10,
        "den": 12,
        "ci": (0.5519636426153274, 0.9530358523851776),
        "role": "standard_gate",
        "note": "validation-sample decomposition",
    },
    {
        "label": "Validation generated subset:\nbroad misleading positive",
        "num": 20,
        "den": 62,
        "ci": (0.2195432864054263, 0.44632145564215464),
        "role": "aggregate",
        "note": "adjudicated broad codebook; kappa=0.040",
    },
    {
        "label": "Broad-positive validation rows:\ntarget shift",
        "num": 14,
        "den": 20,
        "ci": (0.4810232237710206, 0.854524726031006),
        "role": "validation",
        "note": "validation-sample decomposition",
    },
]


def make_figure() -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_COL_IN, 3.9))
    y = np.arange(len(ROWS))[::-1]

    for y_i, row in zip(y, ROWS):
        rate = row["num"] / row["den"]
        lo, hi = row["ci"]
        color = edge_for(row["role"])
        ax.errorbar(
            [rate],
            [y_i],
            xerr=[[max(0.0, rate - lo)], [max(0.0, hi - rate)]],
            fmt="o",
            ms=5.5,
            lw=1.2,
            capsize=2.2,
            color=color,
            markerfacecolor=pastel(color, 0.62),
            markeredgecolor=color,
            zorder=4,
        )
        ax.text(
            min(1.08, hi + 0.025),
            y_i,
            f"{row['num']}/{row['den']} = {100 * rate:.1f}%",
            ha="left",
            va="center",
            fontsize=6.5,
            color="#222222",
            clip_on=False,
        )
        ax.text(
            0.01,
            y_i - 0.31,
            row["note"],
            ha="left",
            va="center",
            fontsize=5.9,
            color="#666666",
            style="italic",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in ROWS])
    ax.set_xlim(0.0, 1.18)
    ax.set_ylim(-0.65, len(ROWS) - 0.35)
    ax.set_xlabel("Estimated proportion with 95% interval")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Validation audit quantities with 95% intervals", loc="left", pad=8)
    style_axes(ax, grid_axis="x")
    ax.axvline(1.0, color="#BBBBBB", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)

    paths = [
        OUT_DIR / "figure_validation_audit_forest.pdf",
        OUT_DIR / "figure_validation_audit_forest.png",
        PAPER_FIG_DIR / "figure_validation_audit_forest.png",
        TACL_FIG_DIR / "figure_validation_audit_forest.png",
        TACL_FINAL_FIG_DIR / "figure_validation_audit_forest.png",
    ]
    for path in paths:
        if path.suffix == ".png":
            fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.12)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return paths[0]


if __name__ == "__main__":
    out = make_figure()
    print(f"wrote: {out}")
    print(f"wrote: {out.with_suffix('.png')}")
    print(f"wrote: {PAPER_FIG_DIR / 'figure_validation_audit_forest.png'}")
    print(f"wrote: {TACL_FIG_DIR / 'figure_validation_audit_forest.png'}")
    print(f"wrote: {TACL_FINAL_FIG_DIR / 'figure_validation_audit_forest.png'}")
