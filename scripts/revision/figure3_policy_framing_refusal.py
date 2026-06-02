#!/usr/bin/env python3
"""Figure 3 — policy-framing gate refusal decomposition.

A single horizontal stacked bar (denom = 1,013) showing the machine-derived
rationale taxonomy under the safety-policy framing. By construction the gate
yields zero downstream target-aware outcomes; the figure shows the *availability*
collapse (censoring), not the absence of underlying susceptibility.

Inputs
------
* analysis_rows.csv  — TACL revision gate-IV run.

Outputs
-------
* figure3_policy_framing_refusal.pdf
* figure3_policy_framing_refusal.png  (600 dpi)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_style import (  # noqa: E402
    SINGLE_COL_IN,
    apply_style,
    edge_for,
    fill_for,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_CSV = ROOT / "outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/analysis_rows.csv"
OUT_DIR = ROOT / "revision_supplementary/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Spec: gate_pass=0, safety_refusal=971, task_conflict_refusal=36, ambiguous_refusal=6
# (REFUSE_AMBIGUOUS=4 + UNPARSEABLE=2 collapsed under "ambiguous_refusal").
SEGMENTS: list[dict] = [
    {"key": "gate_pass",             "count": 0,   "label": "Gate pass"},
    {"key": "safety_refusal",        "count": 971, "label": "Safety refusal"},
    {"key": "task_conflict_refusal", "count": 36,  "label": "Task-conflict refusal"},
    {"key": "ambiguous_refusal",     "count": 6,   "label": "Ambiguous / unparseable refusal"},
]
TOTAL = 1013
assert sum(s["count"] for s in SEGMENTS) == TOTAL


def verify_counts() -> None:
    df = pd.read_csv(ANALYSIS_CSV, low_memory=False)
    sg = df[df["gate_iv_arm"] == "safety_policy_gate"]
    assert len(sg) == TOTAL, f"safety_policy_gate rows {len(sg)} != {TOTAL}"
    decisions = sg["gate_decision"].value_counts(dropna=False).to_dict()
    assert decisions.get("REFUSE_SAFETY", 0) == 971
    assert decisions.get("REFUSE_TASK_CONFLICT", 0) == 36
    # 4 REFUSE_AMBIGUOUS + 2 UNPARSEABLE merged into ambiguous bucket = 6
    assert (decisions.get("REFUSE_AMBIGUOUS", 0)
            + decisions.get("UNPARSEABLE", 0)) == 6
    assert decisions.get("ENGAGE", 0) == 0


def make_figure() -> Path:
    apply_style()
    verify_counts()

    fig = plt.figure(figsize=(SINGLE_COL_IN * 2.0, 3.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.10)
    ax = fig.add_subplot(1, 1, 1)
    # Reserve a real left margin so caption / x-tick labels don't kiss the edge.
    ax.margins(x=0.0)

    # Draw the single stacked bar.
    bar_y = 0.0
    bar_h = 0.56
    cursor = 0
    # Track small-segment annotations so we can stagger their y-position.
    small_idx = 0
    small_y_levels = [bar_h / 2 + 0.42, bar_h / 2 + 1.05]  # alternating heights
    for seg in SEGMENTS:
        cnt = seg["count"]
        if cnt == 0:
            continue
        ax.barh(
            bar_y, cnt, left=cursor, height=bar_h,
            color=fill_for(seg["key"]), edgecolor=edge_for(seg["key"]),
            linewidth=0.9, zorder=3,
        )
        pct = 100.0 * cnt / TOTAL
        center = cursor + cnt / 2
        if cnt / TOTAL >= 0.05:
            ax.text(
                center, bar_y, f"{pct:.2f}%\n(n = {cnt:,})",
                ha="center", va="center", fontsize=6.6, color="#1A1A1A",
                zorder=4,
            )
        else:
            # Stagger small-segment leaders to avoid overlap.
            y_text = small_y_levels[small_idx % len(small_y_levels)]
            small_idx += 1
            ax.annotate(
                f"{seg['label']}\n{pct:.2f}%  (n = {cnt:,})",
                xy=(center, bar_y + bar_h / 2),
                xytext=(center, y_text),
                ha="center", va="bottom", fontsize=6.2, color="#222222",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666",
                                shrinkA=0, shrinkB=0),
            )
        cursor += cnt

    ax.set_xlim(0, TOTAL)
    ax.set_ylim(-0.65, 1.85)
    ax.set_yticks([])
    ax.set_xlabel(
        f"Rows under safety-policy gate (n = {TOTAL:,})\n"
        "Note: rationale categories are a machine-derived taxonomy;\n"
        "this shows generation availability / censoring, not zero underlying susceptibility."
    )

    style_axes(ax, grid_axis="x")
    ax2 = ax.twiny()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of gate (%)")
    ax2.tick_params(direction="out", length=2.2, width=0.6)
    for s in ax2.spines.values():
        s.set_linewidth(0.6); s.set_color("#222222")
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=fill_for(s["key"]),
                       edgecolor=edge_for(s["key"]),
                       linewidth=0.9,
                       label=f"{s['label']} (n = {s['count']:,}; {100*s['count']/TOTAL:.2f}%)")
        for s in SEGMENTS
    ]
    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.45),
        ncol=2, columnspacing=1.4, handletextpad=0.6,
    )

    pdf_path = OUT_DIR / "figure3_policy_framing_refusal.pdf"
    png_path = OUT_DIR / "figure3_policy_framing_refusal.png"
    # Force a generous pad on save so the leftmost tick / caption never clip.
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.30)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return pdf_path


if __name__ == "__main__":
    out = make_figure()
    print(f"wrote: {out}")
    print(f"wrote: {out.with_suffix('.png')}")
