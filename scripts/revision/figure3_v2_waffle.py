#!/usr/bin/env python3
"""Figure 3 v2 — waffle / dot-grid alternative.

One small square per row (n = 1,013) under the safety-policy gate, coloured
by machine-derived rationale category. Cells are laid out in a 30×34 grid
with 7 padding cells. The waffle makes it visually impossible to confuse the
overwhelming dominance of safety_refusal (971 cells, vermillion) with the
two thin trailing categories (task-conflict, ambiguous), and gives an
exact, countable representation of the censoring footprint.

Inputs
------
* analysis_rows.csv  — gate-IV run.

Outputs
-------
* figure3_v2_waffle.pdf
* figure3_v2_waffle.png  (600 dpi)
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
)

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_CSV = ROOT / "outputs/runs/tacl_revision_gate_iv_20260417_ai_completed/analysis_rows.csv"
OUT_DIR = ROOT / "revision_supplementary/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTS = [
    {"key": "gate_pass",             "count": 0,   "label": "Gate pass"},
    {"key": "safety_refusal",        "count": 971, "label": "Safety refusal"},
    {"key": "task_conflict_refusal", "count": 36,  "label": "Task-conflict refusal"},
    {"key": "ambiguous_refusal",     "count": 6,   "label": "Ambiguous / unparseable refusal"},
]
TOTAL = 1013
N_COLS = 38   # 38 × 27 = 1026 → 13 padding cells
N_ROWS = 27
assert N_COLS * N_ROWS >= TOTAL


def verify_counts() -> None:
    df = pd.read_csv(ANALYSIS_CSV, low_memory=False)
    sg = df[df["gate_iv_arm"] == "safety_policy_gate"]
    assert len(sg) == TOTAL
    decisions = sg["gate_decision"].value_counts(dropna=False).to_dict()
    assert decisions.get("REFUSE_SAFETY", 0) == 971
    assert decisions.get("REFUSE_TASK_CONFLICT", 0) == 36
    assert decisions.get("REFUSE_AMBIGUOUS", 0) + decisions.get("UNPARSEABLE", 0) == 6
    assert decisions.get("ENGAGE", 0) == 0


def make_figure() -> Path:
    apply_style()
    verify_counts()

    # Build a flat list of category keys row-major over TOTAL cells, plus padding.
    cell_keys: list[str | None] = []
    for seg in SEGMENTS:
        cell_keys.extend([seg["key"]] * seg["count"])
    while len(cell_keys) < N_COLS * N_ROWS:
        cell_keys.append(None)

    cell_size = 0.13   # inches per cell — physical waffle scale
    fig_w = N_COLS * cell_size + 0.7
    fig_h = N_ROWS * cell_size + 1.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.04, 0.18, 0.92, 0.74])

    # Draw cells.
    for idx, key in enumerate(cell_keys):
        col = idx % N_COLS
        row = idx // N_COLS
        x = col
        y = N_ROWS - 1 - row  # top-down filling
        if key is None:
            # Padding cell: light dotted outline so the grid stays readable.
            rect = mpatches.Rectangle(
                (x + 0.06, y + 0.06), 0.88, 0.88,
                facecolor="#FFFFFF", edgecolor="#DDDDDD",
                linewidth=0.4, linestyle=(0, (1.5, 1.5)), zorder=2,
            )
        else:
            rect = mpatches.Rectangle(
                (x + 0.06, y + 0.06), 0.88, 0.88,
                facecolor=fill_for(key), edgecolor=edge_for(key),
                linewidth=0.35, zorder=3,
            )
        ax.add_patch(rect)

    ax.set_xlim(0, N_COLS)
    ax.set_ylim(0, N_ROWS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Inline annotations for the small categories — leader lines pointing at the centroid
    # of their cells (which sit in the bottom-right of the grid).
    cum = SEGMENTS[1]["count"]  # 971 cells filled with safety_refusal
    # task-conflict centroid
    tc_centre_idx = cum + SEGMENTS[2]["count"] / 2
    tc_col = tc_centre_idx % N_COLS
    tc_row = tc_centre_idx // N_COLS
    tc_x, tc_y = tc_col, N_ROWS - 1 - tc_row
    ax.annotate(
        f"Task-conflict refusal\n{SEGMENTS[2]['count']:,}  (3.55%)",
        xy=(tc_x + 0.5, tc_y + 0.5), xytext=(N_COLS + 0.6, tc_y + 1.5),
        ha="left", va="center", fontsize=6.4, color=edge_for("task_conflict_refusal"),
        arrowprops=dict(arrowstyle="-", lw=0.5,
                        color=edge_for("task_conflict_refusal"), shrinkA=0, shrinkB=2),
    )
    cum += SEGMENTS[2]["count"]  # 1007
    am_centre_idx = cum + SEGMENTS[3]["count"] / 2
    am_col = am_centre_idx % N_COLS
    am_row = am_centre_idx // N_COLS
    am_x, am_y = am_col, N_ROWS - 1 - am_row
    ax.annotate(
        f"Ambiguous / unparseable\n{SEGMENTS[3]['count']:,}  (0.59%)",
        xy=(am_x + 0.5, am_y + 0.5), xytext=(N_COLS + 0.6, am_y - 0.2),
        ha="left", va="center", fontsize=6.4, color=edge_for("ambiguous_refusal"),
        arrowprops=dict(arrowstyle="-", lw=0.5,
                        color=edge_for("ambiguous_refusal"), shrinkA=0, shrinkB=2),
    )
    # Big safety_refusal block label centred on the field
    ax.text(
        N_COLS / 2, N_ROWS / 2 + 1.0,
        "Safety refusal\n971 of 1,013\n(95.85 %)",
        ha="center", va="center", fontsize=10.5, color="#3A1A0A",
        fontweight="bold", zorder=5,
    )

    # Legend
    handles = [
        mpatches.Patch(facecolor=fill_for(s["key"]),
                       edgecolor=edge_for(s["key"]),
                       linewidth=0.6,
                       label=f"{s['label']}  (n = {s['count']:,}; {100*s['count']/TOTAL:.2f}%)")
        for s in SEGMENTS
    ]
    handles.append(mpatches.Patch(
        facecolor="#FFFFFF", edgecolor="#DDDDDD", linewidth=0.4,
        label=f"Grid padding (n = {N_COLS * N_ROWS - TOTAL})",
    ))
    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.03),
        ncol=2, columnspacing=1.4, handletextpad=0.6, fontsize=6.6,
    )

    # Top-of-figure subtitle
    ax.text(
        0, N_ROWS + 0.6,
        f"Each colored cell = one row under the policy-framing gate (n = {TOTAL:,}). "
        "Cells are filled row-major in category order; uncolored cells are grid padding.",
        ha="left", va="bottom", fontsize=6.6, color="#555555", style="italic",
    )

    # Bottom caption
    fig.text(
        0.04, 0.04,
        "Note: rationale categories are a machine-derived taxonomy; the waffle visualises "
        "generation availability / censoring, not zero underlying susceptibility.",
        ha="left", va="bottom", fontsize=6.2, color="#444444",
    )

    pdf_path = OUT_DIR / "figure3_v2_waffle.pdf"
    png_path = OUT_DIR / "figure3_v2_waffle.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.30)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    return pdf_path


if __name__ == "__main__":
    out = make_figure()
    print(f"wrote: {out}")
    print(f"wrote: {out.with_suffix('.png')}")
