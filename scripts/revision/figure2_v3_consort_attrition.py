#!/usr/bin/env python3
"""Figure 2: hash-randomized gate-framing attrition and rates.

Panel A shows only the gate-framing attrition path. Panel B visualizes the
three gate-arm rates requested for the main text: generation availability,
realized target-shift exposure, and conditional target-aware shift among
generated interventions. No selection-adjusted or validation-audit quantities
are embedded in this figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

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

TOTAL_ROWS = 2968
GATES = [
    {
        "key": "standard_gate",
        "label": "Standard gate",
        "role": "standard_gate",
        "n": 993,
        "generated": 454,
        "shifted": 328,
    },
    {
        "key": "safety_short_gate",
        "label": "Brief safety-framing gate",
        "role": "safety_short_gate",
        "n": 962,
        "generated": 52,
        "shifted": 40,
    },
    {
        "key": "safety_policy_gate",
        "label": "Policy-framing gate",
        "role": "safety_policy_gate",
        "n": 1013,
        "generated": 0,
        "shifted": 0,
    },
]

BOOTSTRAP_CI = {
    ("standard_gate", "generated"): (0.39402609361877683, 0.5228011570028398),
    ("standard_gate", "shifted"): (0.27007632122113956, 0.39570923091830734),
    ("standard_gate", "conditional"): (0.6635502945475483, 0.7728285077951003),
    ("safety_short_gate", "generated"): (0.023036046791824576, 0.09487759860004474),
    ("safety_short_gate", "shifted"): (0.014996944739638684, 0.07951246457562801),
    ("safety_short_gate", "conditional"): (0.575, 0.8974358974358975),
    ("safety_policy_gate", "generated"): (0.0, 0.0),
    ("safety_policy_gate", "shifted"): (0.0, 0.0),
}


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%"


def verify_counts() -> None:
    df = pd.read_csv(ANALYSIS_CSV, low_memory=False)
    counts = df["gate_iv_arm"].value_counts().to_dict()
    generated = df.groupby("gate_iv_arm")["engagement_binary"].sum().fillna(0).astype(int)
    shifted = df.groupby("gate_iv_arm")["deception_success_binary"].sum().fillna(0).astype(int)
    assert len(df) == TOTAL_ROWS
    for gate in GATES:
        key = gate["key"]
        assert counts[key] == gate["n"]
        assert int(generated[key]) == gate["generated"]
        assert int(shifted[key]) == gate["shifted"]


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    body: str,
    fc,
    ec,
    title_size: float = 7.0,
    body_size: float = 6.2,
    title_color: str = "#1A1A1A",
    body_color: str = "#222222",
) -> None:
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.8,
        facecolor=fc,
        edgecolor=ec,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y + h * 0.18,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    ax.text(
        x,
        y - h * 0.22,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=body_color,
        zorder=4,
    )


def add_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>,head_length=4,head_width=2.6",
        linewidth=0.65,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("(a) Gate-framing attrition", loc="left", pad=6)

    source_x, source_y = 0.50, 0.91
    add_box(
        ax,
        source_x,
        source_y,
        0.48,
        0.10,
        title="Hash-randomized gate framing",
        body=f"{TOTAL_ROWS:,} scenario-dyad observations",
        fc=pastel(OKABE_ITO["black"], 0.12),
        ec="#222222",
        title_size=7.2,
        body_size=6.3,
    )
    ax.text(
        source_x,
        0.81,
        "Deterministic hash assignment to gate framing",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#666666",
        style="italic",
    )

    gate_xs = [0.18, 0.50, 0.82]
    gate_y = 0.70
    gen_y = 0.47
    out_y = 0.20
    for gate, gx in zip(GATES, gate_xs):
        role = gate["role"]
        edge = edge_for(role)
        total = gate["n"]
        generated = gate["generated"]
        shifted = gate["shifted"]
        refused = total - generated
        no_shift = generated - shifted

        add_arrow(ax, source_x, source_y - 0.05, gx, gate_y + 0.055, "#666666")
        add_box(
            ax,
            gx,
            gate_y,
            0.25,
            0.11,
            title=gate["label"],
            body=f"n = {total:,}",
            fc=pastel(edge, 0.50),
            ec=edge,
            title_size=6.5,
            body_size=5.9,
        )

        gen_x = gx - 0.075
        ref_x = gx + 0.075
        generated_fc = pastel(edge, 0.50) if generated else "#F2F2F2"
        generated_ec = edge if generated else "#BBBBBB"
        generated_color = "#222222" if generated else "#888888"
        add_arrow(ax, gx, gate_y - 0.055, gen_x, gen_y + 0.045, edge if generated else "#999999")
        add_arrow(ax, gx, gate_y - 0.055, ref_x, gen_y + 0.045, OKABE_ITO["grey"])
        add_box(
            ax,
            gen_x,
            gen_y,
            0.135,
            0.09,
            title="Generated",
            body=f"{generated:,}\n({pct(generated, total)})",
            fc=generated_fc,
            ec=generated_ec,
            title_size=5.7,
            body_size=5.0,
            title_color=generated_color,
            body_color=generated_color,
        )
        add_box(
            ax,
            ref_x,
            gen_y,
            0.135,
            0.09,
            title="Non-pass /\nrefusal",
            body=f"{refused:,}\n({pct(refused, total)})",
            fc=pastel(OKABE_ITO["grey"], 0.28),
            ec=OKABE_ITO["grey"],
            title_size=5.2,
            body_size=5.0,
            title_color="#444444",
            body_color="#444444",
        )

        if generated:
            shift_x = gx - 0.075
            no_shift_x = gx + 0.075
            add_arrow(ax, gen_x, gen_y - 0.045, shift_x, out_y + 0.05, edge)
            add_arrow(ax, gen_x, gen_y - 0.045, no_shift_x, out_y + 0.05, "#666666")
            add_box(
                ax,
                shift_x,
                out_y,
                0.135,
                0.10,
                title="Target shift",
                body=f"{shifted:,}\n({pct(shifted, total)} of arm)",
                fc=fill_for("realized"),
                ec=edge_for("realized"),
                title_size=5.5,
                body_size=4.8,
            )
            add_box(
                ax,
                no_shift_x,
                out_y,
                0.135,
                0.10,
                title="No shift",
                body=f"{no_shift:,}\n({pct(no_shift, generated)} of gen.)",
                fc="#FFFFFF",
                ec="#777777",
                title_size=5.5,
                body_size=4.8,
            )
        else:
            add_arrow(ax, gen_x, gen_y - 0.045, gx, out_y + 0.05, "#999999")
            add_box(
                ax,
                gx,
                out_y,
                0.25,
                0.10,
                title="No target-aware outcome",
                body="0 generated interventions",
                fc="#F4F4F4",
                ec="#BBBBBB",
                title_size=5.5,
                body_size=4.8,
                title_color="#777777",
                body_color="#777777",
            )

    ax.text(
        0.50,
        0.055,
        "Non-pass rows do not expose the target-aware defender to a generated intervention.",
        ha="center",
        va="center",
        fontsize=6.2,
        color="#555555",
    )


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_title("(b) Gate-arm rates with scenario-cluster bootstrap 95% intervals", loc="left", pad=6)
    x = np.arange(len(GATES), dtype=float)
    offsets = [-0.18, 0.00, 0.18]
    metrics = [
        ("Generation availability", "generated", "n", "generated", edge_for("generation"), "o"),
        ("Realized target-shift exposure", "shifted", "n", "shifted", edge_for("realized"), "s"),
        (
            "Conditional shift among generated",
            "shifted",
            "generated",
            "conditional",
            edge_for("conditional_shift"),
            "D",
        ),
    ]

    for offset, (label, num_key, den_key, ci_key, color, marker) in zip(offsets, metrics):
        xs: list[float] = []
        ys: list[float] = []
        yerr_low: list[float] = []
        yerr_high: list[float] = []
        for idx, gate in enumerate(GATES):
            num = int(gate[num_key])
            den = int(gate[den_key])
            if den == 0:
                continue
            rate = num / den
            lo, hi = BOOTSTRAP_CI[(gate["key"], ci_key)]
            xs.append(x[idx] + offset)
            ys.append(rate)
            yerr_low.append(rate - lo)
            yerr_high.append(hi - rate)
        ax.errorbar(
            xs,
            ys,
            yerr=[yerr_low, yerr_high],
            fmt=marker,
            ms=5.0,
            lw=1.0,
            capsize=2.0,
            color=color,
            markerfacecolor=pastel(color, 0.65),
            markeredgecolor=color,
            label=label,
            zorder=4,
        )
        for x_i, y_i in zip(xs, ys):
            ax.text(
                x_i,
                y_i + 0.045,
                f"{100 * y_i:.1f}%",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=color,
            )

    ax.text(
        x[-1] + offsets[-1],
        0.075,
        "N/A\n0 generated",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=edge_for("conditional_shift"),
    )
    ax.set_xlim(-0.45, len(GATES) - 0.55)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Standard\n(n=993)",
            "Brief safety-\nframing\n(n=962)",
            "Policy-\nframing\n(n=1,013)",
        ]
    )
    ax.tick_params(axis="x", labelsize=6.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Rate")
    style_axes(ax, grid_axis="y")
    ax.legend(loc="upper right", fontsize=6.2, handletextpad=0.45)


def make_figure() -> Path:
    apply_style()
    verify_counts()
    fig = plt.figure(figsize=(FULL_COL_IN, 4.9))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.35, 1.0],
        left=0.02,
        right=0.985,
        top=0.93,
        bottom=0.12,
        wspace=0.20,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    paths = [
        OUT_DIR / "figure2_v3_consort_attrition.pdf",
        OUT_DIR / "figure2_v3_consort_attrition.png",
        OUT_DIR / "figure2_consort_attrition.pdf",
        OUT_DIR / "figure2_consort_attrition.png",
        PAPER_FIG_DIR / "figure2_consort_attrition.png",
        TACL_FIG_DIR / "figure2_consort_attrition.png",
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
    print(f"wrote: {PAPER_FIG_DIR / 'figure2_consort_attrition.png'}")
    print(f"wrote: {TACL_FIG_DIR / 'figure2_consort_attrition.png'}")
