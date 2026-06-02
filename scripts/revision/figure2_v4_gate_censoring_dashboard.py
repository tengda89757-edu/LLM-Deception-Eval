#!/usr/bin/env python3
"""Figure 2 v4: gate censoring, rates, refusal taxonomy, and domain exposure."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
DOMAIN_CSV = ROOT / "outputs/final_results_20260427/domain_summary.csv"
OUT_DIR = ROOT / "revision_supplementary/figures"
PAPER_FIG_DIR = ROOT / "paper/figures"
TACL_FIG_DIR = ROOT / "TACL_revised_package/figures"
for path in (OUT_DIR, PAPER_FIG_DIR, TACL_FIG_DIR):
    path.mkdir(parents=True, exist_ok=True)

GATES = [
    {"key": "standard_gate", "label": "Standard", "role": "standard_gate", "n": 993, "generated": 454, "shifted": 328},
    {"key": "safety_short_gate", "label": "Brief safety", "role": "safety_short_gate", "n": 962, "generated": 52, "shifted": 40},
    {"key": "safety_policy_gate", "label": "Policy", "role": "safety_policy_gate", "n": 1013, "generated": 0, "shifted": 0},
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

DOMAIN_LABELS = {
    "business_organizations": "Business",
    "governance_law_history": "Gov/law/history",
    "physical_infrastructure_environment": "Infrastructure/env.",
    "finance_markets": "Finance",
    "health_biomedicine": "Health/biomed.",
    "digital_information_systems": "Digital/info",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def load_analysis() -> pd.DataFrame:
    cols = [
        "gate_iv_arm",
        "gate_decision",
        "engagement_binary",
        "deception_success_binary",
        "compact_domain",
    ]
    return pd.read_csv(ANALYSIS_CSV, usecols=cols, low_memory=False)


def verify_counts(df: pd.DataFrame) -> None:
    assert len(df) == 2968
    by_gate = df.groupby("gate_iv_arm").agg(
        n=("gate_iv_arm", "size"),
        generated=("engagement_binary", "sum"),
        shifted=("deception_success_binary", "sum"),
    )
    for gate in GATES:
        row = by_gate.loc[gate["key"]]
        assert int(row["n"]) == gate["n"]
        assert int(row["generated"]) == gate["generated"]
        assert int(row["shifted"]) == gate["shifted"]
    policy = df[df["gate_iv_arm"] == "safety_policy_gate"]["gate_decision"].value_counts()
    assert int(policy.get("REFUSE_SAFETY", 0)) == 971
    assert int(policy.get("REFUSE_TASK_CONFLICT", 0)) == 36
    assert int(policy.get("REFUSE_AMBIGUOUS", 0) + policy.get("UNPARSEABLE", 0)) == 6


def draw_attrition_matrix(ax: plt.Axes) -> None:
    ax.set_title("(a) Common-denominator attrition by gate arm", loc="left", pad=5)
    y = np.arange(len(GATES))[::-1]
    categories = [
        ("shifted", "Realized target shift", "realized"),
        ("no_shift", "Generated, no shift", "conditional_shift"),
        ("refused", "Non-pass/refusal", "aggregate"),
    ]
    for y_i, gate in zip(y, GATES):
        total = gate["n"]
        values = {
            "shifted": gate["shifted"],
            "no_shift": gate["generated"] - gate["shifted"],
            "refused": total - gate["generated"],
        }
        left = 0.0
        for key, _label, role in categories:
            width = values[key] / total
            edge = edge_for(role)
            ax.barh(
                y_i,
                width,
                left=left,
                height=0.58,
                color=fill_for(role),
                edgecolor=edge,
                linewidth=0.8,
                zorder=3,
            )
            if width > 0.075:
                ax.text(
                    left + width / 2,
                    y_i,
                    f"{values[key]:,}\n{pct(width)}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#222222",
                )
            left += width
        ax.text(1.01, y_i, f"n={total:,}", ha="left", va="center", fontsize=6.0, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([gate["label"] for gate in GATES])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Share of gate-arm denominator")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="x")
    handles = [
        mpatches.Patch(facecolor=fill_for(role), edgecolor=edge_for(role), label=label)
        for _key, label, role in categories
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, -0.38), ncol=3, fontsize=5.8)


def draw_gate_rates(ax: plt.Axes) -> None:
    ax.set_title("(b) Gate-arm rates with scenario-cluster bootstrap intervals", loc="left", pad=5)
    x = np.arange(len(GATES), dtype=float)
    metrics = [
        ("Generation availability", "generated", "n", "generated", edge_for("generation"), "o", -0.18),
        ("Realized exposure", "shifted", "n", "shifted", edge_for("realized"), "s", 0.0),
        ("Conditional shift among generated", "shifted", "generated", "conditional", edge_for("conditional_shift"), "D", 0.18),
    ]
    for label, num_key, den_key, ci_key, color, marker, offset in metrics:
        xs, ys, lows, highs = [], [], [], []
        for idx, gate in enumerate(GATES):
            den = int(gate[den_key])
            if den == 0:
                continue
            rate = gate[num_key] / den
            lo, hi = BOOTSTRAP_CI[(gate["key"], ci_key)]
            xs.append(x[idx] + offset)
            ys.append(rate)
            lows.append(rate - lo)
            highs.append(hi - rate)
        ax.errorbar(
            xs,
            ys,
            yerr=[lows, highs],
            fmt=marker,
            ms=5.0,
            lw=1.0,
            capsize=2.0,
            color=color,
            markerfacecolor=pastel(color, 0.62),
            markeredgecolor=color,
            label=label,
            zorder=4,
        )
        for x_i, y_i in zip(xs, ys):
            ax.text(x_i, y_i + 0.045, pct(y_i), ha="center", va="bottom", fontsize=5.8, color=color)
    ax.text(2.18, 0.065, "conditional\nN/A", ha="center", va="bottom", fontsize=5.8, color=edge_for("conditional_shift"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gate['label']}\n(n={gate['n']:,})" for gate in GATES])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="y")
    legend = ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.58), fontsize=5.6, handlelength=1.0, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)


def draw_gate_decisions(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("(c) Generation gate decision taxonomy", loc="left", pad=5)
    decision_specs = [
        ("ENGAGE", "Gate pass", "gate_pass"),
        ("REFUSE_SAFETY", "Safety refusal", "safety_refusal"),
        ("REFUSE_TASK_CONFLICT", "Task conflict", "task_conflict_refusal"),
        ("AMBIG_OR_UNPARSEABLE", "Ambiguous/unparseable", "ambiguous_refusal"),
    ]
    y = np.arange(len(GATES))[::-1]
    for y_i, gate in zip(y, GATES):
        sub = df[df["gate_iv_arm"] == gate["key"]]["gate_decision"].fillna("UNPARSEABLE")
        counts = {
            "ENGAGE": int((sub == "ENGAGE").sum()),
            "REFUSE_SAFETY": int((sub == "REFUSE_SAFETY").sum()),
            "REFUSE_TASK_CONFLICT": int((sub == "REFUSE_TASK_CONFLICT").sum()),
            "AMBIG_OR_UNPARSEABLE": int(sub.isin(["REFUSE_AMBIGUOUS", "UNPARSEABLE"]).sum()),
        }
        left = 0.0
        for key, _label, role in decision_specs:
            width = counts[key] / gate["n"]
            if width <= 0:
                continue
            ax.barh(
                y_i,
                width,
                left=left,
                height=0.58,
                color=fill_for(role),
                edgecolor=edge_for(role),
                linewidth=0.8,
                zorder=3,
            )
            if width > 0.08:
                ax.text(left + width / 2, y_i, f"{counts[key]:,}", ha="center", va="center", fontsize=5.8)
            left += width
    ax.set_yticks(y)
    ax.set_yticklabels([gate["label"] for gate in GATES])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of gate-arm denominator")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="x")
    handles = [
        mpatches.Patch(facecolor=fill_for(role), edgecolor=edge_for(role), label=label)
        for _key, label, role in decision_specs
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2, fontsize=5.7)


def draw_domain_exposure(ax: plt.Axes) -> None:
    ax.set_title("(d) Domain-level generation and realized exposure", loc="left", pad=5)
    dom = pd.read_csv(DOMAIN_CSV).copy()
    dom["label"] = dom["compact_domain"].map(DOMAIN_LABELS)
    dom = dom.sort_values("success_unconditional_rate", ascending=True)
    y = np.arange(len(dom))
    ax.hlines(
        y,
        dom["success_unconditional_rate"],
        dom["engagement_rate"],
        color="#B9B9B9",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        dom["engagement_rate"],
        y,
        s=35,
        color=fill_for("generation"),
        edgecolor=edge_for("generation"),
        linewidth=0.8,
        label="Generation",
        zorder=3,
    )
    ax.scatter(
        dom["success_unconditional_rate"],
        y,
        s=35,
        color=fill_for("realized"),
        edgecolor=edge_for("realized"),
        linewidth=0.8,
        label="Realized exposure",
        zorder=4,
    )
    for y_i, row in zip(y, dom.itertuples(index=False)):
        ax.text(
            max(row.engagement_rate, row.success_unconditional_rate) + 0.012,
            y_i,
            f"n={row.n:,}; cond.={100 * row.success_among_engaged_parseable_rate:.0f}%",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#555555",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(dom["label"])
    ax.set_xlim(0, 0.34)
    ax.set_xlabel("Rate")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    style_axes(ax, grid_axis="x")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.0)


def make_figure() -> list[Path]:
    apply_style()
    df = load_analysis()
    verify_counts(df)

    fig = plt.figure(figsize=(FULL_COL_IN, 5.85))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.985,
        top=0.94,
        bottom=0.10,
        hspace=0.62,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    draw_attrition_matrix(ax_a)
    draw_gate_rates(ax_b)
    draw_gate_decisions(ax_c, df)
    draw_domain_exposure(ax_d)

    note = (
        "All rates use the fixed 2,968-row denominator unless explicitly conditional on generated interventions; "
        "intervals are descriptive scenario-cluster bootstrap references."
    )
    fig.text(0.08, 0.025, note, ha="left", va="bottom", fontsize=5.8, color="#555555")

    pdf_path = OUT_DIR / "figure2_v4_gate_censoring_dashboard.pdf"
    png_path = OUT_DIR / "figure2_v4_gate_censoring_dashboard.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    public_paths = [
        TACL_FIG_DIR / "figure2_consort_attrition.png",
        PAPER_FIG_DIR / "figure2_consort_attrition.png",
    ]
    for path in public_paths:
        shutil.copy2(png_path, path)
    return [pdf_path, png_path, *public_paths]


if __name__ == "__main__":
    for out in make_figure():
        print(f"wrote: {out}")
