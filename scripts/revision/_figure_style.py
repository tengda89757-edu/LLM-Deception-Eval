#!/usr/bin/env python3
"""Shared figure style for TACL revision Figures 2/3/4.

Design principles
-----------------
* Colour-vision-safe: Okabe–Ito 8-colour qualitative palette is used as the
  foundation. It is robust under deuteranopia, protanopia, and tritanopia and
  is the de-facto standard recommended by Wong (Nature Methods, 2011).
* Nature-style typography: sans-serif (Helvetica/Arial fallback), small but
  legible point sizes, hairline spines, ticks pointing outward, no top/right
  spines, light dotted gridlines on secondary axes only.
* Macaron / pastel feel: bar fills use a desaturated alpha-blended copy of the
  qualitative hue with the saturated hue used for the bar edge / accent stroke.
  This keeps the publication-ready aesthetic while preserving hue
  distinguishability for accessibility.
* Reproducible sizes: helpers expose Nature single-column (89 mm) and double
  -column (183 mm) widths in inches.

Module is import-safe (idempotent rcParams patch) and exposes one function
``apply_style()`` plus colour constants and figsize helpers.
"""
from __future__ import annotations

from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba

# ---------------------------------------------------------------------------
# Colour-vision-safe palette (Okabe–Ito).
# ---------------------------------------------------------------------------
OKABE_ITO: Final[dict[str, str]] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#7F7F7F",
}

# Semantic palette used across the three figures. Each role has an "edge" hue
# (saturated, used for outline / accent) and a "fill" hue (alpha-blended onto
# white to give a macaron-style pastel without losing hue separation).
SEMANTIC: Final[dict[str, dict[str, str]]] = {
    # Figure 2 Panel A roles
    "baseline":          {"edge": OKABE_ITO["blue"],          "fill_alpha": 0.55},
    "balanced_advice":   {"edge": OKABE_ITO["orange"],        "fill_alpha": 0.55},
    "generation":        {"edge": OKABE_ITO["sky_blue"],      "fill_alpha": 0.65},
    "realized":          {"edge": OKABE_ITO["vermillion"],    "fill_alpha": 0.55},
    # Figure 2 Panel B (generated-only conditional shift)
    "conditional_shift": {"edge": OKABE_ITO["reddish_purple"], "fill_alpha": 0.55},
    # Figure 2 Panel C (validation audit)
    "validation":        {"edge": OKABE_ITO["bluish_green"],   "fill_alpha": 0.55},
    # Figure 3 stacked bar roles
    "gate_pass":             {"edge": OKABE_ITO["bluish_green"],   "fill_alpha": 0.70},
    "safety_refusal":        {"edge": OKABE_ITO["vermillion"],     "fill_alpha": 0.65},
    "task_conflict_refusal": {"edge": OKABE_ITO["orange"],         "fill_alpha": 0.65},
    "ambiguous_refusal":     {"edge": OKABE_ITO["reddish_purple"], "fill_alpha": 0.65},
    # Figure 4 gate colours
    "standard_gate":     {"edge": OKABE_ITO["blue"],          "fill_alpha": 0.65},
    "safety_short_gate": {"edge": OKABE_ITO["orange"],        "fill_alpha": 0.65},
    "safety_policy_gate":{"edge": OKABE_ITO["bluish_green"],  "fill_alpha": 0.65},
    "aggregate":         {"edge": OKABE_ITO["grey"],          "fill_alpha": 0.45},
}


def pastel(hex_edge: str, alpha: float) -> tuple[float, float, float, float]:
    """Return an alpha-blended-on-white pastel RGBA matching the saturated edge."""
    r, g, b, _ = to_rgba(hex_edge)
    # Blend with white background to obtain the macaron pastel look while
    # remaining a fully opaque RGBA tuple (so it can be combined with edge
    # strokes in patch collections without unintended translucency stacking).
    return (
        r * alpha + 1.0 * (1 - alpha),
        g * alpha + 1.0 * (1 - alpha),
        b * alpha + 1.0 * (1 - alpha),
        1.0,
    )


def fill_for(role: str) -> tuple[float, float, float, float]:
    cfg = SEMANTIC[role]
    return pastel(cfg["edge"], cfg["fill_alpha"])


def edge_for(role: str) -> str:
    return SEMANTIC[role]["edge"]


# ---------------------------------------------------------------------------
# Typography & rcParams.
# ---------------------------------------------------------------------------
_PREFERRED_SANS = [
    "Helvetica",
    "Helvetica Neue",
    "Arial",
    "Liberation Sans",
    "Nimbus Sans",
    "DejaVu Sans",
]


def _resolve_sans_serif() -> list[str]:
    """Filter to fonts actually available on the system, keeping ranking."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in _PREFERRED_SANS if name in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
    return chosen


_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply the Nature-style rcParams. Idempotent."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    sans = _resolve_sans_serif()
    mpl.rcParams.update({
        # Typography
        "font.family":        "sans-serif",
        "font.sans-serif":    sans,
        "font.size":          7.5,
        "axes.titlesize":     8.5,
        "axes.titleweight":   "bold",
        "axes.labelsize":     7.5,
        "xtick.labelsize":    6.8,
        "ytick.labelsize":    6.8,
        "legend.fontsize":    6.8,
        "legend.title_fontsize": 7.0,
        "figure.titlesize":   9.5,
        "figure.titleweight": "bold",
        # Lines & spines
        "axes.linewidth":     0.6,
        "axes.edgecolor":     "#222222",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.labelcolor":    "#222222",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.6,
        "ytick.major.size":   2.6,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Grid (off by default; enabled on a per-axis basis)
        "axes.grid":          False,
        "grid.color":         "#BFBFBF",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.4,
        # Legend
        "legend.frameon":     False,
        "legend.handlelength": 1.6,
        "legend.handleheight": 0.9,
        "legend.borderpad":   0.2,
        "legend.borderaxespad": 0.4,
        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        600,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype":       42,   # TrueType, editor-friendly
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
    })
    _STYLE_APPLIED = True


# ---------------------------------------------------------------------------
# Figure-size helpers (Nature columns).
# ---------------------------------------------------------------------------
MM_PER_INCH: Final[float] = 25.4


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


# Nature single column = 89 mm; 1.5 column = 120 mm; full = 183 mm.
SINGLE_COL_IN: Final[float] = mm_to_in(89.0)   # ~3.504 in
ONE_HALF_COL_IN: Final[float] = mm_to_in(120.0) # ~4.724 in
FULL_COL_IN: Final[float] = mm_to_in(183.0)    # ~7.205 in


def style_axes(ax: plt.Axes, *, grid_axis: str | None = "x") -> None:
    """Apply per-axes touch-ups: outward ticks, hairline spines, optional grid."""
    ax.tick_params(direction="out", length=2.6, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#222222")
    if grid_axis is not None:
        ax.grid(True, axis=grid_axis, linestyle=":", linewidth=0.4, color="#BFBFBF",
                zorder=0)
    ax.set_axisbelow(True)


def annotate_count(ax: plt.Axes, x: float, y: float, text: str, *, ha: str = "left",
                   va: str = "center", color: str = "#222222") -> None:
    """Standardised numeric / fraction annotation."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=6.4, color=color, clip_on=False)


__all__ = [
    "OKABE_ITO",
    "SEMANTIC",
    "apply_style",
    "fill_for",
    "edge_for",
    "pastel",
    "mm_to_in",
    "SINGLE_COL_IN",
    "ONE_HALF_COL_IN",
    "FULL_COL_IN",
    "style_axes",
    "annotate_count",
]
