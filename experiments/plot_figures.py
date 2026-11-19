#!/usr/bin/env python3
"""Render every figure used in the CS5491 final report.

The figures are built from the committed result tables under
``experiments/data/`` only.  No label file is read here, nothing is tuned and
no outcome is used to choose a plotting parameter: every value drawn is a
value that ``EVALUATION.md`` already reports.  The script never touches
``nsc-starter/data/*_labels.json``.

Usage
-----
    python experiments/plot_figures.py                    # every figure -> figures/
    python experiments/plot_figures.py --list             # list figures and captions
    python experiments/plot_figures.py --only fig1 fig5   # a subset
    python experiments/plot_figures.py --formats png      # PNG only
    python experiments/plot_figures.py --formats png pdf --dpi 300
    python experiments/plot_figures.py --out-dir build/figures

Requirements: matplotlib >= 3.5.  The core harness in ``nsc-starter/`` stays
standard-library only; matplotlib is needed for the report figures only.

Colour palette
--------------
A restrained academic palette: four coordinated hues on white, dark-grey
ink, no rainbow, and every series also varies marker or line style so the
figures survive black-and-white printing.

    ink      #333333   axes, text, the NSC reference
    blue     #2563EB   our prefix reweighting
    blue_lt  #3B82F6   position-only control / secondary series
    purple   #7C3AED   positional and structural terms
    green    #10B981   resource baselines
    orange   #EA580C   random selection and residual terms
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiments" / "data"
DEFAULT_OUT = ROOT / "figures"

# --------------------------------------------------------------------------
# Palette and global style
# --------------------------------------------------------------------------
P = {
    "ink": "#333333",
    "ink_soft": "#1F2937",
    "blue": "#2563EB",
    "blue_lt": "#3B82F6",
    "blue_pale": "#93C5FD",
    "green": "#10B981",
    "green_lt": "#34D399",
    "purple": "#7C3AED",
    "purple_lt": "#8B5CF6",
    "orange": "#EA580C",
    "orange_lt": "#F97316",
    "grey": "#9CA3AF",
    "grey_lt": "#E5E7EB",
    "white": "#FFFFFF",
}

SERIF = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
         "DejaVu Serif"]


def apply_style(dpi: int) -> None:
    """Global rcParams: print-friendly, serif, thin borders, no decoration."""
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.0,
        "axes.titlesize": 8.8,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": P["ink"],
        "axes.labelcolor": P["ink"],
        "axes.titlecolor": P["ink"],
        "text.color": P["ink"],
        "xtick.color": P["ink"],
        "ytick.color": P["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "figure.facecolor": P["white"],
        "axes.facecolor": P["white"],
        "savefig.facecolor": P["white"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "grid.color": P["grey_lt"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.columnspacing": 1.4,
        "legend.handletextpad": 0.6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def font_note() -> str:
    """Report which serif font matplotlib will actually use."""
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        return "Times New Roman"
    for name in ("Times", "Liberation Serif", "Nimbus Roman"):
        if name in available:
            return name
    return "DejaVu Serif (Times New Roman not installed)"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def panel_title(ax, title: str, subtitle: str = "") -> None:
    """Two-line panel heading.  A single text object, so it can never overlap."""
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=8.6, pad=5,
                     linespacing=1.35)
    else:
        ax.set_title(title, fontsize=9.0, pad=5)


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def note(ax, text: str, loc: str = "lower left", size: float = 7.0) -> None:
    """Small grey boxed note inside an axes."""
    positions = {"lower left": (0.03, 0.03, "left", "bottom"),
                 "upper left": (0.03, 0.97, "left", "top"),
                 "lower right": (0.97, 0.03, "right", "bottom"),
                 "upper right": (0.97, 0.97, "right", "top")}
    x, y, ha, va = positions[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            color=P["ink_soft"], zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=P["white"],
                      edgecolor=P["grey_lt"], alpha=0.92))


def weighted_quantile(pairs, q: float):
    """Quantile of a discrete distribution given as (value, weight) pairs."""
    items = sorted(pairs)
    total = sum(w for _, w in items)
    if total <= 0:
        return None
    target = q * total
    cumulative = 0.0
    for value, weight in items:
        cumulative += weight
        if cumulative >= target:
            return value
    return items[-1][0]


def spearman(x, y):
    """Spearman rho with average ranks (same convention as the harness)."""
    sys.path.insert(0, str(ROOT / "nsc-starter"))
    try:
        import evaluate as harness  # noqa: WPS433 (local import on purpose)
    finally:
        sys.path.pop(0)
    return harness.correlation(harness.ranks([float(v) for v in x]),
                               harness.ranks([float(v) for v in y]))


def rho_label(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def save(fig, stem: str, out_dir: Path, formats, dpi: int) -> list:
    """Write the figure as PNG (raster, for preview) and PDF (vector, print).

    The PDF metadata is pinned so that regenerating the figures is
    byte-for-byte reproducible: a timestamp in the file would otherwise make
    two runs of the same command differ.
    """
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "experiments/plot_figures.py", "Producer": "matplotlib",
                "CreationDate": None, "ModDate": None}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "pdf":
            fig.savefig(path, format=fmt, metadata=metadata)
        else:
            fig.savefig(path, format=fmt, dpi=dpi)
        written.append(path)
    return written


# ==========================================================================
# Figure 1 - the headline result: the alpha sweep never beats plain NSC
# ==========================================================================
ALPHAS = (0.0, 0.25, 0.5, 1.0)


def fig1_alpha_sweep(ev, an, mc):
    """Spearman rho against the alpha grid, with paired-bootstrap intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    fig.subplots_adjust(wspace=0.10, top=0.80, bottom=0.33)
    for ax, split, subtitle in zip(
            axes,
            ("development", "final"),
            ("Development split (validation PPL)",
             "Final split (test PPL, held out)")):
        ranking = ev["ranking"][split]
        nsc = ranking["original_nsc"]["spearman_rho"]
        boot = an["splits"][split]["paired_architecture_bootstrap"]

        prefix, position = [], []
        prefix_lo, prefix_hi, pos_lo, pos_hi = [], [], [], []
        for alpha in ALPHAS:
            if alpha == 0.0:
                prefix.append(nsc)
                position.append(nsc)
                prefix_lo.append(nsc)
                prefix_hi.append(nsc)
                pos_lo.append(nsc)
                pos_hi.append(nsc)
                continue
            rho_p = ranking[f"sensitivity_a{alpha}"]["spearman_rho"]
            rho_c = ranking[f"position_control_a{alpha}"]["spearman_rho"]
            prefix.append(rho_p)
            position.append(rho_c)
            # CI of (prefix - nsc) shifted onto the prefix score.
            ci = boot.get(f"sensitivity_a{alpha} minus original_nsc", {}).get("ci95")
            prefix_lo.append(nsc + ci[0] if ci else rho_p)
            prefix_hi.append(nsc + ci[1] if ci else rho_p)
            # CI of (prefix - position): position = prefix - delta, bounds flip.
            ci2 = boot.get(f"sensitivity_a{alpha} minus position_control_a{alpha}",
                           {}).get("ci95")
            pos_lo.append(rho_p - ci2[1] if ci2 else rho_c)
            pos_hi.append(rho_p - ci2[0] if ci2 else rho_c)

        xs = list(range(len(ALPHAS)))
        ax.axhline(nsc, color=P["ink"], linestyle="--", linewidth=1.1, zorder=1)
        ax.fill_between(xs, prefix_lo, prefix_hi, color=P["blue"], alpha=0.15,
                        linewidth=0, zorder=2)
        ax.plot(xs, position, color=P["purple"], marker="s", linestyle="--",
                zorder=3, label="position-only control")
        ax.plot(xs, prefix, color=P["blue"], marker="o", zorder=4,
                label="prefix reweighting (ours)")
        ax.plot([0], [nsc], marker="o", markersize=8.0, markerfacecolor="none",
                markeredgecolor=P["ink"], markeredgewidth=1.1, zorder=5)
        ax.axvline(0, color=P["grey"], linewidth=0.9, linestyle=":", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
        ax.set_xlabel("prefix weight $\\alpha$")
        ax.set_xlim(-0.30, len(ALPHAS) - 0.40)
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
        despine(ax)
        panel_title(ax, subtitle, "y-axis zoomed")
        note(ax, f"selected $\\alpha = 0$\n$\\rho$ = {nsc:.4f}", "lower left")

    axes[0].set_ylabel("Spearman $\\rho$ vs negative PPL")
    handles = [
        Line2D([], [], color=P["blue"], marker="o", label="prefix reweighting (ours)"),
        Line2D([], [], color=P["purple"], marker="s", linestyle="--",
               label="position-only control"),
        Line2D([], [], color=P["ink"], linestyle="--", label="original NSC ($\\alpha = 0$)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.012))
    fig.text(0.5, 0.135,
             "Every positive $\\alpha$ lowers the ranking correlation on both "
             "splits; the band is a 95% paired-architecture bootstrap "
             "interval for the prefix score.",
             ha="center", va="bottom", fontsize=7.2, color=P["ink_soft"])
    return fig


# ==========================================================================
# Figure 2 - ranking quality of every method against the required baselines
# ==========================================================================
GROUPS = [
    ("NSC and the $\\alpha = 0$ variants", [
        ("original_nsc", "original NSC"),
        ("student", "student score ($\\alpha = 0$)"),
        ("ablation", "ablation ($\\alpha = 0$)"),
        ("position_only_control", "position-only control ($\\alpha = 0$)"),
    ], P["ink"]),
    ("Prefix reweighting (ours), $\\alpha > 0$", [
        ("sensitivity_a0.25", "prefix $\\alpha = 0.25$"),
        ("sensitivity_a0.5", "prefix $\\alpha = 0.5$"),
        ("sensitivity_a1.0", "prefix $\\alpha = 1.0$"),
        ("without_head_residual_a0.25", "prefix 0.25, no $R$"),
        ("without_head_residual_a0.5", "prefix 0.5, no $R$"),
        ("without_head_residual_a1.0", "prefix 1.0, no $R$"),
    ], P["blue"]),
    ("Positional controls", [
        ("position_example", "position example (given)"),
        ("position_control_a0.25", "position control 0.25"),
        ("position_control_a0.5", "position control 0.5"),
        ("position_control_a1.0", "position control 1.0"),
    ], P["purple"]),
    ("Resource baselines", [
        ("reported_parameters", "reported parameters"),
        ("reference_flops", "reference FLOPs"),
    ], P["green"]),
]


def fig2_ranking_baselines(ev, an, mc):
    """All 16 methods: all-pairs Spearman rho and restricted tau-b."""
    flat = [(g, item, colour) for g, items, colour in GROUPS for item in items]
    count = len(flat)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.0), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06, top=0.90, bottom=0.20, left=0.24,
                        right=0.99)

    values = []
    for _, (key, _), _ in flat:
        for split in ("development", "final"):
            entry = ev["ranking"][split][key]
            values.append(entry["spearman_rho"])
            values.append(entry["size_controlled_pairs"]["tau_b"])
    low = min(values) - 0.012
    high = 1.034

    for ax, split, subtitle in zip(
            axes,
            ("development", "final"),
            ("Development split (validation PPL)",
             "Final split (test PPL, held out)")):
        ranking = ev["ranking"][split]
        # Largest y at the top, so the first method of each group reads first.
        ys = list(range(count))[::-1]
        for y, (_, (key, _), colour) in zip(ys, flat):
            entry = ranking[key]
            rho = entry["spearman_rho"]
            restricted = entry["size_controlled_pairs"]["tau_b"]
            ax.plot([restricted, rho], [y, y], color=colour, linewidth=1.0,
                    alpha=0.35, zorder=2, solid_capstyle="round")
            ax.plot([restricted], [y], marker="D", markersize=3.6,
                    markerfacecolor=P["white"], markeredgecolor=P["ink_soft"],
                    markeredgewidth=0.9, linestyle="none", zorder=4)
            ax.plot([rho], [y], marker="o", markersize=6.2, color=colour,
                    markeredgecolor=P["white"], markeredgewidth=0.6, zorder=5)
            ax.text(rho + 0.005, y, f"{rho:.4f}", va="center", ha="left",
                    fontsize=6.4, color=P["ink_soft"], zorder=6, clip_on=False)
        # separators between method families
        running = 0
        for _, items, _ in GROUPS[:-1]:
            running += len(items)
            ax.axhline(running - 0.5, color=P["grey_lt"], linewidth=1.2,
                       zorder=1)
        ax.axvline(ranking["original_nsc"]["spearman_rho"], color=P["ink"],
                   linestyle="--", linewidth=1.0, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, (_, label), _ in flat])
        ax.set_ylim(-0.8, count - 0.2)
        ax.set_xlim(low, high)
        ax.set_xlabel("Spearman $\\rho$ (all pairs)")
        ax.grid(axis="x", alpha=0.55)
        ax.set_axisbelow(True)
        despine(ax)
        panel_title(ax, subtitle)

    handles = [Line2D([], [], color=colour, marker="o", markersize=6.0,
                      linestyle="none", label=group)
               for group, _, colour in GROUPS]
    handles.append(Line2D([], [], color=P["ink"], marker="D", markersize=4.4,
                           markerfacecolor=P["white"],
                           markeredgecolor=P["ink_soft"], linestyle="none",
                           label="restricted $\\tau_b$ (size-controlled pairs)"))
    handles.append(Line2D([], [], color=P["ink"], ls="--", lw=1.1,
                           label="original NSC"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Ranking quality of every method and baseline",
                 y=0.985, fontsize=9.6)
    return fig


# ==========================================================================
# Figure 3 - mechanism check: the score does respond to layer order
# ==========================================================================
FAMILY_ORDER = ["toy_pair", "ffn_reversed_fixed_heads", "ffn_rotated_fixed_heads",
                "uniform_heads_ffn_reversed", "both_reversed",
                "heads_reversed_fixed_ffn"]

FAMILY_LABELS = {
    "toy_pair": "toy pair (given)",
    "ffn_reversed_fixed_heads": "FFN reversed, fixed heads",
    "ffn_rotated_fixed_heads": "FFN rotated, fixed heads",
    "uniform_heads_ffn_reversed": "FFN reversed, uniform heads",
    "both_reversed": "heads and FFN reversed",
    "heads_reversed_fixed_ffn": "heads only, fixed FFN",
}


def _abs_delta(entry, key):
    values = entry.get(key) or {}
    value = values.get("1.0")
    return 0.0 if value is None else abs(float(value))


def fig3_mechanism(ev, an, mc):
    """Structural sensitivity on the 41 synthetic reorderings."""
    cases = mc["case_details"]
    families = [f for f in FAMILY_ORDER
                if any(c["family"] == f for c in cases)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.34, top=0.86, bottom=0.26)

    series = (("ours_delta", P["blue"], "prefix reweighting ($\\alpha = 1$)"),
              ("position_control_delta", P["purple"],
               "position-only control ($\\alpha = 1$)"))

    # ---- panel (a): largest |delta| per family -------------------------
    ax = axes[0]
    ys = list(range(len(families)))
    height = 0.36
    largest = 0.0
    for offset, (key, colour, _label) in enumerate(series):
        values = [max((_abs_delta(c, key) for c in cases
                       if c["family"] == family), default=0.0)
                  for family in families]
        largest = max(largest, max(values))
        ax.barh([y + (0.5 - offset) * height for y in ys], values,
                height=height * 0.94, color=colour, zorder=2,
                edgecolor=P["white"], linewidth=0.4)
        for y, value in zip(ys, values):
            ax.text(value + largest * 0.012, y + (0.5 - offset) * height,
                    f"{value:.2f}", va="center", ha="left", fontsize=6.6,
                    color=P["ink_soft"])
    ax.set_yticks(ys)
    ax.set_yticklabels([FAMILY_LABELS[f] for f in families])
    ax.set_ylim(len(families) - 0.4, -0.6)
    ax.set_xlim(0, largest * 1.22)
    ax.set_xlabel("$|\\Delta\\,$ score$|$ at $\\alpha = 1$")
    ax.grid(axis="x", alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    panel_title(ax, "Largest reorder effect per family",
                "synthetic reorderings, no trained outcomes")

    # ---- panel (b): head-only cases ------------------------------------
    ax = axes[1]
    head_cases = sorted((c for c in cases
                          if c["family"] == "heads_reversed_fixed_ffn"),
                         key=lambda c: c["case_id"])
    xs = list(range(len(head_cases)))
    ours = [_abs_delta(c, "ours_delta") for c in head_cases]
    pos = [_abs_delta(c, "position_control_delta") for c in head_cases]
    ax.bar([x - 0.19 for x in xs], ours, width=0.36, color=P["blue"], zorder=2,
           edgecolor=P["white"], linewidth=0.4)
    ax.bar([x + 0.19 for x in xs], pos, width=0.36, color=P["purple"], zorder=2,
           edgecolor=P["white"], linewidth=0.4)
    for x, value in zip(xs, ours):
        ax.text(x - 0.19, value, f"{value:.2f}", ha="center", va="bottom",
                fontsize=6.4, color=P["ink_soft"])
    ax.set_xticks(xs)
    ax.set_xticklabels([c["case_id"].split("__")[0].replace("config_", "")
                        for c in head_cases], rotation=45, ha="right",
                       fontsize=6.6)
    ax.set_xlabel("source architecture (head order reversed, FFN fixed)")
    ax.set_ylabel("$|\\Delta\\,$ score$|$")
    ax.set_ylim(0, max(ours) * 1.34)
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    panel_title(ax, "Head-only reorderings",
                "prefix responds; the positional control cannot")
    ax.annotate("positional control\nis exactly 0", xy=(0.19, 0.0),
                xytext=(0.34, 0.62), textcoords="axes fraction", fontsize=7.0,
                color=P["purple"], ha="left",
                arrowprops=dict(arrowstyle="->", color=P["purple"], lw=0.9))

    handles = [Line2D([], [], color=colour, lw=6, label=label)
               for _key, colour, label in series]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.10,
             f"The original NSC changes by exactly 0 on all {mc['cases']} "
             "cases: its layer terms are separable and summed.",
             ha="center", va="bottom", fontsize=7.2, color=P["ink_soft"])
    return fig


# ==========================================================================
# Figure 4 - the key insight: NSC is almost a restatement of compute
# ==========================================================================
WIDTH_COLOURS = {64: P["green_lt"], 128: P["blue"], 256: P["purple"],
                 512: P["orange"]}
WIDTH_MARKERS = {64: "o", 128: "s", 256: "^", 512: "D"}


def fig4_collinearity(ev, an, mc):
    """Outcome and score are both nearly determined by reference FLOPs."""
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    fig.subplots_adjust(wspace=0.28, hspace=0.55, top=0.90, bottom=0.14,
                        left=0.10, right=0.98)
    for row, split in enumerate(("development", "final")):
        profiles = an["splits"][split]["architecture_profiles"]
        ids = sorted(profiles)
        flops = [profiles[i]["reference_flops_per_token"] for i in ids]
        ppl = [profiles[i]["recorded_ppl"] for i in ids]
        nsc = [profiles[i]["original_nsc"] for i in ids]
        widths = sorted({profiles[i]["d_model"] for i in ids})

        for col, (values, ylabel, logy, name) in enumerate((
                (ppl, "perplexity", False, "Perplexity vs compute"),
                (nsc, "original NSC score", True, "NSC score vs compute"))):
            ax = axes[row][col]
            for width in widths:
                keep = [k for k, i in enumerate(ids)
                        if profiles[i]["d_model"] == width]
                ax.scatter([flops[k] for k in keep], [values[k] for k in keep],
                           s=15, marker=WIDTH_MARKERS[width],
                           color=WIDTH_COLOURS[width], alpha=0.80,
                           edgecolors=P["white"], linewidths=0.3, zorder=3,
                           label=f"$d$ = {width}")
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("reference FLOPs per token")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.55)
            ax.set_axisbelow(True)
            despine(ax)
            if logy:
                stat = f"$\\rho$ = {rho_label(spearman(flops, values))}"
            else:
                stat = (f"$\\rho$($-$PPL, FLOPs) = "
                        f"{rho_label(spearman(flops, [-v for v in values]))}")
            panel_title(ax, f"{name} ({split})", stat)
        axes[row][0].text(
            0.03, 0.03,
            f"$\\rho$(NSC, FLOPs) = {rho_label(spearman(flops, nsc))}\n"
            f"$\\rho$(NSC, $-$PPL) = "
            f"{rho_label(spearman(nsc, [-v for v in ppl]))}",
            transform=axes[row][0].transAxes, fontsize=6.8,
            color=P["ink_soft"], va="bottom", zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", facecolor=P["white"],
                      edgecolor=P["grey_lt"], alpha=0.92))
    handles = [Line2D([], [], color=WIDTH_COLOURS[w], marker=WIDTH_MARKERS[w],
                      markersize=4.6, linestyle="none", label=f"$d$ = {w}")
               for w in sorted(WIDTH_COLOURS)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("The recorded outcome and the NSC score are both nearly a "
                 "function of compute", y=0.985, fontsize=9.4)
    return fig


# ==========================================================================
# Figure 5 - budgeted architecture selection against the required baselines
# ==========================================================================
SELECT_METHODS = [
    ("original_nsc", "original NSC", P["ink"], "o"),
    ("sensitivity_a0.25", "prefix $\\alpha = 0.25$ (ours)", P["blue"], "s"),
    ("position_example", "position example (given)", P["purple"], "^"),
    ("reported_parameters", "reported parameters", P["green"], "D"),
]
CAPS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)


def _selection_index(cases):
    return {(case["parameter_cap"], case["k"]): case for case in cases}


def fig5_selection(ev, an, mc):
    """Best selected PPL and regret under the 10/20/30/50M parameter caps."""
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    fig.subplots_adjust(wspace=0.24, hspace=0.42, top=0.90, bottom=0.22,
                        left=0.11, right=0.98)
    for row, (metric, ylabel) in enumerate((
            ("mean_best_ppl", "mean best selected PPL"),
            ("mean_regret", "mean regret to best feasible"))):
        for col, split in enumerate(("development", "final")):
            ax = axes[row][col]
            index = _selection_index(an["splits"][split]["selection_cases"])
            xs = [c / 1e6 for c in CAPS]
            for k in (1, 3):
                style = "-" if k == 1 else "--"
                oracle = [index[(cap, k)]["oracle_ppl"] for cap in CAPS]
                if row == 0:
                    ax.step(xs, oracle, where="post", color=P["ink"],
                            linestyle=(0, (5, 2)), linewidth=1.2, zorder=1)
                else:
                    ax.axhline(0.0, color=P["ink"], linestyle=(0, (5, 2)),
                               linewidth=1.2, zorder=1)
                for key, _label, colour, marker in SELECT_METHODS:
                    values = [index[(cap, k)]["methods"][key][metric]
                              for cap in CAPS]
                    ax.plot(xs, values, color=colour, linestyle=style,
                            marker=marker, markersize=4.2, zorder=4)
                # random selection: 5-95% band from the recorded 200 trials
                lower, upper, median = [], [], []
                for cap in CAPS:
                    case = index[(cap, k)]
                    dist = [(entry["ppl"], entry["trials"]) for entry in
                            case["methods"]["random"]["best_ppl_distribution"]]
                    lower.append(weighted_quantile(dist, 0.05))
                    upper.append(weighted_quantile(dist, 0.95))
                    median.append(weighted_quantile(dist, 0.50))
                if row == 0:
                    ax.fill_between(xs, lower, upper, color=P["orange"],
                                    alpha=0.16, linewidth=0, zorder=2)
                    ax.plot(xs, median, color=P["orange"], linestyle=style,
                            linewidth=1.4, zorder=3)
            if row == 1:
                regrets = []
                for cap in CAPS:
                    case = index[(cap, 1)]
                    dist = [(e["ppl"], e["trials"]) for e in
                            case["methods"]["random"]["best_ppl_distribution"]]
                    regrets.append(weighted_quantile(dist, 0.50)
                                   - case["oracle_ppl"])
                note(ax, "random selection, median regret at\n"
                         "10/20/30/50M: "
                         + ", ".join(f"{v:.1f}" for v in regrets),
                     "upper right", size=6.8)
            ax.set_xticks(xs)
            ax.set_xlabel("total-parameter cap (millions)")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.55)
            ax.set_axisbelow(True)
            despine(ax)
            panel_title(ax, f"{ylabel[0].upper() + ylabel[1:]} ({split})",
                        "$k = 1$ solid, $k = 3$ dashed")

    handles = [Line2D([], [], color=colour, marker=marker, markersize=4.6,
                      label=label)
               for _key, label, colour, marker in SELECT_METHODS]
    handles.append(Line2D([], [], color=P["orange"], lw=4, alpha=0.45,
                           label="random, 5-95% of 200 trials"))
    handles.append(Line2D([], [], color=P["ink"], lw=1.2, linestyle=(0, (5, 2)),
                           label="oracle (best feasible)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.105,
             "All methods choose from the same feasible pool before any "
             "outcome is revealed. Regret is the selected PPL minus the "
             "oracle PPL of that pool; lower is better.",
             ha="center", va="bottom", fontsize=7.2, color=P["ink_soft"])
    return fig


# ==========================================================================
# Figure 6 - exact decomposition of the added prefix term
# ==========================================================================
def fig6_decomposition(ev, an, mc):
    """T = U + P + R: what the added term actually contains."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.3),
                             gridspec_kw={"width_ratios": [0.9, 1.1, 1.0]})
    fig.subplots_adjust(wspace=0.42, top=0.84, bottom=0.26)
    shares = {}
    for split in ("development", "final"):
        rows = an["splits"][split]["decomposition"]["by_architecture"]
        values = []
        for key in sorted(rows):
            row = rows[key]
            total = row["prefix_term"]
            if not total:
                continue
            values.append({
                "u": abs(row["ffn_invariant"]) / abs(total),
                "p": abs(row["position"]) / abs(total),
                "r": abs(row["head_residual"]) / abs(total),
            })
        shares[split] = values

    # ---- panel (a): mean composition -----------------------------------
    ax = axes[0]
    segments = (("u", P["blue"], "$U$ (FFN-invariant)"),
                ("p", P["purple"], "$P$ (positional)"),
                ("r", P["orange"], "$R$ (head residual)"))
    for index, split in enumerate(("development", "final")):
        rows = shares[split]
        means = {key: 100 * sum(r[key] for r in rows) / len(rows)
                 for key, _c, _l in segments}
        left = 0.0
        for key, colour, label in segments:
            value = means[key]
            ax.barh(index, value, left=left, color=colour, zorder=2,
                    edgecolor=P["white"], linewidth=0.5,
                    label=label if index == 0 else None)
            if value > 3.0:
                ax.text(left + value / 2, index, f"{value:.2f}%", ha="center",
                        va="center", fontsize=7.0, color=P["white"])
            else:
                ax.annotate(f"$R$ = {value:.3f}%", xy=(left + value / 2, index),
                            xytext=(0, 13), textcoords="offset points",
                            ha="center", fontsize=6.6, color=colour,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor=P["white"], edgecolor="none",
                                      alpha=0.92))
            left += value
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["development", "final"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0, 104)
    ax.set_xlabel("share of the added term $T$ (%)")
    ax.grid(axis="x", alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    panel_title(ax, "Composition of $T$", "mean over 100 architectures")
    handles = [Line2D([], [], color=colour, lw=6, label=label)
               for _key, colour, label in segments]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))

    # ---- panel (b): ECDF of |R| / T -------------------------------------
    ax = axes[1]
    for split, colour, marker, dy in (("development", P["blue"], "o", -0.16),
                                      ("final", P["orange"], "s", 0.06)):
        ratios = sorted(100 * r["r"] for r in shares[split])
        xs = [0.0] + ratios
        ys = [0.0] + [(i + 1) / len(ratios) for i in range(len(ratios))]
        ax.step(xs, ys, where="post", color=colour, linewidth=1.4,
                label=split)
        median = ratios[len(ratios) // 2]
        ax.plot([median], [0.5], marker=marker, markersize=5.5,
                markerfacecolor=P["white"], markeredgecolor=colour,
                markeredgewidth=1.2, linestyle="none")
        ax.annotate(f"median {median:.4f}%", xy=(median, 0.5),
                    xytext=(median + 0.035, 0.5 + dy), fontsize=6.8,
                    color=colour, ha="left",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.7))
    ax.set_xlabel("$|R| / T$ (%)")
    ax.set_ylabel("cumulative share of architectures")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    panel_title(ax, "Head-residual contribution",
                "removing $R$ leaves every rank unchanged")
    ax.legend(loc="lower right", fontsize=7.2)

    # ---- panel (c): per-architecture strip ------------------------------
    ax = axes[2]
    rng = random.Random(20261119)
    for index, split in enumerate(("development", "final")):
        ratios = [100 * r["r"] for r in shares[split]]
        jitter = [rng.uniform(-0.17, 0.17) for _ in ratios]
        ax.scatter([index + j for j in jitter], ratios, s=9,
                   color=P["blue"] if index == 0 else P["orange"],
                   alpha=0.55, edgecolors="none", zorder=3)
        median = sorted(ratios)[len(ratios) // 2]
        ax.plot([index - 0.30, index + 0.30], [median] * 2, color=P["ink"],
                linewidth=1.8, zorder=5)
        ax.text(index, max(ratios) * 1.10, f"max {max(ratios):.3f}%",
                ha="center", va="bottom", fontsize=6.8, color=P["ink_soft"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["development", "final"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.015, max(max(100 * r["r"] for r in shares["final"]),
                            max(100 * r["r"] for r in shares["development"])) * 1.25)
    ax.set_ylabel("$|R| / T$ (%)")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    panel_title(ax, "Per-architecture residual", "thick line = median")
    return fig


# ==========================================================================
# Figure 7 - cold and warm scoring cost
# ==========================================================================
PRETTY_NAMES = {
    "original_nsc": "original NSC",
    "student": "student score",
    "ablation": "ablation",
    "position_only_control": "position-only control",
    "position_example": "position example",
    "reference_flops": "reference FLOPs",
    "reported_parameters": "reported parameters",
    "sensitivity_a0.25": "prefix $\\alpha$ = 0.25",
    "sensitivity_a0.5": "prefix $\\alpha$ = 0.5",
    "sensitivity_a1.0": "prefix $\\alpha$ = 1.0",
    "position_control_a0.25": "position control 0.25",
    "position_control_a0.5": "position control 0.5",
    "position_control_a1.0": "position control 1.0",
    "without_head_residual_a0.25": "prefix 0.25, no $R$",
    "without_head_residual_a0.5": "prefix 0.5, no $R$",
    "without_head_residual_a1.0": "prefix 1.0, no $R$",
}


def fig7_scoring_cost(ev, an, mc):
    """Warm scoring cost per method against the shared cold quadrature cost."""
    cost = ev["scoring_cost"]["by_method"]
    names = sorted(cost, key=lambda n: -cost[n]["median_warm_seconds"])
    warm = [cost[name]["median_warm_seconds"] for name in names]
    colds = sorted(cost[name]["median_cold_seconds"] for name in names)
    cold_median = colds[len(colds) // 2]

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.85, bottom=0.14)
    ys = list(range(len(names)))
    # Dot plot, not bars: a bar on a log axis would have no meaningful base.
    ax.plot(warm, ys, linestyle="none", marker="o", markersize=5.6,
            color=P["green"], markeredgecolor=P["white"], markeredgewidth=0.7,
            zorder=4)
    for y, value in zip(ys, warm):
        ax.annotate(f"{value * 1000:.1f} ms", xy=(value, y), xytext=(7, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=6.9, color=P["ink_soft"], zorder=5)
    ax.axvline(cold_median, color=P["orange"], linestyle=(0, (5, 2)),
               linewidth=1.3, zorder=3)
    ax.text(cold_median * 0.80, -1.35,
            f"cold panel, median {cold_median:.3f} s\n"
            "(shared MP quadrature;\nnearly identical for every method)",
            va="bottom", ha="right", fontsize=6.9, color=P["orange"], zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY_NAMES.get(name, name.replace("_", " "))
                        for name in names])
    ax.set_ylim(-1.7, len(names) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(min(warm) * 0.52, cold_median * 1.45)
    ax.set_xlabel("median warm seconds to score all 200 configurations "
                  "(log scale)")
    ax.grid(axis="x", alpha=0.55)
    ax.set_axisbelow(True)
    despine(ax)
    environment = ev.get("environment", {})
    panel_title(ax, "Scoring cost per method",
                f"{environment.get('processor', 'CPU')}, Python "
                f"{environment.get('python', '?')}; recorded panel wall time "
                f"{environment.get('total_wall_seconds', 0):.1f} s")
    return fig


# ==========================================================================
# Figure catalogue: stem, one-line title, report-ready caption
# ==========================================================================
FIGURES = {
    "fig1": {
        "stem": "fig1_alpha_sweep",
        "title": "Prefix reweighting never improves ranking",
        "builder": fig1_alpha_sweep,
        "caption": (
            "Spearman rank correlation against negative perplexity for the "
            "prefix-capacity reweighting $S_a = S_0 + a\\sum_l C_l q_l$ and "
            "its position-only control, over the frozen grid "
            "$a \\in \\{0, 0.25, 0.5, 1\\}$. The dashed line is the original "
            "NSC score ($a = 0$); the band is a 95\\% paired-architecture "
            "bootstrap interval for the prefix score (1000 resamples). Every "
            "positive $a$ lowers the correlation on both the development "
            "split (validation PPL) and the held-out final split (test PPL), "
            "so the development protocol selects $a = 0$. The vertical axis "
            "is zoomed."),
        "source": "evaluation_summary.json (ranking, "
                  "frozen.sensitivity_analysis); analysis_summary.json "
                  "(paired_architecture_bootstrap)",
    },
    "fig2": {
        "stem": "fig2_ranking_baselines",
        "title": "Ranking quality of every method and baseline",
        "builder": fig2_ranking_baselines,
        "caption": (
            "Spearman $\\rho$ over all pairs for all sixteen evaluated "
            "methods, with the size-controlled restricted $\\tau_b$ (same "
            "hidden width, total-parameter ratio $\\geq 0.90$) marked as a "
            "diamond. Points are grouped into the original NSC and $a = 0$ "
            "variants, the prefix variants ($a > 0$), the positional controls "
            "and the resource baselines. The $a = 0$ variants coincide "
            "because the selected weight is zero; reported parameters are "
            "clearly the weakest baseline on both splits, while reference "
            "FLOPs is close to NSC."),
        "source": "evaluation_summary.json (ranking)",
    },
    "fig3": {
        "stem": "fig3_mechanism_reorder",
        "title": "The score does respond to layer order",
        "builder": fig3_mechanism,
        "caption": (
            "Structural sensitivity of the scores on the 41 synthetic "
            "reorderings; these configurations have no trained outcomes and "
            "test sensitivity only. Left: the largest absolute score change "
            "at $a = 1$ per family. The original NSC changes by exactly 0 on "
            "every case because its layer terms are separable and summed. "
            "Right: head-only reorderings, which leave every FFN width "
            "untouched; the position-only control is blind to them (change "
            "exactly 0) while the prefix term is not, which isolates the "
            "attention-prefix mechanism."),
        "source": "mechanism_check.json (case_details)",
    },
    "fig4": {
        "stem": "fig4_nsc_flops_collinearity",
        "title": "NSC is almost a restatement of compute",
        "builder": fig4_collinearity,
        "caption": (
            "Recorded perplexity (left) and the original NSC score (right) "
            "against reference FLOPs per token, coloured by hidden width, on "
            "both splits. The score is nearly a deterministic function of "
            "compute ($\\rho \\approx 0.999$), and so is the recorded "
            "outcome, so the score carries little information beyond model "
            "size on this archive. Reference FLOPs are a dense-operation "
            "estimate, not measured latency."),
        "source": "analysis_summary.json (architecture_profiles, "
                  "resource_diagnostics)",
    },
    "fig5": {
        "stem": "fig5_selection_regret",
        "title": "Budgeted architecture selection under parameter caps",
        "builder": fig5_selection,
        "caption": (
            "Finite-pool selection under total-parameter caps of 10M, 20M, "
            "30M and 50M. Top: mean best selected PPL; the dashed line is "
            "the best feasible model in each pool. Bottom: mean regret to "
            "that oracle. Solid lines are $k = 1$, dashed lines are $k = 3$. "
            "The orange band spans the 5-95\\% percentile of 200 fixed-seed "
            "random-selection trials. All methods choose from the same "
            "feasible pool before any outcome is revealed; the variability "
            "shown for random selection is over those trials, not over "
            "training seeds."),
        "source": "analysis_summary.json (splits.*.selection_cases)",
    },
    "fig6": {
        "stem": "fig6_decomposition",
        "title": "What the added prefix term actually contains",
        "builder": fig6_decomposition,
        "caption": (
            "Exact decomposition of the added term $T = \\sum_l C_l q_l$ into "
            "an FFN-permutation-invariant part $U$, a positional part $P$ and "
            "a heterogeneous-attention residual $R$. The invariant part "
            "dominates, the positional part is next, and $R$ contributes a "
            "median of about 0.01\\% of $T$ (at most about 0.3\\%). Because "
            "$R$ is negligible, removing it leaves every rank unchanged at "
            "$a \\in \\{0.25, 0.5, 1\\}$ on both splits, which is why the "
            "structural change does not translate into a better ranking."),
        "source": "analysis_summary.json (splits.*.decomposition)",
    },
    "fig7": {
        "stem": "fig7_scoring_cost",
        "title": "Cold and warm scoring cost per method",
        "builder": fig7_scoring_cost,
        "caption": (
            "Warm scoring time per method for all 200 configurations, after "
            "the Marchenko-Pastur quadrature cache is populated. The dashed "
            "line marks the median cold-panel time, which is dominated by "
            "the shared quadrature and is therefore nearly identical for "
            "every method. Warm time exposes scorer overhead: the "
            "without-$R$ diagnostics cost roughly six times the original "
            "NSC scorer. These are scorer diagnostics, not model inference "
            "latencies, and the timings share one cache, so they are not a "
            "fair cold-start comparison between methods."),
        "source": "evaluation_summary.json (scoring_cost.by_method)",
    },
}


def write_caption_file(out_dir: Path, written: dict) -> Path:
    """Emit a report-ready caption sheet next to the figures."""
    path = out_dir / "FIGURE_CAPTIONS.md"
    lines = ["# Report figures - captions and sources", "",
             "Ready to paste into the final report (Times New Roman 12 pt, "
             "A4, 2.5 cm margins).  Each figure is written as PNG (300 dpi) "
             "and as a vector PDF; the PDF is the one to include.", ""]
    for key in FIGURES:
        if key not in written:
            continue
        spec = FIGURES[key]
        stem = spec["stem"]
        lines += [f"## {stem}.png / {stem}.pdf", "",
                  f"**{spec['title']}**", "", spec["caption"], "",
                  f"*Source:* {spec['source']}", "",
                  "```latex",
                  "\\begin{figure}[t]",
                  "    \\centering",
                  f"    \\includegraphics[width=0.95\\textwidth]{{figures/{stem}.pdf}}",
                  f"    \\caption{{{spec['caption']}}}",
                  f"    \\label{{fig:{key}}}",
                  "\\end{figure}",
                  "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    global DATA_DIR
    parser = argparse.ArgumentParser(
        description="Render the report figures from the committed tables.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="directory holding the committed result tables")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="directory to write the figures into")
    parser.add_argument("--formats", default="png,pdf",
                        help="comma separated: png,pdf")
    parser.add_argument("--dpi", type=int, default=300,
                        help="raster resolution for PNG output (default 300)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="figure ids to build (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="list the figures and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available figures:")
        for key, spec in FIGURES.items():
            print(f"  {key:6s} {spec['stem']:26s} {spec['title']}")
        return 0

    DATA_DIR = args.data_dir
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in ("png", "pdf")]
    if unknown:
        parser.error(f"unsupported format(s): {', '.join(unknown)}")

    apply_style(args.dpi)
    print(f"matplotlib {matplotlib.__version__} | serif font: {font_note()}")

    ev = load("evaluation_summary.json")
    an = load("analysis_summary.json")
    mc = load("mechanism_check.json")

    wanted = list(FIGURES) if not args.only else list(args.only)
    missing = [key for key in wanted if key not in FIGURES]
    if missing:
        parser.error(f"unknown figure id(s): {', '.join(missing)}")

    written = {}
    total = time.perf_counter()
    for key in wanted:
        spec = FIGURES[key]
        started = time.perf_counter()
        fig = spec["builder"](ev, an, mc)
        paths = save(fig, spec["stem"], args.out_dir, formats, args.dpi)
        plt.close(fig)
        elapsed = time.perf_counter() - started
        written[key] = paths
        files = ", ".join(p.name for p in paths)
        print(f"  {spec['stem']:26s} {elapsed:5.2f}s  -> {files}")
    total = time.perf_counter() - total

    caption_path = write_caption_file(args.out_dir, written)
    print(f"wrote {len(written)} figure set(s) in {total:.2f}s "
          f"+ captions: {caption_path.name}")
    print(f"output directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
