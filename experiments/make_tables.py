#!/usr/bin/env python3
"""Render the report tables from the committed result tables.

The tables are built from the committed tables under ``experiments/data/``
only.  No label file is read here, nothing is tuned and no outcome is used to
choose a cell value: every number printed is a number that ``EVALUATION.md``
already reports.  The script never touches ``nsc-starter/data/*_labels.json``.

Each table is written three ways:

* ``tables/<stem>.md``   -- pipe table for Markdown viewers and for pasting
  into Word or Google Docs
* ``tables/<stem>.tex``  -- booktabs table for a LaTeX report
* ``tables/<stem>.csv``  -- the same cells as a spreadsheet

``tables/TABLES.md`` collects every table with its caption, a LaTeX snippet
and a note on whether it belongs in the four-page report or in the appendix.

Usage
-----
    python experiments/make_tables.py                  # every table -> tables/
    python experiments/make_tables.py --list           # list the tables
    python experiments/make_tables.py --only table1 table3
    python experiments/make_tables.py --out-dir build/tables
    python experiments/make_tables.py --formats md tex

Requirements: the standard library only.  The core harness in
``nsc-starter/`` and this script need no third-party package.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "experiments" / "data"
DEFAULT_OUT = ROOT / "tables"

SPLITS = ("development", "final")
SPLIT_LABEL = {"development": "development", "final": "final"}
SPLIT_SHORT = {"development": "dev", "final": "final"}
CAPS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)

METHOD_LABELS = {
    "original_nsc": "original NSC",
    "student": "student (ours, $\\alpha = 0$)",
    "ablation": "ablation ($\\alpha = 0$)",
    "position_only_control": "position-only ctrl ($\\alpha = 0$)",
    "position_example": "position example",
    "position_control_a0.25": "pos. control $\\alpha = 0.25$",
    "position_control_a0.5": "pos. control $\\alpha = 0.5$",
    "position_control_a1.0": "pos. control $\\alpha = 1.0$",
    "sensitivity_a0.25": "prefix $\\alpha = 0.25$",
    "sensitivity_a0.5": "prefix $\\alpha = 0.5$",
    "sensitivity_a1.0": "prefix $\\alpha = 1.0$",
    "without_head_residual_a0.25": "prefix 0.25, no $R$",
    "without_head_residual_a0.5": "prefix 0.5, no $R$",
    "without_head_residual_a1.0": "prefix 1.0, no $R$",
    "reported_parameters": "reported params",
    "reference_flops": "reference FLOPs",
}
# Display order of the method table: baselines first, then the proposed
# variants, so the reader meets the reference points before ours.
METHOD_ORDER = [
    "original_nsc", "position_example", "student", "ablation",
    "position_only_control", "sensitivity_a0.25", "sensitivity_a0.5",
    "sensitivity_a1.0", "without_head_residual_a0.25",
    "without_head_residual_a0.5", "without_head_residual_a1.0",
    "position_control_a0.25", "position_control_a0.5", "position_control_a1.0",
    "reported_parameters", "reference_flops",
]


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def f4(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def f6(value) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def f6s(value) -> str:
    """Signed fixed-point with an explicit plus sign, 6 decimals."""
    if value is None:
        return "n/a"
    return f"{float(value):+.6f}"


def f2(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def f4s(value) -> str:
    """Signed fixed-point with an explicit plus sign."""
    if value is None:
        return "n/a"
    value = float(value)
    return f"{value:+.4f}"


def sci(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}e}"


def integer(value) -> str:
    return "-" if value is None else f"{int(value):d}"


def pct(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{100 * float(value):.{digits}f}%"


def latex_escape(text: str) -> str:
    """Escape a plain string for LaTeX.  Math already in $...$ is kept."""
    out = []
    in_math = False
    for char in text:
        if char == "$":
            in_math = not in_math
            out.append(char)
            continue
        if in_math:
            out.append(char)
            continue
        out.append({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}",
                    "\\": r"\textbackslash{}", "|": r"\textbar{}"}.get(char, char))
    return "".join(out)


# A4 with 2.5 cm margins leaves 16 cm of text width, in points.
TEXT_WIDTH_PT = 16.0 / 2.54 * 72.0
# Width of the padding on each side of a cell (\tabcolsep).
TABCOLSEP_PT = 4.0


def estimate_width_pt(header, rows, font_pt: float = 10.0) -> float:
    """Conservative estimate of the natural width of a table.

    Used only to decide whether the LaTeX output has to be scaled down to
    the text width; it is never used to change a reported value.
    """
    def text_width(text: str) -> float:
        width = 0.0
        in_math = False
        for chunk in text.split("$"):
            if in_math:
                # A LaTeX command is one glyph, not one glyph per character.
                for part in re.split(r"(\\[a-zA-Z]+)", chunk):
                    if not part:
                        continue
                    if part.startswith("\\"):
                        width += 0.60 * font_pt
                    elif part.isdigit():
                        width += 0.50 * font_pt
                    elif part.isalpha():
                        width += 0.52 * font_pt
                    else:
                        width += 0.38 * font_pt
            else:
                for char in chunk:
                    if char == " ":
                        width += 0.25 * font_pt
                    elif char.isdigit():
                        width += 0.50 * font_pt
                    elif char.isupper():
                        width += 0.62 * font_pt
                    elif char.isalpha():
                        width += 0.47 * font_pt
                    else:
                        width += 0.34 * font_pt
            in_math = not in_math
        return width

    columns = [text_width(str(h)) for h in header]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(columns):
                columns[index] = max(columns[index], text_width(str(cell)))
    return sum(columns) + 2 * TABCOLSEP_PT * len(columns)


def load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_markdown(table: dict) -> str:
    header = table["header"]
    align = table["align"]
    rule = ["---"] * len(header)
    lines = [f"**{table['title']}**", "", table["caption"], ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(rule) + "|")
    for row in table["rows"]:
        lines.append("| " + " | ".join(cell.replace("|", "\\|")
                                       for cell in row) + " |")
    lines.append("")
    for note in table.get("notes", []):
        lines.append(f"- {note}")
    if table.get("notes"):
        lines.append("")
    return "\n".join(lines)


def render_latex(table: dict) -> str:
    header = table["header"]
    align = table["align"]
    spec = "".join({"l": "l", "r": "r", "c": "c"}[a] for a in align)
    natural = estimate_width_pt(header, table["rows"])
    scale = TEXT_WIDTH_PT / natural if natural else 1.0
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             f"\\caption{{{latex_escape(table['caption'])}}}",
             f"\\label{{tab:{table['id']}}}", "\\setlength{\\tabcolsep}{4pt}"]
    if scale < 1.0:
        # The table is wider than the text block: scale it to fit exactly.
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
              " & ".join(latex_escape(h) for h in header) + r" \\",
              "\\midrule"]
    for row in table["rows"]:
        lines.append(" & ".join(latex_escape(str(c)) for c in row) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if scale < 1.0:
        lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def render_csv(table: dict) -> str:
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table["header"])
    for row in table["rows"]:
        writer.writerow(row)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------
def table1_ranking(ev, an, mc, bm):
    """Starter protocol steps 2 and 3: ranking of every method and baseline."""
    rows = []
    for key in METHOD_ORDER:
        dev = ev["ranking"]["development"][key]
        fin = ev["ranking"]["final"][key]
        rows.append([
            METHOD_LABELS[key],
            f6(dev["spearman_rho"]),
            f6(dev["all_pairs"]["tau_b"]),
            f6(dev["size_controlled_pairs"]["tau_b"]),
            f6(fin["spearman_rho"]),
            f6(fin["all_pairs"]["tau_b"]),
            f6(fin["size_controlled_pairs"]["tau_b"]),
            f"{integer(dev['size_controlled_pairs']['pairs'])}/"
            f"{integer(fin['size_controlled_pairs']['pairs'])}",
        ])
    return {
        "id": "table1",
        "stem": "table1_ranking_baselines",
        "title": "Ranking of every method and required baseline",
        "priority": "report",
        "caption": (
            "Spearman $\\rho$ and Kendall $\\tau_b$ against negative "
            "perplexity over all pairs, and the size-controlled restricted "
            "$\\tau_b$ (same hidden width, total-parameter ratio "
            "$\\min/\\max \\geq 0.90$), on both splits. The last column "
            "is the number of restricted pairs, which overlap and are not an "
            "independent sample. Score-only tied pairs: none, except one pair "
            "for reference FLOPs on the development split."),
        "header": ["Method", "dev $\\rho$", "dev $\\tau_b$",
                   "dev restr. $\\tau_b$", "final $\\rho$", "final $\\tau_b$",
                   "final restr. $\\tau_b$", "restr. pairs d/f"],
        "align": "l" + "r" * 7,
        "rows": rows,
        "notes": [
            "$\\alpha = 0$ was selected on the development split, so the "
            "student score, its ablation and the position-only control all "
            "coincide with the original NSC and carry no nonzero mechanism "
            "test.",
            "Reported parameters is clearly the weakest baseline; reference "
            "FLOPs is close to NSC on both splits.",
            "The restricted pair statistic is a $\\tau_b$ on the restricted "
            "set, not a standard full-panel correlation and not a causal "
            "control.",
        ],
    }


def table2_alpha(ev, an, mc, bm):
    """The frozen development choice and its sensitivity analysis."""
    rows = []
    for entry in ev["frozen"]["sensitivity_analysis"]:
        alpha = entry["alpha"]
        if alpha == 0.0:
            prefix = (ev["ranking"]["development"]["original_nsc"],
                      ev["ranking"]["final"]["original_nsc"])
            position = (ev["ranking"]["development"]["original_nsc"],
                        ev["ranking"]["final"]["original_nsc"])
            residual = (ev["ranking"]["development"]["original_nsc"],
                        ev["ranking"]["final"]["original_nsc"])
        else:
            prefix = (ev["ranking"]["development"][f"sensitivity_a{alpha}"],
                      ev["ranking"]["final"][f"sensitivity_a{alpha}"])
            position = (ev["ranking"]["development"][f"position_control_a{alpha}"],
                        ev["ranking"]["final"][f"position_control_a{alpha}"])
            residual = (
                ev["ranking"]["development"][f"without_head_residual_a{alpha}"],
                ev["ranking"]["final"][f"without_head_residual_a{alpha}"])
        rows.append([
            f"{alpha:g}",
            f6(prefix[0]["spearman_rho"]),
            f6(position[0]["spearman_rho"]),
            f6(residual[0]["spearman_rho"]),
            f6(prefix[1]["spearman_rho"]),
            f6(position[1]["spearman_rho"]),
            f6(residual[1]["spearman_rho"]),
            f4(entry["kendall_tau_b"]),
            f"{entry['score_range'][0]:.0f} - {entry['score_range'][1]:.0f}",
            integer(entry["score_ties"]),
        ])
    return {
        "id": "table2",
        "stem": "table2_alpha_sensitivity",
        "title": "Sensitivity of the ranking to the prefix weight $\\alpha$",
        "priority": "report",
        "caption": (
            "Spearman $\\rho$ of the prefix reweighting, its position-only "
            "control and the variant without the head residual $R$, over the "
            "frozen grid $\\alpha \\in \\{0, 0.25, 0.5, 1\\}$. The six "
            "middle columns are Spearman $\\rho$ of the prefix score, its "
            "position-only control and the variant without $R$; the last "
            "three are the development Kendall $\\tau_b$, the development "
            "score range and the score-tie count of the prefix score. The "
            "choice was frozen on the development split alone."),
        "header": ["$\\alpha$", "dev prefix", "dev position", "dev no-$R$",
                   "final prefix", "final position", "final no-$R$",
                   "dev $\\tau_b$", "dev score range", "ties"],
        "align": "l" + "r" * 9,
        "rows": rows,
        "notes": [
            "Every prescribed positive weight has a lower global "
            "correlation than the original NSC on both panels, so the "
            "development rule selects $\\alpha = 0$.",
            "The score range is the minimum and maximum of the prefix score "
            "over the development split. No pair was tied on both the score "
            "and the outcome for any setting.",
        ],
    }


def _selection_rows(ev, an, mc, bm, metric, with_oracle=False):
    """One row per (split, cap, k) with the requested selection metric."""
    rows = []
    for split in SPLITS:
        cases = {(case["parameter_cap"], case["k"]): case
                 for case in an["splits"][split]["selection_cases"]}
        for cap in CAPS:
            for k in (1, 3):
                case = cases[(cap, k)]
                methods = case["methods"]
                cells = []
                for key in ("original_nsc", "position_example",
                            "reported_parameters", "reference_flops"):
                    cells.append(f4(methods[key][metric]))
                random = methods["random"]
                cells.append(f4(random[metric]))
                cells.append(f4(random["sd_best_ppl"]))
                row = [
                    SPLIT_SHORT[split],
                    f"{cap // 1000000}M",
                    integer(k),
                    integer(case["eligible"]),
                ]
                if with_oracle:
                    row.append(f4(case["oracle_ppl"]))
                rows.append(row + cells)
    return rows


def table3_selection(ev, an, mc, bm):
    """Starter protocol step 4: budgeted selection, split by metric.

    The best selected PPL and the regret are emitted as two tables so that
    each stays at the full report font size; they share the same rows.
    """
    common = ("Split", "Cap", "$k$", "Pool")
    methods = ("NSC = ours", "Pos. example", "Params", "Ref. FLOPs",
               "Random", "SD")
    best = {
        "id": "table3",
        "stem": "table3_selection_ppl",
        "title": "Budgeted architecture selection: mean best selected PPL",
        "priority": "report",
        "caption": (
            "Finite-pool selection under total-parameter caps of 10M, 20M, "
            "30M and 50M, for $k = 1$ and $k = 3$. Each cell is the mean best "
            "selected PPL over the 200 fixed-seed trials; lower is better. "
            "The oracle column is the best PPL anywhere in the feasible pool. "
            "The NSC column is also the student score, its ablation and the "
            "position-only control, which coincide at the frozen $\\alpha = 0$. "
            "The random column is the mean over 200 fixed-seed trials and SD "
            "its standard deviation, which measures the variability of the "
            "random choice and not training-seed uncertainty."),
        "header": list(common) + ["Oracle"] + list(methods),
        "align": "l" + "r" * 10,
        "rows": [],
        "notes": [
            "The baseline caps and both values of $k$ are reported for every "
            "split, as the protocol requires.",
        ],
    }
    regret = {
        "id": "table3b",
        "stem": "table3_selection_regret",
        "title": "Budgeted architecture selection: mean regret",
        "priority": "report",
        "caption": (
            "Mean regret of the same selections: the selected PPL minus the "
            "best feasible PPL in that pool, so zero means the oracle was "
            "found and lower is better. The rows match the selection table "
            "one for one."),
        "header": list(common) + list(methods),
        "align": "l" + "r" * 10,
        "rows": [],
        "notes": [
            "SD is the standard deviation of the random selection over the "
            "200 trials, not of the regret.",
        ],
    }
    best["rows"] = _selection_rows(ev, an, mc, bm, "mean_best_ppl",
                                   with_oracle=True)
    regret["rows"] = _selection_rows(ev, an, mc, bm, "mean_regret")
    return [best, regret]


def table4_mechanism(ev, an, mc, bm):
    """Starter protocol step 1: the mechanism check."""
    limiting = mc.get("limiting_cases", [])
    rows = [
        ["Mechanism status", mc.get("status", "?"),
         "synthetic reorderings only"],
        ["Synthetic reorder cases", integer(mc.get("cases")),
         "no trained outcomes"],
        ["Head-only cases", integer(mc.get("head_only_cases")), "-"],
        ["Max $\\lvert\\Delta\\rvert$ NSC reorder change",
         f"{float(mc['max_abs_nsc_delta']):.9f}", "NSC is additive"],
        ["Max $\\lvert\\Delta\\rvert$ prefix reorder change, $\\alpha = 1$",
         f"{float(mc['max_abs_ours_delta_a1']):.6f}", "-"],
        ["Max $\\lvert\\Delta\\rvert$ head-only: prefix / position, $\\alpha = 1$",
         f"{float(mc['max_head_only_ours_delta_a1']):.6f} / "
         f"{float(mc['max_head_only_position_control_delta_a1']):.6f}",
         "position ctrl blind to head-only"],
        ["Resource estimates invariant under reordering",
         str(mc.get("all_resource_estimates_invariant")), "under any reorder"],
        ["Single-layer $C_1 = 0$ limiting cases",
         f"{integer(len(limiting))} cases, "
         f"{'all passed' if all(c.get('passed') for c in limiting) else 'FAILED'}",
         "prescribed $\\alpha$, 4 widths"],
        ["Mechanism failures", integer(len(mc.get("failures", []))), "-"],
        ["Homogeneous layers approach positional weights",
         str(mc.get("homogeneous_layers_recover_position_weights")),
         "$\\epsilon$ correction"],
        ["Constant-head prefix/position identity",
         f"{integer(mc.get('constant_head_position_identity_cases'))} cases",
         f"max error {sci(mc.get('max_constant_head_position_identity_error'))}"],
        ["Max scorer reconstruction error",
         sci(mc.get("max_abs_scorer_reconstruction_error")), "-"],
    ]
    return {
        "id": "table4",
        "stem": "table4_mechanism",
        "title": "Mechanism check on the synthetic reorderings",
        "priority": "report",
        "caption": (
            "Checks that the prefix term responds to layer order while the "
            "additive NSC does not, that the $a = 0$ ablation recovers the "
            "original score exactly, and that the resource estimates stay "
            "invariant under reordering. The synthetic configurations carry "
            "no trained outcomes, so this table tests sensitivity only and "
            "says nothing about predictive accuracy."),
        "header": ["Check", "Result", "Note"],
        "align": "lrl",
        "rows": rows,
        "notes": [
            "A reorder response alone is weaker evidence than a predictive "
            "result; see Tables 1-3 for the latter.",
        ],
    }


def table5_decomposition(ev, an, mc, bm):
    """What the added term contains, and how much the residual matters."""
    rows = []
    for split in SPLITS:
        d = an["splits"][split]["decomposition"]
        ranks = d.get("ranks_equal_without_head_residual", {})
        rows.append([
            SPLIT_LABEL[split],
            sci(d.get("max_abs_identity_error")),
            integer(d.get("adjacent_swaps_checked")),
            sci(d.get("max_adjacent_swap_identity_error")),
            sci(d.get("max_abs_scorer_reconstruction_error")),
            pct(d.get("residual_fraction_median"), 5),
            pct(d.get("residual_fraction_max"), 5),
            "/".join(str(ranks.get(a, "?")) for a in ("0.25", "0.5", "1.0")),
        ])
    return {
        "id": "table5",
        "stem": "table5_decomposition",
        "title": "Exact decomposition of the added prefix term",
        "priority": "report",
        "caption": (
            "The added term decomposes exactly into $T = U + P + R$, an "
            "FFN-permutation-invariant part, a positional part and a "
            "heterogeneous-attention residual. The last column records "
            "whether every rank is unchanged when $R$ is removed, for "
            "$a \\in \\{0.25, 0.5, 1\\}$. A tiny residual and unchanged ranks "
            "mean the structural change carries no extra predictive "
            "information on this archive."),
        "header": ["Split", "Max identity err.", "Swaps",
                   "Max swap err.", "Max reconstr. err.",
                   "Median $\\lvert R\\rvert/T$", "Max $\\lvert R\\rvert/T$",
                   "Same ranks w/o $R$"],
        "align": "l" + "r" * 7,
        "rows": rows,
        "notes": [
            "The residual contributes a median of about 0.01\\% of $T$, so "
            "dropping it changes no ranking.",
        ],
    }


def table6_resource_controls(ev, an, mc, bm):
    """Successively stricter size, FLOPs and depth matching."""
    labels = {
        "same_width_parameters": "w+par",
        "same_width_parameters_flops": "w+par+flops",
        "same_width_parameters_flops_depth": "w+par+flops+depth",
    }
    methods = ("original_nsc", "reference_flops", "reported_parameters",
               "sensitivity_a1.0", "position_control_a1.0")
    rows = []
    for split in SPLITS:
        for control, entry in an["splits"][split]["resource_controls"].items():
            rows.append([
                f"{SPLIT_SHORT[split]} / {labels.get(control, control)}",
                integer(entry["pair_count"]),
                integer(entry["covered_architectures"]),
            ] + [f6(entry["ranking"][m]["tau_b"]) for m in methods])
    diagnostics = []
    for split in SPLITS:
        d = an["splits"][split]["resource_diagnostics"]
        diagnostics.append(
            f"{SPLIT_LABEL[split]}: {integer(d['matched_pairs_flops_ratio_below_0_9'])}"
            f" of {integer(d['matched_pairs_flops_ratio_below_0_9'])} matched "
            f"pairs fall below a 0.90 FLOPs ratio (median "
            f"{f4(d['matched_flops_ratio_median'])}, minimum "
            f"{f4(d['matched_flops_ratio_min'])}), "
            f"{integer(d['matched_pairs_different_depth'])} differ in depth, "
            f"and NSC correlates with reference FLOPs at "
            f"$\\rho$ = {f4(d['nsc_flops_spearman_rho'])}.")
    return {
        "id": "table6",
        "stem": "table6_resource_controls",
        "title": "Successively stricter resource matching",
        "priority": "appendix",
        "caption": (
            "Restricted $\\tau_b$ on successively stricter matched subsets. "
            "The prescribed control matches hidden width and total "
            "parameters; the FLOPs and depth controls are additional "
            "diagnostics added after the final outcomes were available, so "
            "they are not causal controls. The last two rows give the "
            "diagnostics of the prescribed parameter-matched pairs."),
        "header": ["Split / control", "Pairs", "Architectures",
                   "original NSC", "Ref. FLOPs", "Params",
                   "Prefix $\\alpha = 1$", "Pos. ctrl $\\alpha = 1$"],
        "align": "l" + "r" * 7,
        "rows": rows,
        "notes": [
            "Matching parameters does not fix FLOPs or depth, so the "
            "restricted statistic keeps a strong resource component.",
            "The FLOPs and depth rows are post-freeze diagnostics and are "
            "reported with all remaining pair and architecture counts.",
        ] + diagnostics,
    }


def table7_within_width(ev, an, mc, bm):
    """Ranking quality inside each hidden-width group."""
    methods = ("original_nsc", "reference_flops", "sensitivity_a1.0",
               "position_control_a1.0")
    rows = []
    for split in SPLITS:
        widths = an["splits"][split]["within_width"]
        for width in sorted(widths, key=int):
            entry = widths[width]
            rows.append([
                f"{SPLIT_LABEL[split]} / {width}",
                integer(entry["architectures"]),
            ] + [f6(entry["ranking"][m]["spearman_rho"]) for m in methods])
    return {
        "id": "table7",
        "stem": "table7_within_width",
        "title": "Ranking quality within each hidden-width group",
        "priority": "appendix",
        "caption": (
            "Spearman $\\rho$ computed inside each hidden-width group, where "
            "model size is most nearly controlled. The pattern of the "
            "all-pairs table survives the width restriction."),
        "header": ["Split / width", "Architectures", "original NSC",
                   "Ref. FLOPs", "Prefix $\\alpha = 1$",
                   "Pos. ctrl $\\alpha = 1$"],
        "align": "l" + "r" * 5,
        "rows": rows,
        "notes": [],
    }


def table8_bootstrap(ev, an, mc, bm):
    """Paired-architecture bootstrap on the contrasts that matter."""
    rows = []
    for split in SPLITS:
        boot = an["splits"][split]["paired_architecture_bootstrap"]
        for name in sorted(boot):
            entry = boot[name]
            ci = entry.get("ci95") or [None, None]
            rows.append([
                f"{SPLIT_LABEL[split]} / {name}",
                f6s(entry.get("estimate_delta_rho")),
                f"[{f6s(ci[0])}, {f6s(ci[1])}]",
                integer(entry.get("valid_replicates")),
            ])
    return {
        "id": "table8",
        "stem": "table8_bootstrap",
        "title": "Paired-architecture bootstrap on the $\\rho$ contrasts",
        "priority": "appendix",
        "caption": (
            "Difference in Spearman $\\rho$ between the two named scores with "
            "a 95\\% percentile interval from 1000 paired bootstrap "
            "resamples of architectures. The unit of resampling is the "
            "architecture, not the pair, so no training-seed interpretation "
            "is implied."),
        "header": ["Split / contrast", "$\\Delta\\rho$", "95\\% interval",
                   "Replicates"],
        "align": "lrrr",
        "rows": rows,
        "notes": [
            "Intervals for the prefix-versus-position contrasts include zero, "
            "so the two are not separated on this archive.",
        ],
    }


def table9_cost(ev, an, mc, bm):
    """Cold and warm scoring cost."""
    cost = ev["scoring_cost"]["by_method"]
    rows = []
    for name in sorted(cost, key=lambda n: -cost[n]["median_cold_seconds"]):
        rows.append([
            METHOD_LABELS.get(name, name.replace("_", " ")),
            f"{cost[name]['median_cold_seconds'] * 1000:.3f}",
            f"{cost[name]['median_warm_seconds'] * 1000:.3f}",
        ])
    environment = ev.get("environment", {})
    return {
        "id": "table9",
        "stem": "table9_scoring_cost",
        "title": "Scoring cost per method",
        "priority": "appendix",
        "caption": (
            "Median time to score all 200 configurations, over three "
            "repetitions per method, with a cold and with a warm "
            "Marchenko-Pastur quadrature cache. Cold time is dominated by the "
            "shared quadrature; warm time exposes scorer overhead. These are "
            "scorer diagnostics, not model inference latencies, and the "
            "timings share one cache."),
        "header": ["Method", "Cold panel (ms)", "Warm panel (ms)"],
        "align": "lrr",
        "rows": rows,
        "notes": [
            f"Measured on {environment.get('processor', 'CPU')} with Python "
            f"{environment.get('python', '?')}; the recorded evaluation wall "
            f"time is {environment.get('total_wall_seconds', 0):.2f} s.",
        ],
    }


def table10_benchmark(ev, an, mc, bm):
    """The data and split composition."""
    rows = []
    for split in SPLITS:
        entry = bm["splits"][split]
        widths = entry["hidden_width_counts"]
        pools = entry["parameter_cap_eligible"]
        if isinstance(pools, dict):
            pool_text = "/".join(str(pools[key]) for key in
                                 sorted(pools, key=lambda k: int(str(k).rstrip("M"))))
        else:
            pool_text = "/".join(str(p) for p in pools)
        low, high = entry["total_parameters_range"]
        rows.append([
            SPLIT_LABEL[split],
            integer(entry["count"]),
            integer(entry["all_pairs"]),
            integer(entry["size_controlled_pairs"]),
            "/".join(str(widths[key]) for key in sorted(widths, key=int)),
            pool_text,
            f"{int(low):d} - {int(high):d}",
            entry["label_field"],
        ])
    return {
        "id": "table10",
        "stem": "table10_benchmark",
        "title": "Benchmark panel and split composition",
        "priority": "report",
        "caption": (
            "The archived GPT-2 / Transformer-XL panel: architectures per "
            "split, all pairs, size-controlled pairs, the number of "
            "architectures per hidden width (64/128/256/512), the feasible "
            "pool size at each parameter cap (10M/20M/30M/50M), the "
            "total-parameter range and the outcome field used. The final "
            "split is an architecture holdout within the same archive, not an "
            "independent benchmark."),
        "header": ["Split", "Architectures", "All pairs", "Restr. pairs",
                   "Widths", "Pools", "Param. range", "Label"],
        "align": "l" + "r" * 6 + "l",
        "rows": rows,
        "notes": [
            "The panel carries one recorded outcome per architecture, without "
            "training seeds, tokens or hardware.",
        ],
    }


BUILDERS = [
    table1_ranking, table2_alpha, table3_selection, table4_mechanism,
    table5_decomposition, table6_resource_controls, table7_within_width,
    table8_bootstrap, table9_cost, table10_benchmark,
]


# --------------------------------------------------------------------------
# Index sheet
# --------------------------------------------------------------------------
def write_index(out_dir: Path, tables: list) -> Path:
    """Collect every table with its caption, sources and LaTeX snippet."""
    path = out_dir / "TABLES.md"
    report = [t for t in tables if t["priority"] == "report"]
    appendix = [t for t in tables if t["priority"] != "report"]
    lines = [
        "# Report tables - captions, sources and LaTeX snippets",
        "",
        "Generated by `experiments/make_tables.py` from the committed result",
        "tables in `experiments/data/`. Every number is already reported in",
        "[EVALUATION.md](../EVALUATION.md); nothing here is recomputed from a",
        "label file.",
        "",
        "Each table is written as `tables/<name>.md` (Markdown, for Word or",
        "Google Docs), `tables/<name>.tex` (booktabs, for LaTeX) and",
        "`tables/<name>.csv` (spreadsheet).",
        "",
        "## How many of these fit in the report",
        "",
        "The final report is **4 pages** and the course counts *figures,",
        "tables and captions* against that limit, so most of these tables",
        "cannot all go in the body. A workable split:",
        "",
        "- **In the report (3 tables):** " +
        ", ".join(f"`{t['stem']}`" for t in report) +
        ". These are the ranking table, the sensitivity table and the",
        "  selection table -- the three the starter protocol requires to be",
        "  reported.",
        "- **In the report (3 figures):** `fig1_alpha_sweep`,",
        "  `fig4_nsc_flops_collinearity` and `fig5_selection_regret`.",
        "- **Appendix or `EVALUATION.md`:** " +
        ", ".join(f"`{t['stem']}`" for t in appendix) +
        ". The full set is already",
        "  in [EVALUATION.md](../EVALUATION.md), so the appendix can point",
        "  there instead of reprinting the tables.",
        "",
        "If the report has to shrink further, Table 10 (the panel",
        "composition) can be folded into one sentence of the data section,",
        "and Table 2 can be merged with Figure 1, which plots the same",
        "numbers.",
        "",
    ]
    for table in tables:
        badge = ("**recommended for the report**" if table["priority"] == "report"
                 else "appendix / EVALUATION.md")
        lines += [
            f"## {table['stem']}",
            "",
            f"{badge}",
            "",
            f"**{table['title']}**",
            "",
            table["caption"],
            "",
            render_markdown(table).split("\n", 3)[3].strip(),
            "",
            "```latex",
            render_latex(table).rstrip(),
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    global DATA_DIR
    parser = argparse.ArgumentParser(
        description="Render the report tables from the committed result tables.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="directory holding the committed result tables")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help="directory to write the tables into")
    parser.add_argument("--formats", default="md,tex,csv",
                        help="comma separated: md,tex,csv")
    parser.add_argument("--only", nargs="*", default=None,
                        help="table ids to build (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="list the tables and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("Available tables:")
        for builder in BUILDERS:
            print(f"  {builder.__name__}")
        return 0

    DATA_DIR = args.data_dir
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = [f for f in formats if f not in ("md", "tex", "csv")]
    if unknown:
        parser.error(f"unsupported format(s): {', '.join(unknown)}")

    ev = load("evaluation_summary.json")
    an = load("analysis_summary.json")
    mc = load("mechanism_check.json")
    bm = load("benchmark_manifest.json")

    wanted = None
    if args.only:
        wanted = set(args.only)
    tables = []
    for builder in BUILDERS:
        built = builder(ev, an, mc, bm)
        if isinstance(built, dict):
            built = [built]
        for table in built:
            if wanted is not None and table["id"] not in wanted:
                continue
            tables.append(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    expected = set()
    for table in tables:
        for row in table["rows"]:
            if len(row) != len(table["header"]):
                raise SystemExit(f"{table['stem']}: row has {len(row)} cells "
                                 f"but the header has {len(table['header'])}")
        for fmt in formats:
            path = args.out_dir / f"{table['stem']}.{fmt}"
            expected.add(path.name)
            if fmt == "md":
                path.write_text(render_markdown(table), encoding="utf-8")
            elif fmt == "tex":
                path.write_text(render_latex(table), encoding="utf-8")
            else:
                path.write_text(render_csv(table), encoding="utf-8")
        print(f"  {table['stem']:28s} {len(table['rows']):3d} rows  "
              f"[{table['priority']}]")
    elapsed = time.perf_counter() - started

    # Remove stale outputs from an earlier run, so the directory always
    # matches the tables this run produced.
    for stale in args.out_dir.iterdir():
        # match table1_..., table2_... but never the TABLES.md index
        if (re.match(r"table\\d+_", stale.name) and stale.suffix in (".md", ".tex", ".csv")
                and stale.name not in expected):
            stale.unlink()
            print(f"  removed stale output: {stale.name}")

    index = write_index(args.out_dir, tables)
    print(f"wrote {len(tables)} table(s) in {elapsed:.2f}s "
          f"+ index: {index.name}")
    print(f"output directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
