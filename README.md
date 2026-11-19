# Structural Sensitivity and Predictive Utility of NSC Reweighting

CS5491 Course Project — 2026/27 Semester A.

Transformer architecture selection is costly when every candidate has to be
trained. Neural Spectral Capacity (NSC) is a training-free alternative, but its
additive form cannot distinguish layer permutations that preserve the same
projections. This project asks whether **prefix-capacity reweighting** adds
predictive information beyond layer position and model size, using the course's
archived panel of 200 GPT-2 / Transformer-XL configurations.

Everything runs on CPU with the supplied Python 3.9+ / standard-library-only
starter. There is no training, GPU, model download or LLM API call.

The submitted proposal is `project_proposal.pdf`. `nsc-starter/` holds the
course harness and our scorer; the supplied baseline, evaluator and raw data
are preserved. Local reference archives can be kept in `reference/` (gitignored).

## Team

| Name | Student ID |
|------|-----------|
| Xiaoyang Yu | 72655540 |
| Guanqi Shao | 72655921 |
| Weihan Yi | 72655496 |

Contact: 72655540@cityu-dg.edu.cn

## Proposed method

Let `B_l` be the supplied per-layer NSC capacity, `q_l = 2 psi_MP(d, f_l, 0.02)`
the FFN term for hidden width `d` and FFN width `f_l`, and
`S0 = sum_l B_l`. With `eps = 1e-12` and depth `L`:

```
C_l = (sum_{j<l} B_j) / (S0 + eps)
S_a = S0 + a * sum_l C_l q_l
```

`C_l` is a prefix-capacity proxy for FFN context. We follow the archive's
post-norm setting, `h_{l+1} = LN(z_l + F_l(z_l))` where `z_l` follows attention,
and retain NSC initialization and projection grouping.

- `a = 0` is the ablation and recovers original NSC.
- `C_l = (l-1)/L` is the position-only control.
- Synthetic FFN reorderings, with fixed and varying head counts, probe
  positional and attention-prefix effects. Limiting cases: NSC recovery at
  `a = 0`, and `C_l = 0` for one layer.
- `a` is chosen from `{0, 0.25, 0.5, 1}` to maximise development Spearman
  against negative perplexity, preferring smaller `a` on ties. All settings
  undergo sensitivity analysis, and **choices are frozen before the final
  split is evaluated**.

Implement `score(config)` and `ablation(config)` in
[`nsc-starter/student_score.py`](nsc-starter/student_score.py). The evaluator
passes configuration only: do not load label files, look up architecture IDs,
or use configuration fingerprints as outcome lookups inside a scorer.

## Benchmark and evaluation protocol

- 200 archived configurations: 100 development cases with validation PPL and
  100 final cases with test PPL. Final outcomes are held out by protocol, not
  by software — do not inspect or tune on them before freezing the method.
- Baselines: original NSC, the supplied positional example, the position-only
  control, the `a = 0` ablation, reported total parameters, reference FLOPs,
  and uniform random selection.
- Ranking: Spearman's rho and Kendall's tau-b against negative PPL, with tie
  counts. Also report restricted tau-b for same-hidden-width pairs whose total
  parameter ratio `min/max >= 0.90`, together with the pair count (overlapping
  pairs; not an independent sample size).
- Selection: parameter caps 10M / 20M / 30M / 50M and `k in {1, 3}`, all
  methods choosing from the same feasible pool before outcomes are revealed.
  Report best selected PPL and regret to the best feasible model, with 200
  fixed-seed trials for random selection and score ties. These are not
  training-seed uncertainties.
- Also report failures, resource effects, cold/warm scoring cost, and the
  actual hardware and elapsed time. Synthetic reorderings test structural
  sensitivity only; they do not imply predictive accuracy.

The starter's [`nsc-starter/README.md`](nsc-starter/README.md) is the
authoritative source for the schema, baseline conventions, resource
arithmetic, provenance and the full evaluation protocol.

## Repository layout

```
.
├── README.md                    # this file
├── EVALUATION.md                # dataset, baselines, dev/test results
├── requirements.txt             # core harness needs no third-party packages
├── .gitignore
├── .gitattributes               # keeps the starter byte-exact (data-integrity checks)
├── nsc-starter/                 # course starter harness (student_score.py is ours)
│   ├── README.md                # protocol, schema, baselines, limitations
│   ├── nsc.py                   # MP integral, per-layer / original NSC, resources
│   ├── student_score.py         # OUR WORK: score / ablation / position-only control
│   ├── evaluate.py              # ranking, size control, fixed-pool selection
│   ├── check_starter.py         # numerical identities and data-integrity checks
│   ├── expected_development.json# reference output for the unedited template
│   └── data/                    # 200 configs, splits, labels, resources, provenance
├── experiments/                 # dataset, development selection, mechanism and evaluation
│   ├── data/                    # generated datasets and result tables (committed)
│   └── plot_figures.py          # renders every report figure from experiments/data
├── figures/                     # report figures (PNG + vector PDF) and caption sheet
└── results/                     # generated result JSON (gitignored)
```

The first commit of this repository held a generic "Automated Heuristic
Design" scaffold (`src/`, `scripts/`, placeholder `data/`); commit `dcef444`
replaced it with this project's layout. `main` and `dev` currently point at
the same commit.

## Setup and reproduction

Python 3.9 or later; no installation step for the core harness.

```bash
cd nsc-starter
python check_starter.py
python evaluate.py --split development --output ../results/development_results.json
```

After freezing the method, the development choices and the code version:

```bash
python evaluate.py --split final --output ../results/final_results.json
```

The full pipeline and the reported numbers are in [EVALUATION.md](EVALUATION.md);
scripts live in `experiments/`.

To reproduce the complete experiments from the repository root:

```bash
python nsc-starter/check_starter.py
python experiments/select_alpha.py
python experiments/dataset.py
python experiments/mechanism.py
python experiments/evaluate_all.py
python experiments/analyze.py
python experiments/plot_figures.py
```

The original development choice remains `a=0`. The extended experiments include
same-a position controls, the exact invariant/position/head-residual decomposition,
FLOPs and depth controls, paired architecture bootstrap, and selected-architecture
profiles for every prescribed budget. Layer-order decomposition and resource
analysis follow the proposal; the detailed component ablations, extra resource
matching and bootstrap settings were added after final outcomes were available.
`select_alpha.py` replays the development calculation without writing to the scorer
or the original selection record.

`--student path/to/module.py` selects another scorer implementation. The
evaluator loads only the chosen split's label file. Expect seconds on an
ordinary laptop. Complete result tables are saved under `experiments/data/`;
official and extended evaluator outputs are saved under `results/`.

`experiments/plot_figures.py` reads these three committed tables:

- `evaluation_summary.json`: ranking, budget-selection metrics, per-architecture
  PPL and all method scores (`architecture_scores`), and cold/warm CPU timing
  samples for each method (`scoring_cost.by_method`).
- `analysis_summary.json`: component values, resource controls, bootstrap
  intervals, architecture profiles and selected-ID frequencies. Selection cases
  also include best-PPL distributions over the 200 trials, including random selection.
- `mechanism_check.json`: synthetic permutation deltas and numerical identities.

These files are under `experiments/data/`. Supplementary analysis and the
figures read this result table directly and check that it matches the current
scores and labels; they do not depend on the gitignored files in `results/`.

## Report figures

`experiments/plot_figures.py` renders every figure of the final report from the
committed result tables in `experiments/data/`. It reads no label file and
recomputes no outcome: every number it draws is already reported in
[EVALUATION.md](EVALUATION.md). Each figure is written as a 300 dpi PNG and as
a vector PDF -- include the PDF in the report -- and the script also writes
`figures/FIGURE_CAPTIONS.md`, a ready-to-paste caption sheet with LaTeX
snippets.

```bash
python -m pip install matplotlib     # only the figures need a third-party package
python experiments/plot_figures.py   # ~7 s, writes figures/*.png and figures/*.pdf
python experiments/plot_figures.py --list
python experiments/plot_figures.py --only fig1 fig5
```

| Figure | Content |
|---|---|
| `fig1_alpha_sweep` | Spearman $\rho$ over the $\alpha$ grid on both splits, with paired-bootstrap intervals |
| `fig2_ranking_baselines` | All sixteen methods and baselines: all-pairs $\rho$ and restricted $\tau_b$ |
| `fig3_mechanism_reorder` | Reorder sensitivity on the 41 synthetic cases, by family and head-only |
| `fig4_nsc_flops_collinearity` | Recorded outcome and NSC score against reference FLOPs |
| `fig5_selection_regret` | Budgeted selection and regret under the 10/20/30/50M parameter caps |
| `fig6_decomposition` | The exact decomposition $T = U + P + R$ of the added term |
| `fig7_scoring_cost` | Cold and warm scoring cost per method |

The core harness in `nsc-starter/` stays standard-library only; matplotlib is
needed for the figures alone. The palette is a restrained academic one -- blue,
purple, green and orange on white with dark-grey ink, no rainbow -- and every
series also varies its marker or line style, so the figures stay readable in
greyscale and match the Times New Roman 12 pt report.

## Scope of the evidence

The panel carries one recorded outcome per architecture, without training
seeds, steps/tokens, hardware or the exact upstream training revision. These
results support an exploratory comparison on this fixed archive; they cannot
establish training-seed significance, full-pretraining performance or
generalisation to another architecture family. The final split is an
architecture holdout within the archive, not an independent benchmark. The
archived configurations contain adaptive embeddings, memory length 192 and
`pre_lnorm=false` — describe the actual family, not an assumed modern
pre-normalised GPT decoder. Cite published methods and distinguish
reproduction from our own contribution.

## References

- Zhu, Zhao and Lu (2026). *Neural Spectral Capacity: Measuring and Designing
  Architectures from Network Specification Alone.*
  [Paper](https://arxiv.org/abs/2609.23087) ·
  [code](https://github.com/Optima-CityU/neural-spectral-capacity)
- Javaheripi et al. (NeurIPS 2022). *LiteTransformerSearch: Training-free
  Neural Architecture Search for Efficient Language Models.*
  [Paper](https://arxiv.org/abs/2203.02094) ·
  [code](https://github.com/microsoft/archai/tree/neurips-lts/archai/nlp)
