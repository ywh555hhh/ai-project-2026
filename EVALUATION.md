# Structural Sensitivity and Predictive Utility of NSC Reweighting

This is an analysis of a fixed archived architecture panel. The original
development-only rule selected **a=0**, so the submitted method equals NSC.
Positive prefix reweighting did not improve global development ranking.
Structural sensitivity and predictive utility are evaluated separately.

The formula, layer-order decomposition, permutation checks and resource/failure
analysis follow the proposal. Detailed component ablations, extra FLOPs/depth
matching and bootstrap settings were added after final outcomes were available.
Those additional diagnostics explain the fixed-panel results; they do not
provide a new independent test or change the development-selected score.

## Data and protocol

The course archive supplies 200 GPT-2 / Transformer-XL-style configurations,
with adaptive embeddings, `pre_lnorm=false`, `mem_len=192`, widths 64–512 and
depth 2–16. The disjoint splits were assigned without examining outcomes.
Development uses validation PPL; final uses test PPL on different architectures.
The final split is an architecture holdout in the same archive.


| Split | Architectures | Parameter-matched pairs | Pools at 10/20/30/50M |
|---|---|---|---|
| development | 100 | 180 | 24/65/83/96 |
| final | 100 | 185 | 25/57/79/96 |

Ranking uses average-rank Spearman rho and Kendall tau-b against **negative
PPL**, with scores canonicalized to 12 significant digits. The required
restricted tau-b uses same-width pairs with total-parameter min/max >=0.90.
These overlapping pairs are not independent samples or causal controls.

Projection parameters are `sum_l(4*d^2+2*d*f_l)`. Reference MACs/query token
add `2*L*384*d`; reference FLOPs are twice MACs. These omit embeddings,
vocabulary output, biases, norms, activation, softmax and cache preparation.
They are not total training/inference FLOPs or measured latency. Caps use
the archive's **total model parameters**, whose component fields overlap.

## Proposed method and structural analysis

The implementation retains the starter's sigma=0.02 and projection grouping:

```text
B_l = 3 H_l psi(d,d/H_l) + psi(d,d) + 2 psi(d,f_l)
q_l = 2 psi(d,f_l); S0 = sum_l B_l; D = S0 + 1e-12
C_l = sum_{j<l} B_j / D
S_a = S0 + a sum_l C_l q_l
position-only control: replace C_l by (l-1)/L
```

The archive is post-norm: `h_(l+1)=LN(z_l+F_l(z_l))`. Prefix capacity is a
structural proxy, not a derived capacity law for this normalized computation.

Let `A_l=B_l-q_l` be attention capacity and `Abar=mean_l A_l`. The added term
has the exact decomposition:

```text
T = sum_l C_l q_l = U + P + R
U = [(sum q)^2 - sum(q^2)] / (2D)
P = Abar/D * sum_l (l-1)q_l
R = sum_l q_l sum_{j<l}(A_j-Abar) / D
```

`U` is invariant to FFN permutations. For constant heads, `R=0`: within a
fixed layer multiset, all order sensitivity is scaled positional weighting,
`Delta S_prefix = A*L/D * Delta S_position`. Across different architectures,
`U` and the scaling vary, so this does not assert identical overall rankings.
For a joint adjacent-layer swap, the exact new-minus-old prefix change is
`(A_(i+1)*q_i-A_i*q_(i+1))/D`. These identities identify what the score sees
beyond position and resource magnitude; a reorder response alone is weaker evidence.


Mechanism status: **PASS**, 41 synthetic pairs from eight development sources and the starter toy pair. The expanded panel includes FFN permutations with constant heads, varying heads, head-only permutations and joint reversals. All synthetic cases have **no trained outcomes**.

| Check | Result |
|---|---|
| Max absolute NSC reorder change | 0.000000000 |
| Max absolute prefix reorder change, a=1 | 100.189884 |
| Max head-only prefix / position change, a=1 | 1.862679 / 0.000000 |
| Resource invariance and a=0 recovery | PASS |
| Single-layer C1=0, all prescribed a, four widths | PASS |
| Homogeneous layers approach positional weights (epsilon correction) | True |
| Constant-head prefix/position delta identity | 9 pairs; max error 2.10e-12 |

| Split | Max decomposition error | Adjacent swaps checked / max error | Median / max abs(R)/T | Same ranks without R at a=.25/.5/1 |
|---|---|---|---|---|
| development | 5.35e-13 | 826 / 2.06e-12 | 0.00990% / 0.27260% | True/True/True |
| final | 3.84e-13 | 844 / 3.10e-12 | 0.00786% / 0.23557% | True/True/True |

The adjacent-swap deltas above are computed from the actual scorer, rather than
only from the decomposed formula. Reconstructing the score from the components
at every positive alpha has maximum absolute error 1.82e-12 on either split.

The heterogeneous-attention residual can affect synthetic reorder scores, yet its predictive contribution must be assessed separately. Removing it above is an explanatory diagnostic; it does not replace the frozen score.

## Development choice and matched sensitivity controls

The original freeze time is `2026-11-13T03:27:07+00:00`. The proposal selects the largest development rho on {0,.25,.5,1}, preferring smaller a on ties. Replaying this rule reproduces **a=0**; the original record is not regenerated. Position controls use the same a, without separately optimizing them. Equal a need not imply equal perturbation amplitude.

| a | dev prefix rho | dev position rho | dev no-R rho | final prefix rho | final position rho | final no-R rho |
|---|---|---|---|---|---|---|
| 0.0 | 0.969745 | 0.969745 | 0.969745 | 0.965941 | 0.965941 | 0.965941 |
| 0.25 | 0.967909 | 0.968293 | 0.967909 | 0.965149 | 0.965269 | 0.965149 |
| 0.5 | 0.966649 | 0.966973 | 0.966649 | 0.964200 | 0.963612 | 0.964200 |
| 1.0 | 0.963432 | 0.963900 | 0.963432 | 0.961224 | 0.961608 | 0.961224 |

Every prescribed positive weight has lower global rho than original NSC on both panels. Prefix is below its same-a position control on development; final is mixed, with prefix slightly higher at a=.5 and lower at .25 and 1. The paired-bootstrap intervals for all prefix-versus-position rho differences include zero. This is evidence about this archive and grid, not all prefix/context models.

## Required ranking baselines

All-pair and restricted tie counts are saved in `experiments/data/evaluation_summary.json`. Reference FLOPs has one score-tied pair on development; the other required baselines have no score ties, and neither panel has outcome ties. Student, ablation and frozen position control coincide because a=0, so their equality supplies no nonzero mechanism test.

| Method | dev rho | dev tau-b | dev restricted | final rho | final tau-b | final restricted |
|---|---|---|---|---|---|---|
| original_nsc | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| position_example | 0.970069 | 0.865051 | 0.622222 | 0.968977 | 0.853333 | 0.318919 |
| student | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| ablation | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| position_only_control | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| reported_parameters | 0.727873 | 0.575758 | 0.133333 | 0.741854 | 0.589495 | 0.275676 |
| reference_flops | 0.967948 | 0.858471 | 0.622222 | 0.965101 | 0.842424 | 0.329730 |

## Resource effects and additional controls

The prescribed parameter matching does not fix FLOPs, projection weights or depth. Adaptive embeddings contribute to reported total parameters but not to NSC. Additional fixed 0.90 FLOPs matching and then equal-depth matching are post-final diagnostics, with all remaining pair and architecture counts reported; none is a causal control.

| Split | Parameter pairs | FLOPs ratio <.90 | Median / min FLOPs ratio | NSC–FLOPs rho |
|---|---|---|---|---|
| development | 180 | 141 | 0.711540 / 0.186343 | 0.999163 |
| final | 185 | 140 | 0.781261 / 0.188546 | 0.999640 |

The following entries are restricted tau-b on each stated pair set, not full-panel rank correlations.

| Split / control | Pairs | Architectures | original_nsc | reference_flops | reported_parameters | sensitivity_a1.0 | position_control_a1.0 |
|---|---|---|---|---|---|---|---|
| development / same_width_parameters | 180 | 94 | 0.622222 | 0.622222 | 0.133333 | 0.633333 | 0.622222 |
| development / same_width_parameters_flops | 39 | 47 | 0.179487 | 0.179487 | 0.435897 | 0.230769 | 0.179487 |
| development / same_width_parameters_flops_depth | 15 | 30 | 0.333333 | 0.333333 | 0.466667 | 0.600000 | 0.466667 |
| final / same_width_parameters | 185 | 96 | 0.329730 | 0.329730 | 0.275676 | 0.297297 | 0.308108 |
| final / same_width_parameters_flops | 45 | 47 | -0.066667 | -0.066667 | 0.377778 | -0.200000 | -0.155556 |
| final / same_width_parameters_flops_depth | 18 | 25 | -0.111111 | -0.111111 | 0.555556 | -0.111111 | -0.111111 |

These statistics should be described as correlations on increasingly restricted subsets. The dev/final comparison also changes architectures and PPL endpoints; a change in restricted tau alone cannot identify a generalization mechanism.

| Split / width | Architectures | NSC rho | FLOPs rho | prefix a=1 rho | position a=1 rho |
|---|---|---|---|---|---|
| development / 64 | 25 | 0.982308 | 0.982308 | 0.980769 | 0.980769 |
| development / 128 | 25 | 0.909231 | 0.909231 | 0.909231 | 0.900769 |
| development / 256 | 26 | 0.948718 | 0.948718 | 0.945299 | 0.945299 |
| development / 512 | 24 | 0.892174 | 0.892174 | 0.893913 | 0.896522 |
| final / 64 | 24 | 0.919130 | 0.924348 | 0.919130 | 0.920870 |
| final / 128 | 26 | 0.917265 | 0.920684 | 0.914530 | 0.905641 |
| final / 256 | 27 | 0.896215 | 0.900488 | 0.901709 | 0.898657 |
| final / 512 | 23 | 0.659091 | 0.658103 | 0.658103 | 0.647233 |

## Paired architecture bootstrap

These exploratory 95% percentile intervals use 1,000 fixed-seed draws of architectures with replacement, shared across methods in each replicate. They describe panel-sampling sensitivity, not training-seed uncertainty. Contrasts are not adjusted for multiple comparisons. All seven contrasts per split, including .25 and .5, are stored in `analysis_summary.json`.

| Split / contrast | Delta rho | 95% percentile interval |
|---|---|---|
| development / original_nsc minus reference_flops | 0.001797 | [-0.001851, 0.006796] |
| development / sensitivity_a1.0 minus original_nsc | -0.006313 | [-0.015774, -0.000215] |
| development / sensitivity_a1.0 minus position_control_a1.0 | -0.000468 | [-0.002991, 0.001515] |
| final / original_nsc minus reference_flops | 0.000840 | [-0.001850, 0.003652] |
| final / sensitivity_a1.0 minus original_nsc | -0.004716 | [-0.012621, 0.000588] |
| final / sensitivity_a1.0 minus position_control_a1.0 | -0.000384 | [-0.002715, 0.001993] |

## Finite-pool selection and failures

Every method uses the same feasible pool, caps 10/20/30/50M and k=1/3, with 200 fixed-seed trials for uniform random draws and score ties. Best selected PPL and regret use archived outcomes, not new online training. Scored methods are deterministic on these panels; their selection SD is zero. Random SD below measures variability over selections.

| Split | Cap | k | Pool | Oracle | NSC best (regret) | Parameters best (regret) | Random best (regret), SD |
|---|---|---|---|---|---|---|---|
| development | 10M | 1 | 24 | 37.6010 | 39.5418 (1.9408) | 37.6010 (0.0000) | 50.8860 (13.2851), 10.8874 |
| development | 10M | 3 | 24 | 37.6010 | 37.6010 (0.0000) | 37.6010 (0.0000) | 41.5064 (3.9054), 4.3481 |
| development | 20M | 1 | 65 | 29.9539 | 33.1287 (3.1748) | 32.4472 (2.4933) | 47.1793 (17.2254), 11.8259 |
| development | 20M | 3 | 65 | 29.9539 | 30.5189 (0.5650) | 29.9539 (0.0000) | 36.1687 (6.2148), 4.6444 |
| development | 30M | 1 | 83 | 28.2660 | 30.0721 (1.8061) | 30.0721 (1.8061) | 44.5123 (16.2464), 12.4264 |
| development | 30M | 3 | 83 | 28.2660 | 28.3506 (0.0846) | 29.3405 (1.0745) | 34.4531 (6.1871), 6.0560 |
| development | 50M | 1 | 96 | 25.9270 | 26.0157 (0.0886) | 26.0157 (0.0886) | 41.3224 (15.3954), 12.6853 |
| development | 50M | 3 | 96 | 25.9270 | 26.0157 (0.0886) | 26.0157 (0.0886) | 31.9696 (6.0426), 5.8318 |
| final | 10M | 1 | 25 | 38.4856 | 38.4856 (0.0000) | 42.2075 (3.7219) | 50.6622 (12.1766), 9.2506 |
| final | 10M | 3 | 25 | 38.4856 | 38.4856 (0.0000) | 42.2075 (3.7219) | 42.4569 (3.9713), 4.1380 |
| final | 20M | 1 | 57 | 32.1837 | 33.3200 (1.1363) | 40.6258 (8.4421) | 47.5466 (15.3630), 10.5438 |
| final | 20M | 3 | 57 | 32.1837 | 33.0335 (0.8499) | 33.6774 (1.4938) | 38.4760 (6.2924), 5.0253 |
| final | 30M | 1 | 79 | 29.1845 | 30.8054 (1.6209) | 29.1845 (0.0000) | 44.7174 (15.5329), 10.7638 |
| final | 30M | 3 | 79 | 29.1845 | 29.9562 (0.7717) | 29.1845 (0.0000) | 35.1146 (5.9301), 5.5506 |
| final | 50M | 1 | 96 | 26.2937 | 27.6652 (1.3715) | 27.4234 (1.1297) | 41.6530 (15.3593), 11.2497 |
| final | 50M | 3 | 96 | 26.2937 | 27.6228 (1.3291) | 27.4234 (1.1297) | 32.1382 (5.8445), 5.6723 |

The JSON includes full selection tables for positional, prefix, position-control and no-R variants. `analysis_summary.json` exposes selected IDs, frequencies and best-PPL distributions for every method (including random) in all eight pools per split, oracle IDs, and architecture/resource profiles. These trial-level summaries are recomputed and checked against the evaluation metrics. Equal global correlations do not guarantee equal selections or low regret.

For the descriptive final 30M, k=1 failure, the selected and feasible-oracle profiles are:

| ID / role | PPL | Width | Depth | Mean FFN | Total params | AdaEmb / total | Reference FLOPs |
|---|---|---|---|---|---|---|---|
| config_13_j1 / NSC | 30.8054 | 512 | 10 | 1468.9 | 29171219 | 0.1188 | 58918912 |
| config_13_j4 / parameter baseline, oracle | 29.1845 | 256 | 14 | 1249.8 | 29812275 | 0.5686 | 30761984 |

NSC ranks larger scored projection capacity highly; the parameter baseline and oracle need not allocate their total budget the same way. These profiles expose the mismatch between the score and total-parameter budget. They do not isolate a causal effect of embeddings, FFN order or depth. No synthetic permutation inherits the original architecture's PPL.

## Reproduction

Python 3.9+ and the standard library suffice for the numeric pipeline. Optional
SciPy independently verifies the rank correlations. From the repository root:

```bash
python nsc-starter/check_starter.py
python experiments/select_alpha.py
python experiments/dataset.py
python experiments/mechanism.py
python experiments/evaluate_all.py
python experiments/analyze.py
```

The development choice is already recorded in `experiments/frozen_config.json`.
`select_alpha.py` reproduces and verifies the complete development sweep without
overwriting that record or changing the scorer. The extended evaluator performs
the same check and reproduces all shared ranking, pair-count and selection fields
from the official evaluator.

Hardware, cold/warm scoring samples and evaluation wall time are recorded in
`experiments/data/evaluation_summary.json`. Each timed method gets three separate
cold-cache/warm-cache panel pairs over all 200 configurations, with samples and
medians saved under `scoring_cost.by_method`. These measure scorer computation,
excluding copying, ranking and selection. `architecture_scores` supplies the
per-architecture PPL and every method's score for plotting. Supplementary analysis
uses this tracked table and validates it against current scores and labels,
without reading the gitignored intermediate outputs. The source panel and split hashes
are checked against the supplied provenance. The complete development ranking
and selection reproduce `nsc-starter/expected_development.json`; independent
SciPy verification agrees to within 2.23e-16.

Representative median CPU times for scoring all 200 configurations are below,
measured on an AMD Ryzen 7 5700X with Python 3.10.20. All three samples for every
timed method are in the JSON. The recorded evaluation wall time is 20.57 seconds,
including the separate runtime measurements.

| Method | Cold panel, ms | Warm panel, ms |
|---|---|---|
| original_nsc | 414.918 | 1.342 |
| student (alpha=0) | 413.169 | 1.412 |
| sensitivity_a1.0 | 419.521 | 2.718 |
| position_control_a1.0 | 427.610 | 2.874 |
| without_head_residual_a1.0 | 431.291 | 5.927 |

Cold timings are dominated by MP quadrature. Warm timings expose scorer overhead
after matrix-shape values have been cached; they are not model inference times.

## Scope of the conclusions

This controlled investigation finds that the specified positive prefix term changes structural sensitivity but supplies no global-ranking improvement under the proposed development rule. Its FFN pair term is order-invariant, its constant-head order effect is positional, and NSC is nearly co-ranked with reference FLOPs on this archive. The nonzero matched controls, resource restrictions and local selection failures explain these observations without establishing that all structural proxies fail.

Each architecture has one recorded endpoint, with no training seeds, tokens, hardware or exact upstream revision. The archive cannot establish training-seed significance, full-pretraining behavior or generalization to another family. Sources: Zhu, Zhao and Lu, Neural Spectral Capacity (2026), and Javaheripi et al., LiteTransformerSearch (NeurIPS 2022); attribution and archive hashes are in `nsc-starter/data/provenance.json`. The contribution is this bounded analysis, not a better heuristic or a new search algorithm.
