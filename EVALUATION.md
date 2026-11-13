# Evaluation

CS5491 archived GPT-2 / Transformer-XL panel (LiteTransformerSearch, NeurIPS
2022): 200 configs, CPU-only, no training. Numbers from
`experiments/evaluate_all.py`; full tables in
`experiments/data/evaluation_summary.json`.

## Dataset / test instances

| | Development | Final (test) |
|---|---:|---:|
| Instances | 100 | 100 |
| Outcome | `valid_ppl` | `test_ppl` |
| Size-controlled pairs | 180 | 185 |
| Eligible at 10M/20M/30M/50M | 24/65/83/96 | 25/57/79/96 |

Widths 64–512, depth 2–16; all post-norm (`pre_lnorm=false`) with adaptive
embeddings and `mem_len=192`. Splits are disjoint, assigned within width groups
without using outcomes; final is held out by protocol. Manifest:
`experiments/data/benchmark_manifest.json`.

Our own dataset: 33 synthetic layer reorderings from 8 dev configs plus the
starter's toy pair (`experiments/data/counterfactuals.json`) — FFN and head
reversals/rotations. No trained labels, so structural claims only.

## Method and freeze

`S_a = S0 + a·Σ C_l q_l`, `C_l = Σ_{j<l} B_j/(S0+ε)`, `q_l = 2ψ_MP(d,f_l,0.02)`,
`ε=1e-12`. `ablation` is `a=0`; `position_only_control` uses `C_l=(l-1)/L`.

Dev-only `a` sweep (Spearman ρ vs −`valid_ppl`):

| `a` | 0 | 0.25 | 0.5 | 1 |
|---|---:|---:|---:|---:|
| ρ | **0.969745** | 0.967909 | 0.966649 | 0.963432 |

Frozen **a = 0** before any final run (`experiments/frozen_config.json`). The
prefix term degrades ranking monotonically, so our submitted score equals
original NSC — a negative result. A post-freeze dev-only diagnostic peaks at
ρ=0.972865 for `a=-1`, outside the pre-registered grid; not used.

## Mechanism check — PASS (33 cases)

- NSC is order-blind (max |Δ| = 0); `S_1` is not (max |Δ| = 100.189884).
- Head-only reorder with all `q_l` fixed: |Δ ours| up to 1.862679,
  position-only control 0 → the prefix term, not position, drives the effect.
- `a=0` recovers NSC exactly; `C_1=0`; resources invariant. Toy pair matches
  the starter README: NSC 121.820169804 both orders, position example
  96.957958136 / 85.772296570.

## Ranking

Spearman ρ / Kendall τ-b / size-controlled τ-b vs −PPL. No score ties on either
split (4,950 pairs each). Controlled τ falls 0.622 → 0.330 on final: with size
held fixed, specification-only scores carry much less signal.

| Method | dev ρ | dev τ | dev ctrl | final ρ | final τ | final ctrl |
|---|---:|---:|---:|---:|---:|---:|
| original NSC | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| positional example | 0.970069 | 0.865051 | 0.622222 | **0.968977** | **0.853333** | 0.318919 |
| **student (ours, a=0)** | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| ablation / position-only control | 0.969745 | 0.861414 | 0.622222 | 0.965941 | 0.843232 | 0.329730 |
| reported parameters | 0.727873 | 0.575758 | 0.133333 | 0.741854 | 0.589495 | 0.275676 |
| reference FLOPs | 0.967948 | 0.858471 | 0.622222 | 0.965101 | 0.842424 | 0.329730 |

Post-freeze sensitivity (never used for selection): final ρ at `a` =
0.25/0.5/1 is 0.965149 / 0.964200 / 0.961224 — the same decline. The official
harness reproduces `expected_development.json` exactly, and our extended
harness agrees with it on every shared statistic.

## Selection on the final test instances

Best selected PPL (regret), 200 fixed-seed trials, same pool for all methods.
On this split NSC, student, ablation, position-only control, positional example
and reference FLOPs select identically.

| Cap | k | pool | oracle | NSC family | parameters | random |
|---:|--:|--:|---:|---:|---:|---:|
| 10M | 1 | 25 | 38.4856 | 38.4856 (0.0000) | 42.2075 (3.7219) | 50.6622 (12.1766) |
| 10M | 3 | 25 | 38.4856 | 38.4856 (0.0000) | 42.2075 (3.7219) | 42.4569 (3.9713) |
| 20M | 1 | 57 | 32.1837 | 33.3200 (1.1363) | 40.6258 (8.4421) | 47.5466 (15.3630) |
| 20M | 3 | 57 | 32.1837 | 33.0335 (0.8499) | 33.6774 (1.4938) | 38.4760 (6.2924) |
| 30M | 1 | 79 | 29.1845 | 30.8054 (1.6209) | **29.1845 (0.0000)** | 44.7174 (15.5329) |
| 30M | 3 | 79 | 29.1845 | 29.9562 (0.7717) | **29.1845 (0.0000)** | 35.1146 (5.9301) |
| 50M | 1 | 96 | 26.2937 | 27.6652 (1.3715) | **27.4234 (1.1297)** | 41.6530 (15.3593) |
| 50M | 3 | 96 | 26.2937 | 27.6228 (1.3291) | **27.4234 (1.1297)** | 32.1382 (5.8445) |

Deterministic selections have sd = 0; `random`'s variability is over draws and
tie-breaking, not training seeds. Dev selection table: `evaluation_summary.json`.

## Reproduce / environment

```bash
python experiments/dataset.py && python experiments/mechanism.py && \
python experiments/select.py && python experiments/evaluate_all.py
```

Windows 11, AMD64 Family 23 Model 104, 16 CPUs, Python 3.13.14. Cold/warm
scoring of 200 configs: 0.533 s / 0.0035 s; full pipeline 4.84 s. MP quadrature
is memoised, so harness timings are diagnostics, not cold-start comparisons.

## Caveats

- One outcome per architecture, no seeds/steps/hardware; final is an
  architecture holdout inside this archive, not an independent benchmark.
- Size-controlled pairs overlap (180/185 is not an independent sample size);
  synthetic reorderings have no outcomes and support no predictive claim.
- Frozen `a=0` makes our score identical to the NSC baseline — the contribution
  is the controlled investigation, not a better score. No final outcome was
  used for any decision; selection reads development labels only.
