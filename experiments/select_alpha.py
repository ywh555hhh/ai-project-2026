#!/usr/bin/env python3
"""Reproduce development-only hyperparameter selection without changing files.

Protocol (from the proposal and the starter README):

* Choose a in {0, 0.25, 0.5, 1} to maximise development Spearman rho against
  NEGATIVE validation PPL; prefer the smaller a on ties.
* Record the full sweep as a sensitivity analysis.
* Verify the recorded choice used for final evaluation.

Only development outcome labels are read. Replaying the calculation does not
overwrite the original choice, its timestamp or student_score.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsc-starter"))

import evaluate as harness  # noqa: E402  (official canonical/ranks/correlation helpers)
import student_score as ours  # noqa: E402

HERE = Path(__file__).resolve().parent
STARTER_DATA = ROOT / "nsc-starter" / "data"
ALPHAS = (0.0, 0.25, 0.5, 1.0)
TIE_TOL = 1e-12


def development_selection():
    configs = json.loads((STARTER_DATA / "architectures.json").read_text(encoding="utf-8"))
    splits = json.loads((STARTER_DATA / "splits.json").read_text(encoding="utf-8"))
    dev_labels = json.loads((STARTER_DATA / "development_labels.json").read_text(encoding="utf-8"))

    keys = list(splits["development"])
    ppl = [dev_labels[k]["valid_ppl"] for k in keys]
    quality = [-v for v in ppl]
    q_ranks = harness.ranks(quality)
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    instances = [configs[k] for k in keys]

    sweep = []
    for a in ALPHAS:
        values = [harness.canonical(ours.reweighted_nsc(c, a, "prefix")) for c in instances]
        rho = harness.correlation(harness.ranks(values), q_ranks)
        all_pairs = harness.pair_summary(values, quality, pairs)
        sweep.append({
            "alpha": a,
            "spearman_rho": rho,
            "kendall_tau_b": all_pairs["tau_b"],
            "score_ties": all_pairs["score_only_ties"],
            "both_ties": all_pairs["both_ties"],
            "score_range": [min(values), max(values)],
        })

    best = sweep[0]
    for row in sweep[1:]:
        if row["spearman_rho"] > best["spearman_rho"] + TIE_TOL:
            best = row
        # Numerically tied rho -> keep the smaller a (sweep is ascending).
    chosen = best["alpha"]

    # Position-only control is reported at the same a but is NOT used to select a.
    control = [harness.canonical(ours.reweighted_nsc(c, chosen, "position")) for c in instances]
    control_rho = harness.correlation(harness.ranks(control), q_ranks)

    return {
        "candidate_alphas": list(ALPHAS),
        "chosen_alpha": chosen,
        "chosen_spearman_rho": best["spearman_rho"],
        "sensitivity_analysis": sweep,
        "position_only_control_at_chosen_alpha": {
            "alpha": chosen, "spearman_rho": control_rho,
            "note": "reported for comparison; not used to select a",
        },
        "epsilon": ours.EPS,
        "split_separation_verified": not (set(splits["development"]) & set(splits["final"])),
    }


def validate_selection():
    record = json.loads((HERE / "frozen_config.json").read_text(encoding="utf-8"))
    replay = development_selection()
    for key, value in replay.items():
        if record.get(key) != value:
            raise RuntimeError(f"Development selection differs from the recorded {key}")
    if ours.ALPHA != replay["chosen_alpha"]:
        raise RuntimeError("Scorer alpha differs from the development-selected alpha")
    return record


def main():
    record = validate_selection()
    chosen = record["chosen_alpha"]
    print("development-only a sweep (Spearman rho vs -valid_ppl):")
    for row in record["sensitivity_analysis"]:
        marker = "  <-- FROZEN" if row["alpha"] == chosen else ""
        print(f"  a={row['alpha']:<5} rho={row['spearman_rho']:.9f} "
              f"tau_b={row['kendall_tau_b']:.9f}{marker}")
    control_rho = record["position_only_control_at_chosen_alpha"]["spearman_rho"]
    print(f"position-only control at a={chosen}: rho={control_rho:.9f}")
    print(f"PASS: recorded alpha={chosen}; scorer and original record unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
