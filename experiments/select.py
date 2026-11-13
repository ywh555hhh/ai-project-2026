#!/usr/bin/env python3
"""Development-only hyperparameter selection and FREEZE of our method.

Protocol (from the proposal and the starter README):

* Choose a in {0, 0.25, 0.5, 1} to maximise development Spearman rho against
  NEGATIVE validation PPL; prefer the smaller a on ties.
* Record the full sweep as a sensitivity analysis.
* Freeze the method, the hyperparameter and the code version BEFORE any
  evaluation on the final split.

This script reads ONLY nsc-starter/data/development_labels.json. It never
opens final_labels.json. Running it after the final split has been evaluated
would violate the protocol, so the frozen record is the audit trail.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsc-starter"))

import evaluate as harness  # noqa: E402  (official canonical/ranks/correlation helpers)
import student_score as ours  # noqa: E402

HERE = Path(__file__).resolve().parent
STARTER_DATA = ROOT / "nsc-starter" / "data"
SCORER = ROOT / "nsc-starter" / "student_score.py"
ALPHAS = (0.0, 0.25, 0.5, 1.0)
TIE_TOL = 1e-12


def main():
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
        # exact/statistically identical rho -> keep the smaller a (sweep is ascending)
    chosen = best["alpha"]

    # Position-only control is reported at the same a but is NOT used to select a.
    control = [harness.canonical(ours.reweighted_nsc(c, chosen, "position")) for c in instances]
    control_rho = harness.correlation(harness.ranks(control), q_ranks)

    # Apply the frozen alpha to the scorer itself, so the submission is self-contained.
    source = SCORER.read_text(encoding="utf-8")
    patched, count = re.subn(r"(?m)^ALPHA = [0-9.]+$", f"ALPHA = {chosen}", source)
    if count != 1:
        raise RuntimeError("could not locate the single ALPHA = ... line in student_score.py")
    if patched != source:
        with SCORER.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(patched)

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "FROZEN - do not tune after this point",
        "method": "prefix-capacity reweighted NSC: S_a = S0 + a*sum_l C_l*q_l",
        "selection_metric": "Spearman rho against negative validation PPL on 100 development architectures",
        "candidate_alphas": list(ALPHAS),
        "tie_break": "prefer smaller a on ties (tolerance 1e-12)",
        "chosen_alpha": chosen,
        "chosen_spearman_rho": best["spearman_rho"],
        "sensitivity_analysis": sweep,
        "position_only_control_at_chosen_alpha": {
            "alpha": chosen, "spearman_rho": control_rho,
            "note": "reported for comparison; not used to select a",
        },
        "labels_read": ["nsc-starter/data/development_labels.json"],
        "final_labels_read": False,
        "scorer_path": "nsc-starter/student_score.py",
        "epsilon": ours.EPS,
        "split_separation_verified": not (set(splits["development"]) & set(splits["final"])),
    }
    (HERE / "frozen_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("development-only a sweep (Spearman rho vs -valid_ppl):")
    for row in sweep:
        marker = "  <-- FROZEN" if row["alpha"] == chosen else ""
        print(f"  a={row['alpha']:<5} rho={row['spearman_rho']:.9f} "
              f"tau_b={row['kendall_tau_b']:.9f}{marker}")
    print(f"position-only control at a={chosen}: rho={control_rho:.9f}")
    print(f"frozen a={chosen}")
    print("written:", HERE / "frozen_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
