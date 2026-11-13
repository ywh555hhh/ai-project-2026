#!/usr/bin/env python3
"""Mechanism check on OUR synthetic counterfactual dataset.

This is deliberately separate from the predictive claim. We verify:

1. Original NSC is invariant to any independent permutation of the head-count
   and FFN-width lists (its layer terms are separable and summed).
2. Prefix-capacity reweighting IS sensitive to those permutations, including
   head-only reorderings that leave every q_l unchanged (attention-prefix effect).
3. The a=0 ablation removes the effect exactly and recovers original NSC.
4. C_1 = 0 for the first layer (the proposal's limiting case).
5. Reference resource estimates stay invariant under reordering.
6. The toy pair reproduces the numbers published in the starter README.

No trained labels are read anywhere in this script.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsc-starter"))

import student_score as ours  # noqa: E402
from nsc import original_nsc, position_example, resource_estimate  # noqa: E402

HERE = Path(__file__).resolve().parent
ALPHAS = (0.0, 0.25, 0.5, 1.0)
TOL = 1e-8


def measure(config):
    """All scores and resource fields for one (synthetic) configuration."""
    prefix_term = math.fsum(c * q for c, q in
                            zip(ours.prefix_weights(config)[1], ours.ffn_terms(config)))
    res = resource_estimate(config)
    return {
        "nsc": original_nsc(config),
        "position_example": position_example(config),
        "ablation_a0": ours.ablation(config),
        "zero_weighting": ours.reweighted_nsc(config, 0.0, "none"),
        "ours": {str(a): ours.reweighted_nsc(config, a, "prefix") for a in ALPHAS},
        "position_control": {str(a): ours.reweighted_nsc(config, a, "position") for a in ALPHAS},
        "prefix_term": prefix_term,
        "first_prefix_weight": ours.prefix_weights(config)[1][0],
        "projection_parameters": res["projection_parameters"],
        "reference_macs_per_token": res["reference_macs_per_token"],
    }


def close(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def main():
    data = json.loads((HERE / "data" / "counterfactuals.json").read_text(encoding="utf-8"))
    case_results, failures = [], []
    nsc_deltas, ours_deltas, head_only_ours, head_only_control = [], [], [], []
    resource_ok = True

    for case in data["cases"]:
        variants = case["variants"]
        names = list(variants)
        measured = {name: measure(cfg) for name, cfg in variants.items()}
        left, right = measured[names[0]], measured[names[1]]

        # 1. additive NSC cannot see the permutation
        nsc_delta = right["nsc"] - left["nsc"]
        nsc_deltas.append(abs(nsc_delta))
        if not close(nsc_delta, 0.0):
            failures.append(f"{case['case_id']}: NSC changed under permutation ({nsc_delta})")

        # 2. resources are invariant (same multiset of FFN widths)
        if (left["projection_parameters"] != right["projection_parameters"]
                or left["reference_macs_per_token"] != right["reference_macs_per_token"]):
            resource_ok = False
            failures.append(f"{case['case_id']}: resource estimate changed")

        # 3. invariants on every variant
        for name, m in measured.items():
            if not close(m["ablation_a0"], m["nsc"]):
                failures.append(f"{case['case_id']}/{name}: a=0 did not recover NSC")
            if not close(m["zero_weighting"], m["nsc"]):
                failures.append(f"{case['case_id']}/{name}: zero weighting did not recover NSC")
            if m["first_prefix_weight"] != 0.0:
                failures.append(f"{case['case_id']}/{name}: C_1 != 0")
            for a in ALPHAS:
                if not math.isfinite(m["ours"][str(a)]):
                    failures.append(f"{case['case_id']}/{name}: non-finite score at a={a}")

        # 4. sensitivity and ablation deltas
        delta = {str(a): right["ours"][str(a)] - left["ours"][str(a)] for a in ALPHAS}
        control_delta = {str(a): right["position_control"][str(a)]
                         - left["position_control"][str(a)] for a in ALPHAS}
        if abs(delta["0.0"]) > TOL:
            failures.append(f"{case['case_id']}: ablation still has an order effect")
        ours_deltas.append(abs(delta["1.0"]))
        if case["family"] == "heads_reversed_fixed_ffn":
            head_only_ours.append(abs(delta["1.0"]))
            head_only_control.append(abs(control_delta["1.0"]))
            if abs(control_delta["1.0"]) > TOL:
                failures.append(f"{case['case_id']}: position-only control reacted to head order")

        case_results.append({
            "case_id": case["case_id"], "family": case["family"],
            "source_config": case["source_config"],
            "nsc": {n: measured[n]["nsc"] for n in names},
            "position_example": {n: measured[n]["position_example"] for n in names},
            "ours_a1": {n: measured[n]["ours"]["1.0"] for n in names},
            "position_control_a1": {n: measured[n]["position_control"]["1.0"] for n in names},
            "nsc_delta": nsc_delta, "ours_delta": delta,
            "position_control_delta": control_delta,
            "resource_invariant": (left["projection_parameters"] == right["projection_parameters"]
                                   and left["reference_macs_per_token"] == right["reference_macs_per_token"]),
        })

    toy = next(c for c in data["cases"] if c["family"] == "toy_pair")
    toy_scores = [measure(cfg) for cfg in toy["variants"].values()]
    toy_reference = {
        "nsc_both_orders": [round(m["nsc"], 9) for m in toy_scores],
        "position_example": [round(m["position_example"], 9) for m in toy_scores],
        "starter_readme_expected": [96.957958136, 85.772296570],
        "ours_a1": [m["ours"]["1.0"] for m in toy_scores],
    }
    if not all(close(a, b) for a, b in zip(toy_reference["position_example"],
                                           toy_reference["starter_readme_expected"])):
        failures.append("toy pair does not reproduce starter README position example")

    if max(ours_deltas) <= TOL:
        failures.append("our score showed no order sensitivity anywhere")
    if max(head_only_ours) <= TOL:
        failures.append("no attention-prefix effect detected")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "cases": len(case_results),
        "max_abs_nsc_delta": max(nsc_deltas),
        "max_abs_ours_delta_a1": max(ours_deltas),
        "cases_with_ours_delta_gt_tol": sum(d > TOL for d in ours_deltas),
        "head_only_cases": len(head_only_ours),
        "max_head_only_ours_delta_a1": max(head_only_ours),
        "max_head_only_position_control_delta_a1": max(head_only_control),
        "all_resource_estimates_invariant": resource_ok,
        "toy_pair": toy_reference,
        "interpretation": "Synthetic reorderings carry no trained outcomes. These results show "
                          "structural sensitivity only, not predictive validity.",
        "failures": failures,
        "case_details": case_results,
    }
    (HERE / "data" / "mechanism_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"mechanism check: {summary['status']} ({len(case_results)} cases)")
    print(f"  max |dNSC| under permutation      = {summary['max_abs_nsc_delta']:.3e}")
    print(f"  max |d ours| at a=1              = {summary['max_abs_ours_delta_a1']:.6f}")
    print(f"  head-only: |d ours| = {summary['max_head_only_ours_delta_a1']:.6f}, "
          f"|d position control| = {summary['max_head_only_position_control_delta_a1']:.3e}")
    print(f"  toy position example: {toy_reference['position_example']}")
    for failure in failures:
        print("  FAIL:", failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
