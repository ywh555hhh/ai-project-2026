#!/usr/bin/env python3
"""Full evaluation on the chosen benchmark instances, AFTER the freeze.

Runs on both splits:

* the official starter harness (``nsc-starter/evaluate.py``) untouched, so its
  numbers remain canonical and comparable with ``expected_development.json``;
* an extended harness that adds the proposal's position-only control and the
  remaining a-sensitivity settings, reusing the official tie/rank/selection
  helpers so both computations agree.

Outputs
-------
results/development_results.json, results/final_results.json   (official, gitignored)
results/extended_*.json                                        (extended, gitignored)
experiments/data/evaluation_summary.json                       (key tables, committed)

IMPORTANT: the frozen choice is experiments/frozen_config.json. Nothing here
selects or changes a. The extra a settings are post-freeze sensitivity
diagnostics only, never used to pick a method.
"""
import copy
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "nsc-starter"
sys.path.insert(0, str(STARTER))

import evaluate as harness  # noqa: E402
import nsc  # noqa: E402
import student_score as ours  # noqa: E402
from nsc import original_nsc, position_example, resource_estimate  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = STARTER / "data"
RESULTS = ROOT / "results"
SPLITS = ("development", "final")


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def run_official():
    """Canonical starter harness, one run per split, no code changes."""
    RESULTS.mkdir(exist_ok=True)
    outputs = {}
    for split in SPLITS:
        out = RESULTS / f"{split}_results.json"
        cmd = [sys.executable, str(STARTER / "evaluate.py"),
               "--split", split, "--output", str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"official evaluate.py failed for {split}: {run.stderr}")
        outputs[split] = json.loads(out.read_text(encoding="utf-8"))
        print(f"official {split}: {run.stdout.strip().splitlines()[-2]}")
    return outputs


def extended_methods():
    """Official four methods + the proposal's control + post-freeze diagnostics."""
    methods = {
        "original_nsc": original_nsc,
        "position_example": position_example,
        "student": ours.score,
        "ablation": ours.ablation,
        "position_only_control": ours.position_only_control,
    }
    for a in (0.25, 0.5, 1.0):
        methods[f"sensitivity_a{a}"] = (
            lambda config, value=a: ours.reweighted_nsc(config, value, "prefix"))
    return methods


def run_split(split, methods):
    """Same protocol as evaluate.evaluate(), with a configurable method set."""
    configs, resources = read("architectures.json"), read("resources.json")
    keys = read("splits.json")[split]
    label_field = "valid_ppl" if split == "development" else "test_ppl"
    labels = read(split + "_labels.json")
    ppl = [labels[k][label_field] for k in keys]
    quality = [-v for v in ppl]
    instances = [configs[k] for k in keys]

    scores, timings = {}, {}
    for name, function in methods.items():
        start = time.perf_counter()
        scores[name] = [harness.canonical(function(copy.deepcopy(c))) for c in instances]
        timings[name] = time.perf_counter() - start
    scores["reported_parameters"] = [resources[k]["total"] for k in keys]
    scores["reference_flops"] = [
        resource_estimate(c)["reference_flops_per_token"] for c in instances]

    total = scores["reported_parameters"]
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if instances[i]["d_model"] == instances[j]["d_model"]
               and min(total[i], total[j]) / max(total[i], total[j]) >= 0.90]
    ranking = {name: {
        "spearman_rho": harness.correlation(harness.ranks(values), harness.ranks(quality)),
        "all_pairs": harness.pair_summary(values, quality, pairs),
        "size_controlled_pairs": harness.pair_summary(values, quality, matched),
    } for name, values in scores.items()}

    selection = []
    for budget in harness.BUDGETS:
        pool = [i for i, size in enumerate(total) if size <= budget]
        for k in harness.K_VALUES:
            if len(pool) < k:
                continue
            seed = 202627 + budget + k
            result = {name: harness.select_trials(pool, values, k, ppl, seed)
                      for name, values in scores.items()}
            result["random"] = harness.select_trials(pool, None, k, ppl, seed)
            selection.append({"parameter_cap": budget, "k": k, "eligible": len(pool),
                              "oracle_ppl": min(ppl[i] for i in pool), "methods": result})
    return {"split": split, "label": label_field, "architectures": len(keys),
            "score_precision": "12 significant digits",
            "matched_rule": "same d_model; min(total)/max(total)>=0.90",
            "scoring_seconds": timings, "ranking": ranking, "selection": selection,
            "size_controlled_pair_count": len(matched)}


def scoring_cost(instances):
    """Cold (empty MP cache) vs warm (shared MP cache) wall time for one panel."""
    nsc.psi_mp.cache_clear()
    start = time.perf_counter()
    for config in instances:
        ours.score(config)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    for config in instances:
        ours.score(config)
    warm = time.perf_counter() - start
    return {"cold_seconds_for_200": cold, "warm_seconds_for_200": warm,
            "note": "MP quadrature is memoised; warm timing is a diagnostic, "
                    "not a fair cold-start comparison between methods."}


def cross_check(official, extended):
    """The shared methods must agree bit-for-bit with the official harness."""
    problems = []
    for split in SPLITS:
        a, b = official[split]["ranking"], extended[split]["ranking"]
        for name in ("original_nsc", "position_example", "student", "ablation",
                     "reported_parameters", "reference_flops"):
            if a[name]["spearman_rho"] != b[name]["spearman_rho"]:
                problems.append(f"{split}/{name}: rho mismatch")
            if a[name]["all_pairs"]["tau_b"] != b[name]["all_pairs"]["tau_b"]:
                problems.append(f"{split}/{name}: tau mismatch")
        for oa, ob in zip(official[split]["selection"], extended[split]["selection"]):
            for name in oa["methods"]:
                if oa["methods"][name]["mean_best_ppl"] != ob["methods"][name]["mean_best_ppl"]:
                    problems.append(f"{split}/{oa['parameter_cap']}/{oa['k']}/{name}: selection mismatch")
    return problems


def main():
    frozen = json.loads((HERE / "frozen_config.json").read_text(encoding="utf-8"))
    print(f"frozen a={frozen['chosen_alpha']} (frozen at {frozen['frozen_at_utc']})")

    started = time.perf_counter()
    official = run_official()
    methods = extended_methods()
    extended = {split: run_split(split, methods) for split in SPLITS}
    for split in SPLITS:
        (RESULTS / f"extended_{split}.json").write_text(
            json.dumps(extended[split], indent=2, allow_nan=False) + "\n", encoding="utf-8")

    problems = cross_check(official, extended)
    if problems:
        raise RuntimeError("extended harness diverged from official: " + "; ".join(problems))

    configs = read("architectures.json")
    all_instances = list(configs.values())
    environment = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "total_wall_seconds": None,
    }

    summary = {
        "generated_by": "experiments/evaluate_all.py",
        "protocol": "method and a frozen on development BEFORE these runs; "
                    "final split evaluated once; no re-selection after seeing final outcomes",
        "frozen": {"chosen_alpha": frozen["chosen_alpha"],
                   "chosen_spearman_rho": frozen["chosen_spearman_rho"],
                   "frozen_at_utc": frozen["frozen_at_utc"],
                   "sensitivity_analysis": frozen["sensitivity_analysis"]},
        "benchmark": {"name": "CS5491 archived GPT-2 / Transformer-XL panel",
                      "architectures": 200, "splits": {s: extended[s]["architectures"] for s in SPLITS},
                      "size_controlled_pair_count": {s: extended[s]["size_controlled_pair_count"]
                                                     for s in SPLITS}},
        "ranking": {s: extended[s]["ranking"] for s in SPLITS},
        "selection": {s: extended[s]["selection"] for s in SPLITS},
        "scoring_cost": scoring_cost(all_instances),
        "official_dev_matches_expected": None,
        "cross_check": "PASS - extended harness agrees with official evaluate.py on shared methods",
        "environment": environment,
        "limitations": [
            "One recorded endpoint per architecture; no training seeds, steps/tokens, hardware or exact upstream revision.",
            "The final split is an architecture holdout within this archive, not an independent benchmark.",
            "200 fixed-seed trials measure tie-breaking / random selection variability, not training-seed uncertainty.",
            "Size-controlled pairs overlap and their count is not an independent sample size.",
            "Synthetic reorderings have no trained outcomes and support only structural claims.",
        ],
    }

    # The unedited-template reference must still reproduce for all shared methods.
    expected = json.loads((STARTER / "expected_development.json").read_text(encoding="utf-8"))
    shared = ("original_nsc", "position_example", "student", "ablation",
              "reported_parameters", "reference_flops")
    summary["official_dev_matches_expected"] = all(
        official["development"]["ranking"][n]["spearman_rho"]
        == expected["ranking"][n]["spearman_rho"] for n in shared)

    environment["total_wall_seconds"] = time.perf_counter() - started
    (HERE / "data" / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("cross-check:", summary["cross_check"])
    print("official dev reproduces expected_development.json:", summary["official_dev_matches_expected"])
    for split in SPLITS:
        print(f"--- {split} ranking ---")
        for name in ("original_nsc", "position_example", "student", "ablation",
                     "position_only_control", "reported_parameters", "reference_flops",
                     "sensitivity_a0.25", "sensitivity_a0.5", "sensitivity_a1.0"):
            r = extended[split]["ranking"][name]
            print(f"  {name:22s} rho={r['spearman_rho']:.6f} "
                  f"tau={r['all_pairs']['tau_b']:.6f} "
                  f"ctrl_tau={r['size_controlled_pairs']['tau_b']:.6f}")
    print(f"total wall: {environment['total_wall_seconds']:.2f}s")
    print("written:", HERE / "data" / "evaluation_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
