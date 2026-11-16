#!/usr/bin/env python3
"""Mechanism, resource and selection analyses of the proposal's fixed score."""
import copy
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsc-starter"))

import evaluate as harness
import nsc
from experiments.components import decompose
from experiments.evaluate_all import extended_methods
from experiments.select_alpha import ALPHAS, validate_selection

HERE = Path(__file__).resolve().parent
DATA = ROOT / "nsc-starter/data"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (index - left) * (values[right] - values[left])


def rho(scores, quality):
    return harness.correlation(harness.ranks(scores), harness.ranks(quality))


def method_name(alpha, family="prefix"):
    if alpha == 0:
        return "original_nsc"
    return {"prefix": "sensitivity_a", "position": "position_control_a",
            "without_residual": "without_head_residual_a"}[family] + str(float(alpha))


def bootstrap(scores, quality, plan):
    """One architecture draw is shared across every contrast in each replicate."""
    contrasts = [(method_name(a), family) for a in ALPHAS[1:]
                 for family in ("original_nsc", method_name(a, "position"))]
    contrasts.append(("original_nsc", "reference_flops"))
    samples = {left + " minus " + right: [] for left, right in contrasts}
    needed_methods = {name for contrast in contrasts for name in contrast}
    point_rhos = {name: rho(scores[name], quality) for name in needed_methods}
    rng = random.Random(plan["seed"])
    n = len(quality)
    for _ in range(plan["repeats"]):
        indices = [rng.randrange(n) for _ in range(n)]
        qr = harness.ranks([quality[i] for i in indices])
        correlations = {name: harness.correlation(harness.ranks([values[i] for i in indices]), qr)
                        for name, values in scores.items()
                        if name in needed_methods}
        for left, right in contrasts:
            if correlations[left] is not None and correlations[right] is not None:
                samples[left + " minus " + right].append(correlations[left] - correlations[right])
    return {left + " minus " + right: {
        "estimate_delta_rho": (point_rhos[left] - point_rhos[right]
                               if point_rhos[left] is not None and point_rhos[right] is not None else None),
        "ci95": ([percentile(samples[left + " minus " + right], p) for p in (0.025, 0.975)]
                 if samples[left + " minus " + right] else None),
        "valid_replicates": len(samples[left + " minus " + right]),
    } for left, right in contrasts}


def profiles(keys, configs, resources, scores, ppl):
    output = {}
    for i, key in enumerate(keys):
        c, r = configs[key], resources[key]
        res = nsc.resource_estimate(c)
        output[key] = {
            "d_model": c["d_model"], "n_layer": c["n_layer"], "d_embed": c["d_embed"],
            "heads": c["n_head"], "ffn_widths": c["d_inner"],
            "mean_ffn_width": statistics.mean(c["d_inner"]),
            "total_parameters": r["total"], "reported_adaptive_embedding_parameters": r["AdaEmb"],
            "reported_adaptive_embedding_fraction": r["AdaEmb"] / r["total"],
            **res, "original_nsc": scores["original_nsc"][i], "recorded_ppl": ppl[i],
        }
    return output


def failure_cases(keys, scores, ppl, sizes, selection):
    """Expose IDs for all prescribed pools, with exactly the harness tie protocol."""
    names = [*scores, "random"]
    key_index = {key: i for i, key in enumerate(keys)}
    output = []
    for row in selection:
        cap, k = row["parameter_cap"], row["k"]
        pool = [i for i, size in enumerate(sizes) if size <= cap]
        oracle = min(ppl[i] for i in pool)
        methods = {}
        for name in names:
            rng = random.Random(202627 + cap + k)
            frequencies, selected_sets, best_ppl_counts = {}, set(), {}
            best_ppl = []
            for _ in range(harness.REPEATS):
                if name == "random":
                    chosen = rng.sample(pool, k)
                else:
                    order = list(pool)
                    rng.shuffle(order)
                    order.sort(key=lambda i: scores[name][i], reverse=True)
                    chosen = order[:k]
                selected_sets.add(tuple(sorted(keys[i] for i in chosen)))
                best = min(ppl[i] for i in chosen)
                best_ppl.append(best)
                best_ppl_counts[best] = best_ppl_counts.get(best, 0) + 1
                for i in chosen:
                    frequencies[keys[i]] = frequencies.get(keys[i], 0) + 1
            metrics = {"mean_best_ppl": statistics.mean(best_ppl),
                       "mean_regret": statistics.mean(best_ppl) - oracle,
                       "sd_best_ppl": statistics.pstdev(best_ppl), "trials": harness.REPEATS}
            if metrics != row["methods"][name]:
                raise RuntimeError(f"Selection details differ from evaluation: {cap}/{k}/{name}")
            selected = sorted(frequencies, key=lambda key: (-frequencies[key], key))
            best_key = min(selected, key=lambda key: ppl[key_index[key]])
            methods[name] = {
                "selected_ids": selected, "selection_frequency": frequencies,
                "selected_ids_note": "Union over trials; frequencies describe random selection or score ties",
                "unique_selected_sets": len(selected_sets),
                "best_revealed_id": best_key if len(selected_sets) == 1 else None,
                "best_ppl_distribution": [{"ppl": value, "trials": count}
                                          for value, count in sorted(best_ppl_counts.items())],
                **metrics,
            }
        output.append({"parameter_cap": cap, "k": k, "eligible": len(pool),
                       "oracle_ids": [keys[i] for i in pool if ppl[i] == oracle],
                       "oracle_ppl": oracle, "methods": methods})
    return output


def analyze_split(split, plan, evaluation):
    configs, resources = read(DATA / "architectures.json"), read(DATA / "resources.json")
    keys = read(DATA / "splits.json")[split]
    labels = read(DATA / (split + "_labels.json"))
    field = "valid_ppl" if split == "development" else "test_ppl"
    ppl = [labels[key][field] for key in keys]
    quality = [-v for v in ppl]
    instances = [configs[key] for key in keys]
    methods = extended_methods()
    scores = {name: [harness.canonical(function(copy.deepcopy(c))) for c in instances]
              for name, function in methods.items()}
    sizes = [resources[key]["total"] for key in keys]
    flops = [nsc.resource_estimate(c)["reference_flops_per_token"] for c in instances]
    scores.update(reported_parameters=sizes, reference_flops=flops)
    architecture_scores = {key: {"ppl": ppl[i],
        "scores": {name: values[i] for name, values in scores.items()}}
        for i, key in enumerate(keys)}
    if architecture_scores != evaluation["architecture_scores"][split]:
        raise RuntimeError("Evaluation data differ from current scores or labels; rerun evaluate_all.py")
    pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if instances[i]["d_model"] == instances[j]["d_model"]
               and min(sizes[i], sizes[j]) / max(sizes[i], sizes[j]) >= plan["parameter_ratio_threshold"]]
    both = [(i, j) for i, j in matched
            if min(flops[i], flops[j]) / max(flops[i], flops[j]) >= plan["flops_ratio_threshold"]]
    same_depth = [(i, j) for i, j in both if instances[i]["n_layer"] == instances[j]["n_layer"]]
    controls = {}
    for name, subset in zip(plan["resource_controls"], (matched, both, same_depth)):
        controls[name] = {
            "pair_count": len(subset), "covered_architectures": len({i for pair in subset for i in pair}),
            "pair_ids": [[keys[i], keys[j]] for i, j in subset],
            "ranking": {name: harness.pair_summary(v, quality, subset) for name, v in scores.items()},
        }
    ratios = [min(flops[i], flops[j]) / max(flops[i], flops[j]) for i, j in matched]
    parts = [decompose(c) for c in instances]
    scorer_errors = [abs(nsc.original_nsc(c) + a * part["prefix_term"]
                         - methods[method_name(a)](c))
                     for c, part in zip(instances, parts) for a in ALPHAS[1:]]
    residual_fractions = [abs(p["head_residual"]) / p["prefix_term"] for p in parts]
    adjacent_errors = []
    for config, part in zip(instances, parts):
        for i in range(config["n_layer"] - 1):
            swapped = copy.deepcopy(config)
            for name in ("n_head", "d_inner"):
                swapped[name][i], swapped[name][i+1] = swapped[name][i+1], swapped[name][i]
            predicted_delta = (part["attention"][i+1] * part["q"][i]
                               - part["attention"][i] * part["q"][i+1]) / part["denominator"]
            actual_delta = (methods["sensitivity_a1.0"](swapped)
                            - methods["sensitivity_a1.0"](config))
            adjacent_errors.append(abs(actual_delta - predicted_delta))
    if (max(scorer_errors) > 1e-8
            or max(abs(p["identity_error"]) for p in parts) > 1e-8
            or max(adjacent_errors) > 1e-8):
        raise RuntimeError("Algebraic decomposition or adjacent-swap identity failed")
    by_width = {}
    for width in sorted({c["d_model"] for c in instances}):
        indices = [i for i, c in enumerate(instances) if c["d_model"] == width]
        by_width[str(width)] = {"architectures": len(indices), "ranking": {
            name: {"spearman_rho": rho([v[i] for i in indices], [quality[i] for i in indices]),
                   "pairs": harness.pair_summary(v, quality, [(i, j) for i, j in pairs
                                                              if i in indices and j in indices])}
            for name, v in scores.items()}}
    diagnostic_errors = [(i, j) for i, j in matched
                         if (scores["original_nsc"][i] - scores["original_nsc"][j])
                         * (quality[i] - quality[j]) < 0]
    diagnostic_errors.sort(key=lambda pair: (-abs(ppl[pair[0]] - ppl[pair[1]]), pair))
    # Independent package validation is optional; it never enters the core scoring path.
    independent = {"status": "SKIPPED: optional SciPy unavailable"}
    try:
        import scipy
        from scipy.stats import kendalltau, spearmanr
    except ImportError:
        pass
    else:
        errors = []
        for name, values in scores.items():
            target = evaluation["ranking"][split][name]
            errors.extend([abs(float(spearmanr(values, quality).statistic) - target["spearman_rho"]),
                           abs(float(kendalltau(values, quality).statistic) - target["all_pairs"]["tau_b"])])
        independent = {"status": "PASS", "scipy_version": scipy.__version__, "max_abs_error": max(errors)}
        if max(errors) > 1e-12:
            raise RuntimeError("Independent ranking validation failed")
    return {
        "split": split, "label": field, "resource_controls": controls, "within_width": by_width,
        "resource_diagnostics": {"matched_flops_ratio_min": min(ratios),
            "matched_flops_ratio_median": statistics.median(ratios),
            "matched_pairs_flops_ratio_below_0_9": sum(v < 0.9 for v in ratios),
            "matched_pairs_different_depth": sum(instances[i]["n_layer"] != instances[j]["n_layer"]
                                                  for i, j in matched),
            "nsc_flops_spearman_rho": rho(scores["original_nsc"], flops)},
        "decomposition": {"max_abs_identity_error": max(abs(p["identity_error"]) for p in parts),
            "max_abs_scorer_reconstruction_error": max(scorer_errors),
            "adjacent_swaps_checked": len(adjacent_errors), "max_adjacent_swap_identity_error": max(adjacent_errors),
            "residual_fraction_median": statistics.median(residual_fractions),
            "residual_fraction_max": max(residual_fractions),
            "by_architecture": {key: {name: part[name] for name in (
                "s0", "ffn_invariant", "position", "head_residual", "prefix_term")}
                for key, part in zip(keys, parts)},
            "ranks_equal_without_head_residual": {str(a): harness.ranks(scores[method_name(a)])
                == harness.ranks(scores[method_name(a, "without_residual")]) for a in ALPHAS[1:]}},
        "paired_architecture_bootstrap": bootstrap(scores, quality, plan["bootstrap"]),
        "architecture_profiles": profiles(keys, configs, resources, scores, ppl),
        "selection_cases": failure_cases(keys, scores, ppl, sizes, evaluation["selection"][split]),
        "largest_matched_pair_errors": [{"ids": [keys[i], keys[j]], "absolute_ppl_gap": abs(ppl[i]-ppl[j]),
            "flops_ratio": min(flops[i], flops[j]) / max(flops[i], flops[j])}
            for i, j in diagnostic_errors[:5]],
        "independent_statistics": independent,
    }


def main():
    validate_selection()
    plan = {
        "status": "Proposal mechanism and resource analyses, with additional diagnostic settings; no method reselection",
        "supplementary_settings": "Detailed component ablations, extra FLOPs/depth matching "
                                  "and bootstrap settings were added after final outcomes were available.",
        "alphas": list(ALPHAS),
        "parameter_ratio_threshold": 0.9,
        "flops_ratio_threshold": 0.9,
        "resource_controls": ["same_width_parameters", "same_width_parameters_flops", "same_width_parameters_flops_depth"],
        "bootstrap": {"unit": "architecture", "repeats": 1000, "seed": 20261116,
                      "interval": "95% percentile; paired resampling, no training-seed interpretation"},
    }
    evaluation = read(HERE / "data/evaluation_summary.json")
    summary = {"status": plan["status"], "settings": plan,
               "splits": {split: analyze_split(split, plan, evaluation)
                          for split in ("development", "final")}}
    write(HERE / "data/analysis_summary.json", summary)
    for split, result in summary["splits"].items():
        print(split, "resource-control pairs:", {n: v["pair_count"] for n, v in result["resource_controls"].items()})
        print(split, "ranks unchanged without head residual:", result["decomposition"]["ranks_equal_without_head_residual"])
    print("written:", HERE / "data/analysis_summary.json")


if __name__ == "__main__":
    main()
