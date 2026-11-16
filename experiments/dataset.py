#!/usr/bin/env python3
"""Build and document the datasets used for evaluation.

Two artefacts are written to experiments/data/:

* ``benchmark_manifest.json`` — the chosen benchmark: the 200 archived
  GPT-2 / Transformer-XL configurations supplied by the course starter,
  their development / final split membership, resource fields, provenance,
  size-control pair counts and parameter-cap pool sizes.
* ``counterfactuals.json`` — OUR constructed dataset: synthetic layer
  reorderings used only for the mechanism check. These configurations have
  NO trained outcomes and must never be used to claim predictive accuracy.

This script deliberately reads no label file, so the dataset construction
cannot leak outcomes into any design decision.
"""
import copy
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "nsc-starter"
DATA = STARTER / "data"
OUT = Path(__file__).resolve().parent / "data"
CAPS = (10_000_000, 20_000_000, 30_000_000, 50_000_000)


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def minimal(config):
    """Keep only the four fields the NSC scorer actually consumes."""
    return {"d_model": config["d_model"], "n_layer": config["n_layer"],
            "n_head": copy.deepcopy(config["n_head"]),
            "d_inner": copy.deepcopy(config["d_inner"])}


def permute(config, reverse_ffn=False, rotate_ffn=False, reverse_heads=False):
    result = minimal(config)
    inner = list(result["d_inner"])
    heads = list(result["n_head"])
    if reverse_ffn:
        inner = inner[::-1]
    if rotate_ffn:
        inner = inner[1:] + inner[:1]
    if reverse_heads:
        heads = heads[::-1]
    result["d_inner"], result["n_head"] = inner, heads
    return result


def build_manifest():
    configs = read("architectures.json")
    resources = read("resources.json")
    splits = read("splits.json")
    full_provenance = read("provenance.json")
    provenance = {key: full_provenance[key] for key in
                  ("version", "source_file", "source_attribution", "source_paper",
                   "nsc_paper", "limitations")}
    provenance["full_record"] = "nsc-starter/data/provenance.json"

    split_info = {}
    for split, label_file, label_field, role in (
        ("development", "development_labels.json", "valid_ppl",
         "method development and a selection (only labels used for decisions)"),
        ("final", "final_labels.json", "test_ppl",
         "architecture holdout for evaluating the development-selected score and sensitivity settings"),
    ):
        keys = splits[split]
        totals = [resources[k]["total"] for k in keys]
        pairs = [(i, j) for i in range(len(keys)) for j in range(i)]
        matched = sum(
            1 for i, j in pairs
            if configs[keys[i]]["d_model"] == configs[keys[j]]["d_model"]
            and min(totals[i], totals[j]) / max(totals[i], totals[j]) >= 0.90
        )
        split_info[split] = {
            "count": len(keys), "label_file": label_file,
            "label_field": label_field, "role": role, "keys": keys,
            "hidden_width_counts": dict(sorted(Counter(configs[k]["d_model"] for k in keys).items())),
            "size_controlled_pairs": matched,
            "all_pairs": len(pairs),
            "parameter_cap_eligible": {str(cap): sum(t <= cap for t in totals) for cap in CAPS},
            "total_parameters_range": [min(totals), max(totals)],
        }

    return {
        "generated_by": "experiments/dataset.py",
        "benchmark": {
            "name": "CS5491 archived GPT-2 / Transformer-XL configuration panel",
            "source": "nsc-starter/data/architectures.json",
            "architectures": len(configs),
            "hidden_widths": sorted({c["d_model"] for c in configs.values()}),
            "depth_range": [min(c["n_layer"] for c in configs.values()),
                            max(c["n_layer"] for c in configs.values())],
            "pre_lnorm_false_count": sum(not c["pre_lnorm"] for c in configs.values()),
            "adaptive_embedding_count": sum(bool(c["adaptive"]) for c in configs.values()),
            "memory_length": sorted({c["mem_len"] for c in configs.values()}),
            "family_note": "Post-norm GPT-2 / Transformer-XL-style archive with adaptive "
                           "embeddings and mem_len=192; NOT a modern pre-normalized GPT decoder.",
        },
        "provenance": provenance,
        "splits": split_info,
        "size_control_rule": "same d_model; min(total)/max(total)>=0.90 (overlapping pairs)",
        "selection_rule": "parameter caps 10M/20M/30M/50M, k in {1,3}, same feasible pool for all methods",
        "resources_note": "Use resources.json 'total' directly; component fields overlap and must not be summed.",
        "label_policy": "No outcome values are copied into this manifest. "
                        "Final outcomes are held out by protocol, not by access control.",
    }


def build_counterfactuals():
    """Deterministic counterfactual panel from development configurations only."""
    configs = read("architectures.json")
    splits = read("splits.json")
    selected, per_width = {}, Counter()
    for key in splits["development"]:
        config = configs[key]
        if (len(config["n_head"]) >= 3 and len(set(config["n_head"])) >= 2
                and len(set(config["d_inner"])) >= 2):
            width = config["d_model"]
            if per_width[width] < 2:  # 2 sources per hidden width -> 8 sources
                selected.setdefault(width, []).append(key)
                per_width[width] += 1
        if sum(per_width.values()) == 8 and len(per_width) == 4:
            break

    cases = [{
        "case_id": "toy_ffn_order",
        "family": "toy_pair",
        "source_config": None,
        "note": "Two-layer teaching pair from the starter README; both orders have identical "
                "projection weights and reference MACs, so additive NSC cannot separate them.",
        "variants": {
            "ffn_256_512": {"d_model": 128, "n_layer": 2, "n_head": [4, 4], "d_inner": [256, 512]},
            "ffn_512_256": {"d_model": 128, "n_layer": 2, "n_head": [4, 4], "d_inner": [512, 256]},
        },
    }]

    transformations = (
        ("ffn_reversed_fixed_heads", dict(reverse_ffn=True),
         "Permute FFN widths, keep head counts: probes positional sensitivity."),
        ("ffn_rotated_fixed_heads", dict(rotate_ffn=True),
         "A second FFN permutation with fixed heads."),
        ("heads_reversed_fixed_ffn", dict(reverse_heads=True),
         "Permute attention head counts, keep FFN widths: isolates the attention-prefix effect."),
        ("both_reversed", dict(reverse_ffn=True, reverse_heads=True),
         "Permute both lists: combined structural change."),
    )
    for width in sorted(selected):
        for key in selected[width]:
            config = configs[key]
            for suffix, kwargs, note in transformations:
                cases.append({
                    "case_id": f"{key}__{suffix}",
                    "family": suffix,
                    "source_config": key,
                    "hidden_width": width,
                    "note": note,
                    "variants": {"original": minimal(config),
                                 "reordered": permute(config, **kwargs)},
                })
            uniform = minimal(config)
            uniform["n_head"] = [uniform["n_head"][0]] * uniform["n_layer"]
            cases.append({
                "case_id": f"{key}__uniform_heads_ffn_reversed",
                "family": "uniform_heads_ffn_reversed", "source_config": key,
                "hidden_width": width,
                "note": "Constant heads isolate positional sensitivity with no head heterogeneity.",
                "variants": {"original": uniform, "reordered": permute(uniform, reverse_ffn=True)},
            })

    return {
        "generated_by": "experiments/dataset.py",
        "purpose": "Mechanism check: does prefix-capacity reweighting respond to layer order, "
                   "and does the a=0 ablation remove that response?",
        "warning": "These are SYNTHETIC reorderings with NO trained outcomes. "
                   "They test structural sensitivity only, never predictive validity. "
                   "Do not assign an existing architecture's PPL to them.",
        "source_split": "development (identifiers only; no labels read)",
        "source_config_count": sum(len(v) for v in selected.values()),
        "case_count": len(cases),
        "cases": cases,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    counterfactuals = build_counterfactuals()
    (OUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "counterfactuals.json").write_text(
        json.dumps(counterfactuals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"benchmark: {manifest['benchmark']['architectures']} configs; "
          f"dev/final = {manifest['splits']['development']['count']}/"
          f"{manifest['splits']['final']['count']}")
    print(f"size-controlled pairs: dev={manifest['splits']['development']['size_controlled_pairs']}, "
          f"final={manifest['splits']['final']['size_controlled_pairs']}")
    print(f"counterfactual cases: {counterfactuals['case_count']} "
          f"from {counterfactuals['source_config_count']} source configs")
    print("written:", OUT / "benchmark_manifest.json", OUT / "counterfactuals.json")


if __name__ == "__main__":
    main()
