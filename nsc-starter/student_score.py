"""Our CS5491 method: prefix-capacity reweighting of Neural Spectral Capacity.

Proposal: "Structural Sensitivity and Predictive Utility of NSC Reweighting".

The original NSC score is additive over layers, so any permutation of layers
that preserves the same projections receives the same score. We add a
prefix-capacity term as a structural proxy for FFN context:

    B_l  = supplied per-layer capacity (nsc.layer_capacities)
    q_l  = 2 * psi_MP(d, f_l, 0.02)          (the FFN part of B_l)
    S0   = sum_l B_l                          (original NSC)
    C_l  = sum_{j<l} B_j / (S0 + eps)        (prefix capacity, eps = 1e-12)
    S_a  = S0 + a * sum_l C_l * q_l

Larger scores are predicted to be better, matching the evaluator convention.
``ablation`` sets a = 0, which disables ONLY the prefix mechanism and recovers
original NSC exactly. ``position_only_control`` replaces C_l by (l-1)/L (with
1-based l), the position-only baseline required by the proposal.

The evaluator passes architecture configuration only. No label file is read,
no architecture ID is looked up, and no configuration fingerprint is used as
an outcome lookup inside any scorer.
"""
import math

from nsc import layer_capacities, layers, original_nsc, psi_mp

EPS = 1e-12

# FROZEN on the development split only (see experiments/frozen_config.json).
# Chosen by the a-sweep in experiments/select.py; do not change after the
# final split has been evaluated.
ALPHA = 0.0


def prefix_weights(config):
    """Return (per-layer capacities B, normalized prefix weights C).

    C[0] is identically zero: no layer precedes the first one. This is the
    "C_l = 0 for one layer" limiting case from the proposal.
    """
    capacities = layer_capacities(config)
    total = math.fsum(capacities)
    weights, running = [], 0.0
    for value in capacities:
        weights.append(running / (total + EPS))
        running += value
    return capacities, weights


def ffn_terms(config):
    """q_l = 2 * psi_MP(d, f_l, 0.02): the FFN contribution to each layer."""
    d, _heads, inner = layers(config)
    return [2.0 * psi_mp(d, f) for f in inner]


def reweighted_nsc(config, alpha, weighting="prefix"):
    """S_a under the requested weighting scheme.

    weighting="prefix"   -> our method (prefix-capacity reweighting)
    weighting="position" -> position-only control, C_l = (l-1)/L
    weighting="none"     -> zero weights, i.e. original NSC
    """
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if weighting not in ("prefix", "position", "none"):
        raise ValueError("unknown weighting scheme")
    capacities, prefix = prefix_weights(config)
    total = math.fsum(capacities)
    q = ffn_terms(config)
    length = len(capacities)
    if weighting == "prefix":
        weights = prefix
    elif weighting == "position":
        weights = [i / length for i in range(length)]  # (l-1)/L, 1-based l
    else:
        weights = [0.0] * length
    return total + alpha * math.fsum(c * term for c, term in zip(weights, q))


def score(config):
    """Our frozen method: prefix-capacity reweighted NSC."""
    return reweighted_nsc(config, ALPHA, "prefix")


def ablation(config):
    """Disable only the proposed mechanism (a = 0) -> original NSC."""
    return reweighted_nsc(config, 0.0, "prefix")


def position_only_control(config, alpha=None):
    """Position-only baseline required by the proposal, at the frozen a."""
    return reweighted_nsc(config, ALPHA if alpha is None else alpha, "position")


def sanity_check():
    """Developer-only self-check; never called by the official evaluator."""
    for config in ({'d_model': 128, 'n_layer': 2, 'n_head': [4, 4],
                    'd_inner': [256, 512]},
                   {'d_model': 128, 'n_layer': 2, 'n_head': [4, 4],
                    'd_inner': [512, 256]}):
        assert ablation(config) == original_nsc(config)
        assert reweighted_nsc(config, 0.0, "none") == original_nsc(config)
        assert prefix_weights(config)[1][0] == 0.0
    return True


if __name__ == "__main__":
    print("student_score sanity_check:", sanity_check())
