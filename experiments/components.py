"""Exact decomposition of prefix-capacity reweighting; no outcome access."""
import math

import student_score as ours
from nsc import layer_capacities, layers, psi_mp


def decompose(config):
    d, heads, inner = layers(config)
    b = layer_capacities(config)
    q = [2 * psi_mp(d, f) for f in inner]
    attention = [3 * h * psi_mp(d, d // h) + psi_mp(d, d) for h in heads]
    total = math.fsum(b)
    denominator = total + ours.EPS
    mean_attention = math.fsum(attention) / len(attention)
    invariant = math.fsum(q[i] * q[j] for i in range(len(q)) for j in range(i)) / denominator
    positional = mean_attention * math.fsum(i * v for i, v in enumerate(q)) / denominator
    residual = math.fsum(q[i] * math.fsum(a - mean_attention for a in attention[:i])
                         for i in range(len(q))) / denominator
    direct = math.fsum(q[i] * math.fsum(b[:i]) for i in range(len(q))) / denominator
    return {"s0": total, "denominator": denominator, "q": q, "attention": attention,
            "ffn_invariant": invariant, "position": positional,
            "head_residual": residual, "prefix_term": direct,
            "identity_error": direct - invariant - positional - residual}


def without_head_residual(config, alpha):
    parts = decompose(config)
    return parts["s0"] + alpha * (parts["ffn_invariant"] + parts["position"])
