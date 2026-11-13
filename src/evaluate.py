"""Evaluation pipeline, baselines and metrics.

Aggregates per-instance results into the tables reported in the final report.
"""


def evaluate(instances, solver, **kwargs) -> dict:
    """Run ``solver`` over ``instances`` and aggregate the metrics."""
    raise NotImplementedError
