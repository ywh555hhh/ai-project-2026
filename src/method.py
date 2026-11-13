"""Our method.

The single entry point is :func:`run`, which takes an instance and returns a
solution plus the metrics the report compares against the baselines.
"""


def run(instance, config: dict | None = None):
    """Solve one instance with our method.

    Args:
        instance: A benchmark instance produced by ``problem.load_instances``.
        config: Optional hyper-parameters; defaults are the reported ones.

    Returns:
        A dict with at least ``{"solution": ..., "score": float, "runtime": float}``.
    """
    raise NotImplementedError
