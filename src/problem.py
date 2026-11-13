"""Problem definition and benchmark instances.

Defines the objective, the instance loader and the deterministic seeds used so
that every experiment in the report is reproducible.
"""


def load_instances(split: str = "test"):
    """Load benchmark instances for the given split.

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        A list of instances in the project's canonical representation.
    """
    raise NotImplementedError
