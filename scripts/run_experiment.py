"""Main entry point: data preparation, method run and baseline comparison.

Usage:
    python scripts/run_experiment.py --prepare
    python scripts/run_experiment.py --method ours
    python scripts/run_experiment.py --method baseline
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="prepare the dataset")
    parser.add_argument("--method", choices=["ours", "baseline"], help="which method to run")
    args = parser.parse_args()
    raise NotImplementedError("implement the corresponding stage")


if __name__ == "__main__":
    main()
