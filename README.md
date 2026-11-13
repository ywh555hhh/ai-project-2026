# Course Project — Automated Heuristic Design (2026)

> One-paragraph summary of what this repository contains and what the task is.
> Replace this block once the topic is finalised: state the problem being solved,
> the approach taken, and the main result reported in the final report.

## Team

| Name | Student ID | Contact email | GitHub |
|------|-----------|---------------|--------|
|  |  |  | @ywh555hhh |
|  |  |  | |
|  |  |  | |

## Repository structure

```
.
├── README.md              # this file
├── requirements.txt       # Python dependencies
├── .gitignore
├── data/                  # small datasets or download scripts (see data/README.md)
├── src/                   # core implementation
│   ├── problem.py         # problem definition / benchmark instances
│   ├── method.py          # our method (heuristic / search procedure / scorer)
│   └── evaluate.py        # evaluation pipeline, baselines, metrics
└── scripts/               # entry points that reproduce every reported result
    ├── run_experiment.py  # main train/eval run
    └── plot_results.py    # regenerates the figures/tables in the report
```

## Environment setup

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Credentials are **never** committed. Export them before running:

```bash
export DEEPSEEK_API_KEY=...      # only if the project uses an LLM API
```

## Reproducing the results

Every row below maps to a table or figure in the final report.
Fill in the exact expected output and running time once the pipeline exists —
the graders rely on this table to reproduce the numbers.

| # | Step | Command | Expected output | Runtime |
|---|------|---------|-----------------|---------|
| 1 | Prepare data | `python scripts/run_experiment.py --prepare` | `data/processed/*.json` | ~1 min |
| 2 | Run our method | `python scripts/run_experiment.py --method ours` | `results/ours.json` | — |
| 3 | Run baseline | `python scripts/run_experiment.py --method baseline` | `results/baseline.json` | — |
| 4 | Figures / tables | `python scripts/plot_results.py` | `figures/*.pdf` | <1 min |

## Data

- Small datasets and download scripts live in `data/` (see `data/README.md`).
- Large files are **not** committed; they are fetched from a stable link and
  materialised by the download script.

## Graded commit

The graded snapshot is identified by a git tag, not by `main`:

```bash
git tag milestone        # submitted for the milestone (Nov 19)
git tag final            # submitted for the final report (Dec 3)
git push origin milestone final
```

The commit hash of the tag is pasted into the Canvas assignment.
