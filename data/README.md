# Data

Rule of thumb: small files (and download scripts) are committed here;
large files are fetched from a stable link by `scripts/download_data.sh`.

```
data/
├── README.md          # this file
├── raw/               # gitignored; populated by the download script
└── processed/         # gitignored; produced by scripts/run_experiment.py --prepare
```

## Benchmark / dataset used

| Item | Value |
|------|-------|
| Dataset / benchmark | TBD |
| Size | TBD |
| Source link | TBD |
| Licence | TBD |

## How to obtain it

```bash
bash scripts/download_data.sh      # TODO: implement, document the expected layout
```
