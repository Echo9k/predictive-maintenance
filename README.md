# Predictive Maintenance — Azure Telemetry

End-to-end pipeline that predicts component failures on the **Microsoft Azure Predictive
Maintenance** dataset (100 machines, one year of hourly telemetry).

The deliverable is `predictive_maintenance.ipynb`. The `src/` modules are the reusable
building blocks the notebook calls into.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then fetch the data — see [`data/README.md`](data/README.md) for the Kaggle download command.

## Run

Open the notebook and run top-to-bottom:

```bash
jupyter lab predictive_maintenance.ipynb
```

The notebook will write `features.parquet`, `artifacts.joblib`, and `metrics.json` into `data/`.

## Layout

```
predictive_maintenance.ipynb   # the deliverable — story + results
src/
  build_features.py            # leakage-safe feature builder
  train_eval.py                # CV, fit, calibration, evaluation
  make_notebook.py             # regenerates the notebook from src/ (optional)
data/                          # raw CSVs (gitignored) + generated artifacts
requirements.txt               # pinned versions used to produce the reported numbers
```

## Notes

- Time-aware train/val/test split — no shuffled k-fold.
- Features are built strictly from past windows; the build step is unit-tested against a
  one-step shift to bound any residual leakage.
- Metrics in `data/metrics_pre_leak_fix.json` are kept on purpose as the before-picture for
  the leakage write-up in the notebook.
