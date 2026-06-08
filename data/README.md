# Data

This directory holds the **Microsoft Azure Predictive Maintenance** dataset from Kaggle.
The CSVs are **not** committed — they're redistributed from a public Kaggle dataset under its
license, so we link to the source instead.

## Files expected by the notebook

```
data/
├── PdM_telemetry.csv     # ~80 MB, hourly sensor readings
├── PdM_errors.csv        # non-failure error codes
├── PdM_failures.csv      # ground-truth failure events
├── PdM_machines.csv      # machine metadata (model, age)
└── PdM_maint.csv         # scheduled + unscheduled maintenance
```

## Download

### Option A — Kaggle CLI (recommended)

```bash
pip install kaggle
# Put your kaggle.json in ~/.config/kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY env vars
kaggle datasets download -d arnabbiswas1/microsoft-azure-predictive-maintenance -p data/ --unzip
```

### Option B — manual

1. Open https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance
2. Click **Download**, unzip into this directory.

## Generated artifacts (also gitignored)

The notebook writes these here at runtime:

- `features.parquet` — built feature table
- `artifacts.joblib` — fitted pipeline + label encoders
- `metrics.json` — final test-set metrics
- `metrics_pre_leak_fix.json` — pre-fix baseline kept for the leakage write-up
