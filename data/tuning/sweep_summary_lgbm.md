# Sweep `lgbm-pdm` (lightgbm) — best config

- Trials: **30**  
- Best trial: **#14**  
- Validation any-failure PR-AUC: **0.9996**  
- Best LightGBM iteration: **383**  

## Best hyperparameters

| param | value |
|---|---|
| `learning_rate` | 0.041058 |
| `num_leaves` | 87 |
| `min_child_samples` | 34 |
| `subsample` | 0.6714 |
| `colsample_bytree` | 0.90176 |
| `reg_lambda` | 9.28 |

## Held-out test metrics

| model | any-failure PR-AUC | macro recall | Brier |
|---|---|---|---|
| lightgbm_tuned_raw | 0.9700 | 0.9341 | 0.0016 |
| lightgbm_tuned_calibrated | 0.9217 | 0.9312 | 0.0016 |
