# Hyperparameter sweep — combined summary

## lgbm — `lgbm-pdm`

- Trials: **30** | Best #14 | val PR-AUC **0.9996**

| param | value |
|---|---|
| `learning_rate` | 0.041058 |
| `num_leaves` | 87 |
| `min_child_samples` | 34 |
| `subsample` | 0.6714 |
| `colsample_bytree` | 0.90176 |
| `reg_lambda` | 9.28 |

## enet — `enet-pdm`

- Trials: **10** | Best #7 | val PR-AUC **0.9711**

| param | value |
|---|---|
| `C` | 0.005337 |
| `l1_ratio` | 0.21506 |

## Held-out test — head-to-head

| model | any-failure PR-AUC | macro recall | Brier |
|---|---|---|---|
| lightgbm_tuned_raw | 0.9700 | 0.9341 | 0.0016 |
| lightgbm_tuned_calibrated | 0.9217 | 0.9312 | 0.0016 |
| elasticnet_tuned | 0.9084 | 0.9302 | 0.0096 |
