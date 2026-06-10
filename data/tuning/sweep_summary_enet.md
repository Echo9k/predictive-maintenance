# Sweep `enet-pdm` (elasticnet) — best config

- Trials: **10**  
- Best trial: **#7**  
- Validation any-failure PR-AUC: **0.9711**  

## Best hyperparameters

| param | value |
|---|---|
| `C` | 0.005337 |
| `l1_ratio` | 0.21506 |

## Held-out test metrics

| model | any-failure PR-AUC | macro recall | Brier |
|---|---|---|---|
| elasticnet_tuned | 0.9084 | 0.9302 | 0.0096 |
