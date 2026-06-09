# Sweep `enet-pdm` (elasticnet) — best config

- Trials: **10**  
- Best trial: **#4**  
- Validation any-failure PR-AUC: **0.9593**  

## Best hyperparameters

| param | value |
|---|---|
| `C` | 0.25378 |
| `l1_ratio` | 0.68727 |

## Held-out test metrics

| model | any-failure PR-AUC | macro recall | Brier |
|---|---|---|---|
| elasticnet_tuned | 0.9076 | 0.9323 | 0.0036 |
