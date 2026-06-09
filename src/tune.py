"""
tune.py
-------
Hyperparameter search for the primary LightGBM model AND the elastic-net
baseline, with W&B experiment tracking. Two studies, one CLI; pick with
`--model {lgbm,enet,both}`.

Design choices (defensible, consistent with train_eval.py):
  * Same TEMPORAL split as train_eval.py. Tune on `fit`, score each trial on
    `cal`, keep the `test` tail untouched. The test slice is never read inside
    the search; it is the held-out generalisation check at the end.
  * LightGBM: Optuna TPE + MedianPruner. Per-trial early stopping via the
    `cal` slice. 6-knob search (capacity vs regularisation).
  * Elastic-net: Optuna TPE over `C` (log) and `l1_ratio` on the same
    stratified subsample as the baseline in train_eval.py — kept comparable
    so the talking point "we tuned both" is honest, not a moving target.
  * Objective for both studies: ANY-FAILURE PR-AUC on `cal`. Aligns the
    search with the headline metric at 1.9% positive rate.
  * W&B: one run per trial, grouped under a single sweep group per model.
    After each search, a summary run logs the best config, retrained
    test metrics, and uploads the model artifact.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

try:
    from .train_eval import (CLASSES, evaluate, feature_cols, load,
                             temporal_split)
except ImportError:  # allow `python src/tune.py` from the src/ directory
    from train_eval import (CLASSES, evaluate, feature_cols, load,
                            temporal_split)

DATA = Path(__file__).resolve().parents[1] / "data"
ARTIFACTS = DATA / "tuning"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def any_failure_pr_auc(proba: np.ndarray, y_true: np.ndarray) -> float:
    """Collapsed any-failure PR-AUC: P(failure) = 1 - P(none) vs binary y!=0."""
    return float(average_precision_score((y_true != 0).astype(int), 1.0 - proba[:, 0]))


def suggest_lgbm_params(trial: optuna.Trial) -> dict:
    """6-knob LightGBM search space — capacity vs regularisation."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def suggest_enet_params(trial: optuna.Trial) -> dict:
    """2-knob elastic-net search: regularisation strength and L1/L2 mix.

    Range matches the baseline in train_eval.py (C=0.5, l1_ratio=0.5) so the
    sweep can both confirm and improve on the hand-picked defaults.
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


def make_lgbm_objective(Xf, yf, Xc, yc, cat_features, wandb_cfg):
    """Build the LightGBM Optuna objective. Captures slices and W&B config."""
    import wandb

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        run = None
        if wandb_cfg["enabled"]:
            run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"] or None,
                group=wandb_cfg["group"],
                job_type="trial",
                name=f"trial-{trial.number:03d}",
                config={**params, "trial_number": trial.number},
                reinit=True,
            )

        model = LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=2000,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            **params,
        )
        t0 = time.perf_counter()
        model.fit(
            Xf, yf, categorical_feature=cat_features,
            eval_set=[(Xc, yc)], eval_metric="multi_logloss",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        fit_seconds = time.perf_counter() - t0

        proba = model.predict_proba(Xc)
        val_prauc = any_failure_pr_auc(proba, yc)
        val_brier = float(brier_score_loss((yc != 0).astype(int), 1.0 - proba[:, 0]))
        best_iter = int(model.best_iteration_ or model.n_estimators)

        if run is not None:
            run.log({
                "val/any_failure_pr_auc": val_prauc,
                "val/any_failure_brier": val_brier,
                "best_iteration": best_iter,
                "fit_seconds": fit_seconds,
            })
            run.summary["val/any_failure_pr_auc"] = val_prauc
            run.summary["best_iteration"] = best_iter
            run.finish()

        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("fit_seconds", fit_seconds)
        return val_prauc

    return objective


def fit_final_lgbm(best_params: dict, best_iter: int, Xf, yf, Xc, yc, cat_features):
    """Retrain LGBM at best_params for best_iter rounds, then isotonic-calibrate on cal."""
    model = LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=best_iter,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        **best_params,
    )
    model.fit(Xf, yf, categorical_feature=cat_features)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(Xc, yc)
    return model, calibrated


# ----------------------------------------------------------------------
# Elastic-net baseline tuning
# ----------------------------------------------------------------------

def _enet_subsample(yf: np.ndarray, seed: int = 42):
    """Stratified subsample mirroring train_eval.py: all positives + 40k negatives.

    Keeping this identical to the baseline keeps the comparison honest — the
    tuned numbers come from the same data the hand-picked baseline saw.
    """
    rng = np.random.RandomState(seed)
    pos_idx = np.where(yf != 0)[0]
    neg_pool = np.where(yf == 0)[0]
    neg_idx = rng.choice(neg_pool, size=min(40000, neg_pool.size), replace=False)
    return np.concatenate([pos_idx, neg_idx])


def make_enet_objective(Xf_s, yf, sub, Xc_s, yc, wandb_cfg):
    """Build the elastic-net Optuna objective. Inputs are pre-scaled arrays."""
    import wandb

    def objective(trial: optuna.Trial) -> float:
        params = suggest_enet_params(trial)
        run = None
        if wandb_cfg["enabled"]:
            run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg["entity"] or None,
                group=wandb_cfg["group"],
                job_type="trial",
                name=f"trial-{trial.number:03d}",
                config={**params, "trial_number": trial.number, "model": "enet"},
                reinit=True,
            )

        enet = LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=500,
            class_weight="balanced", n_jobs=-1, **params,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence warnings are expected at extremes
            enet.fit(Xf_s[sub], yf[sub])
        fit_seconds = time.perf_counter() - t0

        proba = enet.predict_proba(Xc_s)
        val_prauc = any_failure_pr_auc(proba, yc)
        val_brier = float(brier_score_loss((yc != 0).astype(int), 1.0 - proba[:, 0]))

        if run is not None:
            run.log({
                "val/any_failure_pr_auc": val_prauc,
                "val/any_failure_brier": val_brier,
                "fit_seconds": fit_seconds,
            })
            run.summary["val/any_failure_pr_auc"] = val_prauc
            run.finish()

        trial.set_user_attr("fit_seconds", fit_seconds)
        return val_prauc

    return objective


def fit_final_enet(best_params: dict, Xf_s, yf, sub):
    enet = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=1000,
        class_weight="balanced", n_jobs=-1, **best_params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enet.fit(Xf_s[sub], yf[sub])
    return enet


# ----------------------------------------------------------------------
# Study runners
# ----------------------------------------------------------------------

def _make_study(name: str, seed: int) -> optuna.Study:
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=name,
    )


def run_lgbm_study(args, df, fit, cal, test, cols, wandb_cfg):
    cat_features = ["model"] if "model" in cols else "auto"
    Xf, yf = fit[cols], fit["y"].values
    Xc, yc = cal[cols], cal["y"].values
    Xt, yt = test[cols], test["y"].values

    print(f"\n[lgbm] running {args.n_trials} Optuna trials | "
          f"wandb={'on' if wandb_cfg['enabled'] else 'off'}")
    study = _make_study(args.lgbm_study_name, args.seed)
    study.optimize(make_lgbm_objective(Xf, yf, Xc, yc, cat_features, wandb_cfg),
                   n_trials=args.n_trials, show_progress_bar=False)

    best = study.best_trial
    best_iter = int(best.user_attrs.get("best_iteration", 400))
    print(f"\n[lgbm] best #{best.number}: val PR-AUC = {best.value:.4f} | "
          f"best_iter={best_iter}")

    final_model, calibrated = fit_final_lgbm(best.params, best_iter,
                                             Xf, yf, Xc, yc, cat_features)
    test_results = {}
    evaluate("lightgbm_tuned_raw", final_model.predict_proba(Xt), yt, test_results)
    evaluate("lightgbm_tuned_calibrated", calibrated.predict_proba(Xt), yt, test_results)

    artifact_path = ARTIFACTS / "tuned_lgbm.joblib"
    joblib.dump({"model": final_model, "calibrated": calibrated, "cols": cols,
                 "best_params": best.params, "best_iteration": best_iter},
                artifact_path)

    summary = {
        "study_name": args.lgbm_study_name,
        "model_kind": "lightgbm",
        "n_trials": args.n_trials,
        "best_trial_number": best.number,
        "val_any_failure_pr_auc": best.value,
        "best_iteration": best_iter,
        "best_params": best.params,
        "test_metrics": test_results,
    }
    (ARTIFACTS / "sweep_summary_lgbm.json").write_text(json.dumps(summary, indent=2))
    _write_markdown_summary(summary, ARTIFACTS / "sweep_summary_lgbm.md")
    print(f"[lgbm] saved -> {artifact_path}")
    print(f"[lgbm] saved -> {ARTIFACTS / 'sweep_summary_lgbm.md'}")

    if wandb_cfg["enabled"]:
        import wandb
        run = wandb.init(
            project=wandb_cfg["project"], entity=wandb_cfg["entity"] or None,
            group=wandb_cfg["group"], job_type="summary",
            name=f"{args.lgbm_study_name}-best",
            config={**best.params, "best_iteration": best_iter, "model": "lgbm"},
            reinit=True,
        )
        run.summary["val/any_failure_pr_auc"] = best.value
        for split_name, res in test_results.items():
            run.summary[f"test/{split_name}/any_failure_pr_auc"] = res["any_failure_pr_auc"]
            run.summary[f"test/{split_name}/macro_recall"] = res["macro_recall"]
            run.summary[f"test/{split_name}/any_failure_brier"] = res["any_failure_brier"]
        imp = pd.DataFrame({
            "feature": cols,
            "gain": final_model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        run.log({"feature_importance": wandb.Table(dataframe=imp.head(30))})
        art = wandb.Artifact("lgbm_tuned", type="model",
                             description="Tuned LightGBM + isotonic calibration.")
        art.add_file(str(artifact_path))
        art.add_file(str(ARTIFACTS / "sweep_summary_lgbm.json"))
        run.log_artifact(art)
        run.finish()
    return summary


def run_enet_study(args, df, fit, cal, test, cols, wandb_cfg):
    num_cols = [c for c in cols if c != "model"]
    Xf_num, yf = fit[num_cols].values, fit["y"].values
    Xc_num, yc = cal[num_cols].values, cal["y"].values
    Xt_num, yt = test[num_cols].values, test["y"].values

    scaler = StandardScaler().fit(Xf_num)
    Xf_s, Xc_s, Xt_s = scaler.transform(Xf_num), scaler.transform(Xc_num), scaler.transform(Xt_num)
    sub = _enet_subsample(yf, seed=args.seed)

    n_trials = max(8, args.n_trials // 3)  # enet is faster + has fewer knobs
    print(f"\n[enet] running {n_trials} Optuna trials | "
          f"wandb={'on' if wandb_cfg['enabled'] else 'off'}")
    study = _make_study(args.enet_study_name, args.seed)
    study.optimize(make_enet_objective(Xf_s, yf, sub, Xc_s, yc, wandb_cfg),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n[enet] best #{best.number}: val PR-AUC = {best.value:.4f}")

    final_model = fit_final_enet(best.params, Xf_s, yf, sub)
    test_results = {}
    evaluate("elasticnet_tuned", final_model.predict_proba(Xt_s), yt, test_results)

    artifact_path = ARTIFACTS / "tuned_enet.joblib"
    joblib.dump({"model": final_model, "scaler": scaler, "num_cols": num_cols,
                 "best_params": best.params},
                artifact_path)

    summary = {
        "study_name": args.enet_study_name,
        "model_kind": "elasticnet",
        "n_trials": n_trials,
        "best_trial_number": best.number,
        "val_any_failure_pr_auc": best.value,
        "best_params": best.params,
        "test_metrics": test_results,
    }
    (ARTIFACTS / "sweep_summary_enet.json").write_text(json.dumps(summary, indent=2))
    _write_markdown_summary(summary, ARTIFACTS / "sweep_summary_enet.md")
    print(f"[enet] saved -> {artifact_path}")
    print(f"[enet] saved -> {ARTIFACTS / 'sweep_summary_enet.md'}")

    if wandb_cfg["enabled"]:
        import wandb
        run = wandb.init(
            project=wandb_cfg["project"], entity=wandb_cfg["entity"] or None,
            group=wandb_cfg["group"], job_type="summary",
            name=f"{args.enet_study_name}-best",
            config={**best.params, "model": "enet"},
            reinit=True,
        )
        run.summary["val/any_failure_pr_auc"] = best.value
        for split_name, res in test_results.items():
            run.summary[f"test/{split_name}/any_failure_pr_auc"] = res["any_failure_pr_auc"]
            run.summary[f"test/{split_name}/macro_recall"] = res["macro_recall"]
            run.summary[f"test/{split_name}/any_failure_brier"] = res["any_failure_brier"]
        art = wandb.Artifact("enet_tuned", type="model",
                             description="Tuned elastic-net baseline + StandardScaler.")
        art.add_file(str(artifact_path))
        art.add_file(str(ARTIFACTS / "sweep_summary_enet.json"))
        run.log_artifact(art)
        run.finish()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=30,
                    help="LGBM trial count; enet uses max(8, n_trials//3).")
    ap.add_argument("--model", choices=["lgbm", "enet", "both"], default="lgbm")
    ap.add_argument("--lgbm-study-name", type=str, default="lgbm-pdm")
    ap.add_argument("--enet-study-name", type=str, default="enet-pdm")
    ap.add_argument("--no-wandb", action="store_true",
                    help="Skip W&B entirely (useful for smoke tests).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()

    wandb_enabled = (not args.no_wandb) and os.getenv("WANDB_API_KEY", "").strip() != ""
    wandb_cfg = {
        "enabled": wandb_enabled,
        "project": os.getenv("WANDB_PROJECT", "quadsci-predictive-maintenance"),
        "entity": os.getenv("WANDB_ENTITY", "").strip(),
        "group": f"sweep-{int(time.time())}",
    }
    if not wandb_enabled:
        warnings.warn("W&B disabled (no WANDB_API_KEY or --no-wandb set); "
                      "tuning will still run, just without remote logging.")

    df = load()
    fit, cal, test = temporal_split(df)
    cols = feature_cols(df)
    print(f"split sizes  fit:{fit.shape[0]}  cal:{cal.shape[0]}  test:{test.shape[0]}")

    summaries = {}
    if args.model in ("lgbm", "both"):
        summaries["lgbm"] = run_lgbm_study(args, df, fit, cal, test, cols, wandb_cfg)
    if args.model in ("enet", "both"):
        summaries["enet"] = run_enet_study(args, df, fit, cal, test, cols, wandb_cfg)

    if args.model == "both":
        _write_combined_markdown(summaries, ARTIFACTS / "sweep_summary.md")
        print(f"\ncombined -> {ARTIFACTS / 'sweep_summary.md'}")
    else:
        # Keep the canonical filename pointing at whichever single study ran,
        # so the notebook cell can render `sweep_summary.md` unconditionally.
        kind = args.model
        src = ARTIFACTS / f"sweep_summary_{kind}.md"
        (ARTIFACTS / "sweep_summary.md").write_text(src.read_text())


def _write_markdown_summary(summary: dict, path: Path) -> None:
    """Render a single-study sweep summary as a notebook-friendly markdown table."""
    kind = summary.get("model_kind", "model")
    lines = [
        f"# Sweep `{summary['study_name']}` ({kind}) — best config",
        "",
        f"- Trials: **{summary['n_trials']}**  ",
        f"- Best trial: **#{summary['best_trial_number']}**  ",
        f"- Validation any-failure PR-AUC: **{summary['val_any_failure_pr_auc']:.4f}**  ",
    ]
    if "best_iteration" in summary:
        lines.append(f"- Best LightGBM iteration: **{summary['best_iteration']}**  ")
    lines += [
        "",
        "## Best hyperparameters",
        "",
        "| param | value |",
        "|---|---|",
    ]
    for k, v in summary["best_params"].items():
        lines.append(f"| `{k}` | {v:.5g}" if isinstance(v, float) else f"| `{k}` | {v}")
        lines[-1] += " |"
    lines += ["", "## Held-out test metrics", "",
              "| model | any-failure PR-AUC | macro recall | Brier |",
              "|---|---|---|---|"]
    for name, res in summary["test_metrics"].items():
        lines.append(f"| {name} | {res['any_failure_pr_auc']:.4f} "
                     f"| {res['macro_recall']:.4f} | {res['any_failure_brier']:.4f} |")
    path.write_text("\n".join(lines) + "\n")


def _write_combined_markdown(summaries: dict, path: Path) -> None:
    """Combined LGBM + enet markdown for the notebook."""
    parts = ["# Hyperparameter sweep — combined summary", ""]
    test_rows = []
    for kind, s in summaries.items():
        parts += [f"## {kind} — `{s['study_name']}`", "",
                  f"- Trials: **{s['n_trials']}** | "
                  f"Best #{s['best_trial_number']} | "
                  f"val PR-AUC **{s['val_any_failure_pr_auc']:.4f}**", "",
                  "| param | value |", "|---|---|"]
        for k, v in s["best_params"].items():
            parts.append(f"| `{k}` | {v:.5g} |" if isinstance(v, float) else f"| `{k}` | {v} |")
        parts.append("")
        for name, res in s["test_metrics"].items():
            test_rows.append((name, res))
    parts += ["## Held-out test — head-to-head", "",
              "| model | any-failure PR-AUC | macro recall | Brier |",
              "|---|---|---|---|"]
    for name, res in test_rows:
        parts.append(f"| {name} | {res['any_failure_pr_auc']:.4f} "
                     f"| {res['macro_recall']:.4f} | {res['any_failure_brier']:.4f} |")
    path.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
