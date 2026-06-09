"""
train_eval.py
-------------
Temporal-split training + honest evaluation for the 24h-ahead multi-class
component-failure model.

Design choices (defensible to an interpretability-first, metrics-skeptical team):
  * TEMPORAL split, never random k-fold. Fit on the early year, calibrate on a
    middle slice, test on the held-out tail. 1-day gaps at each boundary so a
    training row's 24h label window cannot peek into the next segment.
  * LightGBM multi-class (5 classes: none, comp1..comp4) as the primary model.
  * class_weight="balanced" + threshold/queue logic. NO SMOTE: resampling
    distorts the base rate and wrecks calibration, and calibration is the whole
    point when you rank a work-queue by risk.
  * Elastic-net multinomial logistic as the honest interpretable baseline.
  * Isotonic calibration fit on the middle slice (prefit), evaluated on the tail.
  * Headline metrics: PER-CLASS RECALL, PR-AUC (one-vs-rest), macro-recall,
    any-failure top-decile capture, calibration (Brier). Accuracy/ROC-AUC are
    reported but explicitly de-emphasised at 1.9% positive rate.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (classification_report, confusion_matrix,
                             average_precision_score, brier_score_loss,
                             recall_score)
from lightgbm import LGBMClassifier

DATA = Path(__file__).resolve().parents[1] / "data"
CLASSES = ["none", "comp1", "comp2", "comp3", "comp4"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

FIT_END = pd.Timestamp("2015-07-01")
CAL_END = pd.Timestamp("2015-08-01")
GAP = pd.Timedelta(days=1)


def load():
    """Load the feature table and attach the integer label column `y` (CLASSES order)."""
    df = pd.read_parquet(DATA / "features.parquet")
    df["model"] = df["model"].astype("category")
    df["y"] = df["label"].map(LABEL2ID).astype(int)
    return df


def temporal_split(df):
    """Split into fit/cal/test by time, dropping a 1-day gap at each boundary.

    Fit is everything before FIT_END, calibration is the slice [FIT_END, CAL_END),
    and test is everything from CAL_END on. The GAP removed before FIT_END and
    CAL_END is the airgap: it keeps a fit/cal row's forward 24h label window from
    reaching into the next segment — the inter-segment leak that the strict label
    inequality in `make_labels` does not by itself cover.

    Returns:
        (fit, cal, test) DataFrames.
    """
    t = df["datetime"]
    fit = df[t < FIT_END - GAP]
    cal = df[(t >= FIT_END) & (t < CAL_END - GAP)]
    test = df[t >= CAL_END]
    return fit, cal, test


def feature_cols(df):
    """Model feature columns: every column except the keys, the raw label, and `y`."""
    drop = {"machineID", "datetime", "label", "y"}
    return [c for c in df.columns if c not in drop]


def evaluate(name, proba, y_true, out):
    """Score predictions with imbalance-aware metrics and record them in `out`.

    Computes per-class precision/recall, one-vs-rest PR-AUC, and macro-recall,
    plus — for the collapsed any-failure score (1 - P(none)) — its PR-AUC,
    top-decile capture, and Brier score. Accuracy is stored as
    `accuracy_misleading` to flag that it is uninformative at this base rate.
    Prints a short summary and stores the full result dict at `out[name]`.

    Args:
        name: Label for this model's results.
        proba: (n, 5) class-probability matrix, columns in CLASSES order.
        y_true: Integer true labels.
        out: Results dict accumulated across models (mutated in place).

    Returns:
        The result dict stored at `out[name]`.
    """
    pred = proba.argmax(1)
    rep = classification_report(y_true, pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    # PR-AUC one-vs-rest per class
    prauc = {}
    for i, c in enumerate(CLASSES):
        if (y_true == i).sum() > 0:
            prauc[c] = float(average_precision_score((y_true == i).astype(int), proba[:, i]))
    # any-failure detection
    p_fail = 1.0 - proba[:, 0]
    is_fail = (y_true != 0).astype(int)
    order = np.argsort(-p_fail)
    k = max(1, int(0.10 * len(p_fail)))
    top_decile_capture = float(is_fail[order[:k]].sum() / is_fail.sum())
    any_prauc = float(average_precision_score(is_fail, p_fail))
    # calibration of any-failure score
    brier = float(brier_score_loss(is_fail, p_fail))
    res = {
        "per_class_recall": {c: rep[c]["recall"] for c in CLASSES},
        "per_class_precision": {c: rep[c]["precision"] for c in CLASSES},
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "pr_auc_ovr": prauc,
        "any_failure_pr_auc": any_prauc,
        "any_failure_top_decile_capture": top_decile_capture,
        "any_failure_brier": brier,
        "accuracy_misleading": rep["accuracy"],
        "confusion": confusion_matrix(y_true, pred).tolist(),
    }
    out[name] = res
    print(f"\n=== {name} ===")
    print("per-class recall:", {c: round(v, 3) for c, v in res["per_class_recall"].items()})
    print("macro recall:", round(res["macro_recall"], 3))
    print("any-failure PR-AUC:", round(any_prauc, 3),
          "| top-decile capture:", round(top_decile_capture, 3),
          "| Brier:", round(brier, 4))
    return res


def main():
    """Train the models, evaluate on the held-out tail, and write metrics/artifacts.

    Loads features, makes the temporal split, fits the primary LightGBM, isotonic-
    calibrates it on the cal slice, fits the elastic-net baseline on a stratified
    subsample, evaluates all three on the test tail, and writes metrics.json and
    artifacts.joblib to data/.
    """
    df = load()
    fit, cal, test = temporal_split(df)
    cols = feature_cols(df)
    print("split sizes  fit:", fit.shape[0], "cal:", cal.shape[0], "test:", test.shape[0])
    print("test positives:", int((test.y != 0).sum()),
          "test pos-rate: {:.3f}%".format((test.y != 0).mean() * 100))

    Xf, yf = fit[cols], fit["y"].values
    Xc, yc = cal[cols], cal["y"].values
    Xt, yt = test[cols], test["y"].values
    out = {}

    # ---------- Primary: LightGBM multi-class ----------
    lgb = LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=400,
        learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(Xf, yf, categorical_feature=["model"])
    evaluate("lightgbm_raw", lgb.predict_proba(Xt), yt, out)

    # isotonic calibration on the middle slice (prefit)
    cal_lgb = CalibratedClassifierCV(FrozenEstimator(lgb), method="isotonic")
    cal_lgb.fit(Xc, yc)
    evaluate("lightgbm_calibrated", cal_lgb.predict_proba(Xt), yt, out)

    # ---------- Baseline: elastic-net multinomial logistic ----------
    num_cols = [c for c in cols if c != "model"]
    scaler = StandardScaler().fit(Xf[num_cols])
    # saga elastic-net is slow; the baseline only needs a representative fit.
    # Stratified subsample keeps all positives + a sample of negatives.
    rng = np.random.RandomState(42)
    pos_idx = np.where(yf != 0)[0]
    neg_idx = rng.choice(np.where(yf == 0)[0], size=min(40000, (yf == 0).sum()),
                         replace=False)
    sub = np.concatenate([pos_idx, neg_idx])
    enet = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga",
                              C=0.5, max_iter=500, class_weight="balanced",
                              n_jobs=-1)
    enet.fit(scaler.transform(Xf[num_cols].iloc[sub]), yf[sub])
    evaluate("elasticnet_baseline", enet.predict_proba(scaler.transform(Xt[num_cols])), yt, out)

    Path(DATA / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\nsaved metrics -> data/metrics.json")
    # persist primary model artifacts for the notebook / SHAP
    import joblib
    joblib.dump({"model": lgb, "calibrated": cal_lgb, "cols": cols,
                 "num_cols": num_cols, "scaler": scaler, "enet": enet},
                DATA / "artifacts.joblib")
    print("saved artifacts -> data/artifacts.joblib")


if __name__ == "__main__":
    main()
