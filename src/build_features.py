"""
build_features.py
-----------------
Point-in-time-correct feature engineering for the Azure Predictive Maintenance
dataset. Produces a 3-hourly snapshot table with:

  - rolling telemetry features (mean+std over 3h and 24h, backward-looking only)
  - error counts over the last 24h (per error type)
  - days since last component replacement (per component)
  - machine metadata (model, age)
  - 24h-ahead multi-class label: which component (if any) fails in (t, t+24h]

Leakage discipline
==================
* Every feature at time t is computed from data with timestamp <= t.
* The label looks strictly FORWARD into (t, t+24h]; features never do.
* `failures` is a subset of `maint` (a failure at ft triggers a logged
  replacement at the same timestamp ft). The risk is intra-row: a grid row at
  exactly t == ft would pull that maint record into `days_since_compK = 0`,
  which is the very label we're trying to predict. Two protections close it:
    1. The label mask in `make_labels` is strict t < ft, so no row at t == ft
       is ever labeled positive — the leaky feature value has no positive row
       to attach to.
    2. The temporal split in train_eval.py adds a 1-day gap at each boundary,
       so a fit row's 24h label window cannot reach into the cal/test slices
       (this is the "airgap" — it handles the inter-segment case the strict
       label inequality doesn't cover).

This mirrors the canonical Azure "Predictive Maintenance Modelling Guide"
recipe, implemented from scratch with explicit point-in-time joins.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERRORS = ["error1", "error2", "error3", "error4", "error5"]
COMPS = ["comp1", "comp2", "comp3", "comp4"]
LABEL_HORIZON_H = 24            # predict failure within next 24 hours
SNAPSHOT_STRIDE_H = 3          # score every 3 hours (the standard grid)


def load_raw():
    tel = pd.read_csv(DATA / "PdM_telemetry.csv", parse_dates=["datetime"])
    err = pd.read_csv(DATA / "PdM_errors.csv", parse_dates=["datetime"])
    mnt = pd.read_csv(DATA / "PdM_maint.csv", parse_dates=["datetime"])
    fail = pd.read_csv(DATA / "PdM_failures.csv", parse_dates=["datetime"])
    mac = pd.read_csv(DATA / "PdM_machines.csv")
    return tel, err, mnt, fail, mac


def telemetry_features(tel: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking rolling mean/std over 3h and 24h, sampled on the 3h grid.

    Window is positional (counts rows), not time-based. That is equivalent to a
    true time window here because PdM_telemetry.csv is regular hourly with no
    gaps; on a production stream with sensor dropouts the two would diverge and
    this should switch to `.rolling("24h")` on a datetime index.
    """
    tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    feats = []
    for win in (3, 24):
        g = tel.groupby("machineID")[SENSORS]
        roll = g.rolling(window=win, min_periods=win)
        m = roll.mean().reset_index(level=0, drop=True)
        s = roll.std().reset_index(level=0, drop=True)
        m.columns = [f"{c}_mean_{win}h" for c in SENSORS]
        s.columns = [f"{c}_std_{win}h" for c in SENSORS]
        feats.append(m)
        feats.append(s)
    out = pd.concat([tel[["datetime", "machineID"]]] + feats, axis=1)
    # sample every SNAPSHOT_STRIDE_H hours (06:00 baseline => keep hours 6,9,12,...)
    out = out[out["datetime"].dt.hour % SNAPSHOT_STRIDE_H == 0]
    out = out.dropna().reset_index(drop=True)
    return out


def error_features(err: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Count of each error type over the trailing 24h, evaluated on the grid."""
    e = err.copy()
    e["one"] = 1
    wide = (e.pivot_table(index=["machineID", "datetime"], columns="errorID",
                          values="one", aggfunc="sum")
             .reindex(columns=ERRORS).fillna(0).reset_index())
    # expand to full hourly index per machine so rolling sums are well defined
    pieces = []
    for mid, gdf in wide.groupby("machineID"):
        full = (gdf.set_index("datetime").sort_index()
                   .reindex(pd.date_range(gdf.datetime.min(), gdf.datetime.max(), freq="h"),
                            fill_value=0))
        roll = full[ERRORS].rolling(f"{LABEL_HORIZON_H}h").sum()
        roll.columns = [f"{c}_count_24h" for c in ERRORS]
        roll["machineID"] = mid
        pieces.append(roll.reset_index().rename(columns={"index": "datetime"}))
    allroll = pd.concat(pieces, ignore_index=True)
    return grid.merge(allroll, on=["machineID", "datetime"], how="left").fillna(
        {f"{c}_count_24h": 0 for c in ERRORS})


def maint_features(mnt: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Days since last replacement of each component (point-in-time, backward only)."""
    out = grid.copy()
    for comp in COMPS:
        rep = (mnt[mnt["comp"] == comp][["machineID", "datetime"]]
               .rename(columns={"datetime": "rep_time"})
               .sort_values("rep_time"))
        col = f"days_since_{comp}"
        vals = np.full(len(out), np.nan)
        # merge_asof per machine: most recent replacement at or before t
        g = (pd.merge_asof(out.sort_values("datetime"),
                           rep.sort_values("rep_time"),
                           by="machineID", left_on="datetime", right_on="rep_time",
                           direction="backward"))
        days = (g["datetime"] - g["rep_time"]).dt.total_seconds() / 86400.0
        out = out.sort_values("datetime").reset_index(drop=True)
        out[col] = days.values
    return out


def make_labels(grid: pd.DataFrame, fail: pd.DataFrame) -> pd.DataFrame:
    """Multi-class label: component failing in (t, t+24h], else 'none'. Forward-looking."""
    out = grid.copy()
    out["label"] = "none"
    horizon = pd.Timedelta(hours=LABEL_HORIZON_H)
    # For each failure ft, mark grid rows of the same machine in [ft-24h, ft).
    # Strictly t < ft (not t <= ft): the grid row at t == ft would have access
    # via merge_asof(backward) to the failure-driven maintenance record at ft
    # (because failures are also logged in PdM_maint at the same timestamp),
    # turning `days_since_compK` into a near-perfect tell. Strict-less-than
    # closes that intra-row leak and also matches the "24h in advance" brief:
    # zero-lead-time rows aren't useful predictions anyway.
    out = out.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    idx = {mid: g.index.values for mid, g in out.groupby("machineID")}
    times = out["datetime"].values
    for _, row in fail.iterrows():
        mid, ft, comp = row["machineID"], row["datetime"], row["failure"]
        rows = idx.get(mid)
        if rows is None:
            continue
        t = times[rows]
        mask = (t >= np.datetime64(ft - horizon)) & (t < np.datetime64(ft))
        out.loc[rows[mask], "label"] = comp
    return out


def build():
    tel, err, mnt, fail, mac = load_raw()
    telf = telemetry_features(tel)
    grid = telf[["machineID", "datetime"]].copy()
    errf = error_features(err, grid)
    mntf = maint_features(mnt, grid)

    df = telf.merge(errf, on=["machineID", "datetime"], how="left")
    df = df.merge(mntf, on=["machineID", "datetime"], how="left")
    df = df.merge(mac, on="machineID", how="left")
    df = make_labels(df, fail)

    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    out_path = DATA / "features.parquet"
    df.to_parquet(out_path, index=False)
    print("feature table:", df.shape)
    print("label distribution:\n", df["label"].value_counts())
    print("positive rate: {:.4f}%".format((df["label"] != "none").mean() * 100))
    print("date range:", df.datetime.min(), "->", df.datetime.max())
    print("saved ->", out_path)
    return df


if __name__ == "__main__":
    build()
