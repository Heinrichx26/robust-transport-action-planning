from __future__ import annotations

import numpy as np
import pandas as pd

from ccerts.feature_utils import add_history_features, build_feature_bundle


def valid_periods(frame: pd.DataFrame, min_period_rows: int) -> list[str]:
    counts = frame["period"].astype(str).value_counts()
    return sorted(counts[counts >= min_period_rows].index.tolist())


def sample_frame(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed).copy()


def prepare_features(
    dataset: str,
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    train_h, cal_h, test_h, numeric_cols, categorical_cols = add_dataset_features(dataset, train, cal, test)
    bundle = build_feature_bundle(train_h, cal_h, test_h, numeric_cols, categorical_cols)
    return train_h, cal_h, test_h, bundle.x_train, bundle.x_cal, bundle.x_test


def add_dataset_features(
    dataset: str,
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if dataset == "bts":
        for frame in [train, cal, test]:
            frame["dep_hour"] = pd.to_numeric(frame["dep_hour"], errors="coerce").fillna(0).astype(int)
            frame["route"] = frame["route"].astype(str)
        keys = [["route"], ["origin"], ["carrier"], ["origin", "dep_hour"]]
        train_h = add_history_features(train.copy(), train, keys, "loss", "hist")
        cal_h = add_history_features(cal, train, keys, "loss", "hist")
        test_h = add_history_features(test, train, keys, "loss", "hist")
        numeric = [c for c in ["dep_hour", "distance", "scheduled_elapsed", "schedule_slack"] if c in test_h.columns]
        numeric += [c for c in test_h.columns if c.startswith("hist_")]
        categorical = ["carrier", "origin", "destination"]
        return train_h, cal_h, test_h, numeric, categorical

    for frame in [train, cal, test]:
        frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce").fillna(0).astype(int)
        frame["od"] = frame["od"].astype(str)
    keys = [["od"], ["pickup_zone"], ["hour"], ["pickup_zone", "hour"]]
    train_h = add_history_features(train.copy(), train, keys, "loss", "hist")
    cal_h = add_history_features(cal, train, keys, "loss", "hist")
    test_h = add_history_features(test, train, keys, "loss", "hist")
    if dataset == "divvy":
        numeric = [
            c
            for c in ["hour", "day_of_week", "is_weekend", "capacity_proxy", "inventory_start"]
            if c in test_h.columns
        ]
    elif dataset == "road":
        numeric = [
            c
            for c in [
                "hour",
                "day_of_week",
                "is_weekend",
                "free_flow_speed",
                "lag_speed",
                "lag_network_headroom",
            ]
            if c in test_h.columns
        ]
    else:
        # Current-period trip counts and durations are outcomes at the selection time.
        # Historical aggregates below retain demand context without decision-time leakage.
        numeric = ["hour"]
    numeric += [c for c in test_h.columns if c.startswith("hist_")]
    categorical = ["pickup_zone", "dropoff_zone"]
    return train_h, cal_h, test_h, numeric, categorical
