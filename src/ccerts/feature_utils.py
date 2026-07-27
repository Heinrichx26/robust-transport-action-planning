from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import OneHotEncoder
except Exception:  # pragma: no cover - exercised in constrained runtimes
    OneHotEncoder = None

from ccerts.numpy_ml import HashingEncoder


@dataclass
class FeatureBundle:
    x_train: np.ndarray
    x_cal: np.ndarray
    x_test: np.ndarray
    encoder: object
    feature_names: list[str]


def build_feature_bundle(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> FeatureBundle:
    train_num = _numeric_matrix(train, numeric_cols)
    cal_num = _numeric_matrix(cal, numeric_cols)
    test_num = _numeric_matrix(test, numeric_cols)

    encoder = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
        if OneHotEncoder is not None
        else HashingEncoder(n_features=128)
    )
    train_cat = encoder.fit_transform(train[categorical_cols].astype(str))
    cal_cat = encoder.transform(cal[categorical_cols].astype(str))
    test_cat = encoder.transform(test[categorical_cols].astype(str))

    cat_names = encoder.get_feature_names_out(categorical_cols).tolist()
    return FeatureBundle(
        x_train=np.hstack([train_num, train_cat]),
        x_cal=np.hstack([cal_num, cal_cat]),
        x_test=np.hstack([test_num, test_cat]),
        encoder=encoder,
        feature_names=numeric_cols + cat_names,
    )


def transform_with_bundle(frame: pd.DataFrame, bundle: FeatureBundle, numeric_cols: list[str], categorical_cols: list[str]) -> np.ndarray:
    num = _numeric_matrix(frame, numeric_cols)
    cat = bundle.encoder.transform(frame[categorical_cols].astype(str))
    return np.hstack([num, cat])


def _numeric_matrix(frame: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    arrays = []
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        fill = float(values.median()) if values.notna().any() else 0.0
        arrays.append(values.fillna(fill).to_numpy(dtype=float))
    if not arrays:
        return np.zeros((len(frame), 0), dtype=float)
    return np.column_stack(arrays)


def add_history_features(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    keys: list[list[str]],
    target_col: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.copy()
    global_mean = float(pd.to_numeric(train[target_col], errors="coerce").mean())
    for key in keys:
        name = prefix + "_" + "_".join(key)
        stats = train.groupby(key)[target_col].agg(["mean", "count"]).reset_index()
        stats = stats.rename(columns={"mean": f"{name}_mean", "count": f"{name}_count"})
        out = out.merge(stats, on=key, how="left")
        out[f"{name}_mean"] = out[f"{name}_mean"].fillna(global_mean)
        out[f"{name}_count"] = out[f"{name}_count"].fillna(0.0)
    return out
