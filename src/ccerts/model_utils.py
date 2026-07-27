from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover - exercised in constrained runtimes
    HistGradientBoostingRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

from ccerts.numpy_ml import NumpyRidgeRegressor


@dataclass(frozen=True)
class ScoreBlock:
    name: str
    cal: np.ndarray
    test: np.ndarray
    target: str


def fit_lgbm_regressor(x: np.ndarray, y: pd.Series | np.ndarray, seed: int, *, objective: str = "regression", alpha: float | None = None):
    if LGBMRegressor is not None:
        params = {
            "objective": objective,
            "n_estimators": 260,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 0.1,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if alpha is not None:
            params["alpha"] = alpha
        model = LGBMRegressor(**params)
    elif HistGradientBoostingRegressor is not None:
        if objective == "quantile":
            model = HistGradientBoostingRegressor(loss="quantile", quantile=float(alpha or 0.8), max_iter=220, learning_rate=0.04, random_state=seed)
        else:
            model = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, l2_regularization=0.02, random_state=seed)
    else:
        model = NumpyRidgeRegressor(
            alpha=2.0,
            objective="quantile" if objective == "quantile" else "regression",
            quantile=float(alpha or 0.8),
        )
    model.fit(x, y)
    return model


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / (np.nanstd(values) + 1e-9)


def zscore_with_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return (values - np.nanmean(reference)) / (np.nanstd(reference) + 1e-9)


def group_history_score(train: pd.DataFrame, frame: pd.DataFrame, keys: list[str], target: str) -> np.ndarray:
    global_mean = float(pd.to_numeric(train[target], errors="coerce").mean())
    stats = train.groupby(keys)[target].mean().rename("_hist_target").reset_index()
    joined = frame.merge(stats, on=keys, how="left")
    return joined["_hist_target"].fillna(global_mean).to_numpy(dtype=float)


def top_budget_objective(score: np.ndarray, gain_lower: np.ndarray, budgets: tuple[float, ...]) -> float:
    score = np.asarray(score, dtype=float)
    gain_lower = np.asarray(gain_lower, dtype=float)
    order = np.argsort(score)[::-1]
    total = 0.0
    for budget in budgets:
        k = max(1, int(round(len(score) * budget)))
        total += float(np.sum(gain_lower[order[:k]]))
    return total / max(len(budgets), 1)


def choose_blend_weight(
    cal_left: np.ndarray,
    cal_right: np.ndarray,
    cal_gain_lower: np.ndarray,
    *,
    budgets: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20),
) -> tuple[float, float]:
    left = zscore(cal_left)
    right = zscore(cal_right)
    best_gamma = 1.0
    best_value = -np.inf
    for gamma in np.linspace(0.0, 1.0, 11):
        score = gamma * left + (1.0 - gamma) * right
        value = top_budget_objective(score, cal_gain_lower, budgets)
        if value > best_value:
            best_gamma = float(gamma)
            best_value = float(value)
    return best_gamma, best_value


def blend_scores(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return gamma * np.asarray(left, dtype=float) + (1.0 - gamma) * np.asarray(right, dtype=float)
