from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_lower_quantile(pred_lower: np.ndarray, observed_lower: np.ndarray, alpha: float) -> float:
    scores = np.asarray(pred_lower, dtype=float) - np.asarray(observed_lower, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    level = np.ceil((scores.size + 1) * (1 - alpha)) / scores.size
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(scores, level, method="higher"))


def weighted_conformal_lower_quantile(
    pred_lower: np.ndarray,
    observed_lower: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> float:
    scores = np.asarray(pred_lower, dtype=float) - np.asarray(observed_lower, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(weights) & (weights >= 0)
    scores = scores[mask]
    weights = weights[mask]
    if scores.size == 0 or float(weights.sum()) <= 0:
        return 0.0
    order = np.argsort(scores)
    scores = scores[order]
    weights = weights[order] / weights.sum()
    cdf = np.cumsum(weights)
    idx = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    idx = min(idx, len(scores) - 1)
    return float(scores[idx])


def coverage_rate(certified: np.ndarray, observed_lower: np.ndarray) -> float:
    certified = np.asarray(certified, dtype=float)
    observed_lower = np.asarray(observed_lower, dtype=float)
    mask = np.isfinite(certified) & np.isfinite(observed_lower)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(observed_lower[mask] >= certified[mask]))


def selected_set_quantile(
    calibration: pd.DataFrame,
    score_col: str,
    value_col: str,
    budget_fraction: float,
    alpha: float,
    *,
    block_col: str = "block_id",
) -> tuple[float, int]:
    """Calibrate one scale-normalized nonconformity score per decision block."""
    scores: list[float] = []
    if block_col not in calibration.columns:
        return 0.0, 0
    for _, block in calibration.groupby(block_col):
        if block.empty:
            continue
        k = max(1, int(round(len(block) * budget_fraction)))
        selected = block.sort_values(score_col, ascending=False).head(k)
        predicted_value = float(pd.to_numeric(selected[value_col], errors="coerce").fillna(0.0).sum())
        observed_value = float(pd.to_numeric(selected["gain_lower"], errors="coerce").fillna(0.0).sum())
        scale = selected_set_scale(predicted_value, len(selected))
        scores.append((predicted_value - observed_value) / scale)
    if not scores:
        return 0.0, 0
    values = np.asarray(scores, dtype=float)
    level = np.ceil((len(values) + 1) * (1 - alpha)) / len(values)
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(values, level, method="higher")), int(len(values))


def selected_set_joint_quantile(
    calibration: pd.DataFrame,
    budgets: list[float],
    alpha: float,
    *,
    block_col: str = "block_id",
) -> tuple[float, int]:
    """Calibrate the maximum normalized selected-set error over all reported budgets."""
    scores: list[float] = []
    if block_col not in calibration.columns:
        return 0.0, 0
    for _, block in calibration.groupby(block_col):
        block_scores: list[float] = []
        for budget in budgets:
            key = int(round(budget * 100))
            score_col = f"score_ccerts_b{key}" if f"score_ccerts_b{key}" in block.columns else "score_ccerts"
            value_col = f"value_ccerts_b{key}" if f"value_ccerts_b{key}" in block.columns else score_col
            k = max(1, int(round(len(block) * budget)))
            selected = block.sort_values(score_col, ascending=False).head(k)
            predicted_value = float(pd.to_numeric(selected[value_col], errors="coerce").fillna(0.0).sum())
            observed_value = float(pd.to_numeric(selected["gain_lower"], errors="coerce").fillna(0.0).sum())
            scale = selected_set_scale(predicted_value, len(selected))
            block_scores.append((predicted_value - observed_value) / scale)
        if block_scores:
            scores.append(float(np.max(block_scores)))
    if not scores:
        return 0.0, 0
    values = np.asarray(scores, dtype=float)
    level = np.ceil((len(values) + 1) * (1 - alpha)) / len(values)
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(values, level, method="higher")), int(len(values))


def selected_set_scale(predicted_value: float, selected_n: int) -> float:
    """Pre-outcome scale for heteroscedastic selected-set value errors."""
    root_n = np.sqrt(max(int(selected_n), 1))
    return float(max(1.0, root_n + abs(float(predicted_value)) / root_n))


def evaluate_selected_set_certificate(
    calibration: pd.DataFrame,
    deployment: pd.DataFrame,
    budgets: list[float],
    alpha: float,
    *,
    block_col: str = "block_id",
    joint_budgets: bool = False,
) -> pd.DataFrame:
    """Evaluate coverage of the selected-set total lower-bound value."""
    rows: list[dict[str, float]] = []
    if block_col not in calibration.columns or block_col not in deployment.columns:
        return pd.DataFrame()
    joint_q, joint_blocks = selected_set_joint_quantile(
        calibration, budgets, alpha, block_col=block_col
    ) if joint_budgets else (0.0, 0)
    for budget in budgets:
        key = int(round(budget * 100))
        score_col = f"score_ccerts_b{key}" if f"score_ccerts_b{key}" in deployment.columns else "score_ccerts"
        value_col = f"value_ccerts_b{key}" if f"value_ccerts_b{key}" in deployment.columns else score_col
        if joint_budgets:
            q, calibration_blocks = joint_q, joint_blocks
        else:
            q, calibration_blocks = selected_set_quantile(
                calibration,
                score_col,
                value_col,
                budget,
                alpha,
                block_col=block_col,
            )
        for block_id, block in deployment.groupby(block_col):
            k = max(1, int(round(len(block) * budget)))
            selected = block.sort_values(score_col, ascending=False).head(k)
            fitted_value = float(pd.to_numeric(selected[value_col], errors="coerce").fillna(0.0).sum())
            set_scale = selected_set_scale(fitted_value, len(selected))
            certified_value = max(0.0, fitted_value - q * set_scale)
            observed_value = float(pd.to_numeric(selected["gain_lower"], errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "budget_fraction": budget,
                    "block_id": str(block_id),
                    "selected_n": float(k),
                    "calibration_blocks": float(calibration_blocks),
                    "set_quantile": q,
                    "set_scale": set_scale,
                    "uncalibrated_set_value": fitted_value,
                    "certified_set_value": certified_value,
                    "observed_set_value": observed_value,
                    "uncalibrated_hit": float(observed_value >= fitted_value),
                    "certificate_hit": float(observed_value >= certified_value),
                    "joint_budget_certificate": float(joint_budgets),
                }
            )
    return pd.DataFrame(rows)
