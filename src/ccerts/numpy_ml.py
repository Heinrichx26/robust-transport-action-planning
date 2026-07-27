from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


class HashingEncoder:
    """Deterministic bounded-width replacement for dense one-hot encoding."""

    def __init__(self, n_features: int = 128) -> None:
        self.n_features = int(n_features)
        self.columns_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "HashingEncoder":
        self.columns_ = [str(c) for c in frame.columns]
        return self

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(frame), self.n_features), dtype=np.float32)
        values = frame.astype(str).to_numpy()
        for j, column in enumerate(self.columns_):
            for i, value in enumerate(values[:, j]):
                token = f"{column}={value}".encode("utf-8", errors="replace")
                digest = hashlib.blake2b(token, digest_size=8).digest()
                integer = int.from_bytes(digest, "little", signed=False)
                slot = integer % self.n_features
                sign = 1.0 if ((integer >> 8) & 1) == 0 else -1.0
                out[i, slot] += sign
        scale = np.sqrt(np.maximum(np.sum(out * out, axis=1, keepdims=True), 1.0))
        return out / scale

    def get_feature_names_out(self, columns: list[str] | None = None) -> np.ndarray:
        return np.asarray([f"categorical_hash_{i:03d}" for i in range(self.n_features)], dtype=object)


class NumpyRidgeRegressor:
    """Ridge regression with an optional asymmetric-loss reweighting loop."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        objective: str = "regression",
        quantile: float = 0.8,
        max_iter: int = 8,
    ) -> None:
        self.alpha = float(alpha)
        self.objective = objective
        self.quantile = float(quantile)
        self.max_iter = int(max_iter)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: pd.Series | np.ndarray) -> "NumpyRidgeRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = np.nan_to_num((x - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0)
        design = np.column_stack([np.ones(len(z)), z])
        weights = np.ones(len(z), dtype=np.float64)
        coef = np.zeros(design.shape[1], dtype=np.float64)
        for _ in range(self.max_iter if self.objective == "quantile" else 1):
            root_w = np.sqrt(np.clip(weights, 1e-4, 1e4))
            xw = design * root_w[:, None]
            yw = y * root_w
            gram = xw.T @ xw
            penalty = np.eye(gram.shape[0]) * self.alpha
            penalty[0, 0] = 1e-8
            coef = np.linalg.solve(gram + penalty, xw.T @ yw)
            if self.objective == "quantile":
                residual = y - design @ coef
                weights = np.where(residual >= 0.0, self.quantile, 1.0 - self.quantile)
                weights /= np.maximum(np.abs(residual), 0.05 * (np.std(y) + 1e-8))
        self.coef_ = coef
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        z = np.nan_to_num((np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_, nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack([np.ones(len(z)), z]) @ self.coef_


class NumpyIsotonicRegression:
    """Pool-adjacent-violators isotonic regression with clipped extrapolation."""

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyIsotonicRegression":
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]
        unique_x, inverse = np.unique(xs, return_inverse=True)
        sums = np.bincount(inverse, weights=ys)
        counts = np.bincount(inverse).astype(float)
        levels = (sums / counts).tolist()
        block_counts = counts.tolist()
        left = list(range(len(levels)))
        right = list(range(len(levels)))
        k = 0
        while k < len(levels) - 1:
            if levels[k] <= levels[k + 1] + 1e-15:
                k += 1
                continue
            total = block_counts[k] + block_counts[k + 1]
            pooled = (levels[k] * block_counts[k] + levels[k + 1] * block_counts[k + 1]) / total
            levels[k] = pooled
            block_counts[k] = total
            right[k] = right[k + 1]
            for seq in (levels, block_counts, left, right):
                del seq[k + 1]
            k = max(k - 1, 0)
        fitted = np.empty(len(unique_x), dtype=float)
        for value, lo, hi in zip(levels, left, right):
            fitted[lo : hi + 1] = value
        self.x_ = unique_x
        self.y_ = fitted
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("Calibrator must be fitted before prediction.")
        return np.interp(np.asarray(x, dtype=float), self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


class NumpyLogisticRegression:
    def __init__(self, max_iter: int = 600, C: float = 0.5, random_state: int | None = None, solver: str | None = None) -> None:
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x - self.mean_) / self.scale_])
        coef = np.zeros(z.shape[1], dtype=float)
        learning_rate = 0.08
        penalty = 1.0 / max(self.C, 1e-8)
        for step in range(min(self.max_iter, 500)):
            logits = np.clip(z @ coef, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            grad = z.T @ (prob - y) / len(y)
            grad[1:] += penalty * coef[1:] / len(y)
            coef -= learning_rate / np.sqrt(1.0 + step / 40.0) * grad
        self.coef_ = coef
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Classifier must be fitted before prediction.")
        z = np.column_stack([np.ones(len(x)), (np.asarray(x, dtype=float) - self.mean_) / self.scale_])
        prob = 1.0 / (1.0 + np.exp(-np.clip(z @ self.coef_, -30.0, 30.0)))
        return np.column_stack([1.0 - prob, prob])


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def roc_auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        raise ValueError("AUC requires both classes.")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    return float((np.sum(ranks[pos]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
