from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionResult:
    method: str
    budget_fraction: float
    selected: pd.DataFrame
    metrics: dict[str, float]


def top_budget_select(
    frame: pd.DataFrame,
    score_col: str,
    budget_fraction: float,
    method: str,
    group_col: str | None = None,
    group_cap_fraction: float | None = None,
) -> SelectionResult:
    if frame.empty:
        return SelectionResult(method, budget_fraction, frame.copy(), {})

    k = max(1, int(round(len(frame) * budget_fraction)))
    work = frame.copy()
    score_col = _resolve_score_col(work, score_col, budget_fraction, method)
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(-np.inf)

    if group_col is None:
        selected = work.sort_values("_score", ascending=False).head(k)
    else:
        cap = max(1, int(round(k * float(group_cap_fraction or 0.25))))
        pieces = []
        counts: dict[object, int] = {}
        for _, row in work.sort_values("_score", ascending=False).iterrows():
            group = row[group_col]
            if counts.get(group, 0) >= cap:
                continue
            pieces.append(row)
            counts[group] = counts.get(group, 0) + 1
            if len(pieces) >= k:
                break
        selected = pd.DataFrame(pieces, columns=work.columns)

    metrics = selection_metrics(frame, selected)
    return SelectionResult(method, budget_fraction, selected.drop(columns=["_score"], errors="ignore"), metrics)


def _resolve_score_col(frame: pd.DataFrame, score_col: str, budget_fraction: float, method: str) -> str:
    if method == "C-CERTS":
        budget_key = int(round(budget_fraction * 100))
        candidate = f"{score_col}_b{budget_key}"
        if candidate in frame.columns:
            return candidate
    return score_col


def selection_metrics(frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float]:
    total_loss = float(np.clip(frame["loss"].to_numpy(dtype=float), 0.0, None).sum())
    total_gain_lb = float(np.clip(frame["gain_lower"].to_numpy(dtype=float), 0.0, None).sum())
    selected_loss = float(np.clip(selected["loss"].to_numpy(dtype=float), 0.0, None).sum()) if not selected.empty else 0.0
    selected_gain_lb = (
        float(np.clip(selected["gain_lower"].to_numpy(dtype=float), 0.0, None).sum()) if not selected.empty else 0.0
    )
    selected_cert = (
        float(np.clip(selected["cert_gain"].to_numpy(dtype=float), 0.0, None).sum())
        if "cert_gain" in selected and not selected.empty
        else 0.0
    )
    high_threshold = float(frame["loss"].quantile(0.9)) if len(frame) else 0.0
    high_total = int((frame["loss"] >= high_threshold).sum())
    high_hits = int((selected["loss"] >= high_threshold).sum()) if not selected.empty else 0
    return {
        "selected_n": float(len(selected)),
        "loss_capture": selected_loss / (total_loss + 1e-9),
        "gain_lower_capture": selected_gain_lb / (total_gain_lb + 1e-9),
        "certified_gain_sum": selected_cert,
        "gain_lower_sum": selected_gain_lb,
        "high_loss_recall": high_hits / max(high_total, 1),
    }


def symmetric_difference_rate(left: pd.DataFrame, right: pd.DataFrame, id_col: str = "unit_id") -> float:
    a = set(left[id_col].tolist())
    b = set(right[id_col].tolist())
    denom = max(len(a | b), 1)
    return len(a ^ b) / denom


def budget_curve(
    frame: pd.DataFrame,
    budgets: list[float],
    score_cols: dict[str, str],
    group_col: str | None = None,
    group_cap_fraction: float | None = None,
) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        selections = {
            method: top_budget_select(frame, col, budget, method, group_col, group_cap_fraction)
            for method, col in score_cols.items()
        }
        reference = selections.get("C-CERTS")
        risk_first = selections.get("RiskFirst")
        for method, result in selections.items():
            row = {"method": method, "budget_fraction": budget}
            row.update(result.metrics)
            if reference is not None:
                row["set_diff_vs_ccerts"] = symmetric_difference_rate(result.selected, reference.selected)
            if risk_first is not None and method == "C-CERTS":
                row["set_diff_vs_risk"] = symmetric_difference_rate(result.selected, risk_first.selected)
            rows.append(row)
    return pd.DataFrame(rows)


def block_budget_curve(
    frame: pd.DataFrame,
    budgets: list[float],
    score_cols: dict[str, str],
    *,
    block_col: str = "block_id",
    group_col: str | None = None,
    group_cap_fraction: float | None = None,
) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        selected_by_method: dict[str, list[pd.DataFrame]] = {method: [] for method in score_cols}
        for _, block in frame.groupby(block_col):
            for method, col in score_cols.items():
                selected = top_budget_select(block, col, budget, method, group_col, group_cap_fraction).selected
                selected_by_method[method].append(selected)
        selections: dict[str, SelectionResult] = {}
        for method, pieces in selected_by_method.items():
            selected = pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()
            selections[method] = SelectionResult(method, budget, selected, selection_metrics(frame, selected))
        reference = selections.get("C-CERTS")
        risk_first = selections.get("RiskFirst")
        for method, result in selections.items():
            row = {"method": method, "budget_fraction": budget}
            row.update(result.metrics)
            if reference is not None:
                row["set_diff_vs_ccerts"] = symmetric_difference_rate(result.selected, reference.selected)
            if risk_first is not None and method == "C-CERTS":
                row["set_diff_vs_risk"] = symmetric_difference_rate(result.selected, risk_first.selected)
            rows.append(row)
    return pd.DataFrame(rows)
