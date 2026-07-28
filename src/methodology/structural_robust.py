from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ccerts.robust_endpoints import _ctm_gain, _divvy_gain


@dataclass(frozen=True)
class ScenarioPrediction:
    names: tuple[str, ...]
    calibration: np.ndarray
    deployment: np.ndarray
    joint_correction: float
    rank: int


@dataclass(frozen=True)
class PlanSolution:
    selected: np.ndarray
    predicted_worst_value: float
    feasible: bool
    solver: str
    gap: float


@dataclass(frozen=True)
class AnchoredPlanSolution:
    selected: np.ndarray
    baseline: np.ndarray
    predicted_worst_improvement: float
    changed_actions: int
    feasible: bool
    solver: str
    gap: float


@dataclass(frozen=True)
class FrontierPoint:
    """Best protected improvement for an exact number of changed actions."""

    changed_actions: int
    protected_improvement: float
    selected: np.ndarray
    feasible: bool


def scenario_gain_matrix(dataset: str, frame: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray]:
    """Return loss reductions under every predeclared structural setting."""
    loss = _num(frame, "loss")
    reference = np.minimum(np.maximum(_num(frame, "gain_lower"), 0.0), np.maximum(loss, 0.0))
    values: list[np.ndarray] = [reference]
    names = ["reference"]

    if dataset == "bts":
        slack = np.maximum(_num(frame, "schedule_slack"), 0.0)
        actionable = np.maximum(_num(frame, "actionable_delay"), 0.0)
        reliability = np.clip(_num(frame, "recovery_reliability"), 0.0, 1.0)
        scale = 0.50 + 0.50 * reliability
        for slack_weight, delay_weight in [(0.25, 0.10), (0.50, 0.10), (0.20, 0.25)]:
            values.append(np.minimum(loss, slack_weight * slack + delay_weight * actionable) * scale)
            names.append(f"slack_{slack_weight:.2f}_delay_{delay_weight:.2f}")
    elif dataset == "road":
        speed = _num(frame, "mean_speed", 5.0)
        free_flow = _num(frame, "free_flow_speed", 25.0)
        headroom = _num(frame, "network_headroom")
        for jam_density, cell_length, control_scale in product(
            [120.0, 150.0, 180.0], [0.6, 0.8, 1.0], [0.75, 1.0, 1.25]
        ):
            values.append(
                0.70
                * _ctm_gain(
                    speed,
                    free_flow,
                    headroom,
                    jam_density=jam_density,
                    cell_length_km=cell_length,
                    control_scale=control_scale,
                )
            )
            names.append(f"ctm_{int(jam_density)}_{cell_length:.1f}_{control_scale:.2f}")
    elif dataset == "divvy":
        for multiplier in [0.50, 1.00, 1.50]:
            values.append(_divvy_gain(frame, multiplier))
            names.append(f"truck_{multiplier:.2f}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    matrix = np.column_stack(values)
    matrix = np.clip(matrix, 0.0, np.maximum(loss[:, None], 0.0))
    return tuple(names), matrix


def contamination_robust_values(
    scenario_values: np.ndarray,
    contamination: float,
) -> np.ndarray:
    """Represent a contamination ambiguity set by its extreme distributions.

    The centre assigns equal mass to the declared structural settings. A share
    ``contamination`` may move to any one setting. Applying a plan selector to
    the returned columns therefore maximises the worst expected value over all
    extreme distributions of this ambiguity set.
    """
    values = np.asarray(scenario_values, dtype=float)
    eta = float(contamination)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("contamination must lie in [0, 1].")
    centre = np.mean(values, axis=1, keepdims=True)
    return (1.0 - eta) * centre + eta * values


def fit_low_rank_scenario_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calibration: np.ndarray,
    y_calibration: np.ndarray,
    x_deployment: np.ndarray,
    *,
    alpha: float = 0.10,
    max_rank: int = 6,
) -> ScenarioPrediction:
    """Fit nonlinear models for all structural settings and measure joint error."""
    from ccerts.model_utils import fit_lgbm_regressor

    cal_columns: list[np.ndarray] = []
    deployment_columns: list[np.ndarray] = []
    for scenario in range(y_train.shape[1]):
        model = fit_lgbm_regressor(x_train, y_train[:, scenario], 20260724 + scenario)
        cal_columns.append(np.maximum(model.predict(x_calibration), 0.0))
        deployment_columns.append(np.maximum(model.predict(x_deployment), 0.0))
    cal_raw = np.column_stack(cal_columns)
    deployment_raw = np.column_stack(deployment_columns)
    # The joint error is retained for diagnostics. Complete-plan residuals perform
    # the inferential calibration after optimization.
    row_error = np.max(cal_raw - y_calibration, axis=1)
    n = len(row_error)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1))
    correction = float(np.quantile(row_error, level, method="higher")) if n else 0.0
    correction = max(correction, 0.0)
    return ScenarioPrediction(
        names=tuple(f"scenario_{i}" for i in range(y_train.shape[1])),
        calibration=cal_raw,
        deployment=deployment_raw,
        joint_correction=correction,
        rank=y_train.shape[1],
    )


def robust_plan_select(
    scenario_values: np.ndarray,
    budget: int,
    *,
    groups: np.ndarray | None = None,
    group_cap: int | None = None,
) -> PlanSolution:
    """Maximise the worst total value across common structural settings."""
    values = np.maximum(np.asarray(scenario_values, dtype=float), 0.0)
    n, scenario_count = values.shape
    k = min(max(int(budget), 0), n)
    if k == 0 or n == 0:
        return PlanSolution(np.empty(0, dtype=int), 0.0, True, "empty plan", 0.0)
    if np.allclose(values, values[:, [0]], rtol=1e-10, atol=1e-12):
        selected = np.argsort(values[:, 0])[::-1][:k]
        return PlanSolution(
            selected.astype(int),
            float(values[selected, 0].sum()),
            True,
            "exact sorting for a common score",
            0.0,
        )
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        # Variables are x_1,...,x_n,z.  Minimising -z maximises the common lower value.
        objective = np.zeros(n + 1)
        objective[-1] = -1.0
        rows = []
        lower = []
        upper = []
        budget_row = np.zeros(n + 1)
        budget_row[:n] = 1.0
        rows.append(budget_row)
        lower.append(-np.inf)
        upper.append(float(k))
        for s in range(scenario_count):
            row = np.zeros(n + 1)
            row[:n] = -values[:, s]
            row[-1] = 1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)
        if groups is not None and group_cap is not None:
            group_array = np.asarray(groups)
            for group in pd.unique(group_array):
                row = np.zeros(n + 1)
                row[:n] = (group_array == group).astype(float)
                rows.append(row)
                lower.append(-np.inf)
                upper.append(float(group_cap))
        constraints = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
        lb = np.zeros(n + 1)
        ub = np.ones(n + 1)
        ub[-1] = np.inf
        result = milp(
            c=objective,
            integrality=np.r_[np.ones(n, dtype=int), 0],
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"mip_rel_gap": 1e-6, "time_limit": 30.0},
        )
        if result.success and result.x is not None:
            selected = np.flatnonzero(result.x[:n] > 0.5)
            value = float(np.min(values[selected].sum(axis=0))) if len(selected) else 0.0
            return PlanSolution(
                selected,
                value,
                True,
                "mixed-integer robust epigraph",
                float(getattr(result, "mip_gap", 0.0) or 0.0),
            )
    except Exception:
        pass

    # Deterministic fallback: maximise the pointwise lower score and preserve all limits.
    score = np.min(values, axis=1)
    order = np.argsort(score)[::-1]
    chosen: list[int] = []
    counts: dict[object, int] = {}
    for idx in order:
        if groups is not None and group_cap is not None:
            group = groups[idx]
            if counts.get(group, 0) >= group_cap:
                continue
            counts[group] = counts.get(group, 0) + 1
        chosen.append(int(idx))
        if len(chosen) == k:
            break
    selected = np.asarray(chosen, dtype=int)
    value = float(np.min(values[selected].sum(axis=0))) if len(selected) else 0.0
    return PlanSolution(selected, value, True, "pointwise-lower fallback", np.nan)


def anchored_robust_plan_select(
    scenario_values: np.ndarray,
    baseline: np.ndarray,
    budget: int,
    *,
    groups: np.ndarray | None = None,
    group_cap: int | None = None,
    discrepancy_penalty: float = 0.0,
) -> AnchoredPlanSolution:
    """Maximise certified improvement over a feasible baseline plan."""
    values = np.maximum(np.asarray(scenario_values, dtype=float), 0.0)
    n, scenario_count = values.shape
    baseline = np.unique(np.asarray(baseline, dtype=int))
    if np.any((baseline < 0) | (baseline >= n)):
        raise ValueError("Baseline indices must refer to rows of scenario_values.")
    k = min(max(int(budget), 0), n)
    penalty = max(float(discrepancy_penalty), 0.0)
    if len(baseline) > k:
        raise ValueError("The baseline plan exceeds the action budget.")
    if groups is not None and group_cap is not None:
        group_array = np.asarray(groups)
        baseline_counts = pd.Series(group_array[baseline]).value_counts()
        if len(baseline_counts) and int(baseline_counts.max()) > int(group_cap):
            raise ValueError("The baseline plan exceeds a group limit.")
    if k == 0 or n == 0:
        return AnchoredPlanSolution(
            np.empty(0, dtype=int), baseline, 0.0, 0, True, "empty plan", 0.0
        )

    baseline_totals = values[baseline].sum(axis=0) if len(baseline) else np.zeros(scenario_count)
    if np.allclose(values, values[:, [0]], rtol=1e-10, atol=1e-12):
        adjusted_score = values[:, 0] - penalty
        adjusted_score[baseline] = values[baseline, 0] + penalty
        selected = score_plan_select(adjusted_score, k, groups=groups, group_cap=group_cap)
        improvement = float(
            values[selected, 0].sum()
            - baseline_totals[0]
            - penalty * symmetric_difference_count(selected, baseline)
        )
        return AnchoredPlanSolution(
            selected,
            baseline,
            improvement,
            symmetric_difference_count(selected, baseline),
            True,
            "exact sorting for a common response",
            0.0,
        )

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        # Variables are x_1,...,x_n,z. Each scenario constrains z by the
        # improvement of x over the fixed baseline plan.
        objective = np.zeros(n + 1)
        objective[-1] = -1.0
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        budget_row = np.zeros(n + 1)
        budget_row[:n] = 1.0
        rows.append(budget_row)
        lower.append(-np.inf)
        upper.append(float(k))
        for scenario in range(scenario_count):
            row = np.zeros(n + 1)
            row[:n] = -values[:, scenario]
            change_coefficients = np.ones(n)
            change_coefficients[baseline] = -1.0
            row[:n] += penalty * change_coefficients
            row[-1] = 1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(-baseline_totals[scenario] - penalty * len(baseline)))
        if groups is not None and group_cap is not None:
            group_array = np.asarray(groups)
            for group in pd.unique(group_array):
                row = np.zeros(n + 1)
                row[:n] = (group_array == group).astype(float)
                rows.append(row)
                lower.append(-np.inf)
                upper.append(float(group_cap))
        constraints = LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper))
        lb = np.r_[np.zeros(n), -np.inf]
        ub = np.r_[np.ones(n), np.inf]
        result = milp(
            c=objective,
            integrality=np.r_[np.ones(n, dtype=int), 0],
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"mip_rel_gap": 1e-6, "time_limit": 30.0},
        )
        if result.success and result.x is not None:
            selected = np.flatnonzero(result.x[:n] > 0.5)
            improvement = (
                paired_plan_values(values, selected, baseline)["worst"]
                - penalty * symmetric_difference_count(selected, baseline)
            )
            return AnchoredPlanSolution(
                selected,
                baseline,
                improvement,
                symmetric_difference_count(selected, baseline),
                True,
                "mixed-integer anchored robust epigraph",
                float(getattr(result, "mip_gap", 0.0) or 0.0),
            )
    except Exception:
        pass

    # The baseline is always a valid zero-improvement fallback.
    return AnchoredPlanSolution(
        baseline.copy(), baseline, 0.0, 0, True, "feasible baseline fallback", np.nan
    )


def anchored_change_frontier(
    scenario_values: np.ndarray,
    baseline: np.ndarray,
    budget: int,
    *,
    groups: np.ndarray | None = None,
    group_cap: int | None = None,
) -> list[FrontierPoint]:
    """Compute the exact protected-improvement frontier by change count.

    For each feasible change count ``c`` in ``0,...,2*budget``, this routine
    solves the paired maximin problem with the additional equality ``C(x,b)=c``.
    The regularised certificate for any discrepancy penalty is the upper envelope
    of the returned affine values ``protected_improvement - penalty*c``.
    """
    values = np.maximum(np.asarray(scenario_values, dtype=float), 0.0)
    n, scenario_count = values.shape
    baseline = np.unique(np.asarray(baseline, dtype=int))
    k = min(max(int(budget), 0), n)
    if np.any((baseline < 0) | (baseline >= n)) or len(baseline) > k:
        raise ValueError("The baseline must be a feasible index set within the action budget.")
    if groups is not None and group_cap is not None:
        group_array = np.asarray(groups)
        if len(baseline) and int(pd.Series(group_array[baseline]).value_counts().max()) > int(group_cap):
            raise ValueError("The baseline plan exceeds a group limit.")

    baseline_totals = values[baseline].sum(axis=0) if len(baseline) else np.zeros(scenario_count)
    points: list[FrontierPoint] = []
    for change_count in range(0, 2 * k + 1):
        selected, improvement, feasible = _solve_exact_change_count(
            values,
            baseline,
            k,
            change_count,
            baseline_totals,
            groups=groups,
            group_cap=group_cap,
        )
        points.append(FrontierPoint(change_count, improvement, selected, feasible))
    return points


def _solve_exact_change_count(
    values: np.ndarray,
    baseline: np.ndarray,
    budget: int,
    change_count: int,
    baseline_totals: np.ndarray,
    *,
    groups: np.ndarray | None,
    group_cap: int | None,
) -> tuple[np.ndarray, float, bool]:
    """Solve one exact-change-count paired maximin problem."""
    n, scenario_count = values.shape
    if change_count < 0 or change_count > 2 * budget:
        return np.empty(0, dtype=int), float("-inf"), False
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp

        objective = np.zeros(n + 1)
        objective[-1] = -1.0
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        budget_row = np.zeros(n + 1)
        budget_row[:n] = 1.0
        rows.append(budget_row)
        lower.append(-np.inf)
        upper.append(float(budget))
        # C(x,b) = |b| + sum_{b_a=0}x_a - sum_{b_a=1}x_a.
        change_row = np.zeros(n + 1)
        change_row[:n] = 1.0
        change_row[baseline] = -1.0
        rows.append(change_row)
        lower.append(float(change_count - len(baseline)))
        upper.append(float(change_count - len(baseline)))
        for scenario in range(scenario_count):
            row = np.zeros(n + 1)
            row[:n] = -values[:, scenario]
            row[-1] = 1.0
            rows.append(row)
            lower.append(-np.inf)
            upper.append(float(-baseline_totals[scenario]))
        if groups is not None and group_cap is not None:
            group_array = np.asarray(groups)
            for group in pd.unique(group_array):
                row = np.zeros(n + 1)
                row[:n] = (group_array == group).astype(float)
                rows.append(row)
                lower.append(-np.inf)
                upper.append(float(group_cap))
        result = milp(
            c=objective,
            integrality=np.r_[np.ones(n, dtype=int), 0],
            bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
            constraints=LinearConstraint(np.vstack(rows), np.asarray(lower), np.asarray(upper)),
            options={"mip_rel_gap": 1e-9, "time_limit": 30.0},
        )
        if result.success and result.x is not None:
            selected = np.flatnonzero(result.x[:n] > 0.5).astype(int)
            totals = values[selected].sum(axis=0) - baseline_totals if len(selected) else -baseline_totals
            return selected, float(np.min(totals)), True
    except Exception:
        pass
    return np.empty(0, dtype=int), float("-inf"), False


def score_plan_select(
    scores: np.ndarray,
    budget: int,
    *,
    groups: np.ndarray | None = None,
    group_cap: int | None = None,
) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    chosen: list[int] = []
    counts: dict[object, int] = {}
    for idx in order:
        if groups is not None and group_cap is not None:
            group = groups[idx]
            if counts.get(group, 0) >= group_cap:
                continue
            counts[group] = counts.get(group, 0) + 1
        chosen.append(int(idx))
        if len(chosen) >= budget:
            break
    return np.asarray(chosen, dtype=int)


def plan_values(scenario_values: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    if len(selected) == 0:
        return {"worst": 0.0, "mean": 0.0, "spread": 0.0}
    totals = np.asarray(scenario_values, dtype=float)[selected].sum(axis=0)
    return {
        "worst": float(np.min(totals)),
        "mean": float(np.mean(totals)),
        "spread": float(np.max(totals) - np.min(totals)),
    }


def paired_plan_values(
    scenario_values: np.ndarray,
    selected: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    """Return scenario-wise value differences between two plans."""
    values = np.asarray(scenario_values, dtype=float)
    selected_totals = values[np.asarray(selected, dtype=int)].sum(axis=0) if len(selected) else np.zeros(values.shape[1])
    baseline_totals = values[np.asarray(baseline, dtype=int)].sum(axis=0) if len(baseline) else np.zeros(values.shape[1])
    differences = selected_totals - baseline_totals
    return {
        "worst": float(np.min(differences)),
        "mean": float(np.mean(differences)),
        "spread": float(np.max(differences) - np.min(differences)),
    }


def symmetric_difference_count(first: np.ndarray, second: np.ndarray) -> int:
    """Count actions that appear in exactly one of two plans."""
    return int(len(np.setxor1d(np.asarray(first, dtype=int), np.asarray(second, dtype=int))))


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(dtype=float)
