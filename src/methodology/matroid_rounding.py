"""Linear relaxation and randomized pipage rounding for grouped action limits."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class FractionalRobustPlan:
    marginals: np.ndarray
    value: float
    runtime_seconds: float


def solve_grouped_robust_lp(
    scenario_values: np.ndarray,
    groups: np.ndarray,
    total_cap: int,
    group_cap: int,
) -> FractionalRobustPlan:
    """Solve the robust linear relaxation under total and disjoint-group limits."""
    values = np.maximum(np.asarray(scenario_values, dtype=float), 0.0)
    labels = np.asarray(groups)
    n, scenario_count = values.shape
    unique_groups, inverse = np.unique(labels.astype(str), return_inverse=True)
    objective = np.r_[np.zeros(n), -1.0]
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    total_row = np.zeros(n + 1)
    total_row[:n] = 1.0
    rows.append(total_row)
    bounds.append(float(total_cap))
    for group_index in range(len(unique_groups)):
        row = np.zeros(n + 1)
        row[np.where(inverse == group_index)[0]] = 1.0
        rows.append(row)
        bounds.append(float(group_cap))
    for theta in range(scenario_count):
        rows.append(np.r_[-values[:, theta], 1.0])
        bounds.append(0.0)
    start = perf_counter()
    result = linprog(
        objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(bounds),
        bounds=[(0.0, 1.0)] * n + [(0.0, None)],
        method="highs",
    )
    elapsed = perf_counter() - start
    if not result.success:
        raise RuntimeError(f"Robust linear programme failed: {result.message}")
    return FractionalRobustPlan(
        marginals=np.clip(result.x[:n], 0.0, 1.0),
        value=float(result.x[-1]),
        runtime_seconds=float(elapsed),
    )


def randomized_pipage_round(
    marginals: np.ndarray,
    groups: np.ndarray,
    total_cap: int,
    group_cap: int,
    rng: np.random.Generator,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Round one point in a truncated partition-matroid polytope.

    Zero-value dummy actions extend an independent-set point to a base. Pairwise
    endpoint moves then preserve its expectation and all group constraints.
    """
    original = np.clip(np.asarray(marginals, dtype=float), 0.0, 1.0)
    labels = np.asarray(groups).astype(str)
    if len(original) != len(labels):
        raise ValueError("marginals and groups must have the same length")
    if original.sum() > total_cap + 1e-7:
        raise ValueError("fractional point exceeds the total action limit")

    unique, inverse = np.unique(labels, return_inverse=True)
    group_sums = np.bincount(inverse, weights=original, minlength=len(unique))
    if np.any(group_sums > group_cap + 1e-7):
        raise ValueError("fractional point exceeds a group limit")

    remaining = max(float(total_cap) - float(original.sum()), 0.0)
    dummy_count = int(np.ceil(remaining - tolerance))
    if dummy_count:
        dummy = np.ones(dummy_count)
        dummy[-1] = remaining - float(dummy_count - 1)
        x = np.r_[original, dummy]
        dummy_label = "__dummy_actions__"
        labels = np.r_[labels, np.repeat(dummy_label, dummy_count)]
    else:
        x = original.copy()
    unique, inverse = np.unique(labels, return_inverse=True)
    caps = np.full(len(unique), float(group_cap))
    dummy_positions = np.where(unique == "__dummy_actions__")[0]
    if len(dummy_positions):
        caps[dummy_positions[0]] = float(total_cap)

    max_iterations = 4 * max(len(x), 1)
    for _ in range(max_iterations):
        fractional = np.where((x > tolerance) & (x < 1.0 - tolerance))[0]
        if len(fractional) == 0:
            break

        pair = None
        for group_index in np.unique(inverse[fractional]):
            candidates = fractional[inverse[fractional] == group_index]
            if len(candidates) >= 2:
                pair = (int(candidates[0]), int(candidates[1]))
                break
        if pair is None:
            if len(fractional) < 2:
                raise RuntimeError("A single fractional coordinate remained")
            pair = (int(fractional[0]), int(fractional[1]))
        i, j = pair
        group_i, group_j = int(inverse[i]), int(inverse[j])
        sums = np.bincount(inverse, weights=x, minlength=len(unique))
        plus = min(1.0 - x[i], x[j])
        minus = min(x[i], 1.0 - x[j])
        if group_i != group_j:
            plus = min(plus, caps[group_i] - sums[group_i])
            minus = min(minus, caps[group_j] - sums[group_j])
        plus = max(float(plus), 0.0)
        minus = max(float(minus), 0.0)
        if plus <= tolerance or minus <= tolerance:
            found = False
            for left_index in range(len(fractional)):
                for right_index in range(left_index + 1, len(fractional)):
                    ii, jj = int(fractional[left_index]), int(fractional[right_index])
                    gi, gj = int(inverse[ii]), int(inverse[jj])
                    ap = min(1.0 - x[ii], x[jj])
                    am = min(x[ii], 1.0 - x[jj])
                    if gi != gj:
                        ap = min(ap, caps[gi] - sums[gi])
                        am = min(am, caps[gj] - sums[gj])
                    if ap > tolerance and am > tolerance:
                        i, j, group_i, group_j = ii, jj, gi, gj
                        plus, minus = float(ap), float(am)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise RuntimeError("No feasible pipage direction was found")
        probability_plus = minus / (plus + minus)
        if rng.random() < probability_plus:
            x[i] += plus
            x[j] -= plus
        else:
            x[i] -= minus
            x[j] += minus
        x[np.abs(x) <= tolerance] = 0.0
        x[np.abs(x - 1.0) <= tolerance] = 1.0
    else:
        raise RuntimeError("Pipage rounding did not terminate")

    rounded = np.flatnonzero(x[: len(original)] > 0.5)
    if len(rounded) > total_cap:
        raise RuntimeError("Rounded plan exceeds the total action limit")
    rounded_labels = labels[: len(original)][rounded]
    _, counts = np.unique(rounded_labels, return_counts=True)
    if len(counts) and int(counts.max()) > group_cap:
        raise RuntimeError("Rounded plan exceeds a group limit")
    return rounded
