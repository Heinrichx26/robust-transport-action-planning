from __future__ import annotations

from itertools import combinations

import numpy as np

from methodology.structural_robust import (
    anchored_change_frontier,
    anchored_robust_plan_select,
    paired_plan_values,
    symmetric_difference_count,
)


def brute_frontier(values: np.ndarray, baseline: np.ndarray, budget: int) -> dict[int, float]:
    best: dict[int, float] = {}
    for size in range(budget + 1):
        for selected_tuple in combinations(range(len(values)), size):
            selected = np.asarray(selected_tuple, dtype=int)
            changed = symmetric_difference_count(selected, baseline)
            improvement = paired_plan_values(values, selected, baseline)["worst"]
            best[changed] = max(best.get(changed, float("-inf")), improvement)
    return best


def main() -> None:
    rng = np.random.default_rng(20260728)
    values = rng.uniform(0.1, 5.0, size=(8, 3))
    baseline = np.asarray([0, 2, 5], dtype=int)
    budget = 3
    exact = anchored_change_frontier(values, baseline, budget)
    brute = brute_frontier(values, baseline, budget)
    feasible = {point.changed_actions: point for point in exact if point.feasible}
    if set(feasible) != set(brute):
        raise AssertionError(f"Feasible change counts differ: {set(feasible)} != {set(brute)}")
    for changed, expected in brute.items():
        observed = feasible[changed].protected_improvement
        if not np.isclose(observed, expected, atol=1e-7):
            raise AssertionError(f"Frontier mismatch at C={changed}: {observed} != {expected}")

    previous_changes = 2 * budget + 1
    for penalty in np.linspace(0.0, 3.0, 13):
        direct = anchored_robust_plan_select(
            values,
            baseline,
            budget,
            discrepancy_penalty=float(penalty),
        )
        best_point = max(
            feasible.values(),
            key=lambda point: point.protected_improvement - penalty * point.changed_actions,
        )
        envelope_value = best_point.protected_improvement - penalty * best_point.changed_actions
        if not np.isclose(direct.predicted_worst_improvement, envelope_value, atol=1e-7):
            raise AssertionError(
                f"Regularised value mismatch at penalty={penalty}: "
                f"{direct.predicted_worst_improvement} != {envelope_value}"
            )
        if direct.changed_actions > previous_changes:
            raise AssertionError("The direct regularisation path is not monotone.")
        previous_changes = direct.changed_actions
    print("change-frontier smoke test passed")


if __name__ == "__main__":
    main()
