"""Smoke test for robust linear relaxation and grouped pipage rounding."""

import numpy as np

from methodology.matroid_rounding import (
    randomized_pipage_round,
    solve_grouped_robust_lp,
)
from methodology.structural_robust import plan_values, robust_plan_select


def main() -> None:
    response = np.array(
        [[8.0, 2.0], [3.0, 7.0], [7.0, 3.0], [2.0, 8.0], [6.0, 4.0], [4.0, 6.0]]
    )
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    total_cap = 3
    group_cap = 1
    fractional = solve_grouped_robust_lp(response, groups, total_cap, group_cap)
    exact = robust_plan_select(response, total_cap, groups=groups, group_cap=group_cap)
    rng = np.random.default_rng(20260728)
    draws = [
        randomized_pipage_round(
            fractional.marginals, groups, total_cap, group_cap, rng
        )
        for _ in range(20_000)
    ]
    empirical = np.zeros(len(response))
    values = []
    for selected in draws:
        empirical[selected] += 1.0
        values.append(plan_values(response, selected)["worst"])
        assert len(selected) <= total_cap
        _, counts = np.unique(groups[selected], return_counts=True)
        assert len(counts) == 0 or counts.max() <= group_cap
    empirical /= len(draws)
    marginal_error = float(np.max(np.abs(empirical - fractional.marginals)))
    best = float(np.max(values))
    assert marginal_error < 0.02
    assert best >= exact.predicted_worst_value - 1e-9
    print(
        {
            "lp_value": round(fractional.value, 6),
            "exact_value": round(exact.predicted_worst_value, 6),
            "best_rounded_value": round(best, 6),
            "mean_rounded_value": round(float(np.mean(values)), 6),
            "max_marginal_error": round(marginal_error, 6),
            "feasibility_violations": 0,
        }
    )


if __name__ == "__main__":
    main()
