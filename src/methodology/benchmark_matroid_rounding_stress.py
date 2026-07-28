"""Controlled grouped-action instances with a genuinely fractional robust optimum."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from methodology.matroid_rounding import randomized_pipage_round, solve_grouped_robust_lp
from methodology.structural_robust import plan_values, robust_plan_select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def complementary_instance(module_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Create one complementary response pair and one common action per module."""
    values = np.zeros((3 * module_count, 2 * module_count), dtype=float)
    groups: list[str] = []
    for module in range(module_count):
        action = 3 * module
        setting = 2 * module
        values[action, setting] = 10.0
        values[action + 1, setting + 1] = 10.0
        values[action + 2, setting : setting + 2] = 4.0
        groups.extend([f"choice_{module}", f"choice_{module}", f"common_{module}"])
    return values, np.asarray(groups)


def benchmark(module_count: int, repeats: int, seed: int, verify_exact: bool) -> dict[str, float | int]:
    values, groups = complementary_instance(module_count)
    total_cap = 2 * module_count
    fractional = solve_grouped_robust_lp(values, groups, total_cap, 1)
    fractional_coordinates = int(np.sum((fractional.marginals > 1e-8) & (fractional.marginals < 1 - 1e-8)))

    exact_value = 4.0
    exact_seconds = np.nan
    if verify_exact:
        start = perf_counter()
        exact = robust_plan_select(values, total_cap, groups=groups, group_cap=1)
        exact_seconds = perf_counter() - start
        if not exact.feasible or abs(exact.predicted_worst_value - exact_value) > 1e-7:
            raise RuntimeError("The controlled instance did not attain its analytic integer optimum.")

    rng = np.random.default_rng(seed)
    retained: list[float] = []
    selections = np.zeros(len(values), dtype=float)
    violations = 0
    start = perf_counter()
    for _ in range(repeats):
        selected = randomized_pipage_round(fractional.marginals, groups, total_cap, 1, rng)
        retained.append(plan_values(values, selected)["worst"])
        selections[selected] += 1.0
        _, counts = np.unique(groups[selected], return_counts=True)
        violations += int(len(selected) > total_cap or (len(counts) and counts.max() > 1))
    elapsed = perf_counter() - start
    retained_array = np.asarray(retained)
    marginal_error = np.max(np.abs(selections / repeats - fractional.marginals))
    return {
        "modules": module_count,
        "candidate_count": len(values),
        "setting_count": values.shape[1],
        "action_limit": total_cap,
        "fractional_coordinates": fractional_coordinates,
        "lp_value": fractional.value,
        "exact_value": exact_value,
        "lp_gap_percent": 100.0 * (fractional.value - exact_value) / exact_value,
        "mean_rounded_value": float(retained_array.mean()),
        "minimum_rounded_value": float(retained_array.min()),
        "mean_retained_percent": 100.0 * float(retained_array.mean()) / exact_value,
        "max_marginal_error": float(marginal_error),
        "feasibility_violations": violations,
        "lp_seconds": fractional.runtime_seconds,
        "exact_verification_seconds": exact_seconds,
        "rounding_seconds_per_draw": elapsed / repeats,
        "rounding_repeats": repeats,
    }


def main() -> None:
    args = parse_args()
    sizes = [2] if args.smoke else [10, 50, 200]
    repeats = min(args.repeats, 50) if args.smoke else args.repeats
    rows = [
        benchmark(size, repeats, args.seed + index, verify_exact=args.smoke)
        for index, size in enumerate(sizes)
    ]
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
