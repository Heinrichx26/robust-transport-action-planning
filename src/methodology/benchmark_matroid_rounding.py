"""Benchmark the grouped robust relaxation and pipage rounding on public records."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from methodology.matroid_rounding import (
    randomized_pipage_round,
    solve_grouped_robust_lp,
)
from methodology.structural_robust import (
    plan_values,
    robust_plan_select,
    scenario_gain_matrix,
)


DATASETS = {
    "bts": Path("data/open/bts/ccerts_bts_ready_v2.csv"),
    "road": Path("data/open/road/ccerts_road_ready_v3.csv"),
    "divvy": Path("data/open/divvy/ccerts_divvy_official_capacity.csv"),
}

USE_COLUMNS = {
    "bts": [
        "block_id", "group", "loss", "gain_lower", "schedule_slack",
        "actionable_delay", "recovery_reliability",
    ],
    "road": [
        "block_id", "group", "loss", "gain_lower", "mean_speed",
        "free_flow_speed", "network_headroom",
    ],
    "divvy": [
        "block_id", "group", "loss", "gain_lower", "routine_loss",
        "capacity_proxy", "inventory_start", "departures", "arrivals",
        "relocation_limit",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def latest_block(dataset: str, path: Path) -> str:
    latest = ""
    for chunk in pd.read_csv(path, usecols=["block_id"], chunksize=250_000):
        candidate = str(chunk["block_id"].astype(str).max())
        latest = max(latest, candidate)
    return latest


def read_block(dataset: str, path: Path, block_id: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=USE_COLUMNS[dataset], chunksize=250_000):
        selected = chunk.loc[chunk["block_id"].astype(str) == block_id]
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError(f"No records found for {dataset} block {block_id}")
    return pd.concat(parts, ignore_index=True)


def benchmark_instance(
    dataset: str,
    block_id: str,
    frame: pd.DataFrame,
    candidate_count: int,
    repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    _, all_values = scenario_gain_matrix(dataset, frame)
    score = np.mean(all_values, axis=1)
    order = np.argsort(score)[::-1]
    positive = order[score[order] > 0]
    chosen = positive[: min(candidate_count, len(positive))]
    values = all_values[chosen]
    groups = frame.iloc[chosen]["group"].fillna("unknown").astype(str).to_numpy()
    n = len(values)
    total_cap = max(1, int(np.ceil(0.10 * n)))
    group_cap = max(1, int(np.ceil(0.40 * total_cap)))

    fractional = solve_grouped_robust_lp(values, groups, total_cap, group_cap)
    exact_start = perf_counter()
    exact = robust_plan_select(values, total_cap, groups=groups, group_cap=group_cap)
    exact_seconds = perf_counter() - exact_start
    if not exact.feasible or not np.isfinite(exact.predicted_worst_value):
        raise RuntimeError("Exact robust plan was unavailable")

    rng = np.random.default_rng(seed)
    rounded_values = []
    violations = 0
    rounding_start = perf_counter()
    for _ in range(repeats):
        selected = randomized_pipage_round(
            fractional.marginals, groups, total_cap, group_cap, rng
        )
        rounded_values.append(plan_values(values, selected)["worst"])
        if len(selected) > total_cap:
            violations += 1
        _, counts = np.unique(groups[selected], return_counts=True)
        if len(counts) and int(counts.max()) > group_cap:
            violations += 1
    rounding_seconds = perf_counter() - rounding_start
    rounded = np.asarray(rounded_values, dtype=float)
    exact_value = exact.predicted_worst_value
    upper = float(np.max(values))
    setting_count = values.shape[1]
    delta = 0.05
    additive_term = upper * np.sqrt(2.0 * total_cap * np.log(setting_count / delta))
    relative_factor = max(
        0.0,
        1.0 - np.sqrt(
            2.0 * upper * np.log(setting_count / delta) / max(fractional.value, 1e-12)
        ),
    )
    return {
        "dataset": dataset,
        "block_id": block_id,
        "candidate_count": n,
        "action_limit": total_cap,
        "group_limit": group_cap,
        "setting_count": setting_count,
        "lp_value": fractional.value,
        "exact_value": exact_value,
        "lp_gap_percent": 100.0 * (fractional.value - exact_value) / max(exact_value, 1e-12),
        "best_rounded_value": float(np.max(rounded)),
        "mean_rounded_value": float(np.mean(rounded)),
        "p05_rounded_value": float(np.quantile(rounded, 0.05)),
        "best_gap_percent": 100.0 * (np.max(rounded) - exact_value) / max(exact_value, 1e-12),
        "mean_retained_percent": 100.0 * np.mean(rounded) / max(exact_value, 1e-12),
        "theory_additive_lower": max(0.0, fractional.value - additive_term),
        "theory_relative_factor": relative_factor,
        "lp_seconds": fractional.runtime_seconds,
        "exact_seconds": exact_seconds,
        "rounding_seconds_per_draw": rounding_seconds / repeats,
        "rounding_repeats": repeats,
        "feasibility_violations": violations,
    }


def main() -> None:
    args = parse_args()
    sizes = [100] if args.smoke else [250, 1000, 4000]
    repeats = min(args.repeats, 50) if args.smoke else args.repeats
    rows = []
    for dataset, path in DATASETS.items():
        block_id = latest_block(dataset, path)
        block = read_block(dataset, path, block_id)
        observed_sizes: set[int] = set()
        for size in sizes:
            row = benchmark_instance(
                dataset, block_id, block, size, repeats,
                args.seed + len(rows),
            )
            actual_size = int(row["candidate_count"])
            if actual_size in observed_sizes:
                continue
            observed_sizes.add(actual_size)
            rows.append(row)
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
