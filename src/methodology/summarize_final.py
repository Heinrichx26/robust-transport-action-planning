from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("bts", "road", "divvy")
FRONTIER = (
    "TemporalContext2024",
    "NetworkPropagation2024",
    "CalibratedUncertainty2025",
    "MappedPTO2026",
    "GroupAware2025",
    "RobustImprovement2025",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise completed structural robust experiments.")
    parser.add_argument("--root", type=Path, default=Path("results/structural_robust_final"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    out = root / "combined"
    out.mkdir(parents=True, exist_ok=True)
    plans = pd.concat(
        [pd.read_csv(root / dataset / "tables" / "plan_results.csv") for dataset in DATASETS],
        ignore_index=True,
    )
    coverage = pd.concat(
        [pd.read_csv(root / dataset / "tables" / "plan_coverage.csv") for dataset in DATASETS],
        ignore_index=True,
    )
    test = plans[plans["partition"] == "test"].copy()
    tests = paired_tests(test)
    tests.to_csv(out / "paired_method_tests.csv", index=False)
    main_table = build_main_table(test, tests)
    main_table.to_csv(out / "main_results.csv", index=False)
    coverage_summary = (
        coverage.groupby(["dataset", "budget_fraction"], as_index=False)
        .agg(
            coverage=("covered", "mean"),
            blocks=("covered", "size"),
            mean_lower_bound=("calibrated_plan_lower_bound", "mean"),
            mean_realized_worst=("realized_worst_value", "mean"),
        )
    )
    coverage_summary["retained_share"] = coverage_summary["mean_lower_bound"] / np.maximum(
        coverage_summary["mean_realized_worst"], 1e-12
    )
    coverage_summary.to_csv(out / "coverage_summary.csv", index=False)
    print(main_table.to_string(index=False))
    print(coverage_summary.to_string(index=False))


def paired_tests(test: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260724)
    rows = []
    pivot = test.pivot_table(
        index=["dataset", "fold_id", "block_id", "budget_fraction"],
        columns="method",
        values="realized_worst_value",
        aggfunc="sum",
    )
    for (dataset, budget), part in pivot.groupby(level=["dataset", "budget_fraction"]):
        if "StructuralRobust" not in part:
            continue
        for comparator in [column for column in part.columns if column != "StructuralRobust"]:
            pair = part[["StructuralRobust", comparator]].dropna()
            diff = pair["StructuralRobust"].to_numpy() - pair[comparator].to_numpy()
            if len(diff) == 0:
                continue
            signs = rng.choice(np.array([-1.0, 1.0]), size=(5000, len(diff)), replace=True)
            null = np.mean(signs * diff, axis=1)
            observed = float(np.mean(diff))
            p = float((1 + np.sum(null >= observed)) / 5001) if observed >= 0 else 1.0
            boot = np.mean(rng.choice(diff, size=(5000, len(diff)), replace=True), axis=1)
            comparator_mean = float(pair[comparator].mean())
            rows.append(
                {
                    "dataset": dataset,
                    "budget_fraction": budget,
                    "comparator": comparator,
                    "blocks": len(diff),
                    "mean_difference": observed,
                    "gain_pct": 100.0 * observed / max(comparator_mean, 1e-12),
                    "ci_lower": float(np.quantile(boot, 0.025)),
                    "ci_upper": float(np.quantile(boot, 0.975)),
                    "one_sided_randomization_p": p,
                }
            )
    result = pd.DataFrame(rows)
    risk_mask = result["comparator"] == "RiskFirst"
    result.loc[risk_mask, "holm_p"] = holm(result.loc[risk_mask, "one_sided_randomization_p"].to_numpy())
    return result


def build_main_table(test: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    mean_values = (
        test.groupby(["dataset", "budget_fraction", "method"], as_index=False)["realized_worst_value"]
        .mean()
        .pivot(index=["dataset", "budget_fraction"], columns="method", values="realized_worst_value")
        .reset_index()
    )
    rows = []
    for _, row in mean_values.iterrows():
        dataset = row["dataset"]
        budget = row["budget_fraction"]
        ours = float(row["StructuralRobust"])
        risk = float(row["RiskFirst"])
        frontier_values = {method: float(row[method]) for method in FRONTIER if method in row}
        strongest_name = max(frontier_values, key=frontier_values.get)
        strongest_value = frontier_values[strongest_name]
        risk_test = tests[
            (tests["dataset"] == dataset)
            & (tests["budget_fraction"] == budget)
            & (tests["comparator"] == "RiskFirst")
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "budget_fraction": budget,
                "ours_worst_value": ours,
                "gain_over_loss_priority_pct": 100.0 * (ours - risk) / max(risk, 1e-12),
                "risk_holm_p": risk_test.get("holm_p", np.nan),
                "strongest_frontier_method": strongest_name,
                "gain_over_strongest_frontier_pct": 100.0 * (ours - strongest_value) / max(strongest_value, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def holm(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


if __name__ == "__main__":
    main()
