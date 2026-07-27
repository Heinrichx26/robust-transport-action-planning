from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute sequential complete-plan lower bounds.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target-alpha", type=float, default=0.05)
    parser.add_argument("--adaptation-rate", type=float, default=0.10)
    return parser.parse_args()


def finite_quantile(values: list[float], level: float) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return 0.0
    rank = int(np.ceil((len(array) + 1) * level))
    if rank > len(array):
        return np.inf
    if rank <= 0:
        return -np.inf
    return float(np.partition(array, rank - 1)[rank - 1])


def recompute(dataset_dir: Path, target_alpha: float, adaptation_rate: float) -> None:
    tables = dataset_dir / "tables"
    results = pd.read_csv(tables / "plan_results.csv")
    proposed = results.loc[results["method"].eq("StructuralRobust")].copy()
    rows: list[dict[str, float | str]] = []
    for (fold_id, budget), frame in proposed.groupby(["fold_id", "budget_fraction"]):
        calibration = frame.loc[frame["partition"].eq("set_cal")].sort_values("block_id")
        deployment = frame.loc[frame["partition"].eq("test")].sort_values("block_id")
        residuals = list(
            (
                calibration["predicted_worst_value"].to_numpy(dtype=float)
                - calibration["realized_worst_value"].to_numpy(dtype=float)
            )
            / np.maximum(np.sqrt(calibration["selected_n"].to_numpy(dtype=float)), 1.0)
        )
        adaptive_alpha = float(target_alpha)
        for _, row in deployment.iterrows():
            q = finite_quantile(residuals, 1.0 - adaptive_alpha)
            scale = max(np.sqrt(float(row["selected_n"])), 1.0)
            lower = 0.0 if np.isposinf(q) else max(
                0.0,
                float(row["predicted_worst_value"]) - q * scale,
            )
            covered = float(float(row["realized_worst_value"]) >= lower - 1e-9)
            rows.append(
                {
                    "dataset": row["dataset"],
                    "fold_id": fold_id,
                    "budget_fraction": budget,
                    "block_id": row["block_id"],
                    "calibrated_plan_lower_bound": lower,
                    "realized_worst_value": row["realized_worst_value"],
                    "covered": covered,
                    "calibration_blocks": float(len(residuals)),
                    "adaptive_alpha": adaptive_alpha,
                    "target_alpha": target_alpha,
                    "adaptation_rate": adaptation_rate,
                }
            )
            miss = 1.0 - covered
            adaptive_alpha += adaptation_rate * (target_alpha - miss)
            residuals.append(
                (float(row["predicted_worst_value"]) - float(row["realized_worst_value"]))
                / scale
            )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(tables / "plan_coverage.csv", index=False)
    summary = coverage.groupby(["dataset", "budget_fraction"], as_index=False).agg(
        coverage=("covered", "mean"),
        blocks=("covered", "size"),
    )
    summary.to_csv(tables / "plan_coverage_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    for dataset in ("bts", "road", "divvy"):
        recompute(args.root / dataset, args.target_alpha, args.adaptation_rate)


if __name__ == "__main__":
    main()
