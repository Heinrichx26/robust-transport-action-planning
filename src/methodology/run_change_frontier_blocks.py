"""Audit exact plan-change frontiers on held-out transportation decision blocks.

The script reuses the rolling-origin data preparation and prediction routines,
then solves the exact change-count problems on a bounded audit set from each
held-out block.  It is deliberately separate from plotting and from the full
rolling experiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccerts.pipeline import prepare_features, valid_periods
from ccerts.run_rolling_origin import build_folds, ensure_block_id, period_slice
from ccerts.transparent_methods import fit_method_family
from methodology.structural_robust import (
    anchored_change_frontier,
    contamination_robust_values,
    fit_low_rank_scenario_model,
    scenario_gain_matrix,
    score_plan_select,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["bts", "road", "divvy"], required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/change_frontier_blocks"))
    p.add_argument("--max-rows-per-period", type=int, default=6000)
    p.add_argument("--audit-units", type=int, default=60)
    p.add_argument("--max-blocks", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def area_cap(budget: int) -> int:
    return max(1, int(np.ceil(0.40 * max(int(budget), 1))))


def main() -> None:
    args = parse_args()
    max_rows = min(args.max_rows_per_period, 1200) if args.smoke else args.max_rows_per_period
    audit_units = min(args.audit_units, 24) if args.smoke else args.audit_units
    max_blocks = 1 if args.smoke else args.max_blocks
    frame = pd.read_csv(args.input, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["period", "loss", "gain_lower", "group"])
    if args.dataset == "road":
        hour = pd.to_numeric(frame["hour"], errors="coerce")
        weekday = pd.to_numeric(frame["is_weekend"], errors="coerce").fillna(1).eq(0)
        frame = frame.loc[weekday & hour.between(6, 9, inclusive="both")].copy()
    frame["period"] = frame["period"].astype(str)
    periods = valid_periods(frame, 100 if args.smoke else 1000)
    folds = build_folds(periods, 2, 1, 1, 1)
    if not folds:
        raise ValueError("At least five ordered periods are required.")
    fold = folds[0]
    train = ensure_block_id(args.dataset, period_slice(frame, fold["train"], max_rows, 701))
    policy = ensure_block_id(args.dataset, period_slice(frame, fold["policy_val"], max_rows, 702))
    set_cal = ensure_block_id(args.dataset, period_slice(frame, fold["set_cal"], max_rows, 703))
    test = ensure_block_id(args.dataset, period_slice(frame, fold["test"], max_rows, 704))
    train_h, policy_h, future_h, x_train, x_policy, x_future = prepare_features(
        args.dataset, train, policy, pd.concat([set_cal, test], ignore_index=True)
    )
    _, y_train = scenario_gain_matrix(args.dataset, train_h)
    _, y_policy = scenario_gain_matrix(args.dataset, policy_h)
    _, y_future = scenario_gain_matrix(args.dataset, future_h)
    prediction = fit_low_rank_scenario_model(
        x_train, y_train, x_policy, y_policy, x_future, alpha=0.10
    )
    methods = fit_method_family(
        train_h, policy_h, future_h, x_train, x_policy, x_future, alpha=0.10, seed=20260724
    )
    pred_scores = methods.predictions.reset_index(drop=True)
    test_start = len(set_cal)
    test_frame = future_h.iloc[test_start:].reset_index(drop=True)
    predicted = prediction.deployment[test_start:]
    records: list[dict[str, float | int | str]] = []
    audited_blocks = 0
    for block_id, block in test_frame.groupby("block_id", sort=True):
        if audited_blocks >= max_blocks:
            break
        idx = block.index.to_numpy()
        scores = pred_scores.iloc[test_start + idx]["score_risk_first"].to_numpy(dtype=float)
        response_score = np.mean(np.maximum(predicted[idx], 0.0), axis=1)
        half = max(1, audit_units // 2)
        candidates = np.r_[np.argsort(scores)[::-1][:half], np.argsort(response_score)[::-1][:half]]
        keep = np.unique(candidates)[: min(audit_units, len(block))]
        block = block.iloc[keep].reset_index(drop=True)
        values = np.maximum(predicted[idx][keep], 0.0)
        if len(block) < 4:
            continue
        groups = block["group"].astype(str).to_numpy()
        budget = max(1, int(round(0.10 * len(block))))
        baseline = score_plan_select(
            pred_scores.iloc[test_start + idx].iloc[keep]["score_risk_first"].to_numpy(dtype=float),
            budget,
            groups=groups,
            group_cap=area_cap(budget),
        )
        robust_values = contamination_robust_values(values, 0.20)
        frontier = anchored_change_frontier(
            robust_values, baseline, budget, groups=groups, group_cap=area_cap(budget)
        )
        feasible = [point for point in frontier if point.feasible and np.isfinite(point.protected_improvement)]
        audited_blocks += 1
        for point in feasible:
            records.append(
                {
                    "dataset": args.dataset,
                    "block_id": str(block_id),
                    "audit_units": int(len(block)),
                    "budget": int(budget),
                    "changed_actions": int(point.changed_actions),
                    "protected_improvement": float(point.protected_improvement),
                    "baseline_actions": int(len(baseline)),
                }
            )
    out = args.output / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(out / "exact_change_frontier.csv", index=False)
    if result.empty:
        raise RuntimeError("No feasible audit frontier was produced.")
    print(result.groupby(["dataset", "block_id"], as_index=False).agg(
        points=("changed_actions", "size"), max_change=("changed_actions", "max"),
        max_protected_improvement=("protected_improvement", "max")
    ).to_string(index=False))


if __name__ == "__main__":
    main()
