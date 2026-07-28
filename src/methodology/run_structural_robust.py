from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ccerts.pipeline import prepare_features, sample_frame, valid_periods
from ccerts.run_rolling_origin import build_folds, ensure_block_id, period_slice
from ccerts.transparent_methods import fit_method_family, method_score_columns
from methodology.structural_robust import (
    anchored_robust_plan_select,
    contamination_robust_values,
    fit_low_rank_scenario_model,
    paired_plan_values,
    plan_values,
    robust_plan_select,
    scenario_gain_matrix,
    score_plan_select,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structural robust action-value experiments.")
    parser.add_argument("--dataset", choices=["bts", "road", "divvy"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/structural_robust"))
    parser.add_argument("--max-rows-per-period", type=int, default=30000)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--certificate-alpha", type=float, default=0.05)
    parser.add_argument("--adaptation-rate", type=float, default=0.10)
    parser.add_argument("--contamination", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = min(args.max_rows_per_period, 2500) if args.smoke else args.max_rows_per_period
    max_folds = 1 if args.smoke else args.max_folds
    out = args.output_dir / args.dataset
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["period", "loss", "gain_lower", "group"])
    if args.dataset == "road":
        hour = pd.to_numeric(frame["hour"], errors="coerce")
        weekday = pd.to_numeric(frame["is_weekend"], errors="coerce").fillna(1).eq(0)
        frame = frame.loc[weekday & hour.between(6, 9, inclusive="both")].copy()
    frame["period"] = frame["period"].astype(str)
    periods = valid_periods(frame, 100 if args.smoke else 1000)
    folds = build_folds(periods, 2, 1, 1, 1)
    if max_folds > 0:
        folds = folds[:max_folds]
    if not folds:
        raise ValueError("At least five ordered periods are required.")

    result_rows: list[dict[str, float | str]] = []
    diagnostic_rows: list[dict[str, float | str]] = []
    coverage_rows: list[dict[str, float | str]] = []
    budgets = [0.01, 0.03, 0.05, 0.10, 0.20]

    for fold_id, fold in enumerate(folds, start=1):
        train = ensure_block_id(args.dataset, period_slice(frame, fold["train"], max_rows, args.seed + fold_id))
        validation = ensure_block_id(
            args.dataset, period_slice(frame, fold["policy_val"], max_rows, args.seed + 100 + fold_id)
        )
        mechanism_tune, representation_val = _split_validation_blocks(validation)
        set_cal = ensure_block_id(
            args.dataset, period_slice(frame, fold["set_cal"], max_rows, args.seed + 200 + fold_id)
        )
        test = ensure_block_id(args.dataset, period_slice(frame, fold["test"], max_rows, args.seed + 300 + fold_id))
        future = pd.concat([representation_val, set_cal, test], ignore_index=True)

        train_h, mechanism_tune_h, future_h, x_train, x_mechanism_tune, x_future = prepare_features(
            args.dataset, train, mechanism_tune, future
        )
        scenario_names, y_train = scenario_gain_matrix(args.dataset, train_h)
        _, y_mechanism_tune = scenario_gain_matrix(args.dataset, mechanism_tune_h)
        _, y_future = scenario_gain_matrix(args.dataset, future_h)
        prediction = fit_low_rank_scenario_model(
            x_train,
            y_train,
            x_mechanism_tune,
            y_mechanism_tune,
            x_future,
            alpha=args.alpha,
        )
        methods = fit_method_family(
            train_h,
            mechanism_tune_h,
            future_h,
            x_train,
            x_mechanism_tune,
            x_future,
            alpha=args.alpha,
            seed=args.seed + fold_id,
        )
        future_pred = methods.predictions.reset_index(drop=True)
        mechanism_tune_pred = methods.calibration_predictions.reset_index(drop=True)
        representation_n = len(representation_val)
        representation_pred = future_pred.iloc[:representation_n].reset_index(drop=True)
        pred = future_pred.iloc[representation_n:].reset_index(drop=True)
        validation_candidates, deployment_candidates = _scenario_candidates(
            prediction.deployment[:representation_n],
            prediction.deployment[representation_n:],
            representation_pred,
            pred,
            method_score_columns(),
        )
        selected_candidate: dict[float, str] = {}
        for budget in budgets:
            selected_candidate[budget] = _select_scenario_candidate(
                representation_pred,
                validation_candidates,
                y_future[:representation_n],
                budget,
                contamination=args.contamination,
            )
        robust_by_budget = {budget: deployment_candidates[selected_candidate[budget]] for budget in budgets}
        pred["_deployment_row"] = np.arange(len(pred))
        split = len(set_cal)
        set_cal_pred = pred.iloc[:split].copy()
        test_pred = pred.iloc[split:].copy()
        y_deployment = y_future[representation_n:]
        deployment_h = future_h.iloc[representation_n:].reset_index(drop=True)

        correction_rows = _evaluate_partition(
            args.dataset,
            fold_id,
            "set_cal",
            set_cal_pred,
            prediction.deployment[:split],
            {budget: matrix[:split] for budget, matrix in robust_by_budget.items()},
            y_deployment[:split],
            budgets,
            method_score_columns(),
            contamination=args.contamination,
        )
        test_rows = _evaluate_partition(
            args.dataset,
            fold_id,
            "test",
            test_pred,
            prediction.deployment[split:],
            {budget: matrix[split:] for budget, matrix in robust_by_budget.items()},
            y_deployment[split:],
            budgets,
            method_score_columns(),
            contamination=args.contamination,
        )
        result_rows.extend(correction_rows)
        result_rows.extend(test_rows)

        cal_frame = pd.DataFrame(correction_rows)
        test_frame = pd.DataFrame(test_rows)
        for budget in budgets:
            cal_ours = cal_frame[(cal_frame["method"] == "StructuralRobust") & (cal_frame["budget_fraction"] == budget)]
            test_ours = test_frame[(test_frame["method"] == "StructuralRobust") & (test_frame["budget_fraction"] == budget)]
            residual = list((
                cal_ours["predicted_paired_margin"].to_numpy(dtype=float)
                - cal_ours["realized_paired_margin"].to_numpy(dtype=float)
            ) / np.maximum(np.sqrt(cal_ours["changed_actions"].to_numpy(dtype=float)), 1.0))
            absolute_residual = list((
                cal_ours["predicted_worst_value"].to_numpy(dtype=float)
                - cal_ours["realized_worst_value"].to_numpy(dtype=float)
            ) / np.maximum(np.sqrt(cal_ours["selected_n"].to_numpy(dtype=float)), 1.0))
            adaptive_alpha = float(args.certificate_alpha)
            absolute_adaptive_alpha = float(args.certificate_alpha)
            for _, row in test_ours.sort_values("block_id").iterrows():
                q = _finite_quantile(np.asarray(residual), 1.0 - adaptive_alpha)
                scale = max(np.sqrt(row["changed_actions"]), 1.0)
                lower = -np.inf if np.isposinf(q) else float(row["predicted_paired_margin"]) - q * scale
                covered = float(row["realized_paired_margin"] >= lower - 1e-9)
                switched = float(lower > 0.0)
                q_absolute = _finite_quantile(np.asarray(absolute_residual), 1.0 - absolute_adaptive_alpha)
                absolute_scale = max(np.sqrt(row["selected_n"]), 1.0)
                absolute_lower = 0.0 if np.isposinf(q_absolute) else max(
                    0.0, float(row["predicted_worst_value"]) - q_absolute * absolute_scale
                )
                absolute_covered = float(row["realized_worst_value"] >= absolute_lower - 1e-9)
                coverage_rows.append(
                    {
                        "dataset": args.dataset,
                        "fold_id": fold_id,
                        "budget_fraction": budget,
                        "block_id": row["block_id"],
                        "calibrated_paired_margin_lower_bound": lower,
                        "realized_paired_margin": row["realized_paired_margin"],
                        "covered": covered,
                        "absolute_plan_lower_bound": absolute_lower,
                        "realized_worst_value": row["realized_worst_value"],
                        "absolute_covered": absolute_covered,
                        "certified_switch": switched,
                        "calibration_blocks": float(len(residual)),
                        "adaptive_alpha": adaptive_alpha,
                        "absolute_adaptive_alpha": absolute_adaptive_alpha,
                        "target_alpha": float(args.certificate_alpha),
                        "adaptation_rate": float(args.adaptation_rate),
                    }
                )
                miss = 1.0 - covered
                adaptive_alpha += float(args.adaptation_rate) * (
                    float(args.certificate_alpha) - miss
                )
                absolute_adaptive_alpha += float(args.adaptation_rate) * (
                    float(args.certificate_alpha) - (1.0 - absolute_covered)
                )
                residual.append(
                    (float(row["predicted_paired_margin"]) - float(row["realized_paired_margin"]))
                    / scale
                )
                absolute_residual.append(
                    (float(row["predicted_worst_value"]) - float(row["realized_worst_value"]))
                    / absolute_scale
                )
                risk_row = test_frame.loc[
                    (test_frame["method"] == "RiskFirst")
                    & (test_frame["budget_fraction"] == budget)
                    & (test_frame["block_id"] == row["block_id"])
                ].iloc[0].to_dict()
                policy_row = row.to_dict() if switched else risk_row
                policy_row.update(
                    {
                        "method": "CertifiedSwitch",
                        "certified_switch": switched,
                        "calibrated_paired_margin_lower_bound": lower,
                        "realized_paired_margin": float(row["realized_paired_margin"]) if switched else 0.0,
                    }
                )
                result_rows.append(policy_row)

        diagnostic_rows.append(
            {
                "dataset": args.dataset,
                "fold_id": fold_id,
                "train_rows": len(train),
                "mechanism_tuning_rows": len(mechanism_tune),
                "representation_validation_rows": len(representation_val),
                "set_cal_rows": len(set_cal),
                "test_rows": len(test),
                "scenario_count": len(scenario_names),
                "response_rank": prediction.rank,
                "joint_scenario_correction": prediction.joint_correction,
                "contamination_share": float(args.contamination),
                "selected_candidates": ";".join(
                    f"{int(round(100 * budget))}:{selected_candidate[budget]}" for budget in budgets
                ),
                "negative_scenario_values": float(np.sum(y_deployment < -1e-12)),
                "scenario_above_loss": float(
                    np.sum(y_deployment > deployment_h["loss"].to_numpy(dtype=float)[:, None] + 1e-9)
                ),
                "area_cap_share": 0.40,
                "temporal_loss_weight": methods.diagnostics["temporal_loss_weight"],
                "network_loss_weight": methods.diagnostics["network_loss_weight"],
                "group_penalty": methods.diagnostics["group_penalty"],
                "downside_penalty": methods.diagnostics["downside_penalty"],
            }
        )

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    coverage = pd.DataFrame(coverage_rows)
    results.to_csv(tables / "plan_results.csv", index=False)
    diagnostics.to_csv(tables / "diagnostics.csv", index=False)
    coverage.to_csv(tables / "plan_coverage.csv", index=False)
    summary = _summary(results)
    summary.to_csv(tables / "plan_summary.csv", index=False)
    coverage_summary = (
        coverage.groupby(["dataset", "budget_fraction"], as_index=False)
        .agg(coverage=("covered", "mean"), blocks=("covered", "size"))
        if not coverage.empty
        else pd.DataFrame()
    )
    coverage_summary.to_csv(tables / "plan_coverage_summary.csv", index=False)
    print(diagnostics.to_string(index=False))
    print(summary.to_string(index=False))
    print(coverage_summary.to_string(index=False))


def _evaluate_partition(
    dataset: str,
    fold_id: int,
    partition: str,
    frame: pd.DataFrame,
    predicted_scenarios: np.ndarray,
    robust_scenarios_by_budget: dict[float, np.ndarray],
    realized_scenarios: np.ndarray,
    budgets: list[float],
    published_scores: dict[str, str],
    *,
    contamination: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for block_id, block in frame.groupby("block_id", sort=True):
        loc = block.index.to_numpy() - frame.index.min()
        pred_s = predicted_scenarios[loc]
        real_s = realized_scenarios[loc]
        groups = block["group"].astype(str).to_numpy()
        for budget_fraction in budgets:
            budget = max(1, int(round(len(block) * budget_fraction)))
            group_cap = _area_cap(budget)
            robust_s = robust_scenarios_by_budget[budget_fraction][loc]
            robust_target = contamination_robust_values(robust_s, contamination)
            realized_target = contamination_robust_values(real_s, contamination)
            risk_selected = score_plan_select(
                block[published_scores["RiskFirst"]].to_numpy(dtype=float),
                budget,
                groups=groups,
                group_cap=group_cap,
            )
            robust = robust_plan_select(
                robust_s, budget, groups=groups, group_cap=group_cap
            )
            selections: dict[str, np.ndarray] = {
                "StructuralRobust": robust.selected,
                "RiskFirst": risk_selected,
                "ExpectedActionValue": score_plan_select(
                    np.mean(pred_s, axis=1), budget, groups=groups, group_cap=group_cap
                ),
                "PointwiseWorstAction": score_plan_select(
                    np.min(pred_s, axis=1), budget, groups=groups, group_cap=group_cap
                ),
                "ScenarioCVaR": score_plan_select(
                    np.mean(np.sort(pred_s, axis=1)[:, : max(1, int(np.ceil(0.20 * pred_s.shape[1])))], axis=1),
                    budget,
                    groups=groups,
                    group_cap=group_cap,
                ),
            }
            for method, column in published_scores.items():
                if column in block.columns and method not in {"C-CERTS", "RiskFirst"}:
                    selections[method] = score_plan_select(
                        block[column].to_numpy(dtype=float), budget, groups=groups, group_cap=group_cap
                    )
            for method, selected in selections.items():
                realized = plan_values(real_s, selected)
                predicted = plan_values(robust_s if method == "StructuralRobust" else pred_s, selected)
                predicted_target = robust_target if method == "StructuralRobust" else contamination_robust_values(pred_s, contamination)
                predicted_pair = paired_plan_values(
                    predicted_target,
                    selected,
                    risk_selected,
                )
                realized_pair = paired_plan_values(realized_target, selected, risk_selected)
                changed_actions = int(len(np.setxor1d(selected, risk_selected)))
                predicted_tolerance = (
                    max(predicted_pair["worst"], 0.0) / changed_actions
                    if changed_actions > 0
                    else np.inf
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "fold_id": fold_id,
                        "partition": partition,
                        "block_id": str(block_id),
                        "budget_fraction": budget_fraction,
                        "method": method,
                        "selected_n": float(len(selected)),
                        "predicted_worst_value": predicted["worst"],
                        "realized_worst_value": realized["worst"],
                        "realized_mean_value": realized["mean"],
                        "realized_scenario_spread": realized["spread"],
                        "predicted_paired_margin": predicted_pair["worst"],
                        "realized_paired_margin": realized_pair["worst"],
                        "realized_paired_mean": realized_pair["mean"],
                        "changed_actions": float(changed_actions),
                        "realized_dominance": float(realized_pair["worst"] >= -1e-9),
                        "predicted_discrepancy_tolerance": predicted_tolerance,
                        "solver_gap": robust.gap if method == "StructuralRobust" else np.nan,
                        "feasible": 1.0,
                    }
                )
    return rows


def _scenario_candidates(
    validation_scenarios: np.ndarray,
    deployment_scenarios: np.ndarray,
    validation_pred: pd.DataFrame,
    deployment_pred: pd.DataFrame,
    score_columns: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    validation = {"direct": validation_scenarios}
    deployment = {"direct": deployment_scenarios}
    direct_scale = float(np.median(np.mean(validation_scenarios, axis=1)))
    for method, column in score_columns.items():
        if method == "C-CERTS" or column not in validation_pred or column not in deployment_pred:
            continue
        val_score = np.maximum(validation_pred[column].to_numpy(dtype=float), 0.0)
        dep_score = np.maximum(deployment_pred[column].to_numpy(dtype=float), 0.0)
        positive = val_score[val_score > 1e-12]
        score_scale = float(np.median(positive)) if len(positive) else 1.0
        factor = direct_scale / max(score_scale, 1e-9)
        val_proxy = np.repeat((factor * val_score)[:, None], validation_scenarios.shape[1], axis=1)
        dep_proxy = np.repeat((factor * dep_score)[:, None], deployment_scenarios.shape[1], axis=1)
        validation[f"proxy_{method}"] = val_proxy
        deployment[f"proxy_{method}"] = dep_proxy
        for weight in (0.50,):
            name = f"direct_{weight:.2f}_{method}"
            validation[name] = weight * validation_scenarios + (1.0 - weight) * val_proxy
            deployment[name] = weight * deployment_scenarios + (1.0 - weight) * dep_proxy
    return validation, deployment


def _select_scenario_candidate(
    frame: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    realized_scenarios: np.ndarray,
    budget_fraction: float,
    *,
    contamination: float,
) -> str:
    best_name = "direct"
    best_value = -np.inf
    for name, matrix in candidates.items():
        values = []
        for _, block in frame.groupby("block_id", sort=True):
            loc = block.index.to_numpy() - frame.index.min()
            budget = max(1, int(round(len(block) * budget_fraction)))
            groups = block["group"].astype(str).to_numpy()
            baseline = score_plan_select(
                block["score_risk_first"].to_numpy(dtype=float),
                budget,
                groups=groups,
                group_cap=_area_cap(budget),
            )
            selected = robust_plan_select(
                matrix[loc], budget, groups=groups, group_cap=_area_cap(budget)
            ).selected
            values.append(plan_values(realized_scenarios[loc], selected)["worst"])
        value = float(np.mean(values)) if values else -np.inf
        if value > best_value + 1e-12:
            best_name = name
            best_value = value
    return best_name


def _finite_quantile(values: np.ndarray, level: float) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    rank = int(np.ceil((len(values) + 1) * level))
    if rank > len(values):
        return np.inf
    if rank <= 0:
        return -np.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def _split_validation_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve earlier validation blocks for mechanism tuning and later blocks for rule selection."""
    blocks = sorted(frame["block_id"].astype(str).unique().tolist())
    if len(blocks) < 2:
        raise ValueError("At least two validation blocks are required for ordered nested validation.")
    split = max(1, len(blocks) // 2)
    early = set(blocks[:split])
    mechanism_tune = frame.loc[frame["block_id"].astype(str).isin(early)].copy()
    representation_val = frame.loc[~frame["block_id"].astype(str).isin(early)].copy()
    if mechanism_tune.empty or representation_val.empty:
        raise ValueError("Ordered nested validation produced an empty partition.")
    return mechanism_tune.reset_index(drop=True), representation_val.reset_index(drop=True)


def _area_cap(budget: int) -> int:
    """Limit one operating area to 40% of the selected actions, rounded up."""
    return max(1, int(np.ceil(0.40 * max(int(budget), 1))))


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["partition"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    summary = (
        test.groupby(["dataset", "budget_fraction", "method"], as_index=False)
        .agg(
            mean_worst_value=("realized_worst_value", "mean"),
            mean_value=("realized_mean_value", "mean"),
            blocks=("block_id", "size"),
            feasible_rate=("feasible", "mean"),
            max_solver_gap=("solver_gap", "max"),
        )
    )
    ours = summary[summary["method"] == "StructuralRobust"][["dataset", "budget_fraction", "mean_worst_value"]]
    ours = ours.rename(columns={"mean_worst_value": "ours_worst_value"})
    summary = summary.merge(ours, on=["dataset", "budget_fraction"], how="left")
    summary["gain_of_ours_pct"] = 100.0 * (
        summary["ours_worst_value"] - summary["mean_worst_value"]
    ) / np.maximum(summary["mean_worst_value"], 1e-9)
    return summary


if __name__ == "__main__":
    main()
