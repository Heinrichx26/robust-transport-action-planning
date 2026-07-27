from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ccerts.evaluate import evaluate_block_methods, evaluate_methods, paired_block_tests
from ccerts.calibration import evaluate_selected_set_certificate
from ccerts.transparent_methods import fit_method_family, method_family_table, method_score_columns
from ccerts.pipeline import prepare_features, sample_frame, valid_periods
from ccerts.robust_endpoints import apply_scenario_envelope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin C-CERTS experiments on prepared open data.")
    parser.add_argument("--dataset", choices=["bts", "nyc", "divvy", "road"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ccerts"))
    parser.add_argument("--train-window", type=int, default=2)
    parser.add_argument("--cal-window", type=int, default=1, help="Policy-validation periods.")
    parser.add_argument("--set-cal-window", type=int, default=1, help="Selected-set certificate calibration periods.")
    parser.add_argument("--test-window", type=int, default=1)
    parser.add_argument("--max-rows-per-period", type=int, default=45000)
    parser.add_argument("--max-folds", type=int, default=0, help="0 means all possible folds.")
    parser.add_argument("--min-period-rows", type=int, default=1000)
    parser.add_argument(
        "--period-mode",
        choices=["original", "halfmonth", "week"],
        default="original",
        help="Optionally rebuild ordered periods from block_id for additional deployment folds.",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--certificate-alpha", type=float, default=None)
    parser.add_argument("--joint-budget-certificate", action="store_true")
    parser.add_argument("--endpoint-mode", choices=["reference", "scenario_envelope"], default="reference")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir / args.dataset / "rolling"
    tables = out / "tables"
    figdata = out / "data_for_figures"
    for path in [tables, figdata]:
        path.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, low_memory=False)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["period", "loss", "gain_lower", "group"])
    endpoint_diagnostics: dict[str, float] = {}
    if args.endpoint_mode == "scenario_envelope":
        frame, endpoint_diagnostics = apply_scenario_envelope(args.dataset, frame)
    frame = rebuild_periods(frame, args.period_mode)
    frame["period"] = frame["period"].astype(str)
    periods = valid_periods(frame, args.min_period_rows)
    folds = build_folds(periods, args.train_window, args.cal_window, args.set_cal_window, args.test_window)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("Not enough valid periods for rolling-origin evaluation.")

    all_curves: list[pd.DataFrame] = []
    all_models: list[pd.DataFrame] = []
    all_inversions: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_set_certificates: list[pd.DataFrame] = []
    all_block_values: list[pd.DataFrame] = []
    score_cols = method_score_columns()
    budgets = [0.01, 0.03, 0.05, 0.10, 0.20]

    for fold_id, fold in enumerate(folds, start=1):
        print(
            f"fold {fold_id}: train={fold['train']} policy_val={fold['policy_val']} "
            f"set_cal={fold['set_cal']} test={fold['test']}"
        )
        train = period_slice(frame, fold["train"], args.max_rows_per_period, args.seed + fold_id)
        policy_val = period_slice(frame, fold["policy_val"], args.max_rows_per_period, args.seed + 100 + fold_id)
        set_cal = period_slice(frame, fold["set_cal"], args.max_rows_per_period, args.seed + 150 + fold_id)
        test = period_slice(frame, fold["test"], args.max_rows_per_period, args.seed + 200 + fold_id)
        train = ensure_block_id(args.dataset, train)
        policy_val = ensure_block_id(args.dataset, policy_val)
        set_cal = ensure_block_id(args.dataset, set_cal)
        test = ensure_block_id(args.dataset, test)
        deployment_pool = pd.concat([set_cal, test], ignore_index=True)
        train, policy_val, deployment_pool, x_train, x_val, x_deployment = prepare_features(
            args.dataset, train, policy_val, deployment_pool
        )
        outputs = fit_method_family(
            train,
            policy_val,
            deployment_pool,
            x_train,
            x_val,
            x_deployment,
            alpha=args.alpha,
            seed=args.seed + fold_id,
        )
        if args.endpoint_mode == "scenario_envelope":
            for budget in [0.01, 0.03, 0.05, 0.10, 0.20]:
                key = int(round(budget * 100))
                outputs.predictions[f"score_ccerts_b{key}"] = outputs.predictions["score_ccerts"]
                outputs.predictions[f"value_ccerts_b{key}"] = outputs.predictions["score_ccerts"]
                outputs.calibration_predictions[f"score_ccerts_b{key}"] = outputs.calibration_predictions["score_ccerts"]
                outputs.calibration_predictions[f"value_ccerts_b{key}"] = outputs.calibration_predictions["score_ccerts"]
        set_cal_pred = outputs.predictions[outputs.predictions["period"].isin(fold["set_cal"])].copy()
        test_pred = outputs.predictions[outputs.predictions["period"].isin(fold["test"])].copy()
        curve, model_summary, inversion = evaluate_methods(test_pred, score_cols, budgets)
        certificate_alpha = args.alpha if args.certificate_alpha is None else args.certificate_alpha
        set_certificates = evaluate_selected_set_certificate(
            set_cal_pred,
            test_pred,
            budgets,
            certificate_alpha,
            joint_budgets=args.joint_budget_certificate,
        )
        block_values = evaluate_block_methods(test_pred, score_cols, budgets)

        for table in [curve, model_summary, inversion]:
            table.insert(0, "dataset", args.dataset)
            table.insert(1, "fold_id", fold_id)
            table.insert(2, "train_periods", ",".join(fold["train"]))
            table.insert(3, "policy_val_periods", ",".join(fold["policy_val"]))
            table.insert(4, "set_cal_periods", ",".join(fold["set_cal"]))
            table.insert(5, "test_periods", ",".join(fold["test"]))
        diagnostics = pd.DataFrame([outputs.diagnostics])
        diagnostics["certificate_alpha"] = float(certificate_alpha)
        diagnostics["endpoint_mode"] = args.endpoint_mode
        diagnostics["joint_budget_certificate"] = float(args.joint_budget_certificate)
        diagnostics["endpoint_policy"] = "fixed_validated_stack" if args.endpoint_mode == "scenario_envelope" else "capacity_specific"
        for name, value in endpoint_diagnostics.items():
            diagnostics[name] = value
        diagnostics.insert(0, "dataset", args.dataset)
        diagnostics.insert(1, "fold_id", fold_id)
        diagnostics.insert(2, "train_periods", ",".join(fold["train"]))
        diagnostics.insert(3, "policy_val_periods", ",".join(fold["policy_val"]))
        diagnostics.insert(4, "set_cal_periods", ",".join(fold["set_cal"]))
        diagnostics.insert(5, "test_periods", ",".join(fold["test"]))

        all_curves.append(curve)
        all_models.append(model_summary)
        all_inversions.append(inversion)
        all_diagnostics.append(diagnostics)
        if not set_certificates.empty:
            set_certificates.insert(0, "dataset", args.dataset)
            set_certificates.insert(1, "fold_id", fold_id)
            set_certificates.insert(2, "set_cal_periods", ",".join(fold["set_cal"]))
            set_certificates.insert(3, "test_periods", ",".join(fold["test"]))
            all_set_certificates.append(set_certificates)
        if not block_values.empty:
            block_values.insert(0, "dataset", args.dataset)
            block_values.insert(1, "fold_id", fold_id)
            block_values.insert(2, "test_periods", ",".join(fold["test"]))
            all_block_values.append(block_values)

    curve_df = pd.concat(all_curves, ignore_index=True)
    model_df = pd.concat(all_models, ignore_index=True)
    inversion_df = pd.concat(all_inversions, ignore_index=True)
    diagnostics_df = pd.concat(all_diagnostics, ignore_index=True)
    family_df = method_family_table()
    summary_df = summarize(curve_df, inversion_df)

    curve_df.to_csv(tables / "table_rolling_budget_curve.csv", index=False)
    model_df.to_csv(tables / "table_rolling_model_summary.csv", index=False)
    inversion_df.to_csv(tables / "table_rolling_inversion.csv", index=False)
    diagnostics_df.to_csv(tables / "table_rolling_diagnostics.csv", index=False)
    if all_set_certificates:
        set_certificate_df = pd.concat(all_set_certificates, ignore_index=True)
        set_certificate_df.to_csv(tables / "table_set_certificate.csv", index=False)
        set_certificate_summary = (
            set_certificate_df.groupby(["dataset", "budget_fraction"], as_index=False)
            .agg(
                set_coverage=("certificate_hit", "mean"),
                uncalibrated_coverage=("uncalibrated_hit", "mean"),
                deployment_blocks=("certificate_hit", "size"),
                mean_uncalibrated_set_value=("uncalibrated_set_value", "mean"),
                mean_certified_set_value=("certified_set_value", "mean"),
                mean_observed_set_value=("observed_set_value", "mean"),
                calibration_blocks=("calibration_blocks", "min"),
            )
        )
        set_certificate_summary.to_csv(tables / "table_set_certificate_summary.csv", index=False)
    if all_block_values:
        block_value_df = pd.concat(all_block_values, ignore_index=True)
        block_value_df.to_csv(tables / "table_block_method_gain.csv", index=False)
        paired_df = paired_block_tests(block_value_df, seed=args.seed)
        paired_df.insert(0, "dataset", args.dataset)
        paired_df.to_csv(tables / "table_paired_tests.csv", index=False)
    family_df.to_csv(tables / "table_method_families.csv", index=False)
    summary_df.to_csv(tables / "table_rolling_summary.csv", index=False)
    curve_df.to_csv(figdata / "rolling_budget_curve.csv", index=False)
    inversion_df.to_csv(figdata / "rolling_inversion.csv", index=False)
    print(summary_df.to_string(index=False))


def build_folds(
    periods: list[str],
    train_window: int,
    policy_val_window: int,
    set_cal_window: int,
    test_window: int,
) -> list[dict[str, list[str]]]:
    total = train_window + policy_val_window + set_cal_window + test_window
    folds = []
    for start in range(0, len(periods) - total + 1):
        train = periods[start : start + train_window]
        policy_val = periods[start + train_window : start + train_window + policy_val_window]
        set_cal_start = start + train_window + policy_val_window
        set_cal = periods[set_cal_start : set_cal_start + set_cal_window]
        test = periods[set_cal_start + set_cal_window : start + total]
        folds.append({"train": train, "policy_val": policy_val, "set_cal": set_cal, "test": test})
    return folds


def rebuild_periods(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "original":
        return frame
    if "block_id" not in frame.columns:
        raise ValueError(f"period mode {mode!r} requires block_id.")
    work = frame.copy()
    timestamp = pd.to_datetime(work["block_id"], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"period mode {mode!r} requires date-like block_id values.")
    if mode == "halfmonth":
        half = np.where(timestamp.dt.day <= 15, "H1", "H2")
        work["period"] = timestamp.dt.strftime("%Y-%m-") + half
    elif mode == "week":
        iso = timestamp.dt.isocalendar()
        work["period"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    work["period_id"] = work["period"]
    return work


def period_slice(frame: pd.DataFrame, periods: list[str], max_rows_per_period: int, seed: int) -> pd.DataFrame:
    parts = []
    for offset, period in enumerate(periods):
        part = frame[frame["period"] == period].copy()
        parts.append(sample_frame(part, max_rows_per_period, seed + offset))
    return pd.concat(parts, ignore_index=True)


def ensure_block_id(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "block_id" in work.columns:
        work["block_id"] = work["block_id"].astype(str)
        return work
    if dataset == "bts" and "flight_day" in work.columns:
        work["block_id"] = work["flight_day"].astype(str)
    elif "cell_time" in work.columns:
        work["block_id"] = pd.to_datetime(work["cell_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        work["block_id"] = work["period"].astype(str)
    return work


def summarize(curve: pd.DataFrame, inversion: pd.DataFrame) -> pd.DataFrame:
    unconstrained = curve[curve["constraint"] == "none"].copy()
    method_summary = (
        unconstrained.groupby(["dataset", "method", "budget_fraction"], as_index=False)
        .agg(
            gain_lower_capture_mean=("gain_lower_capture", "mean"),
            gain_lower_capture_std=("gain_lower_capture", "std"),
            certified_gain_sum_mean=("certified_gain_sum", "mean"),
            high_loss_recall_mean=("high_loss_recall", "mean"),
        )
    )
    inv_summary = (
        inversion.groupby(["dataset", "budget_fraction"], as_index=False)
        .agg(
            actionability_gap_mean=("actionability_gap", "mean"),
            risk_inversion_rate_mean=("risk_inversion_rate", "mean"),
        )
    )
    return method_summary.merge(inv_summary, on=["dataset", "budget_fraction"], how="left")


if __name__ == "__main__":
    main()
