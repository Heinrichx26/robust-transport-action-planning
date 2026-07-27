from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover
    from ccerts.numpy_ml import NumpyIsotonicRegression as IsotonicRegression

from ccerts.calibration import conformal_lower_quantile
from ccerts.model_utils import fit_lgbm_regressor, group_history_score, zscore, zscore_with_reference
from ccerts.selection import top_budget_select


@dataclass(frozen=True)
class MethodOutputs:
    predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    diagnostics: dict[str, float]


def fit_method_family(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    deployment: pd.DataFrame,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_deployment: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> MethodOutputs:
    """Fit transparent comparison rules and the finite-class C-CERTS selector.

    The validation outcomes choose one predeclared priority rule for each
    intervention capacity. Deployment outcomes are not used until the later
    selected-intervention calibration and evaluation stages.
    """
    train = train.reset_index(drop=True).copy()
    validation = validation.reset_index(drop=True).copy()
    deployment = deployment.reset_index(drop=True).copy()

    fit_index, score_calibration_index = _split_earlier_data(train)
    fit_train = train.iloc[fit_index].reset_index(drop=True)
    score_calibration = train.iloc[score_calibration_index].reset_index(drop=True)
    x_fit = x_train[fit_index]
    x_score_calibration = x_train[score_calibration_index]

    loss_model = fit_lgbm_regressor(x_fit, fit_train["loss"], seed)
    gain_model = fit_lgbm_regressor(x_fit, fit_train["gain_lower"], seed + 11)
    ratio_model = fit_lgbm_regressor(x_fit, _recoverable_share(fit_train), seed + 23)
    gain_q80_model = fit_lgbm_regressor(
        x_fit, fit_train["gain_lower"], seed + 31, objective="quantile", alpha=0.80
    )
    gain_q20_model = fit_lgbm_regressor(
        x_fit, fit_train["gain_lower"], seed + 37, objective="quantile", alpha=0.20
    )
    loss_q80_model = fit_lgbm_regressor(
        x_fit, fit_train["loss"], seed + 41, objective="quantile", alpha=0.80
    )

    validation_scores, deployment_scores = _base_scores(
        score_calibration,
        validation,
        deployment,
        x_score_calibration,
        x_validation,
        x_deployment,
        loss_model,
        gain_model,
        ratio_model,
        gain_q80_model,
        gain_q20_model,
        loss_q80_model,
        alpha,
    )
    comparison_validation, comparison_deployment, comparison_diagnostics = _published_comparisons(
        fit_train,
        validation,
        deployment,
        x_train,
        x_validation,
        x_deployment,
        alpha,
        seed + 101,
        validation_scores,
        deployment_scores,
    )
    operational_validation, operational_deployment = _operational_response_scores(
        train,
        validation,
        deployment,
        validation_scores["LossPriority"],
        deployment_scores["LossPriority"],
    )
    validation_scores["OperationalResponse"] = operational_validation
    deployment_scores["OperationalResponse"] = operational_deployment

    candidate_order = [
        "LossPriority",
        "OperationalResponse",
        "DirectLowerImprovement",
        "RecoverableShare",
        "UpperTailImprovement",
        "PositiveImprovement",
    ]
    selected_rules: dict[int, str] = {}
    for capacity in _capacities():
        selected_rules[_capacity_key(capacity)] = _select_rule(
            validation,
            {name: validation_scores[name] for name in candidate_order},
            capacity,
        )

    pred = _prediction_frame(deployment)
    cal_pred = _prediction_frame(validation)
    _add_score_columns(pred, deployment_scores)
    _add_score_columns(cal_pred, validation_scores)
    _add_score_columns(pred, comparison_deployment)
    _add_score_columns(cal_pred, comparison_validation)

    for capacity in _capacities():
        key = _capacity_key(capacity)
        name = selected_rules[key]
        pred[f"score_ccerts_b{key}"] = deployment_scores[name]
        cal_pred[f"score_ccerts_b{key}"] = validation_scores[name]
        pred[f"value_ccerts_b{key}"] = deployment_scores[name]
        cal_pred[f"value_ccerts_b{key}"] = validation_scores[name]

    default_rule = selected_rules[_capacity_key(0.05)]
    pred["score_ccerts"] = deployment_scores[default_rule]
    cal_pred["score_ccerts"] = validation_scores[default_rule]
    pred["cert_gain"] = deployment_scores[default_rule]
    cal_pred["cert_gain"] = validation_scores[default_rule]

    diagnostics: dict[str, float] = {
        "alpha": float(alpha),
        "train_rows": float(len(train)),
        "score_calibration_rows": float(len(score_calibration)),
        "model_fit_rows": float(len(fit_train)),
        "validation_rows": float(len(validation)),
        "deployment_rows": float(len(deployment)),
        "finite_priority_rule_count": float(len(candidate_order)),
    }
    diagnostics.update(comparison_diagnostics)
    for capacity in _capacities():
        key = _capacity_key(capacity)
        diagnostics[f"ccerts_rule_b{key}"] = float(candidate_order.index(selected_rules[key]) + 1)
        diagnostics[f"ccerts_validation_value_b{key}"] = _validation_value(
            validation,
            validation_scores[selected_rules[key]],
            capacity,
        )

    pred = pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cal_pred = cal_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return MethodOutputs(predictions=pred, calibration_predictions=cal_pred, diagnostics=diagnostics)


def method_score_columns() -> dict[str, str]:
    return {
        "RiskFirst": "score_risk_first",
        "TemporalContext2024": "score_TemporalContext2024",
        "NetworkPropagation2024": "score_NetworkPropagation2024",
        "CalibratedUncertainty2025": "score_CalibratedUncertainty2025",
        "MappedPTO2026": "score_MappedPTO2026",
        "GroupAware2025": "score_GroupAware2025",
        "RobustImprovement2025": "score_RobustImprovement2025",
        "C-CERTS": "score_ccerts",
    }


def method_family_table() -> pd.DataFrame:
    rows = [
        {
            "method": "RiskFirst",
            "source": "Predicted-loss priority",
            "venue": "Common operational rule",
            "year": 2026,
            "decision_score": r"$\widehat r_a$",
        },
        {
            "method": "TemporalContext2024",
            "source": "Mamdouh et al.",
            "venue": "Expert Systems with Applications",
            "year": 2024,
            "decision_score": r"$0.65\widehat r_a+0.35h_a^{time}$",
        },
        {
            "method": "NetworkPropagation2024",
            "source": "Sun et al.",
            "venue": "Expert Systems with Applications",
            "year": 2024,
            "decision_score": r"$0.55\widehat r_a+0.45h_a^{network}$",
        },
        {
            "method": "CalibratedUncertainty2025",
            "source": "Tang et al.",
            "venue": "Machine Learning",
            "year": 2025,
            "decision_score": r"$\widehat r_a^{(0.80)}-q_{1-\alpha}$",
        },
        {
            "method": "MappedPTO2026",
            "source": "Guo et al.",
            "venue": "Transportation Research Part B: Methodological",
            "year": 2026,
            "decision_score": r"$\widehat m(\widehat r_a)$",
        },
        {
            "method": "GroupAware2025",
            "source": "Chen et al.",
            "venue": "Transportation Research Part B: Methodological",
            "year": 2025,
            "decision_score": r"$z(\widehat r_a)-0.20z(\bar r_{g(a)})$",
        },
        {
            "method": "RobustImprovement2025",
            "source": "Shadman et al.",
            "venue": "Expert Systems with Applications",
            "year": 2025,
            "decision_score": r"$\widehat g_a-0.15\rho_a$",
        },
        {
            "method": "C-CERTS",
            "source": "This study",
            "venue": "Proposed method",
            "year": 2026,
            "decision_score": r"$s_a^{\widehat\pi_B}$",
        },
    ]
    return pd.DataFrame(rows)


def _base_scores(
    score_calibration: pd.DataFrame,
    validation: pd.DataFrame,
    deployment: pd.DataFrame,
    x_score_calibration: np.ndarray,
    x_validation: np.ndarray,
    x_deployment: np.ndarray,
    loss_model,
    gain_model,
    ratio_model,
    gain_q80_model,
    gain_q20_model,
    loss_q80_model,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    score_calibration_loss = np.clip(loss_model.predict(x_score_calibration), 0.0, None)
    score_calibration_gain = np.clip(gain_model.predict(x_score_calibration), 0.0, None)
    score_calibration_ratio = np.clip(
        ratio_model.predict(x_score_calibration), 0.0, 1.0
    )
    score_calibration_q80 = np.clip(
        gain_q80_model.predict(x_score_calibration), 0.0, None
    )
    score_calibration_loss_q80 = np.clip(
        loss_q80_model.predict(x_score_calibration), 0.0, None
    )
    validation_loss = np.clip(loss_model.predict(x_validation), 0.0, None)
    deployment_loss = np.clip(loss_model.predict(x_deployment), 0.0, None)
    validation_gain = np.clip(gain_model.predict(x_validation), 0.0, None)
    deployment_gain = np.clip(gain_model.predict(x_deployment), 0.0, None)
    validation_ratio = np.clip(ratio_model.predict(x_validation), 0.0, 1.0)
    deployment_ratio = np.clip(ratio_model.predict(x_deployment), 0.0, 1.0)
    validation_q80 = np.clip(gain_q80_model.predict(x_validation), 0.0, None)
    deployment_q80 = np.clip(gain_q80_model.predict(x_deployment), 0.0, None)
    validation_q20 = np.clip(gain_q20_model.predict(x_validation), 0.0, None)
    deployment_q20 = np.clip(gain_q20_model.predict(x_deployment), 0.0, None)
    validation_loss_q80 = np.clip(loss_q80_model.predict(x_validation), 0.0, None)
    deployment_loss_q80 = np.clip(loss_q80_model.predict(x_deployment), 0.0, None)

    direct_q = conformal_lower_quantile(
        score_calibration_gain,
        score_calibration["gain_lower"].to_numpy(dtype=float),
        alpha,
    )
    ratio_raw_score_calibration = score_calibration_ratio * score_calibration_loss
    ratio_raw_validation = validation_ratio * validation_loss
    ratio_raw_deployment = deployment_ratio * deployment_loss
    ratio_q = conformal_lower_quantile(
        ratio_raw_score_calibration,
        score_calibration["gain_lower"].to_numpy(dtype=float),
        alpha,
    )
    tail_q = conformal_lower_quantile(
        score_calibration_q80,
        score_calibration["gain_lower"].to_numpy(dtype=float),
        alpha,
    )
    positive_raw_score_calibration = np.sqrt(
        score_calibration_gain * score_calibration_q80
    )
    positive_raw_validation = np.sqrt(validation_gain * validation_q80)
    positive_raw_deployment = np.sqrt(deployment_gain * deployment_q80)
    positive_q = conformal_lower_quantile(
        positive_raw_score_calibration,
        score_calibration["gain_lower"].to_numpy(dtype=float),
        alpha,
    )

    validation_scores = {
        "LossPriority": validation_loss,
        "DirectLowerImprovement": np.clip(validation_gain - direct_q, 0.0, None),
        "RecoverableShare": np.clip(ratio_raw_validation - ratio_q, 0.0, None),
        "UpperTailImprovement": np.clip(validation_q80 - tail_q, 0.0, None),
        "PositiveImprovement": np.clip(positive_raw_validation - positive_q, 0.0, None),
        "LossQuantile": validation_loss_q80,
        "GainQ20": validation_q20,
        "GainMean": validation_gain,
    }
    deployment_scores = {
        "LossPriority": deployment_loss,
        "DirectLowerImprovement": np.clip(deployment_gain - direct_q, 0.0, None),
        "RecoverableShare": np.clip(ratio_raw_deployment - ratio_q, 0.0, None),
        "UpperTailImprovement": np.clip(deployment_q80 - tail_q, 0.0, None),
        "PositiveImprovement": np.clip(positive_raw_deployment - positive_q, 0.0, None),
        "LossQuantile": deployment_loss_q80,
        "GainQ20": deployment_q20,
        "GainMean": deployment_gain,
    }
    return validation_scores, deployment_scores


def _split_earlier_data(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keep score calibration earlier than rule validation and final evaluation."""
    if len(frame) < 10:
        split = max(1, len(frame) - 1)
        return np.arange(split), np.arange(split, len(frame))

    if "block_id" in frame.columns:
        block = frame["block_id"].astype(str)
        unique_blocks = sorted(block.unique().tolist())
        calibration_blocks = max(1, int(np.ceil(0.20 * len(unique_blocks))))
        calibration_values = set(unique_blocks[-calibration_blocks:])
        calibration_mask = block.isin(calibration_values).to_numpy()
        if calibration_mask.any() and (~calibration_mask).any():
            return np.flatnonzero(~calibration_mask), np.flatnonzero(calibration_mask)

    split = max(1, int(np.floor(0.80 * len(frame))))
    split = min(split, len(frame) - 1)
    return np.arange(split), np.arange(split, len(frame))


def _published_comparisons(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    deployment: pd.DataFrame,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_deployment: np.ndarray,
    alpha: float,
    seed: int,
    validation_scores: dict[str, np.ndarray],
    deployment_scores: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    loss_validation = validation_scores["LossPriority"]
    loss_deployment = deployment_scores["LossPriority"]

    time_validation, time_deployment = _context_scores(
        train, validation, deployment, loss_validation, loss_deployment, "time"
    )
    network_validation, network_deployment = _context_scores(
        train, validation, deployment, loss_validation, loss_deployment, "network"
    )
    loss_quantile_q = conformal_lower_quantile(
        validation_scores["LossQuantile"], validation["loss"].to_numpy(dtype=float), alpha
    )

    mapper = IsotonicRegression(out_of_bounds="clip")
    try:
        mapper.fit(loss_validation, validation["gain_lower"].to_numpy(dtype=float))
        mapped_validation = np.clip(mapper.predict(loss_validation), 0.0, None)
        mapped_deployment = np.clip(mapper.predict(loss_deployment), 0.0, None)
    except Exception:
        mapped_validation = validation_scores["DirectLowerImprovement"]
        mapped_deployment = deployment_scores["DirectLowerImprovement"]

    group_validation = group_history_score(train, validation, ["group"], "loss")
    group_deployment = group_history_score(train, deployment, ["group"], "loss")
    group_loss_validation = zscore(loss_validation)
    group_context_validation = zscore(group_validation)
    group_loss_deployment = zscore_with_reference(loss_deployment, loss_validation)
    group_context_deployment = zscore_with_reference(group_deployment, group_validation)

    downside_validation = np.maximum(
        validation_scores["GainMean"] - validation_scores["GainQ20"], 0.0
    )
    downside_deployment = np.maximum(
        deployment_scores["GainMean"] - deployment_scores["GainQ20"], 0.0
    )
    blend_grid = (0.25, 0.50, 0.75)
    penalty_grid = (0.10, 0.25, 0.50, 1.00)
    temporal_weight = _select_parameter(
        validation,
        {
            weight: weight * loss_validation + (1.0 - weight) * time_validation
            for weight in blend_grid
        },
    )
    network_weight = _select_parameter(
        validation,
        {
            weight: weight * loss_validation + (1.0 - weight) * network_validation
            for weight in blend_grid
        },
    )
    group_penalty = _select_parameter(
        validation,
        {
            penalty: group_loss_validation - penalty * group_context_validation
            for penalty in penalty_grid
        },
    )
    downside_penalty = _select_parameter(
        validation,
        {
            penalty: validation_scores["DirectLowerImprovement"] - penalty * downside_validation
            for penalty in penalty_grid
        },
    )

    group_rule_validation = group_loss_validation - group_penalty * group_context_validation
    group_rule_deployment = group_loss_deployment - group_penalty * group_context_deployment
    robust_validation = (
        validation_scores["DirectLowerImprovement"] - downside_penalty * downside_validation
    )
    robust_deployment = (
        deployment_scores["DirectLowerImprovement"] - downside_penalty * downside_deployment
    )

    return (
        {
            "TemporalContext2024": temporal_weight * loss_validation + (1.0 - temporal_weight) * time_validation,
            "NetworkPropagation2024": network_weight * loss_validation + (1.0 - network_weight) * network_validation,
            "CalibratedUncertainty2025": validation_scores["LossQuantile"] - loss_quantile_q,
            "MappedPTO2026": mapped_validation,
            "GroupAware2025": group_rule_validation,
            "RobustImprovement2025": robust_validation,
        },
        {
            "TemporalContext2024": temporal_weight * loss_deployment + (1.0 - temporal_weight) * time_deployment,
            "NetworkPropagation2024": network_weight * loss_deployment + (1.0 - network_weight) * network_deployment,
            "CalibratedUncertainty2025": deployment_scores["LossQuantile"] - loss_quantile_q,
            "MappedPTO2026": mapped_deployment,
            "GroupAware2025": group_rule_deployment,
            "RobustImprovement2025": robust_deployment,
        },
        {
            "temporal_loss_weight": float(temporal_weight),
            "network_loss_weight": float(network_weight),
            "group_penalty": float(group_penalty),
            "downside_penalty": float(downside_penalty),
        },
    )


def _select_parameter(
    validation: pd.DataFrame,
    candidates: dict[float, np.ndarray],
) -> float:
    """Choose one predeclared parameter on earlier blocks across all capacities."""
    capacities = _capacities()
    values = {
        parameter: np.asarray(
            [_validation_value(validation, score, capacity) for capacity in capacities],
            dtype=float,
        )
        for parameter, score in candidates.items()
    }
    scale = np.maximum(
        np.max(np.vstack(list(values.values())), axis=0),
        1e-9,
    )
    objective = {
        parameter: float(np.mean(candidate_values / scale))
        for parameter, candidate_values in values.items()
    }
    return max(sorted(objective), key=lambda parameter: objective[parameter])


def _context_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    deployment: pd.DataFrame,
    validation_loss: np.ndarray,
    deployment_loss: np.ndarray,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "time":
        if "hist_route_mean" in validation and "hist_route_mean" in deployment:
            validation_context = pd.to_numeric(
                validation["hist_route_mean"], errors="coerce"
            ).fillna(float(np.mean(validation_loss))).to_numpy(dtype=float)
            deployment_context = pd.to_numeric(
                deployment["hist_route_mean"], errors="coerce"
            ).fillna(float(np.mean(validation_loss))).to_numpy(dtype=float)
            return validation_context, deployment_context
        if "hist_od_mean" in validation and "hist_od_mean" in deployment:
            validation_context = pd.to_numeric(
                validation["hist_od_mean"], errors="coerce"
            ).fillna(float(np.mean(validation_loss))).to_numpy(dtype=float)
            deployment_context = pd.to_numeric(
                deployment["hist_od_mean"], errors="coerce"
            ).fillna(float(np.mean(validation_loss))).to_numpy(dtype=float)
            return validation_context, deployment_context
        return validation_loss, deployment_loss

    if {"origin", "dep_hour"}.issubset(train.columns):
        keys = ["origin", "dep_hour"]
    elif {"pickup_zone", "hour"}.issubset(train.columns):
        keys = ["pickup_zone", "hour"]
    else:
        keys = ["group"]
    return (
        group_history_score(train, validation, keys, "loss"),
        group_history_score(train, deployment, keys, "loss"),
    )


def _select_rule(
    validation: pd.DataFrame,
    scores: dict[str, np.ndarray],
    capacity: float,
) -> str:
    values = {
        name: _validation_value(validation, score, capacity)
        for name, score in scores.items()
    }
    return max(values, key=values.get)


def _validation_value(
    validation: pd.DataFrame,
    score: np.ndarray,
    capacity: float,
) -> float:
    work = validation[["unit_id", "loss", "gain_lower", "group"]].copy()
    work["priority_score"] = np.asarray(score, dtype=float)
    if "block_id" not in validation.columns:
        return float(
            top_budget_select(
                work,
                "priority_score",
                capacity,
                "validation",
                group_col="group",
                group_cap_fraction=0.40,
            ).selected[
                "gain_lower"
            ].sum()
        )
    values = []
    work["block_id"] = validation["block_id"].astype(str).to_numpy()
    for _, block in work.groupby("block_id"):
        selected = top_budget_select(
            block,
            "priority_score",
            capacity,
            "validation",
            group_col="group",
            group_cap_fraction=0.40,
        ).selected
        values.append(float(selected["gain_lower"].sum()))
    return float(np.mean(values)) if values else 0.0


def _prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["unit_id", "period", "loss", "gain_lower", "group"]
    columns += [
        column
        for column in ["block_id", "observed_or_simulated_gain"]
        if column in frame.columns
    ]
    return frame[columns].copy()


def _add_score_columns(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> None:
    column_names = {
        "LossPriority": "score_risk_first",
        "DirectLowerImprovement": "score_direct_lower",
        "RecoverableShare": "score_recoverable_share",
        "UpperTailImprovement": "score_upper_tail_improvement",
        "PositiveImprovement": "score_positive_improvement",
        "OperationalResponse": "score_operational_response",
        "TemporalContext2024": "score_TemporalContext2024",
        "NetworkPropagation2024": "score_NetworkPropagation2024",
        "CalibratedUncertainty2025": "score_CalibratedUncertainty2025",
        "MappedPTO2026": "score_MappedPTO2026",
        "GroupAware2025": "score_GroupAware2025",
        "RobustImprovement2025": "score_RobustImprovement2025",
    }
    for name, column in column_names.items():
        if name in scores:
            frame[column] = scores[name]


def _recoverable_share(frame: pd.DataFrame) -> np.ndarray:
    loss = np.clip(frame["loss"].to_numpy(dtype=float), 0.0, None)
    gain = np.clip(frame["gain_lower"].to_numpy(dtype=float), 0.0, None)
    return np.divide(gain, np.maximum(loss, 1e-6), out=np.zeros_like(gain), where=loss > 0)


def _operational_response_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    deployment: pd.DataFrame,
    validation_loss: np.ndarray,
    deployment_loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = str(train["dataset"].iloc[0]) if "dataset" in train and len(train) else ""
    if dataset == "bts":
        keys = ["carrier", "route", "dep_hour"]
        if {"actionable_delay", "recovery_reliability", "schedule_slack"}.difference(train.columns):
            validation_history = group_history_score(train, validation, keys, "gain_lower")
            deployment_history = group_history_score(train, deployment, keys, "gain_lower")
            return (
                np.minimum(validation_loss, np.clip(validation_history, 0.0, None)),
                np.minimum(deployment_loss, np.clip(deployment_history, 0.0, None)),
            )
        validation_delay = group_history_score(train, validation, keys, "actionable_delay")
        deployment_delay = group_history_score(train, deployment, keys, "actionable_delay")
        validation_recovery = np.clip(
            group_history_score(train, validation, keys, "recovery_reliability"), 0.0, 1.0
        )
        deployment_recovery = np.clip(
            group_history_score(train, deployment, keys, "recovery_reliability"), 0.0, 1.0
        )
        validation_slack = pd.to_numeric(
            validation["schedule_slack"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        deployment_slack = pd.to_numeric(
            deployment["schedule_slack"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        validation_score = np.minimum(
            validation_loss, 0.35 * validation_slack + 0.15 * validation_delay
        ) * (0.50 + 0.50 * validation_recovery)
        deployment_score = np.minimum(
            deployment_loss, 0.35 * deployment_slack + 0.15 * deployment_delay
        ) * (0.50 + 0.50 * deployment_recovery)
        return validation_score, deployment_score

    if dataset == "road":
        validation_room = pd.to_numeric(
            validation["lag_network_headroom"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        deployment_room = pd.to_numeric(
            deployment["lag_network_headroom"], errors="coerce"
        ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        return (
            validation_loss * (0.03 + 0.15 * validation_room),
            deployment_loss * (0.03 + 0.15 * deployment_room),
        )

    validation_capacity = pd.to_numeric(
        validation["capacity_proxy"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    deployment_capacity = pd.to_numeric(
        deployment["capacity_proxy"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    validation_inventory = pd.to_numeric(
        validation["inventory_start"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    deployment_inventory = pd.to_numeric(
        deployment["inventory_start"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    validation_limit = pd.to_numeric(
        validation["relocation_limit"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    deployment_limit = pd.to_numeric(
        deployment["relocation_limit"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    validation_imbalance = np.abs(validation_inventory - 0.5 * validation_capacity)
    deployment_imbalance = np.abs(deployment_inventory - 0.5 * deployment_capacity)
    return (
        np.minimum(validation_loss, 0.80 * np.minimum(validation_limit, validation_imbalance)),
        np.minimum(deployment_loss, 0.80 * np.minimum(deployment_limit, deployment_imbalance)),
    )


def _capacities() -> tuple[float, ...]:
    return (0.01, 0.03, 0.05, 0.10, 0.20)


def _capacity_key(capacity: float) -> int:
    return int(round(100 * capacity))
