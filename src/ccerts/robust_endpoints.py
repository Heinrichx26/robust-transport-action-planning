from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def apply_scenario_envelope(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replace gain_lower with a pointwise lower envelope over prespecified response scenarios."""
    out = frame.copy()
    reference = pd.to_numeric(out["gain_lower"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if dataset == "bts":
        envelope, scenario_count = _bts_envelope(out, reference)
    elif dataset == "road":
        envelope, scenario_count = _road_envelope(out, reference)
    elif dataset == "divvy":
        envelope, scenario_count = _divvy_envelope(out, reference)
    else:
        envelope, scenario_count = reference.copy(), 1

    loss = pd.to_numeric(out["loss"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    upper_col = "gain_upper" if "gain_upper" in out.columns else "observed_or_simulated_gain"
    upper = pd.to_numeric(out[upper_col], errors="coerce").to_numpy(dtype=float)
    upper = np.where(np.isfinite(upper), upper, loss)
    envelope = np.clip(envelope, 0.0, np.minimum(np.maximum(loss, 0.0), np.maximum(upper, 0.0)))
    envelope = np.minimum(envelope, np.maximum(reference, 0.0))
    discrepancy = np.maximum(reference - envelope, 0.0)

    out["gain_reference"] = reference
    out["gain_lower"] = envelope
    out["endpoint_scenario_deduction"] = discrepancy
    positive_reference = reference > 1e-12
    diagnostics = {
        "endpoint_scenario_count": float(scenario_count),
        "endpoint_envelope_positive_pct": 100.0 * float(np.mean(envelope > 1e-12)),
        "endpoint_reference_positive_pct": 100.0 * float(np.mean(positive_reference)),
        "endpoint_mean_deduction": float(np.mean(discrepancy)),
        "endpoint_relative_deduction_pct": 100.0 * float(np.sum(discrepancy)) / max(float(np.sum(reference)), 1e-12),
        "endpoint_envelope_above_reference": float(np.sum(envelope > reference + 1e-9)),
        "endpoint_envelope_negative": float(np.sum(envelope < -1e-9)),
        "endpoint_envelope_above_loss": float(np.sum(envelope > loss + 1e-9)),
    }
    return out, diagnostics


def _bts_envelope(frame: pd.DataFrame, reference: np.ndarray) -> tuple[np.ndarray, int]:
    loss = pd.to_numeric(frame["loss"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    slack = np.clip(pd.to_numeric(frame["schedule_slack"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, None)
    actionable = np.clip(pd.to_numeric(frame["actionable_delay"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, None)
    reliability = np.clip(pd.to_numeric(frame["recovery_reliability"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    scale = 0.50 + 0.50 * reliability
    envelope = np.maximum(reference, 0.0).copy()
    for slack_weight, delay_weight in [(0.25, 0.10), (0.50, 0.10), (0.20, 0.25)]:
        scenario = np.minimum(loss, slack_weight * slack + delay_weight * actionable) * scale
        envelope = np.minimum(envelope, np.maximum(scenario, 0.0))
    return envelope, 4


def _ctm_gain(
    speed: np.ndarray,
    free_flow_speed: np.ndarray,
    network_headroom: np.ndarray,
    *,
    jam_density: float,
    cell_length_km: float,
    control_scale: float,
) -> np.ndarray:
    speed = np.clip(np.asarray(speed, dtype=float), 5.0, None)
    free_flow = np.maximum(np.asarray(free_flow_speed, dtype=float), 25.0)
    headroom = np.clip(np.asarray(network_headroom, dtype=float), 0.0, 1.0)
    interval_hours = 5.0 / 60.0
    density = jam_density * np.clip(1.0 - speed / free_flow, 0.0, 0.95)
    capacity_flow = free_flow * jam_density / 4.0
    control_share = np.clip(control_scale * (0.03 + 0.15 * headroom), 0.0, 0.30)
    controllable_outflow = np.minimum(control_share * capacity_flow, density * cell_length_km / interval_hours)
    action_density = np.maximum(density - controllable_outflow * interval_hours / cell_length_km, 0.0)
    action_speed = free_flow * np.maximum(1.0 - action_density / jam_density, 0.05)
    routine_loss = np.maximum(60.0 * (1.0 / speed - 1.0 / free_flow), 0.0)
    action_loss = np.maximum(60.0 * (1.0 / np.maximum(action_speed, 5.0) - 1.0 / free_flow), 0.0)
    return np.maximum(routine_loss - action_loss, 0.0)


def _road_envelope(frame: pd.DataFrame, reference: np.ndarray) -> tuple[np.ndarray, int]:
    speed = pd.to_numeric(frame["mean_speed"], errors="coerce").fillna(5.0).to_numpy(dtype=float)
    free_flow = pd.to_numeric(frame["free_flow_speed"], errors="coerce").fillna(25.0).to_numpy(dtype=float)
    headroom = pd.to_numeric(frame["network_headroom"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    envelope = np.maximum(reference, 0.0).copy()
    scenarios = list(product([120.0, 150.0, 180.0], [0.6, 0.8, 1.0], [0.75, 1.0, 1.25]))
    for jam_density, cell_length, control_scale in scenarios:
        scenario = 0.70 * _ctm_gain(
            speed,
            free_flow,
            headroom,
            jam_density=jam_density,
            cell_length_km=cell_length,
            control_scale=control_scale,
        )
        envelope = np.minimum(envelope, scenario)
    return envelope, len(scenarios) + 1


def _hour_loss(inventory: np.ndarray, capacity: np.ndarray, departures: np.ndarray, arrivals: np.ndarray) -> np.ndarray:
    served_departures = np.minimum(inventory, departures)
    remaining = inventory - served_departures
    accepted_arrivals = np.minimum(arrivals, np.maximum(capacity - remaining, 0.0))
    return np.maximum(departures - served_departures, 0.0) + np.maximum(arrivals - accepted_arrivals, 0.0)


def _divvy_gain(frame: pd.DataFrame, multiplier: float) -> np.ndarray:
    capacity = pd.to_numeric(frame["capacity_proxy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    initial = pd.to_numeric(frame["inventory_start"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    departures = pd.to_numeric(frame["departures"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    arrivals = pd.to_numeric(frame["arrivals"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    routine = pd.to_numeric(frame["routine_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    radius = np.ceil(multiplier * pd.to_numeric(frame["relocation_limit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)).clip(1.0, 15.0)
    best = routine.copy()
    for offset in range(-15, 16):
        allowed = np.abs(offset) <= radius
        candidate_inventory = np.clip(initial + offset, 0.0, capacity)
        candidate_loss = _hour_loss(candidate_inventory, capacity, departures, arrivals)
        best = np.where(allowed, np.minimum(best, candidate_loss), best)
    return 0.80 * np.maximum(routine - best, 0.0)


def _divvy_envelope(frame: pd.DataFrame, reference: np.ndarray) -> tuple[np.ndarray, int]:
    envelope = np.maximum(reference, 0.0).copy()
    for multiplier in [0.50, 1.00, 1.50]:
        envelope = np.minimum(envelope, _divvy_gain(frame, multiplier))
    return envelope, 4
