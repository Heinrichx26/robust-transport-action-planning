from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare METR-LA morning-peak records for roadway local-control analysis.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/open/road"))
    parser.add_argument("--output", type=Path, default=Path("data/open/road/ccerts_road_ready.csv"))
    parser.add_argument("--start", default="2012-03-01 00:00:00")
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--period-days", type=int, default=14)
    parser.add_argument("--smoke-samples", type=int, default=0, help="Use the first N sequential samples; 0 uses all files.")
    parser.add_argument("--peak-start-hour", type=int, default=6, help="Inclusive local morning-peak start hour.")
    parser.add_argument("--peak-end-hour", type=int, default=10, help="Exclusive local morning-peak end hour.")
    parser.add_argument("--include-weekends", action="store_true", help="Retain weekend records in the peak window.")
    return parser.parse_args()


def load_sequential_speed(input_dir: Path, smoke_samples: int = 0) -> np.ndarray:
    parts: list[np.ndarray] = []
    for name in ["train.npz", "val.npz", "test.npz"]:
        path = input_dir / name
        if not path.exists():
            continue
        bundle = np.load(path)
        x = np.asarray(bundle["x"], dtype=float)
        if x.ndim != 4 or x.shape[1] < 1 or x.shape[-1] < 1:
            raise ValueError(f"Unexpected METR-LA array shape in {path}: {x.shape}")
        parts.append(x[:, -1, :, 0])
        if smoke_samples > 0 and sum(len(part) for part in parts) >= smoke_samples:
            break
    if not parts:
        raise FileNotFoundError(f"No METR-LA train/val/test npz files found in {input_dir}")
    speed = np.vstack(parts)
    if smoke_samples > 0:
        speed = speed[:smoke_samples]
    return speed


def _sensor_metadata(input_dir: Path, n_sensors: int) -> pd.DataFrame:
    path = input_dir / "graph_sensor_locations.csv"
    if not path.exists():
        return pd.DataFrame(
            {
                "sensor_id": [f"sensor_{i:03d}" for i in range(n_sensors)],
                "latitude": np.nan,
                "longitude": np.nan,
                "group": [str(i // 12) for i in range(n_sensors)],
            }
        )
    locations = pd.read_csv(path)
    locations = locations.iloc[:n_sensors].copy()
    locations["sensor_id"] = locations["sensor_id"].astype(str)
    lat_bin = np.floor((pd.to_numeric(locations["latitude"], errors="coerce") - 33.7) / 0.04)
    lon_bin = np.floor((pd.to_numeric(locations["longitude"], errors="coerce") + 118.7) / 0.04)
    locations["group"] = "grid_" + lat_bin.fillna(-1).astype(int).astype(str) + "_" + lon_bin.fillna(-1).astype(int).astype(str)
    return locations[["sensor_id", "latitude", "longitude", "group"]]


def _ctm_counterfactual(hourly: pd.DataFrame) -> pd.DataFrame:
    work = hourly.copy()
    baseline_end = work["hour_ts"].min() + pd.Timedelta(days=14)
    baseline = work.loc[work["hour_ts"] < baseline_end].groupby("sensor_id")["mean_speed"].quantile(0.95)
    fallback = float(work.loc[work["hour_ts"] < baseline_end, "mean_speed"].quantile(0.95))
    work["free_flow_speed"] = np.maximum(work["sensor_id"].map(baseline).fillna(fallback), 25.0)
    speed = np.clip(work["mean_speed"].to_numpy(dtype=float), 5.0, None)
    vf = work["free_flow_speed"].to_numpy(dtype=float)

    # Speed-calibrated one-step cell-transmission response. Density follows a
    # Greenshields inversion and the action removes at most ten percent of the
    # calibrated cell capacity during one five-minute control interval.
    jam_density = 150.0
    cell_length_km = 0.8
    dt_hours = 5.0 / 60.0
    density = jam_density * np.clip(1.0 - speed / vf, 0.0, 0.95)
    capacity_flow = vf * jam_density / 4.0
    speed_ratio = np.clip(speed / vf, 0.0, 1.0)
    work["network_headroom"] = pd.Series(speed_ratio, index=work.index).groupby(
        [work["group"], work["hour_ts"]]
    ).transform("mean")
    control_share = 0.03 + 0.15 * work["network_headroom"].to_numpy(dtype=float)
    controllable_outflow = np.minimum(control_share * capacity_flow, density * cell_length_km / dt_hours)
    density_action = np.maximum(density - controllable_outflow * dt_hours / cell_length_km, 0.0)
    action_speed = vf * np.maximum(1.0 - density_action / jam_density, 0.05)

    routine_loss = np.maximum(60.0 * (1.0 / speed - 1.0 / vf), 0.0)
    intervention_loss = np.maximum(60.0 * (1.0 / np.maximum(action_speed, 5.0) - 1.0 / vf), 0.0)
    simulated_gain = np.maximum(routine_loss - intervention_loss, 0.0)
    work["routine_loss"] = routine_loss
    work["intervention_loss"] = intervention_loss
    work["observed_or_simulated_gain"] = simulated_gain
    work["gain_upper"] = simulated_gain
    work["gain_lower"] = 0.70 * simulated_gain
    work["loss"] = routine_loss
    return work


def prepare_road(
    speed: np.ndarray,
    input_dir: Path,
    start: str,
    freq: str,
    period_days: int,
    peak_start_hour: int,
    peak_end_hour: int,
    include_weekends: bool,
) -> pd.DataFrame:
    if not 0 <= peak_start_hour < peak_end_hour <= 24:
        raise ValueError("Morning-peak hours must satisfy 0 <= start < end <= 24.")
    n_times, n_sensors = speed.shape
    timestamps = pd.date_range(start=start, periods=n_times, freq=freq)
    metadata = _sensor_metadata(input_dir, n_sensors)
    sensor_ids = metadata["sensor_id"].tolist()
    wide = pd.DataFrame(speed, index=timestamps, columns=sensor_ids)
    wide = wide.replace(0.0, np.nan).interpolate(limit_direction="both")
    long = wide.stack().rename("speed").reset_index()
    long.columns = ["timestamp", "sensor_id", "speed"]
    long["hour_ts"] = long["timestamp"].dt.floor("h")
    hourly = long.groupby(["sensor_id", "hour_ts"], as_index=False).agg(
        mean_speed=("speed", "mean"),
        min_speed=("speed", "min"),
        speed_std=("speed", "std"),
    )
    hourly["speed_std"] = hourly["speed_std"].fillna(0.0)
    hourly = hourly.merge(metadata, on="sensor_id", how="left")
    hourly = _ctm_counterfactual(hourly)
    group_history = hourly[["group", "hour_ts", "network_headroom"]].drop_duplicates().sort_values(
        ["group", "hour_ts"]
    )
    group_history["lag_network_headroom"] = group_history.groupby("group")["network_headroom"].shift(1)
    group_history["lag_network_headroom"] = group_history["lag_network_headroom"].fillna(1.0)
    hourly = hourly.merge(
        group_history[["group", "hour_ts", "lag_network_headroom"]],
        on=["group", "hour_ts"],
        how="left",
    )
    elapsed_days = (hourly["hour_ts"] - pd.Timestamp(start)).dt.days
    hourly["period"] = "road_" + (elapsed_days // max(period_days, 1) + 1).astype(int).astype(str).str.zfill(2)
    hourly["period_id"] = hourly["period"]
    hourly["block_id"] = hourly["hour_ts"].dt.strftime("%Y-%m-%d")
    hourly["hour"] = hourly["hour_ts"].dt.hour
    hourly["day_of_week"] = hourly["hour_ts"].dt.dayofweek
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    hourly["lag_speed"] = hourly.groupby("sensor_id")["mean_speed"].shift(1)
    hourly["lag_speed"] = hourly["lag_speed"].fillna(hourly["free_flow_speed"])
    hourly["group_id"] = hourly["group"]
    hourly["pickup_zone"] = hourly["sensor_id"]
    hourly["dropoff_zone"] = hourly["sensor_id"]
    hourly["od"] = hourly["sensor_id"]
    hourly["unit_id"] = hourly["sensor_id"] + "_" + hourly["hour_ts"].astype(str)
    hourly["dataset"] = "road"
    hourly["domain"] = "freeway_bottleneck_control"
    hourly["budget_family"] = "cardinality_partition"
    hourly["trip_count"] = 0.0
    hourly["mean_duration"] = 0.0
    hourly["p90_duration"] = 0.0
    hourly["distance_mean"] = 0.8
    peak_mask = (hourly["hour"] >= peak_start_hour) & (hourly["hour"] < peak_end_hour)
    if not include_weekends:
        peak_mask &= hourly["day_of_week"] < 5
    hourly = hourly.loc[peak_mask].copy()
    hourly["morning_peak_start_hour"] = peak_start_hour
    hourly["morning_peak_end_hour"] = peak_end_hour
    return hourly


def main() -> None:
    args = parse_args()
    speed = load_sequential_speed(args.input_dir, args.smoke_samples)
    ready = prepare_road(
        speed,
        args.input_dir,
        args.start,
        args.freq,
        args.period_days,
        args.peak_start_hour,
        args.peak_end_hour,
        args.include_weekends,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(args.output, index=False)
    print(f"saved {args.output} rows={len(ready):,} periods={ready['period'].nunique()}")


if __name__ == "__main__":
    main()
