from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Divvy trip records for C-CERTS station rebalancing.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/open/divvy"))
    parser.add_argument("--output", type=Path, default=Path("data/open/divvy/ccerts_divvy_ready.csv"))
    parser.add_argument("--months", nargs="*", default=["202401", "202402", "202403", "202404", "202405", "202406"])
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce").to_numpy(dtype=float))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce").to_numpy(dtype=float))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce").to_numpy(dtype=float))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce").to_numpy(dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0, None)))


def read_month(path: Path, chunksize: int) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        if not csv_names:
            raise ValueError(f"No CSV found in {path}")
        with zf.open(csv_names[0]) as handle:
            for chunk in pd.read_csv(handle, chunksize=chunksize):
                required = [
                    "started_at",
                    "ended_at",
                    "start_station_id",
                    "end_station_id",
                    "start_lat",
                    "start_lng",
                    "end_lat",
                    "end_lng",
                ]
                chunk = chunk.dropna(subset=required).copy()
                if chunk.empty:
                    continue
                chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
                chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")
                chunk = chunk.dropna(subset=["started_at", "ended_at"])
                if chunk.empty:
                    continue
                chunk["duration_min"] = (chunk["ended_at"] - chunk["started_at"]).dt.total_seconds() / 60.0
                chunk = chunk[(chunk["duration_min"] > 1.0) & (chunk["duration_min"] <= 180.0)].copy()
                if chunk.empty:
                    continue
                chunk["distance_km"] = haversine_km(
                    chunk["start_lat"],
                    chunk["start_lng"],
                    chunk["end_lat"],
                    chunk["end_lng"],
                )
                chunk["hour_ts"] = chunk["started_at"].dt.floor("h")
                start = (
                    chunk.groupby(["start_station_id", "hour_ts"], as_index=False)
                    .agg(
                        departures=("ride_id", "count"),
                        mean_duration=("duration_min", "mean"),
                        p90_duration=("duration_min", lambda x: float(np.quantile(x, 0.9))),
                        distance_mean=("distance_km", "mean"),
                        lat=("start_lat", "mean"),
                        lng=("start_lng", "mean"),
                    )
                    .rename(columns={"start_station_id": "station_id"})
                )
                end = (
                    chunk.groupby(["end_station_id", "hour_ts"], as_index=False)
                    .agg(arrivals=("ride_id", "count"))
                    .rename(columns={"end_station_id": "station_id"})
                )
                rows.append(start.merge(end, on=["station_id", "hour_ts"], how="outer"))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby(["station_id", "hour_ts"], as_index=False).agg(
        departures=("departures", "sum"),
        arrivals=("arrivals", "sum"),
        mean_duration=("mean_duration", "mean"),
        p90_duration=("p90_duration", "mean"),
        distance_mean=("distance_mean", "mean"),
        lat=("lat", "mean"),
        lng=("lng", "mean"),
    )
    return out


def station_group(lat: float, lng: float) -> str:
    if not math.isfinite(lat) or not math.isfinite(lng):
        return "unknown"
    lat_bin = int(math.floor((lat - 41.6) / 0.05))
    lng_bin = int(math.floor((lng + 87.9) / 0.05))
    return f"grid_{lat_bin}_{lng_bin}"


def _hour_loss(inventory: float, capacity: float, departures: float, arrivals: float) -> tuple[float, float]:
    """Return unmet demand and end inventory for one station-hour."""
    served_departures = min(max(inventory, 0.0), max(departures, 0.0))
    after_departures = inventory - served_departures
    unmet_departures = max(departures - served_departures, 0.0)
    free_docks = max(capacity - after_departures, 0.0)
    served_arrivals = min(free_docks, max(arrivals, 0.0))
    unmet_arrivals = max(arrivals - served_arrivals, 0.0)
    return unmet_departures + unmet_arrivals, after_departures + served_arrivals


def _best_one_hour_action(
    inventory: float,
    capacity: float,
    departures: float,
    arrivals: float,
    relocation_limit: float,
) -> tuple[float, float]:
    """Evaluate a bounded pre-hour relocation and return its loss and chosen inventory."""
    lower = max(0.0, inventory - relocation_limit)
    upper = min(capacity, inventory + relocation_limit)
    candidates = {
        lower,
        upper,
        min(max(inventory, lower), upper),
        min(max(departures, lower), upper),
        min(max(capacity - arrivals + departures, lower), upper),
    }
    best_loss = float("inf")
    best_inventory = inventory
    for candidate in candidates:
        loss, _ = _hour_loss(candidate, capacity, departures, arrivals)
        movement = abs(candidate - inventory)
        incumbent_movement = abs(best_inventory - inventory)
        if loss < best_loss - 1e-12 or (abs(loss - best_loss) <= 1e-12 and movement < incumbent_movement):
            best_loss = loss
            best_inventory = candidate
    return best_loss, best_inventory


def simulate_station_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct the public-demand inventory counterfactual used by the station domain."""
    work = frame.copy()
    work["service_day"] = pd.to_datetime(work["hour_ts"]).dt.strftime("%Y-%m-%d")
    first_month = pd.to_datetime(work["hour_ts"]).min().to_period("M").to_timestamp()
    reference_end = first_month + pd.offsets.MonthBegin(1)
    reference = work.loc[pd.to_datetime(work["hour_ts"]) < reference_end].copy()
    if reference.empty:
        reference = work.copy()
    station_turnover = reference.groupby("station_id")["trip_count"].quantile(0.90)
    fallback_turnover = float(reference["trip_count"].quantile(0.90))
    fallback_capacity = float(np.ceil(max(8.0, 1.6 * fallback_turnover)))
    station_capacity = np.ceil(np.maximum(8.0, 1.6 * station_turnover)).clip(upper=80.0)
    work["capacity_proxy"] = work["station_id"].map(station_capacity).fillna(fallback_capacity).astype(float)
    work["relocation_limit"] = np.maximum(2.0, np.ceil(0.20 * work["capacity_proxy"])).clip(upper=10.0)

    records: list[dict[str, float | str]] = []
    for (_, _), day in work.groupby(["station_id", "service_day"], sort=False):
        day = day.sort_values("hour_ts")
        capacity = float(day["capacity_proxy"].iloc[0])
        inventory = 0.5 * capacity
        for idx, row in day.iterrows():
            departures = float(row["departures"])
            arrivals = float(row["arrivals"])
            inventory_start = inventory
            routine_loss, inventory = _hour_loss(inventory_start, capacity, departures, arrivals)
            action_loss, action_inventory = _best_one_hour_action(
                inventory_start,
                capacity,
                departures,
                arrivals,
                float(row["relocation_limit"]),
            )
            simulated_gain = max(routine_loss - action_loss, 0.0)
            records.append(
                {
                    "_index": idx,
                    "inventory_start": inventory_start,
                    "routine_loss": routine_loss,
                    "intervention_loss": action_loss,
                    "simulated_gain": simulated_gain,
                    "action_inventory": action_inventory,
                }
            )
    simulation = pd.DataFrame(records).set_index("_index")
    return work.join(simulation)


def main() -> None:
    args = parse_args()
    parts = []
    for month in args.months:
        path = args.input_dir / f"{month}-divvy-tripdata.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"preparing {path}")
        month_df = read_month(path, args.chunksize)
        if month_df.empty:
            continue
        month_df["period"] = f"{month[:4]}-{month[4:]}"
        parts.append(month_df)
    if not parts:
        raise ValueError("No Divvy rows prepared.")

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.groupby(["station_id", "hour_ts", "period"], as_index=False).agg(
        departures=("departures", "sum"),
        arrivals=("arrivals", "sum"),
        mean_duration=("mean_duration", "mean"),
        p90_duration=("p90_duration", "mean"),
        distance_mean=("distance_mean", "mean"),
        lat=("lat", "mean"),
        lng=("lng", "mean"),
    )
    frame["departures"] = pd.to_numeric(frame["departures"], errors="coerce").fillna(0.0)
    frame["arrivals"] = pd.to_numeric(frame["arrivals"], errors="coerce").fillna(0.0)
    frame["trip_count"] = frame["departures"] + frame["arrivals"]
    frame = frame[frame["trip_count"] > 0].copy()
    frame["imbalance"] = (frame["departures"] - frame["arrivals"]).abs()
    frame = simulate_station_inventory(frame)
    frame["loss"] = frame["routine_loss"]
    frame["gain_upper"] = frame["simulated_gain"]
    frame["gain_lower"] = 0.80 * frame["simulated_gain"]
    frame["hour"] = pd.to_datetime(frame["hour_ts"]).dt.hour
    frame["day_of_week"] = pd.to_datetime(frame["hour_ts"]).dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["block_id"] = frame["service_day"]
    frame["pickup_zone"] = frame["station_id"].astype(str)
    frame["dropoff_zone"] = frame["station_id"].astype(str)
    frame["od"] = frame["station_id"].astype(str)
    frame["group"] = [station_group(a, b) for a, b in zip(frame["lat"], frame["lng"])]
    frame["unit_id"] = frame["period"].astype(str) + "_" + frame["station_id"].astype(str) + "_" + frame["hour_ts"].astype(str)
    frame["dataset"] = "divvy"
    frame["domain"] = "station_rebalancing"
    frame["period_id"] = frame["period"]
    frame["group_id"] = frame["group"]
    frame["routine_loss"] = frame["loss"]
    frame["observed_or_simulated_gain"] = frame["simulated_gain"]
    frame["budget_family"] = "cardinality_partition"
    for col in ["mean_duration", "p90_duration", "distance_mean"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    keep = [
        "unit_id",
        "dataset",
        "domain",
        "period",
        "period_id",
        "block_id",
        "loss",
        "routine_loss",
        "gain_lower",
        "gain_upper",
        "observed_or_simulated_gain",
        "intervention_loss",
        "budget_family",
        "group",
        "group_id",
        "hour",
        "day_of_week",
        "is_weekend",
        "capacity_proxy",
        "relocation_limit",
        "inventory_start",
        "action_inventory",
        "departures",
        "arrivals",
        "trip_count",
        "mean_duration",
        "p90_duration",
        "distance_mean",
        "od",
        "pickup_zone",
        "dropoff_zone",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["loss", "gain_lower"]).to_csv(args.output, index=False)
    print(f"saved {args.output} rows={len(frame):,}")


if __name__ == "__main__":
    main()
