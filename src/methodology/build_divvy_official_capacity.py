"""Rebuild Divvy inventory replay with official station dock counts."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ccerts.prepare_divvy import _best_one_hour_action, _hour_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("data/open/divvy/ccerts_divvy_ready.csv"))
    parser.add_argument("--trip-dir", type=Path, default=Path("data/open/divvy"))
    parser.add_argument(
        "--station-file",
        type=Path,
        default=Path("data/shared_mobility/external/chicago_divvy_stations_20260728.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def extract_station_id(unit_id: str) -> str:
    match = re.match(r"^[^_]+_(.*?)_\d{4}-\d{2}-\d{2}", str(unit_id))
    return match.group(1).strip() if match else ""


def station_names(trip_dir: Path) -> pd.DataFrame:
    mappings = []
    for path in sorted(trip_dir.glob("2024??-divvy-tripdata.zip")):
        with zipfile.ZipFile(path) as archive:
            csv_name = next(
                name for name in archive.namelist()
                if name.endswith(".csv") and not name.startswith("__MACOSX")
            )
            with archive.open(csv_name) as handle:
                for chunk in pd.read_csv(
                    handle,
                    usecols=[
                        "start_station_id", "start_station_name",
                        "end_station_id", "end_station_name",
                    ],
                    chunksize=250_000,
                ):
                    start = chunk[["start_station_id", "start_station_name"]].rename(
                        columns={"start_station_id": "station_id", "start_station_name": "station_name"}
                    )
                    end = chunk[["end_station_id", "end_station_name"]].rename(
                        columns={"end_station_id": "station_id", "end_station_name": "station_name"}
                    )
                    mappings.extend([start, end])
        if mappings:
            current = pd.concat(mappings, ignore_index=True).dropna()
            current["station_id"] = current["station_id"].astype(str).str.strip()
            mappings = [current.drop_duplicates("station_id")]
    result = pd.concat(mappings, ignore_index=True).drop_duplicates("station_id")
    return result


def official_capacities(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(payload)
    frame["official_capacity"] = pd.to_numeric(frame["total_docks"], errors="coerce")
    frame = frame[["station_name", "official_capacity"]].dropna()
    frame = frame.loc[frame["official_capacity"] > 0].drop_duplicates("station_name")
    return frame


def replay(frame: pd.DataFrame, capacity_multiplier: float) -> pd.DataFrame:
    records = []
    for (_, _), day in frame.groupby(["station_id", "block_id"], sort=False):
        day = day.sort_values("hour_ts")
        capacity = float(day["official_capacity"].iloc[0]) * capacity_multiplier
        inventory = 0.5 * capacity
        relocation_limit = float(np.clip(np.ceil(0.20 * capacity), 1.0, 10.0))
        for index, row in day.iterrows():
            departures = float(row["departures"])
            arrivals = float(row["arrivals"])
            routine_loss, inventory_after = _hour_loss(
                inventory, capacity, departures, arrivals
            )
            action_loss, action_inventory = _best_one_hour_action(
                inventory, capacity, departures, arrivals, relocation_limit
            )
            gain = max(routine_loss - action_loss, 0.0)
            records.append(
                {
                    "index": index,
                    "capacity_proxy": capacity,
                    "inventory_start": inventory,
                    "relocation_limit": relocation_limit,
                    "routine_loss": routine_loss,
                    "loss": routine_loss,
                    "intervention_loss": action_loss,
                    "observed_or_simulated_gain": gain,
                    "gain_upper": gain,
                    "gain_lower": 0.80 * gain,
                    "action_inventory": action_inventory,
                }
            )
            inventory = inventory_after
    updated = pd.DataFrame(records).set_index("index")
    output = frame.copy()
    for column in updated.columns:
        output[column] = updated[column]
    output["capacity_source"] = "City of Chicago Divvy Bicycle Stations snapshot, 2026-07-28"
    output["capacity_multiplier"] = capacity_multiplier
    return output


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.prepared, low_memory=False)
    frame["station_id"] = frame["unit_id"].map(extract_station_id)
    frame["hour_ts"] = pd.to_datetime(frame["unit_id"].str[-19:], errors="coerce")
    frame = frame.merge(station_names(args.trip_dir), on="station_id", how="left")
    frame = frame.merge(official_capacities(args.station_file), on="station_name", how="left")
    frame = frame.dropna(subset=["hour_ts", "official_capacity"]).copy()
    if args.smoke:
        days = sorted(frame["block_id"].astype(str).unique())[:3]
        frame = frame.loc[frame["block_id"].astype(str).isin(days)].copy()
    if args.capacity_multiplier <= 0:
        raise ValueError("capacity-multiplier must be positive")
    output = replay(frame, args.capacity_multiplier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        {
            "rows": len(output),
            "stations": int(output["station_id"].nunique()),
            "days": int(output["block_id"].nunique()),
            "negative_gain": int((output["gain_lower"] < 0).sum()),
            "gain_above_loss": int((output["gain_lower"] > output["loss"] + 1e-9).sum()),
            "inventory_outside_capacity": int(
                (
                    (output["inventory_start"] < -1e-9)
                    | (output["inventory_start"] > output["capacity_proxy"] + 1e-9)
                ).sum()
            ),
        }
    )


if __name__ == "__main__":
    main()
