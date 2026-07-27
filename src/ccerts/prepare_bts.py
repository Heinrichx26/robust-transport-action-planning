from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BTS Airline On-Time Performance files for C-CERTS.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/open/bts"))
    parser.add_argument("--output", type=Path, default=Path("data/open/bts/ccerts_bts_ready.csv"))
    parser.add_argument("--min-year", type=int, default=2022)
    return parser.parse_args()


def _read_bts_files(input_dir: Path) -> pd.DataFrame:
    files = sorted(list(input_dir.glob("*.csv")) + list(input_dir.glob("*.csv.gz")) + list(input_dir.glob("*.zip")))
    if not files:
        raise FileNotFoundError(
            f"No BTS csv or zip files found in {input_dir}. Download open BTS On-Time Performance files first."
        )
    frames = []
    for path in files:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as fh:
                    frames.append(pd.read_csv(fh, low_memory=False))
        else:
            frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def _first_existing(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"None of the expected columns exists: {names}")


def _optional_numeric(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(frame), dtype=float)


def prepare_bts(frame: pd.DataFrame, min_year: int) -> pd.DataFrame:
    year_col = _first_existing(frame, ["Year", "YEAR"])
    month_col = _first_existing(frame, ["Month", "MONTH"])
    day_col = _first_existing(frame, ["DayofMonth", "DAY_OF_MONTH", "DayOfMonth"])
    carrier_col = _first_existing(frame, ["Reporting_Airline", "OP_UNIQUE_CARRIER", "UniqueCarrier"])
    origin_col = _first_existing(frame, ["Origin", "ORIGIN"])
    dest_col = _first_existing(frame, ["Dest", "DEST"])
    distance_col = _first_existing(frame, ["Distance", "DISTANCE"])
    sched_dep_col = _first_existing(frame, ["CRSDepTime", "CRS_DEP_TIME"])
    arr_delay_col = _first_existing(frame, ["ArrDelay", "ARR_DELAY"])
    dep_delay_col = _first_existing(frame, ["DepDelay", "DEP_DELAY"])
    crs_elapsed_col = _first_existing(frame, ["CRSElapsedTime", "CRS_ELAPSED_TIME"])
    actual_elapsed_col = _first_existing(frame, ["ActualElapsedTime", "ACTUAL_ELAPSED_TIME"])

    work = frame.copy()
    work = work[pd.to_numeric(work[year_col], errors="coerce") >= min_year].copy()
    for col in [distance_col, sched_dep_col, arr_delay_col, dep_delay_col, crs_elapsed_col, actual_elapsed_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[year_col, month_col, day_col, carrier_col, origin_col, dest_col, arr_delay_col])

    sched_dep = work[sched_dep_col].fillna(0).astype(int)
    dep_hour = np.clip(sched_dep // 100, 0, 23)
    period = (
        pd.to_numeric(work[year_col], errors="coerce").astype(int).astype(str)
        + "-"
        + pd.to_numeric(work[month_col], errors="coerce").astype(int).astype(str).str.zfill(2)
    )
    flight_day = (
        period
        + "-"
        + pd.to_numeric(work[day_col], errors="coerce").astype(int).astype(str).str.zfill(2)
    )

    loss = np.clip(work[arr_delay_col].to_numpy(dtype=float), 0.0, None)
    dep_loss = np.clip(work[dep_delay_col].fillna(0).to_numpy(dtype=float), 0.0, None)
    elapsed_recovery = np.clip(
        work[crs_elapsed_col].fillna(work[actual_elapsed_col]).to_numpy(dtype=float)
        - work[actual_elapsed_col].fillna(work[crs_elapsed_col]).to_numpy(dtype=float),
        0.0,
        None,
    )
    carrier_delay = _optional_numeric(work, ["CarrierDelay", "CARRIER_DELAY"])
    nas_delay = _optional_numeric(work, ["NASDelay", "NAS_DELAY"])
    late_aircraft_delay = _optional_numeric(work, ["LateAircraftDelay", "LATE_AIRCRAFT_DELAY"])
    actionable_delay = np.clip(carrier_delay + nas_delay + late_aircraft_delay, 0.0, None)

    scheduled_elapsed = work[crs_elapsed_col].fillna(work[crs_elapsed_col].median()).to_numpy(dtype=float)
    schedule_table = pd.DataFrame(
        {
            "carrier": work[carrier_col].astype(str).to_numpy(),
            "origin": work[origin_col].astype(str).to_numpy(),
            "destination": work[dest_col].astype(str).to_numpy(),
            "dep_hour": dep_hour.to_numpy(),
            "scheduled_elapsed": scheduled_elapsed,
        }
    )
    context_median = schedule_table.groupby(
        ["carrier", "origin", "destination", "dep_hour"]
    )["scheduled_elapsed"].transform("median")
    schedule_slack = np.clip(schedule_table["scheduled_elapsed"].to_numpy() - context_median.to_numpy(), 0.0, None)
    observed_absorption = np.clip(dep_loss - loss, 0.0, None) + elapsed_recovery
    recovery_reliability = np.clip(observed_absorption / (dep_loss + 5.0), 0.0, 1.0)
    gain_upper = np.minimum(loss, schedule_slack + 0.35 * actionable_delay)
    structural_gain = np.minimum(loss, 0.35 * schedule_slack + 0.15 * actionable_delay)
    gain_lower = structural_gain * (0.50 + 0.50 * recovery_reliability)

    ready = pd.DataFrame(
        {
            "unit_id": np.arange(len(work), dtype=int).astype(str),
            "dataset": "bts",
            "domain": "airline_delay_recovery",
            "period": period.to_numpy(),
            "period_id": period.to_numpy(),
            "flight_day": flight_day.to_numpy(),
            "block_id": flight_day.to_numpy(),
            "carrier": work[carrier_col].astype(str).to_numpy(),
            "origin": work[origin_col].astype(str).to_numpy(),
            "destination": work[dest_col].astype(str).to_numpy(),
            "group": work[origin_col].astype(str).to_numpy(),
            "dep_hour": dep_hour.to_numpy(),
            "distance": work[distance_col].fillna(work[distance_col].median()).to_numpy(dtype=float),
            "loss": loss,
            "routine_loss": loss,
            "gain_lower": gain_lower,
            "gain_upper": gain_upper,
            "observed_or_simulated_gain": gain_upper,
            "scheduled_elapsed": scheduled_elapsed,
            "schedule_slack": schedule_slack,
            "actionable_delay": actionable_delay,
            "recovery_reliability": recovery_reliability,
            "budget_family": "cardinality_partition",
        }
    )
    ready["route"] = ready["origin"] + "_" + ready["destination"]
    ready["group_id"] = ready["group"]
    return ready


def main() -> None:
    args = parse_args()
    frame = _read_bts_files(args.input_dir)
    ready = prepare_bts(frame, args.min_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ready.to_csv(args.output, index=False)
    print(f"Saved BTS C-CERTS-ready file: {args.output} ({len(ready):,} rows)")


if __name__ == "__main__":
    main()
