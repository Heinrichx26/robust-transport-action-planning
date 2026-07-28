"""Pre-analysis checks for the public-data structural replay inputs.

The checks use only quantities observed before the decision and physical
identities used by the replay. They document boundary compliance and expose
records that would make a structural calculation impossible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_SRC = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from methodology.structural_robust import scenario_gain_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate structural replay boundaries and identities.")
    parser.add_argument("--dataset", choices=["bts", "road", "divvy"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def validate(dataset: str, frame: pd.DataFrame) -> dict[str, float | str]:
    if dataset == "road":
        frame = frame.loc[
            pd.to_numeric(frame["is_weekend"], errors="coerce").fillna(1).eq(0)
            & pd.to_numeric(frame["hour"], errors="coerce").between(6, 9)
        ].copy()
    names, values = scenario_gain_matrix(dataset, frame)
    loss = pd.to_numeric(frame["loss"], errors="coerce").to_numpy(float)
    rows: dict[str, float | str] = {
        "dataset": dataset,
        "rows": float(len(frame)),
        "scenario_count": float(len(names)),
        "negative_response_rate": float(np.mean(values < -1e-10)) if len(values) else 0.0,
        "above_loss_rate": float(np.mean(values > loss[:, None] + 1e-10)) if len(values) else 0.0,
        "nonfinite_response_rate": float(np.mean(~np.isfinite(values))) if len(values) else 0.0,
    }
    if dataset == "bts":
        reliability = pd.to_numeric(frame["recovery_reliability"], errors="coerce").to_numpy(float)
        rows["invalid_reliability_rate"] = float(np.mean((reliability < 0) | (reliability > 1)))
        rows["negative_slack_rate"] = float(np.mean(pd.to_numeric(frame["schedule_slack"], errors="coerce") < 0))
    elif dataset == "road":
        speed = pd.to_numeric(frame["mean_speed"], errors="coerce").to_numpy(float)
        free_flow = pd.to_numeric(frame["free_flow_speed"], errors="coerce").to_numpy(float)
        rows["invalid_speed_rate"] = float(np.mean((speed < 0) | (free_flow <= 0)))
        rows["headroom_outside_unit_rate"] = float(
            np.mean(~pd.to_numeric(frame["network_headroom"], errors="coerce").between(0, 1))
        )
    else:
        capacity = pd.to_numeric(frame["capacity_proxy"], errors="coerce").to_numpy(float)
        inventory = pd.to_numeric(frame["inventory_start"], errors="coerce").to_numpy(float)
        rows["inventory_outside_capacity_rate"] = float(np.mean((inventory < 0) | (inventory > capacity)))
        rows["invalid_relocation_limit_rate"] = float(
            np.mean(pd.to_numeric(frame["relocation_limit"], errors="coerce").to_numpy(float) < 0)
        )
    return rows


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    if args.max_rows > 0:
        frame = frame.head(args.max_rows).copy()
    result = pd.DataFrame([validate(args.dataset, frame)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
