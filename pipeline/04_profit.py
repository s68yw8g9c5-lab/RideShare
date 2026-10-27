"""Turn pay into profit per hour for a driver working each zone and hour.

  profit per hour = (driver pay + tips - cost of every mile driven)
                    / (time with a rider + time without one)

The trip data has no driver ID, so the time and miles without a rider are estimated,
under three sets of assumptions (best, mid, worst case) so no single guess drives a finding:
  - Manhattan: a share of a driver's time has no rider (50%, 60%, 70%). Half of that
    empty time is spent driving at the zone's usual speed, half parked.
  - Airports: each pickup costs a wait in the taxi lot (30, 60, 90 minutes),
    plus about 2 miles from the lot to the terminal.
Cost per mile is the IRS business rate for that year (gas, wear, insurance, depreciation).
A gas-only view is also kept, for drivers who only count fuel.

Usage:  python pipeline/04_profit.py
Output: data/clean/model_table.parquet  (features + profit per hour, one row per zone per hour)
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("data/clean")
AIRPORTS = (132, 138)
IRS_RATE = {2024: 0.67, 2025: 0.70, 2026: 0.725}  # dollars per mile
MPG = 30                                           # for the gas-only view
EMPTY_MOVING = 0.5                                 # share of empty time spent driving
AIRPORT_LOT_MILES = 2
SCENARIOS = {  # name: (Manhattan share of time with no rider, airport wait in minutes)
    "best": (0.5, 30),
    "mid": (0.6, 60),
    "worst": (0.7, 90),
}


def add_profit(df, name, empty_share, airport_wait_min, cost_per_mile):
    airport = df["zone"].isin(AIRPORTS)
    speed = df["trip_miles"] / df["trip_hours"]
    empty_hours = np.where(airport, df["trips"] * airport_wait_min / 60,
                           df["trip_hours"] * empty_share / (1 - empty_share))
    empty_miles = np.where(airport, df["trips"] * AIRPORT_LOT_MILES,
                           empty_hours * EMPTY_MOVING * speed)
    hours = df["trip_hours"] + empty_hours
    profit = df["driver_pay"] + df["tips"] - cost_per_mile * (df["trip_miles"] + empty_miles)
    df[f"profit_hr_{name}"] = profit / hours
    return df


def main():
    df = pd.read_parquet(CLEAN / "zone_hour_features.parquet")
    df["earn_per_rider_hr"] = (df["driver_pay"] + df["tips"]) / df["trip_hours"]
    full_cost = df["year"].map(IRS_RATE)
    for name, (share, wait) in SCENARIOS.items():
        add_profit(df, name, share, wait, full_cost)
    add_profit(df, "mid_gas_only", *SCENARIOS["mid"], df["gas_price"] / MPG)
    # with only a handful of trips, one odd ride swings the hourly number wildly
    df["enough_trips"] = (df["trips"] >= 5).astype(int)
    df.to_parquet(CLEAN / "model_table.parquet", index=False, compression="zstd")

    # sanity check: hour-weighted averages, so a busy hour counts more than a dead one
    df["is_airport"] = df["zone"].isin(AIRPORTS).map({True: "airport", False: "manhattan"})
    cols = ["profit_hr_best", "profit_hr_mid", "profit_hr_worst", "profit_hr_mid_gas_only"]
    df = df[df["enough_trips"] == 1]
    w = df["trip_hours"]
    print("Average profit per hour (dollars):")
    print(df[cols].mul(w, axis=0).groupby(df["is_airport"]).sum()
            .div(w.groupby(df["is_airport"]).sum(), axis=0).round(2).to_string())


if __name__ == "__main__":
    main()
