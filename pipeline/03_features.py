"""Join everything a model needs onto the trip summaries: weather (what happened and
what was forecast the day before), calendar, holidays and the weekly gas price.

Uber and Lyft are added together, since a driver can work for both.
Each zone uses its nearest weather station: JFK and LaGuardia use their own airport
readings, every Manhattan zone uses Central Park.

Usage:  python pipeline/03_features.py
Output: data/clean/zone_hour_features.parquet  (one row per zone per hour)
"""
from pathlib import Path

import duckdb
import holidays
import pandas as pd
import requests

CLEAN = Path("data/clean")
GAS_URL = "https://www.eia.gov/dnav/pet/hist_xls/EMM_EPMR_PTE_Y35NY_DPGw.xls"  # NYC regular, weekly
STATION_OF_ZONE = {132: "jfk", 138: "lga"}  # everything else: central_park


def gas_prices():
    raw = Path("data/raw/gas_nyc_weekly.xls")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(requests.get(GAS_URL, timeout=60).content)
    gas = pd.read_excel(raw, sheet_name="Data 1", skiprows=2)
    gas.columns = ["week", "gas_price"]
    return gas.dropna()


def main():
    trips = duckdb.sql("""
        SELECT zone, hour,
               sum(trips) AS trips, sum(driver_pay) AS driver_pay, sum(tips) AS tips,
               sum(trip_miles) AS trip_miles, sum(trip_hours) AS trip_hours,
               sum(avg_wait_min * trips) / sum(trips) AS avg_wait_min,
               sum(rider_fare) AS rider_fare
        FROM 'data/clean/zone_hour/*.parquet'
        GROUP BY ALL ORDER BY zone, hour
    """).df()
    trips["station"] = trips["zone"].map(STATION_OF_ZONE).fillna("central_park")

    actual = pd.read_parquet(CLEAN / "weather_actual.parquet").rename(columns={
        "temperature_2m": "temp_f", "precipitation": "precip_in", "wind_speed_10m": "wind_mph"})
    # Central Park's wind sensor is often out; borrow LaGuardia's reading for those hours
    lga_wind = actual.loc[actual["station"] == "lga"].set_index("time")["wind_mph"]
    gap = actual["wind_mph"].isna()
    actual.loc[gap, "wind_mph"] = actual.loc[gap, "time"].map(lga_wind)
    forecast = pd.read_parquet(CLEAN / "weather_forecast_d1.parquet").rename(columns={
        "temperature_2m": "fc_temp_f", "precipitation": "fc_precip_in",
        "snowfall": "fc_snow_in", "wind_speed_10m": "fc_wind_mph"}).drop(columns="rain")  # rain is often blank; precip covers it
    df = (trips.merge(actual, left_on=["station", "hour"], right_on=["station", "time"], how="left")
               .drop(columns="time")
               .merge(forecast, left_on=["station", "hour"], right_on=["station", "time"], how="left")
               .drop(columns="time"))

    # calendar
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek  # 0 = Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["hour"].dt.month
    df["year"] = df["hour"].dt.year
    us_ny = holidays.US(state="NY", years=range(2024, 2027))
    df["is_holiday"] = df["hour"].dt.date.map(lambda d: d in us_ny).astype(int)

    # gas: the price from the most recent weekly survey on or before that hour
    gas = gas_prices()
    df = pd.merge_asof(df.sort_values("hour"), gas.sort_values("week"),
                       left_on="hour", right_on="week").drop(columns="week")

    df = df.sort_values(["zone", "hour"]).reset_index(drop=True)
    df.to_parquet(CLEAN / "zone_hour_features.parquet", index=False, compression="zstd")
    print(f"{len(df):,} zone-hours, {df['zone'].nunique()} zones")
    print("share missing:\n", df.isna().mean().round(3)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
