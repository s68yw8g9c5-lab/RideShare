"""Hourly weather for the three NYC reference points, from Open-Meteo:
  actual   = what happened (archive API)
  forecast = what the forecast said the day before (Previous Runs API),
             used for the honest "plan tomorrow" backtest.

Usage:  python pipeline/02_weather.py 2024-01-01 2026-08-31
Output: data/clean/weather_actual.parquet, data/clean/weather_forecast_d1.parquet
"""
import sys
from pathlib import Path

import pandas as pd
import requests

STATIONS = {  # name: (lat, lon)
    "central_park": (40.7789, -73.9692),
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
}
VARS = ["temperature_2m", "precipitation", "rain", "snowfall", "wind_speed_10m"]
OUT = Path("data/clean")


def fetch(url, params):
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"])


def main():
    start, end = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(start_date=start, end_date=end, timezone="America/New_York",
                  temperature_unit="fahrenheit", precipitation_unit="inch",
                  wind_speed_unit="mph")
    actual, forecast = [], []
    for name, (lat, lon) in STATIONS.items():
        loc = dict(latitude=lat, longitude=lon, **common)
        a = fetch("https://archive-api.open-meteo.com/v1/archive",
                  {**loc, "hourly": ",".join(VARS)})
        f = fetch("https://previous-runs-api.open-meteo.com/v1/forecast",
                  {**loc, "hourly": ",".join(f"{v}_previous_day1" for v in VARS)})
        f.columns = [c.replace("_previous_day1", "") for c in f.columns]
        actual.append(a.assign(station=name))
        forecast.append(f.assign(station=name))
        print(f"{name}: {len(a):,} hours")
    for frames, fname in [(actual, "weather_actual"), (forecast, "weather_forecast_d1")]:
        df = pd.concat(frames)
        df["time"] = pd.to_datetime(df["time"])
        df.to_parquet(OUT / f"{fname}.parquet", index=False)


if __name__ == "__main__":
    main()
