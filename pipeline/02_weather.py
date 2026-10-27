"""Hourly weather for the three NYC reference points:
  actual   = what happened: official hourly readings from the National Weather
             Service stations at Central Park, JFK and LaGuardia (via Iowa
             Environmental Mesonet, free, no daily limit)
  forecast = what the forecast said the day before (Open-Meteo Previous Runs API),
             used for the honest "plan tomorrow" backtest.

Usage:  python pipeline/02_weather.py 2024-01-01 2026-08-31
Output: data/clean/weather_actual.parquet, data/clean/weather_forecast_d1.parquet

Each file is saved on its own and skipped if it already exists, so if Open-Meteo's
free daily limit is hit (it is shared by everyone on the same network), just run
the script again the next day.
"""
import io
import sys
import time
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
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
ASOS_STATIONS = {"NYC": "central_park", "JFK": "jfk", "LGA": "lga"}
FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def fetch(url, params, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429 and "Daily" not in r.text and i < tries - 1:
            time.sleep(65)  # per-minute limit: wait it out
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        return pd.DataFrame(r.json()["hourly"])


def year_chunks(start, end):
    """Split the date range into calendar years to keep each request small."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for y in range(s.year, e.year + 1):
        yield (max(s, pd.Timestamp(f"{y}-01-01")).date().isoformat(),
               min(e, pd.Timestamp(f"{y}-12-31")).date().isoformat())


def download(url, suffix, start, end):
    common = dict(timezone="America/New_York", temperature_unit="fahrenheit",
                  precipitation_unit="inch", wind_speed_unit="mph",
                  hourly=",".join(v + suffix for v in VARS))
    frames = []
    cache = OUT / "weather_parts"  # each piece saved, so a retry picks up where it stopped
    cache.mkdir(exist_ok=True)
    for name, (lat, lon) in STATIONS.items():
        for s, e in year_chunks(start, end):
            part = cache / f"{name}{suffix}_{s}_{e}.parquet"
            if part.exists():
                frames.append(pd.read_parquet(part))
                continue
            df = fetch(url, dict(latitude=lat, longitude=lon, start_date=s,
                                 end_date=e, **common))
            df.columns = [c.replace(suffix, "") if suffix else c for c in df.columns]
            df = df.assign(station=name)
            df.to_parquet(part, index=False)
            frames.append(df)
            print(f"  {name} {s[:4]}: {len(df):,} hours", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def download_actual(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    params = [("station", st) for st in ASOS_STATIONS] + [
        ("data", d) for d in ["tmpf", "p01i", "sknt", "wxcodes"]] + [
        ("year1", s.year), ("month1", s.month), ("day1", s.day),
        ("year2", e.year), ("month2", e.month), ("day2", e.day),
        ("tz", "America/New_York"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "empty"), ("trace", "0.0001"), ("report_type", "3")]  # 3 = routine hourly
    r = requests.get(ASOS_URL, params=params, timeout=600)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text), parse_dates=["valid"])
    wx = raw["wxcodes"].fillna("")
    df = pd.DataFrame({
        "station": raw["station"].map(ASOS_STATIONS),
        # a reading at 3:51 describes the 3:00-3:59 hour
        "time": raw["valid"].dt.floor("h"),
        "temperature_2m": raw["tmpf"],
        "precipitation": raw["p01i"],                       # inches in the past hour
        "wind_speed_10m": raw["sknt"] * 1.15078,             # knots to mph
        "is_rain": wx.str.contains("RA|DZ").astype(int),
        "is_snow": wx.str.contains("SN").astype(int),
    })
    df = df.groupby(["station", "time"], as_index=False).last()
    for name, g in df.groupby("station"):
        print(f"  {name}: {len(g):,} hours", flush=True)
    return df


def main():
    start, end = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {"weather_actual": lambda: download_actual(start, end),
            "weather_forecast_d1": lambda: download(FORECAST_URL, "_previous_day1", start, end)}
    for fname, job in jobs.items():
        out = OUT / f"{fname}.parquet"
        if out.exists():
            print(f"{fname}: already done")
            continue
        print(f"{fname}: downloading...")
        try:
            job().to_parquet(out, index=False)
        except (RuntimeError, requests.RequestException) as err:
            print(f"{fname}: FAILED, try again later ({err})")


if __name__ == "__main__":
    main()
