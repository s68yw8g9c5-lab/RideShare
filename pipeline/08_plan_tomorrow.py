"""Plan tomorrow: expected profit per hour for every zone and hour of the next driving day,
from the National Weather Service forecast.

Preview model: a regression on the last 12 months of data. Each zone gets its own baseline
for every day-of-week and hour, plus a month baseline, and weather moves it up or down
(light rain by time of day, heavy rain, snow, temperature, wind, holidays).
It also works out a "what if it rains" version of the same day, so a driver can see how a
change in the forecast would change the plan.

Usage:  python pipeline/08_plan_tomorrow.py [YYYY-MM-DD]
Output: reports/plan/plan_tomorrow.json
"""
import json
import sys
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd
import pyfixest as pf
import requests

viz = import_module("05_explore")
OUT = Path("reports/plan")
NY = ZoneInfo("America/New_York")
GRID_URL = "https://api.weather.gov/gridpoints/OKX/34,45"  # the NWS forecast grid square over Central Park
HEADERS = {"User-Agent": "rideshare-forecast student project (github.com/s68yw8g9c5-lab/RideShare)"}
EVENT_VENUES = [194]
BLOCKS = [("Morning rush", 0, 7, 9), ("Midday", 0, 10, 15), ("Evening rush", 0, 16, 19),
          ("Weekday night", 0, 20, 23), ("Overnight", 0, 0, 6), ("Weekend", 1, 0, 23)]
TEMPS = [("below 32F", -50, 32), ("32-50F", 32, 50), ("70-85F", 70, 85), ("above 85F", 85, 130)]
RHS = ("light_rain:C(block) + heavy_rain + is_snow + " + " + ".join(f"temp_{i}" for i in range(len(TEMPS)))
       + " + windy + is_holiday")


def block_of(hour, weekend):
    if weekend:
        return "Weekend"
    for name, _, h0, h1 in BLOCKS[:-1]:
        if h0 <= hour <= h1:
            return name


def add_terms(df):
    for i, (_, lo, hi) in enumerate(TEMPS):
        df[f"temp_{i}"] = ((df["temp_f"] >= lo) & (df["temp_f"] < hi)).astype(int)
    df["windy"] = (df["wind_mph"] >= 20).astype(int)
    df["block"] = [block_of(h, w) for h, w in zip(df["hour_of_day"], df["is_weekend"])]
    df["zone_dow_hour"] = df["zone"] * 1000 + df["day_of_week"] * 24 + df["hour_of_day"]
    return df


def fit():
    df = viz.load()
    df = df[df["hour"] >= df["hour"].max() - pd.DateOffset(months=12)]
    df = add_terms(df.dropna(subset=["temp_f", "wind_mph", "is_rain", "is_snow"]).copy())
    return pf.feols(f"profit_hr_busy_mid ~ {RHS} | zone_dow_hour + month", data=df,
                    weights="driver_hours", vcov={"CRV1": "zone"}), df


def nws_hours(day):
    """Hourly forecast for one day in New York time, from the NWS gridpoint data."""
    props = requests.get(GRID_URL, headers=HEADERS, timeout=60).json()["properties"]
    hours = pd.date_range(day, periods=24, freq="h", tz=NY)

    def spread(name, per_hour=False):
        s = pd.Series(np.nan, index=hours)
        for v in props[name]["values"]:
            start, dur = v["validTime"].split("/")
            start = pd.Timestamp(start).tz_convert(NY)
            n = int(pd.Timedelta(dur).total_seconds() // 3600)
            for k in range(n):
                t = start + pd.Timedelta(hours=k)
                if t in s.index and v["value"] is not None:
                    s[t] = v["value"] / n if per_hour else v["value"]
        return s

    wx = pd.Series("", index=hours)
    for v in props["weather"]["values"]:
        start, dur = v["validTime"].split("/")
        start = pd.Timestamp(start).tz_convert(NY)
        kinds = " ".join(w.get("weather") or "" for w in v["value"])
        for k in range(int(pd.Timedelta(dur).total_seconds() // 3600)):
            t = start + pd.Timedelta(hours=k)
            if t in wx.index:
                wx[t] = kinds
    f = pd.DataFrame({
        "temp_f": spread("temperature") * 9 / 5 + 32,
        "rain_chance": spread("probabilityOfPrecipitation"),
        "precip_in": spread("quantitativePrecipitation", per_hour=True) / 25.4,
        "snow_in": spread("snowfallAmount", per_hour=True) / 25.4,
        "wind_mph": spread("windSpeed") / 1.609,
        "weather": wx,
    })
    likely = f["rain_chance"] >= 50
    f["is_snow"] = (likely & (f["weather"].str.contains("snow") | (f["snow_in"] > 0))).astype(int)
    f["is_rain"] = (likely & ~f["is_snow"].astype(bool)
                    & (f["weather"].str.contains("rain|thunder|drizzle") | (f["precip_in"] > 0))).astype(int)
    f["heavy_rain"] = ((f["is_rain"] == 1) & (f["precip_in"] >= 0.1)).astype(int)
    f["light_rain"] = ((f["is_rain"] == 1) & (f["heavy_rain"] == 0)).astype(int)
    return f.reset_index(names="time")


def condition(r):
    if r.is_snow:
        return "Snow"
    if r.heavy_rain:
        return "Heavy rain"
    if r.is_rain:
        return "Rain"
    if r.rain_chance >= 30:
        return "Chance of rain"
    return "Dry"


def main():
    now = datetime.now(NY)
    if len(sys.argv) > 1:
        day = pd.Timestamp(sys.argv[1]).date()
    else:  # before 5 am, "tomorrow" still means the day that is starting
        day = (now if now.hour < 5 else now + timedelta(days=1)).date()
    OUT.mkdir(parents=True, exist_ok=True)

    model, hist = fit()
    fc = nws_hours(str(day))
    zones = sorted(hist["zone"].unique())
    names = pd.read_csv("data/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]
    weekend = int(pd.Timestamp(day).dayofweek >= 5)
    is_holiday = int(day in holidays.US(state="NY", years=day.year))

    grid = pd.DataFrame([(z, h) for z in zones for h in range(24)], columns=["zone", "hour_of_day"])
    grid = grid.merge(fc.assign(hour_of_day=fc["time"].dt.hour), on="hour_of_day")
    grid = grid.assign(day_of_week=pd.Timestamp(day).dayofweek, is_weekend=weekend,
                       is_holiday=is_holiday, month=pd.Timestamp(day).month)
    grid["temp_f"] = grid["temp_f"].fillna(hist["temp_f"].median())
    grid["wind_mph"] = grid["wind_mph"].fillna(0)
    grid = add_terms(grid)
    known = grid["zone_dow_hour"].isin(hist["zone_dow_hour"].unique())
    grid["forecast"] = np.nan
    grid.loc[known, "forecast"] = model.predict(grid[known])
    rainy = grid[known].assign(is_rain=1, light_rain=1, heavy_rain=0, is_snow=0)
    grid.loc[known, "if_rain"] = model.predict(rainy)

    zone_rows = []
    for z, g in grid.groupby("zone"):
        zone_rows.append({
            "zone_id": int(z), "zone": names.get(z, str(z)),
            "area": "Airport" if z in viz.AIRPORTS else "Manhattan",
            "pickable": z not in EVENT_VENUES,
            "forecast": [None if pd.isna(v) else round(float(v), 2) for v in g.sort_values("hour_of_day")["forecast"]],
            "if_rain": [None if pd.isna(v) else round(float(v), 2) for v in g.sort_values("hour_of_day")["if_rain"]],
        })
    hours = [{"hour": int(r.time.hour), "temp_f": None if pd.isna(r.temp_f) else round(float(r.temp_f)),
              "rain_chance": None if pd.isna(r.rain_chance) else int(r.rain_chance),
              "precip_in": None if pd.isna(r.precip_in) else round(float(r.precip_in), 3),
              "condition": condition(r)} for r in fc.itertuples()]
    plan = {
        "date": str(day), "weekday": pd.Timestamp(day).day_name(), "holiday": bool(is_holiday),
        "made_at": now.isoformat(timespec="minutes"), "weather_source": "National Weather Service, Central Park forecast",
        "dollars": "August 2026 dollars, after car costs (IRS mileage rate) and estimated empty time",
        "airport_wait_min": 60, "hours": hours, "zones": zone_rows,
    }
    (OUT / "plan_tomorrow.json").write_text(json.dumps(plan))

    pick = grid[known & ~grid["zone"].isin(EVENT_VENUES)]
    best = pick.loc[pick.groupby("hour_of_day")["forecast"].idxmax()]
    print(f"Plan for {plan['weekday']} {day}: " + ", ".join(f"{h['hour']}h {h['condition']}" for h in hours[::6]))
    for r in best.itertuples():
        print(f"  {r.hour_of_day:2d}:00  {names.get(r.zone, r.zone):30s} ${r.forecast:5.2f}  (if rain ${r.if_rain:5.2f})")


if __name__ == "__main__":
    main()
