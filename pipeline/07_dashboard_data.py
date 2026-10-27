"""Export the small summary tables behind the online dashboard.

Usage:  python pipeline/07_dashboard_data.py   (after 05 and 06)
Output: reports/dashboard/*.csv and zones.json
"""
import json
from importlib import import_module
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

viz = import_module("05_explore")
OUT = Path("reports/dashboard")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = viz.load()
    man = df[~df["zone"].isin(viz.AIRPORTS)].copy()
    names = pd.read_csv("data/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]

    # zones, with simplified outlines in longitude / latitude
    by_zone = viz.avg(man, ["zone"]).rename("profit_hr")
    trips = man.groupby("zone")["trips"].sum()
    shapes = gpd.read_file("zip://data/raw/taxi_zones.zip!taxi_zones/taxi_zones.shp")
    shapes = shapes[shapes["borough"] == "Manhattan"].copy()
    shapes["geometry"] = shapes.geometry.simplify(40).to_crs(4326)
    rows = []
    for _, r in shapes.iterrows():
        z = r["LocationID"]
        geom = json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"]
        geom["coordinates"] = json.loads(json.dumps(geom["coordinates"]),
                                         parse_float=lambda x: round(float(x), 5))
        rows.append({"zone_id": int(z), "zone": names.get(z, r["zone"]),
                     "profit_hr": round(float(by_zone[z]), 2) if z in by_zone else None,
                     "trips": int(trips.get(z, 0)), "geometry": geom})
    (OUT / "zones.json").write_text(json.dumps(rows))

    grid = viz.avg(man, ["day_of_week", "hour_of_day"]).rename("profit_hr").reset_index()
    grid["day"] = grid["day_of_week"].map(dict(enumerate(DAYS)))
    grid.round(2)[["day", "day_of_week", "hour_of_day", "profit_hr"]].to_csv(
        OUT / "day_hour.csv", index=False)

    man["weather"] = np.select([man["is_snow"] == 1, man["is_rain"] == 1], ["Snow", "Rain"], "Dry")
    w = viz.avg(man, ["weather"]).rename("profit_hr").reindex(["Dry", "Rain", "Snow"])
    share = man.groupby("weather")["driver_hours"].sum() / man["driver_hours"].sum()
    pd.DataFrame({"weather": w.index, "profit_hr": w.values.round(2),
                  "share_of_hours": share.reindex(w.index).values.round(4)}).to_csv(
        OUT / "weather.csv", index=False)

    a = df[df["zone"].isin(viz.AIRPORTS)]
    profit = ((a["driver_pay"] + a["tips"] - a["year"].map(viz.IRS_RATE)
               * (a["trip_miles"] + viz.AIRPORT_LOT_MILES * a["trips"])) * a["cpi_factor"]).sum()
    waits = np.arange(0, 181, 5)
    curve = profit / (a["trip_hours"].sum() + a["trips"].sum() * waits / 60)
    manhattan = float((man["profit_hr_busy_mid"] * man["driver_hours"]).sum() / man["driver_hours"].sum())
    pd.DataFrame({"wait_min": waits, "airport_profit_hr": curve.round(2),
                  "manhattan_profit_hr": round(manhattan, 2)}).to_csv(
        OUT / "airport_curve.csv", index=False)

    even = 60 * (profit / manhattan - a["trip_hours"].sum()) / a["trips"].sum()
    airports = viz.avg(a, ["zone"])
    summary = [
        ("manhattan_profit_hr", round(manhattan, 2)),
        ("airport_breakeven_min", round(float(even))),
        ("jfk_profit_hr", round(float(airports[132]), 2)),
        ("lga_profit_hr", round(float(airports[138]), 2)),
        ("rain_lift_pct", round(float(w["Rain"] / w["Dry"] - 1), 4)),
        ("trips", int(df["trips"].sum())),
    ]
    reg_summary = Path("reports/regression_summary.csv")
    if reg_summary.exists():
        summary += [tuple(r) for r in pd.read_csv(reg_summary).itertuples(index=False)]
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(OUT / "summary.csv", index=False)

    reg = Path("reports/rain_by_time.csv")
    if reg.exists():
        pd.read_csv(reg).to_csv(OUT / "rain_by_time.csv", index=False)
    for f in sorted(OUT.iterdir()):
        print(f, f"{f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
