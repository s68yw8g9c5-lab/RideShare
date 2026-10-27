"""Download NYC Uber/Lyft (HVFHV) trips one month at a time and shrink each
month to one row per pickup zone per hour, for Manhattan + JFK + LaGuardia.

Usage:  python pipeline/01_trips.py 2024-01 2026-08 [--keep-raw]
Output: data/clean/zone_hour/YYYY-MM.parquet  (a few MB per month)
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests

BASE = "https://d37ci6vzurychx.cloudfront.net"
RAW = Path("data/raw")
CLEAN = Path("data/clean/zone_hour")
AIRPORTS = (132, 138)  # JFK, LaGuardia


def months(start, end):
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def download(url, dest):
    if dest.exists():
        return
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def zone_ids():
    lookup = Path("data/taxi_zone_lookup.csv")
    download(f"{BASE}/misc/taxi_zone_lookup.csv", lookup)
    zones = pd.read_csv(lookup)
    manhattan = zones.loc[zones["Borough"] == "Manhattan", "LocationID"].tolist()
    return sorted(set(manhattan) | set(AIRPORTS))


SUMMARY_SQL = """
SELECT
    PULocationID                              AS zone,
    date_trunc('hour', pickup_datetime)       AS hour,
    hvfhs_license_num                         AS company,   -- HV0003 Uber, HV0005 Lyft
    count(*)                                  AS trips,
    sum(driver_pay)                           AS driver_pay,
    sum(tips)                                 AS tips,
    sum(trip_miles)                           AS trip_miles,
    sum(trip_time) / 3600.0                   AS trip_hours,
    avg(trip_miles / nullif(trip_time / 3600.0, 0)) AS avg_mph,
    avg(epoch(pickup_datetime - request_datetime) / 60.0) AS avg_wait_min,
    sum(base_passenger_fare)                  AS rider_fare
FROM read_parquet(?)
WHERE PULocationID IN (SELECT unnest(?::INTEGER[]))
  AND trip_time > 0 AND trip_miles > 0 AND driver_pay >= 0
GROUP BY ALL
"""


def summarize(raw_file, out_file, zones):
    con = duckdb.connect()
    df = con.execute(SUMMARY_SQL, [str(raw_file), zones]).df()
    df.sort_values(["zone", "hour", "company"]).to_parquet(out_file, index=False, compression="zstd")
    return len(df)


def main():
    start, end = sys.argv[1], sys.argv[2]
    keep_raw = "--keep-raw" in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    zones = zone_ids()
    for m in months(start, end):
        out = CLEAN / f"{m}.parquet"
        if out.exists():
            print(f"{m}: already done")
            continue
        raw = RAW / f"fhvhv_tripdata_{m}.parquet"
        print(f"{m}: downloading...", flush=True)
        download(f"{BASE}/trip-data/fhvhv_tripdata_{m}.parquet", raw)
        rows = summarize(raw, out, zones)
        print(f"{m}: {rows:,} zone-hour rows")
        if not keep_raw:
            raw.unlink()


if __name__ == "__main__":
    main()
