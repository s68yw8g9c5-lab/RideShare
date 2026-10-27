# Rideshare Profit Forecast (NYC)

**Question:** Where and when should a rideshare driver work tomorrow to make the most money, given the weather?

A weather-aware forecast of driver *profit* per hour (pay + tips minus driving costs and empty time),
by zone and hour, for Manhattan, JFK and LaGuardia. Built entirely on public data. Chicago follows as a
generalization test.

## Status
- [x] Plan and tools
- [x] Trips summarized to zone x hour, Jan 2024 to Aug 2026 (`pipeline/01_trips.py`)
- [x] Weather, actual and day-ahead forecast (`pipeline/02_weather.py`)
- [x] Weather, calendar, holidays and gas joined (`pipeline/03_features.py`)
- [x] Profit per hour, best / mid / worst case (`pipeline/04_profit.py`)
- [ ] Regression, random forest / XGBoost
- [ ] Forecast backtest on 2026
- [ ] Map dashboard and "plan tomorrow"

## Run it
```
pip install -r requirements.txt
python pipeline/01_trips.py 2024-01 2026-08
python pipeline/02_weather.py 2024-01-01 2026-08-31
python pipeline/03_features.py
python pipeline/04_profit.py
```

## Data in this repo
- `data/clean/zone_hour/YYYY-MM.parquet`: one row per pickup zone, hour and company (HV0003 Uber,
  HV0005 Lyft) with trips, driver pay, tips, miles, time with a rider, speed, rider wait and fare.
  Manhattan's 67 zones plus JFK (132) and LaGuardia (138). About 266 million trips in total.
- `data/clean/weather_actual.parquet`: hourly NWS readings at Central Park, JFK and LaGuardia
  (temperature F, precipitation inches, wind mph, rain and snow flags).
- `data/clean/weather_forecast_d1.parquet`: what the forecast said the day before, for the same hours.
- `data/clean/model_table.parquet`: the table the models use. One row per zone per hour (Uber + Lyft
  combined) with weather, forecast, calendar, holiday, gas price and profit per hour under best, mid
  and worst case assumptions. Use rows with `enough_trips == 1`. Too big for GitHub, so rebuild it
  with steps 03 and 04 (under a minute).

## Sources
NYC TLC High Volume FHV trip records, NWS station readings (via Iowa Environmental Mesonet),
Open-Meteo Previous Runs forecasts,
National Weather Service API, IRS 2026 mileage rate (72.5 cents/mile), EIA gas prices.
