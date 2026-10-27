# Rideshare Profit Forecast (NYC)

**Question:** Where and when should a rideshare driver work tomorrow to make the most money, given the weather?

A weather-aware forecast of driver *profit* per hour (pay + tips minus driving costs and empty time),
by zone and hour, for Manhattan, JFK and LaGuardia. Built entirely on public data. Chicago follows as a
generalization test.

## Status
- [x] Plan and tools
- [ ] Trips summarized to zone x hour (`pipeline/01_trips.py`)
- [ ] Weather, actual and day-ahead forecast (`pipeline/02_weather.py`)
- [ ] Profit per hour, regression, random forest / XGBoost
- [ ] Forecast backtest on 2026
- [ ] Map dashboard and "plan tomorrow"

## Run it
```
pip install -r requirements.txt
python pipeline/01_trips.py 2024-01 2026-08
python pipeline/02_weather.py 2024-01-01 2026-08-31
```

## Data
NYC TLC High Volume FHV trip records, Open-Meteo archive and Previous Runs forecasts,
National Weather Service API, IRS 2026 mileage rate (72.5 cents/mile), EIA gas prices.
