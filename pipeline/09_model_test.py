"""The accuracy test: which model plans a driver's day best?

Every model learns from 2024-2025 only. Then, for each day of January-August 2026, it
predicts profit per hour for every zone and hour using the weather forecast issued the
day before (what a driver would really have had), and we compare with what drivers
actually made.

Models:
  history only   each zone's average for that day of week and hour (no weather)
  regression     the same, moved by weather (light/heavy rain by time of day, snow,
                 temperature, wind, holidays)
  random forest  and XGBoost: tree models given the same facts, free to find their own patterns

Scores (weighted by driver hours, August 2026 dollars):
  typical miss   average gap between predicted and actual profit per hour
  hit rate       for each day and time of day, does the model's #1 zone land in the
                 actual top 3? (picking at random would hit about 4% of the time)
  dollars lost   actual profit of the best zone minus actual profit of the model's pick
  citywide miss  for all zones together, hour by hour: is this a good hour to drive at all?

Usage:  python pipeline/09_model_test.py
Output: reports/model_test.csv, reports/model_test.txt
"""
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

viz = import_module("05_explore")
plan = import_module("08_plan_tomorrow")
REPORTS = Path("reports")
EVENT_VENUES = [194]
PERIODS = {"Late 12-6am": range(0, 6), "Morning 6-10am": range(6, 10), "Midday 10am-4pm": range(10, 16),
           "Rush 4-8pm": range(16, 20), "Night 8pm-12am": range(20, 24)}
TREE_FEATURES = ["zone", "day_of_week", "hour_of_day", "month", "is_holiday", "temp_f", "precip_in",
                 "wind_mph", "is_rain", "light_rain", "heavy_rain", "is_snow", "in_congestion_zone"]


def load():
    df = viz.load()
    df = df[~df["zone"].isin(EVENT_VENUES)].dropna(subset=["temp_f", "wind_mph", "is_rain", "is_snow"]).copy()
    return plan.add_terms(df)


def with_forecast_weather(test):
    """Swap the actual weather for what was forecast the day before."""
    t = test.copy()
    t["temp_f"] = t["fc_temp_f"].fillna(t["temp_f"])
    t["wind_mph"] = t["fc_wind_mph"].fillna(t["wind_mph"])
    t["precip_in"] = t["fc_precip_in"].fillna(0)
    t["is_snow"] = (t["fc_snow_in"].fillna(0) > 0).astype(int)
    t["is_rain"] = ((t["precip_in"] >= 0.01) & (t["is_snow"] == 0)).astype(int)
    t["heavy_rain"] = ((t["is_rain"] == 1) & (t["precip_in"] >= 0.1)).astype(int)
    t["light_rain"] = ((t["is_rain"] == 1) & (t["heavy_rain"] == 0)).astype(int)
    return plan.add_terms(t)


def fit_models(train):
    y, w = train["profit_hr_busy_mid"], train["driver_hours"]
    models = {}
    hist = (train.assign(p=y * w).groupby("zone_dow_hour")["p"].sum() / w.groupby(train["zone_dow_hour"]).sum())
    models["History only"] = lambda d: d["zone_dow_hour"].map(hist).to_numpy()
    reg = pf.feols(f"profit_hr_busy_mid ~ {plan.RHS} | zone_dow_hour + month", data=train,
                   weights="driver_hours", vcov="iid")
    models["Regression"] = lambda d: reg.predict(d)
    sample = train.sample(n=min(500_000, len(train)), random_state=1)
    rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=20, max_features=0.6, n_jobs=-1, random_state=1)
    rf.fit(sample[TREE_FEATURES], sample["profit_hr_busy_mid"], sample_weight=sample["driver_hours"])
    models["Random forest"] = lambda d: rf.predict(d[TREE_FEATURES])
    xg = xgb.XGBRegressor(n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=1)
    xg.fit(train[TREE_FEATURES], y, sample_weight=w)
    models["XGBoost"] = lambda d: xg.predict(d[TREE_FEATURES])
    return models


def score(test, pred):
    t = test.assign(pred=pred).dropna(subset=["pred"])
    w = t["driver_hours"]
    miss = float(np.average((t["pred"] - t["profit_hr_busy_mid"]).abs(), weights=w))
    wet = t["actually_rained"] == 1
    miss_wet = float(np.average((t.loc[wet, "pred"] - t.loc[wet, "profit_hr_busy_mid"]).abs(), weights=w[wet]))
    # "Is this a good hour to drive at all?": all of Manhattan + airports together, hour by hour
    city = t.assign(wp=t["pred"] * w, wa=t["profit_hr_busy_mid"] * w).groupby("hour")[["wp", "wa", "driver_hours"]].sum()
    city_miss = float(((city["wp"] - city["wa"]) / city["driver_hours"]).abs().mean())
    t["period"] = pd.cut(t["hour_of_day"], [-1, 5, 9, 15, 19, 23], labels=list(PERIODS))
    t["date"] = t["hour"].dt.date
    t["wp"], t["wa"] = t["pred"] * w, t["profit_hr_busy_mid"] * w
    cell = t.groupby(["date", "period", "zone"], observed=True).agg(
        wp=("wp", "sum"), wa=("wa", "sum"), w=("driver_hours", "sum"), n=("hour_of_day", "size"))
    cell = cell[cell["n"] == cell.groupby(["date", "period"], observed=True)["n"].transform("max")]
    cell["pred"], cell["actual"] = cell["wp"] / cell["w"], cell["wa"] / cell["w"]
    hits, lost, picks = [], [], 0
    for _, g in cell.groupby(level=["date", "period"], observed=True):
        if len(g) < 10:
            continue
        top3 = set(g["actual"].nlargest(3).index.get_level_values("zone"))
        pick = g["pred"].idxmax()
        hits.append(pick[2] in top3)
        lost.append(g["actual"].max() - g.loc[pick, "actual"])
        picks += 1
    return {"typical_miss": round(miss, 2), "miss_rainy_hours": round(miss_wet, 2), "citywide_hour_miss": round(city_miss, 2), "hit_rate_top3": round(float(np.mean(hits)), 4),
            "dollars_lost": round(float(np.mean(lost)), 2), "plans_tested": picks}


def main():
    df = load()
    df["actually_rained"] = df["is_rain"]
    train, test = df[df["year"] <= 2025], df[df["year"] == 2026]
    test_fc = with_forecast_weather(test)
    models = fit_models(train)
    rows = []
    for name, predict in models.items():
        for weather, t in [("forecast the day before", test_fc), ("actual weather", test)]:
            r = score(t, predict(t))
            rows.append({"model": name, "weather": weather, **r})
            print(name, "|", weather, "|", r, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "model_test.csv", index=False)
    fc = res[res["weather"] == "forecast the day before"].sort_values("hit_rate_top3", ascending=False)
    lines = ["Accuracy test on Jan-Aug 2026 (models trained on 2024-2025 only), using the day-before forecast",
             f"{int(fc['plans_tested'].iloc[0]):,} plans tested (each day x 5 times of day); "
             "random picking would land in the top 3 about 4% of the time", ""]
    for r in fc.itertuples():
        lines.append(f"{r.model:14s} top pick in actual top 3: {r.hit_rate_top3:.0%}   "
                     f"typical miss ${r.typical_miss:.2f}/hr (rainy hours ${r.miss_rainy_hours:.2f})   "
                 f"citywide hour miss ${r.citywide_hour_miss:.2f}   lost vs best zone ${r.dollars_lost:.2f}/hr")
    (REPORTS / "model_test.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
