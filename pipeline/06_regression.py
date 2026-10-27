"""How much does weather move a Manhattan driver's profit per hour, once time of day,
day of week, season and zone are held fixed?

A weighted regression on 2024-2025 (2026 is kept back to test the models later).
Weights are driver hours, so busy zone-hours count more than quiet ones.
Zone, day-of-week x hour and month are "fixed effects": each gets its own
baseline, so the weather numbers compare like with like (a rainy Tuesday 6pm in
Midtown against a dry Tuesday 6pm in Midtown). Change over time is a steady trend
plus a step at each rule change (congestion pricing, TLC pay-rate updates).
Dollars are August 2026 dollars.
Standard errors are clustered by day, because every zone shares the same weather on a
given day; treating them as independent would make the results look more certain than
they are.

Usage:  python pipeline/06_regression.py
Output: reports/regression_weather.txt, reports/rain_by_time.csv, reports/regression_summary.csv,
        figures/weather_effect_by_time.png
"""
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

viz = import_module("05_explore")  # same colors and helpers as the exploration charts

REPORTS, FIG = Path("reports"), Path("figures")
AIRPORTS = (132, 138)
BLOCKS = [  # (name, weekend?, first hour, last hour)
    ("Weekday 7-10am", 0, 7, 9), ("Weekday 10am-4pm", 0, 10, 15),
    ("Weekday 4-8pm", 0, 16, 19), ("Weekday 8pm-midnight", 0, 20, 23),
    ("Weekday midnight-7am", 0, 0, 6), ("Weekend, any hour", 1, 0, 23),
]
TEMPS = ["below 32F", "32-50F", "70-85F", "above 85F"]  # compared with 50-70F
RULES = {"congestion_pricing_zone": "Congestion pricing, below 60th St (Jan 2025)",
         "pay_rate_2024_03": "TLC pay update (Mar 2024)", "pay_rate_2025_03": "TLC pay update (Mar 2025)",
         "pay_rate_2025_08": "TLC 5% pay raise (Aug 2025)"}
# The March 2026 pay update falls after the 2024-2025 training years; the 2026 test uses it.
CONTROLS = (" + ".join(f"temp_{i}" for i in range(len(TEMPS))) + " + windy + is_holiday"
            " + months_since_start + " + " + ".join(RULES))
FIXED = "zone + dow_hour + month"


def prepare():
    df = viz.load()
    df = df[~df["zone"].isin(AIRPORTS)].dropna(
        subset=["profit_hr_busy_mid", "is_rain", "is_snow", "temp_f", "wind_mph"]).copy()
    band = pd.cut(df["temp_f"], [-50, 32, 50, 70, 85, 130],
                  labels=["below 32F", "32-50F", "50-70F", "70-85F", "above 85F"])
    for i, b in enumerate(TEMPS):
        df[f"temp_{i}"] = (band == b).astype(int)
    df["windy"] = (df["wind_mph"] >= 20).astype(int)
    df["dow_hour"] = df["day_of_week"] * 24 + df["hour_of_day"]
    df["day"] = df["hour"].dt.strftime("%Y-%m-%d")
    for i, (name, weekend, h0, h1) in enumerate(BLOCKS):
        inside = (df["is_weekend"] == weekend) & df["hour_of_day"].between(h0, h1)
        df[f"rain_b{i}"] = (inside & (df["is_rain"] == 1)).astype(int)
    df["fixed_hours"] = df["trip_hours"] / (1 - 0.6)
    keep = ["light_rain", "heavy_rain", "profit_hr_busy_mid", "profit_hr_fixed_mid", "fixed_hours", "months_since_start", *RULES, "driver_hours", "is_rain", "is_snow", "windy", "is_holiday",
            "zone", "dow_hour", "month", "year", "day", "trip_hours", "empty_share_mid"]
    keep += [c for c in df.columns if c.startswith(("temp_", "rain_b"))]
    return df[keep]


def fit(rhs, data, y="profit_hr_busy_mid", w="driver_hours"):
    return pf.feols(f"{y} ~ {rhs} | {FIXED}", data=data, weights=w, vcov={"CRV1": "day"})


def row(model, term):
    b = model.coef()[term]
    lo, hi = model.confint().loc[term]
    return b, lo, hi


def main():
    REPORTS.mkdir(exist_ok=True)
    df = prepare()
    train = df[df["year"] <= 2025]
    dry = train[(train["is_rain"] == 0) & (train["is_snow"] == 0)]
    dry_avg = np.average(dry["profit_hr_busy_mid"], weights=dry["driver_hours"])

    main_model = fit(f"is_rain + is_snow + {CONTROLS}", train)
    intensity = fit(f"light_rain + heavy_rain + is_snow + {CONTROLS}", train)
    blocks = " + ".join(f"rain_b{i}" for i in range(len(BLOCKS)))
    by_time = fit(f"{blocks} + is_snow + {CONTROLS}", train)

    lines = ["Weather effect on Manhattan driver profit per hour (middle-case assumptions)",
             f"Trained on 2024-2025, {len(train):,} zone-hours; same zone, day-of-week x hour",
             "and month compared, plus a trend and rule-change steps; August 2026 dollars;",
             "95% ranges clustered by day",
             f"Average dry-hour profit: ${dry_avg:.2f}", ""]
    for term, label in [("is_rain", "Rain"), ("is_snow", "Snow"), ("windy", "Wind 20+ mph")]:
        b, lo, hi = row(main_model, term)
        lines.append(f"{label:20s} {b:+.2f} $/hr  (95% range {lo:+.2f} to {hi:+.2f})"
                     f"  = {b / dry_avg:+.1%} of a dry hour")
    for i, band in enumerate(TEMPS):
        b, lo, hi = row(main_model, f"temp_{i}")
        lines.append(f"{band + ' vs 50-70F':20s} {b:+.2f} $/hr  (95% range {lo:+.2f} to {hi:+.2f})")
    for term, label in [("light_rain", "  light rain"), ("heavy_rain", "  heavy rain (0.1 in/hr+)")]:
        b, lo, hi = row(intensity, term)
        lines.append(f"{label:20s} {b:+.2f} $/hr  (95% range {lo:+.2f} to {hi:+.2f})"
                     f"  = {b / dry_avg:+.1%} of a dry hour")
    b, lo, hi = row(main_model, "months_since_start")
    lines.append(f"{'Trend, per year':20s} {12 * b:+.2f} $/hr  (95% range {12 * lo:+.2f} to {12 * hi:+.2f})")
    lines += ["", "Steps at rule changes (on top of the trend):"]
    for term, label in RULES.items():
        b, lo, hi = row(main_model, term)
        lines.append(f"  {label:44s} {b:+.2f} $/hr  (95% range {lo:+.2f} to {hi:+.2f})")
    fixed = fit(f"is_rain + is_snow + {CONTROLS}", train, y="profit_hr_fixed_mid", w="fixed_hours")
    rain, fixed_rain = row(main_model, "is_rain"), row(fixed, "is_rain")
    pd.DataFrame([(f"rain_{k}{s}", round(v, 2)) for s, vals in [("", rain), ("_fixed", fixed_rain)]
                  for k, v in zip(["effect", "low", "high"], vals)]
                 + [("rain_effect_pct", round(rain[0] / dry_avg, 4))],
                 columns=["metric", "value"]).to_csv(REPORTS / "regression_summary.csv", index=False)
    b, lo, hi = fixed_rain
    lines += ["", "Check: with the same empty time in every hour (no busy-ness adjustment),",
              f"rain adds {b:+.2f} $/hr (95% range {lo:+.2f} to {hi:+.2f}). So the true rain effect",
              "most likely sits between the two estimates, depending on how much rain cuts drivers' waits."]
    lines += ["", f"R-squared (with fixed effects): {main_model._r2:.2f}", "", "Rain effect by time of day:"]
    rows = []
    for i, (name, *_) in enumerate(BLOCKS):
        b, lo, hi = row(by_time, f"rain_b{i}")
        rows.append({"block": name, "rain_effect": round(b, 2), "low": round(lo, 2), "high": round(hi, 2),
                     "pct_of_dry": round(b / dry_avg, 4)})
        lines.append(f"  {name:22s} {b:+.2f} $/hr  (95% range {lo:+.2f} to {hi:+.2f})")
    text = "\n".join(lines)
    (REPORTS / "regression_weather.txt").write_text(text + "\n")
    pd.DataFrame(rows).to_csv(REPORTS / "rain_by_time.csv", index=False)
    print(text)
    effect_chart(rows, row(main_model, "is_rain")[0])


def effect_chart(rows, overall):
    rows = sorted(rows, key=lambda r: r["rain_effect"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, r in enumerate(rows):
        ax.plot([r["low"], r["high"]], [i, i], color=viz.BLUE, lw=2, solid_capstyle="round", alpha=0.45)
        ax.plot([r["rain_effect"]], [i], "o", ms=9, color=viz.BLUE, mec=viz.SURFACE, mew=2)
        ax.text(r["high"] + 0.15, i, f"{r['rain_effect']:+.2f}".replace("+", "+\\$"),
                va="center", fontsize=10, color=viz.INK)
    ax.axvline(0, color=viz.REF, lw=1)
    ax.set_yticks(range(len(rows)), [r["block"] for r in rows])
    ax.set_xlabel("Extra profit per hour when it rains ($), with 95% range")
    ax.xaxis.grid(True, color=viz.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    viz.title(ax, f"Rain pays most: {rows[-1]['block'].lower()}",
              f"Manhattan, same zone, hour and season compared (overall +\\${overall:.2f}/hr)")
    fig.savefig(FIG / "weather_effect_by_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
