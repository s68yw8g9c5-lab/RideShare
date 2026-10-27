"""First look at where and when driving pays: a zone map, a day-by-hour grid,
the weather effect, and the airport-versus-Manhattan break-even.

All numbers use the main target, profit_hr_busy_mid (middle-case assumptions,
empty time that moves with how busy the zone is), and only zone-hours with
enough trips. Averages are weighted by driver hours, so they read as
"what an average hour of driving earned".

Usage:  python pipeline/05_explore.py
Output: figures/*.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

FIG = Path("figures")
AIRPORTS = {132: "JFK", 138: "LaGuardia"}
IRS_RATE = {2024: 0.67, 2025: 0.70, 2026: 0.725}
AIRPORT_LOT_MILES = 2
RANDALLS_ISLAND = 194

# colors: one blue ramp for magnitude, ink for text, gray for reference lines
SURFACE, INK, INK_2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE = "#2a78d6"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    df = pd.read_parquet("data/clean/model_table.parquet")
    df = df[df["enough_trips"] == 1].copy()
    df["driver_hours"] = df["trip_hours"] / (1 - df["empty_share_mid"])
    df.loc[df["zone"].isin(AIRPORTS), "driver_hours"] = df["trip_hours"] + df["trips"]  # 60 min wait
    return df


def avg(df, by):
    """Profit per driver hour, weighted by driver hours."""
    p = df["profit_hr_busy_mid"] * df["driver_hours"]
    return p.groupby([df[b] for b in by]).sum() / df.groupby(by)["driver_hours"].sum()


def title(ax, headline, sub):
    ax.set_title(headline, loc="left", fontsize=15, fontweight="bold", color=INK, pad=28)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=INK_2, va="bottom")


def zone_map(df):
    shapes = gpd.read_file("zip://data/raw/taxi_zones.zip!taxi_zones/taxi_zones.shp")
    man = shapes[shapes["borough"] == "Manhattan"].copy()
    by_zone = avg(df[~df["zone"].isin(AIRPORTS)], ["zone"])
    man["profit"] = man["LocationID"].map(by_zone)
    # Randalls Island is stadiums and festivals: real, but it would wash out the scale
    event = man["LocationID"] == RANDALLS_ISLAND
    vmax = man.loc[~event, "profit"].max()
    fig, ax = plt.subplots(figsize=(7.5, 10))
    man.plot(column="profit", cmap=BLUES, ax=ax, edgecolor=SURFACE, linewidth=0.8,
             vmax=vmax, legend=True, missing_kwds={"color": GRID},
             legend_kwds={"label": "Profit per hour ($)", "shrink": 0.5, "pad": 0.02,
                          "orientation": "horizontal", "anchor": (0.0, 1.0)})
    top = man[~event].dropna(subset=["profit"]).nlargest(3, "profit")
    top = top.assign(y=top.geometry.representative_point().y).sort_values("y", ascending=False)
    for i, (_, r) in enumerate(top.iterrows()):  # labels stacked in map order, so lines don't cross
        p = r.geometry.representative_point()
        ax.annotate(f"{r['zone']}  \\${r['profit']:.0f}/hr", (p.x, p.y),
                    xytext=(-60, 70 - 45 * i), textcoords="offset points", ha="right",
                    fontsize=9.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    r = man[event].iloc[0]
    p = r.geometry.representative_point()
    ax.annotate(f"Randalls Island  \\${r['profit']:.0f}/hr\n(stadium and festival crowds)",
                (p.x, p.y), xytext=(15, 55), textcoords="offset points", fontsize=9.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    airports = avg(df[df["zone"].isin(AIRPORTS)], ["zone"])
    note = ",  ".join(f"{AIRPORTS[z]} \\${v:.0f}/hr" for z, v in airports.items())
    ax.text(0, 0.97, f"Airports (off map, assuming a 1-hour lot wait): {note}",
            transform=ax.transAxes, fontsize=10, color=INK_2)
    ax.set_axis_off()
    title(ax, "Midtown pays best in Manhattan",
          "Average driver profit per hour after car costs, Jan 2024 to Aug 2026, in Aug 2026 dollars")
    fig.savefig(FIG / "map_profit_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return by_zone


def day_hour_grid(df):
    m = df[~df["zone"].isin(AIRPORTS)]
    grid = avg(m, ["day_of_week", "hour_of_day"]).unstack()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(grid.values, cmap=BLUES, aspect="auto")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    hours = [f"{(h % 12) or 12}{'am' if h < 12 else 'pm'}" for h in range(24)]
    ax.set_xticks(range(0, 24, 2), hours[::2])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    best = np.unravel_index(np.nanargmax(grid.values), grid.shape)
    worst = np.unravel_index(np.nanargmin(grid.values), grid.shape)
    for (r, c) in (best, worst):
        v = grid.values[r, c]
        ax.text(c, r, f"${v:.0f}", ha="center", va="center", fontsize=9,
                color="white" if v > grid.values.mean() else INK)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.9)
    cb.set_label("Profit per hour ($)")
    cb.outline.set_visible(False)
    title(ax, "Weeknights 9 to 11 pm pay most; late mornings pay least",
          "Manhattan, average driver profit per hour by day and hour")
    fig.savefig(FIG / "heatmap_day_hour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return grid


def weather_bars(df):
    m = df[~df["zone"].isin(AIRPORTS)].copy()
    m["weather"] = np.select([m["is_snow"] == 1, m["is_rain"] == 1], ["Snow", "Rain"], "Dry")
    vals = avg(m, ["weather"]).reindex(["Dry", "Rain", "Snow"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(vals.index, vals.values, color=BLUE, width=0.55)
    for b, v in zip(bars, vals.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"${v:.2f}", ha="center",
                fontsize=11, color=INK)
    ax.set_ylabel("Profit per hour ($)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    pct = vals["Rain"] / vals["Dry"] - 1
    title(ax, f"Rainy hours pay about {pct:.0%} more",
          "Manhattan, average driver profit per hour (not yet adjusted for time of day)")
    fig.savefig(FIG / "weather_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return vals


def airport_breakeven(df):
    a = df[df["zone"].isin(AIRPORTS)]
    profit = ((a["driver_pay"] + a["tips"]
               - a["year"].map(IRS_RATE) * (a["trip_miles"] + AIRPORT_LOT_MILES * a["trips"]))
              * a["cpi_factor"]).sum()  # in August 2026 dollars, like the Manhattan figures
    waits = np.arange(0, 181, 5)
    per_hr = profit / (a["trip_hours"].sum() + a["trips"].sum() * waits / 60)
    manhattan = avg(df[~df["zone"].isin(AIRPORTS)].assign(k=1), ["k"]).iloc[0]
    even = 60 * (profit / manhattan - a["trip_hours"].sum()) / a["trips"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(waits, per_hr, color=BLUE, lw=2)
    ax.axhline(manhattan, color=REF, lw=1.5, ls="--")
    ax.text(178, manhattan + 0.8, f"Manhattan average ${manhattan:.0f}/hr", ha="right",
            color=INK_2, fontsize=10)
    ax.text(20, np.interp(20, waits, per_hr) + 2, "Airport pickups", color=INK, fontsize=10)
    ax.plot([even], [manhattan], "o", ms=8, color=BLUE, mec=SURFACE, mew=2)
    ax.annotate(f"Break-even: {even:.0f} min wait", (even, manhattan), xytext=(12, 18),
                textcoords="offset points", fontsize=10, color=INK)
    ax.set_xlabel("Minutes waiting in the airport taxi lot per pickup")
    ax.set_ylabel("Profit per hour ($)")
    ax.set_xlim(0, 180)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.tick_params(length=0)
    title(ax, f"The airport beats Manhattan until the lot wait passes {even:.0f} minutes",
          "JFK and LaGuardia combined, driver profit per hour after car costs")
    fig.savefig(FIG / "airport_breakeven.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return even


def main():
    FIG.mkdir(exist_ok=True)
    df = load()
    zones = zone_map(df)
    grid = day_hour_grid(df)
    weather = weather_bars(df)
    even = airport_breakeven(df)
    names = pd.read_csv("data/taxi_zone_lookup.csv").set_index("LocationID")["Zone"]
    print("Top zones:", zones.sort_values(ascending=False).head(5).rename(names).round(2).to_dict())
    print("Day x hour range:", round(grid.min().min(), 2), "to", round(grid.max().max(), 2))
    print("Weather:", weather.round(2).to_dict())
    print("Airport break-even wait (min):", round(even))


if __name__ == "__main__":
    main()
