"""Exploratory data analysis: summary statistics (JSON) and publication-style plots (PNG)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import data_loader, preprocessing

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def summary_statistics(sessions: pd.DataFrame, hourly: pd.DataFrame) -> dict:
    """Numbers reported in the README/dashboard - all computed from the data at hand."""
    by_hour = hourly.groupby("hour")["energy_kwh"].mean()
    daily = hourly.set_index("timestamp")["energy_kwh"].resample("D").sum()
    util = preprocessing.station_utilization(sessions)
    arrivals = sessions.groupby("arrival_hour_int").size()
    return {
        "data_source": data_loader.get_data_source(),
        "n_sessions": int(len(sessions)),
        "n_stations": int(sessions["stationID"].nunique()),
        "period_start": str(sessions["connection"].min()),
        "period_end": str(sessions["disconnect"].max()),
        "total_energy_kwh": float(sessions["kWhDelivered"].sum()),
        "energy_per_session": sessions["kWhDelivered"].describe().round(3).to_dict(),
        "duration_h": sessions["duration_h"].describe().round(3).to_dict(),
        "charge_duration_h": sessions["charge_duration_h"].describe().round(3).to_dict(),
        "avg_power_kw": sessions["avg_power_kw"].describe().round(3).to_dict(),
        "mean_daily_energy_kwh": float(daily.mean()),
        "peak_hour_of_day": int(by_hour.idxmax()),
        "peak_hour_mean_kwh": float(by_hour.max()),
        "top3_peak_hours": [int(h) for h in by_hour.sort_values(ascending=False).index[:3]],
        "peak_arrival_hour": int(arrivals.idxmax()),
        "max_hourly_energy_kwh": float(hourly["energy_kwh"].max()),
        "mean_station_occupancy_pct": float(util["occupancy_pct"].mean()),
        "mean_station_charging_pct": float(util["charging_pct"].mean()),
        "weekday_vs_weekend_session_energy": sessions.groupby("is_weekend")["kWhDelivered"].mean().round(3).to_dict(),
        "weekday_vs_weekend_duration_h": sessions.groupby("is_weekend")["duration_h"].mean().round(3).to_dict(),
        "corr_duration_energy": float(sessions[["duration_h", "kWhDelivered"]].corr().iloc[0, 1]),
    }


def make_plots(sessions: pd.DataFrame, hourly: pd.DataFrame, outdir=config.PLOTS_DIR) -> list[str]:
    """Create the six required plots; returns the written file paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    src = data_loader.get_data_source()
    paths = []

    def _save(fig, name):
        p = outdir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))

    # 1. arrivals through the day
    fig, ax = plt.subplots(figsize=(7, 4))
    cnt = sessions.groupby(["arrival_hour_int", "is_weekend"]).size().unstack(fill_value=0)
    ax.bar(cnt.index - 0.2, cnt.get(0, 0), width=0.4, label="Weekday")
    ax.bar(cnt.index + 0.2, cnt.get(1, 0), width=0.4, label="Weekend")
    ax.set(xlabel="Hour of arrival", ylabel="Sessions", title=f"EV arrivals by hour ({src})")
    ax.legend()
    _save(fig, "01_arrivals_by_hour.png")

    # 2. duration distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sessions["duration_h"].clip(upper=20), bins=50, alpha=0.7, label="Connection duration")
    ax.hist(sessions["charge_duration_h"].clip(upper=20), bins=50, alpha=0.7, label="Charging duration")
    ax.set(xlabel="Hours (clipped at 20)", ylabel="Sessions", title="Duration distributions")
    ax.legend()
    _save(fig, "02_duration_distribution.png")

    # 3. energy distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sessions["kWhDelivered"], bins=50, color="tab:green", alpha=0.8)
    ax.axvline(sessions["kWhDelivered"].mean(), color="k", ls="--", label=f"mean {sessions['kWhDelivered'].mean():.1f} kWh")
    ax.set(xlabel="Energy delivered per session (kWh)", ylabel="Sessions", title="Energy per session")
    ax.legend()
    _save(fig, "03_energy_distribution.png")

    # 4. hourly demand profile (weekday vs weekend)
    fig, ax = plt.subplots(figsize=(7, 4))
    prof = hourly.groupby(["hour", "is_weekend"])["energy_kwh"].mean().unstack()
    for k, lab in [(0, "Weekday"), (1, "Weekend")]:
        if k in prof:
            ax.plot(prof.index, prof[k], marker="o", label=lab)
    ax.set(xlabel="Hour of day", ylabel="Mean energy (kWh/h)", title="Average hourly charging demand")
    ax.legend()
    _save(fig, "04_hourly_demand.png")

    # 5. daily consumption
    fig, ax = plt.subplots(figsize=(9, 4))
    daily = hourly.set_index("timestamp")["energy_kwh"].resample("D").sum()
    ax.plot(daily.index, daily.values, lw=0.8)
    ax.plot(daily.index, daily.rolling(7, min_periods=1).mean(), lw=2, label="7-day mean")
    ax.set(xlabel="Date", ylabel="Energy (kWh/day)", title="Daily electricity consumption")
    ax.legend()
    _save(fig, "05_daily_consumption.png")

    # 6. station utilisation
    fig, ax = plt.subplots(figsize=(9, 4))
    util = preprocessing.station_utilization(sessions).head(30)
    ax.bar(util["stationID"], util["occupancy_pct"], label="Occupied")
    ax.bar(util["stationID"], util["charging_pct"], label="Charging", alpha=0.8)
    ax.set(ylabel="% of observation period", title="Charging-station utilisation")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    _save(fig, "06_station_utilization.png")
    return paths


def run_eda() -> dict:
    sessions = data_loader.load_processed_sessions()
    hourly = data_loader.load_hourly_demand()
    stats = summary_statistics(sessions, hourly)
    (config.METRICS_DIR / "eda_summary.json").write_text(json.dumps(stats, indent=2, default=str))
    make_plots(sessions, hourly)
    return stats


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2, default=str))
