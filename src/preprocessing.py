"""Cleaning, feature engineering and hourly aggregation of EV charging sessions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import data_loader

LOCAL_TZ = "America/Los_Angeles"   # ACN-Data timestamps are UTC; the Caltech site is in LA.


def _parse_time(series: pd.Series) -> pd.Series:
    """Parse RFC-1123 / ISO timestamps to timezone-aware UTC datetimes (NaT on failure)."""
    out = pd.to_datetime(series, format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce", utc=True)
    if out.isna().mean() > 0.5:                      # different format -> generic parser
        out = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return out


def clean_sessions(raw: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    """Return ``(clean_df, report)`` where ``report`` counts removed rows per reason."""
    df = raw.copy()
    report: dict[str, int] = {"rows_raw": int(len(df))}

    for col, new in [("connectionTime", "connection"), ("disconnectTime", "disconnect"),
                     ("doneChargingTime", "done_charging")]:
        df[new] = _parse_time(df[col])

    # Real data: convert UTC -> site local time (hour-of-day analysis needs local time).
    # Synthetic sample: timestamps are already "local" by construction.
    real = source.startswith("ACN")
    for c in ("connection", "disconnect", "done_charging"):
        df[c] = df[c].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None) if real else df[c].dt.tz_localize(None)

    # --- missing values -------------------------------------------------------------
    n0 = len(df)
    df = df.dropna(subset=["connection", "disconnect", "kWhDelivered"])
    report["dropped_missing_core_fields"] = int(n0 - len(df))
    # doneChargingTime is frequently missing in ACN-Data: assume charging lasted until disconnect.
    report["imputed_done_charging_with_disconnect"] = int(df["done_charging"].isna().sum())
    df["done_charging"] = df["done_charging"].fillna(df["disconnect"])

    # --- invalid sessions -----------------------------------------------------------
    df["duration_h"] = (df["disconnect"] - df["connection"]).dt.total_seconds() / 3600.0
    n0 = len(df)
    df = df[(df["duration_h"] > 0.1) & (df["duration_h"] <= 48)]
    report["removed_bad_duration"] = int(n0 - len(df))

    n0 = len(df)
    df = df[(df["kWhDelivered"] > 0.1) & (df["kWhDelivered"] <= 100)]
    report["removed_bad_energy"] = int(n0 - len(df))

    # charging window must lie inside the connection window
    df["done_charging"] = df[["done_charging", "disconnect"]].min(axis=1)
    df["done_charging"] = df[["done_charging", "connection"]].max(axis=1)
    df["charge_duration_h"] = (df["done_charging"] - df["connection"]).dt.total_seconds() / 3600.0
    n0 = len(df)
    df = df[df["charge_duration_h"] > 0.05]
    report["removed_zero_charge_window"] = int(n0 - len(df))

    df["avg_power_kw"] = df["kWhDelivered"] / df["charge_duration_h"]
    n0 = len(df)
    df = df[df["avg_power_kw"] <= 25]            # implausible for AC Level-2 charging
    report["removed_implausible_power"] = int(n0 - len(df))

    # --- feature engineering --------------------------------------------------------
    df["idle_h"] = (df["duration_h"] - df["charge_duration_h"]).clip(lower=0)
    df["arrival_hour"] = df["connection"].dt.hour + df["connection"].dt.minute / 60.0
    df["departure_hour"] = df["disconnect"].dt.hour + df["disconnect"].dt.minute / 60.0
    df["arrival_hour_int"] = df["connection"].dt.hour
    df["day_of_week"] = df["connection"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["date"] = df["connection"].dt.date.astype(str)
    df = df.sort_values("connection").reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    report["data_source"] = source
    return df, report


def hourly_demand(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to an hourly series.

    Each session's delivered energy is spread *uniformly* over its charging window
    ``[connection, done_charging]`` (the dataset contains no per-minute power trace), then summed per hour.
    Columns: timestamp, energy_kwh, n_active (sessions overlapping the hour), n_arrivals.
    """
    t0 = sessions["connection"].min().floor("h")
    t1 = sessions["disconnect"].max().ceil("h")
    n_hours = int((t1 - t0).total_seconds() // 3600) + 1
    energy = np.zeros(n_hours)
    active = np.zeros(n_hours)

    s = (sessions["connection"] - t0).dt.total_seconds().to_numpy() / 3600.0
    e = (sessions["done_charging"] - t0).dt.total_seconds().to_numpy() / 3600.0
    d = (sessions["disconnect"] - t0).dt.total_seconds().to_numpy() / 3600.0
    kwh = sessions["kWhDelivered"].to_numpy()
    for si, ei, di, ki in zip(s, e, d, kwh):
        span = max(ei - si, 1e-6)
        for h in range(int(si), min(int(np.ceil(ei)), n_hours)):
            overlap = min(ei, h + 1) - max(si, h)
            if overlap > 0:
                energy[h] += ki * overlap / span
        for h in range(int(si), min(int(np.ceil(di)), n_hours)):
            if min(di, h + 1) - max(si, h) > 0:
                active[h] += 1
    arrivals = np.zeros(n_hours)
    for h in np.floor(s).astype(int):
        arrivals[min(h, n_hours - 1)] += 1
    out = pd.DataFrame({
        "timestamp": pd.date_range(t0, periods=n_hours, freq="h"),
        "energy_kwh": energy, "n_active": active, "n_arrivals": arrivals,
    })
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def station_utilization(sessions: pd.DataFrame) -> pd.DataFrame:
    """Share of the observation period each station was occupied / actively charging."""
    period_h = (sessions["disconnect"].max() - sessions["connection"].min()).total_seconds() / 3600.0
    g = sessions.groupby("stationID").agg(
        sessions=("sessionID", "count"),
        energy_kwh=("kWhDelivered", "sum"),
        occupied_h=("duration_h", "sum"),
        charging_h=("charge_duration_h", "sum"),
    ).reset_index()
    g["occupancy_pct"] = 100 * g["occupied_h"] / period_h
    g["charging_pct"] = 100 * g["charging_h"] / period_h
    return g.sort_values("energy_kwh", ascending=False).reset_index(drop=True)


def run_pipeline(prefer_real: bool = True) -> dict:
    """Load -> clean -> save processed sessions + hourly demand; returns the cleaning report."""
    raw, source = data_loader.load_raw_sessions(prefer_real=prefer_real)
    clean, report = clean_sessions(raw, source)
    if clean.empty:
        raise ValueError("No valid sessions remain after cleaning.")
    clean.to_csv(config.SESSIONS_CSV, index=False)
    hourly = hourly_demand(clean)
    hourly.to_csv(config.HOURLY_CSV, index=False)
    config.DATA_SOURCE_FILE.write_text(json.dumps({"source": source, "n_sessions": len(clean)}))
    (config.METRICS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(run_pipeline(), indent=2))
