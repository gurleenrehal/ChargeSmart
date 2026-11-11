"""Data loading for ChargeSmart.

Real data source: Caltech ACN-Data (https://ev.caltech.edu/dataset), fetched with the
official ``acnportal`` DataClient (or an equivalent REST fallback).

IMPORTANT - real vs synthetic
-----------------------------
ACN-Data needs a (free) API token and network access. To keep the project runnable
offline, a **synthetic, schema-compatible sample** can be generated
(``generate_synthetic_sample``). It is *not* real data: it is produced by a seeded
generator whose parameters are documented below, and every downstream artefact records
which source was used (``data/processed/data_source.json``).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config

ACN_API_URL = "https://ev.caltech.edu/api/v1/"
RFC_FMT = "%a, %d %b %Y %H:%M:%S GMT"

# Columns of the (flattened) ACN-Data session schema used by this project.
SESSION_COLUMNS = [
    "sessionID", "siteID", "stationID", "spaceID", "userID",
    "connectionTime", "disconnectTime", "doneChargingTime",
    "kWhDelivered", "kWhRequested", "minutesAvailable",
]


# --------------------------------------------------------------------------- #
# Real data: ACN-Data download
# --------------------------------------------------------------------------- #
def _flatten_session(s: dict) -> dict:
    """Flatten one ACN-Data JSON session (userInputs is a list of dicts)."""
    ui = s.get("userInputs") or []
    last = ui[-1] if ui else {}
    return {
        "sessionID": s.get("sessionID") or s.get("_id"),
        "siteID": s.get("siteID"),
        "stationID": s.get("stationID"),
        "spaceID": s.get("spaceID"),
        "userID": s.get("userID"),
        "connectionTime": s.get("connectionTime"),
        "disconnectTime": s.get("disconnectTime"),
        "doneChargingTime": s.get("doneChargingTime"),
        "kWhDelivered": s.get("kWhDelivered"),
        "kWhRequested": last.get("kWhRequested"),
        "minutesAvailable": last.get("minutesAvailable"),
    }


def download_acn_data(token: str, site: str = "caltech", start: datetime | None = None,
                      end: datetime | None = None, out_csv: Path = config.RAW_CSV) -> pd.DataFrame:
    """Download sessions from ACN-Data and save them to ``out_csv``.

    Uses the official ``acnportal.acndata.DataClient`` when installed
    (``pip install acnportal``), otherwise the same REST endpoint through ``requests``.
    Raises ``RuntimeError`` with a readable message on failure.
    """
    start = start or datetime(2019, 1, 1)
    end = end or datetime(2019, 12, 31)
    records: list[dict] = []
    try:
        from acnportal.acndata import DataClient  # type: ignore

        client = DataClient(token)
        for s in client.get_sessions_by_time(site, start, end):
            records.append(_flatten_session(s))
    except ImportError:
        import requests

        url = f"{ACN_API_URL}sessions/{site}"
        params = {"where": f'connectionTime>="{start.strftime(RFC_FMT)}" and '
                           f'connectionTime<="{end.strftime(RFC_FMT)}"'}
        while url:
            try:
                r = requests.get(url, params=params, auth=(token, ""), timeout=30)
                r.raise_for_status()
            except Exception as exc:  # network / auth problems
                raise RuntimeError(f"ACN-Data request failed: {exc}") from exc
            payload = r.json()
            records.extend(_flatten_session(s) for s in payload.get("_items", []))
            nxt = (payload.get("_links") or {}).get("next")
            url = ("https://ev.caltech.edu" + "/api/v1/" + nxt["href"]) if nxt else None
            params = None
    except Exception as exc:
        raise RuntimeError(f"ACN-Data download failed: {exc}") from exc

    if not records:
        raise RuntimeError("ACN-Data returned no sessions for the requested period.")
    df = pd.DataFrame.from_records(records, columns=SESSION_COLUMNS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# Synthetic, schema-compatible sample (NOT real observations)
# --------------------------------------------------------------------------- #
def generate_synthetic_sample(n_days: int = 180, n_stations: int = 20, seed: int = 7,
                              out_csv: Path = config.SAMPLE_CSV) -> pd.DataFrame:
    """Create a *synthetic* workplace-style session dataset with the ACN-Data schema.

    Generator assumptions (all arbitrary, documented for transparency):
      * weekdays ~28 sessions/day, weekends ~6 sessions/day (Poisson);
      * arrivals: 70% morning N(8.5h, 1.2), 20% midday N(12.5h, 2.0), 10% evening N(17.5h, 1.5);
      * dwell time: morning ~N(8h,1.5), midday ~N(3.5h,1.2), evening ~N(2h,0.8) hours;
      * charging power per session ~ U(3.3, 6.6) kW, energy requested ~ lognormal (median 13 kWh);
      * delivered energy = min(requested, power * dwell) with 5% measurement noise.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2019, 1, 1)
    rows = []
    sid = 0
    for d in range(n_days):
        day = start + timedelta(days=d)
        weekend = day.weekday() >= 5
        n = rng.poisson(6 if weekend else 28)
        for _ in range(n):
            u = rng.random()
            if u < 0.70:
                arr_h, dwell = rng.normal(8.5, 1.2), rng.normal(8.0, 1.5)
            elif u < 0.90:
                arr_h, dwell = rng.normal(12.5, 2.0), rng.normal(3.5, 1.2)
            else:
                arr_h, dwell = rng.normal(17.5, 1.5), rng.normal(2.0, 0.8)
            arr_h = float(np.clip(arr_h, 0.2, 23.5))
            dwell = float(np.clip(dwell, 0.4, 14.0))
            conn = day + timedelta(hours=arr_h)
            disc = conn + timedelta(hours=dwell)
            power = rng.uniform(3.3, 6.6)
            req = float(np.clip(rng.lognormal(np.log(13.0), 0.55), 2.0, 45.0))
            delivered = min(req, power * dwell) * rng.normal(1.0, 0.03)
            charge_h = delivered / power
            done = conn + timedelta(hours=min(dwell, charge_h))
            st = int(rng.integers(1, n_stations + 1))
            sid += 1
            rows.append({
                "sessionID": f"SYN-{sid:06d}", "siteID": "SYNTHETIC",
                "stationID": f"SYN-ST-{st:02d}", "spaceID": f"SYN-SP-{st:02d}",
                "userID": int(rng.integers(1, 400)),
                "connectionTime": conn.strftime(RFC_FMT),
                "disconnectTime": disc.strftime(RFC_FMT),
                "doneChargingTime": done.strftime(RFC_FMT),
                "kWhDelivered": round(float(delivered), 3),
                "kWhRequested": round(req, 3),
                "minutesAvailable": round(dwell * 60.0, 1),
            })
    df = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    # Inject a small amount of realistic dirt so the cleaning pipeline has work to do.
    idx = rng.choice(len(df), size=max(5, len(df) // 100), replace=False)
    df.loc[idx[: len(idx) // 3], "kWhDelivered"] = -1.0
    df.loc[idx[len(idx) // 3: 2 * len(idx) // 3], "doneChargingTime"] = np.nan
    df.loc[idx[2 * len(idx) // 3:], "disconnectTime"] = df.loc[idx[2 * len(idx) // 3:], "connectionTime"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw_sessions(prefer_real: bool = True) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source_label)``.

    ``source_label`` is ``"ACN-Data (real)"`` when a downloaded file is available,
    otherwise ``"Synthetic sample (not real data)"``.
    """
    if prefer_real and config.RAW_CSV.exists():
        return pd.read_csv(config.RAW_CSV), "ACN-Data (real)"
    if not config.SAMPLE_CSV.exists():
        generate_synthetic_sample()
    return pd.read_csv(config.SAMPLE_CSV), "Synthetic sample (not real data)"


def get_data_source() -> str:
    """Label of the data the processed files were built from."""
    if config.DATA_SOURCE_FILE.exists():
        try:
            return json.loads(config.DATA_SOURCE_FILE.read_text())["source"]
        except (json.JSONDecodeError, KeyError):
            pass
    return "unknown"


def load_processed_sessions() -> pd.DataFrame:
    """Load cleaned sessions, running the preprocessing pipeline if needed."""
    from src import preprocessing

    if not config.SESSIONS_CSV.exists():
        preprocessing.run_pipeline()
    df = pd.read_csv(config.SESSIONS_CSV, parse_dates=["connection", "disconnect", "done_charging"])
    return df


def load_hourly_demand() -> pd.DataFrame:
    from src import preprocessing

    if not config.HOURLY_CSV.exists():
        preprocessing.run_pipeline()
    return pd.read_csv(config.HOURLY_CSV, parse_dates=["timestamp"])
