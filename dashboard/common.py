"""Shared helpers for the Streamlit pages."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from src import data_loader, preprocessing, trainer
from src.defaults import default_reward_config, default_sim_config

CSS = """
<style>
.kpi{background:linear-gradient(135deg,#0f2b46,#1d5c8a);color:#fff;border-radius:12px;padding:14px 16px;margin-bottom:8px}
.kpi .v{font-size:1.6rem;font-weight:700}.kpi .l{font-size:.8rem;opacity:.85}
</style>
"""


def kpi(col, label: str, value: str) -> None:
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


def source_banner() -> None:
    src = data_loader.get_data_source()
    if src.startswith("Synthetic"):
        st.warning("Data source: **synthetic sample** (ACN-data schema, NOT real observations). "
                   "Run `python scripts/download_data.py --token ...` for real ACN-Data.")
    else:
        st.success(f"Data source: {src}")


@st.cache_data(show_spinner=False)
def sessions() -> pd.DataFrame:
    return data_loader.load_processed_sessions()


@st.cache_data(show_spinner=False)
def hourly() -> pd.DataFrame:
    return data_loader.load_hourly_demand()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def eval_summary() -> pd.DataFrame | None:
    p = config.METRICS_DIR / "eval_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def eval_episodes() -> pd.DataFrame | None:
    p = config.METRICS_DIR / "eval_episodes.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_model_cached(algo: str, stamp: float):
    return trainer.load_trained(algo)


def model_stamp(algo: str) -> float:
    ps = [p.stat().st_mtime for p in (trainer.model_path(algo), trainer.model_path(algo, True)) if p.exists()]
    return max(ps) if ps else 0.0
