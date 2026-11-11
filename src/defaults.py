"""Shared default configurations (keeps scripts, trainer, evaluator and dashboard consistent)."""
from __future__ import annotations

import pandas as pd

import config
from src.ev_simulator import SimConfig
from src.reward_function import RewardConfig


def default_sim_config(**overrides) -> SimConfig:
    """Default simulation parameters + the demand-forecast profile (if the forecaster has been run)."""
    from src import demand_forecasting

    cfg = SimConfig()
    prof = demand_forecasting.load_demand_profile()
    if prof:
        cfg.demand_profile_weekday = prof["weekday"]
        cfg.demand_profile_weekend = prof["weekend"]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def default_reward_config(**overrides) -> RewardConfig:
    r = RewardConfig()
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


def load_empirical_sessions() -> pd.DataFrame | None:
    """Processed sessions (real if ACN-Data was downloaded, else the synthetic sample) for 'empirical' mode."""
    if not config.SESSIONS_CSV.exists():
        return None
    df = pd.read_csv(config.SESSIONS_CSV, usecols=["arrival_hour", "duration_h", "kWhDelivered"])
    return df
