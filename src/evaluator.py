"""Evaluation of charging policies on identical scenarios."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import config
from src.baseline_policies import BASELINES, get_baseline
from src.charging_environment import EVChargingEnv
from src.ev_simulator import Scenario, ScenarioGenerator, SimConfig
from src.reward_function import RewardConfig, TERM_NAMES

METRIC_COLUMNS = [
    "total_cost", "avg_cost_per_ev", "peak_kw", "peak_to_average_ratio", "grid_utilization_pct",
    "pct_fully_charged", "avg_unmet_kwh", "avg_charging_duration_h", "station_utilization_pct",
    "total_energy_kwh", "deadline_violations", "grid_overload_events", "fairness_index", "episode_reward",
]


class RLPolicy:
    """Wraps a Stable-Baselines3 model (deterministic inference)."""

    def __init__(self, model, algo: str, action_mode: str):
        self.model, self.name, self.action_mode = model, algo, action_mode

    def reset(self):
        pass

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action) if self.action_mode == "discrete" else action


@dataclass
class EpisodeResult:
    policy: str
    summary: dict
    total_reward: float
    reward_terms: dict
    scenario: Scenario
    trace: dict = field(default_factory=dict)


def make_eval_env(policy, sim_cfg: SimConfig, rcfg: RewardConfig, scenarios=None) -> EVChargingEnv:
    return EVChargingEnv(sim_cfg, rcfg, action_mode=policy.action_mode, fixed_scenarios=scenarios)


def run_episode(policy, scenario: Scenario, sim_cfg: SimConfig | None = None, rcfg: RewardConfig | None = None,
                record_decisions: bool = True) -> EpisodeResult:
    sim_cfg, rcfg = sim_cfg or SimConfig(), rcfg or RewardConfig()
    env = make_eval_env(policy, sim_cfg, rcfg)
    obs, _ = env.reset(options={"scenario": scenario})
    policy.reset()
    total, done, decisions = 0.0, False, []
    info = {}
    while not done:
        t = env.sim.t
        action = policy.act(env, obs)
        frac = env.decode_action(action)
        obs, r, done, _, info = env.step(action)
        total += r
        if record_decisions:
            decisions.append({"t": t, "hour": t * sim_cfg.dt_hours, "price": info["price"],
                              **{f"frac_port{p}": float(frac[p]) for p in range(env.P)},
                              **{f"kw_port{p}": float(info["power_kw"][p]) for p in range(env.P)},
                              "load_kw": info["load_kw"], "reward": r})
    sim = env.sim
    summary = sim.summary()
    summary["episode_reward"] = total
    trace = {"load": sim.load.copy(), "load_requested": sim.load_requested.copy(), "price": scenario.price.copy(),
             "cost": sim.cost.copy(), "port_power": sim.port_power.copy(), "ev_power": sim.ev_power.copy(),
             "ev_soc": sim.ev_soc.copy(), "final_unmet": sim.final_unmet.copy(), "decisions": decisions}
    return EpisodeResult(policy.name, summary, total, info["episode_reward_terms"], scenario, trace)


def make_scenarios(sim_cfg: SimConfig, seeds: Iterable[int], empirical_sessions=None) -> list[Scenario]:
    gen = ScenarioGenerator(sim_cfg, empirical_sessions)
    return [gen.generate(int(s)) for s in seeds]


def evaluate_policy(policy, scenarios: list[Scenario], sim_cfg: SimConfig | None = None,
                    rcfg: RewardConfig | None = None) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        res = run_episode(policy, sc, sim_cfg, rcfg, record_decisions=False)
        d = dict(res.summary)
        d.update({f"term_{k}": v for k, v in res.reward_terms.items()})
        d.update({"policy": policy.name, "seed": sc.seed})
        rows.append(d)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and 95 % confidence interval (normal approx.) per policy for every metric."""
    cols = [c for c in METRIC_COLUMNS if c in df.columns]
    out = []
    for name, g in df.groupby("policy", sort=False):
        row = {"policy": name, "n_episodes": len(g)}
        for c in cols:
            m, s = g[c].mean(), g[c].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{c}_mean"], row[f"{c}_std"] = m, s
            row[f"{c}_ci95"] = 1.96 * s / np.sqrt(len(g)) if len(g) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def pretty_table(agg: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or ["total_cost", "peak_kw", "peak_to_average_ratio", "pct_fully_charged", "avg_unmet_kwh",
                    "grid_utilization_pct", "fairness_index", "deadline_violations", "grid_overload_events",
                    "episode_reward"]
    t = pd.DataFrame({"policy": agg["policy"]})
    for c in cols:
        if f"{c}_mean" in agg:
            t[c] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    return t


def evaluate_baselines(scenarios, sim_cfg=None, rcfg=None) -> pd.DataFrame:
    frames = [evaluate_policy(get_baseline(n), scenarios, sim_cfg, rcfg) for n in BASELINES]
    return pd.concat(frames, ignore_index=True)


def evaluate_models(models: dict, scenarios, sim_cfg=None, rcfg=None) -> pd.DataFrame:
    """``models``: {algo_name: (sb3_model, action_mode)}."""
    frames = []
    for name, (model, mode) in models.items():
        frames.append(evaluate_policy(RLPolicy(model, name, mode), scenarios, sim_cfg, rcfg))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
