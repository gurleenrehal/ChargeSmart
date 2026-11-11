"""Tabular Q-Learning on a *simplified, discretised* EV-charging problem.

Why simplified?  A tabular method needs a small, discrete state space, so this module uses the same physics and
reward as ``EVChargingEnv`` but with a **single charging port** and a hand-made state discretisation:

    state = (time-of-day bin [12], price level [3], remaining-energy bin [5], urgency bin [5])   ->  900 states
    action = {0: off, 1: low, 2: medium, 3: high}  (fraction of the port's maximum power)

Theory implemented here
-----------------------
* **State-action value**  Q(s,a) = E[ sum_k gamma^k r_{t+k} | s_t = s, a_t = a ] under the greedy policy.
* **Bellman optimality**  Q*(s,a) = E[ r + gamma * max_a' Q*(s',a') ].
* **Temporal-difference update** (off-policy, bootstrapped):
      Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
  (no bootstrap term at the terminal step).
* **Exploration vs exploitation**: epsilon-greedy - with probability epsilon pick a random action, otherwise
  argmax_a Q(s,a); epsilon decays exponentially from ``eps_start`` to ``eps_end``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
from src.charging_environment import DEFAULT_LEVELS, EVChargingEnv
from src.ev_simulator import SimConfig
from src.reward_function import RewardConfig

N_TIME, N_PRICE, N_NEED, N_URG = 12, 3, 5, 5
N_STATES = N_TIME * N_PRICE * N_NEED * N_URG
N_ACTIONS = len(DEFAULT_LEVELS)
QTABLE_PATH = config.MODELS_DIR / "q_learning_qtable.npy"
QCONFIG_PATH = config.MODELS_DIR / "q_learning_config.json"


def simplified_config(base: SimConfig | None = None) -> SimConfig:
    """One station, one port, two EVs per day, grid capacity equal to one charger."""
    b = base or SimConfig()
    d = b.to_dict()
    d.update(n_stations=1, ports_per_station=1, n_evs=2, grid_capacity_kw=b.port_max_power_kw)
    return SimConfig.from_dict(d)


class SimplifiedEnv:
    """Thin wrapper exposing a 4-action, single-port view of :class:`EVChargingEnv` + a discrete state."""

    def __init__(self, sim_cfg: SimConfig | None = None, rcfg: RewardConfig | None = None):
        self.cfg = simplified_config(sim_cfg)
        self.env = EVChargingEnv(self.cfg, rcfg, action_mode="discrete")
        self.levels = np.asarray(DEFAULT_LEVELS)

    def reset(self, seed=None, scenario=None):
        _, info = self.env.reset(seed=seed, options={"scenario": scenario} if scenario is not None else None)
        return self.state()

    def step(self, a: int):
        # digit 0 -> port 0 gets level ``a``; all other (inactive) ports are ignored by the simulator
        _, r, done, _, info = self.env.step(int(a))
        return self.state(), r, done, info

    def state(self) -> int:
        sim = self.env.sim
        t = min(sim.t, sim.T - 1)
        ps = sim.port_state()
        t_bin = min(int(t * sim.dt / 2.0), N_TIME - 1)
        price = sim.sc.price[t]
        rel = price / max(float(np.max(sim.sc.price)), 1e-9)
        p_bin = 0 if rel < 0.45 else (1 if rel < 0.75 else 2)
        if sim.t >= sim.T or not ps["present"][0]:
            need_bin = urg_bin = 0
        else:
            i = ps["idx"][0]
            frac = ps["need_kwh"][0] / max(sim.requested[i], 1e-6)
            need_bin = 0 if ps["need_kwh"][0] <= 1e-6 else 1 + min(int(frac * 4), 3)
            rem_h = max(ps["remaining_steps"][0] * sim.dt, sim.dt)
            urg = ps["need_kwh"][0] / max(ps["pmax"][0] * rem_h, 1e-6)
            urg_bin = 0 if urg < 0.2 else 1 if urg < 0.4 else 2 if urg < 0.6 else 3 if urg < 0.85 else 4
        return ((t_bin * N_PRICE + p_bin) * N_NEED + need_bin) * N_URG + urg_bin


@dataclass
class QLearningConfig:
    episodes: int = 6000
    alpha: float = 0.1
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.9985     # multiplicative, per episode
    seed: int = config.GLOBAL_SEED


def train_q_learning(qcfg: QLearningConfig | None = None, sim_cfg: SimConfig | None = None,
                     rcfg: RewardConfig | None = None, log_every: int = 100, save: bool = True,
                     progress_cb=None) -> tuple[np.ndarray, pd.DataFrame]:
    qcfg = qcfg or QLearningConfig()
    rng = np.random.default_rng(qcfg.seed)
    env = SimplifiedEnv(sim_cfg, rcfg)
    Q = np.zeros((N_STATES, N_ACTIONS))
    eps = qcfg.eps_start
    rows = []
    for ep in range(qcfg.episodes):
        s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done, ret = False, 0.0
        while not done:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))   # epsilon-greedy
            s2, r, done, info = env.step(a)
            target = r if done else r + qcfg.gamma * Q[s2].max()                              # Bellman target
            Q[s, a] += qcfg.alpha * (target - Q[s, a])                                          # TD update
            s, ret = s2, ret + r
        summ = info["episode_summary"]
        rows.append({"episode": ep, "reward": ret, "cost": summ["total_cost"], "epsilon": eps,
                     "pct_fully_charged": summ["pct_fully_charged"]})
        eps = max(qcfg.eps_end, eps * qcfg.eps_decay)
        if progress_cb and ep % log_every == 0:
            progress_cb(ep + 1, qcfg.episodes, float(np.mean([r["reward"] for r in rows[-log_every:]])))
    log = pd.DataFrame(rows)
    if save:
        np.save(QTABLE_PATH, Q)
        QCONFIG_PATH.write_text(json.dumps(asdict(qcfg), indent=2))
        log.to_csv(config.METRICS_DIR / "train_log_q_learning.csv", index=False)
    return Q, log


class QPolicy:
    """Greedy policy from a trained Q-table (runs on the simplified single-port environment)."""
    name = "Q-Learning"

    def __init__(self, Q: np.ndarray):
        self.Q = Q

    def act(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))


def evaluate_on_simplified(policy, seeds, sim_cfg=None, rcfg=None) -> pd.DataFrame:
    """``policy`` is a QPolicy or a baseline (which uses the continuous interface on the same 1-port physics)."""
    from src.baseline_policies import BaselinePolicy
    from src.evaluator import run_episode

    scfg = simplified_config(sim_cfg)
    rows = []
    for sd in seeds:
        if isinstance(policy, QPolicy):
            env = SimplifiedEnv(sim_cfg, rcfg)
            sc = env.env.generator.generate(int(sd))
            s = env.reset(scenario=sc)
            done, ret = False, 0.0
            while not done:
                s, r, done, info = env.step(policy.act(s))
                ret += r
            d = dict(info["episode_summary"]); d["episode_reward"] = ret
        else:
            from src.ev_simulator import ScenarioGenerator
            sc = ScenarioGenerator(scfg).generate(int(sd))
            res = run_episode(policy, sc, scfg, rcfg, record_decisions=False)
            d = dict(res.summary)
        d.update(policy=policy.name, seed=int(sd))
        rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    Q, log = train_q_learning(QLearningConfig(episodes=300), log_every=50,
                              progress_cb=lambda e, n, m: print(f"ep {e}/{n} mean reward {m:.2f}"))
    print("done", time.time() - t0)
