"""Gymnasium environment ``EVChargingEnv`` (discrete- and continuous-action variants).

* Time step: 15 minutes, one episode = one day (96 steps).
* Observation: flat ``Box`` in [-1, 1] (see ``observation_layout``).
* Actions
    - ``action_mode="continuous"`` : ``Box(-1, 1, (n_ports,))`` - mapped to a power fraction in [0, 1] of the
      maximum permissible rate of the EV on that port (symmetric bounds are what SB3 expects).
    - ``action_mode="discrete"``   : ``Discrete(L ** n_ports)`` - one of ``L`` power levels per port
      (default L = 4: off / low / medium / high), enumerated jointly (base-L digits).
* Constraints are enforced by :class:`src.ev_simulator.ChargingSim` (never by the reward alone).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ev_simulator import ChargingSim, Scenario, ScenarioGenerator, SimConfig
from src.reward_function import RewardConfig, TERM_NAMES, compute_reward

DEFAULT_LEVELS = (0.0, 1 / 3, 2 / 3, 1.0)     # off / low / medium / high (fraction of max rate)
EV_FEATURES = ["occupied", "arrival_time", "departure_time", "soc", "target_soc", "battery_capacity",
               "max_power", "remaining_time", "required_energy", "urgency", "current_power"]
PRICE_LOOKAHEAD_H = (0, 1, 2, 4, 8)


class EVChargingEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, sim_config: SimConfig | None = None, reward_config: RewardConfig | None = None,
                 action_mode: str = "continuous", levels=DEFAULT_LEVELS, render_mode: Optional[str] = None,
                 empirical_sessions=None, fixed_scenarios: list[Scenario] | None = None):
        super().__init__()
        if action_mode not in ("continuous", "discrete"):
            raise ValueError("action_mode must be 'continuous' or 'discrete'")
        self.cfg = sim_config or SimConfig()
        self.rcfg = reward_config or RewardConfig()
        self.action_mode = action_mode
        self.levels = np.asarray(levels, dtype=float)
        self.render_mode = render_mode
        self.generator = ScenarioGenerator(self.cfg, empirical_sessions)
        self.fixed_scenarios = fixed_scenarios
        self.P = self.cfg.n_ports
        self.n_stations_max = self.cfg.max_stations
        self.T = self.cfg.steps_per_day
        self.dt = self.cfg.dt_hours

        n_obs = self.P * len(EV_FEATURES) + 3 + len(PRICE_LOOKAHEAD_H) + 5 + 3 * self.n_stations_max
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(n_obs,), dtype=np.float32)
        if action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.P,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.levels) ** self.P)
        self.sim: ChargingSim | None = None
        self.scenario: Scenario | None = None
        self._ep_terms = {k: 0.0 for k in TERM_NAMES}
        self._ep_return = 0.0
        self._scenario_counter = 0

    # ----------------------------------------------------------------------------- actions
    def decode_action(self, action) -> np.ndarray:
        """Map a raw agent action to the fraction (0..1) of each port's maximum power."""
        if self.action_mode == "continuous":
            a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
            return (a + 1.0) / 2.0
        a = int(action)
        L = len(self.levels)
        digits = np.empty(self.P, dtype=int)
        for k in range(self.P):
            digits[k] = a % L
            a //= L
        return self.levels[digits]

    def fraction_to_action(self, frac: np.ndarray):
        """Inverse mapping used by baseline policies (continuous) - discrete uses the nearest level."""
        frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
        if self.action_mode == "continuous":
            return (frac * 2.0 - 1.0).astype(np.float32)
        L = len(self.levels)
        digits = np.abs(frac[:, None] - self.levels[None, :]).argmin(axis=1)
        return int(sum(int(d) * L ** k for k, d in enumerate(digits)))

    # ----------------------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if "scenario" in options:
            self.scenario = options["scenario"]
        elif self.fixed_scenarios:
            self.scenario = self.fixed_scenarios[self._scenario_counter % len(self.fixed_scenarios)]
            self._scenario_counter += 1
        else:
            self.scenario = self.generator.generate(self.np_random)
        # context from the demand forecaster (expected relative demand per hour of day)
        self.sim = ChargingSim(self.cfg, self.scenario)
        self._ep_terms = {k: 0.0 for k in TERM_NAMES}
        self._ep_return = 0.0
        self._last_frac = np.zeros(self.P)
        return self._obs(), {"scenario_seed": self.scenario.seed}

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")
        frac = self.decode_action(action)
        ps = self.sim.port_state()
        p_req = frac * ps["pmax"]
        self._last_frac = frac
        info = self.sim.step(p_req)
        reward, terms = compute_reward(self.rcfg, info, self.cfg.grid_capacity_kw, self.dt, self.T,
                                       max(self.scenario.n_evs, 1))
        for k, v in terms.items():
            self._ep_terms[k] += v
        self._ep_return += reward
        terminated = self.sim.done
        info_out = {"reward_terms": terms, "power_kw": info["port_power"], "price": info["price"],
                    "load_kw": info["load_kw"], "cost": info["cost"], "overload_kw": info["overload_kw"]}
        if terminated:
            info_out["episode_summary"] = self.sim.summary()
            info_out["episode_reward_terms"] = dict(self._ep_terms)
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), reward, terminated, False, info_out

    # ----------------------------------------------------------------------- observation
    def _hour_idx(self, t: int) -> int:
        return min(int(t * self.dt), 23)

    def _obs(self) -> np.ndarray:
        sim, sc, cfg, t, T = self.sim, self.scenario, self.cfg, self.sim.t, self.T
        tt = min(t, T - 1)
        ps = sim.port_state() if t < T else {"present": np.zeros(self.P, bool), "idx": np.zeros(self.P, int),
                                             "need_kwh": np.zeros(self.P), "pmax": np.zeros(self.P),
                                             "remaining_steps": np.zeros(self.P)}
        present, idx = ps["present"], ps["idx"]
        feats = np.zeros((self.P, len(EV_FEATURES)))
        if sc.n_evs and present.any():
            e = idx[present]
            rem_h = ps["remaining_steps"][present] * self.dt
            need = ps["need_kwh"][present]
            pm = np.maximum(ps["pmax"][present], 1e-6)
            urgency = need / np.maximum(pm * np.maximum(rem_h, self.dt), 1e-6)
            feats[present, 0] = 1.0
            feats[present, 1] = sc.arrival[e] / T
            feats[present, 2] = sc.departure[e] / T
            feats[present, 3] = sim.soc[e]
            feats[present, 4] = sc.soc_target[e]
            feats[present, 5] = sc.capacity_kwh[e] / 100.0
            feats[present, 6] = pm / cfg.port_max_power_kw
            feats[present, 7] = np.clip(rem_h / 12.0, 0, 1)
            feats[present, 8] = np.clip(need / 40.0, 0, 1)
            feats[present, 9] = np.clip(urgency / 2.0, 0, 1)
            feats[present, 10] = sim.last_port_power[present] / cfg.port_max_power_kw
        ang = 2 * np.pi * tt / T
        time_f = [np.sin(ang), np.cos(ang), tt / T]
        pmax_price = max(float(np.max(sc.price)), 1e-6)
        price_f = [sc.price[min(tt + int(round(h / self.dt)), T - 1)] / pmax_price for h in PRICE_LOOKAHEAD_H]
        prof = np.asarray(cfg.demand_profile_weekend if sc.weekend else cfg.demand_profile_weekday, float)
        if prof.size != 24:
            prof = np.zeros(24)
        hr = self._hour_idx(tt)
        load_prev = float(sim.load[t - 1]) if t > 0 else 0.0
        grid_f = [load_prev / cfg.grid_capacity_kw, cfg.grid_capacity_kw / 50.0, sim.peak / cfg.grid_capacity_kw,
                  prof[hr], prof[(hr + 1) % 24]]
        st_f = []
        for s in range(self.n_stations_max):
            ports = [p for p in range(self.P) if cfg.station_of_port(p) == s and sim.active[p]]
            if not ports:
                st_f += [0.0, 0.0, 0.0]
                continue
            avail = sum(1 for p in ports if not present[p]) / len(ports)
            pw = float(sim.last_port_power[ports].sum())
            cap_s = len(ports) * cfg.port_max_power_kw
            util = sum(1 for p in ports if present[p]) / len(ports)
            st_f += [avail, pw / cap_s, util]
        obs = np.concatenate([feats.ravel(), time_f, price_f, grid_f, st_f]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------------------ render
    def render(self):
        if self.sim is None:
            return None
        sim = self.sim
        t = min(sim.t, self.T - 1)
        lines = [f"t={sim.t:3d} ({sim.t * self.dt:5.2f} h)  price={sim.sc.price[t]:.3f} $/kWh  "
                 f"load={sim.load[sim.t - 1] if sim.t else 0:5.1f}/{self.cfg.grid_capacity_kw:.1f} kW"]
        ps = sim.port_state() if sim.t < self.T else None
        for p in range(self.P):
            if not sim.active[p]:
                continue
            if ps is not None and ps["present"][p]:
                i = ps["idx"][p]
                lines.append(f"  port {p}: EV{i} soc={sim.soc[i]:.2f}->{sim.sc.soc_target[i]:.2f} "
                             f"need={ps['need_kwh'][p]:5.1f} kWh  left={ps['remaining_steps'][p] * self.dt:4.1f} h  "
                             f"P={sim.last_port_power[p]:4.1f} kW")
            else:
                lines.append(f"  port {p}: free")
        text = "\n".join(lines)
        return text if self.render_mode in (None, "ansi") else text

    def close(self):
        self.sim = None


def make_env(action_mode: str = "continuous", seed: Optional[int] = None, sim_config: SimConfig | None = None,
             reward_config: RewardConfig | None = None, empirical_sessions=None, fixed_scenarios=None) -> EVChargingEnv:
    env = EVChargingEnv(sim_config, reward_config, action_mode=action_mode, empirical_sessions=empirical_sessions,
                        fixed_scenarios=fixed_scenarios)
    return env
