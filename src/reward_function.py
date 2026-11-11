"""Reward function for EVChargingEnv (fully configurable, decomposed).

Per-step reward (before the global ``reward_scale``)::

    R(t) = - [ a*Cost(t) + b*Peak(t) + g*Unmet(t) + d*Overload(t) + e*Fairness(t) + xi*Violations(t) ]
           + g*Progress(t) + w*Successes(t)

Terms (all dimension-less, normalised so a whole day has O(1) magnitude):

* **Cost(t)**      = price_t * energy_t / (price_ref * E_day_ref).  Sum over a day ~ average price / reference price.
* **Peak(t)**      = max(0, load_t - peak_{t-1}) / grid_capacity.  It telescopes: the episode sum equals
                     peak_load / capacity, i.e. the cost of the *final* peak is split over the steps that created it.
* **Unmet(t)**     = sum of the energy shortfall of EVs that depart at t / E_day_ref.   (sparse, terminal)
* **Progress(t)**  = useful energy delivered at t / E_day_ref (dense shaping).  For one EV the two terms
                     together give  -g*(2*U_dep - U_0)/E_day_ref: the *optimal policy is the one that minimises
                     the final shortfall U_dep*, the dense part only gives immediate credit assignment. Because
                     progress is paid whenever energy is delivered, staying idle to save money is never rewarded.
* **Overload(t)**  = max(0, requested_total - capacity) / capacity * dt_hours.  Requests above the grid capacity
                     are curtailed by the simulator, but the agent is still penalised for asking.
* **Fairness(t)**  = (1 - Jain index of fulfilment ratios of connected EVs) / steps_per_day.
* **Violations**   = number of EVs leaving with unmet energy (> tolerance);  **Successes** = EVs leaving fully charged.

All weights are fields of :class:`RewardConfig`; ``reward_scale`` multiplies everything uniformly (helps
gradient-based learners, does not change the optimal policy).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class RewardConfig:
    alpha_cost: float = 1.0          # electricity cost
    beta_peak: float = 0.5           # new-peak penalty
    gamma_unmet: float = 3.0         # unmet energy at departure (+ matching dense progress reward)
    delta_overload: float = 2.0      # requested power above grid capacity
    epsilon_fairness: float = 0.3    # 1 - Jain index
    xi_deadline: float = 0.3         # fixed penalty per deadline violation
    omega_success: float = 0.2       # bonus per EV fully charged before departure
    reward_scale: float = 10.0       # uniform scaling of the final reward
    energy_ref_kwh: float = 13.0     # reference energy per EV (E_ref); E_day_ref = n_evs * E_ref
    price_ref: float = 0.35          # reference price ($/kWh)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


TERM_NAMES = ["electricity_cost", "peak_demand", "unmet_energy", "grid_overload",
              "fairness", "deadline_violation", "charging_progress", "departure_success"]


def compute_reward(cfg: RewardConfig, info: dict, grid_capacity_kw: float, dt_hours: float,
                   steps_per_day: int, n_evs: int) -> tuple[float, dict]:
    """Return ``(scaled_reward, terms)``; ``terms`` holds the *signed* contribution of each term
    (negative = penalty) already multiplied by its weight and by ``reward_scale``."""
    e_day = max(n_evs, 1) * cfg.energy_ref_kwh
    s = cfg.reward_scale
    cost = info["cost"] / (cfg.price_ref * e_day)
    peak = max(0.0, info["load_kw"] - info["peak_prev"]) / grid_capacity_kw
    unmet = info["departed_unmet_kwh"] / e_day
    overload = info["overload_kw"] / grid_capacity_kw * dt_hours
    fairness = (1.0 - info["fairness_index"]) / steps_per_day
    progress = info["energy_kwh"] / e_day
    terms = {
        "electricity_cost": -cfg.alpha_cost * cost * s,
        "peak_demand": -cfg.beta_peak * peak * s,
        "unmet_energy": -cfg.gamma_unmet * unmet * s,
        "grid_overload": -cfg.delta_overload * overload * s,
        "fairness": -cfg.epsilon_fairness * fairness * s,
        "deadline_violation": -cfg.xi_deadline * info["n_violation"] * s,
        "charging_progress": cfg.gamma_unmet * progress * s,
        "departure_success": cfg.omega_success * info["n_success"] * s,
    }
    return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}
