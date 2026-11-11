"""Physical EV-charging simulator (pure NumPy, no RL dependencies).

The simulator owns *all* physics and constraints:

* a charging port can only serve the EV currently connected to it (arrival <= t < departure);
* per-EV power <= min(EV acceptance rate, port limit);
* charging stops at the EV's target state of charge (no energy is wasted above the request);
* the total power drawn is **hard-limited** to the grid capacity (proportional curtailment);
* no charging after departure.

Real vs simulated: when ``SimConfig.arrival_pattern == "empirical"`` arrival times, dwell times and
energy demand are *resampled from real/processed ACN-Data sessions*. Battery capacity, state of charge,
EV charging limits, tariff and grid capacity are always simulation parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import numpy as np
import pandas as pd

# Piece-wise constant hourly time-of-use tariff in $/kWh (simulation assumption, not from ACN-Data):
# cheap solar-hours dip (10-16h), expensive evening peak (16-21h).
DEFAULT_TOU_PRICE = [0.14] * 6 + [0.20] * 4 + [0.10] * 6 + [0.35] * 5 + [0.20] * 3

ARRIVAL_PATTERNS = ("mixed", "workplace", "evening_peak", "uniform", "empirical")


@dataclass
class SimConfig:
    # --- layout (fixed padded layout: max_stations x max_ports_per_station slots) ---------
    n_stations: int = 2
    ports_per_station: int = 2
    max_stations: int = 2
    max_ports_per_station: int = 2
    # --- time -----------------------------------------------------------------------------
    dt_hours: float = 0.25
    steps_per_day: int = 96
    # --- grid / charger limits ------------------------------------------------------------
    grid_capacity_kw: float = 14.0
    port_max_power_kw: float = 7.2
    # --- tariff ---------------------------------------------------------------------------
    price_profile: list = field(default_factory=lambda: list(DEFAULT_TOU_PRICE))
    price_scale_noise: float = 0.10     # per-episode multiplicative tariff noise (+/-)
    price_hourly_jitter: float = 0.05
    # --- EV population --------------------------------------------------------------------
    n_evs: int = 8
    arrival_pattern: str = "mixed"
    dwell_scale: float = 1.0            # multiplies sampled parking durations ("departure deadlines")
    battery_choices_kwh: list = field(default_factory=lambda: [35.0, 50.0, 60.0, 75.0])
    ev_power_choices_kw: list = field(default_factory=lambda: [3.7, 6.6, 7.2])
    soc0_range: tuple = (0.40, 0.70)
    soc_target_range: tuple = (0.75, 0.90)
    min_dwell_h: float = 1.5
    max_dwell_h: float = 12.0
    feasibility_margin: float = 0.85    # requested energy <= margin * pmax * dwell (before grid limits)
    # --- context for the RL agent ----------------------------------------------------------
    demand_profile_weekday: list = field(default_factory=lambda: [0.0] * 24)
    demand_profile_weekend: list = field(default_factory=lambda: [0.0] * 24)
    # --- misc ------------------------------------------------------------------------------
    weekend_probability: float = 0.0

    @property
    def n_ports(self) -> int:
        return self.max_stations * self.max_ports_per_station

    def active_ports(self) -> np.ndarray:
        """Boolean mask over the padded port layout selecting the configured stations/ports."""
        mask = np.zeros(self.n_ports, dtype=bool)
        for p in range(self.n_ports):
            st, slot = divmod(p, self.max_ports_per_station)
            mask[p] = st < self.n_stations and slot < self.ports_per_station
        return mask

    def station_of_port(self, p: int) -> int:
        return p // self.max_ports_per_station

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Scenario:
    """One simulated day: EV sessions + prices. Identical scenarios are replayed for every policy."""
    arrival: np.ndarray        # step index
    departure: np.ndarray      # step index (exclusive), <= steps_per_day
    port: np.ndarray
    capacity_kwh: np.ndarray
    soc0: np.ndarray
    soc_target: np.ndarray
    pmax_kw: np.ndarray        # min(EV acceptance, port limit)
    price: np.ndarray          # length steps_per_day, $/kWh
    weekend: bool = False
    n_rejected: int = 0
    seed: Optional[int] = None

    @property
    def n_evs(self) -> int:
        return len(self.arrival)

    def to_frame(self, dt_hours: float = 0.25) -> pd.DataFrame:
        return pd.DataFrame({
            "ev": np.arange(self.n_evs), "port": self.port,
            "arrival_h": self.arrival * dt_hours, "departure_h": self.departure * dt_hours,
            "battery_kwh": self.capacity_kwh, "soc_initial": self.soc0, "soc_target": self.soc_target,
            "max_power_kw": self.pmax_kw,
            "energy_needed_kwh": (self.soc_target - self.soc0) * self.capacity_kwh,
        })


# ----------------------------------------------------------------------------- scenarios
class ScenarioGenerator:
    """Seeded generator of EV arrival/departure scenarios."""

    def __init__(self, cfg: SimConfig, empirical_sessions: pd.DataFrame | None = None):
        self.cfg = cfg
        self.empirical = empirical_sessions

    def _sample_pattern(self, rng: np.random.Generator, n: int):
        cfg = self.cfg
        pat = cfg.arrival_pattern
        if pat == "empirical" and self.empirical is not None and len(self.empirical) > 0:
            rows = self.empirical.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
            return (rows["arrival_hour"].to_numpy(float), rows["duration_h"].to_numpy(float),
                    rows["kWhDelivered"].to_numpy(float))
        if pat == "workplace":
            comp = rng.choice(3, size=n, p=[0.7, 0.2, 0.1])
            mu = np.array([8.5, 12.5, 17.5])[comp]; sd = np.array([1.2, 2.0, 1.5])[comp]
            dmu = np.array([8.0, 3.5, 2.0])[comp]; dsd = np.array([1.5, 1.2, 0.8])[comp]
        elif pat == "evening_peak":
            mu, sd, dmu, dsd = np.full(n, 17.0), np.full(n, 1.5), np.full(n, 4.0), np.full(n, 1.0)
        elif pat == "uniform":
            mu = rng.uniform(5, 20, n); sd = np.zeros(n); dmu = np.full(n, 4.5); dsd = np.full(n, 1.5)
        else:  # "mixed" (default): morning commuters, midday visitors, evening visitors
            comp = rng.choice(3, size=n, p=[0.40, 0.35, 0.25])
            mu = np.array([8.0, 12.5, 17.0])[comp]; sd = np.array([1.5, 2.0, 1.5])[comp]
            dmu = np.array([6.0, 4.0, 3.0])[comp]; dsd = np.array([1.5, 1.0, 1.0])[comp]
        arr = rng.normal(mu, sd)
        dwell = rng.normal(dmu, dsd)
        return arr, dwell, None

    def generate(self, seed: int | np.random.Generator | None = None) -> Scenario:
        cfg = self.cfg
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        T, dt = cfg.steps_per_day, cfg.dt_hours
        n = cfg.n_evs
        arr_h, dwell_h, kwh = self._sample_pattern(rng, n)
        arr_h = np.clip(arr_h, 0.0, 23.0 - 0.25)
        dwell_h = np.clip(dwell_h * cfg.dwell_scale, cfg.min_dwell_h, cfg.max_dwell_h)
        order = np.argsort(arr_h)
        arr_h, dwell_h = arr_h[order], dwell_h[order]
        kwh = kwh[order] if kwh is not None else None

        arrival = np.floor(arr_h / dt).astype(int)
        departure = np.minimum(arrival + np.ceil(dwell_h / dt).astype(int), T)
        keep_dwell = departure - arrival >= int(np.ceil(cfg.min_dwell_h / dt) / 2) + 1

        active = np.flatnonzero(cfg.active_ports())
        port_free_at = {int(p): 0 for p in active}
        recs, rejected = [], 0
        for i in range(n):
            if not keep_dwell[i]:
                rejected += 1
                continue
            free = [p for p, t_free in port_free_at.items() if t_free <= arrival[i]]
            if not free:
                rejected += 1
                continue
            p = int(rng.choice(free))
            port_free_at[p] = int(departure[i])
            cap = float(rng.choice(cfg.battery_choices_kwh))
            ev_pmax = float(rng.choice(cfg.ev_power_choices_kw))
            pmax = min(ev_pmax, cfg.port_max_power_kw)
            target = float(rng.uniform(*cfg.soc_target_range))
            dwell_eff = (departure[i] - arrival[i]) * dt
            if kwh is not None:                       # empirical energy demand (real data)
                need = float(np.clip(kwh[i], 2.0, 0.95 * cap * 0.8))
                soc0 = float(np.clip(target - need / cap, 0.05, target - 0.05))
            else:
                soc0 = float(rng.uniform(*cfg.soc0_range))
                soc0 = min(soc0, target - 0.05)
            need = (target - soc0) * cap
            max_need = cfg.feasibility_margin * pmax * dwell_eff        # keep sessions physically feasible
            if need > max_need:
                soc0 = target - max_need / cap
            recs.append((arrival[i], departure[i], p, cap, soc0, target, pmax))
        if recs:
            a = np.array(recs, dtype=float)
            arrival_a, dep_a, port_a, cap_a, soc0_a, tgt_a, pmax_a = a.T
        else:
            arrival_a = dep_a = port_a = cap_a = soc0_a = tgt_a = pmax_a = np.zeros(0)

        # tariff: hourly profile, per-episode scale + per-hour jitter, repeated per sub-hour step
        base = np.asarray(cfg.price_profile, dtype=float)
        scale = 1.0 + rng.uniform(-cfg.price_scale_noise, cfg.price_scale_noise)
        hourly = base * scale * (1.0 + rng.uniform(-cfg.price_hourly_jitter, cfg.price_hourly_jitter, 24))
        steps_per_hour = int(round(1.0 / dt))
        price = np.repeat(hourly, steps_per_hour)[:T]
        weekend = bool(rng.random() < cfg.weekend_probability)
        return Scenario(arrival=arrival_a.astype(int), departure=dep_a.astype(int), port=port_a.astype(int),
                        capacity_kwh=cap_a, soc0=soc0_a, soc_target=tgt_a, pmax_kw=pmax_a,
                        price=price, weekend=weekend, n_rejected=rejected,
                        seed=seed if isinstance(seed, (int, np.integer)) else None)


# ------------------------------------------------------------------------------ simulator
class ChargingSim:
    """Time-stepped physics. ``step`` takes requested kW per port and returns what actually happened."""

    DEPARTURE_TOL_KWH = 0.1

    def __init__(self, cfg: SimConfig, scenario: Scenario):
        self.cfg = cfg
        self.sc = scenario
        self.P = cfg.n_ports
        self.T = cfg.steps_per_day
        self.dt = cfg.dt_hours
        self.active = cfg.active_ports()
        self.reset()

    # -- lifecycle ------------------------------------------------------------------------
    def reset(self) -> None:
        sc, T, P = self.sc, self.T, self.P
        self.t = 0
        self.soc = sc.soc0.copy()
        self.delivered = np.zeros(sc.n_evs)
        self.requested = (sc.soc_target - sc.soc0) * sc.capacity_kwh
        self.charge_steps = np.zeros(sc.n_evs)
        self.departed = np.zeros(sc.n_evs, dtype=bool)
        self.final_unmet = np.full(sc.n_evs, np.nan)
        # occupancy table: occ[t, port] = ev index or -1
        self.occ = -np.ones((T + 1, P), dtype=int)
        for i in range(sc.n_evs):
            self.occ[sc.arrival[i]:sc.departure[i], sc.port[i]] = i
        self.dep_at = [np.flatnonzero(sc.departure == k) for k in range(T + 1)]
        # logs
        self.load = np.zeros(T)
        self.load_requested = np.zeros(T)
        self.cost = np.zeros(T)
        self.port_power = np.zeros((T, P))
        self.ev_power = np.zeros((sc.n_evs, T))
        self.ev_soc = np.full((sc.n_evs, T + 1), np.nan)
        for i in range(sc.n_evs):
            self.ev_soc[i, sc.arrival[i]] = sc.soc0[i]
        self.peak = 0.0
        self.last_port_power = np.zeros(P)

    @property
    def done(self) -> bool:
        return self.t >= self.T

    # -- helpers --------------------------------------------------------------------------
    def occupant(self, t: int | None = None) -> np.ndarray:
        return self.occ[self.t if t is None else t]

    def need_kwh(self) -> np.ndarray:
        return np.maximum((self.sc.soc_target - self.soc) * self.sc.capacity_kwh, 0.0)

    def port_state(self) -> dict:
        """Per-port arrays for the current step (zeros for empty ports)."""
        occ = self.occupant()
        present = occ >= 0
        idx = np.where(present, occ, 0)
        sc = self.sc
        need = np.where(present, self.need_kwh()[idx] if sc.n_evs else 0.0, 0.0)
        pmax = np.where(present, sc.pmax_kw[idx] if sc.n_evs else 0.0, 0.0)
        remaining_steps = np.where(present, sc.departure[idx] - self.t if sc.n_evs else 0, 0)
        return {"present": present, "idx": idx, "need_kwh": need, "pmax": pmax,
                "remaining_steps": remaining_steps}

    # -- dynamics -------------------------------------------------------------------------
    def step(self, p_req_kw: np.ndarray) -> dict:
        """Advance one interval. ``p_req_kw`` is the requested power per port (any values; they are
        clipped to the physical limits here)."""
        if self.done:
            raise RuntimeError("Episode finished; call reset().")
        sc, dt, t = self.sc, self.dt, self.t
        ps = self.port_state()
        present, idx = ps["present"], ps["idx"]

        p = np.clip(np.nan_to_num(np.asarray(p_req_kw, dtype=float)), 0.0, None)
        p = np.minimum(p, ps["pmax"])                          # charger / vehicle limit
        p = np.minimum(p, ps["need_kwh"] / dt)                 # stop at target SoC
        p = np.where(present, p, 0.0)                          # nothing flows into empty ports
        total_req = float(p.sum())
        overload = max(0.0, total_req - self.cfg.grid_capacity_kw)
        if total_req > self.cfg.grid_capacity_kw:              # hard grid constraint (curtailment)
            p = p * (self.cfg.grid_capacity_kw / total_req)
        energy = p * dt
        e_total = float(energy.sum())

        # update batteries
        if present.any():
            ev = idx[present]
            self.soc[ev] += energy[present] / sc.capacity_kwh[ev]
            self.delivered[ev] += energy[present]
            self.charge_steps[ev] += (energy[present] > 1e-9)
            self.ev_power[ev, t] = p[present]
            self.ev_soc[ev, t + 1] = self.soc[ev]
        price = float(sc.price[t])
        cost = e_total * price
        load = float(p.sum())
        peak_prev = self.peak
        self.peak = max(self.peak, load)
        self.load[t], self.load_requested[t], self.cost[t] = load, total_req, cost
        self.port_power[t] = p
        self.last_port_power = p

        # fairness over EVs currently connected and with a non-zero request
        fair_idx = 1.0
        if present.sum() >= 2:
            ev = idx[present]
            x = self.delivered[ev] / np.maximum(self.requested[ev], 1e-6)
            if x.sum() > 1e-9:
                fair_idx = float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))

        # advance time, departures
        self.t += 1
        departed_unmet, n_success, n_violation = 0.0, 0, 0
        for i in self.dep_at[self.t]:
            unmet = float(max((sc.soc_target[i] - self.soc[i]) * sc.capacity_kwh[i], 0.0))
            self.final_unmet[i] = unmet
            self.departed[i] = True
            departed_unmet += unmet
            if unmet <= self.DEPARTURE_TOL_KWH:
                n_success += 1
            else:
                n_violation += 1
        return {
            "t": t, "price": price, "energy_kwh": e_total, "cost": cost, "load_kw": load,
            "requested_kw": total_req, "overload_kw": overload, "peak_prev": peak_prev, "peak": self.peak,
            "departed_unmet_kwh": departed_unmet, "n_success": n_success, "n_violation": n_violation,
            "fairness_index": fair_idx, "port_power": p.copy(),
        }

    # -- episode summary ------------------------------------------------------------------
    def summary(self) -> dict:
        sc, T, dt = self.sc, self.t, self.dt
        n = sc.n_evs
        unmet = np.where(np.isnan(self.final_unmet), self.need_kwh(), self.final_unmet)
        load = self.load[:T]
        n_ports_active = int(self.active.sum())
        port_time = max(n_ports_active * T, 1)
        occupied = (self.occ[:T][:, self.active] >= 0).sum()
        charging = (self.port_power[:T][:, self.active] > 1e-9).sum()
        ratio = self.delivered / np.maximum(self.requested, 1e-6) if n else np.array([])
        jain = float(ratio.sum() ** 2 / (n * (ratio ** 2).sum())) if n and (ratio ** 2).sum() > 0 else 1.0
        mean_load = float(load.mean()) if T else 0.0
        return {
            "n_evs": int(n),
            "n_rejected": int(sc.n_rejected),
            "total_cost": float(self.cost[:T].sum()),
            "avg_cost_per_ev": float(self.cost[:T].sum() / n) if n else 0.0,
            "peak_kw": float(load.max()) if T else 0.0,
            "peak_to_average_ratio": float(load.max() / mean_load) if mean_load > 1e-9 else float("nan"),
            "grid_utilization_pct": 100 * mean_load / self.cfg.grid_capacity_kw,
            "peak_grid_utilization_pct": 100 * (float(load.max()) if T else 0.0) / self.cfg.grid_capacity_kw,
            "pct_fully_charged": float(100 * np.mean(unmet <= self.DEPARTURE_TOL_KWH)) if n else 100.0,
            "avg_unmet_kwh": float(unmet.mean()) if n else 0.0,
            "total_unmet_kwh": float(unmet.sum()),
            "avg_charging_duration_h": float(self.charge_steps.mean() * dt) if n else 0.0,
            "station_utilization_pct": float(100 * occupied / port_time),
            "charging_utilization_pct": float(100 * charging / port_time),
            "total_energy_kwh": float(self.delivered.sum()),
            "requested_energy_kwh": float(self.requested.sum()),
            "deadline_violations": int(np.sum(unmet > self.DEPARTURE_TOL_KWH)),
            "grid_overload_events": int(np.sum(self.load_requested[:T] > self.cfg.grid_capacity_kw + 1e-6)),
            "fairness_index": jain,
        }
