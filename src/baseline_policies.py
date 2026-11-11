"""Traditional (non-learning) charging policies.

Every policy sees exactly the same information the simulator exposes (connected EVs, their energy needs
and deadlines, the tariff schedule, the grid capacity) and returns the power it *wants* per port in kW.
The simulator then enforces the physical limits, so all policies run under identical constraints.
"""
from __future__ import annotations

import numpy as np

from src.ev_simulator import ChargingSim


def _allocate(order: list[int], desired: np.ndarray, cap_kw: float) -> np.ndarray:
    """Give power to ports in priority order until ``cap_kw`` is exhausted."""
    out = np.zeros_like(desired)
    left = cap_kw
    for p in order:
        give = min(desired[p], left)
        out[p] = max(give, 0.0)
        left -= out[p]
        if left <= 1e-9:
            break
    return out


def _min_steps_needed(need_kwh: np.ndarray, pmax: np.ndarray, dt: float) -> np.ndarray:
    return np.ceil(need_kwh / np.maximum(pmax * dt, 1e-9) - 1e-9)


class BaselinePolicy:
    name = "baseline"
    action_mode = "continuous"

    def reset(self) -> None:  # stateless by default
        pass

    def power_request(self, sim: ChargingSim) -> np.ndarray:  # kW per port
        raise NotImplementedError

    def act(self, env, obs=None):
        sim = env.sim
        ps = sim.port_state()
        p = self.power_request(sim)
        frac = np.where(ps["pmax"] > 0, p / np.maximum(ps["pmax"], 1e-9), 0.0)
        return env.fraction_to_action(np.clip(frac, 0, 1))


class UncontrolledCharging(BaselinePolicy):
    """Every connected EV charges immediately at its maximum power. No coordination: if the sum exceeds the
    grid capacity the simulator curtails all vehicles proportionally (and counts an overload event)."""
    name = "Uncontrolled"

    def power_request(self, sim):
        return sim.port_state()["pmax"].copy()


class FirstComeFirstServed(BaselinePolicy):
    """Serve vehicles in arrival order at full power until the grid capacity is used up."""
    name = "FCFS"

    def power_request(self, sim):
        ps = sim.port_state()
        ports = [p for p in range(sim.P) if ps["present"][p]]
        ports.sort(key=lambda p: sim.sc.arrival[ps["idx"][p]])
        return _allocate(ports, ps["pmax"], sim.cfg.grid_capacity_kw)


class TimeOfUseCharging(BaselinePolicy):
    """Charge in the cheapest intervals before each vehicle's own departure.

    For every EV: r = steps needed at full power; the vehicle charges now if the current interval is among
    the ``r + margin`` cheapest intervals of its remaining window (or if it has no slack left).
    Vehicles wanting to charge are served earliest-departure-first within the grid capacity."""
    name = "Time-of-Use"

    def __init__(self, margin_steps: int = 1):
        self.margin = margin_steps

    def power_request(self, sim):
        ps = sim.port_state()
        t, dt, sc = sim.t, sim.dt, sim.sc
        want = np.zeros(sim.P)
        r_all = _min_steps_needed(ps["need_kwh"], ps["pmax"], dt)
        for p in range(sim.P):
            if not ps["present"][p] or ps["need_kwh"][p] <= 1e-9:
                continue
            i = ps["idx"][p]
            window = sc.price[t:sc.departure[i]]
            r = int(r_all[p]) + self.margin
            if r >= len(window):
                want[p] = ps["pmax"][p]
                continue
            cheapest = np.lexsort((np.arange(len(window)), window))[:r]
            if 0 in cheapest:
                want[p] = ps["pmax"][p]
        ports = [p for p in range(sim.P) if want[p] > 0]
        ports.sort(key=lambda p: sc.departure[ps["idx"][p]])
        return _allocate(ports, want, sim.cfg.grid_capacity_kw)


class SmartRuleBased(BaselinePolicy):
    """Least-laxity-first with price-aware deferral and load smoothing.

    laxity = remaining steps - steps needed at full power.
    * critical EVs (laxity <= ``critical_steps``) are served first, up to the full grid capacity;
    * non-critical EVs charge only when the current price is at or below the ``price_quantile`` of the
      prices left in their window, and only into the headroom below a soft grid target (``soft_frac`` * capacity),
      most-urgent first.
    """
    name = "Rule-Based"

    def __init__(self, critical_steps: int = 3, price_quantile: float = 0.35, soft_frac: float = 0.7):
        self.critical_steps, self.q, self.soft = critical_steps, price_quantile, soft_frac

    def power_request(self, sim):
        ps = sim.port_state()
        t, dt, sc, cap = sim.t, sim.dt, sim.sc, sim.cfg.grid_capacity_kw
        r_all = _min_steps_needed(ps["need_kwh"], ps["pmax"], dt)
        lax = ps["remaining_steps"] - r_all
        crit, norm = [], []
        for p in range(sim.P):
            if not ps["present"][p] or ps["need_kwh"][p] <= 1e-9:
                continue
            (crit if lax[p] <= self.critical_steps else norm).append(p)
        crit.sort(key=lambda p: lax[p])
        out = _allocate(crit, ps["pmax"], cap)
        used = out.sum()
        soft_room = max(self.soft * cap - used, 0.0)
        cheap = []
        for p in norm:
            i = ps["idx"][p]
            window = sc.price[t:sc.departure[i]]
            if sc.price[t] <= np.quantile(window, self.q) + 1e-12:
                cheap.append(p)
        cheap.sort(key=lambda p: lax[p])
        extra = _allocate(cheap, ps["pmax"], min(soft_room, cap - used))
        return out + extra


BASELINES = {
    "Uncontrolled": UncontrolledCharging,
    "FCFS": FirstComeFirstServed,
    "Time-of-Use": TimeOfUseCharging,
    "Rule-Based": SmartRuleBased,
}


def get_baseline(name: str) -> BaselinePolicy:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Choose from {list(BASELINES)}")
    return BASELINES[name]()
