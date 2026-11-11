"""Business insights computed from actual simulation runs (no hard-coded numbers)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import evaluator, preprocessing
from src.baseline_policies import get_baseline
from src.ev_simulator import SimConfig
from src.reward_function import RewardConfig


def cheapest_window(price_profile, hours: int = 4) -> tuple[int, int, float]:
    p = np.asarray(price_profile, float)
    best = min(range(24 - hours + 1), key=lambda h: p[h:h + hours].mean())
    return best, best + hours, float(p[best:best + hours].mean())


def compute_insights(sim_cfg: SimConfig, rcfg: RewardConfig, policy, seeds, sessions: pd.DataFrame | None = None) -> dict:
    """Run ``policy`` and Uncontrolled on the same scenarios and derive recommendations."""
    scenarios = evaluator.make_scenarios(sim_cfg, seeds)
    unc, pol = get_baseline("Uncontrolled"), policy
    dt = sim_cfg.dt_hours
    T = sim_cfg.steps_per_day
    en_unc, en_pol = np.zeros(24), np.zeros(24)
    over_hours = np.zeros(24)
    peaks_req, peaks_pol, costs_unc, costs_pol, ports_pol = [], [], [], [], []
    urgent = []
    for sc in scenarios:
        ru = evaluator.run_episode(unc, sc, sim_cfg, rcfg, False)
        rp = evaluator.run_episode(pol, sc, sim_cfg, rcfg, False)
        for res, acc in ((ru, en_unc), (rp, en_pol)):
            per_h = res.trace["load"].reshape(-1, int(round(1 / dt))).sum(axis=1) * dt
            acc += per_h
        req = ru.trace["load_requested"].reshape(-1, int(round(1 / dt)))
        over_hours += (req > sim_cfg.grid_capacity_kw + 1e-6).any(axis=1)
        peaks_req.append(ru.trace["load_requested"].max())
        peaks_pol.append(rp.trace["load"].max())
        costs_unc.append(ru.summary["total_cost"]); costs_pol.append(rp.summary["total_cost"])
        ports_pol.append(rp.summary["station_utilization_pct"])
    # urgent vehicles in the first scenario: highest (energy needed / energy deliverable before departure)
    sc0 = scenarios[0]
    f = sc0.to_frame(dt)
    f["urgency"] = f["energy_needed_kwh"] / (f["max_power_kw"] * (f["departure_h"] - f["arrival_h"]))
    urgent = f.sort_values("urgency", ascending=False).head(5).round(2)

    w0, w1, wp = cheapest_window(sim_cfg.price_profile)
    share_pol = float(en_pol[w0:w1].sum() / max(en_pol.sum(), 1e-9) * 100)
    share_unc = float(en_unc[w0:w1].sum() / max(en_unc.sum(), 1e-9) * 100)
    out = {
        "cheap_window": (w0, w1, wp), "share_energy_cheap_policy": share_pol, "share_energy_cheap_uncontrolled": share_unc,
        "peak_requested_uncontrolled": float(np.mean(peaks_req)), "peak_policy": float(np.mean(peaks_pol)),
        "cost_uncontrolled": float(np.mean(costs_unc)), "cost_policy": float(np.mean(costs_pol)),
        "overload_hours": pd.Series(over_hours / len(scenarios), index=range(24), name="share_of_days_overloaded"),
        "energy_by_hour_uncontrolled": pd.Series(en_unc / len(scenarios)), "energy_by_hour_policy": pd.Series(en_pol / len(scenarios)),
        "urgent_evs": urgent, "policy": pol.name, "n_scenarios": len(scenarios),
        "sim_station_utilization_pct": float(np.mean(ports_pol)),
    }
    recs = []
    recs.append(f"Best charging window: {w0:02d}:00-{w1:02d}:00 (mean tariff ${wp:.2f}/kWh). {pol.name} delivers "
                f"{share_pol:.1f}% of its energy there vs {share_unc:.1f}% under uncontrolled charging.")
    pk_red = 100 * (out["peak_requested_uncontrolled"] - out["peak_policy"]) / max(out["peak_requested_uncontrolled"], 1e-9)
    recs.append(f"Unmanaged charging would request a mean peak of {out['peak_requested_uncontrolled']:.1f} kW against a "
                f"{sim_cfg.grid_capacity_kw:.0f} kW limit; {pol.name} peaks at {out['peak_policy']:.1f} kW ({pk_red:.1f}% lower than requested).")
    sv = 100 * (out["cost_uncontrolled"] - out["cost_policy"]) / max(out["cost_uncontrolled"], 1e-9)
    recs.append(f"Mean daily energy cost: ${out['cost_uncontrolled']:.2f} (uncontrolled) vs ${out['cost_policy']:.2f} ({pol.name}), a {sv:.1f}% difference.")
    worst = out["overload_hours"].sort_values(ascending=False)
    if worst.iloc[0] > 0:
        hrs = ", ".join(f"{h:02d}:00" for h in worst.index[:3] if worst[h] > 0)
        recs.append(f"Grid bottleneck hours under uncontrolled charging: {hrs} (overload on up to {100 * worst.iloc[0]:.0f}% of simulated days).")
    else:
        recs.append("No grid-capacity bottleneck occurred under uncontrolled charging in these scenarios.")
    if sessions is not None and len(sessions):
        util = preprocessing.station_utilization(sessions)
        hi, lo = util.iloc[0], util.iloc[-1]
        recs.append(f"Data: station {hi['stationID']} is occupied {hi['occupancy_pct']:.1f}% of the time vs {lo['occupancy_pct']:.1f}% for "
                    f"{lo['stationID']}; steering new arrivals to the least-used stations would balance utilisation.")
    out["recommendations"] = recs
    return out
