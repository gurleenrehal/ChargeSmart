import numpy as np
import pandas as pd
import streamlit as st

from dashboard import common
from src import evaluator, trainer, visualization
from src.baseline_policies import BASELINES, get_baseline
from src.defaults import default_reward_config, default_sim_config
from src.ev_simulator import ARRIVAL_PATTERNS, DEFAULT_TOU_PRICE, ScenarioGenerator

LEVEL_NAMES = {0.0: "off", 0.33: "low", 0.67: "medium", 1.0: "high"}


def render():
    st.header("EV Charging Simulator")
    base = default_sim_config()
    with st.expander("Scenario configuration", expanded=True):
        a, b, c = st.columns(3)
        stations = a.slider("Charging stations", 1, base.max_stations, base.n_stations)
        pps = a.slider("Ports per station", 1, base.max_ports_per_station, base.ports_per_station)
        n_evs = a.slider("Number of EVs", 1, 14, base.n_evs)
        cap = b.slider("Grid capacity (kW)", 4.0, 30.0, base.grid_capacity_kw)
        pmax = b.selectbox("Max charging power per port (kW)", [3.7, 7.2, 11.0], index=1)
        pattern = b.selectbox("Arrival pattern", ARRIVAL_PATTERNS, index=0)
        dwell = c.slider("Departure-deadline scale (x parking time)", 0.5, 2.0, 1.0, 0.1)
        batt = c.multiselect("Battery capacities (kWh)", [30.0, 35.0, 50.0, 60.0, 75.0, 100.0], default=[35.0, 50.0, 60.0, 75.0])
        soc0 = c.slider("Initial SoC range", 0.05, 0.95, (0.40, 0.70))
        tgt = c.slider("Target SoC range", 0.5, 1.0, (0.75, 0.90))
        t1, t2 = st.columns(2)
        tariff = t1.selectbox("Electricity pricing", ["Time-of-use (default)", "Flat"])
        spread = t2.slider("Price spread (x default TOU spread)" if tariff.startswith("Time") else "Flat price ($/kWh)",
                           0.0, 2.0 if tariff.startswith("Time") else 0.6, 1.0 if tariff.startswith("Time") else 0.2, 0.05)
        seed = st.number_input("Scenario seed", 0, 100000, 1000)
    mean_p = float(np.mean(DEFAULT_TOU_PRICE))
    profile = [max(0.01, mean_p + spread * (p - mean_p)) for p in DEFAULT_TOU_PRICE] if tariff.startswith("Time") else [spread] * 24
    cfg = default_sim_config(n_stations=stations, ports_per_station=pps, n_evs=n_evs, grid_capacity_kw=cap,
                             port_max_power_kw=pmax, ev_power_choices_kw=[min(pmax, x) for x in (3.7, 6.6, 7.2, 11.0)] if pmax > 7.2 else [3.7, 6.6, pmax],
                             arrival_pattern=pattern, dwell_scale=dwell, battery_choices_kwh=batt or [50.0],
                             soc0_range=tuple(soc0), soc_target_range=tuple(tgt), price_profile=profile)
    rcfg = default_reward_config()
    options = list(BASELINES) + ["DQN", "PPO", "SAC"]
    choice = st.selectbox("Charging policy", options)
    rl = choice in ("DQN", "PPO", "SAC")
    if rl:
        meta = common.read_json(trainer.meta_path(choice))
        if not trainer.model_path(choice).exists() and not trainer.model_path(choice, True).exists():
            st.error(f"{choice} has not been trained yet - use the Training Centre.")
            return
        trained_cfg = (meta or {}).get("sim_config", {})
        diff = [k for k in ("grid_capacity_kw", "n_evs", "port_max_power_kw", "n_stations", "ports_per_station",
                            "arrival_pattern", "dwell_scale", "price_profile") if trained_cfg.get(k) != getattr(cfg, k)]
        if diff:
            st.warning("The agent was trained on different settings (" + ", ".join(diff) + "). Results show how well it generalises.")
    if st.button("Run simulation", type="primary"):
        sc = ScenarioGenerator(cfg).generate(int(seed))
        if sc.n_evs == 0:
            st.error("No EV could be placed with these settings - increase ports or reduce EV count.")
            return
        if rl:
            model, mode, _ = common.load_model_cached(choice, common.model_stamp(choice))
            policy = evaluator.RLPolicy(model, choice, mode)
        else:
            policy = get_baseline(choice)
        st.session_state["sim_result"] = (evaluator.run_episode(policy, sc, cfg, rcfg), cfg, mode if rl else "continuous")
    if "sim_result" not in st.session_state:
        return
    res, cfg, mode = st.session_state["sim_result"]
    s = res.summary
    k = st.columns(6)
    common.kpi(k[0], "Total cost", f"${s['total_cost']:.2f}")
    common.kpi(k[1], "Peak load (kW)", f"{s['peak_kw']:.1f}")
    common.kpi(k[2], "Fully charged", f"{s['pct_fully_charged']:.0f}%")
    common.kpi(k[3], "Avg unmet (kWh)", f"{s['avg_unmet_kwh']:.2f}")
    common.kpi(k[4], "Grid util.", f"{s['grid_utilization_pct']:.1f}%")
    common.kpi(k[5], "EVs served / rejected", f"{s['n_evs']} / {s['n_rejected']}")
    st.caption(f"Policy: **{res.policy}**, episode reward {res.total_reward:.2f}")
    dt = cfg.dt_hours
    a, b = st.columns(2)
    a.plotly_chart(visualization.fig_schedule(res, dt), use_container_width=True)
    b.plotly_chart(visualization.fig_soc(res, dt), use_container_width=True)
    a.plotly_chart(visualization.fig_grid_load(res, cfg.grid_capacity_kw, dt), use_container_width=True)
    b.plotly_chart(visualization.fig_port_power(res, dt), use_container_width=True)
    st.subheader("Vehicle departure status")
    st.dataframe(visualization.departure_table(res, dt), use_container_width=True, hide_index=True)
    st.subheader("Agent / policy decisions per time step")
    dec = pd.DataFrame(res.trace["decisions"])
    if mode == "discrete":
        for p in range(cfg.n_ports):
            dec[f"level_port{p}"] = dec[f"frac_port{p}"].round(2).map(lambda x: LEVEL_NAMES.get(round(x, 2), f"{x:.2f}"))
    only_active = [c for c in dec.columns if not (c.endswith(tuple(str(p) for p in range(cfg.n_ports))) and
                   int(c[-1]) not in np.flatnonzero(cfg.active_ports()))]
    t = st.slider("Time step", 0, len(dec) - 1, 36)
    st.dataframe(dec[only_active].round(3).iloc[max(0, t - 5): t + 6], use_container_width=True)
    with st.expander("All time steps"):
        st.dataframe(dec[only_active].round(3), use_container_width=True)
