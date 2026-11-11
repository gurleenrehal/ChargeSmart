import streamlit as st

from dashboard import common
from src import preprocessing, visualization
from src.ev_simulator import DEFAULT_TOU_PRICE


def render():
    st.header("Executive Dashboard")
    common.source_banner()
    s, h = common.sessions(), common.hourly()
    util = preprocessing.station_utilization(s)
    price = DEFAULT_TOU_PRICE
    cost = float((h["energy_kwh"] * h["hour"].map(lambda x: price[x])).sum())
    summ = common.eval_summary()
    c = st.columns(4)
    common.kpi(c[0], "EV sessions analysed", f"{len(s):,}")
    common.kpi(c[1], "Charging stations", f"{s['stationID'].nunique()}")
    common.kpi(c[2], "Total energy (MWh)", f"{s['kWhDelivered'].sum() / 1000:.1f}")
    common.kpi(c[3], "Avg cost/session (simulated TOU tariff)", f"${cost / len(s):.2f}")
    c = st.columns(4)
    common.kpi(c[0], "Peak hourly demand (kWh)", f"{h['energy_kwh'].max():.0f}")
    common.kpi(c[1], "Avg station occupancy", f"{util['occupancy_pct'].mean():.1f}%")
    if summ is not None:
        rl = summ[summ["policy"].isin(["DQN", "PPO", "SAC"])]
        best = (rl if len(rl) else summ).sort_values("episode_reward_mean", ascending=False).iloc[0]
        common.kpi(c[2], f"Best policy by reward: {best['policy']}", f"{best['episode_reward_mean']:.1f}")
        common.kpi(c[3], f"{best['policy']} cost/day (sim)", f"${best['total_cost_mean']:.2f}")
    else:
        common.kpi(c[2], "RL performance", "run evaluate.py")
        common.kpi(c[3], "", "-")
    a, b = st.columns(2)
    a.plotly_chart(visualization.fig_hourly_demand(h), use_container_width=True)
    if summ is not None:
        b.plotly_chart(visualization.fig_metric_bars(summ, "total_cost"), use_container_width=True)
    else:
        b.plotly_chart(visualization.fig_arrivals(s), use_container_width=True)
    st.caption("Prices, battery sizes and grid capacity are simulation parameters; session data are from the source shown above.")
