import streamlit as st

from dashboard import common
from src import insights, trainer
from src.baseline_policies import BASELINES, get_baseline
from src.defaults import default_reward_config, default_sim_config
import config
from src import evaluator


def render():
    st.header("Business Insights")
    common.source_banner()
    options = list(BASELINES) + trainer.trained_algorithms()
    default = "Rule-Based"
    pol_name = st.selectbox("Policy to analyse", options, index=options.index(default))
    n = st.slider("Scenarios", 5, 30, 10)
    if st.button("Compute insights", type="primary"):
        if pol_name in BASELINES:
            pol = get_baseline(pol_name)
        else:
            m, mode, _ = trainer.load_trained(pol_name)
            pol = evaluator.RLPolicy(m, pol_name, mode)
        st.session_state["insights"] = insights.compute_insights(default_sim_config(), default_reward_config(), pol,
                                                                  config.EVAL_SEEDS[:n], common.sessions())
    r = st.session_state.get("insights")
    if not r:
        st.info("Press 'Compute insights' to run the simulations the recommendations are based on.")
        return
    st.subheader(f"Recommendations (from {r['n_scenarios']} simulated days, policy: {r['policy']})")
    for rec in r["recommendations"]:
        st.markdown(f"- {rec}")
    import pandas as pd
    e = pd.DataFrame({"Uncontrolled": r["energy_by_hour_uncontrolled"], r["policy"]: r["energy_by_hour_policy"]})
    st.bar_chart(e)
    st.caption("Mean energy delivered per hour of day (kWh)")
    st.bar_chart(r["overload_hours"].rename("share of days with requested load above grid capacity (uncontrolled)"))
    st.subheader("Vehicles with the tightest charging deadline (first scenario)")
    st.dataframe(r["urgent_evs"], hide_index=True, use_container_width=True)
