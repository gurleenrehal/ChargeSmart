import pandas as pd
import streamlit as st

import config
from dashboard import common
from src import evaluator, trainer, visualization
from src.baseline_policies import BASELINES
from src.defaults import default_reward_config, default_sim_config
from src.reward_function import TERM_NAMES


def render():
    st.header("Algorithm Comparison")
    with st.expander("Re-run benchmark now", expanded=False):
        n = st.slider("Number of identical test scenarios", 5, 30, 10)
        if st.button("Run benchmark"):
            cfg, rc = default_sim_config(), default_reward_config()
            scs = evaluator.make_scenarios(cfg, config.EVAL_SEEDS[:n])
            frames = [evaluator.evaluate_baselines(scs, cfg, rc)]
            models = {a: m[:2] for a in trainer.trained_algorithms() if (m := trainer.load_trained(a))}
            if models:
                frames.append(evaluator.evaluate_models(models, scs, cfg, rc))
            df = pd.concat(frames, ignore_index=True)
            df.to_csv(config.METRICS_DIR / "eval_episodes.csv", index=False)
            evaluator.aggregate(df).to_csv(config.METRICS_DIR / "eval_summary.csv", index=False)
            st.rerun()
    agg, eps = common.eval_summary(), common.eval_episodes()
    if agg is None:
        st.info("No results yet - run `python scripts/evaluate.py` or use the button above.")
        return
    st.info("DQN acts in the **discrete** environment (4 power levels per port); PPO/SAC act in the **continuous** one. "
            "Both share identical physics, scenarios, prices and reward, but the action sets differ, so they are not "
            "strictly like-for-like. Baselines are evaluated in the continuous environment.")
    st.write(f"Mean ± std over **{int(agg['n_episodes'].iloc[0])}** identical scenarios")
    st.dataframe(evaluator.pretty_table(agg), use_container_width=True, hide_index=True)
    metric = st.selectbox("Metric", list(visualization.METRIC_LABELS), format_func=lambda m: visualization.METRIC_LABELS[m])
    a, b = st.columns(2)
    a.plotly_chart(visualization.fig_metric_bars(agg, metric), use_container_width=True)
    if eps is not None:
        b.plotly_chart(visualization.fig_reward_terms(eps.groupby("policy", sort=False)[[f"term_{t}" for t in TERM_NAMES if f"term_{t}" in eps]]
                       .mean().rename(columns=lambda c: c[5:])), use_container_width=True)
        box = visualization.go.Figure()
        for p, g in eps.groupby("policy", sort=False):
            box.add_box(y=g[metric], name=p)
        box.update_layout(title=f"Distribution across scenarios: {visualization.METRIC_LABELS[metric]}", **visualization.LAYOUT)
        st.plotly_chart(box, use_container_width=True)
    pp = config.METRICS_DIR / "paired_differences.csv"
    if pp.exists():
        st.subheader("Paired differences vs baselines (same scenarios, mean ± 95% CI)")
        st.dataframe(pd.read_csv(pp).round(3), use_container_width=True, hide_index=True)
    q = config.METRICS_DIR / "eval_q_learning_simplified_summary.csv"
    if q.exists():
        st.subheader("Q-Learning (simplified single-port environment)")
        st.dataframe(evaluator.pretty_table(pd.read_csv(q), ["total_cost", "pct_fully_charged", "avg_unmet_kwh", "episode_reward"]),
                     use_container_width=True, hide_index=True)
