import pandas as pd
import streamlit as st

import config
from dashboard import common
from src import demand_forecasting as df_, visualization


@st.cache_data(show_spinner="Forecasting...")
def _forecast(h: int, stamp: float):
    return df_.forecast_next(h)


def render():
    st.header("Demand Forecasting")
    common.source_banner()
    meta = common.read_json(df_.METRICS_JSON)
    if meta is None:
        if st.button("Train forecasting models"):
            df_.train_and_evaluate()
            st.rerun()
        st.info("No forecaster trained yet.")
        return
    horizon = st.slider("Forecast horizon (hours)", 6, 168, 48, step=6)
    fc = _forecast(horizon, df_.MODEL_PATH.stat().st_mtime)
    st.plotly_chart(visualization.fig_forecast(common.hourly(), fc), use_container_width=True)
    daily = fc.set_index("timestamp")["forecast_kwh"].resample("D").sum().reset_index()
    st.write("Daily forecast totals (partial days at the edges)")
    st.dataframe(daily.assign(forecast_kwh=daily["forecast_kwh"].round(1)), use_container_width=True)
    st.subheader("Model performance on the held-out (later) 20% of the data")
    st.dataframe(pd.DataFrame(meta["metrics"]).T.round(3), use_container_width=True)
    pred = pd.read_csv(df_.FORECAST_CSV, parse_dates=["timestamp"])
    models = [c for c in pred.columns if c not in ("timestamp", "actual")]
    sel = st.multiselect("Models to plot", models, default=["random_forest", "moving_average_24h"])
    st.plotly_chart(visualization.fig_forecast_test(pred, sel), use_container_width=True)
    st.caption(f"Train rows: {meta['train_rows']}, test rows: {meta['test_rows']}, split at {meta['split_timestamp']}. "
               "XGBoost appears only if installed.")
