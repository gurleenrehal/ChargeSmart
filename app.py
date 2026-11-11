"""ChargeSmart - Streamlit entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="ChargeSmart", page_icon="⚡", layout="wide")

from dashboard import (common, comparison, data_explorer, demand_forecasting, insights, overview,  # noqa: E402
                       simulator, training_centre)

PAGES = {
    "1. Executive Dashboard": overview,
    "2. Data Explorer": data_explorer,
    "3. Demand Forecasting": demand_forecasting,
    "4. RL Training Centre": training_centre,
    "5. EV Charging Simulator": simulator,
    "6. Algorithm Comparison": comparison,
    "7. Business Insights": insights,
}

st.markdown(common.CSS, unsafe_allow_html=True)
st.sidebar.title("⚡ ChargeSmart")
st.sidebar.caption("RL-based EV charging & smart-grid optimisation (simulation prototype)")
page = st.sidebar.radio("Navigation", list(PAGES))
try:
    PAGES[page].render()
except FileNotFoundError as exc:
    st.error(f"Missing data file: {exc}. Run `python scripts/preprocess_data.py` first.")
