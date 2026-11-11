import streamlit as st

from dashboard import common
from src import preprocessing, visualization


def render():
    st.header("EV Charging Data Explorer")
    common.source_banner()
    s = common.sessions()
    d0, d1 = s["connection"].min().date(), s["connection"].max().date()
    c = st.columns(3)
    dr = c[0].date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
    stations = c[1].multiselect("Stations", sorted(s["stationID"].unique()))
    e0, e1 = float(s["kWhDelivered"].min()), float(s["kWhDelivered"].max())
    er = c[2].slider("Energy per session (kWh)", e0, e1, (e0, e1))
    f = s[(s["kWhDelivered"] >= er[0]) & (s["kWhDelivered"] <= er[1])]
    if len(dr) == 2:
        f = f[(f["connection"].dt.date >= dr[0]) & (f["connection"].dt.date <= dr[1])]
    if stations:
        f = f[f["stationID"].isin(stations)]
    if f.empty:
        st.info("No sessions match the filters.")
        return
    st.write(f"**{len(f):,}** sessions selected")
    hrly = preprocessing.hourly_demand(f)
    a, b = st.columns(2)
    a.plotly_chart(visualization.fig_arrivals(f), use_container_width=True)
    b.plotly_chart(visualization.fig_duration(f), use_container_width=True)
    a.plotly_chart(visualization.fig_energy(f), use_container_width=True)
    b.plotly_chart(visualization.fig_hourly_demand(hrly), use_container_width=True)
    a.plotly_chart(visualization.fig_daily(hrly), use_container_width=True)
    b.plotly_chart(visualization.fig_station_util(preprocessing.station_utilization(f)), use_container_width=True)
    st.subheader("Session records")
    st.dataframe(f[["sessionID", "stationID", "connection", "disconnect", "duration_h", "charge_duration_h",
                    "kWhDelivered", "avg_power_kw"]].head(2000), use_container_width=True)
