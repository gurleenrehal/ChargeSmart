"""Plotly (dashboard) and Matplotlib (saved result plots) figure builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from src.reward_function import TERM_NAMES

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2))


def _fig(title: str, **kw) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, **LAYOUT, **kw)
    return f


# ------------------------------------------------------------------------ data explorer
def fig_arrivals(sessions: pd.DataFrame) -> go.Figure:
    cnt = sessions.groupby(["arrival_hour_int", "is_weekend"]).size().unstack(fill_value=0)
    f = _fig("EV arrivals by hour of day", xaxis_title="Hour", yaxis_title="Sessions", barmode="group")
    for k, name in [(0, "Weekday"), (1, "Weekend")]:
        if k in cnt:
            f.add_bar(x=cnt.index, y=cnt[k], name=name)
    return f


def fig_duration(sessions: pd.DataFrame) -> go.Figure:
    f = _fig("Connection vs charging duration", xaxis_title="Hours", yaxis_title="Sessions", barmode="overlay")
    f.add_histogram(x=sessions["duration_h"].clip(upper=20), nbinsx=50, name="Connected", opacity=0.65)
    f.add_histogram(x=sessions["charge_duration_h"].clip(upper=20), nbinsx=50, name="Charging", opacity=0.65)
    return f


def fig_energy(sessions: pd.DataFrame) -> go.Figure:
    f = _fig("Energy delivered per session", xaxis_title="kWh", yaxis_title="Sessions")
    f.add_histogram(x=sessions["kWhDelivered"], nbinsx=50, marker_color=PALETTE[2], name="Sessions")
    return f


def fig_hourly_demand(hourly: pd.DataFrame) -> go.Figure:
    prof = hourly.groupby(["hour", "is_weekend"])["energy_kwh"].mean().unstack()
    f = _fig("Average hourly charging demand", xaxis_title="Hour of day", yaxis_title="kWh per hour")
    for k, name in [(0, "Weekday"), (1, "Weekend")]:
        if k in prof:
            f.add_scatter(x=prof.index, y=prof[k], mode="lines+markers", name=name)
    return f


def fig_daily(hourly: pd.DataFrame) -> go.Figure:
    daily = hourly.set_index("timestamp")["energy_kwh"].resample("D").sum()
    f = _fig("Daily electricity consumption", xaxis_title="Date", yaxis_title="kWh/day")
    f.add_scatter(x=daily.index, y=daily.values, mode="lines", name="Daily", line=dict(width=1))
    f.add_scatter(x=daily.index, y=daily.rolling(7, min_periods=1).mean(), mode="lines", name="7-day mean",
                  line=dict(width=3))
    return f


def fig_station_util(util: pd.DataFrame) -> go.Figure:
    f = _fig("Charging-station utilisation", yaxis_title="% of observation period", barmode="overlay")
    f.add_bar(x=util["stationID"], y=util["occupancy_pct"], name="Occupied")
    f.add_bar(x=util["stationID"], y=util["charging_pct"], name="Charging")
    return f


# ----------------------------------------------------------------------------- forecasting
def fig_forecast(hourly: pd.DataFrame, fc: pd.DataFrame, lookback_h: int = 24 * 7) -> go.Figure:
    h = hourly.tail(lookback_h)
    f = _fig("Charging demand: history and forecast", xaxis_title="Time", yaxis_title="kWh per hour")
    f.add_scatter(x=h["timestamp"], y=h["energy_kwh"], name="History", mode="lines")
    f.add_scatter(x=fc["timestamp"], y=fc["forecast_kwh"], name="Forecast (Random Forest)", mode="lines",
                  line=dict(dash="dash", width=3))
    return f


def fig_forecast_test(pred: pd.DataFrame, models: list[str], last_days: int = 14) -> go.Figure:
    p = pred[pred["timestamp"] >= pred["timestamp"].max() - pd.Timedelta(days=last_days)]
    f = _fig(f"Held-out test period (last {last_days} days)", xaxis_title="Time", yaxis_title="kWh per hour")
    f.add_scatter(x=p["timestamp"], y=p["actual"], name="Actual", line=dict(color="black", width=1.5))
    for m in models:
        if m in p:
            f.add_scatter(x=p["timestamp"], y=p[m], name=m.replace("_", " "), mode="lines", opacity=0.8)
    return f


# ------------------------------------------------------------------------------- training
def fig_training(logs: dict[str, pd.DataFrame], x: str = "timesteps", y: str = "episode_reward",
                 window: int = 50) -> go.Figure:
    f = _fig(f"Training curves ({y}, {window}-episode moving average)", xaxis_title=x, yaxis_title=y)
    for i, (name, df) in enumerate(logs.items()):
        if df is None or df.empty or y not in df:
            continue
        xs = df[x] if x in df else df.index
        f.add_scatter(x=xs, y=df[y].rolling(window, min_periods=1).mean(), name=name, line=dict(color=PALETTE[i % 8]))
    return f


# ------------------------------------------------------------------------------ simulator
def fig_schedule(result, dt: float = 0.25) -> go.Figure:
    """Heat-map of charging power per EV over time."""
    P = result.trace["ev_power"]
    hours = np.arange(P.shape[1]) * dt
    labels = [f"EV{i} (port {p})" for i, p in enumerate(result.scenario.port)]
    f = go.Figure(go.Heatmap(z=P, x=hours, y=labels, colorscale="Viridis", colorbar=dict(title="kW")))
    f.update_layout(title="Individual EV charging schedules", xaxis_title="Hour of day", **LAYOUT)
    return f


def fig_soc(result, dt: float = 0.25) -> go.Figure:
    S = result.trace["ev_soc"]
    hours = np.arange(S.shape[1]) * dt
    f = _fig("Battery state of charge", xaxis_title="Hour of day", yaxis_title="SoC", yaxis=dict(range=[0, 1.02]))
    for i in range(S.shape[0]):
        c = PALETTE[i % 8]
        f.add_scatter(x=hours, y=S[i], mode="lines", name=f"EV{i}", line=dict(color=c), connectgaps=False)
        f.add_scatter(x=[hours[result.scenario.departure[i]] if result.scenario.departure[i] < len(hours) else hours[-1]],
                      y=[result.scenario.soc_target[i]], mode="markers", marker=dict(symbol="x", size=9, color=c),
                      showlegend=False, hovertext=f"EV{i} target")
    return f


def fig_grid_load(result, grid_capacity_kw: float, dt: float = 0.25) -> go.Figure:
    tr = result.trace
    hours = np.arange(len(tr["load"])) * dt
    f = make_subplots(specs=[[{"secondary_y": True}]])
    f.add_scatter(x=hours, y=tr["load"], name="Aggregate load (kW)", fill="tozeroy", line=dict(color=PALETTE[0]))
    f.add_scatter(x=hours, y=tr["load_requested"], name="Requested (kW)", line=dict(color=PALETTE[1], dash="dot"))
    f.add_hline(y=grid_capacity_kw, line_dash="dash", line_color="red", annotation_text="Grid capacity")
    f.add_scatter(x=hours, y=tr["price"], name="Price ($/kWh)", line=dict(color=PALETTE[2], width=2), secondary_y=True)
    f.update_layout(title="Aggregate grid load and electricity price", xaxis_title="Hour of day", **LAYOUT)
    f.update_yaxes(title_text="kW", secondary_y=False)
    f.update_yaxes(title_text="$/kWh", secondary_y=True)
    return f


def fig_port_power(result, dt: float = 0.25) -> go.Figure:
    pp = result.trace["port_power"]
    hours = np.arange(pp.shape[0]) * dt
    f = _fig("Charging power allocation per port", xaxis_title="Hour of day", yaxis_title="kW", barmode="stack")
    for p in range(pp.shape[1]):
        if pp[:, p].any():
            f.add_bar(x=hours, y=pp[:, p], name=f"Port {p}", width=dt)
    return f


def departure_table(result, dt: float = 0.25) -> pd.DataFrame:
    sc, tr = result.scenario, result.trace
    df = sc.to_frame(dt)
    df["unmet_kwh"] = np.round(tr["final_unmet"], 2)
    df["status"] = np.where(df["unmet_kwh"] <= 0.1, "fully charged", "DEADLINE MISSED")
    df["final_soc"] = np.round(np.nanmax(np.where(np.isnan(tr["ev_soc"]), -1, tr["ev_soc"]), axis=1), 3)
    return df.round(2)


# ------------------------------------------------------------------------------ comparison
METRIC_LABELS = {
    "total_cost": "Total electricity cost ($)", "peak_kw": "Peak grid demand (kW)",
    "pct_fully_charged": "Vehicles fully charged (%)", "avg_unmet_kwh": "Avg unmet energy (kWh/EV)",
    "grid_utilization_pct": "Mean grid utilisation (%)", "fairness_index": "Fairness (Jain index)",
    "peak_to_average_ratio": "Peak-to-average load ratio", "episode_reward": "Episode reward",
    "deadline_violations": "Deadline violations", "grid_overload_events": "Grid overload events",
    "avg_cost_per_ev": "Cost per EV ($)", "station_utilization_pct": "Station utilisation (%)",
}


def fig_metric_bars(agg: pd.DataFrame, metric: str) -> go.Figure:
    f = _fig(METRIC_LABELS.get(metric, metric))
    f.add_bar(x=agg["policy"], y=agg[f"{metric}_mean"], error_y=dict(type="data", array=agg[f"{metric}_ci95"]),
              marker_color=[PALETTE[i % 8] for i in range(len(agg))])
    f.update_layout(showlegend=False, yaxis_title=METRIC_LABELS.get(metric, metric))
    return f


def fig_reward_terms(terms: pd.DataFrame) -> go.Figure:
    """``terms``: index = policy, columns = reward term names (episode sums, mean over scenarios)."""
    f = _fig("Reward decomposition (mean episode contribution per term)", barmode="relative", yaxis_title="reward")
    for i, c in enumerate([t for t in TERM_NAMES if t in terms.columns]):
        f.add_bar(x=terms.index, y=terms[c], name=c.replace("_", " "), marker_color=PALETTE[i % 8])
    return f


# ----------------------------------------------------------------------- matplotlib (saved)
def save_comparison_plots(agg: pd.DataFrame, outdir=config.PLOTS_DIR) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["total_cost", "peak_kw", "pct_fully_charged", "avg_unmet_kwh", "grid_utilization_pct",
               "fairness_index"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, m in zip(axes.ravel(), metrics):
        ax.bar(agg["policy"], agg[f"{m}_mean"], yerr=agg[f"{m}_ci95"], capsize=3,
               color=[PALETTE[i % 8] for i in range(len(agg))])
        ax.set_title(METRIC_LABELS[m], fontsize=10)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("Policy comparison (mean ± 95% CI over identical test scenarios)")
    fig.tight_layout()
    p = outdir / "policy_comparison.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return [str(p)]


def save_training_curves(logs: dict[str, pd.DataFrame], outdir=config.PLOTS_DIR, window: int = 50) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, df in logs.items():
        if df is None or df.empty:
            continue
        x = df["timesteps"] if "timesteps" in df else df["episode"]
        axes[0].plot(x, df["episode_reward" if "episode_reward" in df else "reward"].rolling(window, min_periods=1).mean(), label=name)
        axes[1].plot(x, (df["total_cost" if "total_cost" in df else "cost"]).rolling(window, min_periods=1).mean(), label=name)
    axes[0].set(title="Training reward (moving average)", xlabel="timesteps / episodes", ylabel="episode reward")
    axes[1].set(title="Episode electricity cost during training", xlabel="timesteps / episodes", ylabel="$")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    p = outdir / "training_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))
    return paths
