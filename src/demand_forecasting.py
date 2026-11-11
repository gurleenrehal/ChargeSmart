"""Hourly charging-demand forecasting (Moving Average baseline, Random Forest, optional XGBoost).

* Time-based split only: the last 20 % of the series is the test set (no shuffling).
* Features use information available *before* the forecast hour (lags, rolling means).
* The forecast is **context** for the RL agent (``expected demand`` in its observation);
  the agent still has to learn charging decisions on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from src import data_loader

MODEL_PATH = config.MODELS_DIR / "demand_rf.joblib"
FORECAST_CSV = config.METRICS_DIR / "forecast_test_predictions.csv"
METRICS_JSON = config.METRICS_DIR / "forecast_metrics.json"
PROFILE_JSON = config.METRICS_DIR / "demand_profile.json"
FEATURES = ["hour_sin", "hour_cos", "dow", "is_weekend", "lag_1", "lag_2", "lag_3",
            "lag_24", "lag_168", "roll_mean_24", "roll_mean_168"]


# ----------------------------------------------------------------------------- features
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised feature matrix for a complete hourly series (``timestamp``, ``energy_kwh``)."""
    out = pd.DataFrame(index=df.index)
    ts = df["timestamp"]
    y = df["energy_kwh"]
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    for k in (1, 2, 3, 24, 168):
        out[f"lag_{k}"] = y.shift(k)
    out["roll_mean_24"] = y.shift(1).rolling(24).mean()
    out["roll_mean_168"] = y.shift(1).rolling(168).mean()
    return out


def _row_features(ts: pd.Timestamp, hist: np.ndarray) -> np.ndarray:
    """Single feature row for ``ts`` given the history vector (used for recursive forecasts)."""
    n = len(hist)
    dow = ts.dayofweek
    return np.array([[np.sin(2 * np.pi * ts.hour / 24), np.cos(2 * np.pi * ts.hour / 24), dow, int(dow >= 5),
                      hist[n - 1], hist[n - 2], hist[n - 3], hist[n - 24], hist[n - 168],
                      hist[n - 24:].mean(), hist[n - 168:].mean()]])


# ------------------------------------------------------------------------------ models
class MovingAverageForecaster:
    """Forecast = mean of the previous ``window`` hours (window=24 -> trailing-day average)."""

    def __init__(self, window: int = 24):
        self.window = window

    def predict_series(self, y: pd.Series) -> pd.Series:
        return y.shift(1).rolling(self.window, min_periods=1).mean()


class SeasonalMovingAverage:
    """Mean of the same hour on the previous ``days`` days (captures the daily cycle)."""

    def __init__(self, days: int = 7):
        self.days = days

    def predict_series(self, y: pd.Series) -> pd.Series:
        parts = [y.shift(24 * k) for k in range(1, self.days + 1)]
        return pd.concat(parts, axis=1).mean(axis=1)


def _metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ok = ~np.isnan(y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true[ok], y_pred[ok])),
        "RMSE": float(np.sqrt(mean_squared_error(y_true[ok], y_pred[ok]))),
        "R2": float(r2_score(y_true[ok], y_pred[ok])),
    }


def _make_xgb():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, random_state=config.GLOBAL_SEED)
    except Exception:  # not installed -> silently skip (it is optional)
        return None


# --------------------------------------------------------------------------- pipeline
@dataclass
class ForecastResult:
    metrics: dict
    test_predictions: pd.DataFrame
    profile: dict


def train_and_evaluate(test_frac: float = 0.2, save: bool = True) -> ForecastResult:
    df = data_loader.load_hourly_demand().reset_index(drop=True)
    if len(df) < 168 * 3:
        raise ValueError("Need at least three weeks of hourly data to train the forecaster.")
    feats = make_features(df)
    valid = feats.dropna().index
    split_ts = df.loc[valid, "timestamp"].iloc[int(len(valid) * (1 - test_frac))]
    train_idx = [i for i in valid if df.loc[i, "timestamp"] < split_ts]
    test_idx = [i for i in valid if df.loc[i, "timestamp"] >= split_ts]
    X_tr, y_tr = feats.loc[train_idx, FEATURES], df.loc[train_idx, "energy_kwh"]
    X_te, y_te = feats.loc[test_idx, FEATURES], df.loc[test_idx, "energy_kwh"]

    preds = pd.DataFrame({"timestamp": df.loc[test_idx, "timestamp"], "actual": y_te.values})
    metrics: dict[str, dict] = {}

    ma = MovingAverageForecaster(24).predict_series(df["energy_kwh"])
    sma = SeasonalMovingAverage(7).predict_series(df["energy_kwh"])
    preds["moving_average_24h"] = ma.loc[test_idx].values
    preds["seasonal_moving_average_7d"] = sma.loc[test_idx].values
    metrics["moving_average_24h"] = _metrics(y_te, preds["moving_average_24h"])
    metrics["seasonal_moving_average_7d"] = _metrics(y_te, preds["seasonal_moving_average_7d"])

    rf = RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=3,
                               random_state=config.GLOBAL_SEED, n_jobs=1)
    rf.fit(X_tr, y_tr)
    preds["random_forest"] = rf.predict(X_te)
    metrics["random_forest"] = _metrics(y_te, preds["random_forest"])

    xgb = _make_xgb()
    if xgb is not None:
        xgb.fit(X_tr, y_tr)
        preds["xgboost"] = xgb.predict(X_te)
        metrics["xgboost"] = _metrics(y_te, preds["xgboost"])

    # Day-ahead (recursive 24 h) evaluation of the RF from each test midnight.
    day_true, day_pred = [], []
    ts_all = df["timestamp"]
    for midnight in sorted({t.normalize() for t in df.loc[test_idx, "timestamp"]}):
        pos = ts_all.index[ts_all == midnight]
        if len(pos) == 0 or pos[0] + 24 > len(df) or pos[0] < 168:
            continue
        p0 = int(pos[0])
        hist = df["energy_kwh"].to_numpy()[:p0].copy()
        for h in range(24):
            yhat = float(rf.predict(pd.DataFrame(_row_features(ts_all.iloc[p0 + h], hist), columns=FEATURES))[0])
            hist = np.append(hist, yhat)
            day_pred.append(yhat)
            day_true.append(df["energy_kwh"].iloc[p0 + h])
    if day_true:
        metrics["random_forest_day_ahead_24h"] = _metrics(day_true, day_pred)

    # Average predicted demand profile by hour of day (used as RL context), normalised to [0, 1].
    prof_df = preds.assign(hour=preds["timestamp"].dt.hour, wk=preds["timestamp"].dt.dayofweek >= 5)
    profile = {}
    for name, flag in (("weekday", False), ("weekend", True)):
        sub = prof_df[prof_df["wk"] == flag]
        vec = sub.groupby("hour")["random_forest"].mean().reindex(range(24)).fillna(0.0).to_numpy()
        profile[name] = (vec / max(vec.max(), 1e-9)).round(4).tolist()
    result_meta = {
        "data_source": data_loader.get_data_source(),
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "split_timestamp": str(split_ts), "metrics": metrics,
        "feature_importance": dict(zip(FEATURES, map(float, rf.feature_importances_))),
    }
    if save:
        joblib.dump(rf, MODEL_PATH)
        preds.to_csv(FORECAST_CSV, index=False)
        METRICS_JSON.write_text(json.dumps(result_meta, indent=2))
        PROFILE_JSON.write_text(json.dumps(profile, indent=2))
    return ForecastResult(metrics=result_meta, test_predictions=preds, profile=profile)


def load_demand_profile() -> dict | None:
    if PROFILE_JSON.exists():
        return json.loads(PROFILE_JSON.read_text())
    return None


def forecast_next(horizon_hours: int = 24, model_path=MODEL_PATH) -> pd.DataFrame:
    """Recursive forecast for the ``horizon_hours`` hours after the end of the historical series."""
    df = data_loader.load_hourly_demand()
    if not model_path.exists():
        train_and_evaluate()
    rf = joblib.load(model_path)
    hist = df["energy_kwh"].to_numpy().copy()
    ts = df["timestamp"].iloc[-1]
    rows = []
    for _ in range(horizon_hours):
        ts = ts + pd.Timedelta(hours=1)
        yhat = max(0.0, float(rf.predict(pd.DataFrame(_row_features(ts, hist), columns=FEATURES))[0]))
        hist = np.append(hist, yhat)
        rows.append({"timestamp": ts, "forecast_kwh": yhat})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = train_and_evaluate()
    print(json.dumps(r.metrics["metrics"], indent=2))
