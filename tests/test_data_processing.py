import numpy as np
import pandas as pd

from src import data_loader, preprocessing


def _raw():
    return data_loader.generate_synthetic_sample(n_days=40, seed=1, out_csv=__import__("pathlib").Path("/tmp/_sample_test.csv"))


def test_cleaning_removes_invalid_rows():
    raw = _raw()
    clean, rep = preprocessing.clean_sessions(raw, "Synthetic sample (not real data)")
    assert len(clean) < len(raw)
    assert (clean["kWhDelivered"] > 0).all()
    assert (clean["disconnect"] > clean["connection"]).all()
    assert clean["done_charging"].between(clean["connection"], clean["disconnect"]).all()
    assert rep["rows_clean"] == len(clean)


def test_hourly_energy_is_conserved():
    clean, _ = preprocessing.clean_sessions(_raw(), "Synthetic sample (not real data)")
    h = preprocessing.hourly_demand(clean)
    assert np.isclose(h["energy_kwh"].sum(), clean["kWhDelivered"].sum(), rtol=1e-6)
    assert (h["energy_kwh"] >= 0).all()


def test_sample_is_deterministic():
    a = data_loader.generate_synthetic_sample(n_days=10, seed=3, out_csv=__import__("pathlib").Path("/tmp/_a.csv"))
    b = data_loader.generate_synthetic_sample(n_days=10, seed=3, out_csv=__import__("pathlib").Path("/tmp/_b.csv"))
    pd.testing.assert_frame_equal(a, b)


def test_time_split_has_no_leakage():
    from src import demand_forecasting as df
    res = df.train_and_evaluate(save=False)
    pred = res.test_predictions
    assert pred["timestamp"].is_monotonic_increasing
    assert pd.Timestamp(res.metrics["split_timestamp"]) <= pred["timestamp"].min()
