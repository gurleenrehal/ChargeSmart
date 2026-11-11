"""Headless smoke test: every dashboard page renders without raising."""
import pytest
from streamlit.testing.v1 import AppTest

import config

PAGES = ["1. Executive Dashboard", "2. Data Explorer", "3. Demand Forecasting", "4. RL Training Centre",
         "5. EV Charging Simulator", "6. Algorithm Comparison", "7. Business Insights"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("policy", ["Rule-Based", "Uncontrolled", "DQN", "PPO", "SAC"])
def test_simulator_run_button(policy):
    from src import trainer
    if policy in ("DQN", "PPO", "SAC") and policy not in trainer.trained_algorithms():
        pytest.skip(f"{policy} not trained")
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value("5. EV Charging Simulator").run()
    sel = [s for s in at.selectbox if s.label == "Charging policy"][0]
    sel.set_value(policy).run()
    [b for b in at.button if b.label == "Run simulation"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert "sim_result" in at.session_state
    assert at.session_state["sim_result"][0].summary["peak_kw"] <= at.session_state["sim_result"][1].grid_capacity_kw + 1e-9


def test_insights_button():
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value("7. Business Insights").run()
    [b for b in at.button if b.label == "Compute insights"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["insights"]["recommendations"]) >= 4
