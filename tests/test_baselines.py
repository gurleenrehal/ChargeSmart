import numpy as np
import pytest

from src import evaluator
from src.baseline_policies import BASELINES, get_baseline
from src.ev_simulator import SimConfig


@pytest.fixture(scope="module")
def scenarios():
    return evaluator.make_scenarios(SimConfig(), range(2000, 2008))


@pytest.mark.parametrize("name", list(BASELINES))
def test_baseline_runs_and_respects_grid(name, scenarios):
    for sc in scenarios:
        res = evaluator.run_episode(get_baseline(name), sc, SimConfig(), None, False)
        assert res.trace["load"].max() <= SimConfig().grid_capacity_kw + 1e-9
        assert 0 <= res.summary["pct_fully_charged"] <= 100


def test_identical_scenarios_give_identical_results(scenarios):
    a = evaluator.run_episode(get_baseline("Rule-Based"), scenarios[0], SimConfig(), None, False).summary
    b = evaluator.run_episode(get_baseline("Rule-Based"), scenarios[0], SimConfig(), None, False).summary
    assert a == b


def test_only_uncontrolled_overloads_the_grid(scenarios):
    df = evaluator.evaluate_baselines(scenarios)
    g = df.groupby("policy")["grid_overload_events"].sum()
    assert g["FCFS"] == 0 and g["Rule-Based"] == 0 and g["Time-of-Use"] == 0
    assert g["Uncontrolled"] > 0


def test_tou_shifts_energy_to_cheap_hours(scenarios):
    cfg = SimConfig()
    df = evaluator.evaluate_baselines(scenarios)
    cost = df.groupby("policy")["total_cost"].mean()
    assert cost["Time-of-Use"] < cost["Uncontrolled"]


def test_q_learning_td_update_moves_toward_target():
    from src import q_learning
    Q, log = q_learning.train_q_learning(q_learning.QLearningConfig(episodes=150), save=False)
    assert Q.shape == (q_learning.N_STATES, q_learning.N_ACTIONS) and np.abs(Q).sum() > 0
    assert log["reward"].tail(30).mean() > log["reward"].head(30).mean()
