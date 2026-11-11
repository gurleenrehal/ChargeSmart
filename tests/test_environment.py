import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.charging_environment import EVChargingEnv


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_gymnasium_checker(mode):
    check_env(EVChargingEnv(action_mode=mode), skip_render_check=True)


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_episode_runs_and_is_reproducible(mode):
    def run(seed):
        env = EVChargingEnv(action_mode=mode)
        env.reset(seed=seed)
        rng = np.random.default_rng(0)
        total, done = 0.0, False
        while not done:
            a = env.action_space.sample() if False else (rng.uniform(-1, 1, env.P).astype("float32") if mode == "continuous" else int(rng.integers(env.action_space.n)))
            obs, r, done, _, info = env.step(a)
            assert env.observation_space.contains(obs)
            total += r
        return total, info["episode_summary"]["total_cost"]
    assert run(5) == run(5)


def test_reward_terms_sum_to_reward():
    env = EVChargingEnv(action_mode="continuous")
    env.reset(seed=2)
    _, r, _, _, info = env.step(np.ones(env.P, dtype="float32"))
    assert np.isclose(sum(info["reward_terms"].values()), r)


def test_idle_agent_is_not_rewarded_more_than_charging_agent():
    """The reward must not encourage skipping charging to save cost."""
    from src.baseline_policies import get_baseline
    from src.evaluator import make_scenarios, run_episode
    from src.ev_simulator import SimConfig
    from src.baseline_policies import BaselinePolicy

    class Idle(BaselinePolicy):
        name = "idle"
        def power_request(self, sim):
            return np.zeros(sim.P)

    cfg = SimConfig()
    sc = make_scenarios(cfg, range(10))
    idle = np.mean([run_episode(Idle(), s, cfg, None, False).total_reward for s in sc])
    rule = np.mean([run_episode(get_baseline("Rule-Based"), s, cfg, None, False).total_reward for s in sc])
    assert rule > idle
