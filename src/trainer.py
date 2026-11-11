"""Training of the RL agents (DQN on the discrete env, PPO/SAC on the continuous env) with Stable-Baselines3."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import config
from src.charging_environment import EVChargingEnv
from src.defaults import default_reward_config, default_sim_config
from src.evaluator import RLPolicy, evaluate_policy, make_scenarios
from src.ev_simulator import SimConfig
from src.reward_function import RewardConfig

ALGO_MODE = {"DQN": "discrete", "PPO": "continuous", "SAC": "continuous"}
VALIDATION_SEEDS = list(range(5000, 5006))      # disjoint from the test seeds (config.EVAL_SEEDS)
DEFAULT_TIMESTEPS = {"DQN": 200_000, "PPO": 400_000, "SAC": 100_000}


def default_hyperparams(algo: str) -> dict:
    common = dict(learning_rate=3e-4, gamma=0.99, batch_size=128, net_arch=[128, 128])
    if algo == "DQN":
        return {**common, "learning_rate": 5e-4, "buffer_size": 100_000, "learning_starts": 5_000,
                "train_freq": 4, "gradient_steps": 1, "target_update_interval": 1_000,
                "exploration_fraction": 0.35, "exploration_final_eps": 0.05}
    if algo == "PPO":
        return {**common, "gamma": 0.995, "n_steps": 1920, "n_epochs": 10, "gae_lambda": 0.95, "ent_coef": 0.005,
                "clip_range": 0.2}
    if algo == "SAC":
        return {**common, "batch_size": 256, "buffer_size": 100_000, "learning_starts": 5_000, "train_freq": 1,
                "gradient_steps": 1, "tau": 0.005}
    raise KeyError(algo)


def model_path(algo: str, best: bool = False) -> Path:
    return config.MODELS_DIR / f"{algo.lower()}{'_best' if best else ''}.zip"


def meta_path(algo: str) -> Path:
    return config.MODELS_DIR / f"{algo.lower()}_meta.json"


def progress_path(algo: str) -> Path:
    return config.METRICS_DIR / f"train_progress_{algo.lower()}.json"


def _write_progress(algo: str, **kw) -> None:
    progress_path(algo).write_text(json.dumps({"algo": algo, "updated": time.time(), **kw}))


def _build_callbacks(algo, total, sim_cfg, rcfg, seed, eval_freq, checkpoint_freq, log_rows, empirical):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    mode = ALGO_MODE[algo]
    val_scenarios = make_scenarios(sim_cfg, VALIDATION_SEEDS, empirical)

    class LogAndValidate(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.t0 = time.time()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "episode_summary" in info:
                    s = info["episode_summary"]
                    log_rows.append({"timesteps": self.num_timesteps, "episode_reward": info["episode"]["r"],
                                     "total_cost": s["total_cost"], "peak_kw": s["peak_kw"],
                                     "pct_fully_charged": s["pct_fully_charged"],
                                     "avg_unmet_kwh": s["avg_unmet_kwh"]})
            if self.num_timesteps % 2000 < self.training_env.num_envs and log_rows:
                recent = [r["episode_reward"] for r in log_rows[-20:]]
                _write_progress(algo, status="running", timesteps=int(self.num_timesteps), total=int(total),
                                mean_reward_recent=float(np.mean(recent)), episodes=len(log_rows),
                                elapsed_s=time.time() - self.t0)
            if self.num_timesteps % eval_freq < self.training_env.num_envs:
                df = evaluate_policy(RLPolicy(self.model, algo, mode), val_scenarios, sim_cfg, rcfg)
                score = float(df["episode_reward"].mean())
                if self.verbose:
                    print(f"[{algo}] step {self.num_timesteps}: validation reward {score:.2f}", flush=True)
                if score > self.best:
                    self.best = score
                    self.model.save(model_path(algo, best=True))
                    self.logger.record("validation/best_reward", score)
            return True

    ckpt_dir = config.MODELS_DIR / "checkpoints"
    ckpt = CheckpointCallback(save_freq=checkpoint_freq, save_path=str(ckpt_dir), name_prefix=f"{algo.lower()}_s{seed}")
    return [LogAndValidate(), ckpt]


def train(algo: str, total_timesteps: Optional[int] = None, seed: int = config.GLOBAL_SEED,
          hyperparams: Optional[dict] = None, sim_cfg: Optional[SimConfig] = None,
          rcfg: Optional[RewardConfig] = None, eval_freq: int = 10_000, checkpoint_freq: int = 50_000,
          verbose: int = 1, save: bool = True) -> dict:
    """Train one agent; returns metadata (also written next to the model)."""
    import torch
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed

    from src.defaults import load_empirical_sessions

    algo = algo.upper()
    if algo not in ALGO_MODE:
        raise ValueError(f"Unknown algorithm {algo}; choose from {list(ALGO_MODE)}")
    torch.set_num_threads(1)
    total_timesteps = total_timesteps or DEFAULT_TIMESTEPS[algo]
    sim_cfg, rcfg = sim_cfg or default_sim_config(), rcfg or default_reward_config()
    hp = {**default_hyperparams(algo), **(hyperparams or {})}
    empirical = load_empirical_sessions() if sim_cfg.arrival_pattern == "empirical" else None
    set_random_seed(seed)

    env = Monitor(EVChargingEnv(sim_cfg, rcfg, action_mode=ALGO_MODE[algo], empirical_sessions=empirical))
    env.reset(seed=seed)
    net_arch = hp.pop("net_arch")
    kwargs = dict(policy_kwargs=dict(net_arch=net_arch), seed=seed, verbose=0, device="cpu", **hp)
    cls = {"DQN": DQN, "PPO": PPO, "SAC": SAC}[algo]
    model = cls("MlpPolicy", env, **kwargs)

    log_rows: list = []
    cbs = _build_callbacks(algo, total_timesteps, sim_cfg, rcfg, seed, eval_freq, checkpoint_freq, log_rows, empirical)
    cbs[0].verbose = verbose
    t0 = time.time()
    _write_progress(algo, status="running", timesteps=0, total=int(total_timesteps), mean_reward_recent=None,
                    episodes=0, elapsed_s=0.0)
    try:
        model.learn(total_timesteps=total_timesteps, callback=cbs, progress_bar=False)
    except Exception as exc:
        _write_progress(algo, status=f"failed: {exc}", timesteps=int(model.num_timesteps), total=int(total_timesteps))
        raise
    elapsed = time.time() - t0
    log = pd.DataFrame(log_rows)
    meta = {"algo": algo, "action_mode": ALGO_MODE[algo], "seed": seed, "total_timesteps": int(total_timesteps),
            "train_seconds": elapsed, "hyperparams": {**hp, "net_arch": net_arch},
            "sim_config": sim_cfg.to_dict(), "reward_config": rcfg.to_dict(),
            "best_validation_reward": float(cbs[0].best), "episodes": len(log)}
    if save:
        model.save(model_path(algo))
        meta_path(algo).write_text(json.dumps(meta, indent=2, default=str))
        log.to_csv(config.METRICS_DIR / f"train_log_{algo.lower()}.csv", index=False)
    _write_progress(algo, status="finished", timesteps=int(total_timesteps), total=int(total_timesteps),
                    mean_reward_recent=float(log["episode_reward"].tail(20).mean()) if len(log) else None,
                    episodes=len(log), elapsed_s=elapsed)
    return meta


def load_trained(algo: str, prefer_best: bool = True):
    """Return ``(model, action_mode, meta)`` or ``None`` when no model has been trained yet."""
    from stable_baselines3 import DQN, PPO, SAC

    algo = algo.upper()
    path = model_path(algo, best=True) if prefer_best and model_path(algo, best=True).exists() else model_path(algo)
    if not path.exists():
        return None
    cls = {"DQN": DQN, "PPO": PPO, "SAC": SAC}[algo]
    model = cls.load(path, device="cpu")
    meta = json.loads(meta_path(algo).read_text()) if meta_path(algo).exists() else {}
    return model, ALGO_MODE[algo], meta


def trained_algorithms() -> list[str]:
    return [a for a in ALGO_MODE if model_path(a).exists() or model_path(a, best=True).exists()]
