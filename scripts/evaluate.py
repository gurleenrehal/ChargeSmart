"""Evaluate baselines and trained RL agents on identical scenarios and write all results to results/.

Every number in the README / dashboard comes from the files written here - nothing is hard-coded.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src import evaluator, q_learning, trainer, visualization
from src.baseline_policies import BASELINES, get_baseline
from src.defaults import default_reward_config, default_sim_config, load_empirical_sessions


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=len(config.EVAL_SEEDS), help="number of identical test scenarios")
    args = ap.parse_args()

    seeds = config.EVAL_SEEDS[: args.episodes]
    sim_cfg, rcfg = default_sim_config(), default_reward_config()
    # Use exactly the configuration the agents were trained with, if recorded.
    for algo in trainer.trained_algorithms():
        mp = trainer.meta_path(algo)
        if mp.exists():
            meta = json.loads(mp.read_text())
            from src.ev_simulator import SimConfig
            from src.reward_function import RewardConfig
            sim_cfg, rcfg = SimConfig.from_dict(meta["sim_config"]), RewardConfig.from_dict(meta["reward_config"])
            break
    empirical = load_empirical_sessions() if sim_cfg.arrival_pattern == "empirical" else None
    scenarios = evaluator.make_scenarios(sim_cfg, seeds, empirical)
    print(f"Evaluating on {len(scenarios)} identical scenarios (seeds {seeds[0]}..{seeds[-1]})")

    frames = [evaluator.evaluate_baselines(scenarios, sim_cfg, rcfg)]
    models, notes = {}, {}
    for algo in ("DQN", "PPO", "SAC"):
        loaded = trainer.load_trained(algo)
        if loaded:
            model, mode, meta = loaded
            models[algo] = (model, mode)
            notes[algo] = {"action_mode": mode, "train_timesteps": meta.get("total_timesteps"),
                           "train_seconds": meta.get("train_seconds"), "seed": meta.get("seed")}
        else:
            print(f"  (no trained {algo} model found - skipped)")
    if models:
        frames.append(evaluator.evaluate_models(models, scenarios, sim_cfg, rcfg))
    df = pd.concat(frames, ignore_index=True)
    agg = evaluator.aggregate(df)
    df.to_csv(config.METRICS_DIR / "eval_episodes.csv", index=False)
    agg.to_csv(config.METRICS_DIR / "eval_summary.csv", index=False)
    table = evaluator.pretty_table(agg)
    md = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False)]
    (config.METRICS_DIR / "results_table.md").write_text("\n".join(md) + "\n")
    print(table.to_string(index=False))

    # Paired differences (same scenario) of every policy vs each baseline.
    pair_rows = []
    pols = list(df["policy"].unique())
    for p in pols:
        for ref in ("Uncontrolled", "Time-of-Use", "Rule-Based"):
            if p == ref:
                continue
            a = df[df.policy == p].sort_values("seed"); b = df[df.policy == ref].sort_values("seed")
            for m in ("total_cost", "peak_kw", "avg_unmet_kwh", "pct_fully_charged", "episode_reward"):
                mean, ci = paired_ci(a[m].to_numpy(), b[m].to_numpy())
                pair_rows.append({"policy": p, "reference": ref, "metric": m, "mean_diff": mean, "ci95": ci,
                                  "ref_mean": float(b[m].mean()),
                                  "rel_diff_pct": 100 * mean / b[m].mean() if abs(b[m].mean()) > 1e-9 else float("nan")})
    pd.DataFrame(pair_rows).to_csv(config.METRICS_DIR / "paired_differences.csv", index=False)

    # Q-learning on the simplified (single-port) environment, compared with baselines on the same physics.
    qsummary = None
    if q_learning.QTABLE_PATH.exists():
        Q = np.load(q_learning.QTABLE_PATH)
        qdf = [q_learning.evaluate_on_simplified(q_learning.QPolicy(Q), seeds, sim_cfg, rcfg)]
        qdf += [q_learning.evaluate_on_simplified(get_baseline(n), seeds, sim_cfg, rcfg) for n in BASELINES]
        qdf = pd.concat(qdf, ignore_index=True)
        qdf.to_csv(config.METRICS_DIR / "eval_q_learning_simplified.csv", index=False)
        qagg = evaluator.aggregate(qdf)
        qagg.to_csv(config.METRICS_DIR / "eval_q_learning_simplified_summary.csv", index=False)
        print("\nSimplified single-port environment (Q-Learning vs baselines):")
        print(evaluator.pretty_table(qagg, ["total_cost", "pct_fully_charged", "avg_unmet_kwh", "episode_reward"]).to_string(index=False))
        qsummary = True

    # training logs + plots
    logs = {}
    for algo in ("dqn", "ppo", "sac"):
        p = config.METRICS_DIR / f"train_log_{algo}.csv"
        if p.exists():
            logs[algo.upper()] = pd.read_csv(p)
    visualization.save_comparison_plots(agg)
    if logs:
        visualization.save_training_curves(logs)
    qlog = config.METRICS_DIR / "train_log_q_learning.csv"
    if qlog.exists():
        qdir = config.PLOTS_DIR / "q_learning"
        qdir.mkdir(exist_ok=True)
        visualization.save_training_curves({"Q-Learning (single port)": pd.read_csv(qlog)}, outdir=qdir)

    (config.METRICS_DIR / "eval_meta.json").write_text(json.dumps({
        "n_scenarios": len(scenarios), "seeds": [int(s) for s in seeds], "sim_config": sim_cfg.to_dict(),
        "reward_config": rcfg.to_dict(), "models": notes, "data_source": __import__("src.data_loader", fromlist=["x"]).get_data_source(),
        "mean_rejected_evs_per_scenario": float(np.mean([s.n_rejected for s in scenarios])),
        "mean_served_evs_per_scenario": float(np.mean([s.n_evs for s in scenarios])),
    }, indent=2, default=str))
    print("\nSaved results to", config.METRICS_DIR)


if __name__ == "__main__":
    main()
