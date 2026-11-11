"""Train the RL agents.

Examples:
    python scripts/train.py --algo q_learning
    python scripts/train.py --algo dqn --timesteps 150000
    python scripts/train.py --algo ppo --timesteps 300000 --lr 3e-4
    python scripts/train.py --algo all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import q_learning, trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="all", choices=["q_learning", "dqn", "ppo", "sac", "all"])
    ap.add_argument("--timesteps", type=int, default=None, help="training steps (episodes for q_learning)")
    ap.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--skip-sac", action="store_true", help="with --algo all: skip the slow SAC run")
    args = ap.parse_args()

    algos = ["q_learning", "dqn", "ppo", "sac"] if args.algo == "all" else [args.algo]
    if args.skip_sac and "sac" in algos:
        algos.remove("sac")
    for a in algos:
        if a == "q_learning":
            qc = q_learning.QLearningConfig(seed=args.seed)
            if args.timesteps:
                qc.episodes = args.timesteps
            if args.lr:
                qc.alpha = args.lr
            if args.gamma:
                qc.gamma = args.gamma
            Q, log = q_learning.train_q_learning(qc, progress_cb=lambda e, n, m: print(f"[Q-learning] episode {e}/{n} mean reward {m:.2f}", flush=True))
            print(f"Saved Q-table {Q.shape} -> {q_learning.QTABLE_PATH}")
            continue
        hp = {}
        if args.lr:
            hp["learning_rate"] = args.lr
        if args.gamma:
            hp["gamma"] = args.gamma
        if args.batch_size:
            hp["batch_size"] = args.batch_size
        meta = trainer.train(a.upper(), args.timesteps, args.seed, hp or None)
        print(f"[{a.upper()}] finished in {meta['train_seconds']:.0f}s, best validation reward "
              f"{meta['best_validation_reward']:.2f}")


if __name__ == "__main__":
    main()
