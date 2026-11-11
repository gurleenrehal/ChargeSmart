import subprocess
import sys
import time

import pandas as pd
import streamlit as st

import config
from dashboard import common
from src import evaluator, q_learning, trainer, visualization
from src.defaults import default_reward_config, default_sim_config

ALGOS = ["Q-Learning", "DQN", "PPO", "SAC"]
LOG = config.LOGS_DIR / "dashboard_training.log"


def _running() -> bool:
    p = st.session_state.get("train_proc")
    return p is not None and p.poll() is None


@st.fragment(run_every=3)
def _progress():
    algo = st.session_state.get("train_algo")
    if not algo:
        st.caption("No training started from this session.")
        return
    if algo == "Q-Learning":
        st.write("Q-Learning log (last lines):")
        st.code("\n".join(LOG.read_text().splitlines()[-4:]) if LOG.exists() else "")
    else:
        prog = common.read_json(trainer.progress_path(algo))
        if prog:
            frac = min(prog["timesteps"] / max(prog["total"], 1), 1.0)
            st.progress(frac, text=f"{algo}: {prog['timesteps']:,}/{prog['total']:,} steps - {prog['status']} - "
                                   f"recent mean reward {prog.get('mean_reward_recent')}")
    st.write("Process:", "running" if _running() else "not running")


def render():
    st.header("RL Training Centre")
    rows = []
    for a in ALGOS:
        if a == "Q-Learning":
            ok = q_learning.QTABLE_PATH.exists()
            rows.append({"algorithm": a, "environment": "single-port, discretised", "trained": ok, "timesteps": "-", "train_s": "-"})
            continue
        meta = common.read_json(trainer.meta_path(a))
        rows.append({"algorithm": a, "environment": f"{trainer.ALGO_MODE[a]} actions", "trained": meta is not None,
                     "timesteps": meta["total_timesteps"] if meta else "-",
                     "train_s": round(meta["train_seconds"]) if meta else "-"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Train")
    c = st.columns(5)
    algo = c[0].selectbox("Algorithm", ALGOS, index=1)
    unit = "episodes" if algo == "Q-Learning" else "timesteps"
    default_n = 6000 if algo == "Q-Learning" else trainer.DEFAULT_TIMESTEPS[algo]
    n = c[1].number_input(f"Number of {unit}", 500, 2_000_000, default_n, step=500)
    lr = c[2].number_input("Learning rate", 1e-5, 1e-1, 0.1 if algo == "Q-Learning" else float(trainer.default_hyperparams(algo)["learning_rate"]), format="%.5f")
    gamma = c[3].number_input("Discount factor", 0.5, 0.9999, 0.99, format="%.4f")
    bs = c[4].number_input("Batch size", 16, 1024, 128, disabled=algo == "Q-Learning")
    seed = st.number_input("Seed", 0, 10_000, config.GLOBAL_SEED)
    b1, b2 = st.columns(2)
    if b1.button("Start training", disabled=_running(), type="primary"):
        cmd = [sys.executable, str(config.ROOT / "scripts" / "train.py"), "--algo", algo.lower().replace("-", "_"),
               "--timesteps", str(int(n)), "--seed", str(int(seed)), "--lr", str(lr), "--gamma", str(gamma)]
        if algo != "Q-Learning":
            cmd += ["--batch-size", str(int(bs))]
        st.session_state["train_proc"] = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, cwd=config.ROOT)
        st.session_state["train_algo"] = algo
        st.rerun()
    if b2.button("Stop training", disabled=not _running()):
        st.session_state["train_proc"].terminate()
        st.rerun()
    _progress()

    st.subheader("Training curves")
    logs = {}
    for a in ("dqn", "ppo", "sac"):
        p = config.METRICS_DIR / f"train_log_{a}.csv"
        if p.exists():
            logs[a.upper()] = pd.read_csv(p)
    if logs:
        st.plotly_chart(visualization.fig_training(logs), use_container_width=True)
    qp = config.METRICS_DIR / "train_log_q_learning.csv"
    if qp.exists():
        q = pd.read_csv(qp)
        st.plotly_chart(visualization.fig_training({"Q-Learning (single port)": q}, x="episode", y="reward", window=100), use_container_width=True)

    st.subheader("Checkpoints and saved models")
    files = sorted(config.MODELS_DIR.rglob("*.zip")) + sorted(config.MODELS_DIR.glob("*.npy"))
    st.dataframe(pd.DataFrame([{"file": str(f.relative_to(config.MODELS_DIR)), "size_kB": round(f.stat().st_size / 1024),
                                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))} for f in files]),
                 use_container_width=True, hide_index=True)

    st.subheader("Load a saved model")
    trained = trainer.trained_algorithms()
    if not trained:
        st.info("No trained SB3 models yet.")
        return
    pick = st.selectbox("Saved model", trained)
    if st.button("Load and quick-evaluate (6 scenarios)"):
        model, mode, meta = trainer.load_trained(pick)
        df = evaluator.evaluate_policy(evaluator.RLPolicy(model, pick, mode),
                                       evaluator.make_scenarios(default_sim_config(), range(7000, 7006)),
                                       default_sim_config(), default_reward_config())
        st.success(f"Loaded {pick} ({mode} actions). Mean reward {df['episode_reward'].mean():.2f}, "
                   f"mean cost ${df['total_cost'].mean():.2f}, fully charged {df['pct_fully_charged'].mean():.1f}%")
        st.json(meta.get("hyperparams", {}))
