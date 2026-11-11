# models/

Trained artefacts (created by `scripts/train.py` and `scripts/preprocess_data.py`):

| File | Content |
|---|---|
| `dqn.zip`, `ppo.zip`, `sac.zip` | final Stable-Baselines3 models |
| `*_best.zip` | best checkpoint on 6 validation scenarios (seeds 5000-5005, disjoint from the test seeds) - used by the evaluator and dashboard |
| `*_meta.json` | training configuration (hyper-parameters, simulation and reward config, seed, time) |
| `q_learning_qtable.npy` | tabular Q-table (900 states x 4 actions) for the simplified single-port problem |
| `demand_rf.joblib` | Random Forest demand forecaster |
| `checkpoints/` | periodic training checkpoints (git-ignored) |
