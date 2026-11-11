# Architecture

```mermaid
flowchart TD
    A[ACN-Data / synthetic sample] --> B[data_loader + preprocessing]
    B --> C[EDA]
    B --> D[Hourly demand series]
    D --> E[Demand forecasting: MA / RF / XGB]
    E -->|expected-demand profile| F[EVChargingEnv]
    G[ChargingSim physics + constraints] --> F
    H[ScenarioGenerator: arrivals, prices] --> G
    F --> I[Baselines]
    F --> J[Q-Learning / DQN / PPO / SAC]
    I --> K[Evaluator: identical scenarios]
    J --> K
    K --> L[results/metrics, results/plots]
    L --> M[Streamlit dashboard]
    J --> M
    G --> M
```

* `ev_simulator.py` - physics and hard constraints (no RL dependency).
* `charging_environment.py` - Gymnasium wrapper (observation, actions, reward).
* `evaluator.py` - runs any policy on identical `Scenario` objects and aggregates metrics.
* `trainer.py` - SB3 training, validation-based best-model selection, progress files used by the dashboard.
* `dashboard/` - one module per page; the Training Centre starts `scripts/train.py` as a subprocess.
