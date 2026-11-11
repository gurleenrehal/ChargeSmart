# Methodology

**Data.** ACN-Data sessions (connection / disconnection / done-charging times, kWh delivered) are cleaned
(invalid durations, non-positive energy, implausible power removed; missing `doneChargingTime` imputed with the
disconnect time). Per-hour demand is obtained by spreading each session's energy uniformly over its charging window.
When ACN-Data is unavailable, a **synthetic** schema-compatible sample is used and every artefact says so.

**Simulated (not observed) variables:** battery capacity, initial/target SoC, EV acceptance rate, charger limit,
tariff, grid capacity, number of ports. In `empirical` arrival mode arrival time, dwell time and energy demand are
resampled from the processed sessions.

**Forecasting.** Chronological 80/20 split. Moving average (24 h), seasonal moving average (7 d) and Random Forest on
lag/rolling/calendar features; day-ahead recursive evaluation. The forecast enters the RL observation as context only.

**Environment.** 15-min steps, one day per episode, 2 stations x 2 ports (4 ports), 14 kW grid limit, 7.2 kW chargers.
Discrete actions: 4 levels per port, jointly enumerated (4^4 = 256 actions).

**Evaluation.** All policies run on the same 30 pre-generated scenarios (seeds 1000-1029). Agents are selected on
validation scenarios (seeds 5000-5005). Paired per-scenario differences with 95% CIs are reported.

**Algorithms.** Q-Learning (tabular TD, simplified single-port problem); DQN (discrete env); PPO and SAC (continuous
env). DQN's discrete and PPO/SAC's continuous action sets differ, so the comparison is not strictly like-for-like.

**Syllabus mapping.** Implemented: RL agent/environment formulation, Temporal-Difference learning (Q-Learning),
Gymnasium environment, deep RL (DQN), RL in continuous action spaces (PPO, SAC), model training and deployment
(trainer + dashboard). **Not implemented:** dynamic programming and Monte-Carlo methods.
