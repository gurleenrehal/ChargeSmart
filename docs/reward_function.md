# Reward function

Implemented in `src/reward_function.py`; all weights are fields of `RewardConfig`.

```
R(t) = s * ( - [ a*Cost(t) + b*Peak(t) + g*Unmet(t) + d*Overload(t) + e*Fairness(t) + xi*Violations(t) ]
             + g*Progress(t) + w*Successes(t) )
```

| Term | Definition | Default weight |
|---|---|---|
| Cost(t) | `price_t * energy_t / (price_ref * E_day_ref)`, `E_day_ref = n_evs * E_ref` | a = 1.0 |
| Peak(t) | `max(0, load_t - peak_{t-1}) / grid_capacity` (telescopes to `peak/capacity` over the episode) | b = 0.5 |
| Unmet(t) | energy shortfall of EVs departing at t, divided by `E_day_ref` | g = 3.0 |
| Progress(t) | useful energy delivered at t, divided by `E_day_ref` | g = 3.0 (same as Unmet) |
| Overload(t) | `max(0, requested_kW - capacity) / capacity * dt_hours` | d = 2.0 |
| Fairness(t) | `(1 - Jain index of fulfilment ratios of connected EVs) / steps_per_day` | e = 0.3 |
| Violations(t) | number of EVs leaving with more than 0.1 kWh unmet | xi = 0.3 |
| Successes(t) | number of EVs leaving fully charged | w = 0.2 |
| s | global reward scale (does not change the optimal policy) | 10 |

## Why it cannot be gamed by not charging
For one EV, Unmet and Progress together contribute `-g*(2*U_dep - U_0)/E_day_ref`. Unmet energy is therefore penalised
twice as hard as the saving it buys, so skipping charging to avoid cost is never optimal. Progress is the dense
(potential-based) counterpart of the terminal shortfall penalty: it gives immediate credit assignment without changing
which policies minimise the final shortfall. `tests/test_environment.py::test_idle_agent_is_not_rewarded_more_than_charging_agent`
checks that an idle policy scores worse than a sensible one.

## Constraints are not in the reward
Grid capacity, charger limits, target SoC and departures are enforced by `ChargingSim`. Requests above the grid capacity
are curtailed proportionally; the Overload term only teaches the agent not to ask for them.

## Reward decomposition
`info["reward_terms"]` holds every term of each step and `info["episode_reward_terms"]` the episode sums
(shown in the dashboard's Algorithm Comparison page).

## Limitation
The weights were set by reasoning about magnitudes, not tuned systematically.
