# ADR 0003: reward = mission availability, with expensive restores

* Status: accepted (2026-10)

## Context

The first reward averaged packet delivery over benign *online* drones. On 30 validation
scenarios a uniformly random policy then scored 54 against 13 for doing nothing, because
re-flashing drones at random cleaned implants for free. With an 8-step restore, blanket
re-flashing was close to optimal and detection barely mattered.

## Decision

* `availability = (1/n) * sum_i [benign_i and online_i] * P(delivery_i)`: compromised and
  offline drones count as lost service.
* A restore takes the drone offline for 20 steps (`EnvConfig.restore_steps`).
* Small explicit penalties: active threat (0.25), per action (0.02 / n), per false block
  (0.25 / n).

## Consequences

On 40 validation scenarios: no-op 31.5, random 28.2, default watchdog 156.1, a lenient
watchdog 50.1, privileged oracle 184.2. Thresholds now matter, and "do everything" and
"do nothing" both lose (`tests/test_reward_hacking.py`). Changing these constants defines a
new benchmark version.
