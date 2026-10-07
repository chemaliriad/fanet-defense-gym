# Code walkthrough

A guided tour for reviewers (and for me, to explain every line). Read it with the code
open; each section names the function to look at and the question it answers.

## The pipeline in one picture

```
ScenarioSampler.sample(seed) ──> ScenarioSpec ──> FanetSim ──> FanetParallelEnv (PettingZoo)
        (scenario.py)              (frozen, hashed)   (sim.py)        (env_parallel.py)
                                                         │
                         policies.py / tabular.py / rollout.py (act on observations)
                                                         │
                         evaluate.py (paired runs, intervals) ──> scripts/run_experiments.py
```

## 1. Scenarios (`scenario.py`)

* **`ScenarioSpec`** is everything that varies between episodes. It is frozen, validated in
  `__post_init__`, JSON round-trips, and `scenario_id` hashes its canonical JSON. Question it
  answers: *can you replay exactly the scenario a model failed on?* Yes, from the spec alone.
* **`ScenarioSampler.sample(seed)`** is a pure function of the seed (it builds its own RNG from
  `SeedSequence([seed, 7919])`). Difficulty profiles are ranges, not fixed values.
* **`split_seeds` / `generate_suite` / `iter_suite`**: train, val and test are different
  `SeedSequence` namespaces, so a test scenario can never leak into training. `iter_suite`
  shards round-robin; the union of shards is the suite (tested).

## 2. The simulator (`sim.py`)

Read `FanetSim.step` first: it is the whole model in a dozen lines, in this order:

1. `_finish_restores`: re-flashed drones come back clean.
2. `_apply_actions`: decodes actions against the neighbour slots the agent *observed*
   (`self.slots`), not the current geometry, so an action means what the agent saw.
3. `_move`: random waypoint.
4. `_red_update`: implants wake up, misbehave unless stealthy, and exploit neighbours that have
   not blocked them. It draws a fixed number of uniforms per step (common random numbers).
5. `_build_network`: links, blocks (a block cuts the link both ways), multi-source BFS where
   active blackholes are extra "destinations" (`multi_source_bfs`), flood congestion, then
   `delivery_probabilities` (product of per-hop success along the next-hop tree).
6. `_update_evidence`: the only place observations are generated, from noisy evidence.
7. `_compute_slots`, `_obs_vector`: the K nearest neighbours become fixed-size slots.
8. `_reward_and_info`: availability minus small costs; also `local_reward`, whose mean is the
   team reward (tested), used for credit assignment during training only.

Questions to expect:

* *Why expected delivery probability instead of simulating packets?* A lower-variance reward and
  a cheaper step; observations still carry sampling noise (binomial watchdog counts and ACKs).
* *Why can't the agent see compromise?* `_obs_vector` only reads `anom`, `ema_pdr`, `ema_f`,
  slots, distances and its own blocks. The calibration test pins the anomaly AUC.
* *Why a TTL on blocks?* So false positives heal themselves at a cost, instead of fragmenting
  the network forever.

## 3. The environment adapter (`env_parallel.py`)

Thin by design: `reset` builds a `FanetSim` from a fixed spec, an `options["scenario"]`, or the
sampler; `step` converts the action dict to an array. Every agent has the same spaces, so one
network serves any swarm size. `state()` exposes privileged state for centralised critics.

## 4. Policies (`policies.py`, `tabular.py`, `rollout.py`)

* `WatchdogPolicy` is the classic heuristic in five vectorised lines; `tune_watchdog` searches
  its thresholds on validation only.
* `OraclePolicy` is privileged and says so (`privileged = True`).
* `tabular.discretize` maps the same observation to 144 cells (anomaly, delivery, worst
  usable neighbour and its slot). `action_mask` forbids blocking empty or already-blocked slots.
* `rollout.py` runs the policy network in numpy so worker processes never import torch, and
  checkpoints are plain `.npz` files.

## 5. Learning (`ppo.py`)

* One shared actor-critic (parameter sharing). The critic output is scaled by `1/(1-gamma)`.
* `collate` concatenates episodes along the agent axis and mixes local and team rewards
  (`local_weight = 0.5`); the sum over agents equals the team objective.
* `gae` is checked against a hand-computed example; `ppo_update` is the clipped surrogate with
  advantage normalisation, entropy bonus, gradient clipping and a KL early stop.
* Evaluation can be greedy or sampled; the mode is chosen on validation, never on test.

Design history worth telling: with the team reward alone, PPO barely moved (KL near zero for
100k steps); per-agent credit assignment fixed the learning signal. See `docs/design.md`.

## 6. Evaluation (`evaluate.py`)

* `run_episode` hands the simulator only to privileged policies (leakage boundary).
* `evaluate` keeps scenario order, so two policies' results are paired.
* `bootstrap_ci` (over scenarios), `seed_interval` (Student t over training seeds),
  `paired_difference` (refuses unpaired inputs).

## 7. Running it in five minutes

```bash
pip install -e ".[dev,train,viz]"
pytest -m "not slow" -q                                   # about a minute
fanet-defense evaluate --policy noop --n 20 --workers 2
fanet-defense evaluate --policy watchdog --n 20 --workers 2
fanet-defense demo --policy watchdog --seed 3 --step 90 --out snapshot.png
```

The return gap between `noop` and `watchdog` on the same 20 scenarios is the first thing to
show; `docs/results.md` has the full protocol.
