# Environment design

This document is the contract of `fanet_defense_v0`: what is simulated, what each agent
sees and does, how reward is computed, how scenarios are generated and how policies are
evaluated. Code references point to `src/fanet_defense/`.

## 1. Scope

**What it is.** A fast, procedurally generated, partially observable cyber-defense
environment for a swarm of drones that relay telemetry to a ground control station (GCS)
over an ad hoc network. Defenders are decentralised: one local agent per drone. The
environment is meant for training and *evaluating* defensive policies (scripted, RL, or
LLM-driven) across thousands of distinct scenarios, with honest statistics.

**What it is not.** It is not a packet-level network emulator (no MAC layer, no real AODV
messages, no radio propagation model) and makes no claim about performance on real drones.
Links, losses and attacks are abstracted into probabilities so that one step takes about a
millisecond; [section 9](#9-limitations) lists the simplifications.

## 2. Threat model

Inspired by the public TTCP **CAGE Challenge 3** narrative (drones carrying hardware Trojans
that wake up and spread a worm; decentralised blue agents; keep data flowing), re-designed
from scratch for speed and procedural generation. No CybORG code is used.

* **Implants.** `n_implants` drones carry a supply-chain implant that stays dormant until a
  wake-up time drawn in `[wake_min, wake_max]`.
* **Behaviours once awake** (drawn per implant from `attack_probs`):
  * *blackhole*: forges fresh routes (advertises itself as the destination), attracts
    traffic and drops all of it (the classic AODV blackhole);
  * *greyhole*: relays honestly-routed traffic but drops a share `greyhole_drop` of it;
  * *flooder*: jams the channel, adding loss `flood_loss` to every neighbour that has not
    blocked it.
* **Stealth.** Each step, an awake implant behaves benignly with probability `stealth`.
* **Lateral movement.** Every awake implant exploits each reachable benign neighbour with
  probability `exploit_prob` per step. A neighbour that blocks the attacker cannot be
  exploited by it. New implants wake 1 to 5 steps later.
* **Benign confounders.** A share `flaky_frac` of healthy drones relays unreliably (battery,
  saturation), so a low forwarding ratio is evidence, not proof, of compromise.

The blue agent of a drone is assumed to run in a protected environment (e.g. a TEE): a
compromised application processor cannot disable it. This mirrors CAGE-3, where the
defender keeps acting on its host until the host is re-imaged.

## 3. Dynamics (one step = one second)

`FanetSim.step` applies, in this order:

1. **Restores that finish** clear the implant and bring the drone back online.
2. **Blue actions** (section 4) are applied, using the neighbour slots the agent observed.
3. **Mobility**: random waypoint in a square arena; the GCS is fixed at the centre.
4. **Adversary**: wake-ups, stealth draw, lateral movement.
5. **Network**: unit-disk links between online nodes, minus blocked links (a block cuts the
   link in both directions); multi-source BFS from the GCS *and* every active blackhole
   gives each drone its next hop (lowest index on ties). Per-hop success combines
   `base_loss` and flood congestion; relays forward with probability 1 (benign), the flaky
   factor, `1 - greyhole_drop`, or 0 (blackhole, offline). The probability that a drone's
   packet reaches the GCS is the product along its path (`delivery_probabilities`).
6. **Evidence** (what agents will observe), with independent noise.
7. **Reward and metrics.**

### Common random numbers

Randomness lives in four streams spawned from the scenario seed: initial state, mobility,
adversary, observation noise. The mobility and adversary streams draw a fixed number of
uniforms per step regardless of the defender's actions, so two policies evaluated on the
same scenario face the same motion and the same attacker dice. This makes paired
comparisons low-variance (tested in `tests/test_sim.py`).

## 4. Agents, observations, actions

Agents: `drone_0 .. drone_{n-1}` (PettingZoo `ParallelEnv`), all sharing the same spaces,
so one policy network can serve swarms of any size.

**Observation** (`Box(-1, 1, (19,))` with the default `k_neighbors = 4`):

| index | feature | source of evidence |
|---|---|---|
| 0 | host anomaly score, `tanh(z/2)` | `z ~ N(d' * mu, 1)`, `mu = 1` while misbehaving, `0.25` while dormant, `0` otherwise (`detector_dprime = d'`) |
| 1 | own delivery ratio (EMA) | ACKs: `Binomial(packets_per_step, p_deliver) / packets_per_step` |
| 2 | time `t / horizon` | clock |
| 3 + 4k | slot k present | k-th nearest online neighbour |
| 4 + 4k | distance / radio range | positions |
| 5 + 4k | watchdog forwarding ratio (EMA) | overheard retransmissions: `Binomial(relayed, fwd * (1 - base_loss) * (1 - congestion))` |
| 6 + 4k | already blocked by me | own firewall state |

**Actions** (`Discrete(6)`): `0` no-op; `1..4` block the neighbour in slot k for
`block_ttl = 25` steps; `5` restore (re-flash) this drone, offline for
`restore_steps = 20` steps, after which it is clean (its flakiness, being hardware, stays).
Blocking an empty slot, or acting while offline, is ignored and counted as `invalid`.

**What agents never see**: which drones are compromised, attack types, wake-up times,
positions of non-neighbours. Policies that need hidden state must declare
`privileged = True` (only the oracle does), and the harness passes the simulator to nobody
else (`tests/test_evaluate.py::test_non_privileged_policies_never_receive_the_simulator`).
The informativeness of the anomaly score is pinned by a calibration test: AUC within
`[0.78, 0.93]` for `d' = 1.5` (theory: 0.86).

## 5. Reward

Shared by all agents, per step:

```
reward = availability - 0.25 * threat - 0.02 * actions / n - 0.25 * false_blocks / n
availability = (1/n) * sum_i [drone i benign and online] * P(packet of i reaches the GCS)
threat       = (1/n) * #{misbehaving implants that still have a usable link}
```

Design choices, and the experiment behind each:

* **Availability counts every drone in the denominator.** An earlier version averaged over
  benign *online* drones only; a uniformly random policy then scored far above doing
  nothing (restores were free). Counting compromised and offline drones as lost service
  fixed it.
* **Restores are expensive (20 offline steps).** With 8 steps, blanket re-flashing was
  close to optimal and detection hardly mattered. With 20, on 40 validation scenarios:
  no-op 31, random 28, default watchdog 156, privileged oracle 184. Doing everything and
  doing nothing both lose (`tests/test_reward_hacking.py`).
* **Explicit penalties are small** (threat, actions, false blocks); most of the cost of a
  wrong action comes through availability, which is what an operator would measure.

Episodes have a fixed horizon (200 steps by default) and time is in the observation, so
the value after the last step is exactly zero.

**Credit assignment (training only).** The team reward is exactly the mean of per-drone
terms: the drone's own availability term, minus the threat it poses, minus the cost of its
own actions and false blocks (`FanetSim.local_reward`, pinned by a test). With a shared team
reward alone, one drone's action moves the signal by about `1/n`, and PPO barely learned
(KL near zero for 100k steps). Learners therefore train on
`0.5 * local + 0.5 * team` per agent; the sum over agents equals the team objective, and
**evaluation always uses the team reward**.

## 6. Scenario generation at scale

`ScenarioSpec` (frozen dataclass, JSON round-trip, 12-hex content hash) holds every
parameter of an episode. `ScenarioSampler` draws specs from difficulty profiles:

| field | easy | medium | hard |
|---|---|---|---|
| implants (share of swarm) | 5–15 % | 10–25 % | 15–30 % |
| `exploit_prob` | 0–0.004 | 0.004–0.012 | 0.010–0.030 |
| `stealth` | 0 | 0–0.2 | 0.1–0.4 |
| `detector_dprime` | 2.0–3.0 | 1.2–2.2 | 0.6–1.6 |
| `base_loss` | 0–0.03 | 0.01–0.06 | 0.02–0.08 |
| `flaky_frac` | 0–0.10 | 0.10–0.25 | 0.20–0.40 |

Shared across profiles: 8–20 drones; radio range set from a target mean degree of 4–7;
attack mix from a Dirichlet; speeds, packet rate, greyhole drop and flood loss drawn
uniformly. `mixed` picks a profile per scenario.

* **Splits are seed namespaces**, not slices of one trace: `train`, `val` and `test` seeds
  come from different `SeedSequence` streams and never overlap (tested on 2 000 seeds each).
* **Sharding**: `iter_suite(..., shard=i, num_shards=N)` yields disjoint subsets whose union
  is the suite, which is what the Kubernetes indexed job uses.
* **Out-of-distribution families** for evaluation: larger swarms (21–24 drones),
  flood-heavy attack mixes, and stealth floored at 0.5.

## 7. Evaluation protocol

* One episode per scenario, same scenarios for every policy (paired), in the same order.
* Metrics per episode: return, mean availability, mean compromised share, false blocks,
  false restores, time to containment (first step after the first active threat followed
  by 5 threat-free steps, censored at the horizon) and whether containment happened.
* Fixed policies: mean and 95 % percentile-bootstrap interval over scenarios.
* Learned policies: one number per training seed (mean over scenarios), then mean and 95 %
  Student-t interval over seeds. Paired differences against the tuned watchdog are
  reported with bootstrap intervals.
* Hyper-parameters and watchdog thresholds are chosen on `val`; `test` is touched once.
* PPO is evaluated either greedily or with sampled actions (a stochastic policy can be the
  better memoryless policy under partial observability); the mode is chosen once, on `val`,
  from the final validation scores of all seeds, and recorded in `results/summary.json`.

## 8. Baselines

* **No-op**, **random** (uniform over actions).
* **Watchdog** (Marti et al., 2000, in local form): restore when the anomaly score is high
  and own delivery is poor; otherwise block the worst neighbour if its forwarding ratio is
  below a threshold. A **tuned** variant gets its three thresholds from random search on
  `val`.
* **Oracle (privileged)**: restores every compromised drone, dormant or not. It reads the
  hidden state, so it is an upper reference, not a deployable defender.
* **Learned**: tabular Q-learning and SARSA on a discretised observation (same inputs,
  coarsened), and PPO with parameter sharing (independent PPO).

PPO can start by behaviour-cloning the watchdog on fresh training scenarios, using only
the local observations and the same action mask as PPO. Cross-entropy fits the shared
actor before PPO fine-tuning; the critic is not cloned. The `ppo_bc` baseline uses 64
scenarios and the validation-tuned watchdog, with its own greedy-or-sampled validation
choice. Its curve starts at `env_steps = 0` with validation returns and training agreement.
`bc_env_steps` records the demonstration simulator steps separately (not per-drone rows);
the training interaction budget is these steps plus the subsequent PPO `env_steps`.
Setting `bc_scenarios = 0` keeps the original PPO training path.

## 8b. Text interface for language models

`FanetTextEnv` (Gymnasium, `Text` observation and action spaces) exposes the same scenarios
to a central defender that answers in JSON. Design rules:

* **Same information**: the prompt is rendered from the observation vectors (plus neighbour
  identifiers), so a language model sees exactly the local evidence the drone agents see.
* **No injection channel**: prompts contain numbers and fixed tokens only; a test checks the
  character set and the absence of hidden-state words.
* **Defensive parsing**: answers can be wrapped in prose; malformed JSON, unknown drones,
  offline drones, illegal block targets and duplicates are ignored and reported; an optional
  per-error penalty exists for training.
* **Verifiable reward**: the simulator scores every answer; `decision_interval` lets a model
  act every few seconds to bound the number of calls per episode.
* **Equivalence test**: a policy played through text gets exactly its numeric return.

`dataset.iter_records` turns any teacher policy into prompt / answer / reward records
(JSONL), sharded like scenario suites, for supervised fine-tuning or as RL prompts.

## 9. Limitations

* Abstract link and loss model; no MAC contention, no real AODV route errors or timers.
* Routing re-converges instantly every step; real AODV takes time and can loop.
* One defender type per drone; no inter-agent messages (CAGE-3 has a broadcast channel).
* Attack behaviours are scripted, not learned; no adaptive attacker yet.
* The host anomaly score is a calibrated abstraction of an IDS, not a trained detector.
* Rewards use ground truth (as every simulator does); observations never do.

## 10. References

* TTCP CAGE Challenge 3 and CybORG: M. Standen et al., *CybORG: A Gym for the Development of
  Autonomous Cyber Agents*, IJCAI-21 ACD workshop; github.com/cage-challenge (MIT).
* C. Perkins, E. Belding-Royer, S. Das, *Ad hoc On-Demand Distance Vector (AODV) Routing*,
  RFC 3561, 2003.
* S. Marti, T. Giuli, K. Lai, M. Baker, *Mitigating routing misbehavior in mobile ad hoc
  networks*, MobiCom 2000 (watchdog and pathrater).
* J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017; *High-Dimensional
  Continuous Control Using Generalized Advantage Estimation*, 2016.
* C. Schroeder de Witt et al., *Is Independent Learning All You Need in the StarCraft
  Multi-Agent Challenge?*, 2020; C. Yu et al., *The Surprising Effectiveness of PPO in
  Cooperative Multi-Agent Games*, NeurIPS 2022.
* S. Huang et al., *CleanRL: High-quality Single-file Implementations of Deep
  Reinforcement Learning Algorithms*, JMLR 2022.
* J. K. Terry et al., *PettingZoo: Gym for Multi-Agent Reinforcement Learning*, NeurIPS 2021.
