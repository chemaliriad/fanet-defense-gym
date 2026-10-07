# ADR 0002: observations come from noisy local evidence only

* Status: accepted (2026-10)

## Context

A defender trained with access to the true compromise state learns to read labels, not to
detect attacks. The author's first prototypes (2025, not published) had exactly this flaw:
the reward and the routing used the ground-truth list of malicious nodes, and one variant put
the label in the observation. Such results do not transfer and cannot be defended.

## Decision

* Observations are computed only from simulated telemetry with explicit noise models: host
  anomaly score with a calibrated separation `d'`, watchdog overhearing (binomial counts),
  end-to-end delivery feedback, own firewall state.
* Policies that need hidden state must set `privileged = True`; the evaluation harness passes
  the simulator to those policies only. The only privileged policy is the oracle baseline.
* Rewards may use ground truth (the simulator knows what happened); observations may not.

## Consequences

* Tests pin the boundary: non-privileged policies never receive the simulator, and the
  anomaly score's AUC stays in a calibrated band (informative, not an oracle).
* Detection quality becomes a scenario parameter that can be swept.
