# ADR 0004: evaluation protocol

* Status: accepted (2026-10)

## Context

Single-run learning curves on the training distribution (what the author's 2025 prototypes
reported) do not support claims such as "algorithm A beats algorithm B": they mix
exploration noise, seed luck and over-fitting to one hand-made scenario.

## Decision

* Train on fresh scenarios every episode (seed stream of the `train` split).
* Tune every threshold and hyper-parameter on `val`; touch `test` once.
* Evaluate all policies on the same scenarios (paired), greedy, no learning during
  evaluation: 200 `test` scenarios plus three out-of-distribution families.
* Learned methods: 5 training seeds, identical interaction budget; report mean and 95 %
  Student-t interval over seeds. Fixed policies: 95 % bootstrap over scenarios. Paired
  differences against the tuned watchdog.
* Keep every run, including bad ones; numbers in the README come from `results/summary.json`.

## Consequences

The full protocol (5 seeds x 300k steps) costs a few hours of CPU on a laptop and runs with
`make repro`. The published results come from a lighter run (3 seeds x 150k steps, 2 cores)
whose exact settings are recorded in `results/summary.json`.
