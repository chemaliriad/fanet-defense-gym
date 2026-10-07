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

Experiments cost about an hour of CPU on a laptop; `scripts/run_experiments.py` reproduces
them end to end.
