# ADR 0001: an independent environment, not a CybORG fork

* Status: accepted (2026-10)

## Context

CybORG's CAGE Challenge 3 (DroneSwarm) is the closest public benchmark to the problem of
defending a drone swarm. Its CC3 release pins `gym==0.23.1`, predates NumPy 2 and runs a
detailed action/host model that is slow per step. The goal here is different: generate
thousands of varied scenarios, train and evaluate many policies quickly, and expose the
same world to scripted, RL and LLM defenders.

## Decision

Write a new, small simulator (pure numpy) whose threat model is inspired by the CAGE-3
narrative (dormant hardware implants, worm-like spread, decentralised blue agents, keep data
flowing), with AODV-style routing attacks (blackhole, greyhole, flooding). No CybORG code is
copied. A faithful CybORG reproduction, if done, lives in a separate repository pinned to the
CC3 commit.

## Consequences

* Full control over determinism, seeding, sharding and speed (about 1 ms per step).
* Results are not comparable with the CAGE-3 leaderboard; the README says so.
* The abstractions (probabilistic links, instant route convergence) are documented as
  limitations in `docs/design.md`.
