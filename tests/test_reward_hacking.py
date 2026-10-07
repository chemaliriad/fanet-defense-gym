"""Degenerate strategies must not pay: the reward cannot be gamed by blanket actions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fanet_defense import EnvConfig, FanetSim, ScenarioSampler, ScenarioSpec, generate_suite
from fanet_defense.evaluate import evaluate, run_episode
from fanet_defense.policies import NoOpPolicy, OraclePolicy, Policy, WatchdogPolicy

CFG = EnvConfig()


class BlockEverything(Policy):
    """Every drone blocks its first usable neighbour at every step."""

    name = "block-everything"

    def act(self, obs, sim=None):
        slots = obs[:, 3:].reshape(obs.shape[0], CFG.k_neighbors, 4)
        usable = (slots[..., 0] > 0.5) & (slots[..., 3] < 0.5)
        first = usable.argmax(axis=1)
        return np.where(usable.any(axis=1), first + 1, 0).astype(np.int64)


class RestoreEverything(Policy):
    """Every drone re-flashes itself as soon as it is back online."""

    name = "restore-everything"

    def act(self, obs, sim=None):
        return np.full(obs.shape[0], CFG.restore_action, dtype=np.int64)


class InvalidOnly(Policy):
    """Only actions that the simulator must ignore (blocks on empty slots)."""

    name = "invalid-only"

    def act(self, obs, sim=None):
        return np.full(obs.shape[0], CFG.k_neighbors, dtype=np.int64)  # last slot


@pytest.fixture(scope="module")
def suite() -> list[ScenarioSpec]:
    return generate_suite(12, master_seed=31, split="val", sampler=ScenarioSampler("mixed"))


@pytest.fixture(scope="module")
def watchdog_mean(suite) -> float:
    return float(np.mean([r.ret for r in evaluate(WatchdogPolicy(CFG), suite, CFG)]))


def test_blocking_everything_does_not_pay(suite, watchdog_mean):
    res = evaluate(BlockEverything(), suite, CFG)
    assert np.mean([r.ret for r in res]) < watchdog_mean
    assert np.mean([r.false_blocks for r in res]) > 50


def test_restoring_everything_does_not_pay(suite, watchdog_mean):
    res = evaluate(RestoreEverything(), suite, CFG)
    assert np.mean([r.ret for r in res]) < watchdog_mean
    assert np.mean([r.availability for r in res]) < 0.2  # drones are offline most of the time


def test_invalid_actions_are_exactly_a_no_op():
    """Ignored actions must leave the world untouched, bit for bit."""
    spec = ScenarioSpec(seed=8, n_drones=10, comm_range=1.0, n_implants=2, horizon=60)
    a = run_episode(NoOpPolicy(), spec, CFG)
    b = run_episode(InvalidOnly(), spec, CFG)
    assert a.ret == b.ret and a.availability == b.availability and b.actions == 0


def test_without_implants_doing_nothing_is_optimal():
    spec = ScenarioSpec(seed=2, n_drones=12, comm_range=420.0, n_implants=0, horizon=80)
    noop = run_episode(NoOpPolicy(), spec, CFG)
    oracle = run_episode(OraclePolicy(CFG), spec, CFG)
    watchdog = run_episode(WatchdogPolicy(CFG), spec, CFG)
    assert oracle.ret == noop.ret  # nothing to fix, nothing done
    assert watchdog.ret <= noop.ret + 1e-9  # any action is a false positive here


def test_isolated_swarm_has_zero_availability_but_no_crash():
    spec = ScenarioSpec(seed=5, n_drones=8, comm_range=1.0, n_implants=1, horizon=30)
    sim = FanetSim(spec, CFG)
    while not sim.done:
        _, r, _, info = sim.step(np.zeros(sim.n, dtype=np.int64))
        assert info["availability"] == 0.0 and np.isfinite(r)


def test_reward_never_exceeds_one_per_step():
    spec = dataclasses.replace(ScenarioSampler("easy").sample(3), horizon=50)
    sim = FanetSim(spec, CFG)
    rng = np.random.default_rng(0)
    while not sim.done:
        _, r, _, _ = sim.step(rng.integers(0, CFG.n_actions, size=sim.n))
        assert r <= 1.0 + 1e-12
