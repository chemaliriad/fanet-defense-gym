"""Evaluation harness, baselines ordering and information-leakage checks."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fanet_defense import EnvConfig, FanetSim, ScenarioSampler, ScenarioSpec, generate_suite
from fanet_defense.evaluate import _time_to_containment as ttc
from fanet_defense.evaluate import (
    bootstrap_ci,
    evaluate,
    paired_difference,
    run_episode,
    seed_interval,
    summarize,
)
from fanet_defense.policies import NoOpPolicy, OraclePolicy, Policy, RandomPolicy, WatchdogPolicy

CFG = EnvConfig()


@pytest.fixture(scope="module")
def suite() -> list[ScenarioSpec]:
    return generate_suite(
        16, master_seed=21, split="val", sampler=ScenarioSampler("mixed", n_range=(8, 12))
    )


# ---------------------------------------------------------------- containment metric
def test_ttc_when_nothing_ever_attacked():
    assert ttc(np.zeros(50)) == (0.0, True)


def test_ttc_counts_from_first_threat_to_first_quiet_window():
    threat = np.array([0, 0, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0.3, 0])
    assert ttc(threat, hold=5) == (3.0, True)  # threat starts at 2, quiet from 5 on


def test_ttc_is_censored_when_the_threat_is_never_contained():
    threat = np.array([0, 0.5, 0.5, 0.5, 0.5])
    assert ttc(threat) == (4.0, False)


# ---------------------------------------------------------------- statistics
def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.normal(5.0, 2.0, size=10)
    large = rng.normal(5.0, 2.0, size=400)
    e1, lo1, hi1 = bootstrap_ci(small)
    e2, lo2, hi2 = bootstrap_ci(large)
    assert lo1 <= e1 <= hi1 and lo2 <= e2 <= hi2
    assert (hi2 - lo2) < (hi1 - lo1)


def test_bootstrap_of_a_single_value_is_degenerate_not_an_error():
    assert bootstrap_ci([3.0]) == (3.0, 3.0, 3.0)


def test_seed_interval_uses_student_t_for_small_samples():
    m, _, hi = seed_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0
    assert hi - m == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))


def test_paired_difference_requires_identical_scenarios(suite):
    a = evaluate(NoOpPolicy(), suite[:4], CFG)
    b = evaluate(NoOpPolicy(), suite[1:5], CFG)
    with pytest.raises(ValueError):
        paired_difference(a, b)


# ---------------------------------------------------------------- policies
def test_evaluation_is_deterministic(suite):
    r1 = evaluate(WatchdogPolicy(CFG), suite[:5], CFG)
    r2 = evaluate(WatchdogPolicy(CFG), suite[:5], CFG)
    assert [r.ret for r in r1] == [r.ret for r in r2]


def test_random_policy_is_reproducible_per_scenario(suite):
    p = RandomPolicy(CFG, seed=1)
    assert run_episode(p, suite[0], CFG).ret == run_episode(p, suite[0], CFG).ret


def test_baseline_ordering_noop_below_watchdog_below_oracle(suite):
    means = {
        p.name: np.mean([r.ret for r in evaluate(p, suite, CFG)])
        for p in (NoOpPolicy(), RandomPolicy(CFG), WatchdogPolicy(CFG), OraclePolicy(CFG))
    }
    assert means["noop"] < means["watchdog"] < means["oracle"]
    assert means["random"] < means["watchdog"]


def test_oracle_contains_threats_and_never_restores_a_benign_drone(suite):
    res = evaluate(OraclePolicy(CFG), suite, CFG)
    assert all(r.false_restores == 0 for r in res)
    assert np.mean([r.compromised for r in res]) < 0.1


def test_summarize_reports_every_metric_with_an_interval(suite):
    s = summarize(evaluate(NoOpPolicy(), suite[:6], CFG))
    assert {"ret", "availability", "ttc", "contained"} <= set(s)
    assert all(lo <= m <= hi for m, lo, hi in s.values())


def test_parallel_evaluation_matches_serial(suite):
    serial = evaluate(WatchdogPolicy(CFG), suite[:6], CFG, workers=1)
    parallel = evaluate(WatchdogPolicy(CFG), suite[:6], CFG, workers=2)
    assert [r.ret for r in serial] == [r.ret for r in parallel]


# ---------------------------------------------------------------- information boundaries
class _SimSpy(Policy):
    name = "spy"

    def __init__(self) -> None:
        self.saw_sim = False

    def act(self, obs, sim=None):
        self.saw_sim = self.saw_sim or sim is not None
        return np.zeros(obs.shape[0], dtype=np.int64)


def test_non_privileged_policies_never_receive_the_simulator(suite):
    spy = _SimSpy()
    run_episode(spy, suite[0], CFG)
    assert not spy.saw_sim


def test_oracle_cannot_run_without_the_simulator():
    with pytest.raises(ValueError):
        OraclePolicy(CFG).act(np.zeros((3, CFG.obs_dim), dtype=np.float32), None)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def test_anomaly_score_is_informative_but_not_an_oracle():
    """With d'=1.5 the expected AUC is Phi(d'/sqrt2) ~ 0.86: far from 0.5 and from 1."""
    spec = ScenarioSpec(
        seed=0,
        n_drones=14,
        comm_range=450.0,
        n_implants=4,
        wake_min=1,
        wake_max=1,
        exploit_prob=0.0,
        detector_dprime=1.5,
        horizon=200,
    )
    pos, neg = [], []
    for seed in range(6):
        sim = FanetSim(dataclasses.replace(spec, seed=seed), CFG)
        while not sim.done:
            obs, *_ = sim.step(np.zeros(sim.n, dtype=np.int64))
            bad = sim.behaving[: sim.n]
            pos.extend(obs[bad, 0])
            neg.extend(obs[~sim.comp[: sim.n], 0])
    auc = _auc(np.asarray(pos), np.asarray(neg))
    assert 0.78 < auc < 0.93
