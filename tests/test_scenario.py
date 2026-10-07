"""Scenario specification, sampler and suite generation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fanet_defense import ScenarioSampler, ScenarioSpec, generate_suite, iter_suite, split_seeds


def test_json_round_trip_preserves_the_spec_and_its_id():
    spec = ScenarioSampler().sample(123)
    again = ScenarioSpec.from_dict(json.loads(spec.to_json()))
    assert again == spec
    assert again.scenario_id == spec.scenario_id


def test_scenario_id_is_stable_and_discriminating():
    a, b = ScenarioSpec(seed=1), ScenarioSpec(seed=1)
    assert a.scenario_id == b.scenario_id
    assert ScenarioSpec(seed=2).scenario_id != a.scenario_id
    assert ScenarioSpec(seed=1, n_drones=13).scenario_id != a.scenario_id
    assert len(a.scenario_id) == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_drones": 2},
        {"comm_range": 0.0},
        {"speed_min": 10.0, "speed_max": 5.0},
        {"horizon": 0},
        {"base_loss": 1.0},
        {"n_implants": 99},
        {"wake_min": 0},
        {"wake_min": 9, "wake_max": 3},
        {"attack_probs": (0.5, 0.5, 0.5)},
        {"attack_probs": (1.0, 0.0)},
        {"stealth": 1.0},
        {"flaky_frac": 1.5},
        {"flaky_fwd": 0.0},
        {"detector_dprime": -1.0},
    ],
)
def test_invalid_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ScenarioSpec(seed=0, **kwargs)


def test_sampler_is_a_pure_function_of_the_seed():
    s = ScenarioSampler("mixed")
    assert s.sample(7) == s.sample(7)
    assert s.sample(7) != s.sample(8)


def test_sampler_respects_its_bounds():
    sampler = ScenarioSampler("mixed", n_range=(8, 14))
    for seed in range(200):
        spec = sampler.sample(seed)
        assert 8 <= spec.n_drones <= 14
        assert 1 <= spec.n_implants <= max(1, spec.n_drones // 3)
        assert spec.speed_min <= spec.speed_max
        assert spec.wake_min <= spec.wake_max
        assert np.isclose(sum(spec.attack_probs), 1.0)
        assert 0 < spec.comm_range <= 0.6 * spec.arena


def test_difficulty_orders_the_detector_quality_and_spread():
    def mean_of(diff: str, attr: str) -> float:
        specs = generate_suite(300, 0, "val", ScenarioSampler(diff))
        return float(np.mean([getattr(s, attr) for s in specs]))

    assert mean_of("easy", "detector_dprime") > mean_of("medium", "detector_dprime")
    assert mean_of("medium", "detector_dprime") > mean_of("hard", "detector_dprime")
    assert mean_of("easy", "exploit_prob") < mean_of("medium", "exploit_prob")
    assert mean_of("medium", "exploit_prob") < mean_of("hard", "exploit_prob")
    assert mean_of("easy", "flaky_frac") < mean_of("hard", "flaky_frac")


def test_a_suite_of_thousands_has_no_duplicate_scenarios():
    suite = generate_suite(3000, master_seed=11, split="train")
    ids = {s.scenario_id for s in suite}
    assert len(ids) == 3000


def test_splits_use_disjoint_seed_streams():
    seeds = {sp: set(split_seeds(5, sp, 2000)) for sp in ("train", "val", "test")}
    assert not seeds["train"] & seeds["val"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["val"] & seeds["test"]


def test_suite_is_reproducible_and_depends_on_master_seed():
    assert generate_suite(20, 1, "test") == generate_suite(20, 1, "test")
    assert generate_suite(20, 1, "test") != generate_suite(20, 2, "test")


def test_shards_are_disjoint_and_cover_the_suite_exactly():
    full = generate_suite(103, 4, "test")
    shards = [list(iter_suite(103, 4, "test", shard=i, num_shards=5)) for i in range(5)]
    flat = [s for shard in shards for s in shard]
    assert len(flat) == 103
    assert {s.scenario_id for s in flat} == {s.scenario_id for s in full}
    assert sum(len(sh) for sh in shards) == 103
    ids = [s.scenario_id for s in flat]
    assert len(set(ids)) == len(ids)


def test_shard_arguments_are_validated():
    with pytest.raises(ValueError):
        list(iter_suite(10, 0, "test", shard=3, num_shards=3))
