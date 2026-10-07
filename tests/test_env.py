"""PettingZoo adapter: API conformance, seeding and spaces."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from fanet_defense import EnvConfig, ScenarioSampler, ScenarioSpec
from fanet_defense.env_parallel import FanetParallelEnv


def fixed_env(**kw) -> FanetParallelEnv:
    spec = ScenarioSpec(seed=9, n_drones=9, comm_range=450.0, n_implants=2, horizon=40)
    return FanetParallelEnv(scenario=spec, **kw)


def test_pettingzoo_parallel_api_conformance_on_a_fixed_scenario():
    parallel_api_test(fixed_env(), num_cycles=120)


def test_pettingzoo_api_conformance_with_a_sampler():
    env = FanetParallelEnv(sampler=ScenarioSampler("mixed", n_range=(6, 12)), max_drones=12)
    parallel_api_test(env, num_cycles=120)


def test_reset_seed_reproduces_the_same_scenario_and_observations():
    env = FanetParallelEnv(sampler=ScenarioSampler("hard"))
    o1, _ = env.reset(seed=3)
    spec1 = env.sim.spec
    o2, _ = env.reset(seed=3)
    assert env.sim.spec == spec1
    assert all(np.array_equal(o1[a], o2[a]) for a in o1)


def test_number_of_agents_follows_the_scenario():
    env = FanetParallelEnv(sampler=ScenarioSampler("mixed", n_range=(6, 14)), max_drones=14)
    sizes = set()
    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        sizes.add(len(obs))
        assert env.agents == env.possible_agents[: len(obs)]
    assert len(sizes) > 1


def test_all_agents_removed_after_truncation_and_step_before_reset_fails():
    env = fixed_env()
    env.reset(seed=0)
    for _ in range(40):
        _, _, term, trunc, _ = env.step(dict.fromkeys(env.agents, 0))
    assert env.agents == []
    assert all(trunc.values()) and not any(term.values())
    with pytest.raises(RuntimeError):
        env.step({})


def test_options_can_force_a_specific_scenario():
    env = FanetParallelEnv(sampler=ScenarioSampler("easy"))
    spec = ScenarioSpec(seed=77, n_drones=7, comm_range=500.0, horizon=10)
    obs, _ = env.reset(seed=0, options={"scenario": spec})
    assert len(obs) == 7 and env.sim.spec == spec


def test_scenarios_larger_than_max_drones_are_refused():
    env = FanetParallelEnv(max_drones=8)
    with pytest.raises(ValueError):
        env.reset(options={"scenario": ScenarioSpec(seed=0, n_drones=9)})


def test_spaces_match_the_config_and_contain_observations():
    cfg = EnvConfig(k_neighbors=3)
    env = fixed_env(config=cfg)
    obs, _ = env.reset(seed=0)
    for a, o in obs.items():
        assert env.observation_space(a).contains(o)
        assert o.shape == (cfg.obs_dim,)
    assert env.action_space("drone_0").n == cfg.n_actions == 5


def test_reward_is_shared_by_every_agent():
    env = fixed_env()
    env.reset(seed=0)
    _, rew, *_ = env.step(dict.fromkeys(env.agents, 0))
    assert len(set(rew.values())) == 1


def test_state_is_a_flat_privileged_vector():
    env = fixed_env()
    env.reset(seed=0)
    st = env.state()
    assert st.shape == (3 * env.sim.n,) and st.dtype == np.float32
