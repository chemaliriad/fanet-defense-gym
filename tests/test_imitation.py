"""Watchdog demonstrations and supervised warm starts for PPO."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from fanet_defense import EnvConfig, ScenarioSampler, ScenarioSpec, generate_suite
from fanet_defense.policies import WatchdogPolicy
from fanet_defense.sim import FanetSim

torch = pytest.importorskip("torch")
from fanet_defense.imitation import behaviour_clone, collect_demonstrations  # noqa: E402
from fanet_defense.ppo import ActorCritic, PPOConfig, export_params, train_ppo  # noqa: E402
from fanet_defense.rollout import NumpyPPOPolicy  # noqa: E402

CFG = EnvConfig()


@pytest.fixture(autouse=True)
def single_torch_thread():
    torch.set_num_threads(1)


def test_demonstrations_are_reproducible_and_match_rollout_rows():
    class RestoreTeacher(WatchdogPolicy):
        def act(self, obs, sim=None):
            assert sim is None
            return np.full(len(obs), self.cfg.restore_action, dtype=np.int64)

    teacher = RestoreTeacher(CFG)
    specs = [ScenarioSpec(seed=s, n_drones=n, horizon=5) for s, n in ((1, 5), (2, 7))]
    obs, actions = collect_demonstrations(teacher, specs, CFG)
    again = collect_demonstrations(teacher, specs, CFG)
    assert obs.shape == (60, CFG.obs_dim) and obs.dtype == np.float32
    assert actions.shape == (60,) and actions.dtype == np.int64
    np.testing.assert_array_equal(obs, again[0])
    np.testing.assert_array_equal(actions, again[1])
    sim = FanetSim(specs[0], CFG)
    sim.step(teacher.act(sim.obs))
    assert not sim.online[: sim.n].any()
    np.testing.assert_array_equal(obs[5:10], sim.obs)


def test_clone_generalizes_to_held_out_demonstrations():
    teacher = WatchdogPolicy(CFG)
    sampler = ScenarioSampler("mixed", horizon=100)
    obs, actions = collect_demonstrations(teacher, generate_suite(6, 12, "train", sampler), CFG)
    held_obs, held_actions = collect_demonstrations(
        teacher, generate_suite(3, 12, "val", sampler), CFG
    )
    torch.manual_seed(0)
    model = ActorCritic(CFG, hidden=64, gamma=0.97)
    before = export_params(model)
    stats = behaviour_clone(model, obs, actions, epochs=30, lr=1e-3, batch_size=256, seed=0)
    policy = NumpyPPOPolicy(export_params(model), CFG)
    agreement = float(np.mean(policy.act(held_obs) == held_actions))
    print(f"held-out agreement: {agreement:.6f}")
    assert agreement >= 0.85
    assert np.isfinite(stats["loss"]) and stats["agreement"] >= 0.85
    for key in before:
        if key.startswith("c"):
            np.testing.assert_array_equal(before[key], policy.params[key])


def test_ppo_warm_start_records_validation_and_demo_budget():
    sampler = ScenarioSampler("easy", n_range=(6, 8), horizon=20)
    val = generate_suite(2, 0, "val", sampler)
    pcfg = PPOConfig(
        total_env_steps=40,
        episodes_per_iter=2,
        epochs=1,
        minibatch=64,
        workers=1,
        torch_threads=1,
        bc_scenarios=2,
        bc_epochs=2,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        policy, curve = train_ppo(pcfg, CFG, sampler, 0, val, executor=pool)
    assert isinstance(policy, NumpyPPOPolicy)
    assert len(curve) == 2 and curve[0]["env_steps"] == 0
    assert 0 <= curve[0]["bc_agreement"] <= 1
    assert curve[0]["bc_env_steps"] == 40
    assert np.isfinite(curve[0]["val_return"])
    assert np.isfinite(curve[0]["val_return_sampled"])
    assert curve[-1]["env_steps"] == 40


def test_ppo_default_bc_fields_preserve_exported_parameters():
    pcfg = PPOConfig(
        total_env_steps=40,
        episodes_per_iter=2,
        epochs=1,
        minibatch=64,
        workers=1,
        torch_threads=1,
    )
    explicit = replace(pcfg, bc_scenarios=0, bc_epochs=10, bc_lr=1e-3)
    assert (pcfg.bc_scenarios, pcfg.bc_epochs, pcfg.bc_lr) == (0, 10, 1e-3)
    sampler = ScenarioSampler("easy", n_range=(6, 8), horizon=20)
    with ThreadPoolExecutor(max_workers=1) as pool:
        before, curve = train_ppo(pcfg, CFG, sampler, 7, executor=pool)
        after, again = train_ppo(explicit, CFG, sampler, 7, executor=pool)
    for key in before.params:
        np.testing.assert_array_equal(before.params[key], after.params[key])
    assert len(curve) == len(again) == 1
    assert "bc_agreement" not in curve[0]
