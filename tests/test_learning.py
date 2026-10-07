"""Learned agents: discretisation, masks, GAE, numpy/torch parity, smoke training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fanet_defense import EnvConfig, ScenarioSampler, ScenarioSpec, generate_suite
from fanet_defense.rollout import NumpyPPOPolicy, actor_logits, critic_value, rollout_episode
from fanet_defense.tabular import (
    TabularConfig,
    TabularPolicy,
    action_mask,
    discretize,
    n_states,
    train_tabular,
)

CFG = EnvConfig()
K = CFG.k_neighbors


def make_obs(anom=0.0, pdr=1.0, t=0.5, slots=()):
    """One observation row; ``slots`` is a list of (present, dist, fwd, blocked)."""
    row = np.zeros(CFG.obs_dim, dtype=np.float32)
    row[:3] = (anom, pdr, t)
    for k, feat in enumerate(slots):
        row[3 + 4 * k : 3 + 4 * k + 4] = feat
    return row[None, :]


# ---------------------------------------------------------------- tabular helpers
def test_discretize_maps_every_observation_into_the_table():
    rng = np.random.default_rng(0)
    obs = rng.uniform(-1, 1, size=(500, CFG.obs_dim)).astype(np.float32)
    s = discretize(obs, CFG)
    assert s.min() >= 0 and s.max() < n_states(CFG)


def test_discretize_points_at_the_worst_unblocked_neighbour():
    obs = make_obs(
        anom=0.9,
        pdr=0.2,
        slots=[(1, 0.2, 0.9, 0), (1, 0.4, 0.1, 1), (1, 0.6, 0.3, 0)],
    )
    s = int(discretize(obs, CFG)[0])
    slot = s % K
    level = (s // K) % 4
    assert slot == 2  # slot 1 is worse but already blocked
    assert level == 0  # 0.3 < 0.4 -> lowest forwarding bin
    assert s // (K * 4) == 2 * 3 + 0  # anomaly bin 2, delivery bin 0


def test_discretize_flags_when_no_neighbour_is_usable():
    obs = make_obs(slots=[(0, 0, 0, 0)])
    s = int(discretize(obs, CFG)[0])
    assert (s // K) % 4 == 3 and s % K == 0


def test_action_mask_blocks_absent_or_already_blocked_slots():
    obs = make_obs(slots=[(1, 0.1, 0.9, 0), (1, 0.2, 0.8, 1)])
    mask = action_mask(obs, CFG)[0]
    assert mask[0] and mask[CFG.restore_action]  # no-op and restore always allowed
    assert mask[1] and not mask[2]  # slot 0 usable, slot 1 already blocked
    assert not mask[3] and not mask[4]  # empty slots


def test_tabular_policy_never_picks_a_masked_action():
    q = np.zeros((n_states(CFG), CFG.n_actions))
    q[:, 3] = 100.0  # the table loves "block slot 2", which does not exist here
    obs = make_obs(slots=[(1, 0.1, 0.9, 0)])
    assert TabularPolicy(q, CFG).act(obs)[0] != 3


def test_tabular_training_smoke_updates_the_table():
    sampler = ScenarioSampler("easy", n_range=(6, 8), horizon=20)
    val = generate_suite(2, 0, "val", sampler)
    for algo in ("q", "sarsa"):
        tcfg = TabularConfig(algo=algo, total_env_steps=120, eval_every_steps=60)
        pol, curve = train_tabular(tcfg, CFG, sampler, seed=0, evaluator=lambda p: float(len(val)))
        assert np.abs(pol.q).sum() > 0
        assert len(curve) >= 2 and curve[-1]["env_steps"] <= 140


# ---------------------------------------------------------------- PPO
torch = pytest.importorskip("torch")
from fanet_defense.ppo import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    collate,
    export_params,
    gae,
    train_ppo,
)


def test_gae_matches_a_hand_computed_example():
    rew = np.array([[1.0], [1.0]], dtype=np.float32)
    val = np.array([[0.5], [0.5]], dtype=np.float32)
    adv, ret = gae(rew, val, gamma=0.9, lam=0.8)
    # t=1: delta = 1 + 0 - 0.5 = 0.5 ; t=0: delta = 1 + 0.9*0.5 - 0.5 = 0.95, adv = 0.95 + 0.72*0.5
    assert adv[:, 0] == pytest.approx([1.31, 0.5])
    assert ret[:, 0] == pytest.approx([1.81, 1.0])


def test_numpy_forward_matches_torch_forward():
    torch.manual_seed(0)
    model = ActorCritic(CFG, hidden=32, gamma=0.97)
    obs = np.random.default_rng(1).uniform(-1, 1, size=(64, CFG.obs_dim)).astype(np.float32)
    params = export_params(model)
    with torch.no_grad():
        t_logits = model.logits(torch.as_tensor(obs)).numpy()
        t_value = model.value(torch.as_tensor(obs)).numpy()
    np.testing.assert_allclose(actor_logits(params, obs, CFG), t_logits, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(critic_value(params, obs), t_value, rtol=1e-4, atol=1e-4)


def test_numpy_policy_round_trips_through_npz(tmp_path):
    torch.manual_seed(0)
    params = export_params(ActorCritic(CFG, hidden=16, gamma=0.97))
    pol = NumpyPPOPolicy(params, CFG)
    path = tmp_path / "policy.npz"
    pol.save(path)
    again = NumpyPPOPolicy.load(path, CFG)
    obs = np.random.default_rng(2).uniform(-1, 1, size=(30, CFG.obs_dim)).astype(np.float32)
    assert np.array_equal(pol.act(obs), again.act(obs))


def test_rollout_is_reproducible_and_well_shaped():
    torch.manual_seed(0)
    params = export_params(ActorCritic(CFG, hidden=16, gamma=0.97))
    spec = ScenarioSpec(seed=4, n_drones=7, comm_range=450.0, horizon=25)
    a = rollout_episode((params, spec.to_dict(), CFG, 123))
    b = rollout_episode((params, spec.to_dict(), CFG, 123))
    assert a["obs"].shape == (25, 7, CFG.obs_dim) and a["act"].shape == (25, 7)
    assert a["rew"].shape == (25,) and a["n"] == 7
    assert np.array_equal(a["act"], b["act"]) and a["ret"] == b["ret"]


def test_collate_concatenates_agents_and_broadcasts_team_reward():
    torch.manual_seed(0)
    params = export_params(ActorCritic(CFG, hidden=16, gamma=0.97))
    eps = [
        rollout_episode((params, ScenarioSpec(seed=s, n_drones=n, horizon=10).to_dict(), CFG, s))
        for s, n in ((1, 5), (2, 6))
    ]
    batch = collate(eps, gamma=0.97, lam=0.95)
    assert batch["obs"].shape == (10 * 11, CFG.obs_dim)
    assert batch["adv"].shape == batch["ret"].shape == (110,)


def test_ppo_training_smoke_runs_end_to_end():
    sampler = ScenarioSampler("easy", n_range=(6, 8), horizon=20)
    val = generate_suite(2, 0, "val", sampler)
    pcfg = PPOConfig(
        total_env_steps=80,
        episodes_per_iter=2,
        eval_every_iters=1,
        minibatch=64,
        workers=1,
        torch_threads=1,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pol, curve = train_ppo(pcfg, CFG, sampler, seed=0, val_specs=val, executor=pool)
    assert len(curve) == 2
    assert all("val_return" in row for row in curve)
    assert pol.act(np.zeros((3, CFG.obs_dim), dtype=np.float32)).shape == (3,)
