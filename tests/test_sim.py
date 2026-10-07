"""Unit tests of the simulator on small, hand-checkable cases."""

from __future__ import annotations

import numpy as np
import pytest

from fanet_defense import EnvConfig, FanetSim, ScenarioSpec
from fanet_defense.sim import (
    BLACKHOLE,
    delivery_probabilities,
    multi_source_bfs,
    upstream_flows,
)

CFG = EnvConfig()


def line_graph(n: int) -> np.ndarray:
    """0 - 1 - 2 - ... - (n-1) path graph."""
    a = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = True
    return a


# ---------------------------------------------------------------- pure helpers
def test_bfs_on_a_line_with_gcs_at_one_end():
    usable = line_graph(5)
    sources = np.array([False, False, False, False, True])  # node 4 is the GCS
    depth, nh = multi_source_bfs(usable, sources)
    assert depth.tolist() == [4, 3, 2, 1, 0]
    assert nh.tolist() == [1, 2, 3, 4, -1]


def test_bfs_marks_disconnected_nodes_unreachable():
    usable = line_graph(4)
    usable[1, 2] = usable[2, 1] = False  # cut the line in two halves
    sources = np.array([False, False, False, True])
    depth, nh = multi_source_bfs(usable, sources)
    assert depth.tolist() == [-1, -1, 1, 0]
    assert nh[0] == -1 and nh[1] == -1


def test_blackhole_source_attracts_nearest_nodes():
    usable = line_graph(5)  # 0 - 1 - 2 - 3 - 4(gcs); node 1 is a blackhole
    sources = np.array([False, True, False, False, True])
    depth, nh = multi_source_bfs(usable, sources)
    assert depth.tolist() == [1, 0, 1, 1, 0]
    assert nh[0] == 1  # node 0 sends to the blackhole
    assert nh[2] == 1 or nh[2] == 3  # tie between blackhole and the way to the GCS
    assert nh[2] == 1  # lowest index wins ties, deterministic


def test_delivery_probability_is_product_along_the_path():
    n = 4  # 0 - 1 - 2 - 3(gcs)
    usable = line_graph(n)
    sources = np.array([False, False, False, True])
    depth, nh = multi_source_bfs(usable, sources)
    link_ok = np.full((n, n), 0.9)
    fwd = np.array([1.0, 0.5, 1.0, 1.0])  # node 1 forwards only half of what it receives
    p = delivery_probabilities(depth, nh, link_ok, fwd, gcs=3)
    # node 2 -> gcs: one hop
    assert p[2] == pytest.approx(0.9)
    # node 1 -> 2 -> gcs: two hops, relay 2 forwards everything
    assert p[1] == pytest.approx(0.9 * 1.0 * 0.9)
    # node 0 -> 1 -> 2 -> gcs: relay 1 forwards half
    assert p[0] == pytest.approx(0.9 * 0.5 * 0.9 * 1.0 * 0.9)
    assert p[3] == 0.0  # the GCS itself generates no flow


def test_blackhole_zeroes_delivery_of_flows_through_it():
    n = 3  # 0 - 1(blackhole) - 2(gcs)
    usable = line_graph(n)
    sources = np.array([False, True, True])
    depth, nh = multi_source_bfs(usable, sources)
    link_ok = np.full((n, n), 1.0)
    fwd = np.array([1.0, 0.0, 1.0])
    p = delivery_probabilities(depth, nh, link_ok, fwd, gcs=2)
    assert p[0] == 0.0


def test_upstream_flows_counts_descendants_not_self():
    usable = line_graph(4)  # 0 - 1 - 2 - 3(gcs)
    sources = np.array([False, False, False, True])
    depth, nh = multi_source_bfs(usable, sources)
    is_flow = np.array([True, True, True, False])
    up = upstream_flows(depth, nh, is_flow)
    assert up.tolist() == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------- simulator
def small_spec(**kw) -> ScenarioSpec:
    base = {"seed": 3, "n_drones": 10, "comm_range": 450.0, "n_implants": 2, "horizon": 60}
    base.update(kw)
    return ScenarioSpec(**base)


def noop(sim: FanetSim) -> np.ndarray:
    return np.zeros(sim.n, dtype=np.int64)


def test_reset_is_deterministic_for_a_seed():
    a, b = FanetSim(small_spec()), FanetSim(small_spec())
    assert np.array_equal(a.obs, b.obs)
    for _ in range(20):
        oa, ra, _, _ = a.step(noop(a))
        ob, rb, _, _ = b.step(noop(b))
        assert np.array_equal(oa, ob) and ra == rb


def test_different_seeds_give_different_worlds():
    a, b = FanetSim(small_spec(seed=1)), FanetSim(small_spec(seed=2))
    assert not np.array_equal(a.pos, b.pos)


def test_observation_shape_range_and_dtype():
    sim = FanetSim(small_spec())
    assert sim.obs.shape == (10, CFG.obs_dim)
    assert sim.obs.dtype == np.float32
    for _ in range(30):
        obs, *_ = sim.step(noop(sim))
        assert obs.min() >= -1.0 and obs.max() <= 1.0
        assert np.isfinite(obs).all()


def test_team_reward_is_the_mean_of_local_rewards():
    sim = FanetSim(small_spec(n_implants=3, wake_min=1, wake_max=5, exploit_prob=0.05))
    rng = np.random.default_rng(1)
    for _ in range(40):
        _, r, _, _ = sim.step(rng.integers(0, CFG.n_actions, size=sim.n))
        assert sim.local_reward.shape == (sim.n,)
        assert r == pytest.approx(float(sim.local_reward.mean()), abs=1e-12)


def test_episode_ends_exactly_at_horizon():
    sim = FanetSim(small_spec(horizon=15))
    steps = 0
    while not sim.done:
        sim.step(noop(sim))
        steps += 1
    assert steps == 15
    with pytest.raises(RuntimeError):
        sim.step(noop(sim))


def test_invalid_actions_are_rejected():
    sim = FanetSim(small_spec())
    with pytest.raises(ValueError):
        sim.step(np.zeros(sim.n + 1, dtype=np.int64))
    with pytest.raises(ValueError):
        sim.step(np.full(sim.n, CFG.n_actions, dtype=np.int64))
    with pytest.raises(ValueError):
        sim.step(np.full(sim.n, -1, dtype=np.int64))


def test_no_implants_means_no_threat_and_high_availability():
    sim = FanetSim(small_spec(n_implants=0, base_loss=0.0, flaky_frac=0.0))
    rewards = []
    for _ in range(40):
        _, r, _, info = sim.step(noop(sim))
        rewards.append(r)
        assert info["threat"] == 0.0 and info["n_compromised"] == 0
    assert np.mean(rewards) > 0.9


def test_mobility_and_adversary_streams_do_not_depend_on_actions():
    """Common random numbers: positions and implant wake-ups are identical across policies."""
    a, b = FanetSim(small_spec(seed=5)), FanetSim(small_spec(seed=5))
    rng = np.random.default_rng(0)
    for _ in range(30):
        a.step(noop(a))
        b.step(rng.integers(0, CFG.n_actions, size=b.n))
        assert np.allclose(a.pos, b.pos)


def test_blackhole_drops_traffic_once_awake():
    spec = small_spec(
        n_implants=1, attack_probs=(1.0, 0.0, 0.0), wake_min=3, wake_max=3, exploit_prob=0.0
    )
    sim = FanetSim(spec)
    implant = int(np.flatnonzero(sim.comp)[0])
    assert sim.atk[implant] == BLACKHOLE
    first = sim.step(noop(sim))[3]
    pdr_before = first["pdr"]
    pdrs = [sim.step(noop(sim))[3]["pdr"] for _ in range(10)]
    assert any(sim.behaving)
    assert min(pdrs) < pdr_before  # delivery degrades after the blackhole wakes up


def test_restore_cleans_a_compromised_drone_after_the_offline_period():
    spec = small_spec(n_implants=1, exploit_prob=0.0, wake_min=1, wake_max=1)
    sim = FanetSim(spec)
    implant = int(np.flatnonzero(sim.comp)[0])
    acts = noop(sim)
    acts[implant] = CFG.restore_action
    sim.step(acts)
    assert not sim.online[implant]  # offline while re-flashing
    for _ in range(CFG.restore_steps - 1):
        sim.step(noop(sim))
        assert not sim.online[implant]
    sim.step(noop(sim))
    assert sim.online[implant] and not sim.comp[implant]


def test_blocking_a_neighbour_cuts_the_link_for_the_ttl_then_expires():
    sim = FanetSim(small_spec(n_implants=0, exploit_prob=0.0))
    i = 0
    slot = int(np.flatnonzero(sim.slots[i] >= 0)[0])
    target = int(sim.slots[i, slot])
    acts = noop(sim)
    acts[i] = slot + 1
    sim.step(acts)
    assert sim.block_until[i, target] > sim.t
    assert not sim.net.usable[i, target] and not sim.net.usable[target, i]
    # the block silently expires after block_ttl steps
    for _ in range(CFG.block_ttl):
        sim.step(noop(sim))
    assert sim.block_until[i, target] <= sim.t


def test_blocking_a_benign_neighbour_is_counted_as_false_block():
    sim = FanetSim(small_spec(n_implants=0))
    slot = int(np.flatnonzero(sim.slots[0] >= 0)[0])
    acts = noop(sim)
    acts[0] = slot + 1
    _, _, _, info = sim.step(acts)
    assert info["fp_blocks"] == 1 and info["tp_blocks"] == 0


def test_action_on_an_empty_slot_is_invalid_but_harmless():
    sim = FanetSim(small_spec(comm_range=1.0, n_implants=0))  # nobody in range
    acts = np.ones(sim.n, dtype=np.int64)
    _, _, _, info = sim.step(acts)
    assert info["invalid"] == sim.n and info["actions"] == 0


def test_offline_drones_cannot_act():
    sim = FanetSim(small_spec(n_implants=0))
    acts = noop(sim)
    acts[0] = CFG.restore_action
    sim.step(acts)
    acts[0] = CFG.restore_action  # still offline: must be ignored
    _, _, _, info = sim.step(acts)
    assert info["invalid"] == 1 and info["restores"] == 0


def test_lateral_movement_spreads_without_defense():
    spec = small_spec(
        n_implants=2,
        exploit_prob=0.08,
        attack_probs=(0.0, 1.0, 0.0),
        wake_min=1,
        wake_max=2,
        horizon=120,
    )
    sim = FanetSim(spec)
    start = int(sim.comp.sum())
    while not sim.done:
        sim.step(noop(sim))
    assert int(sim.comp.sum()) > start


def test_blocking_prevents_lateral_movement_to_the_blocker():
    """A drone that blocks every other node can no longer be exploited by them."""
    spec = small_spec(
        n_implants=1,
        exploit_prob=0.5,
        wake_min=1,
        wake_max=1,
        attack_probs=(0.0, 1.0, 0.0),
        horizon=40,
    )

    def run(shield: bool) -> bool:
        sim = FanetSim(spec)
        implant = int(np.flatnonzero(sim.comp)[0])
        victim = next(j for j in range(sim.n) if j != implant and sim.net.adj[j, implant])
        for _ in range(30):
            if shield:
                sim.block_until[victim, :] = 10**6  # victim blocks everyone, permanently
            sim.step(noop(sim))
        return bool(sim.comp[victim])

    assert run(shield=False), "control: without blocking the victim must get infected"
    assert not run(shield=True)
