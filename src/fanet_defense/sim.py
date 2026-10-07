"""Core simulator: mobility, adversary, routing, delivery model and local telemetry.

Everything is plain numpy and deterministic given ``ScenarioSpec.seed`` and the sequence of
blue actions. Randomness is split into independent streams (mobility, adversary,
observation noise). The mobility and adversary streams consume a *fixed* number of uniform
draws per step whatever the defender does, so two policies facing the same scenario see the
same drone motion and the same attacker dice (common random numbers). Observation noise is
state dependent and is therefore not shared across policies.

Model in one paragraph
----------------------
``n`` drones move by random waypoint in a square arena; a ground control station (GCS) sits
at the centre. Two nodes share a link when they are within radio range. Every drone sends
telemetry to the GCS along hop-count shortest paths (AODV-like next-hop tree). Some drones
carry supply-chain implants that wake up later and then misbehave: *blackholes* forge fresh
routes so that neighbours send them traffic, then drop it; *greyholes* drop a share of the
traffic they relay; *flooders* jam the channel around them. Awake implants also exploit
neighbouring drones (lateral movement). A local blue agent on each drone sees only noisy,
local evidence and can block a neighbour (temporary) or restore itself (re-flash, offline for
a few steps). The reward is the mission availability (benign, online drones whose telemetry
reaches the GCS) minus the cost of active threats and of wasteful actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .scenario import ScenarioSpec

NOOP = 0
BLACKHOLE, GREYHOLE, FLOOD = 0, 1, 2
_NEVER = np.iinfo(np.int64).max


# --------------------------------------------------------------------------------------
# Pure helpers (unit-tested on hand-built graphs)
# --------------------------------------------------------------------------------------
def multi_source_bfs(usable: np.ndarray, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hop distance to the nearest source and the next hop towards it.

    ``usable`` is a symmetric boolean adjacency matrix, ``sources`` a boolean mask. Sources
    have depth 0 and no next hop; unreachable nodes have depth -1. Ties go to the lowest
    node index, which makes the result deterministic.
    """
    n_nodes = usable.shape[0]
    depth = np.full(n_nodes, -1, dtype=np.int64)
    next_hop = np.full(n_nodes, -1, dtype=np.int64)
    depth[sources] = 0
    fidx = np.flatnonzero(sources)
    level = 0
    while fidx.size:
        reach = usable[:, fidx]
        cand = np.flatnonzero(reach.any(axis=1) & (depth < 0))
        if cand.size == 0:
            break
        level += 1
        next_hop[cand] = fidx[reach[cand].argmax(axis=1)]
        depth[cand] = level
        fidx = cand
    return depth, next_hop


def delivery_probabilities(
    depth: np.ndarray,
    next_hop: np.ndarray,
    link_ok: np.ndarray,
    fwd_prob: np.ndarray,
    gcs: int,
) -> np.ndarray:
    """Probability that a packet generated at each node reaches the GCS.

    ``p[i] = link_ok[i, nh(i)] * q[nh(i)]`` and ``q[u] = fwd_prob[u] * p[u]`` with
    ``q[gcs] = 1`` and ``q = 0`` at other sources (blackholes). Unreachable nodes get 0.
    """
    p = np.zeros(depth.size)
    q = np.zeros(depth.size)
    q[gcs] = 1.0
    for lvl in range(1, int(depth.max()) + 1):
        idx = np.flatnonzero(depth == lvl)
        nxt = next_hop[idx]
        p[idx] = link_ok[idx, nxt] * q[nxt]
        q[idx] = fwd_prob[idx] * p[idx]
    return p


def upstream_flows(depth: np.ndarray, next_hop: np.ndarray, is_flow: np.ndarray) -> np.ndarray:
    """Number of other flows whose route traverses each node (excluding its own flow)."""
    own = (is_flow & (depth > 0)).astype(np.float64)
    sub = own.copy()
    for lvl in range(int(depth.max()), 0, -1):
        idx = np.flatnonzero(depth == lvl)
        sub += np.bincount(next_hop[idx], weights=sub[idx], minlength=sub.size)
    return sub - own


@dataclass
class Network:
    """Routing and delivery state for one step."""

    adj: np.ndarray  # (N,N) physical links between online nodes
    usable: np.ndarray  # (N,N) links not cut by a block
    depth: np.ndarray
    next_hop: np.ndarray
    p_deliver: np.ndarray  # (N,) packet delivery probability to the GCS
    inflow: np.ndarray  # (N,) upstream flows traversing the node
    fwd_prob: np.ndarray  # (N,) probability the node retransmits a received packet
    extra_loss: np.ndarray  # (N,) congestion loss caused by active flooders
    n_cut_links: int  # physical links currently cut by a block


class FanetSim:
    """One episode of one scenario. Drones are nodes ``0..n-1``; the GCS is node ``n``."""

    def __init__(self, spec: ScenarioSpec, config: EnvConfig | None = None) -> None:
        self.spec = spec
        self.cfg = config or EnvConfig()
        self.n = spec.n_drones
        self.n_nodes = self.n + 1
        self.gcs = self.n
        self._eye = np.eye(self.n_nodes, dtype=bool)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> np.ndarray:
        s, n, nn = self.spec, self.n, self.n_nodes
        c_init, c_mob, c_red, c_obs = np.random.SeedSequence(s.seed).spawn(4)
        self._rng_mob = np.random.default_rng(c_mob)
        self._rng_red = np.random.default_rng(c_red)
        self._rng_obs = np.random.default_rng(c_obs)
        rng = np.random.default_rng(c_init)

        self.t = 0
        self.pos = np.zeros((nn, 2))
        self.pos[:n] = rng.uniform(0.0, s.arena, size=(n, 2))
        self.pos[self.gcs] = (s.arena / 2.0, s.arena / 2.0)
        self.wp = rng.uniform(0.0, s.arena, size=(nn, 2))
        self.speed = rng.uniform(s.speed_min, s.speed_max, size=nn)

        self.comp = np.zeros(nn, dtype=bool)
        self.atk = np.full(nn, -1, dtype=np.int64)
        self.wake = np.full(nn, _NEVER, dtype=np.int64)
        implants = rng.choice(n, size=s.n_implants, replace=False)
        self.comp[implants] = True
        self.atk[implants] = rng.choice(3, size=s.n_implants, p=np.asarray(s.attack_probs))
        self.wake[implants] = rng.integers(s.wake_min, s.wake_max + 1, size=s.n_implants)

        # Benign relays are not all equally reliable; restoring a drone does not fix this.
        self.benign_fwd = np.ones(nn)
        flaky = rng.random(n) < s.flaky_frac
        jitter = np.clip(rng.normal(s.flaky_fwd, 0.07, size=n), 0.4, 0.95)
        self.benign_fwd[:n] = np.where(flaky, jitter, 1.0)

        self.behaving = np.zeros(nn, dtype=bool)  # awake implants that misbehave this step
        self.restore_until = np.zeros(nn, dtype=np.int64)  # 0 = online
        self.online = np.ones(nn, dtype=bool)
        self.block_until = np.zeros((nn, nn), dtype=np.int64)  # [i, j] > now: i blocks j
        self.ema_f = np.ones((nn, nn))  # watchdog ratio of j as estimated by i
        self.ema_pdr = np.ones(n)
        self.anom = np.zeros(n)
        self.slots = np.full((n, self.cfg.k_neighbors), -1, dtype=np.int64)

        self._dist = self._pairwise_distances()
        adj = self._physical_links()
        self.net = self._build_network(0, adj)
        self.ema_pdr = self.net.p_deliver[:n].copy()
        self._update_evidence(self.net)
        self._compute_slots(self.net)
        self.obs = self._obs_vector(0)
        self.last_info: dict[str, Any] = {}
        self.local_reward = np.zeros(n)
        self._acted = np.zeros(n)
        self._false_blocked = np.zeros(n)
        return self.obs

    @property
    def done(self) -> bool:
        return self.t >= self.spec.horizon

    # ------------------------------------------------------------------ stepping
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one second. ``actions`` has shape ``(n,)`` with values in ``[0, K+1]``."""
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError(f"expected actions of shape ({self.n},), got {actions.shape}")
        if actions.min(initial=0) < 0 or actions.max(initial=0) >= self.cfg.n_actions:
            raise ValueError(f"actions must be in [0, {self.cfg.n_actions - 1}]")
        if self.done:
            raise RuntimeError("episode is over; call reset()")

        now = self.t + 1
        self._finish_restores(now)
        counters = self._apply_actions(actions, now)
        self._move()
        self._dist = self._pairwise_distances()
        adj = self._physical_links()
        self._red_update(now, adj)
        self.net = self._build_network(now, adj)
        self._update_evidence(self.net)
        self._compute_slots(self.net)
        self.t = now
        self.obs = self._obs_vector(now)
        reward, info = self._reward_and_info(counters, now)
        self.last_info = info
        return self.obs, reward, self.done, info

    # ------------------------------------------------------------------ blue actions
    def _finish_restores(self, now: int) -> None:
        finished = (self.restore_until > 0) & (self.restore_until <= now)
        if finished.any():
            self.comp[finished] = False
            self.atk[finished] = -1
            self.wake[finished] = _NEVER
            self.restore_until[finished] = 0
            self.online[finished] = True

    def _apply_actions(self, actions: np.ndarray, now: int) -> dict[str, int]:
        cfg = self.cfg
        c = {"fp_blocks": 0, "tp_blocks": 0, "restores": 0, "fp_restores": 0, "invalid": 0}
        self._acted = np.zeros(self.n)
        self._false_blocked = np.zeros(self.n)
        for i in np.flatnonzero(actions != NOOP):
            a = int(actions[i])
            if self.restore_until[i] > 0:  # offline drones cannot act
                c["invalid"] += 1
            elif a == cfg.restore_action:
                self.restore_until[i] = now + cfg.restore_steps
                self.online[i] = False
                c["restores"] += 1
                c["fp_restores"] += int(not self.comp[i])
                self._acted[i] = 1.0
            else:
                target = int(self.slots[i, a - 1])
                if target < 0:
                    c["invalid"] += 1
                    continue
                self.block_until[i, target] = now + cfg.block_ttl
                self._acted[i] = 1.0
                if self.comp[target]:
                    c["tp_blocks"] += 1
                else:
                    c["fp_blocks"] += 1
                    self._false_blocked[i] = 1.0
        return c

    # ------------------------------------------------------------------ world dynamics
    def _pairwise_distances(self) -> np.ndarray:
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1))

    def _physical_links(self) -> np.ndarray:
        on = self.online
        return (self._dist <= self.spec.comm_range) & ~self._eye & on[:, None] & on[None, :]

    def _move(self) -> None:
        s, n = self.spec, self.n
        d = self.wp[:n] - self.pos[:n]
        dist = np.sqrt((d * d).sum(axis=1))
        step_len = self.speed[:n]
        arrive = dist <= step_len
        safe = np.where(dist > 0.0, dist, 1.0)
        moved = self.pos[:n] + d * (step_len / safe)[:, None]
        self.pos[:n] = np.where(arrive[:, None], self.wp[:n], moved)
        new_wp = self._rng_mob.uniform(0.0, s.arena, size=(n, 2))
        new_sp = self._rng_mob.uniform(s.speed_min, s.speed_max, size=n)
        self.wp[:n] = np.where(arrive[:, None], new_wp, self.wp[:n])
        self.speed[:n] = np.where(arrive, new_sp, self.speed[:n])

    def _red_update(self, now: int, adj_phys: np.ndarray) -> None:
        """Implants wake up, misbehave (or lie low) and exploit neighbours."""
        s, n = self.spec, self.n
        online = self.online[:n]
        awake = self.comp[:n] & (self.wake[:n] <= now) & online
        # Fixed number of uniform draws per step, independent of the defender's behaviour.
        u_stealth = self._rng_red.random(n)
        u_exploit = self._rng_red.random(n)
        u_type = self._rng_red.random(n)
        u_delay = self._rng_red.random(n)

        self.behaving[:n] = awake & (u_stealth >= s.stealth)

        blocked_by_target = self.block_until[:n, :n] > now  # [j, a]: j blocks attacker a
        reach = adj_phys[:n, :n] & ~blocked_by_target
        n_attackers = (reach & self.behaving[None, :n]).sum(axis=1)
        p_infect = 1.0 - (1.0 - s.exploit_prob) ** n_attackers
        newly = (~self.comp[:n]) & online & (u_exploit < p_infect)
        if newly.any():
            cum = np.cumsum(np.asarray(s.attack_probs))
            new_type = np.minimum(np.searchsorted(cum, u_type, side="right"), 2)
            self.comp[:n] |= newly
            self.atk[:n] = np.where(newly, new_type, self.atk[:n])
            self.wake[:n] = np.where(newly, now + 1 + (u_delay * 5).astype(np.int64), self.wake[:n])

    def _build_network(self, now: int, adj: np.ndarray) -> Network:
        s, nn = self.spec, self.n_nodes
        blk = self.block_until > now
        cut = blk | blk.T
        usable = adj & ~cut

        beh = self.behaving
        is_bh = beh & (self.atk == BLACKHOLE)
        is_gh = beh & (self.atk == GREYHOLE)
        is_fl = beh & (self.atk == FLOOD)

        sources = np.zeros(nn, dtype=bool)
        sources[self.gcs] = True
        sources |= is_bh  # blackholes advertise themselves as the destination
        depth, next_hop = multi_source_bfs(usable, sources)

        # A node suffers congestion from every adjacent flooder it has not blocked.
        extra = np.zeros(nn)
        if is_fl.any():
            exposed = (adj[:, is_fl] & ~blk[:, is_fl]).sum(axis=1)
            extra = 1.0 - (1.0 - s.flood_loss) ** exposed

        keep = 1.0 - extra
        link_ok = (1.0 - s.base_loss) * keep[:, None] * keep[None, :]
        fwd_prob = self.benign_fwd.copy()
        fwd_prob[is_gh] = 1.0 - s.greyhole_drop
        fwd_prob[is_bh] = 0.0
        fwd_prob[~self.online] = 0.0
        p = delivery_probabilities(depth, next_hop, link_ok, fwd_prob, self.gcs)

        is_flow = self.online.copy()
        is_flow[self.gcs] = False
        inflow = upstream_flows(depth, next_hop, is_flow)
        n_cut = int((adj & cut).sum() // 2)
        return Network(adj, usable, depth, next_hop, p, inflow, fwd_prob, extra, n_cut)

    # ------------------------------------------------------------------ evidence
    def _update_evidence(self, net: Network) -> None:
        """Noisy local telemetry: watchdog overhearing, delivery feedback, host anomaly score."""
        s, n, a = self.spec, self.n, self.cfg.ema_alpha
        rng = self._rng_obs
        pps = s.packets_per_step

        # Watchdog: neighbours overhear whether a relay retransmits what it received.
        relayed = np.rint(pps * net.inflow[:n]).astype(np.int64)  # packets each drone relays
        obs_i, obs_j = np.nonzero(net.adj[:n, :n] & (relayed[None, :] > 0))
        if obs_i.size:
            p_see = net.fwd_prob[obs_j] * (1.0 - s.base_loss) * (1.0 - net.extra_loss[obs_i])
            seen = rng.binomial(relayed[obs_j], np.clip(p_see, 0.0, 1.0))
            ratio = seen / relayed[obs_j]
            self.ema_f[obs_i, obs_j] = (1.0 - a) * self.ema_f[obs_i, obs_j] + a * ratio

        # End-to-end delivery feedback for the drone's own flow (ACKs from the GCS).
        acked = rng.binomial(pps, np.clip(net.p_deliver[:n], 0.0, 1.0)) / pps
        self.ema_pdr = (1.0 - a) * self.ema_pdr + a * acked

        # Host-based anomaly score: weak signal while dormant, strong while misbehaving.
        mu = np.where(self.behaving[:n], 1.0, np.where(self.comp[:n] & self.online[:n], 0.25, 0.0))
        z = s.detector_dprime * mu + rng.standard_normal(n)
        self.anom = np.tanh(z / 2.0)

    def _compute_slots(self, net: Network) -> None:
        n, k = self.n, self.cfg.k_neighbors
        cand = net.adj[:n, :n]
        dd = np.where(cand, self._dist[:n, :n], np.inf)
        order = np.argsort(dd, axis=1, kind="stable")[:, :k]
        valid = np.take_along_axis(cand, order, axis=1)
        slots = np.where(valid, order, -1)
        if slots.shape[1] < k:  # tiny swarms: pad with empty slots
            pad = np.full((n, k - slots.shape[1]), -1, dtype=np.int64)
            slots = np.concatenate([slots, pad], axis=1)
        self.slots = slots.astype(np.int64)

    def _obs_vector(self, now: int) -> np.ndarray:
        n, k = self.n, self.cfg.k_neighbors
        obs = np.zeros((n, self.cfg.obs_dim), dtype=np.float32)
        on = self.online[:n]
        obs[:, 0] = np.where(on, self.anom, 0.0)
        obs[:, 1] = np.where(on, self.ema_pdr, 0.0)
        obs[:, 2] = now / self.spec.horizon
        rows = np.arange(n)[:, None]
        present = (self.slots >= 0) & on[:, None]
        tgt = np.where(self.slots >= 0, self.slots, 0)
        slot_feats = obs[:, 3:].reshape(n, k, 4)
        slot_feats[:, :, 0] = present
        slot_feats[:, :, 1] = present * self._dist[rows, tgt] / self.spec.comm_range
        slot_feats[:, :, 2] = present * self.ema_f[rows, tgt]
        slot_feats[:, :, 3] = present * (self.block_until[rows, tgt] > now)
        return obs

    # ------------------------------------------------------------------ reward and metrics
    def _reward_and_info(self, c: dict[str, int], now: int) -> tuple[float, dict[str, Any]]:
        cfg, n = self.cfg, self.n
        on = self.online[:n]
        benign = (~self.comp[:n]) & on
        p = self.net.p_deliver[:n]
        availability = float((p * benign).sum() / n)
        pdr = float(p[benign].mean()) if benign.any() else 0.0
        has_link = self.net.usable[:n].any(axis=1)
        threat = float((self.behaving[:n] & has_link).sum() / n)
        n_valid_actions = c["tp_blocks"] + c["fp_blocks"] + c["restores"]
        reward = (
            availability
            - cfg.w_threat * threat
            - cfg.w_action * n_valid_actions / n
            - cfg.w_false_block * c["fp_blocks"] / n
        )
        # The team reward is exactly the mean of these per-drone terms. Learners may mix them
        # in for credit assignment; evaluation always uses the team reward.
        self.local_reward = (
            p * benign
            - cfg.w_threat * (self.behaving[:n] & has_link)
            - cfg.w_action * self._acted
            - cfg.w_false_block * self._false_blocked
        )
        info = {
            "t": now,
            "availability": availability,
            "pdr": pdr,
            "threat": threat,
            "n_compromised": int(self.comp[:n].sum()),
            "n_misbehaving": int(self.behaving[:n].sum()),
            "n_offline": int((~on).sum()),
            "n_cut_links": self.net.n_cut_links,
            "actions": n_valid_actions,
            **c,
        }
        return float(reward), info

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict[str, Any]:
        """Copy of the privileged world state, for plots, tests and oracle baselines."""
        now = self.t
        blk = (self.block_until > now) & self.net.adj
        return {
            "t": now,
            "pos": self.pos.copy(),
            "compromised": self.comp.copy(),
            "misbehaving": self.behaving.copy(),
            "attack": self.atk.copy(),
            "offline": ~self.online,
            "usable": self.net.usable.copy(),
            "blocked": (blk | blk.T),
            "next_hop": self.net.next_hop.copy(),
            "p_deliver": self.net.p_deliver.copy(),
            "gcs": self.gcs,
        }
