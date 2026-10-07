"""Tabular Q-learning and SARSA with one table shared by every drone (parameter sharing).

The local observation is discretised into 3 x 3 x 4 x K cells: anomaly level, own delivery
level, the level of the least reliable unblocked neighbour (or "none") and *which* slot it
occupies. The tabular agents see exactly the same information as the neural agent, only
coarsened, so the comparison between them is about function approximation and learning
rule, not about privileged inputs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .config import EnvConfig
from .policies import Policy
from .scenario import ScenarioSampler, split_seeds
from .sim import FanetSim

Algo = Literal["q", "sarsa"]
ANOM_BINS = (0.35, 0.70)
PDR_BINS = (0.4, 0.8)
FWD_BINS = (0.4, 0.75)


def n_states(cfg: EnvConfig) -> int:
    return 3 * 3 * 4 * cfg.k_neighbors


def discretize(obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Map ``(n, obs_dim)`` observations to integer cells ``(n,)``."""
    n, k = obs.shape[0], cfg.k_neighbors
    a = np.digitize(obs[:, 0], ANOM_BINS)
    p = np.digitize(obs[:, 1], PDR_BINS)
    slots = obs[:, 3:].reshape(n, k, 4)
    usable = (slots[..., 0] > 0.5) & (slots[..., 3] < 0.5)
    fwd = np.where(usable, slots[..., 2], np.inf)
    worst = fwd.argmin(axis=1)
    fmin = fwd.min(axis=1)
    none = ~np.isfinite(fmin)
    f = np.where(none, 3, np.digitize(np.where(none, 0.0, fmin), FWD_BINS))
    slot = np.where(none, 0, worst)
    return ((a * 3 + p) * 4 + f) * k + slot


def action_mask(obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    """Valid actions: no-op and restore always; block slot k only for a present, unblocked one."""
    n, k = obs.shape[0], cfg.k_neighbors
    slots = obs[:, 3:].reshape(n, k, 4)
    mask = np.ones((n, cfg.n_actions), dtype=bool)
    mask[:, 1 : k + 1] = (slots[..., 0] > 0.5) & (slots[..., 3] < 0.5)
    return mask


@dataclass
class TabularPolicy(Policy):
    """Greedy policy over a learned Q table."""

    q: np.ndarray
    cfg: EnvConfig = field(default_factory=EnvConfig)
    name: str = "q-table"

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        s = discretize(obs, self.cfg)
        q = np.where(action_mask(obs, self.cfg), self.q[s], -np.inf)
        return q.argmax(axis=1).astype(np.int64)


@dataclass
class TabularConfig:
    algo: Algo = "q"
    gamma: float = 0.97
    alpha: float = 0.1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6  # fraction of the budget over which epsilon decays linearly
    total_env_steps: int = 200_000
    eval_every_steps: int = 20_000


def train_tabular(
    tcfg: TabularConfig,
    cfg: EnvConfig,
    sampler: ScenarioSampler,
    seed: int,
    master_seed: int = 0,
    evaluator: Callable[[Policy], float] | None = None,
    log: Callable[[dict[str, float]], None] | None = None,
) -> tuple[TabularPolicy, list[dict[str, float]]]:
    """Train on a stream of fresh training scenarios; return the greedy policy and the curve."""
    rng = np.random.default_rng([seed, 1234])
    q = np.zeros((n_states(cfg), cfg.n_actions))
    n_scen = tcfg.total_env_steps // sampler.horizon + 2
    train_seeds = split_seeds(master_seed * 1000 + seed, "train", n_scen)
    curve: list[dict[str, float]] = []
    steps, next_eval = 0, 0
    ep_returns: list[float] = []
    t0 = time.perf_counter()

    def eps_at(step: int) -> float:
        frac = min(1.0, step / max(1.0, tcfg.eps_decay_frac * tcfg.total_env_steps))
        return tcfg.eps_start + frac * (tcfg.eps_end - tcfg.eps_start)

    def choose(obs: np.ndarray, s: np.ndarray, eps: float) -> np.ndarray:
        mask = action_mask(obs, cfg)
        greedy = np.where(mask, q[s], -np.inf).argmax(axis=1)
        rand = (rng.random((obs.shape[0], cfg.n_actions)) * mask).argmax(axis=1)
        return np.where(rng.random(obs.shape[0]) < eps, rand, greedy)

    for scen_seed in train_seeds:
        if steps >= tcfg.total_env_steps:
            break
        sim = FanetSim(sampler.sample(scen_seed), cfg)
        obs = sim.obs
        s = discretize(obs, cfg)
        a = choose(obs, s, eps_at(steps))
        ep_ret = 0.0
        while not sim.done:
            obs2, r, _, _ = sim.step(a)
            s2 = discretize(obs2, cfg)
            a2 = choose(obs2, s2, eps_at(steps))
            if tcfg.algo == "q":
                nxt = np.where(action_mask(obs2, cfg), q[s2], -np.inf).max(axis=1)
            else:
                nxt = q[s2, a2]
            # Bootstrapping through the time limit: the cell does not encode the clock.
            td = r + tcfg.gamma * nxt - q[s, a]
            np.add.at(q, (s, a), tcfg.alpha * td)
            s, a, obs = s2, a2, obs2
            ep_ret += r
            steps += 1
            if evaluator is not None and steps >= next_eval:
                row = {
                    "env_steps": float(steps),
                    "val_return": float(evaluator(TabularPolicy(q.copy(), cfg, tcfg.algo))),
                    "train_return": float(np.mean(ep_returns[-20:]))
                    if ep_returns
                    else float("nan"),
                    "wall_s": time.perf_counter() - t0,
                }
                curve.append(row)
                if log:
                    log(row)
                next_eval += tcfg.eval_every_steps
        ep_returns.append(ep_ret)
    return TabularPolicy(q, cfg, tcfg.algo), curve
