"""Torch-free policy forward pass and episode rollouts (runs inside worker processes).

Keeping this module free of torch means rollout workers start in milliseconds, evaluation
of a saved checkpoint needs only numpy, and checkpoints are plain ``.npz`` files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import EnvConfig
from .policies import Policy
from .scenario import ScenarioSpec
from .sim import FanetSim
from .tabular import action_mask

Params = dict[str, np.ndarray]


def _mlp(params: Params, prefix: str, x: np.ndarray) -> np.ndarray:
    h = np.tanh(x @ params[f"{prefix}0.w"] + params[f"{prefix}0.b"])
    h = np.tanh(h @ params[f"{prefix}1.w"] + params[f"{prefix}1.b"])
    return h @ params[f"{prefix}2.w"] + params[f"{prefix}2.b"]


def actor_logits(params: Params, obs: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    logits = _mlp(params, "a", obs.astype(np.float64))
    return np.where(action_mask(obs, cfg), logits, -1e9)


def critic_value(params: Params, obs: np.ndarray) -> np.ndarray:
    return _mlp(params, "c", obs.astype(np.float64))[:, 0] * float(params["vscale"])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


@dataclass
class NumpyPPOPolicy(Policy):
    """Actor of a trained PPO agent, greedy by default, evaluated with numpy only."""

    params: Params
    cfg: EnvConfig = field(default_factory=EnvConfig)
    deterministic: bool = True
    seed: int = 0
    name: str = "ppo"
    _rng: np.random.Generator = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def reset(self, spec: ScenarioSpec) -> None:
        self._rng = np.random.default_rng([self.seed, spec.seed])

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        logits = actor_logits(self.params, obs, self.cfg)
        if self.deterministic:
            return logits.argmax(axis=1).astype(np.int64)
        p = softmax(logits)
        u = self._rng.random((obs.shape[0], 1))
        return (np.cumsum(p, axis=1) > u).argmax(axis=1).astype(np.int64)

    def save(self, path: str | Path) -> None:
        # numpy's stubs type **kwargs of savez_compressed too narrowly; arrays are what it takes.
        np.savez_compressed(path, **self.params)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path, cfg: EnvConfig | None = None, **kw: Any) -> NumpyPPOPolicy:
        with np.load(path) as f:
            params = {k: f[k] for k in f.files}
        return cls(params, cfg or EnvConfig(), **kw)


def rollout_episode(args: tuple[Params, dict[str, Any], EnvConfig, int]) -> dict[str, Any]:
    """Sample one full episode from the stochastic policy. Returns trajectory arrays."""
    params, spec_dict, cfg, seed = args
    sim = FanetSim(ScenarioSpec.from_dict(spec_dict), cfg)
    rng = np.random.default_rng(seed)
    n = sim.n
    obs_l, act_l, logp_l, val_l, rew_l, loc_l = [], [], [], [], [], []
    obs = sim.obs
    rows = np.arange(n)
    while not sim.done:
        p = softmax(actor_logits(params, obs, cfg))
        u = rng.random((n, 1))
        act = (np.cumsum(p, axis=1) > u).argmax(axis=1)
        obs_l.append(obs)
        act_l.append(act)
        logp_l.append(np.log(p[rows, act] + 1e-12))
        val_l.append(critic_value(params, obs))
        obs, reward, _, _ = sim.step(act)
        rew_l.append(reward)
        loc_l.append(sim.local_reward.copy())
    return {
        "obs": np.stack(obs_l).astype(np.float32),
        "act": np.stack(act_l).astype(np.int64),
        "logp": np.stack(logp_l).astype(np.float32),
        "val": np.stack(val_l).astype(np.float32),
        "rew": np.asarray(rew_l, dtype=np.float32),
        "rew_local": np.stack(loc_l).astype(np.float32),
        "ret": float(np.sum(rew_l)),
        "n": n,
    }
