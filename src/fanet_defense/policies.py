"""Reference policies: no-op, random, watchdog heuristic and a privileged oracle.

All policies share one interface so the evaluation harness treats learned and scripted
defenders identically: ``reset(spec)`` once per episode, then ``act(obs, sim)`` per step
where ``obs`` has shape ``(n_drones, obs_dim)`` and the result has shape ``(n_drones,)``.
Only policies with ``privileged = True`` may read ``sim`` (hidden compromise state); the
harness hands ``sim=None`` to everyone else, so leakage is impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .scenario import ScenarioSpec
from .sim import FanetSim


class Policy:
    """Base class. Subclasses override :meth:`act`."""

    name = "policy"
    privileged = False

    def reset(self, spec: ScenarioSpec) -> None:
        """Called once at the start of each episode."""

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NoOpPolicy(Policy):
    name: str = "noop"

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        return np.zeros(obs.shape[0], dtype=np.int64)


@dataclass
class RandomPolicy(Policy):
    """Uniform over all actions (including invalid block slots, which are ignored)."""

    cfg: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 0
    name: str = "random"
    _rng: np.random.Generator = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def reset(self, spec: ScenarioSpec) -> None:
        self._rng = np.random.default_rng([self.seed, spec.seed])

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        return np.asarray(
            self._rng.integers(0, self.cfg.n_actions, size=obs.shape[0]), dtype=np.int64
        )


@dataclass
class WatchdogPolicy(Policy):
    """Classic watchdog / pathrater logic expressed on the local observation.

    * restore this drone when its host anomaly score is high *and* its own delivery is bad;
    * otherwise block the neighbour whose overheard forwarding ratio is lowest, if it is
      below ``th_fwd``.
    """

    cfg: EnvConfig = field(default_factory=EnvConfig)
    th_fwd: float = 0.5
    th_anom: float = 0.6
    th_pdr: float = 0.8
    name: str = "watchdog"

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        n, k = obs.shape[0], self.cfg.k_neighbors
        slots = obs[:, 3:].reshape(n, k, 4)
        present, fwd, blocked = slots[..., 0] > 0.5, slots[..., 2], slots[..., 3] > 0.5
        cand = present & ~blocked & (fwd < self.th_fwd)
        worst = np.where(cand, fwd, np.inf).argmin(axis=1)
        restore = (obs[:, 0] > self.th_anom) & (obs[:, 1] < self.th_pdr)
        return np.where(restore, self.cfg.restore_action, np.where(cand.any(axis=1), worst + 1, 0))


@dataclass
class OraclePolicy(Policy):
    """Privileged upper bound: restores every compromised drone, dormant or awake.

    It reads the hidden state, so it is not a defender anyone could deploy; it only bounds
    the achievable return and sanity-checks the environment.
    """

    cfg: EnvConfig = field(default_factory=EnvConfig)
    name: str = "oracle"
    privileged: bool = True

    def act(self, obs: np.ndarray, sim: FanetSim | None = None) -> np.ndarray:
        if sim is None:
            raise ValueError("OraclePolicy needs the simulator")
        n = obs.shape[0]
        need = sim.comp[:n] & sim.online[:n]
        return np.where(need, self.cfg.restore_action, 0).astype(np.int64)


def tune_watchdog(
    specs: list[ScenarioSpec],
    cfg: EnvConfig,
    n_trials: int = 40,
    seed: int = 0,
) -> WatchdogPolicy:
    """Random search over the three thresholds on *validation* scenarios (never test)."""
    from .evaluate import evaluate  # local import: evaluate depends on this module

    rng = np.random.default_rng(seed)
    best, best_ret = WatchdogPolicy(cfg), -np.inf
    for _ in range(n_trials):
        cand = WatchdogPolicy(
            cfg,
            th_fwd=float(rng.uniform(0.3, 0.8)),
            th_anom=float(rng.uniform(0.2, 0.8)),
            th_pdr=float(rng.uniform(0.5, 1.0)),
            name="watchdog-tuned",
        )
        ret = float(np.mean([r.ret for r in evaluate(cand, specs, cfg)]))
        if ret > best_ret:
            best, best_ret = cand, ret
    return best
