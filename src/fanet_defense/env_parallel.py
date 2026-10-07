"""PettingZoo ``ParallelEnv`` adapter: one local blue agent per drone, shared team reward."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import BENCHMARK_VERSION, EnvConfig
from .scenario import ScenarioSampler, ScenarioSpec
from .sim import FanetSim


class FanetParallelEnv(ParallelEnv):
    """Decentralised defense of a drone swarm.

    Agents are ``drone_0 .. drone_{n-1}``. Every agent has the same observation and action
    spaces, so one policy can be shared (parameter sharing) across agents and across
    scenarios with different numbers of drones.

    Pass ``scenario`` for a fixed scenario, or a ``sampler`` to draw a fresh one at every
    ``reset``. ``reset(options={"scenario": spec})`` overrides both for one episode.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": BENCHMARK_VERSION,
        "render_modes": ["rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        scenario: ScenarioSpec | None = None,
        sampler: ScenarioSampler | None = None,
        config: EnvConfig | None = None,
        max_drones: int = 24,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        self._fixed = scenario
        self._sampler = sampler or ScenarioSampler()
        # With a fixed scenario the set of possible agents is exactly its drones.
        self.max_drones = scenario.n_drones if scenario is not None else max_drones
        self.render_mode = render_mode
        self.possible_agents = [f"drone_{i}" for i in range(max_drones)]
        self.agents: list[str] = []
        self.sim: FanetSim | None = None
        self._rng: np.random.Generator | None = None
        self._obs_space = spaces.Box(-1.0, 1.0, shape=(self.cfg.obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(self.cfg.n_actions)

    # ---- spaces -----------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    # ---- API --------------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        spec = (options or {}).get("scenario") or self._fixed
        if spec is None:
            spec = self._sampler.sample(int(self._rng.integers(2**62)))
        if spec.n_drones > self.max_drones:
            raise ValueError(f"scenario has {spec.n_drones} drones; max_drones={self.max_drones}")
        self.sim = FanetSim(spec, self.cfg)
        self.agents = self.possible_agents[: spec.n_drones]
        obs = {a: self.sim.obs[i] for i, a in enumerate(self.agents)}
        return obs, {a: {} for a in self.agents}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self.sim is None or not self.agents:
            raise RuntimeError("call reset() before step()")
        agents = list(self.agents)
        arr = np.fromiter(
            (int(actions.get(a, 0)) for a in agents), dtype=np.int64, count=len(agents)
        )
        obs, reward, done, info = self.sim.step(arr)
        obs_d = {a: obs[i] for i, a in enumerate(agents)}
        rew_d = dict.fromkeys(agents, reward)
        term_d = dict.fromkeys(agents, False)
        trunc_d = dict.fromkeys(agents, done)
        info_d = dict.fromkeys(agents, info)
        if done:
            self.agents = []
        return obs_d, rew_d, term_d, trunc_d, info_d

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array" or self.sim is None:
            return None
        from .viz import render_frame

        return render_frame(self.sim.snapshot(), self.sim.spec)

    def state(self) -> np.ndarray:
        """Privileged global state (positions, compromise flags) for centralised critics."""
        if self.sim is None:
            raise RuntimeError("call reset() first")
        snap = self.sim.snapshot()
        n = self.sim.n
        return np.concatenate(
            [snap["pos"][:n].ravel() / self.sim.spec.arena, snap["compromised"][:n].astype(float)]
        ).astype(np.float32)
