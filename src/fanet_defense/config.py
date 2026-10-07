"""Interface and reward constants shared by every adapter (PettingZoo, Gymnasium, text).

Unlike :class:`~fanet_defense.scenario.ScenarioSpec`, these do not vary per scenario: they
define the benchmark itself (what an action means, what the reward is). Changing them
creates a different benchmark version, so keep them fixed when comparing policies.
"""

from __future__ import annotations

from dataclasses import dataclass

BENCHMARK_VERSION = "fanet_defense_v0"


@dataclass(frozen=True)
class EnvConfig:
    # Interface -------------------------------------------------------------------------
    k_neighbors: int = 4  # neighbour slots in a local observation (and block-actions)
    # Defensive actions -----------------------------------------------------------------
    block_ttl: int = 25  # steps a block lasts before it silently expires
    restore_steps: int = 20  # steps a drone is offline while it is re-flashed
    # Evidence model --------------------------------------------------------------------
    ema_alpha: float = 0.3  # smoothing of watchdog / delivery evidence
    # Reward = availability - w_threat*threat - w_action*actions/n - w_false_block*false_blocks/n
    # where availability is the mean, over ALL drones, of [benign and online] x P(delivery).
    # Compromised and offline drones therefore count as lost service, which is what makes
    # "restore everyone all the time" and "do nothing" both lose.
    w_threat: float = 0.25
    w_action: float = 0.02
    w_false_block: float = 0.25

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.block_ttl < 1 or self.restore_steps < 1:
            raise ValueError("block_ttl and restore_steps must be >= 1")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

    @property
    def n_actions(self) -> int:
        """0 = no-op, 1..K = block neighbour slot k, K+1 = restore (re-flash) this drone."""
        return self.k_neighbors + 2

    @property
    def restore_action(self) -> int:
        return self.k_neighbors + 1

    @property
    def obs_dim(self) -> int:
        """[anomaly, own delivery, time] + K x [present, distance, watchdog ratio, blocked]."""
        return 3 + 4 * self.k_neighbors
