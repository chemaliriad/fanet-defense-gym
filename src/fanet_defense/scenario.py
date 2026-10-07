"""Scenario specification and procedural generation.

A :class:`ScenarioSpec` holds everything that makes one episode different from another:
the world (size, mobility, link quality) and the adversary (how many implants, what they
do, how fast they spread). It is a frozen, JSON-serialisable dataclass with a stable
content hash, so a scenario can be stored, shipped to a worker and replayed bit-for-bit.

:class:`ScenarioSampler` draws specs from difficulty profiles. ``generate_suite`` turns a
master seed into thousands of distinct scenarios, split into disjoint ``train`` / ``val`` /
``test`` families by construction (different seed namespaces), never by slicing one trace.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

import numpy as np

ATTACK_TYPES: tuple[str, ...] = ("blackhole", "greyhole", "flood")
Difficulty = Literal["easy", "medium", "hard", "mixed"]
Split = Literal["train", "val", "test"]
_SPLIT_ID = {"train": 0, "val": 1, "test": 2}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScenarioSpec:
    """One fully specified episode (world + adversary). Units: metres, seconds (1 step = 1 s)."""

    seed: int
    n_drones: int = 12
    arena: float = 1000.0
    comm_range: float = 380.0
    speed_min: float = 4.0
    speed_max: float = 12.0
    horizon: int = 200
    base_loss: float = 0.03
    packets_per_step: int = 6
    # Adversary: supply-chain implants that wake up later, then attack and spread laterally.
    n_implants: int = 2
    wake_min: int = 5
    wake_max: int = 40
    attack_probs: tuple[float, float, float] = (0.5, 0.3, 0.2)  # blackhole, greyhole, flood
    greyhole_drop: float = 0.6
    stealth: float = 0.0
    exploit_prob: float = 0.01
    flood_loss: float = 0.35
    # Observation quality of the host-based detector (separation of its score, in sigmas).
    detector_dprime: float = 1.5
    # Benign confounders: a share of healthy drones relays unreliably (battery, saturation),
    # so a low watchdog ratio is not proof of compromise.
    flaky_frac: float = 0.15
    flaky_fwd: float = 0.8

    def __post_init__(self) -> None:
        _check(self.n_drones >= 3, "n_drones must be >= 3")
        _check(self.arena > 0 and self.comm_range > 0, "arena and comm_range must be > 0")
        _check(0 < self.speed_min <= self.speed_max, "need 0 < speed_min <= speed_max")
        _check(self.horizon >= 1, "horizon must be >= 1")
        _check(0.0 <= self.base_loss < 1.0, "base_loss must be in [0, 1)")
        _check(self.packets_per_step >= 1, "packets_per_step must be >= 1")
        _check(0 <= self.n_implants <= self.n_drones, "n_implants must be in [0, n_drones]")
        _check(1 <= self.wake_min <= self.wake_max, "need 1 <= wake_min <= wake_max")
        _check(
            len(self.attack_probs) == len(ATTACK_TYPES)
            and all(p >= 0 for p in self.attack_probs)
            and math.isclose(sum(self.attack_probs), 1.0, abs_tol=1e-6),
            "attack_probs must be non-negative and sum to 1",
        )
        for name in ("greyhole_drop", "stealth", "exploit_prob", "flood_loss"):
            _check(0.0 <= getattr(self, name) < 1.0, f"{name} must be in [0, 1)")
        _check(self.detector_dprime >= 0.0, "detector_dprime must be >= 0")
        _check(0.0 <= self.flaky_frac <= 1.0, "flaky_frac must be in [0, 1]")
        _check(0.0 < self.flaky_fwd <= 1.0, "flaky_fwd must be in (0, 1]")

    # -- serialisation -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["attack_probs"] = [float(p) for p in self.attack_probs]
        d["schema"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScenarioSpec:
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "attack_probs" in kwargs:
            kwargs["attack_probs"] = tuple(float(p) for p in kwargs["attack_probs"])
        return cls(**kwargs)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def scenario_id(self) -> str:
        """Stable 12-hex-digit content hash (identical specs always share an id)."""
        canon = json.dumps(_round_floats(self.to_dict()), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon.encode()).hexdigest()[:12]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


# --------------------------------------------------------------------------------------
# Difficulty profiles: (low, high) ranges the sampler draws uniformly from.
# --------------------------------------------------------------------------------------
_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "easy": {
        "implants_frac": (0.05, 0.15),
        "exploit_prob": (0.000, 0.004),
        "stealth": (0.0, 0.0),
        "dprime": (2.0, 3.0),
        "base_loss": (0.00, 0.03),
        "wake": (5, 20),
        "flaky_frac": (0.00, 0.10),
    },
    "medium": {
        "implants_frac": (0.10, 0.25),
        "exploit_prob": (0.004, 0.012),
        "stealth": (0.0, 0.20),
        "dprime": (1.2, 2.2),
        "base_loss": (0.01, 0.06),
        "wake": (5, 35),
        "flaky_frac": (0.10, 0.25),
    },
    "hard": {
        "implants_frac": (0.15, 0.30),
        "exploit_prob": (0.010, 0.030),
        "stealth": (0.10, 0.40),
        "dprime": (0.6, 1.6),
        "base_loss": (0.02, 0.08),
        "wake": (3, 45),
        "flaky_frac": (0.20, 0.40),
    },
}


@dataclass(frozen=True)
class ScenarioSampler:
    """Draw scenarios from a difficulty profile. Pure function of ``seed``."""

    difficulty: Difficulty = "mixed"
    n_range: tuple[int, int] = (8, 20)
    degree_range: tuple[float, float] = (4.0, 7.0)
    horizon: int = 200
    attack_concentration: tuple[float, float, float] = (2.0, 1.5, 1.0)
    # Field floors applied after sampling, e.g. (("stealth", 0.5),) to build a family that is
    # harder than anything seen in training (out-of-distribution evaluation).
    floors: tuple[tuple[str, float], ...] = ()

    def sample(self, seed: int) -> ScenarioSpec:
        spec = self._sample(seed)
        if self.floors:
            spec = replace(
                spec, **{name: max(getattr(spec, name), value) for name, value in self.floors}
            )
        return spec

    def _sample(self, seed: int) -> ScenarioSpec:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 7919]))
        diff = self.difficulty
        if diff == "mixed":
            diff = ("easy", "medium", "hard")[int(rng.integers(3))]
        prof = _PROFILES[diff]

        n = int(rng.integers(self.n_range[0], self.n_range[1] + 1))
        arena = 1000.0
        # Choose the radio range from a target mean degree on a uniform deployment.
        degree = float(rng.uniform(*self.degree_range))
        comm_range = min(0.6 * arena, arena * math.sqrt(degree / (math.pi * (n - 1))))
        implants_frac = float(rng.uniform(*prof["implants_frac"]))
        n_implants = int(np.clip(round(implants_frac * n), 1, max(1, n // 3)))
        wake_lo, wake_hi = (int(v) for v in prof["wake"])
        wake_min = int(rng.integers(wake_lo, wake_lo + 5))
        wake_max = max(wake_min, int(rng.integers(wake_hi - 4, wake_hi + 1)))
        probs = rng.dirichlet(np.asarray(self.attack_concentration))
        speed_max = float(rng.uniform(8.0, 16.0))
        speed_min = float(rng.uniform(2.0, 6.0))

        return ScenarioSpec(
            seed=int(seed),
            n_drones=n,
            arena=arena,
            comm_range=float(comm_range),
            speed_min=min(speed_min, speed_max),
            speed_max=speed_max,
            horizon=self.horizon,
            base_loss=float(rng.uniform(*prof["base_loss"])),
            packets_per_step=int(rng.integers(4, 9)),
            n_implants=n_implants,
            wake_min=wake_min,
            wake_max=wake_max,
            attack_probs=tuple(float(p) for p in probs),  # type: ignore[arg-type]
            greyhole_drop=float(rng.uniform(0.4, 0.8)),
            stealth=float(rng.uniform(*prof["stealth"])),
            exploit_prob=float(rng.uniform(*prof["exploit_prob"])),
            flood_loss=float(rng.uniform(0.2, 0.5)),
            detector_dprime=float(rng.uniform(*prof["dprime"])),
            flaky_frac=float(rng.uniform(*prof["flaky_frac"])),
            flaky_fwd=float(rng.uniform(0.65, 0.9)),
        )


def split_seeds(master_seed: int, split: Split, n: int) -> list[int]:
    """``n`` distinct scenario seeds for a split. Different splits never share a seed stream."""
    ss = np.random.SeedSequence([int(master_seed), _SPLIT_ID[split]])
    # 62-bit seeds: collisions are astronomically unlikely, and we verify below anyway.
    raw = np.random.default_rng(ss).integers(0, 2**62, size=n, dtype=np.int64)
    seeds = [int(s) for s in raw]
    if len(set(seeds)) != len(seeds):  # pragma: no cover - probability ~ n^2 / 2^63
        raise RuntimeError("seed collision; choose another master seed")
    return seeds


def generate_suite(
    n: int,
    master_seed: int = 0,
    split: Split = "test",
    sampler: ScenarioSampler | None = None,
) -> list[ScenarioSpec]:
    """A reproducible family of ``n`` scenarios for one split."""
    sampler = sampler or ScenarioSampler()
    return [sampler.sample(s) for s in split_seeds(master_seed, split, n)]


def iter_suite(
    n: int,
    master_seed: int = 0,
    split: Split = "test",
    sampler: ScenarioSampler | None = None,
    shard: int = 0,
    num_shards: int = 1,
) -> Iterator[ScenarioSpec]:
    """Stream the scenarios of shard ``shard`` out of ``num_shards`` (round-robin by index).

    Shards are disjoint and their union is exactly :func:`generate_suite`, so workers can
    generate or evaluate their share independently (see ``k8s/``).
    """
    if not 0 <= shard < num_shards:
        raise ValueError("need 0 <= shard < num_shards")
    sampler = sampler or ScenarioSampler()
    seeds = split_seeds(master_seed, split, n)
    for idx in range(shard, n, num_shards):
        yield sampler.sample(seeds[idx])
