"""Evaluation harness: run policies on scenario suites and aggregate with honest uncertainty.

Aggregation is done at the level of independent units (scenarios, or training seeds), never
over steps of the same trajectory. Comparisons between two policies use *paired* scenarios.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .policies import Policy
from .scenario import ScenarioSpec
from .sim import FanetSim

METRICS = (
    "ret",
    "availability",
    "threat",
    "compromised",
    "false_blocks",
    "false_restores",
    "actions",
    "ttc",
    "contained",
)


@dataclass
class EpisodeResult:
    scenario_id: str
    seed: int
    n_drones: int
    ret: float  # undiscounted team return
    availability: float  # mean over steps of the availability term
    threat: float  # mean over steps of the active-threat term
    compromised: float  # mean over steps of the compromised fraction of the swarm
    false_blocks: int
    false_restores: int
    actions: int
    ttc: float  # steps from first active threat to containment (censored at the horizon)
    contained: bool  # whether the threat was ever contained for >= 5 consecutive steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_to_containment(threat: np.ndarray, hold: int = 5) -> tuple[float, bool]:
    active = np.flatnonzero(threat > 0)
    if active.size == 0:
        return 0.0, True  # nothing ever attacked
    start = int(active[0])
    quiet = threat[start:] == 0
    run = 0
    for k, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run >= hold:
            return float(k - hold + 1), True
    return float(threat.size - start), False


def run_episode(policy: Policy, spec: ScenarioSpec, cfg: EnvConfig | None = None) -> EpisodeResult:
    cfg = cfg or EnvConfig()
    sim = FanetSim(spec, cfg)
    policy.reset(spec)
    obs = sim.obs
    ret = 0.0
    avail, threat, comp = [], [], []
    fb = fr = acts = 0
    while not sim.done:
        actions = policy.act(obs, sim if policy.privileged else None)
        obs, reward, _, info = sim.step(actions)
        ret += reward
        avail.append(info["availability"])
        threat.append(info["threat"])
        comp.append(info["n_compromised"] / sim.n)
        fb += info["fp_blocks"]
        fr += info["fp_restores"]
        acts += info["actions"]
    ttc, contained = _time_to_containment(np.asarray(threat))
    return EpisodeResult(
        scenario_id=spec.scenario_id,
        seed=spec.seed,
        n_drones=spec.n_drones,
        ret=float(ret),
        availability=float(np.mean(avail)),
        threat=float(np.mean(threat)),
        compromised=float(np.mean(comp)),
        false_blocks=fb,
        false_restores=fr,
        actions=acts,
        ttc=ttc,
        contained=contained,
    )


def _run_one(args: tuple[Policy, ScenarioSpec, EnvConfig]) -> EpisodeResult:
    policy, spec, cfg = args
    return run_episode(policy, spec, cfg)


def evaluate(
    policy: Policy,
    specs: Sequence[ScenarioSpec],
    cfg: EnvConfig | None = None,
    workers: int = 1,
    executor: Executor | None = None,
) -> list[EpisodeResult]:
    """One episode per scenario, in scenario order (so results of two policies are paired).

    Pass ``executor`` to reuse a long-lived process pool (e.g. during training) instead of
    paying process start-up on every call.
    """
    cfg = cfg or EnvConfig()
    jobs = [(policy, s, cfg) for s in specs]
    if executor is not None:
        return list(executor.map(_run_one, jobs, chunksize=2))
    if workers <= 1 or len(specs) < 2:
        return [run_episode(policy, s, cfg) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs, chunksize=4))


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(estimate, low, high) with a percentile bootstrap over independent units."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no values")
    est = float(stat(x))
    if x.size == 1:
        return est, est, est
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def summarize(
    results: Sequence[EpisodeResult], seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """Mean and 95% bootstrap CI of every metric over scenarios."""
    out = {}
    for m in METRICS:
        vals = [float(getattr(r, m)) for r in results]
        out[m] = bootstrap_ci(vals, seed=seed)
    return out


def paired_difference(
    a: Sequence[EpisodeResult],
    b: Sequence[EpisodeResult],
    metric: str = "ret",
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of ``a - b`` over the same scenarios, with a bootstrap CI."""
    if [r.scenario_id for r in a] != [r.scenario_id for r in b]:
        raise ValueError("results must be on the same scenarios, in the same order")
    diffs = [
        float(getattr(x, metric)) - float(getattr(y, metric)) for x, y in zip(a, b, strict=True)
    ]
    return bootstrap_ci(diffs, seed=seed)


def seed_interval(per_seed_means: Sequence[float]) -> tuple[float, float, float]:
    """Mean and 95% t-interval across independent training seeds (small n)."""
    x = np.asarray(per_seed_means, dtype=np.float64)
    m = float(x.mean())
    if x.size < 2:
        return m, m, m
    # two-sided 95% Student t critical values for df = 1..30, then ~normal
    t_crit = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(x.size - 1, 2.0)
    half = t_crit * float(x.std(ddof=1)) / np.sqrt(x.size)
    return m, m - half, m + half
