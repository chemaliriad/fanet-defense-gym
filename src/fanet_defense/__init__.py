"""fanet-defense-gym: a procedurally generated cyber-defense environment for UAV ad hoc networks."""

from .config import BENCHMARK_VERSION, EnvConfig
from .scenario import (
    ATTACK_TYPES,
    ScenarioSampler,
    ScenarioSpec,
    generate_suite,
    iter_suite,
    split_seeds,
)
from .sim import FanetSim

__version__ = "0.1.0"

__all__ = [
    "ATTACK_TYPES",
    "BENCHMARK_VERSION",
    "EnvConfig",
    "FanetSim",
    "ScenarioSampler",
    "ScenarioSpec",
    "__version__",
    "generate_suite",
    "iter_suite",
    "split_seeds",
]
