"""Single-process timings for watchdog-driven swarm steps and scenario sampling."""

import argparse
import platform
import time

from fanet_defense import EnvConfig, ScenarioSampler, generate_suite
from fanet_defense.policies import WatchdogPolicy
from fanet_defense.sim import FanetSim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--specs", type=int, default=8000)
    args = parser.parse_args()
    if args.episodes < 1 or args.specs < 1:
        parser.error("--episodes and --specs must be positive")

    cfg = EnvConfig()
    policy = WatchdogPolicy(cfg)
    suite = generate_suite(args.episodes, 0, "val", ScenarioSampler("mixed"))
    steps = 0
    elapsed = 0.0
    for spec in suite:
        sim = FanetSim(spec, cfg)
        policy.reset(spec)
        start = time.perf_counter()
        while not sim.done:
            sim.step(policy.act(sim.obs, None))
            steps += 1
        elapsed += time.perf_counter() - start

    print(f"Processor: {platform.processor()}")
    print(f"Episodes: {len(suite)}")
    print(f"Swarm steps: {steps}")
    print(f"Mean swarm size: {sum(spec.n_drones for spec in suite) / len(suite):.2f}")
    print(f"{1000 * elapsed / steps:.3f} ms per swarm step")
    print(f"{steps / elapsed:.2f} steps per second")

    start = time.perf_counter()
    specs = generate_suite(args.specs, 0, "train", ScenarioSampler("mixed"))
    sampling_seconds = time.perf_counter() - start
    print(f"Sampled {len(specs)} scenario specs in {sampling_seconds:.6f} seconds")


if __name__ == "__main__":
    main()
