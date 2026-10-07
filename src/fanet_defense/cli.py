"""Command line interface: ``fanet-defense <command> --help``.

Commands
--------
generate-suite  write scenario specs (JSONL), optionally one shard of a larger suite
evaluate        run a policy on a suite and print metrics with 95% confidence intervals
dataset         write prompt / answer / reward records (text interface) for LLM training
train           train PPO, Q-learning or SARSA and save the policy
demo            render one scenario snapshot to PNG
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .evaluate import evaluate, summarize
from .policies import NoOpPolicy, OraclePolicy, Policy, RandomPolicy, WatchdogPolicy
from .scenario import ScenarioSampler, ScenarioSpec, iter_suite


def _sampler(args: argparse.Namespace) -> ScenarioSampler:
    return ScenarioSampler(difficulty=args.difficulty)


def make_policy(name: str, cfg: EnvConfig) -> Policy:
    """``noop``, ``random``, ``watchdog``, ``oracle``, ``ppo:<file.npz>`` or ``q:<file.npz>``."""
    if name == "noop":
        return NoOpPolicy()
    if name == "random":
        return RandomPolicy(cfg)
    if name == "watchdog":
        return WatchdogPolicy(cfg)
    if name == "oracle":
        return OraclePolicy(cfg)
    kind, _, path = name.partition(":")
    if kind == "ppo" and path:
        from .rollout import NumpyPPOPolicy

        return NumpyPPOPolicy.load(path, cfg)
    if kind in {"q", "sarsa"} and path:
        from .tabular import TabularPolicy

        with np.load(path) as f:
            return TabularPolicy(f["q"], cfg, name=kind)
    raise SystemExit(f"unknown policy {name!r}")


def _suite(args: argparse.Namespace) -> list[ScenarioSpec]:
    if args.suite_file:
        lines = Path(args.suite_file).read_text(encoding="utf-8").splitlines()
        return [ScenarioSpec.from_dict(json.loads(line)) for line in lines if line.strip()]
    return list(
        iter_suite(
            args.n, args.master_seed, args.split, _sampler(args), args.shard, args.num_shards
        )
    )


def cmd_generate_suite(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for spec in _suite(args):
            f.write(json.dumps({**spec.to_dict(), "scenario_id": spec.scenario_id}) + "\n")
            count += 1
    print(f"wrote {count} scenarios to {out} in {time.perf_counter() - t0:.1f}s")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = EnvConfig()
    specs = _suite(args)
    policy = make_policy(args.policy, cfg)
    t0 = time.perf_counter()
    results = evaluate(policy, specs, cfg, workers=args.workers)
    summary = summarize(results)
    print(f"{args.policy} on {len(specs)} scenarios ({time.perf_counter() - t0:.0f}s)")
    for metric, (m, lo, hi) in summary.items():
        print(f"  {metric:15s} {m:10.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": args.policy,
            "summary": {k: list(v) for k, v in summary.items()},
            "episodes": [r.to_dict() for r in results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .scenario import generate_suite

    cfg = EnvConfig()
    sampler = _sampler(args)
    val = generate_suite(args.val_n, args.master_seed, "val", sampler)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(row: dict[str, float]) -> None:
        print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    if args.algo == "ppo":
        from .ppo import PPOConfig, train_ppo

        pcfg = PPOConfig(total_env_steps=args.steps, workers=args.workers)
        policy, curve = train_ppo(pcfg, cfg, sampler, args.seed, val, args.master_seed, log=log)
        policy.save(out / "policy.npz")
    else:
        from .tabular import TabularConfig, train_tabular

        def evaluator(p: Policy) -> float:
            return float(np.mean([r.ret for r in evaluate(p, val, cfg)]))

        tcfg = TabularConfig(algo=args.algo, total_env_steps=args.steps)
        tab, curve = train_tabular(
            tcfg, cfg, sampler, args.seed, args.master_seed, evaluator=evaluator, log=log
        )
        np.savez_compressed(out / "policy.npz", q=tab.q)
    (out / "curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")
    print(f"saved {out / 'policy.npz'}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .sim import FanetSim
    from .viz import render_frame

    cfg = EnvConfig()
    spec = _sampler(args).sample(args.seed)
    policy = make_policy(args.policy, cfg)
    sim = FanetSim(spec, cfg)
    policy.reset(spec)
    obs = sim.obs
    while sim.t < args.step and not sim.done:
        obs, *_ = sim.step(policy.act(obs, sim if policy.privileged else None))
    image = render_frame(sim.snapshot(), spec)
    import matplotlib.pyplot as plt

    plt.imsave(args.out, image)
    print(f"saved {args.out} (scenario {spec.scenario_id}, t={sim.t})")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import iter_records, write_jsonl

    cfg = EnvConfig()
    teacher = make_policy(args.teacher, cfg)
    t0 = time.perf_counter()
    count = write_jsonl(
        iter_records(_suite(args), teacher, cfg, decision_interval=args.interval), args.out
    )
    print(f"wrote {count} records to {args.out} in {time.perf_counter() - t0:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fanet-defense", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def suite_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--n", type=int, default=200, help="number of scenarios in the suite")
        sp.add_argument("--split", choices=["train", "val", "test"], default="test")
        sp.add_argument("--master-seed", type=int, default=0)
        sp.add_argument(
            "--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed"
        )
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)
        sp.add_argument("--suite-file", default=None, help="JSONL of specs (overrides --n/--split)")

    g = sub.add_parser("generate-suite", help="write scenario specs as JSONL")
    suite_args(g)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate_suite)

    e = sub.add_parser("evaluate", help="evaluate a policy on a suite")
    suite_args(e)
    e.add_argument("--policy", default="watchdog")
    e.add_argument("--workers", type=int, default=1)
    e.add_argument("--out", default=None, help="write summary and per-episode results as JSON")
    e.set_defaults(func=cmd_evaluate)

    ds = sub.add_parser("dataset", help="write prompt/answer/reward records for LLM training")
    suite_args(ds)
    ds.add_argument("--teacher", default="watchdog", help="policy that writes the answers")
    ds.add_argument("--interval", type=int, default=5, help="seconds between two decisions")
    ds.add_argument("--out", required=True)
    ds.set_defaults(func=cmd_dataset)

    t = sub.add_parser("train", help="train a policy")
    t.add_argument("--algo", choices=["ppo", "q", "sarsa"], default="ppo")
    t.add_argument("--steps", type=int, default=300_000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--master-seed", type=int, default=0)
    t.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    t.add_argument("--val-n", type=int, default=32)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)

    d = sub.add_parser("demo", help="render a scenario snapshot")
    d.add_argument("--policy", default="watchdog")
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--step", type=int, default=60)
    d.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    d.add_argument("--out", default="snapshot.png")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
