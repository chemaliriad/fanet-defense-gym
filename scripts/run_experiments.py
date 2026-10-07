"""Benchmark protocol: tune, train, evaluate. Writes ``results/`` (JSON + CSV).

    python scripts/run_experiments.py --steps 300000 --seeds 0 1 2 3 4 --workers 5

Protocol (see docs/design.md, "Evaluation protocol"):
* training scenarios are fresh at every episode (seed stream of the ``train`` split);
* the watchdog thresholds and every hyper-parameter are chosen on ``val`` only;
* final numbers come from 200 ``test`` scenarios plus three out-of-distribution families,
  identical for every policy (paired), with 95% intervals (bootstrap over scenarios for
  fixed policies, Student t over training seeds for learned ones);
* every run is kept, including the ones that look bad.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fanet_defense import EnvConfig, ScenarioSampler, generate_suite
from fanet_defense.evaluate import (
    EpisodeResult,
    evaluate,
    paired_difference,
    seed_interval,
    summarize,
)
from fanet_defense.policies import (
    NoOpPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    WatchdogPolicy,
    tune_watchdog,
)

ROOT = Path(__file__).resolve().parents[1]
VAL_MASTER, TEST_MASTER = 1, 2026
SUITES: dict[str, tuple[ScenarioSampler, int]] = {
    "test": (ScenarioSampler("mixed"), 200),
    "ood_large_swarm": (ScenarioSampler("mixed", n_range=(21, 24)), 100),
    "ood_flood_heavy": (ScenarioSampler("mixed", attack_concentration=(0.5, 0.5, 4.0)), 100),
    "ood_stealthy": (ScenarioSampler("hard", floors=(("stealth", 0.5),)), 100),
}
METRICS = (
    "ret",
    "availability",
    "compromised",
    "false_blocks",
    "false_restores",
    "ttc",
    "contained",
)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "uncommitted"
    except OSError:  # pragma: no cover
        return "unknown"


def _train_tabular_job(
    args: tuple[str, int, int],
) -> tuple[str, int, dict[str, Any], list[dict[str, float]]]:
    from fanet_defense.tabular import TabularConfig, train_tabular

    algo, seed, steps = args
    cfg = EnvConfig()
    sampler = ScenarioSampler("mixed")
    val = generate_suite(16, VAL_MASTER, "val", sampler)

    def evaluator(p: Policy) -> float:
        return float(np.mean([r.ret for r in evaluate(p, val, cfg)]))

    tcfg = TabularConfig(algo=algo, total_env_steps=steps, eval_every_steps=max(steps // 10, 1))
    pol, curve = train_tabular(tcfg, cfg, sampler, seed, evaluator=evaluator)
    return algo, seed, {"q": pol.q}, curve


def summarize_fixed(res: list[EpisodeResult]) -> dict[str, list[float]]:
    s = summarize(res)
    return {m: list(s[m]) for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--algos", nargs="+", default=["ppo", "q", "sarsa"])
    args = ap.parse_args()

    out = Path(args.out)
    (out / "policies").mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    t_start = time.perf_counter()
    log = lambda msg: print(f"[{time.perf_counter() - t_start:7.0f}s] {msg}", flush=True)  # noqa: E731

    val = generate_suite(32, VAL_MASTER, "val", ScenarioSampler("mixed"))
    suites = {
        name: generate_suite(n, TEST_MASTER, "test", sampler)
        for name, (sampler, n) in SUITES.items()
    }

    # 1. Strong scripted baseline: watchdog thresholds tuned on validation only.
    tuned = tune_watchdog(val, cfg, n_trials=40, seed=0)
    log(
        f"tuned watchdog on val: fwd={tuned.th_fwd:.3f} anom={tuned.th_anom:.3f} pdr={tuned.th_pdr:.3f}"
    )
    val_reference = {
        name: float(np.mean([r.ret for r in evaluate(pol, val, cfg, workers=args.workers)]))
        for name, pol in (
            ("noop", NoOpPolicy()),
            ("watchdog_tuned", tuned),
            ("oracle", OraclePolicy(cfg)),
        )
    }
    log(f"validation references: {val_reference}")

    # 2. Learned policies.
    learned: dict[str, dict[int, Policy]] = {a: {} for a in args.algos}
    curves: dict[str, dict[int, list[dict[str, float]]]] = {a: {} for a in args.algos}
    ppo_mode = "n/a"
    if "ppo" in args.algos:
        from fanet_defense.ppo import PPOConfig, train_ppo

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for seed in args.seeds:
                pcfg = PPOConfig(total_env_steps=args.steps, workers=args.workers)
                pol, curve = train_ppo(
                    pcfg, cfg, ScenarioSampler("mixed"), seed, val, executor=pool
                )
                pol.save(out / "policies" / f"ppo_seed{seed}.npz")
                learned["ppo"][seed], curves["ppo"][seed] = pol, curve
                log(
                    f"ppo seed {seed}: final val return greedy {curve[-1].get('val_return', float('nan')):.1f}"
                    f" / sampled {curve[-1].get('val_return_sampled', float('nan')):.1f}"
                )
        # Greedy or sampled actions? Decided on validation only, once for all seeds.
        greedy = np.mean([curves["ppo"][s][-1]["val_return"] for s in learned["ppo"]])
        sampled = np.mean(
            [curves["ppo"][s][-1].get("val_return_sampled", -np.inf) for s in learned["ppo"]]
        )
        ppo_mode = "sampled" if sampled > greedy else "greedy"
        if ppo_mode == "sampled":
            from fanet_defense.rollout import NumpyPPOPolicy

            learned["ppo"] = {
                s: NumpyPPOPolicy(p.params, cfg, deterministic=False, seed=s, name="ppo")  # type: ignore[attr-defined]
                for s, p in learned["ppo"].items()
            }
        log(
            f"ppo evaluation mode chosen on val: {ppo_mode} (greedy {greedy:.1f}, sampled {sampled:.1f})"
        )
    tab = [(a, s, args.steps) for a in ("q", "sarsa") if a in args.algos for s in args.seeds]
    if tab:
        from fanet_defense.tabular import TabularPolicy

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for algo, seed, params, curve in pool.map(_train_tabular_job, tab):
                np.savez_compressed(out / "policies" / f"{algo}_seed{seed}.npz", q=params["q"])
                learned[algo][seed] = TabularPolicy(params["q"], cfg, name=algo)
                curves[algo][seed] = curve
                log(f"{algo} seed {seed}: final val return {curve[-1]['val_return']:.1f}")

    # 3. Evaluation on identical scenarios.
    fixed: dict[str, Policy] = {
        "noop": NoOpPolicy(),
        "random": RandomPolicy(cfg),
        "watchdog": WatchdogPolicy(cfg),
        "watchdog_tuned": tuned,
        "oracle (privileged)": OraclePolicy(cfg),
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for suite_name, specs in suites.items():
            for name, pol in fixed.items():
                res = evaluate(pol, specs, cfg, executor=pool)
                episodes[(name, suite_name)] = res
                summary.setdefault(name, {})[suite_name] = summarize_fixed(res)
                rows += [
                    {"policy": name, "seed": -1, "suite": suite_name, **r.to_dict()} for r in res
                ]
            log(f"baselines evaluated on {suite_name}")
            for algo, by_seed in learned.items():
                per_seed: dict[str, list[float]] = {m: [] for m in METRICS}
                for seed, pol in by_seed.items():
                    res = evaluate(pol, specs, cfg, executor=pool)
                    episodes[(f"{algo}_seed{seed}", suite_name)] = res
                    for m in METRICS:
                        per_seed[m].append(float(np.mean([getattr(r, m) for r in res])))
                    rows += [
                        {"policy": algo, "seed": seed, "suite": suite_name, **r.to_dict()}
                        for r in res
                    ]
                if by_seed:
                    summary.setdefault(algo, {})[suite_name] = {
                        m: list(seed_interval(v)) for m, v in per_seed.items()
                    } | {"per_seed_ret": per_seed["ret"]}
            log(f"learned policies evaluated on {suite_name}")

    # 4. Paired differences against the tuned watchdog on the test suite (per training seed).
    paired: dict[str, Any] = {}
    base = episodes[("watchdog_tuned", "test")]
    for (name, suite_name), res in episodes.items():
        if suite_name == "test" and name != "watchdog_tuned":
            paired[name] = list(paired_difference(res, base, "ret"))

    meta = {
        "benchmark": "fanet_defense_v0",
        "git_commit": git_commit(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.processor() or platform.machine(),
        "steps_per_run": args.steps,
        "seeds": args.seeds,
        "env_config": asdict(cfg),
        "suites": {
            k: {"n": n, "master_seed": TEST_MASTER, "sampler": repr(s)}
            for k, (s, n) in SUITES.items()
        },
        "val": {"n": len(val), "master_seed": VAL_MASTER},
        "watchdog_tuned": {
            "th_fwd": tuned.th_fwd,
            "th_anom": tuned.th_anom,
            "th_pdr": tuned.th_pdr,
        },
        "val_reference_return": val_reference,
        "ppo_eval_mode": ppo_mode,
        "wall_clock_s": round(time.perf_counter() - t_start, 1),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    payload = {
        "meta": meta,
        "summary": summary,
        "paired_vs_watchdog_tuned_test": paired,
        "curves": curves,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "episodes.csv.gz", index=False)
    log(f"done: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
