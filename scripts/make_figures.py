"""Figures (light and dark) and result tables from ``results/summary.json``.

    python scripts/make_figures.py

Writes docs/figures/*.png, docs/results.md, and refreshes the README block between
``<!-- results:start -->`` and ``<!-- results:end -->`` so the README can never drift from the
committed results.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fanet_defense import EnvConfig, FanetSim, ScenarioSampler
from fanet_defense.policies import NoOpPolicy, Policy, WatchdogPolicy
from fanet_defense.viz import THEMES, draw_snapshot, legend_handles

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"
LABELS = {
    "noop": "No-op",
    "random": "Random",
    "watchdog": "Watchdog (default thresholds)",
    "watchdog_tuned": "Watchdog (tuned on val)",
    "q": "Q-learning (tabular)",
    "sarsa": "SARSA (tabular)",
    "ppo": "PPO (shared policy)",
    "ppo_bc": "PPO + behaviour cloning",
    "oracle (privileged)": "Oracle (privileged)",
}
LEARNED = ("ppo", "q", "sarsa", "ppo_bc")
SUITE_LABELS = {
    "test": "Test (in-distribution)",
    "ood_large_swarm": "Larger swarms (21-24 drones)",
    "ood_flood_heavy": "Flood-heavy attacks",
    "ood_stealthy": "Stealthy implants (stealth >= 0.5)",
}


def _style(ax: Any, th: dict[str, str]) -> None:
    ax.set_facecolor(th["surface"])
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=8, length=0)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])
    ax.grid(axis="x", color=th["grid"], lw=1)
    ax.set_axisbelow(True)


def fig_results(summary: dict[str, Any], theme: str) -> Path:
    th = THEMES[theme]
    s = summary["summary"]
    rows = [(n, *s[n]["test"]["ret"]) for n in s if n != "oracle (privileged)"]
    rows.sort(key=lambda r: r[1])
    oracle = s["oracle (privileged)"]["test"]["ret"][0]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.2), dpi=150, facecolor=th["surface"])
    _style(ax, th)
    for k, (name, m, lo, hi) in enumerate(rows):
        color = th["s1"] if name in LEARNED else th["muted"]
        ax.plot([lo, hi], [k, k], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([m], [k], s=64, color=color, edgecolor=th["surface"], linewidth=2, zorder=3)
        ax.text(hi + 2, k, f"{m:.1f}", va="center", fontsize=8, color=th["ink2"])
    ax.axvline(oracle, color=th["ink2"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(
        oracle - 1,
        len(rows) - 0.55,
        f"oracle (privileged) {oracle:.1f}",
        ha="right",
        fontsize=8,
        color=th["ink2"],
    )
    ax.set_yticks(range(len(rows)), [LABELS.get(n, n) for n, *_ in rows])
    n_test = summary["meta"]["suites"]["test"]["n"]
    ax.set_xlabel(f"Episode return on {n_test} held-out test scenarios (mean, 95% interval)")
    lo_all = min(r[2] for r in rows)
    ax.set_xlim(min(0, lo_all - 5), oracle + 12)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=th["s1"],
            label=f"learned ({len(summary['meta']['seeds'])} training seeds, t-interval)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=th["muted"],
            label="scripted (bootstrap over scenarios)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False, labelcolor=th["ink2"])
    fig.tight_layout()
    out = FIG / f"results_test_{theme}.png"
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)
    return out


def fig_curves(summary: dict[str, Any], theme: str) -> Path:
    th = THEMES[theme]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=150, facecolor=th["surface"])
    _style(ax, th)
    ax.grid(axis="y", color=th["grid"], lw=1)
    for slot, algo in zip(("s1", "s2", "s3", "s4"), LEARNED, strict=True):
        if algo not in summary["summary"]:
            continue
        runs = summary["curves"].get(algo) or {}
        sampled = (
            algo in ("ppo", "ppo_bc") and summary["meta"].get(f"{algo}_eval_mode") == "sampled"
        )
        key = "val_return_sampled" if sampled else "val_return"
        label = LABELS[algo]
        if algo == "ppo":
            label = f"PPO (shared policy, {'sampled actions' if sampled else 'greedy'})"
        elif algo == "ppo_bc":
            label += f" ({'sampled actions' if sampled else 'greedy'})"
        series = [[r for r in rows if key in r] for rows in runs.values()]
        series = [s for s in series if s]
        if not series:
            continue
        n = min(len(s) for s in series)
        x = np.array([r["env_steps"] for r in series[0][:n]]) / 1000
        y = np.array([[r[key] for r in s[:n]] for s in series])
        mean = y.mean(axis=0)
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=th[slot], alpha=0.10, lw=0)
        ax.plot(x, mean, color=th[slot], lw=2, solid_capstyle="round", label=label)
        ax.scatter(
            [x[-1]],
            [mean[-1]],
            s=40,
            color=th[slot],
            edgecolor=th["surface"],
            linewidth=2,
            zorder=3,
        )
    refs = summary["meta"].get("val_reference_return", {})
    for key, label in (
        ("oracle", "oracle (privileged)"),
        ("watchdog_tuned", "tuned watchdog"),
        ("noop", "no-op"),
    ):
        if key in refs:
            ax.axhline(refs[key], color=th["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(
                ax.get_xlim()[0],
                refs[key] + 2,
                f" {label}",
                fontsize=7,
                color=th["ink2"],
                va="bottom",
            )
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean return on validation scenarios")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, 0.84),
        fontsize=7,
        frameon=False,
        labelcolor=th["ink2"],
    )
    fig.tight_layout()
    out = FIG / f"learning_curves_{theme}.png"
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)
    return out


def _run_to(policy: Policy, spec: Any, t_stop: int) -> FanetSim:
    sim = FanetSim(spec, EnvConfig())
    policy.reset(spec)
    obs = sim.obs
    while sim.t < t_stop:
        obs, *_ = sim.step(policy.act(obs, sim if policy.privileged else None))
    return sim


def defended_policy(summary: dict[str, Any]) -> tuple[Policy, str]:
    s = summary["summary"]
    candidates = [
        algo
        for algo in ("ppo", "ppo_bc")
        if algo in s and (RES / "policies" / f"{algo}_seed0.npz").exists()
    ]
    for algo in sorted(candidates, key=lambda n: s[n]["test"]["ret"][0], reverse=True):
        if s[algo]["test"]["ret"][0] >= s["watchdog_tuned"]["test"]["ret"][0]:
            from fanet_defense.rollout import NumpyPPOPolicy

            return NumpyPPOPolicy.load(RES / "policies" / f"{algo}_seed0.npz"), _label(
                f"{algo}_seed0"
            )
    w = summary["meta"]["watchdog_tuned"]
    return WatchdogPolicy(EnvConfig(), w["th_fwd"], w["th_anom"], w["th_pdr"]), "tuned watchdog"


def fig_snapshot(summary: dict[str, Any], theme: str, t_stop: int = 90) -> Path:
    th = THEMES[theme]
    sampler = ScenarioSampler("medium")
    defended, name = defended_policy(summary)
    for seed in range(200):  # first seed (deterministic) where doing nothing visibly loses
        spec = sampler.sample(seed)
        if spec.n_drones < 14:
            continue
        a, b = _run_to(NoOpPolicy(), spec, t_stop), _run_to(defended, spec, t_stop)
        if a.comp[: a.n].sum() >= 4 and b.comp[: b.n].sum() <= 1:
            break
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0), dpi=150, facecolor=th["surface"])
    for ax, sim, title in ((axes[0], a, "No defense"), (axes[1], b, f"Defended by {name}")):
        info = f"{title}\n{int(sim.comp[: sim.n].sum())} of {sim.n} drones compromised at t = {sim.t} s"
        draw_snapshot(ax, sim.snapshot(), spec, theme, info)
    fig.legend(
        handles=legend_handles(theme),
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        labelcolor=th["ink2"],
    )
    fig.suptitle(
        f"Same scenario {spec.scenario_id}, same mobility and attacker dice",
        fontsize=9,
        color=th["ink2"],
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    out = FIG / f"snapshot_{theme}.png"
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)
    return out


def _ci(v: list[float], fmt: str = "{:.1f}") -> str:
    m, lo, hi = v
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _label(name: str) -> str:
    """``ppo_seed0`` -> ``PPO (shared policy), seed 0``; fixed policies keep their label."""
    algo, sep, seed = name.rpartition("_seed")
    if sep and algo in LEARNED and seed.isdigit():
        return f"{LABELS[algo]}, seed {seed}"
    return LABELS.get(name, name)


def caption(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    seeds = meta["seeds"]
    bc_mode = ""
    if "ppo_bc" in summary["summary"]:
        mode = "stochastically" if meta.get("ppo_bc_eval_mode") == "sampled" else "greedily"
        bc_mode = f"; PPO + behaviour cloning acts {mode} (chosen on validation)"
    return (
        f"Held-out test suite of {meta['suites']['test']['n']} scenarios, identical for every"
        f" policy. Learned policies: {len(seeds)} training seeds x {meta['steps_per_run']:,}"
        f" environment steps, mean over seeds with a Student-t 95 % interval; PPO acts"
        f" {'stochastically' if meta.get('ppo_eval_mode') == 'sampled' else 'greedily'}"
        f" (chosen on validation){bc_mode}. Scripted policies: bootstrap 95 % interval over scenarios.\n"
    )


def results_tables(summary: dict[str, Any]) -> tuple[str, str]:
    s = summary["summary"]
    order = [
        "noop",
        "random",
        "watchdog",
        "watchdog_tuned",
        "q",
        "sarsa",
        "ppo",
        "ppo_bc",
        "oracle (privileged)",
    ]
    names = [n for n in order if n in s]
    head = "| Policy | Return | Availability | Compromised share | False blocks | Contained |\n|---|---|---|---|---|---|\n"
    rows = ""
    for n in names:
        t = s[n]["test"]
        rows += (
            f"| {LABELS[n]} | {_ci(t['ret'])} | {t['availability'][0]:.3f} | {t['compromised'][0]:.3f} "
            f"| {t['false_blocks'][0]:.1f} | {100 * t['contained'][0]:.0f} % |\n"
        )
    main = head + rows
    suites = [k for k in SUITE_LABELS if k in s[names[0]]]
    ood = "| Policy | " + " | ".join(SUITE_LABELS[k] for k in suites) + " |\n"
    ood += "|---|" + "---|" * len(suites) + "\n"
    for n in names:
        ood += f"| {LABELS[n]} | " + " | ".join(_ci(s[n][k]["ret"]) for k in suites) + " |\n"
    return main, ood


def write_results_md(summary: dict[str, Any]) -> Path:
    meta = summary["meta"]
    main, ood = results_tables(summary)
    s = summary["summary"]
    random = s["random"]["test"]
    tuned_false_blocks = s["watchdog_tuned"]["test"]["false_blocks"][0]
    learned_false_blocks = "; ".join(
        f"{LABELS[algo]} {s[algo]['test']['false_blocks'][0]:.1f} vs tuned watchdog"
        f" {tuned_false_blocks:.1f}"
        + (
            f" (x{s[algo]['test']['false_blocks'][0] / tuned_false_blocks:.1f})"
            if tuned_false_blocks
            else " (ratio undefined: tuned watchdog has zero false blocks)"
        )
        for algo in LEARNED
        if algo in s
    )
    seed_reports = []
    for algo in LEARNED:
        if algo not in s:
            continue
        returns = s[algo]["test"]["per_seed_ret"]
        threshold = max(returns) / 2
        per_seed = ", ".join(
            f"seed {seed}: {ret:.1f}" + (" (failed)" if ret < threshold else "")
            for seed, ret in zip(meta["seeds"], returns, strict=True)
        )
        seed_reports.append(f"{LABELS[algo]} per-seed test returns: {per_seed}.")
    cloning_report = []
    if "ppo_bc" in s:
        initial = [
            next(row for row in rows if row["env_steps"] == 0)
            for rows in summary["curves"]["ppo_bc"].values()
        ]
        agreement = np.mean([row["bc_agreement"] for row in initial])
        budget = np.mean([row["bc_env_steps"] for row in initial])
        cloning_report = [
            f"PPO + behaviour cloning agreement at env_steps = 0: {agreement:.1%}"
            f" (mean over seeds); demonstration budget (bc_env_steps): {budget:,.0f}"
            " environment steps per seed (mean over seeds)."
        ]
        mode = "with sampled actions" if meta.get("ppo_bc_eval_mode") == "sampled" else "greedily"
        cloning_report.append(f"PPO + behaviour cloning is evaluated {mode}, chosen on validation.")
    paired = summary.get("paired_vs_watchdog_tuned_test", {})
    n_val = meta["val"]["n"]
    ood_sizes = ", ".join(
        f"{SUITE_LABELS[k].lower()} {v['n']}" for k, v in meta["suites"].items() if k != "test"
    )
    lines = [
        "# Results",
        "",
        f"Generated by `scripts/make_figures.py` from `results/summary.json` (commit `{meta['git_commit']}`,"
        f" {meta['date']}, {meta['machine']}, Python {meta['python']}, NumPy {meta['numpy']},"
        f" PyTorch {meta.get('torch', 'n/a')}). Wall clock for the whole protocol: {meta['wall_clock_s'] / 60:.0f} min.",
        "",
        f"Protocol: {meta['steps_per_run']:,} environment steps per training run, training seeds {meta['seeds']},",
        f"fresh training scenarios every episode, thresholds and hyper-parameters chosen on {n_val} validation",
        "scenarios, evaluation on identical held-out scenarios. Intervals are 95 %: Student t over training",
        "seeds for learned policies, percentile bootstrap over scenarios for scripted ones. PPO is evaluated",
        f"{'with sampled actions' if meta.get('ppo_eval_mode') == 'sampled' else 'greedily'}, the mode with",
        "the higher validation return (decided once for all seeds, before the test suite was touched).",
        "",
        f"## Test suite ({meta['suites']['test']['n']} scenarios)",
        "",
        main,
        "Availability is the per-step mission availability averaged over the episode; the compromised share",
        "is averaged over the episode; contained means that the active threat was suppressed for at least",
        "5 consecutive steps.",
        "",
        f"Containment alone is gamed by blanket blocking: the random policy achieves"
        f" {100 * random['contained'][0]:.0f} % containment with"
        f" {random['false_blocks'][0]:.1f} false blocks per episode. Mean false blocks"
        f" per episode for each learned policy versus the tuned watchdog (ratios to the"
        f" tuned watchdog in parentheses): {learned_false_blocks}.",
        "",
        "A training seed is marked failed only when its test return is below half of that policy's best seed return.",
        *seed_reports,
        *cloning_report,
        "",
        f"## Out-of-distribution families (return; scenarios per family: {ood_sizes})",
        "",
        ood,
        "## Paired difference against the tuned watchdog (test, return)",
        "",
        "Positive means better than the tuned watchdog on the same scenarios; one row per training seed",
        "for learned policies.",
        "",
        "| Policy | Mean difference [95 % CI] |",
        "|---|---|",
    ]
    lines += [f"| {_label(k)} | {_ci(v)} |" for k, v in sorted(paired.items())]
    w = meta["watchdog_tuned"]
    lines += [
        "",
        f"Tuned watchdog thresholds (random search, 40 trials on val): forwarding < {w['th_fwd']:.3f},"
        f" anomaly > {w['th_anom']:.3f}, own delivery < {w['th_pdr']:.3f}.",
        "",
        f"Learning curves: PPO{' and PPO + behaviour cloning are' if 'ppo_bc' in s else ' is'} scored on the {n_val} validation scenarios, the tabular learners on",
        "the first 16 of them (cheaper evaluation inside the training workers).",
        "",
    ]
    out = ROOT / "docs" / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def refresh_readme(summary: dict[str, Any]) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    main, _ = results_tables(summary)
    block = f"<!-- results:start -->\n{caption(summary)}\n{main}<!-- results:end -->"
    new = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", block, text, flags=re.S)
    if new != text:
        readme.write_text(new, encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    for theme in ("light", "dark"):
        for fn in (fig_results, fig_curves, fig_snapshot):
            print("wrote", fn(summary, theme))
    print("wrote", write_results_md(summary))
    refresh_readme(summary)


if __name__ == "__main__":
    main()
