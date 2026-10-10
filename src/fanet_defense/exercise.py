"""Keyboard training with paired text baselines and a delayed, deterministic debrief."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import EnvConfig
from .evaluate import _time_to_containment
from .llm import PolicyTextDefender, run_text_episode
from .policies import NoOpPolicy, WatchdogPolicy
from .scenario import ATTACK_TYPES, ScenarioSpec
from .sim import FanetSim
from .text_env import FanetTextEnv, parse_actions

_NOOP = '{"actions": []}'
_COMMAND_ERROR = "Unknown or malformed command; use block d05 d07, restore d03, or pass."


def parse_command(line: str) -> str:
    """Keep JSON verbatim; invalid shorthand cancels the line and adds a JSON message."""
    try:
        json.loads(line)
    except (ValueError, RecursionError):
        pass
    else:
        return line
    actions: list[dict[str, int | str]] = []
    for command in line.lower().split(";"):
        words = command.split()
        if not words or words == ["pass"]:
            continue
        op, *ids = words
        if (
            op not in {"restore", "block"}
            or len(ids) != (1 if op == "restore" else 2)
            or any(re.fullmatch(r"d?[0-9]+", drone) is None for drone in ids)
        ):
            return json.dumps({"actions": [], "message": _COMMAND_ERROR})
        try:
            action: dict[str, int | str] = {"drone": int(ids[0].removeprefix("d")), "op": op}
            if op == "block":
                action["target"] = int(ids[1].removeprefix("d"))
        except ValueError:  # Includes Python's limit on extremely long integer strings.
            return json.dumps({"actions": [], "message": _COMMAND_ERROR})
        actions.append(action)
    return json.dumps({"actions": actions})


@dataclass
class DebriefAction:
    time: int
    drone: int
    op: str
    target: int | None
    correct: bool

    @property
    def verdict(self) -> str:
        if self.correct:
            return "correct"
        return "false block" if self.op == "block" else "unnecessary restore"


@dataclass
class InvalidInput:
    time: int
    line: str
    messages: list[str]


@dataclass
class ExerciseReport:
    scenario_id: str
    seed: int
    decision_interval: int
    trainee_return: float
    noop_return: float
    watchdog_return: float
    actions: list[DebriefAction]
    compromised_drones: dict[int, list[str]]
    attackers_never_addressed: list[int]
    ttc: float
    contained: bool
    invalid_inputs: list[InvalidInput]
    quit_early: bool

    @property
    def false_blocks(self) -> int:
        return sum(row.op == "block" and not row.correct for row in self.actions)

    @property
    def unnecessary_restores(self) -> int:
        return sum(row.op == "restore" and not row.correct for row in self.actions)

    def to_markdown(self) -> str:
        lines = [
            "# After-action report",
            f"Scenario: `{self.scenario_id}` | seed={self.seed} | "
            f"interval={self.decision_interval}s",
            "",
            "| Defender | Return | vs watchdog |",
            "| --- | ---: | ---: |",
            f"| Trainee | {self.trainee_return:.1f} | "
            f"{self.trainee_return - self.watchdog_return:+.1f} |",
            f"| No-op | {self.noop_return:.1f} | {self.noop_return - self.watchdog_return:+.1f} |",
            f"| Watchdog (default) | {self.watchdog_return:.1f} | +0.0 |",
            "",
            f"False blocks: {self.false_blocks}; "
            f"unnecessary restores: {self.unnecessary_restores}.",
            f"Invalid inputs: {len(self.invalid_inputs)}; quit early: {self.quit_early}.",
            f"Time to containment: {self.ttc:g}s; contained: {self.contained}.",
            "Containment uses five consecutive threat-free seconds after the first active threat; "
            "uncontained times are censored at the horizon (no active threat: 0s).",
            "",
            "## Ground truth (trainee trajectory)",
        ]
        lines.extend(
            f"- d{drone:02d}: compromised; attack types: {', '.join(types)}"
            for drone, types in sorted(self.compromised_drones.items())
        )
        if not self.compromised_drones:
            lines.append("No compromised drones.")
        missed = ", ".join(f"d{i:02d}" for i in self.attackers_never_addressed) or "none"
        lines.extend(
            [
                f"Attackers never blocked or restored while compromised: {missed}.",
                "Includes dormant implants and infections observed at any simulation step.",
                "",
                "## Action debrief",
                "Correctness uses hidden state at decision time; only accepted actions are listed.",
                "| Time (s) | Drone | Operation | Target | Verdict |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for row in self.actions:
            target = "-" if row.target is None else f"d{row.target:02d}"
            lines.append(f"| {row.time} | d{row.drone:02d} | {row.op} | {target} | {row.verdict} |")
        lines.extend(["", "## Invalid inputs"])
        for entry in self.invalid_inputs:
            # JSON escaping keeps arbitrary terminal input on a single Markdown line.
            escaped = json.dumps(entry.line, ensure_ascii=True).replace("<", "&lt;")
            escaped = escaped.replace("`", "&#96;").replace("|", "&#124;")
            lines.append(f"- t={entry.time}s: {escaped}")
        if not self.invalid_inputs:
            lines.append("None.")
        lines.extend(["", "## What to look at next time"])
        if self.false_blocks:
            lines.append(
                "- Check forwarding evidence before blocking: clean neighbours were blocked."
            )
        if self.unnecessary_restores:
            lines.append("- Combine anomaly and delivery evidence before taking a drone offline.")
        if self.attackers_never_addressed:
            lines.append(
                "- Revisit persistent anomalies: some compromised drones received no action."
            )
        if not self.contained:
            lines.append(
                "- Follow up after interventions: the threat did not stay quiet for five seconds."
            )
        if self.invalid_inputs:
            lines.append("- Check command syntax and the current neighbour list before acting.")
        if self.trainee_return < self.watchdog_return:
            lines.append(
                "- Review intervention timing: the default watchdog achieved a higher return."
            )
        if lines[-1] == "## What to look at next time":
            lines.append(
                "- Maintain evidence checks and monitor for renewed anomalies after actions."
            )
        return "\n".join(lines) + "\n"


def _debrief(
    scenario: ScenarioSpec, cfg: EnvConfig, decisions: dict[int, str]
) -> tuple[list[DebriefAction], dict[int, list[str]], list[int], float, bool]:
    """Replay only after the exercise, sampling truth at every second (including t=0)."""
    sim = FanetSim(scenario, cfg)
    rows: list[DebriefAction] = []
    compromised: dict[int, list[str]] = {}
    addressed: set[int] = set()
    threats = []
    while True:
        for drone in np.flatnonzero(sim.comp[: sim.n]):
            types = compromised.setdefault(int(drone), [])
            attack = ATTACK_TYPES[int(sim.atk[drone])]
            if attack not in types:
                types.append(attack)
        if sim.done:
            break
        actions = parse_actions(decisions.get(sim.t, _NOOP), sim).actions
        for drone in np.flatnonzero(actions):
            restoring = actions[drone] == cfg.restore_action
            target = None if restoring else int(sim.slots[drone, actions[drone] - 1])
            subject = int(drone) if target is None else target
            correct = bool(sim.comp[subject])
            rows.append(
                DebriefAction(
                    sim.t, int(drone), "restore" if restoring else "block", target, correct
                )
            )
            if correct:
                addressed.add(subject)
        threats.append(sim.step(actions)[3]["threat"])
    ttc, contained = _time_to_containment(np.asarray(threats))
    return rows, compromised, sorted(set(compromised) - addressed), ttc, contained


def run_exercise(
    scenario: ScenarioSpec,
    *,
    ask: Callable[[], str],
    say: Callable[[str], object],
    config: EnvConfig | None = None,
    decision_interval: int = 10,
) -> ExerciseReport:
    """Play, finish quit episodes with no-ops, then print and return the debrief.

    ``ask`` takes no arguments (like ``input()``); ``say`` receives each display string.
    Invalid lines consume a decision, with valid JSON actions retained by the text parser.
    """
    cfg = config or EnvConfig()
    env = FanetTextEnv(scenario, config=cfg, decision_interval=decision_interval)
    prompt, _ = env.reset()
    assert env.sim is not None
    decisions: dict[int, str] = {}
    invalid: list[InvalidInput] = []
    ret = 0.0
    quit_early = False
    say("Commands: block d05 d07; restore d03; pass (or Enter); JSON; quit.")
    while not env.sim.done:
        answer = _NOOP
        if not quit_early:
            say(f"Decision t={env.sim.t}s")
            say(prompt)
            try:
                line = ask()
            except EOFError:
                line = "quit"
            quit_early = line.strip().lower() == "quit"
            if not quit_early:
                answer = parse_command(line)
                messages: list[str] = []
                try:
                    payload = json.loads(answer)
                    if isinstance(payload, dict) and payload.get("message") == _COMMAND_ERROR:
                        messages.append(_COMMAND_ERROR)
                    parsed = parse_actions(answer, env.sim)
                    messages.extend(parsed.notes)
                    if parsed.parse_error:
                        messages.append(parsed.parse_error)
                except (ValueError, TypeError, OverflowError, RecursionError):
                    # Guard malformed JSON values the underlying numeric parser cannot convert.
                    answer = _NOOP
                    messages.append("Malformed JSON action.")
                if messages:
                    invalid.append(InvalidInput(env.sim.t, line, messages))
                    say("Invalid input: " + "; ".join(messages))
        decisions[env.sim.t] = answer
        prompt, reward, _, _, _ = env.step(answer)
        ret += reward
    rows, compromised, missed, ttc, contained = _debrief(scenario, cfg, decisions)
    baselines = []
    for policy in (NoOpPolicy(), WatchdogPolicy(cfg)):
        baseline = FanetTextEnv(scenario, config=cfg, decision_interval=decision_interval)
        baselines.append(run_text_episode(baseline, PolicyTextDefender(policy, baseline)).ret)
    report = ExerciseReport(
        scenario.scenario_id,
        scenario.seed,
        decision_interval,
        ret,
        baselines[0],
        baselines[1],
        rows,
        compromised,
        missed,
        ttc,
        contained,
        invalid,
        quit_early,
    )
    say(report.to_markdown())
    return report
