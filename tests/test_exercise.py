"""Human exercise commands, paired scores and delayed ground truth."""

import json
from dataclasses import replace

import numpy as np
import pytest

from fanet_defense.config import EnvConfig
from fanet_defense.exercise import parse_command, run_exercise
from fanet_defense.policies import WatchdogPolicy
from fanet_defense.scenario import ScenarioSpec
from fanet_defense.sim import FanetSim
from fanet_defense.text_env import FanetTextEnv, actions_to_json, render


@pytest.mark.parametrize(
    ("line", "actions"),
    [
        ("block d05 d07", [{"drone": 5, "op": "block", "target": 7}]),
        ("RESTORE D03", [{"drone": 3, "op": "restore"}]),
        ("restore d3", [{"drone": 3, "op": "restore"}]),
        ("restore 3", [{"drone": 3, "op": "restore"}]),
        ("pass", []),
        ("", []),
        ("  ", []),
        (
            "restore 3; block D5 7; pass",
            [{"drone": 3, "op": "restore"}, {"drone": 5, "op": "block", "target": 7}],
        ),
    ],
)
def test_parse_command_shorthand(line, actions):
    assert json.loads(parse_command(line))["actions"] == actions


def test_parse_command_json_passthrough():
    line = ' {"actions": [{"drone": 3, "op": "restore"}]} '
    assert parse_command(line) == line


@pytest.mark.parametrize("line", ["fly 3", "restore", "block d5 x", "restore 3; fly 2"])
def test_parse_command_unknown_has_message(line):
    result = json.loads(parse_command(line))
    assert result["actions"] == []
    assert result["message"]


def scenario(**kwargs):
    return ScenarioSpec(seed=4, n_drones=10, horizon=43, comm_range=2000, **kwargs)


@pytest.mark.parametrize("interval", [1, 10, 17])
def test_watchdog_trainee_matches_watchdog(interval):
    spec = scenario()
    cfg = EnvConfig(restore_steps=3)
    env = FanetTextEnv(spec, config=cfg, decision_interval=interval)
    env.reset()
    policy = WatchdogPolicy(cfg)
    policy.reset(spec)

    def ask():
        answer = actions_to_json(policy.act(env.sim.obs), env.sim)
        env.step(answer)
        return answer

    report = run_exercise(spec, ask=ask, say=lambda _: None, config=cfg, decision_interval=interval)
    assert report.trainee_return == report.watchdog_return


def test_pass_matches_noop():
    report = run_exercise(scenario(), ask=lambda: "pass", say=lambda _: None)
    assert report.trainee_return == report.noop_return


def test_pre_report_transcript_matches_text_env_exactly():
    spec = scenario()
    env = FanetTextEnv(spec, decision_interval=10)
    env.reset()
    target = int(env.sim.slots[0, 0])
    answers = [f"block 0 {target}", "pass", "fly 3", "pass", "pass"]
    trainee = iter(answers)
    output = []
    report = run_exercise(spec, ask=lambda: next(trainee), say=output.append)

    expected = ["Commands: block d05 d07; restore d03; pass (or Enter); JSON; quit."]
    for line in answers:
        assert not env.sim.done
        expected.extend([f"Decision t={env.sim.t}s", render(env.sim)])
        if line == "fly 3":
            expected.append(
                "Invalid input: Unknown or malformed command; "
                "use block d05 d07, restore d03, or pass."
            )
        _, _, _, _, info = env.step(parse_command(line))
        assert info["n_valid"] == int(line.startswith("block"))
    assert env.sim.done
    assert next(trainee, None) is None
    assert output[:-1] == expected
    assert output[-1] == report.to_markdown()
    for drone in report.compromised_drones:
        assert f"d{drone:02d}: compromised" in output[-1]


def test_quit_finishes_with_noops_and_report():
    output = []
    calls = []

    def ask():
        calls.append(1)
        return " QUIT "

    report = run_exercise(scenario(), ask=ask, say=output.append)
    assert len(calls) == 1
    assert report.quit_early
    assert report.trainee_return == report.noop_return
    assert output[-1] == report.to_markdown()


def test_clean_neighbour_is_one_false_block():
    spec = scenario(n_implants=0)
    sim = FanetSim(spec)
    target = int(sim.slots[0, 0])
    answers = iter([f"block 0 {target}", "quit"])
    report = run_exercise(spec, ask=lambda: next(answers), say=lambda _: None)
    assert report.false_blocks == 1
    assert len(report.actions) == 1
    row = report.actions[0]
    assert (row.time, row.drone, row.op, row.target, row.correct) == (0, 0, "block", target, False)
    assert "false block" in report.to_markdown()


def test_invalid_inputs_are_reported_before_next_question():
    output = []
    answers = iter(["fly 3", '{"actions":[{"drone":99,"op":"restore"}]}', "quit"])

    def ask():
        if len([line for line in output if line.startswith("Decision")]) > 1:
            assert output[-3].startswith("Invalid input:")
        return next(answers)

    report = run_exercise(scenario(), ask=ask, say=output.append)
    assert len(report.invalid_inputs) == 2
    assert report.trainee_return == report.noop_return


def test_containment_uses_every_simulation_step():
    from fanet_defense.evaluate import _time_to_containment

    spec = scenario(wake_min=1, wake_max=1)
    sim = FanetSim(spec)
    threats = []
    while not sim.done:
        threats.append(sim.step(np.zeros(sim.n, dtype=np.int64))[3]["threat"])
    report = run_exercise(spec, ask=lambda: "pass", say=lambda _: None)
    assert (report.ttc, report.contained) == _time_to_containment(np.asarray(threats))


def test_cli_play_writes_report(tmp_path, monkeypatch, capsys):
    from fanet_defense.cli import main

    monkeypatch.setattr("builtins.input", lambda: "quit")
    path = tmp_path / "report.md"
    assert main(["play", "--report", str(path)]) == 0
    report = path.read_text(encoding="utf-8")
    assert report.startswith("# After-action report")
    assert report in capsys.readouterr().out


def test_report_returns_and_signed_watchdog_differences():
    report = run_exercise(scenario(), ask=lambda: "quit", say=lambda _: None)
    report = replace(report, trainee_return=123.456, noop_return=80.123, watchdog_return=100.04)
    assert report.to_markdown().splitlines()[3:8] == [
        "| Defender | Return | vs watchdog |",
        "| --- | ---: | ---: |",
        "| Trainee | 123.5 | +23.4 |",
        "| No-op | 80.1 | -19.9 |",
        "| Watchdog (default) | 100.0 | +0.0 |",
    ]


@pytest.mark.parametrize("gap", ["-1", "nan", "inf"])
def test_cli_play_rejects_invalid_gap(gap):
    from fanet_defense.cli import main

    with pytest.raises(SystemExit, match="--min-gap must be finite and >= 0"):
        main(["play", f"--min-gap={gap}"])


@pytest.mark.parametrize("options", [[], ["--seed", "1", "--interval", "17", "--min-gap", "40"]])
def test_cli_play_selects_first_scenario_with_gap(options, monkeypatch, capsys):
    from fanet_defense.cli import build_parser, main
    from fanet_defense.llm import PolicyTextDefender, run_text_episode
    from fanet_defense.policies import NoOpPolicy
    from fanet_defense.scenario import ScenarioSampler

    args = build_parser().parse_args(["play", *options])
    assert args.difficulty == "medium"
    if not options:
        assert (args.seed, args.interval, args.min_gap) == (0, 10, 30.0)
    reports = []

    def capture_report(spec, **kwargs):
        report = run_exercise(spec, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr("fanet_defense.exercise.run_exercise", capture_report)
    monkeypatch.setattr("builtins.input", lambda: "quit")
    assert main(["play", *options]) == 0
    report = reports[0]
    assert report.watchdog_return - report.noop_return >= args.min_gap
    assert report.seed >= args.seed
    assert report.decision_interval == args.interval
    assert f"Exercise seed: {report.seed}" in capsys.readouterr().out
    sampler = ScenarioSampler(difficulty=args.difficulty)
    assert report.scenario_id == sampler.sample(report.seed).scenario_id
    for seed in range(args.seed, report.seed):
        returns = []
        for policy in (NoOpPolicy(), WatchdogPolicy(EnvConfig())):
            env = FanetTextEnv(sampler.sample(seed), decision_interval=args.interval)
            returns.append(run_text_episode(env, PolicyTextDefender(policy, env)).ret)
        assert returns[1] - returns[0] < args.min_gap
