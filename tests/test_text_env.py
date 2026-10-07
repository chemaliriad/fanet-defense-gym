"""Text interface, dataset generation and the LLM client (no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fanet_defense import EnvConfig, FanetSim, ScenarioSampler, ScenarioSpec, generate_suite
from fanet_defense.dataset import iter_records, write_jsonl
from fanet_defense.evaluate import run_episode
from fanet_defense.llm import ChatCompletionsDefender, PolicyTextDefender, run_text_episode
from fanet_defense.policies import OraclePolicy, WatchdogPolicy
from fanet_defense.text_env import CHARSET, FanetTextEnv, actions_to_json, parse_actions, render

CFG = EnvConfig()


def spec(**kw) -> ScenarioSpec:
    base = {"seed": 4, "n_drones": 10, "comm_range": 420.0, "n_implants": 3, "horizon": 60}
    base.update(kw)
    return ScenarioSpec(**base)


def stepped_sim(steps: int = 25) -> FanetSim:
    sim = FanetSim(spec(wake_min=1, wake_max=3), CFG)
    pol = WatchdogPolicy(CFG)
    for _ in range(steps):
        sim.step(pol.act(sim.obs))
    return sim


# ---------------------------------------------------------------- rendering
def test_prompt_uses_only_the_allowed_charset_and_no_hidden_state():
    sim = stepped_sim()
    text = render(sim)
    assert set(text) <= set(CHARSET)
    for word in ("compromised", "implant", "blackhole", "greyhole", "flood", "attack"):
        assert word not in text.lower()


def test_prompt_lists_every_drone_once():
    sim = stepped_sim()
    lines = render(sim).splitlines()[2:]
    assert [line.split()[0] for line in lines] == [f"d{i:02d}" for i in range(sim.n)]


# ---------------------------------------------------------------- parsing
def test_actions_round_trip_through_json():
    sim = stepped_sim()
    rng = np.random.default_rng(0)
    for _ in range(20):
        acts = rng.integers(0, CFG.n_actions, size=sim.n)
        text = actions_to_json(acts, sim)
        back = parse_actions(text, sim).actions
        # Only legal actions survive the round trip; legal ones are preserved exactly.
        legal = parse_actions(actions_to_json(back, sim), sim).actions
        assert np.array_equal(back, legal)


@pytest.mark.parametrize(
    "answer",
    ["", "no idea", "{not json}", '{"actions": "restore everything"}', "[1, 2, 3]"],
)
def test_malformed_answers_do_nothing_and_are_reported(answer):
    sim = stepped_sim()
    res = parse_actions(answer, sim)
    assert not res.actions.any()
    assert res.parse_error is not None or res.n_invalid > 0


def test_illegal_actions_are_counted_not_applied():
    sim = stepped_sim()
    d = next(i for i in range(2, sim.n) if sim.online[i])
    answer = json.dumps(
        {
            "actions": [
                {"drone": 99, "op": "restore"},  # unknown drone
                {"drone": 0, "op": "block", "target": 0},  # not a neighbour of itself
                {"drone": 1, "op": "launch"},  # unknown op
                {"drone": d, "op": "restore"},
                {"drone": d, "op": "restore"},  # duplicate
            ]
        }
    )
    res = parse_actions(answer, sim)
    assert res.n_invalid == 4 and res.n_valid == 1
    assert res.actions[d] == CFG.restore_action and res.actions.sum() == CFG.restore_action


def test_json_can_be_wrapped_in_prose():
    sim = stepped_sim(steps=0)
    res = parse_actions('Sure! Here you go:\n{"actions": [{"drone": 3, "op": "restore"}]}\n', sim)
    assert res.actions[3] == CFG.restore_action and res.parse_error is None


# ---------------------------------------------------------------- environment
def test_gymnasium_env_checker_passes():
    check_env(FanetTextEnv(scenario=spec()), skip_render_check=True)


def test_text_interface_is_equivalent_to_the_numeric_env():
    """Playing the watchdog through text gives exactly the numeric return."""
    s = spec(wake_min=2, wake_max=10, exploit_prob=0.02)
    numeric = run_episode(WatchdogPolicy(CFG), s, CFG).ret
    env = FanetTextEnv(scenario=s, config=CFG)
    text = run_text_episode(env, PolicyTextDefender(WatchdogPolicy(CFG), env))
    assert text.ret == pytest.approx(numeric, abs=1e-9)
    assert text.invalid_actions == 0 and text.parse_errors == 0


def test_decision_interval_divides_the_number_of_calls():
    env = FanetTextEnv(scenario=spec(horizon=60), config=CFG, decision_interval=5)
    ep = run_text_episode(env, lambda prompt: '{"actions": []}')
    assert ep.decisions == 12


def test_invalid_penalty_is_applied_per_invalid_action():
    s = spec(n_implants=0)
    plain = FanetTextEnv(scenario=s, config=CFG)
    strict = FanetTextEnv(scenario=s, config=CFG, invalid_penalty=0.5)
    plain.reset()
    strict.reset()
    bad = '{"actions": [{"drone": 99, "op": "restore"}]}'
    _, r_plain, *_ = plain.step(bad)
    _, r_strict, *_ = strict.step(bad)
    assert r_plain - r_strict == pytest.approx(0.5)


# ---------------------------------------------------------------- datasets
def test_dataset_records_are_valid_reproducible_and_replayable(tmp_path):
    specs = generate_suite(3, 0, "train", ScenarioSampler("medium", horizon=40))
    a = list(iter_records(specs, WatchdogPolicy(CFG), CFG, decision_interval=5))
    b = list(iter_records(specs, WatchdogPolicy(CFG), CFG, decision_interval=5))
    assert a == b and len(a) == 3 * 8
    for rec in a:
        assert set(rec["prompt"]) <= set(CHARSET)
        assert json.loads(rec["completion"])["actions"] is not None
        assert isinstance(rec["reward"], float) and rec["n_invalid"] == 0
    n = write_jsonl(a, tmp_path / "sft.jsonl")
    lines = (tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == 24 and json.loads(lines[0])["scenario_id"] == specs[0].scenario_id


def test_privileged_teacher_is_labelled_as_such():
    recs = list(iter_records([spec(horizon=10)], OraclePolicy(CFG), CFG, decision_interval=5))
    assert {r["teacher"] for r in recs} == {"oracle"}


# ---------------------------------------------------------------- LLM client
def test_chat_client_builds_the_request_from_the_environment(monkeypatch):
    seen = {}

    def fake_transport(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=json.loads(body), timeout=timeout)
        content = '{"actions": []}'
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    client = ChatCompletionsDefender(model="some-model", transport=fake_transport)
    assert client("state") == '{"actions": []}'
    assert seen["url"] == "https://api.mistral.ai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["messages"][0]["content"] == "state"


def test_chat_client_refuses_to_run_without_a_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ChatCompletionsDefender(model="m", transport=lambda *a: b"")("state")
