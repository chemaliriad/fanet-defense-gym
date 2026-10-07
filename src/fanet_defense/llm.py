"""Run a language-model defender on the text interface.

``TextDefender`` is any callable from a prompt to an answer. Two are provided: a wrapper that
turns a numeric policy into text answers (for tests, baselines and data generation), and a
minimal client for OpenAI-compatible chat-completions endpoints (such as Mistral's API),
written with the standard library only. The API key is read from an environment variable at
call time and is never logged.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .policies import Policy
from .text_env import FanetTextEnv, actions_to_json

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


class TextDefender(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass
class TextEpisode:
    scenario_id: str
    ret: float
    decisions: int
    invalid_actions: int
    parse_errors: int


def run_text_episode(
    env: FanetTextEnv, defender: TextDefender, seed: int | None = None
) -> TextEpisode:
    """Play one episode; the defender sees the instructions followed by the current state."""
    prompt, info = env.reset(seed=seed)
    ret, decisions, invalid, errors = 0.0, 0, 0, 0
    done = False
    while not done:
        answer = defender(env.instructions + "\n\n" + prompt)
        prompt, reward, terminated, truncated, step_info = env.step(answer)
        ret += reward
        decisions += 1
        invalid += step_info["n_invalid"]
        errors += step_info["parse_error"] is not None
        done = terminated or truncated
    return TextEpisode(info["scenario_id"], ret, decisions, invalid, errors)


@dataclass
class PolicyTextDefender:
    """Answers with the JSON form of a numeric policy's actions on the env's current state."""

    policy: Policy
    env: FanetTextEnv

    def __call__(self, prompt: str) -> str:
        sim = self.env.sim
        assert sim is not None, "reset the environment first"
        if sim.t == 0:
            self.policy.reset(sim.spec)
        actions = self.policy.act(sim.obs, sim if self.policy.privileged else None)
        return actions_to_json(actions, sim)


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return bytes(resp.read())


@dataclass
class ChatCompletionsDefender:
    """Minimal client for an OpenAI-compatible ``/chat/completions`` endpoint."""

    model: str
    base_url: str = "https://api.mistral.ai/v1"
    api_key_env: str = "MISTRAL_API_KEY"
    temperature: float = 0.0
    timeout_s: float = 60.0
    transport: Transport = field(default=_urllib_transport, repr=False)

    def request(self, prompt: str) -> tuple[str, dict[str, str], bytes]:
        """Build the HTTP request (exposed for tests); the key comes from the environment."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"set {self.api_key_env} to call {self.base_url}")
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        url = self.base_url.rstrip("/") + "/chat/completions"
        return url, headers, json.dumps(body).encode()

    def __call__(self, prompt: str) -> str:
        url, headers, body = self.request(prompt)
        payload = json.loads(self.transport(url, headers, body, self.timeout_s))
        return str(payload["choices"][0]["message"]["content"])
