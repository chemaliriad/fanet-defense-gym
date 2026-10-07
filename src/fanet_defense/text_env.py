"""Text interface: the same scenarios, played by a language-model defender.

A central "SOC" defender reads every drone's *local* evidence rendered as text and answers
with JSON actions. It sees exactly what the decentralised agents see (anomaly score, own
delivery ratio, neighbours' overheard forwarding ratios, its own blocks), only aggregated in
one place; hidden state never appears in the prompt. Rewards come from the simulator, so they
are verifiable: no model grades another model.

The prompt contains numbers and fixed tokens only, never free text taken from the
environment, so there is no channel for prompt injection through observations.

Answer format::

    {"actions": [{"drone": 3, "op": "restore"},
                 {"drone": 5, "op": "block", "target": 7}]}

``block`` is allowed only towards a drone currently listed as a neighbour and not already
blocked. Anything else (bad JSON, unknown drone, illegal target, two actions for one drone)
is ignored and counted, optionally with a penalty.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .scenario import ScenarioSampler, ScenarioSpec
from .sim import FanetSim

CHARSET = string.ascii_letters + string.digits + " \n.,:;%*=/|()[]{}\"'<>_-+#!?"
MAX_OBS_CHARS = 20_000
MAX_ACTION_CHARS = 4_096

INSTRUCTIONS = (
    "You defend a drone swarm whose telemetry must reach a ground station. Some drones may "
    "carry implants that drop or jam traffic and spread to neighbours. Each line shows one "
    "drone: its host anomaly score (higher is more suspicious), the share of its own packets "
    "acknowledged by the station, and its neighbours with the share of relayed packets they "
    "were overheard forwarding (* = already blocked by this drone). "
    "Actions: restore a drone (20 s offline, then clean) or make a drone block one neighbour "
    "for 25 s. Unnecessary actions cost service. "
    'Answer with JSON only: {"actions": [{"drone": <id>, "op": "restore"} | '
    '{"drone": <id>, "op": "block", "target": <neighbour id>}]}. An empty list does nothing.'
)


@dataclass
class ParseResult:
    actions: np.ndarray  # (n,) action ids for the simulator
    n_valid: int = 0
    n_invalid: int = 0
    parse_error: str | None = None
    notes: list[str] = field(default_factory=list)


def render(sim: FanetSim) -> str:
    """Prompt for the current state, built from the observation vectors only."""
    n, k = sim.n, sim.cfg.k_neighbors
    obs = sim.obs
    lines = [
        f"t={sim.t}/{sim.spec.horizon} | drones={n}",
        "drone anomaly delivery neighbours(id:forwarding%)",
    ]
    for i in range(n):
        if not sim.online[i]:
            lines.append(f"d{i:02d} offline (re-flashing)")
            continue
        slots = obs[i, 3:].reshape(k, 4)
        neigh = []
        for s in range(k):
            j = int(sim.slots[i, s])
            if j < 0 or slots[s, 0] < 0.5:
                continue
            mark = "*" if slots[s, 3] > 0.5 else ""
            neigh.append(f"d{j:02d}:{round(100 * float(slots[s, 2]))}{mark}")
        lines.append(
            f"d{i:02d} {obs[i, 0]:+.2f} {obs[i, 1]:.2f} " + (" ".join(neigh) if neigh else "-")
        )
    return "\n".join(lines)


def actions_to_json(actions: np.ndarray, sim: FanetSim) -> str:
    """Inverse of :func:`parse_actions` for a simulator action vector (used by teachers)."""
    out: list[dict[str, Any]] = []
    for i, a in enumerate(np.asarray(actions)):
        a = int(a)
        if a == sim.cfg.restore_action:
            out.append({"drone": i, "op": "restore"})
        elif 1 <= a <= sim.cfg.k_neighbors and sim.slots[i, a - 1] >= 0:
            out.append({"drone": i, "op": "block", "target": int(sim.slots[i, a - 1])})
    return json.dumps({"actions": out}, separators=(",", ":"))


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_actions(text: str, sim: FanetSim) -> ParseResult:
    """Map a model answer to simulator actions; never raises on malformed input."""
    n, cfg = sim.n, sim.cfg
    actions = np.zeros(n, dtype=np.int64)
    result = ParseResult(actions)
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        result.parse_error = "no JSON object found"
        return result
    try:
        payload = json.loads(match.group(0))
        items = payload["actions"]
        if not isinstance(items, list):
            raise TypeError("'actions' must be a list")
    except (ValueError, KeyError, TypeError) as exc:
        result.parse_error = f"invalid JSON: {exc}"
        return result

    seen: set[int] = set()
    usable = (sim.slots >= 0) & (sim.obs[:, 3:].reshape(n, cfg.k_neighbors, 4)[..., 3] < 0.5)
    for item in items:
        try:
            drone = int(item["drone"])
            op = str(item["op"])
        except (KeyError, TypeError, ValueError):
            result.n_invalid += 1
            result.notes.append("malformed action")
            continue
        if not 0 <= drone < n or drone in seen:
            result.n_invalid += 1
            result.notes.append(f"drone {drone}: unknown or duplicated")
            continue
        if not sim.online[drone]:
            result.n_invalid += 1
            result.notes.append(f"drone {drone}: offline, cannot act")
            continue
        if op == "restore":
            actions[drone] = cfg.restore_action
        elif op == "block":
            try:
                target = int(item["target"])
            except (KeyError, TypeError, ValueError):
                result.n_invalid += 1
                result.notes.append(f"drone {drone}: block without a valid target")
                continue
            slot = np.flatnonzero((sim.slots[drone] == target) & usable[drone])
            if slot.size == 0:
                result.n_invalid += 1
                result.notes.append(f"drone {drone}: d{target:02d} is not a blockable neighbour")
                continue
            actions[drone] = int(slot[0]) + 1
        else:
            result.n_invalid += 1
            result.notes.append(f"drone {drone}: unknown op {op!r}")
            continue
        seen.add(drone)
        result.n_valid += 1
    return result


class FanetTextEnv(gym.Env[str, str]):
    """Gymnasium environment with text observations and JSON-text actions.

    ``decision_interval`` lets the defender act every few seconds (no-op in between), which
    divides the number of model calls per episode; the reward of a call is the sum over the
    steps it covers. ``invalid_penalty`` is subtracted per invalid action or unparsable answer.
    """

    metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium convention

    def __init__(
        self,
        scenario: ScenarioSpec | None = None,
        sampler: ScenarioSampler | None = None,
        config: EnvConfig | None = None,
        decision_interval: int = 1,
        invalid_penalty: float = 0.0,
    ) -> None:
        if decision_interval < 1:
            raise ValueError("decision_interval must be >= 1")
        self.cfg = config or EnvConfig()
        self._fixed = scenario
        self._sampler = sampler or ScenarioSampler()
        self.decision_interval = decision_interval
        self.invalid_penalty = invalid_penalty
        self.observation_space = spaces.Text(max_length=MAX_OBS_CHARS, charset=CHARSET)
        self.action_space = spaces.Text(max_length=MAX_ACTION_CHARS, charset=CHARSET)
        self.sim: FanetSim | None = None
        self._rng: np.random.Generator | None = None

    @property
    def instructions(self) -> str:
        return INSTRUCTIONS

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        spec = (options or {}).get("scenario") or self._fixed
        if spec is None:
            spec = self._sampler.sample(int(self._rng.integers(2**62)))
        self.sim = FanetSim(spec, self.cfg)
        return render(self.sim), {"scenario_id": spec.scenario_id}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("call reset() first")
        sim = self.sim
        parsed = parse_actions(action, sim)
        reward = -self.invalid_penalty * (parsed.n_invalid + (parsed.parse_error is not None))
        acts = parsed.actions
        last: dict[str, Any] = {}
        for _ in range(self.decision_interval):
            _, r, done, last = sim.step(acts)
            reward += r
            acts = np.zeros(sim.n, dtype=np.int64)
            if done:
                break
        info = {
            "t": sim.t,
            "n_valid": parsed.n_valid,
            "n_invalid": parsed.n_invalid,
            "parse_error": parsed.parse_error,
            "notes": parsed.notes,
            **{k: last.get(k) for k in ("availability", "threat", "n_compromised")},
        }
        return render(sim), float(reward), False, sim.done, info
