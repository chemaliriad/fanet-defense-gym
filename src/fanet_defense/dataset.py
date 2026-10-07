"""Synthetic training instances at scale: prompt / action / reward records as JSONL.

Each record is one decision of a *teacher* policy in the text interface: the prompt the model
would see, the teacher's JSON answer, and the verifiable reward that answer earned. The same
generator serves supervised fine-tuning (imitate the teacher) and reinforcement learning
(re-score a model's own answer on the same state). Shards are disjoint and reproducible,
so many workers (or Kubernetes pods) can each write their part of one dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .config import BENCHMARK_VERSION, EnvConfig
from .policies import Policy
from .scenario import ScenarioSpec
from .text_env import INSTRUCTIONS, FanetTextEnv, actions_to_json


def iter_records(
    specs: Iterable[ScenarioSpec],
    teacher: Policy,
    config: EnvConfig | None = None,
    decision_interval: int = 5,
) -> Iterator[dict[str, Any]]:
    """Yield one record per teacher decision, scenario by scenario, in order."""
    cfg = config or EnvConfig()
    for spec in specs:
        env = FanetTextEnv(scenario=spec, config=cfg, decision_interval=decision_interval)
        prompt, _ = env.reset()
        sim = env.sim
        assert sim is not None
        teacher.reset(spec)
        done = False
        while not done:
            actions = teacher.act(sim.obs, sim if teacher.privileged else None)
            answer = actions_to_json(actions, sim)
            t = sim.t
            next_prompt, reward, terminated, truncated, info = env.step(answer)
            yield {
                "benchmark": BENCHMARK_VERSION,
                "scenario_id": spec.scenario_id,
                "seed": spec.seed,
                "t": t,
                "teacher": teacher.name,
                "instructions": INSTRUCTIONS,
                "prompt": prompt,
                "completion": answer,
                "reward": reward,
                "n_invalid": info["n_invalid"],
            }
            prompt = next_prompt
            done = terminated or truncated


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write records to ``path`` (one JSON object per line); return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            count += 1
    return count
