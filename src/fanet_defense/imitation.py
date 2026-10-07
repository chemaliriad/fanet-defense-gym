"""Non-privileged demonstrations and masked actor behaviour cloning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from .config import EnvConfig
from .policies import Policy
from .scenario import ScenarioSpec
from .sim import FanetSim

if TYPE_CHECKING:
    from .ppo import ActorCritic


def collect_demonstrations(
    teacher: Policy, specs: Sequence[ScenarioSpec], cfg: EnvConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 ``[N, obs_dim]`` observations and int64 ``[N]`` actions.

    Like rollout_episode, retain all sim.n drones, including offline drones; empty
    neighbour slots are observation features, not extra agent rows.
    """
    observations, actions = [], []
    for spec in specs:
        sim = FanetSim(spec, cfg)
        teacher.reset(spec)
        while not sim.done:
            obs = sim.obs
            act = teacher.act(obs, None)
            observations.append(obs.copy())
            actions.append(np.asarray(act, dtype=np.int64).copy())
            sim.step(act)
    if not observations:
        return np.empty((0, cfg.obs_dim), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(observations).astype(np.float32), np.concatenate(actions)


def behaviour_clone(
    model: ActorCritic,
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Fit only the actor; report full-dataset loss and greedy agreement after fitting."""
    if len(obs) == 0 or actions.shape != (len(obs),):
        raise ValueError("need nonempty observations and one action per row")
    if epochs < 0 or batch_size < 1 or lr <= 0:
        raise ValueError("need epochs >= 0, batch_size >= 1 and lr > 0")
    device = next(model.parameters()).device
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    y = torch.as_tensor(actions, dtype=torch.int64, device=device)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    for _ in range(epochs):
        perm = rng.permutation(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(perm[start : start + batch_size], device=device)
            loss = F.cross_entropy(model.logits(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            logits = model.logits(x[start : start + batch_size])
            target = y[start : start + batch_size]
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            correct += int((logits.argmax(dim=1) == target).sum().item())
    return {"loss": loss_sum / len(obs), "agreement": correct / len(obs)}
