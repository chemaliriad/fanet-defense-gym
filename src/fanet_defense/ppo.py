"""PPO with parameter sharing across drones (independent PPO, "IPPO").

One actor-critic network is shared by every drone of every scenario, so it transfers to
swarms of different sizes. Rollouts run in worker processes with a numpy copy of the
weights (see :mod:`fanet_defense.rollout`); this module only holds the torch learner.

The implementation follows the usual recipe (GAE, clipped surrogate, advantage
normalisation, entropy bonus, linear learning-rate decay, gradient clipping), in the spirit
of CleanRL's single-file PPO. Time is part of the observation and episodes have a fixed
horizon, so the value after the last step is exactly zero and no truncation correction is
needed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import EnvConfig
from .evaluate import evaluate
from .imitation import behaviour_clone, collect_demonstrations
from .policies import Policy, WatchdogPolicy
from .rollout import NumpyPPOPolicy, Params, rollout_episode
from .scenario import ScenarioSampler, ScenarioSpec, split_seeds


@dataclass(frozen=True)
class PPOConfig:
    total_env_steps: int = 300_000
    episodes_per_iter: int = 16
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 5e-4
    epochs: int = 4
    minibatch: int = 8192
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden: int = 64
    eval_every_iters: int = 5
    workers: int = 4
    torch_threads: int = 2  # leave the other cores to the rollout workers
    # Credit assignment (training only): each agent learns from
    # local_weight * its own share of the team reward + (1 - local_weight) * team reward.
    # The team reward is the mean of the local shares, so the sum over agents is unchanged.
    local_weight: float = 0.5
    eval_sampled: bool = True  # also evaluate the stochastic policy (chosen on val)
    bc_scenarios: int = 0
    bc_epochs: int = 10
    bc_lr: float = 1e-3


def _layer(i: int, o: int, std: float = float(np.sqrt(2))) -> nn.Linear:
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, cfg: EnvConfig, hidden: int, gamma: float) -> None:
        super().__init__()
        d, a = cfg.obs_dim, cfg.n_actions
        self.k = cfg.k_neighbors
        self.vscale = 1.0 / (1.0 - gamma)  # returns live in [0, ~1/(1-gamma)]
        self.actor = nn.Sequential(
            _layer(d, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, a, 0.01)
        )
        self.critic = nn.Sequential(
            _layer(d, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0)
        )

    def mask(self, obs: torch.Tensor) -> torch.Tensor:
        slots = obs[:, 3:].reshape(obs.shape[0], self.k, 4)
        valid_block = (slots[..., 0] > 0.5) & (slots[..., 3] < 0.5)
        ones = torch.ones(obs.shape[0], 1, dtype=torch.bool, device=obs.device)
        return torch.cat([ones, valid_block, ones], dim=1)

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).masked_fill(~self.mask(obs), -1e9)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1) * self.vscale


def export_params(model: ActorCritic) -> Params:
    """Weights as plain numpy arrays in (in, out) layout, for rollout workers and ``.npz``."""
    out: Params = {"vscale": np.asarray(model.vscale)}
    for prefix, net in (("a", model.actor), ("c", model.critic)):
        linears = [m for m in net if isinstance(m, nn.Linear)]
        for j, lin in enumerate(linears):
            out[f"{prefix}{j}.w"] = lin.weight.detach().numpy().T.astype(np.float64).copy()
            out[f"{prefix}{j}.b"] = lin.bias.detach().numpy().astype(np.float64).copy()
    return out


def gae(
    rew: np.ndarray, val: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation on ``(T, N)`` arrays; the value after step T is 0."""
    t_max = rew.shape[0]
    adv = np.zeros_like(val)
    last = np.zeros(val.shape[1], dtype=val.dtype)
    for t in range(t_max - 1, -1, -1):
        nxt = val[t + 1] if t + 1 < t_max else 0.0
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def collate(
    episodes: Sequence[dict[str, Any]], gamma: float, lam: float, local_weight: float = 0.0
) -> dict[str, np.ndarray]:
    """Concatenate episodes along the agent axis and compute advantages and returns.

    Each agent's training reward is ``local_weight * local + (1 - local_weight) * team``.
    """
    obs = np.concatenate([e["obs"] for e in episodes], axis=1)
    act = np.concatenate([e["act"] for e in episodes], axis=1)
    logp = np.concatenate([e["logp"] for e in episodes], axis=1)
    val = np.concatenate([e["val"] for e in episodes], axis=1)
    team = np.concatenate([np.repeat(e["rew"][:, None], e["n"], axis=1) for e in episodes], axis=1)
    if local_weight > 0.0:
        local = np.concatenate([e["rew_local"] for e in episodes], axis=1)
        rew = local_weight * local + (1.0 - local_weight) * team
    else:
        rew = team
    adv, ret = gae(rew, val, gamma, lam)
    flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
    return {
        "obs": flat(obs),
        "act": flat(act),
        "logp": flat(logp),
        "adv": flat(adv),
        "ret": flat(ret),
    }


def ppo_update(
    model: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    pcfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    obs = torch.as_tensor(batch["obs"])
    act = torch.as_tensor(batch["act"])
    old_logp = torch.as_tensor(batch["logp"])
    adv = torch.as_tensor(batch["adv"])
    ret = torch.as_tensor(batch["ret"])
    size = obs.shape[0]
    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
    n_mb = 0
    stop = False
    for _ in range(pcfg.epochs):
        perm = torch.as_tensor(rng.permutation(size))
        for start in range(0, size, pcfg.minibatch):
            idx = perm[start : start + pcfg.minibatch]
            dist = torch.distributions.Categorical(logits=model.logits(obs[idx]))
            new_logp = dist.log_prob(act[idx])
            ratio = (new_logp - old_logp[idx]).exp()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - pcfg.clip, 1 + pcfg.clip)).mean()
            vf = 0.5 * ((model.value(obs[idx]) - ret[idx]) ** 2).mean() / model.vscale**2
            ent = dist.entropy().mean()
            loss = pg + pcfg.vf_coef * vf - pcfg.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - (ratio.log())).mean().item()
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += ent.item()
                stats["kl"] += kl
                stats["clipfrac"] += ((ratio - 1).abs() > pcfg.clip).float().mean().item()
            n_mb += 1
            if kl > 1.5 * pcfg.target_kl:
                stop = True
                break
        if stop:
            break
    return {k: v / max(1, n_mb) for k, v in stats.items()}


def train_ppo(
    pcfg: PPOConfig,
    cfg: EnvConfig,
    sampler: ScenarioSampler,
    seed: int,
    val_specs: Sequence[ScenarioSpec] = (),
    master_seed: int = 0,
    executor: Executor | None = None,
    log: Callable[[dict[str, float]], None] | None = None,
    teacher: Policy | None = None,
) -> tuple[NumpyPPOPolicy, list[dict[str, float]]]:
    """Train on fresh scenarios, optionally cloning a teacher before PPO updates."""
    torch.set_num_threads(pcfg.torch_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 4321])
    model = ActorCritic(cfg, pcfg.hidden, pcfg.gamma)
    opt = torch.optim.Adam(model.parameters(), lr=pcfg.lr, eps=1e-5)
    steps_per_iter = pcfg.episodes_per_iter * sampler.horizon
    n_iters = max(1, int(np.ceil(pcfg.total_env_steps / steps_per_iter)))
    train_seeds = split_seeds(master_seed * 1000 + seed, "train", n_iters * pcfg.episodes_per_iter)

    own_pool = executor is None
    pool: Executor = executor or ProcessPoolExecutor(max_workers=pcfg.workers)
    curve: list[dict[str, float]] = []
    t0 = time.perf_counter()
    try:
        if pcfg.bc_scenarios > 0:
            # Separate namespace: neither scenario collection nor BC shuffling advances
            # the PPO scenario/action/minibatch streams.
            bc_rng = np.random.default_rng([master_seed, seed, 0xBC])
            bc_specs = [
                sampler.sample(int(s)) for s in bc_rng.integers(0, 2**62, size=pcfg.bc_scenarios)
            ]
            obs, actions = collect_demonstrations(
                teacher if teacher is not None else WatchdogPolicy(cfg), bc_specs, cfg
            )
            bc_stats = behaviour_clone(
                model, obs, actions, pcfg.bc_epochs, pcfg.bc_lr, pcfg.minibatch, seed
            )
            params = export_params(model)
            row = {
                "env_steps": 0.0,
                "bc_env_steps": float(sum(s.horizon for s in bc_specs)),
                "bc_agreement": bc_stats["agreement"],
                "val_return": float("nan"),
                "val_return_sampled": float("nan"),
            }
            if val_specs:
                res = evaluate(NumpyPPOPolicy(params, cfg), val_specs, cfg, executor=pool)
                row["val_return"] = float(np.mean([r.ret for r in res]))
                sampled = NumpyPPOPolicy(params, cfg, deterministic=False, seed=seed)
                res = evaluate(sampled, val_specs, cfg, executor=pool)
                row["val_return_sampled"] = float(np.mean([r.ret for r in res]))
            curve.append(row)
            if log:
                log(row)
        for it in range(n_iters):
            opt.param_groups[0]["lr"] = pcfg.lr * max(0.1, 1.0 - it / n_iters)
            params = export_params(model)
            chunk = train_seeds[it * pcfg.episodes_per_iter : (it + 1) * pcfg.episodes_per_iter]
            jobs = [
                (params, sampler.sample(s).to_dict(), cfg, int(rng.integers(2**31))) for s in chunk
            ]
            episodes = list(pool.map(rollout_episode, jobs))
            batch = collate(episodes, pcfg.gamma, pcfg.gae_lambda, pcfg.local_weight)
            stats = ppo_update(model, opt, batch, pcfg, rng)
            row = {
                "env_steps": float((it + 1) * steps_per_iter),
                "train_return": float(np.mean([e["ret"] for e in episodes])),
                "wall_s": time.perf_counter() - t0,
                **stats,
            }
            if val_specs and ((it + 1) % pcfg.eval_every_iters == 0 or it == n_iters - 1):
                params = export_params(model)
                res = evaluate(NumpyPPOPolicy(params, cfg), val_specs, cfg, executor=pool)
                row["val_return"] = float(np.mean([r.ret for r in res]))
                if pcfg.eval_sampled:
                    sampled = NumpyPPOPolicy(params, cfg, deterministic=False, seed=seed)
                    res = evaluate(sampled, val_specs, cfg, executor=pool)
                    row["val_return_sampled"] = float(np.mean([r.ret for r in res]))
            curve.append(row)
            if log:
                log(row)
    finally:
        if own_pool:
            pool.shutdown()
    return NumpyPPOPolicy(export_params(model), cfg, name="ppo"), curve
