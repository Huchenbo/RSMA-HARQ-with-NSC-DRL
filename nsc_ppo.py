"""
nsc_ppo.py
==========

Neuro-symbolic constrained PPO (NSC-PPO) for energy-efficiency (EE)
maximisation in a cognitive satellite-terrestrial network with RSMA and HARQ.

The agent is PPO plus the two neuro-symbolic blocks of Section V-C of the
manuscript:

1. **Soft validity mask** -- the differentiable violation degree

       v_valid(a) = lam_sinr v_sinr + lam_op v_op + lam_aoi v_aoi/lambda_AoI
                    + lam_pow v_pow/P_max                               (27)

   added to the actor loss.  It biases the policy towards actions for which
   the target SINRs are achievable and the OP/AAoI/power budgets hold.

2. **Differentiable projection layer** -- I projected gradient steps on

       L_pen(a) = ||a - a_n||^2 + rho ( max(0, max_k P_out,k,T - delta_s)
                                       + max(0, AAoI - Delta_bar) )    (29)

   each followed by the closed-form repair (box clipping, common rescaling of
   the powers, rate lowering until both SINR inequalities hold) and, at the
   last step, by a feasibility restoration that backs the target rates off and,
   if needed, raises the powers until the outage budget is met.  The
   environment executes the projected action a*; the PPO ratio is formed with
   the raw action a, and the alignment loss keeps the two close:

       L_actor = -L_PPO + lam_valid v_valid(a) + lam_align ||a - a*||^2

The 21-dimensional action of (26) is *not* produced by 21 independent outputs.  The
actor emits a latent vector of dimension 2T + T + T(K+1) and a deterministic
"budget x softmax" map turns it into the action (see :class:`Actor`), which keeps every
stream strictly positive and keeps the transmit power inside the budget P_max by
construction.  Sampling independent powers instead violates the SINR feasibility
condition of (26-e) in most frames and the outage probability saturates at one.

Usage
-----
    python nsc_ppo.py                        # train (4e4 episodes, CPU)
    python nsc_ppo.py --episodes 300         # short run
    python nsc_ppo.py --op-budget 5e-2       # looser outage requirement
    python nsc_ppo.py --no-project           # ablation: closed-form repair only
    python nsc_ppo.py --no-restore           # ablation: PGD + repair, no restoration
    python nsc_ppo.py --eval runs/final/checkpoints/nsc_ppo.pt

Outputs
    runs/final/train_log.csv       one row every 10 episodes
    runs/final/curve.png           EE / OP / AAoI versus episode
    results/curve.png              same figure, kept in the repository
    runs/final/metrics.json        final evaluation (mean action and sampled)
    runs/final/checkpoints/{actor,critic}.pt  network weights (state_dict)
    runs/final/checkpoints/nsc_ppo.pt   full checkpoint (networks + optimisers)
    runs/final/checkpoints/{best,final}.pt   best-evaluated and last snapshot

Released under the MIT licence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from estn_env import EnvConfig, ESTNEnv

__all__ = ["Actor", "Critic", "NSCProjector", "NSCPPO", "TrainConfig"]


# --------------------------------------------------------------------------- #
# configuration (defaults follow Table I of the manuscript)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # --- PPO --------------------------------------------------------------- #
    gamma: float = 0.45              # discount factor
    gae_lambda: float = 0.95         # GAE lambda, eq. (31)
    clip: float = 0.2                # clipping parameter epsilon
    entropy: float = 0.01            # entropy bonus coefficient (initial)
    entropy_final: float = 1e-3      # annealed value, see Actor.distribution
    std_cap_final: float = -2.5      # log sigma reached at the last episode
    lr_actor: float = 3e-4           # Table I: 5e-5 (raised for short runs)
    lr_critic: float = 8e-4
    lr_final: float = 0.0            # cosine decay target, see train()
    lr_decay_frac: float = 0.4     # all schedules freeze at this fraction
    hidden: int = 128                # Table I: two hidden layers, 128 units each
    epochs: int = 4                  # optimisation epochs per PPO update
    minibatch: int = 64
    grad_clip: float = 0.5

    # --- training schedule -------------------------------------------------- #
    episodes: int = 40000             # Table I uses 4e4
    frames_per_episode: int = 16     # N_T
    rollout: int = 32                # transitions per PPO update
    seed: int = 0

    # --- neuro-symbolic layer ----------------------------------------------- #
    project: bool = True             # False -> closed-form repair only (ablation)
    restore: bool = True             # False -> no feasibility restoration (ablation)
    proj_steps: int = 3              # I in Algorithm 1
    proj_lr: float = 1.0             # eta, in normalised action units
    proj_rho: float = 1.0            # rho in (29)
    lam_valid: float = 1.5           # weight of the soft mask
    lam_align: float = 1.8           # weight of the alignment loss
    lam_sinr: float = 1.0            # weights inside (27)
    lam_op: float = 1.0
    lam_aoi: float = 1.0
    lam_pow: float = 1.0


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    """Gaussian policy with a structured, always-feasible output map.

    The network emits a latent vector ``z`` of dimension ``2T + T + T(K+1)``
    (6 + 3 + 15 = 24 for ``K = 4``, ``T = 3``) and the 21-dimensional action of
    (26) is a deterministic function of it:

        R_c,th,t, R_p,th,t = r_min + (r_max - r_min) * sigmoid(z)
        P_s,t              = p_max * sigmoid(z)               (power budget)
        alpha_t            = softmax(z_t) over the K+1 streams
        P_c,t, P_k,t       = P_s,t * alpha_t

    The shares sum to one, so every stream is strictly positive and the total
    power respects the budget by construction; this is the same
    "budget x softmax" parameterisation that is standard for power allocation.
    The effective number of degrees of freedom is ``2T + T + T*K = T(K+3)``,
    i.e. the action dimension of the manuscript.
    """

    def __init__(self, obs_dim: int, cfg, hidden: int = 128):
        super().__init__()
        self.T, self.K = cfg.T, cfg.K
        self.r_min, self.r_max = cfg.r_min, cfg.r_max
        self.p_max = cfg.p_max
        n_lat = 2 * cfg.T + cfg.T + cfg.T * (cfg.K + 1)
        self.n_latent = n_lat
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, n_lat)
        self.log_std = nn.Parameter(torch.full((n_lat,), -1.0))
        self.std_cap = 0.0                # annealed by the training loop

    def set_std_cap(self, cap: float) -> None:
        """Upper bound on ``log sigma``; annealed to ``-2.5`` (sigma = 0.08)."""
        self.std_cap = float(cap)

    def distribution(self, obs: Tensor) -> Normal:
        mu = self.mu(self.body(obs))
        # The standard deviation is capped from above so that the entropy bonus
        # cannot inflate the exploration noise indefinitely, and from below so
        # that the policy can become almost deterministic.  The transition of
        # this MDP is deterministic (the next state is a function of the
        # action), so the optimal policy is deterministic and sigma should
        # decay; the cap and the entropy coefficient are both annealed for the
        # same reason.
        std = self.log_std.clamp(-4.0, self.std_cap).exp().expand_as(mu)
        return Normal(mu, std.clamp(1e-3, 1.0))

    def to_action(self, z: Tensor) -> Tensor:
        """Deterministic map from the latent vector to the action of (26)."""
        T, K = self.T, self.K
        n_r = 2 * T
        rates = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(z[..., :n_r])
        total = self.p_max * torch.sigmoid(z[..., n_r:n_r + T])
        logits = z[..., n_r + T:].reshape(*z.shape[:-1], T, K + 1)
        share = torch.softmax(logits, dim=-1)
        pc = total * share[..., 0]
        pk = total.unsqueeze(-1) * share[..., 1:]
        return torch.cat([rates, pc, pk.reshape(*z.shape[:-1], T * K)], dim=-1)

    def forward(self, obs: Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        z = dist.mean if deterministic else dist.rsample()
        return self.to_action(z), z, dist

    def log_prob(self, z: Tensor, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Log-density of the latent vector (the action map is deterministic)."""
        dist = self.distribution(obs)
        return dist.log_prob(z).sum(-1), dist.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1)


# --------------------------------------------------------------------------- #
# neuro-symbolic layer
# --------------------------------------------------------------------------- #
class NSCProjector:
    """Soft validity mask and differentiable projection (Section V-C)."""

    def __init__(self, env: ESTNEnv, cfg: TrainConfig):
        self.env = env
        self.cfg = cfg
        self.lo = env.lo.to(env.dtype)
        self.hi = env.hi.to(env.dtype)
        self.span = (self.hi - self.lo)

    # --- normalised coordinates: makes the PGD step size scale-free -------- #
    def _to_z(self, a: Tensor) -> Tensor:
        return (a - self.lo) / self.span

    def _to_a(self, z: Tensor) -> Tensor:
        return self.lo + z * self.span

    # --- eq. (27) ---------------------------------------------------------- #
    def violation(self, a: Tensor) -> Dict[str, Tensor]:
        c = self.env.cfg
        v = self.env.violations(a)
        total = (self.cfg.lam_sinr * v["sinr"]
                 + self.cfg.lam_op * v["op"]
                 + self.cfg.lam_aoi * v["aoi"] / c.aaoi_budget
                 + self.cfg.lam_pow * v["pow"] / c.p_max)
        return {**v, "total": total}

    # --- penalty used by (29) ---------------------------------------------- #
    def _penalty(self, a: Tensor) -> Tensor:
        return self._penalty_ps(a).mean()

    def _penalty_ps(self, a: Tensor) -> Tensor:
        """Penalty of (29), one value per leading row of ``a`` (not averaged)."""
        c = self.env.cfg
        out = self.env.evaluate(a)
        v_op = (out["op_final"] - c.op_budget).clamp_min(0.0).mean(-1)
        v_aoi = (out["aaoi"] - c.aaoi_budget).clamp_min(0.0) / c.aaoi_budget
        return v_op + v_aoi

    # --- feasibility restoration ------------------------------------------- #
    def restore(self, a: Tensor, iters: int = 5) -> Tensor:
        """Bring the action back to the OP budget (optional, see README).

        Three monotone moves, all cheap:

        * spread a small fraction of the power uniformly over the ``K + 1``
          streams (renormalised, so the budget still holds exactly), which
          keeps every SINR condition satisfiable;
        * the outage probability is roughly inversely proportional to the
          transmit power, so one Newton-like step ``P <- P * (OP / delta_s)``
          (capped by the power budget) lifts a power-starved action back to the
          boundary of the feasible set;
        * the target rates are then bisected down to the largest common scale
          that keeps ``max_k OP_k,T <= delta_s``.  If the action already meets
          the budget, the rates are left untouched.

        This is *not* part of Algorithm 1 of the paper, where the OP/AAoI
        feasible set is only approached by the ``I`` projected gradient steps;
        it is a one-dimensional closed-form certification that makes the
        logged rows satisfy (26-b) exactly.  Disable it with ``--no-restore``.
        """
        env, c = self.env, self.env.cfg
        T, K = c.T, c.K
        a = env.repair(a)

        # (0) keep every stream alive without spending extra power: a stream
        #     that receives (almost) no power makes its SINR condition
        #     unsatisfiable at any rate and its outage saturates at one.  A
        #     small amount of power is spread uniformly over the K + 1 streams
        #     and the result is renormalised, so the budget of (26-d) still
        #     holds exactly.
        eps = 0.05
        pk = a[..., 3 * T:3 * T + T * K].reshape(*a.shape[:-1], T, K)
        pc = a[..., 2 * T:3 * T].unsqueeze(-1)                 # (..., T, 1)
        p_tot = pc + pk.sum(-1, keepdim=True)                  # (..., T, 1)
        mean_p = p_tot / (K + 1)
        pc = pc + eps * mean_p
        pk = pk + eps * mean_p
        scale = p_tot / (pc + pk.sum(-1, keepdim=True)).clamp_min(1e-9)
        pc = pc * scale
        pk = pk * scale
        a = torch.cat([a[..., :2 * T], pc.squeeze(-1),
                       pk.reshape(*a.shape[:-1], T * K)], dim=-1)
        a = env.repair(a)

        # (i) power step: only fires when the outage budget is missed
        #     ``op`` is a scalar for one action and ``(N,)`` for a batch of N,
        #     while the powers are ``(..., T)``, so the boost factor is built
        #     without an extra trailing axis to keep the two shapes aligned.
        op = torch.amax(env.evaluate(a)["op_final"], dim=-1)      # (...)
        pk = a[..., 3 * T:3 * T + T * K].reshape(*a.shape[:-1], T, K)
        pc = a[..., 2 * T:3 * T]                                  # (..., T)
        p_tot = pc + pk.sum(-1)                                   # (..., T)
        boost = torch.minimum((op / c.op_budget).clamp_min(1.0).unsqueeze(-1),
                              c.p_max / p_tot.clamp_min(1e-9))
        a = torch.cat([a[..., :2 * T], pc * boost,
                       (pk * boost.unsqueeze(-1)).reshape(*a.shape[:-1], T * K)],
                      dim=-1)
        a = env.repair(a)

        # The boost above is the identity whenever the action already meets the
        # budget, and in that case the back-off below returns ``s = 1`` anyway.
        # Returning here skips the five bisection evaluations for every
        # feasible action, which is the great majority of them near
        # convergence; the returned tensor is the same one.
        if bool(torch.all(op <= c.op_budget)):
            return env.repair(a, r_lo=0.0)

        # (ii) rate back-off: the largest common scale ``s in (0, 1]`` with
        #      max_k OP_k,T(a; s R) <= delta_s.  OP is monotone in the target
        #      rate, so ``lo`` always stays feasible and ``hi`` infeasible; the
        #      current rates (s = 1) are kept whenever they already satisfy the
        #      budget -- scaling them down unconditionally would collapse the
        #      policy to R = 0.
        full_ok = torch.amax(env.evaluate(a)["op_final"], dim=-1) <= c.op_budget
        lo = torch.zeros(a.shape[:-1], dtype=a.dtype, device=a.device)
        hi = torch.ones_like(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            trial = torch.cat([a[..., :2 * T] * mid.unsqueeze(-1), a[..., 2 * T:]],
                              dim=-1)
            ok = torch.amax(env.evaluate(trial)["op_final"], dim=-1) <= c.op_budget
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        s = torch.where(full_ok, torch.ones_like(lo), lo)
        # rates may fall below the policy range here: (26-f) only asks for R >= 0
        return env.repair(torch.cat([a[..., :2 * T] * s.unsqueeze(-1), a[..., 2 * T:]],
                                    dim=-1), r_lo=0.0)

    # --- eq. (28) ---------------------------------------------------------- #
    def project(self, a: Tensor) -> Tensor:
        """Projected gradient descent on (28) with backtracking.

        A step is kept only if the penalised objective decreases, so the OP and
        AAoI violations never grow during the unrolled iterations.  Every
        iterate goes through the closed-form repair, hence the box, power and
        SINR conditions hold at every step.  The line search walks down a
        geometric ladder of step sizes: the proximal term ``||a - a_n||^2``
        and the penalty have very different scales (the former is in
        normalised action units, the latter is a probability), so a fixed
        step size either overshoots and is rejected, or crawls.
        """
        env = self.env
        z0 = self._to_z(env.repair(a))
        z = z0
        rho = self.cfg.proj_rho
        ladder = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003)

        def objective(zz: Tensor) -> Tensor:
            return ((zz - z0) ** 2).sum(-1).mean() + rho * self._penalty(self._to_a(zz))

        def objective_rows(zz: Tensor) -> Tensor:
            """Same objective, one value per row instead of the batch mean."""
            return ((zz - z0) ** 2).sum(-1) + rho * self._penalty_ps(self._to_a(zz))

        best = float(objective(z))
        for _ in range(self.cfg.proj_steps):
            z_lin = z.detach().requires_grad_(True)
            grad = torch.autograd.grad(objective(z_lin), z_lin)[0].detach()
            # The whole ladder is evaluated as one batch: the six trial steps
            # differ only in the step size, so batching them keeps the search
            # order of the original loop (the first improving step wins) while
            # cutting six evaluations of ``evaluate`` down to one.
            cand = torch.stack([
                self._to_z(env.repair(self._to_a(
                    (z - self.cfg.proj_lr * scale * grad).clamp(0.0, 1.0))))
                for scale in ladder])
            with torch.no_grad():
                value = objective_rows(cand).detach()
            ok = value < best - 1e-12
            if not bool(ok.any()):
                break
            j = int(torch.nonzero(ok, as_tuple=False)[0])
            z, best = cand[j], float(value[j])
        a_out = self._to_a(z)
        # Algorithm 1 stops at the closed-form repair; the optional restoration
        # below additionally certifies max_k OP_k,T <= delta_s (see README).
        # ``r_lo = 0`` because (26-f) only requires R >= 0: when the power share
        # of a stream is small, the SINR cap can fall below R_min, and clamping
        # back to R_min would make that stream unable to decode at any power.
        return (self.restore(a_out) if self.cfg.restore
                else env.repair(a_out, r_lo=0.0))


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
class NSCPPO:
    def __init__(self, env: ESTNEnv, cfg: TrainConfig, device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.actor = Actor(env.obs_dim, env.cfg, cfg.hidden).to(self.device)
        self.critic = Critic(env.obs_dim, cfg.hidden).to(self.device)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.projector = NSCProjector(env, cfg)
        self.ent_coef = cfg.entropy          # annealed by the training loop

        keys = ("obs", "nxt", "z", "a", "astar", "logp", "r")
        self.buf: Dict[str, List[Tensor]] = {k: [] for k in keys}
        self.last_raw = torch.zeros(env.act_dim, dtype=env.dtype)

    # --- interaction -------------------------------------------------------- #
    def constrain(self, a_raw: Tensor) -> Tensor:
        """Map a raw action onto the constraint set.

        With ``cfg.project`` this is the neuro-symbolic layer of Section V-C
        (soft mask + projection + repair).  Without it, the symbolic part is
        still applied -- the closed-form repair of the box, the power budget
        and the SINR feasibility -- so that ``--no-project`` isolates the
        *differentiable projection* and not the whole constraint handling.
        """
        if self.cfg.project:
            with torch.enable_grad():
                return self.projector.project(a_raw)
        return self.env.repair(a_raw, r_lo=0.0)

    def act(self, obs: Tensor, deterministic: bool = False):
        """Return ``(a_raw, a_star, z, logp)``; ``a_star`` is what runs."""
        with torch.no_grad():
            a_raw, z, _ = self.actor(obs, deterministic)
            logp, _ = self.actor.log_prob(z, obs)
        a_star = self.constrain(a_raw)
        return a_raw.detach(), a_star.detach(), z.detach(), logp.detach()

    def collect(self, obs: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        cfg = self.cfg
        acc: Dict[str, float] = {}
        for _ in range(cfg.frames_per_episode):
            a_raw, a_star, z, logp = self.act(obs)
            nxt, r, info = self.env.step(a_star)
            self.buf["obs"].append(obs)
            self.buf["nxt"].append(nxt)
            self.buf["z"].append(z)
            self.buf["a"].append(a_raw)
            self.buf["astar"].append(a_star)
            self.buf["logp"].append(logp)
            self.buf["r"].append(r)
            self.last_raw = a_raw
            for k, v in info.items():
                acc[k] = acc.get(k, 0.0) + v
            obs = nxt
        return obs, {k: v / cfg.frames_per_episode for k, v in acc.items()}

    # --- PPO update --------------------------------------------------------- #
    def update(self) -> Dict[str, float]:
        cfg = self.cfg
        obs = torch.stack(self.buf["obs"]).to(self.device)
        nxt = torch.stack(self.buf["nxt"]).to(self.device)
        z = torch.stack(self.buf["z"]).to(self.device)
        a = torch.stack(self.buf["a"]).to(self.device)
        astar = torch.stack(self.buf["astar"]).to(self.device)
        logp_old = torch.stack(self.buf["logp"]).to(self.device)
        r = torch.tensor(self.buf["r"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            v = self.critic(obs)
            delta = r + cfg.gamma * self.critic(nxt) - v
            adv = torch.zeros_like(r)
            running = torch.zeros((), dtype=r.dtype, device=r.device)
            for t in reversed(range(len(r))):
                running = delta[t] + cfg.gamma * cfg.gae_lambda * running
                adv[t] = running
            returns = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(r)
        order = np.arange(n)
        stats = dict(pi_loss=0.0, v_loss=0.0, v_valid=0.0, align=0.0)
        n_mb = 0
        for _ in range(cfg.epochs):
            np.random.shuffle(order)
            for start in range(0, n, cfg.minibatch):
                mb = torch.as_tensor(order[start:start + cfg.minibatch], device=self.device)
                logp, ent = self.actor.log_prob(z[mb], obs[mb])
                ratio = torch.exp(torch.clamp(logp - logp_old[mb], -20.0, 20.0))
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                viol = self.projector.violation(a[mb])["total"].mean()
                align = (((a[mb] - astar[mb]) / self.projector.span) ** 2).sum(-1).mean()
                loss = (pi_loss + cfg.lam_valid * viol + cfg.lam_align * align
                        - self.ent_coef * ent.mean())

                self.opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
                self.opt_a.step()

                v_loss = nn.functional.mse_loss(self.critic(obs[mb]), returns[mb])
                self.opt_c.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                self.opt_c.step()

                stats["pi_loss"] += float(pi_loss)
                stats["v_loss"] += float(v_loss)
                stats["v_valid"] += float(viol)
                stats["align"] += float(align)
                n_mb += 1

        for k in stats:
            stats[k] /= max(1, n_mb)
        for key in self.buf:
            self.buf[key].clear()
        return stats

    # --- evaluation --------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(self, episodes: int = 4, deterministic: bool = True) -> Dict[str, float]:
        """Average metrics over whole frames.

        ``deterministic=True`` runs the mean of the Gaussian policy,
        ``deterministic=False`` samples from it.  Both are reported, because
        the MDP transition is deterministic and a well-converged policy should
        give nearly the same numbers either way; a large gap means that the
        exploration noise has not yet decayed.
        """
        acc: Dict[str, float] = {}
        for _ in range(episodes):
            obs = self.env.reset()
            for _ in range(self.cfg.frames_per_episode):
                a, _, _ = self.actor(obs, deterministic)
                a = self.constrain(a)
                obs, _, info = self.env.step(a.detach())
                for k, v in info.items():
                    acc[k] = acc.get(k, 0.0) + v
        n = episodes * self.cfg.frames_per_episode
        return {k: v / n for k, v in acc.items()}

    # --- persistence -------------------------------------------------------- #
    def save(self, path: str, episode: int) -> None:
        base = os.path.dirname(path) or "."
        os.makedirs(base, exist_ok=True)
        torch.save({"episode": episode,
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "opt_actor": self.opt_a.state_dict(),
                    "opt_critic": self.opt_c.state_dict(),
                    "std_cap": self.actor.std_cap,
                    "ent_coef": self.ent_coef,
                    "train_cfg": asdict(self.cfg),
                    "env_cfg": asdict(self.env.cfg)}, path)
        torch.save(self.actor.state_dict(), os.path.join(base, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(base, "critic.pt"))

    def load(self, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:                                  # torch < 1.13
            ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        # the exploration noise is annealed, so the effective sigma at the end of
        # training is part of the model: restore it, otherwise the sampled
        # evaluation uses the un-annealed value and looks much worse
        self.actor.set_std_cap(ckpt.get("std_cap", 0.0))
        self.ent_coef = ckpt.get("ent_coef", self.cfg.entropy_final)
        if "opt_actor" in ckpt:
            self.opt_a.load_state_dict(ckpt["opt_actor"])
            self.opt_c.load_state_dict(ckpt["opt_critic"])


# --------------------------------------------------------------------------- #
# training / evaluation entry points
# --------------------------------------------------------------------------- #
FIELDS = ["episode", "reward", "ee", "op_max", "aaoi", "power", "v_sinr",
          "v_op", "v_aoi", "pi_loss", "v_loss", "align", "log_std", "seconds"]


def _rolling(y: np.ndarray, w: int):
    """Centred moving mean/std over ``w`` log points (the "valid" part only)."""
    w = max(1, min(w, len(y)))
    k = np.ones(w) / w
    mean = np.convolve(y, k, "valid")
    var = np.convolve(y * y, k, "valid") - mean ** 2
    idx = np.arange(len(mean)) + w // 2
    return mean, np.sqrt(np.maximum(var, 0.0)), idx


def plot_curves(log_path: str, out_dirs, window: int = 10,
                op_budget: float = None, aaoi_budget: float = None) -> None:
    """Convergence figure built from ``train_log.csv``.

    Each panel shows the raw log value as a thin trace, a centred moving average
    over ``window`` log points as a thick line, and the ``+-1 std`` spread of the
    samples inside the same window as a band.  The band is what tells a genuine
    plateau apart from a flat line drawn through noisy data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.genfromtxt(log_path, delimiter=",", names=True)
    ep = np.asarray(d["episode"], dtype=float)
    op_budget = EnvConfig().op_budget if op_budget is None else op_budget
    aaoi_budget = EnvConfig().aaoi_budget if aaoi_budget is None else aaoi_budget

    plt.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.0))
    panels = (("ee", "EE (bit/J)", None),
              ("op_max", "max OP", op_budget),
              ("aaoi", "AAoI (s)", aaoi_budget))
    for j, (key, label, budget) in enumerate(panels):
        y = np.asarray(d[key], dtype=float)
        m, s, ix = _rolling(y, window)
        ax[j].fill_between(ep[ix], m - s, m + s, color="#f6b26b", alpha=0.35,
                           lw=0, label=r"$\pm1$ s.d.")
        ax[j].plot(ep, y, lw=0.5, color="#9ecae1", alpha=0.55, label="raw")
        ax[j].plot(ep[ix], m, lw=1.9, color="#c0392b",
                   label=f"moving average ({window * 10} ep)")
        if budget is not None:
            ax[j].axhline(budget, ls="--", c="k", lw=0.9)
            ax[j].set_ylim(0.0, 1.25 * budget)
            ax[j].text(0.985, budget * 1.03, "budget", ha="right", va="bottom",
                       transform=ax[j].get_yaxis_transform(), fontsize=7.5)
        ax[j].set_xlim(0, ep[-1] * 1.01)
        ax[j].set_xlabel("episode")
        ax[j].set_ylabel(label)
        ax[j].set_title(f"({chr(97 + j)})", fontsize=9, loc="left")
        ax[j].grid(alpha=0.25, lw=0.5)
    ax[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    for path in out_dirs:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=200)
        print("saved", path)


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    env_cfg = EnvConfig(op_budget=args.op_budget, aaoi_budget=args.aaoi_budget)
    env = ESTNEnv(env_cfg, device="cpu")
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed,
                      project=not args.no_project, restore=not args.no_restore,
                      lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                      lr_final=args.lr_final, rollout=args.rollout,
                      epochs=args.epochs, lr_decay_frac=args.decay_frac)
    agent = NSCPPO(env, cfg, device="cpu")

    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(env.describe())
    print(f"{cfg.episodes} episodes x {cfg.frames_per_episode} frames | "
          f"projection={'on' if cfg.project else 'off'} | "
          f"lr=({cfg.lr_actor:g}, {cfg.lr_critic:g})")

    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(FIELDS)

    obs = env.reset()
    best_ee = -float("inf")
    stats = dict(pi_loss=0.0, v_loss=0.0, v_valid=0.0, align=0.0)
    # A single episode is only ``frames_per_episode`` frames long, so its return
    # carries a large sampling variance.  The logged point is therefore the mean
    # over the whole logging interval, not the last episode of it.
    acc = dict(ee=0.0, op_max=0.0, aaoi=0.0, power=0.0)
    n_acc = 0
    t0 = time.time()
    for ep in range(1, cfg.episodes + 1):
        progress = ep / cfg.episodes
        # All three schedules (learning rate, exploration noise, entropy bonus)
        # reach their final value at ``lr_decay_frac`` of the budget.  After that
        # point the weights and the noise level are frozen, so the policy sits on
        # a fixed point and the training curve is flat by construction; annealing
        # them over the whole run only keeps the sampled return creeping upwards.
        frac = min(1.0, progress / cfg.lr_decay_frac)
        cos = 0.5 * (1.0 + np.cos(np.pi * frac))
        agent.ent_coef = cfg.entropy_final + (cfg.entropy - cfg.entropy_final) * cos
        agent.actor.set_std_cap(cfg.std_cap_final * frac)
        for opt, lr0 in ((agent.opt_a, cfg.lr_actor), (agent.opt_c, cfg.lr_critic)):
            lr = cfg.lr_final + (lr0 - cfg.lr_final) * cos
            for group in opt.param_groups:
                group["lr"] = lr
        obs, info = agent.collect(obs)
        if len(agent.buf["r"]) >= cfg.rollout:
            stats = agent.update()
        for k in acc:
            acc[k] += info[k]
        n_acc += 1

        if ep % 10 == 0 or ep == 1:
            avg = {k: v / n_acc for k, v in acc.items()}
            acc = {k: 0.0 for k in acc}
            n_acc = 0
            viol = agent.projector.violation(agent.last_raw)
            log_std = float(torch.clamp(agent.actor.log_std.mean(),
                                        max=agent.actor.std_cap))
            row = [ep, avg["ee"] / env_cfg.reward_scale, avg["ee"], avg["op_max"],
                   avg["aaoi"], avg["power"], float(viol["sinr"]),
                   float(viol["op"]), float(viol["aoi"]),
                   stats["pi_loss"], stats["v_loss"], stats["align"],
                   log_std, time.time() - t0]
            with open(log_path, "a", newline="") as fh:
                csv.writer(fh).writerow([round(x, 6) if isinstance(x, float) else x
                                         for x in row])
            print(f"ep {ep:5d} | r {row[1]:6.3f} | EE {row[2]:9.1f} | "
                  f"OP {row[3]:.2e} | AAoI {row[4]:.4f} | "
                  f"pi {row[9]:+.4f} | V {row[10]:.4f} | "
                  f"ls {row[12]:+.2f} | {row[13]:6.1f}s")

        if ep % 100 == 0:
            agent.save(os.path.join(ckpt_dir, "nsc_ppo.pt"), ep)
            ev = agent.evaluate(episodes=2)
            if ev.get("ee", 0.0) > best_ee:
                best_ee = ev["ee"]
                agent.save(os.path.join(ckpt_dir, "best.pt"), ep)

    agent.save(os.path.join(ckpt_dir, "final.pt"), cfg.episodes)
    ev = agent.evaluate(episodes=4, deterministic=True)
    evs = agent.evaluate(episodes=4, deterministic=False)
    print("final evaluation (mean action) :", {k: round(v, 6) for k, v in ev.items()})
    print("final evaluation (sampled)     :", {k: round(v, 6) for k, v in evs.items()})
    print("final log_std (effective):", round(float(torch.clamp(
        agent.actor.log_std.mean(), max=agent.actor.std_cap)), 3))
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"eval_mean_action": ev, "eval_sampled": evs,
                   "best_ee": best_ee, "train": asdict(cfg),
                   "env": asdict(env_cfg)}, fh, indent=2)

    if not args.no_plot:
        try:
            plot_curves(log_path,
                        [os.path.join(args.results, "curve.png"),
                         os.path.join(args.out, "curve.png")],
                        window=args.smooth,
                        op_budget=env_cfg.op_budget,
                        aaoi_budget=env_cfg.aaoi_budget)
        except Exception as exc:                            # pragma: no cover
            print("plot skipped:", exc)


def evaluate(args: argparse.Namespace) -> None:
    env = ESTNEnv(EnvConfig(op_budget=args.op_budget, aaoi_budget=args.aaoi_budget))
    cfg = TrainConfig(project=not args.no_project, restore=not args.no_restore)
    agent = NSCPPO(env, cfg, device="cpu")
    agent.load(args.eval)
    print(env.describe())
    n = args.eval_episodes
    print("mean action :", {k: round(v, 6)
                           for k, v in agent.evaluate(n, True).items()})
    print("sampled     :", {k: round(v, 6)
                           for k, v in agent.evaluate(n, False).items()})


def main() -> None:
    p = argparse.ArgumentParser(description="NSC-PPO for CSTN with RSMA/HARQ")
    p.add_argument("--episodes", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--op-budget", type=float, default=1e-2)
    p.add_argument("--aaoi-budget", type=float, default=1e-2)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=8e-4)
    p.add_argument("--lr-final", type=float, default=0.0)
    p.add_argument("--rollout", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--decay-frac", type=float, default=0.4,
                   help="fraction of the run at which lr/entropy/std are frozen")
    p.add_argument("--threads", type=int, default=1,
                   help="torch CPU threads; 1 is fastest for these tensor sizes")
    p.add_argument("--out", type=str, default="runs/final")
    p.add_argument("--no-project", action="store_true",
                   help="ablation: closed-form repair only, no projection layer")
    p.add_argument("--no-restore", action="store_true",
                   help="stop at the closed-form repair (Algorithm 1 as written)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--results", type=str, default="results",
                   help="directory for the figures")
    p.add_argument("--smooth", type=int, default=10,
                   help="moving-average window, in log points (10 episodes each)")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the figure from an existing train_log.csv")
    p.add_argument("--eval", type=str, default="", help="checkpoint to evaluate")
    p.add_argument("--eval-episodes", type=int, default=8)
    args = p.parse_args()
    if args.plot_only:
        plot_curves(os.path.join(args.out, "train_log.csv"),
                    [os.path.join(args.results, "curve.png"),
                     os.path.join(args.out, "curve.png")], window=args.smooth,
                    op_budget=args.op_budget, aaoi_budget=args.aaoi_budget)
    else:
        evaluate(args) if args.eval else train(args)


if __name__ == "__main__":
    main()
