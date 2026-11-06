"""
estn_env.py
===========

Cognitive satellite-terrestrial network (CSTN) environment with rate-splitting
multiple access (RSMA) and hybrid ARQ (HARQ), written for the NSC-PPO agent in
``nsc_ppo.py``.  The environment is a *statistical-CSI* environment: the
satellite knows the channel statistics (large-scale gains and the Nakagami-m
small-scale parameters) but not the instantaneous channel, so every quantity
that the agent sees is obtained from the closed-form outage probability.

Frame structure
---------------
One MDP step = one frame = ``T`` HARQ rounds.  A frame-level action allocates,
for every round ``t``,

    [R_c,th,t, R_p,th,t, P_c,t, P_k,t for k = 1..K]

so the action dimension is ``T * (K + 3)``.  The number of rounds that are
actually executed is random: a user that decodes successfully in round ``t`` has
its powers and rates set to zero in the following rounds, which is the
"wait-for-all" HARQ of the paper.

Everything in :meth:`ESTNEnv.evaluate` is written in PyTorch and is
differentiable with respect to the action, because the projection layer of
NSC-PPO differentiates through the closed-form OP and AAoI.

Equations of the manuscript implemented here
--------------------------------------------
(8)-(9)   : SINR of the common and private streams
(15)      : exact outage probability (triple sum, Pochhammer symbols)
(23)      : average age of information (wait-for-all)
(24)-(25) : effective throughput and average power
(26)      : constraints (OP budget, AAoI budget, power budget)
(27)      : soft validity mask (violation degrees)

Only the public API of :class:`ESTNEnv` is used by the agent:

    env.reset()          -> state, shape (T*(K+3) + K + 1,)
    env.evaluate(a)      -> dict of differentiable tensors
    env.step(a)          -> (next_state, reward, info)
    env.violations(a)    -> dict of violation degrees, eq. (27)
    env.repair(a)        -> closed-form feasible repair

Author: <your name>.  Released under the MIT licence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["EnvConfig", "ESTNEnv"]


# --------------------------------------------------------------------------- #
# small numerical helpers
# --------------------------------------------------------------------------- #
def _besselj(nu: int, x: float, terms: int = 50) -> float:
    """Bessel function of the first kind J_nu(x) from its power series.

    Only used once, to evaluate the satellite antenna gain of the paper; keeping
    it here avoids a SciPy dependency.
    """
    total = 0.0
    for m in range(terms):
        term = (-1) ** m / (math.factorial(m) * math.gamma(m + nu + 1))
        total += term * (x / 2.0) ** (2 * m + nu)
    return total


def _pochhammer_1m(m_s: int, n: int) -> float:
    """(1 - m_s)_n, the Pochhammer symbol of the outage expression (15)."""
    value = 1.0
    for j in range(n):
        value *= (1 - m_s + j)
    return value


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Numerical setting of the simulation (Section VI of the manuscript)."""

    # --- system ---------------------------------------------------------- #
    K: int = 4                       # number of secondary users
    T: int = 3                       # HARQ rounds per frame
    bandwidth: float = 50e6          # B, Hz
    slot: float = 5e-3               # T_slot, s
    p_max: float = 3500.0            # satellite power budget, W  (eq. (26-d))
    p_bs: float = 10.0               # base-station transmit power, W
    r_min: float = 0.1               # lower bound of the target rates, bit/s/Hz
    r_max: float = 3.0               # upper bound of the target rates, bit/s/Hz

    # --- requirements ---------------------------------------------------- #
    op_budget: float = 1e-2          # delta_s    in (26-b)
    aaoi_budget: float = 1e-2        # Delta_bar  in (26-c), s

    # --- impairment ------------------------------------------------------ #
    sic_residual: float = 0.0        # Xi_k, residual SIC factor
    err_s: float = 0.0               # sigma_{S,k}, CSI error on the satellite link
    err_b: float = 0.0               # sigma_{B,k}, CSI error on the terrestrial link

    # --- satellite link (Shadowed-Rician, eq. (2)) ------------------------ #
    b_s: float = 0.063
    m_s: int = 1
    omega_s: float = 7e-4

    # --- terrestrial link (Nakagami-m, eq. (5)) --------------------------- #
    m_g: int = 1
    omega_g: float = 1.0

    # --- geometry --------------------------------------------------------- #
    # The distances enter the large-scale gain (1)-(2).  The pair (600, 10) is
    # the one used by the original simulator (CODE_SIM: ``ss = [600]*K``,
    # ``D = 10``), which measures the slant range in km-valued numbers and the
    # powers in the same normalised unit as ``P_max = 3500``.  That convention
    # is kept here unchanged, so the energy efficiency is in normalised units
    # and must not be compared with the absolute values of Fig. 7.
    d_s: float = 600.0               # slant range, km-valued number
    d_b: float = 10.0                # BS-UE distance, same normalised unit
    carrier: float = 1.55e9          # Hz
    noise_temp: float = 290.0        # K

    # --- reward ----------------------------------------------------------- #
    reward_scale: float = 1e4        # r = eta_EE / reward_scale

    # derived quantities (filled in by ESTNEnv, kept out of __init__) ------ #
    n_rounds: int = 0
    obs_dim: int = 0
    act_dim: int = 0

    def __post_init__(self) -> None:
        if self.K < 1 or self.T < 1:
            raise ValueError("K and T must be positive")
        self.n_rounds = self.T
        self.act_dim = self.T * (self.K + 3)
        self.obs_dim = self.act_dim + self.K + 1


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
class ESTNEnv:
    """Statistical-CSI CSTN environment (differentiable in the action)."""

    def __init__(self, cfg: EnvConfig | None = None, device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        c = self.cfg

        # ---- constants ---------------------------------------------------- #
        self.k_boltz = 1.380649e-23
        self.n0 = self.k_boltz * c.bandwidth * c.noise_temp        # noise power, W
        self.c_k = 1.0 / c.K                                       # common split ratio

        # ---- large-scale gains, eqs. (1)-(2) and (4) ---------------------- #
        g_s = self._large_scale(c.d_s)                             # satellite -> SU
        g_b = self._large_scale(c.d_b)                             # SU -> base station
        self.g_s2 = torch.full((c.K,), g_s ** 2, dtype=torch.float64,
                               device=self.device)
        self.g_b2 = torch.full((c.K,), g_b ** 2, dtype=torch.float64,
                               device=self.device)

        # ---- Shadowed-Rician parameters, eq. (2) -------------------------- #
        b, m, om = c.b_s, c.m_s, c.omega_s
        self.lambda_s = (1.0 / (2 * b)) * ((2 * b * m) / (2 * b * m + om)) ** m
        self.beta_s = 1.0 / (2 * b)
        self.delta_s = (1.0 / (2 * b)) * (om / (2 * b * m + om))
        self.be_s = self.beta_s - self.delta_s

        # ---- Nakagami-m parameters, eq. (5) ------------------------------- #
        self.xi_g = c.m_g / c.omega_g

        # ---- pre-computed thresholds -------------------------------------- #
        # tan_s is G_S^2 / N0, i.e. Tan_c1 = tan_s * P_c and so on.
        self.tan_s = (self.g_s2 / self.n0).reshape(1, c.K)
        self.tan_b = (self.g_b2 * c.p_bs / self.n0).reshape(1, c.K)
        self.eps = 1e-12

        # ---- action box ---------------------------------------------------- #
        lo = torch.empty(c.act_dim, dtype=torch.float64)
        hi = torch.empty(c.act_dim, dtype=torch.float64)
        lo[: 2 * c.T] = c.r_min
        hi[: 2 * c.T] = c.r_max
        lo[2 * c.T:] = 0.0
        hi[2 * c.T:] = c.p_max
        self.lo, self.hi = lo, hi
        self.span = (hi - lo)

        # ---- nominal action used to initialise the state ------------------- #
        a0 = torch.zeros(c.act_dim, dtype=torch.float64)
        a0[: 2 * c.T] = 0.5 * (c.r_min + c.r_max)
        a0[2 * c.T:] = c.p_max / (c.K + 1)
        self.a0 = self.repair(a0)          # start from a feasible allocation

        self._reset_internal()

    # ------------------------------------------------------------------ #
    # channel
    # ------------------------------------------------------------------ #
    def _large_scale(self, d: float) -> float:
        """Large-scale gain, eqs. (1)-(2): free-space below 100 m, otherwise
        the satellite antenna pattern of the manuscript."""
        c = self.cfg
        if d <= 100.0:
            return math.sqrt(3e8 ** 2 * 10.0 * 10.0) / (4 * math.pi * c.carrier * d ** 1.5)
        g_max = 10 ** 3.5
        w = 0.5
        g_t = g_max * (_besselj(1, w) / (2 * w) + 36 * _besselj(3, w) / w ** 3) ** 2
        g_r = 100.0
        return 3e8 * math.sqrt(g_t * g_r) / (4 * math.pi * c.carrier * d)

    # ------------------------------------------------------------------ #
    # closed-form outage, eq. (15)
    # ------------------------------------------------------------------ #
    def _outage(self, a: Tensor) -> Tensor:
        """Per-round outage probability, shape (..., T, K)."""
        c = self.cfg
        T, K = c.T, c.K
        rc = a[..., :T].unsqueeze(-1)                       # (...,T,1)
        rp = a[..., T:2 * T].unsqueeze(-1)
        pc = a[..., 2 * T:3 * T].unsqueeze(-1)
        pk = a[..., 3 * T:3 * T + T * K].reshape(*a.shape[:-1], T, K)

        g_c = torch.pow(2.0, rc) - 1.0                      # gamma_c,th
        g_p = torch.pow(2.0, rp) - 1.0                      # gamma_p,th
        p_priv = pk.sum(-1, keepdim=True)                   # sum_k P_k,t
        p_tot = pc + p_priv                                 # P_s,t of (26-d)

        # the common stream is decoded first and sees the private streams as
        # interference, so Tan_c2 is built from p_priv, not from p_tot
        tan_c1 = self.tan_s * pc
        tan_c2 = self.tan_s * p_priv
        tan_p1 = self.tan_s * pk
        tan_p2 = self.tan_s * (p_priv - pk)
        tan_p3 = c.sic_residual * tan_c1

        den_c = tan_c1 - g_c * tan_c2
        den_p = tan_p1 - g_p * (tan_p2 + tan_p3)

        ok = (den_c > self.eps) & (den_p > self.eps)

        gam_c = g_c / den_c.clamp_min(self.eps)
        gam_p = g_p / den_p.clamp_min(self.eps)
        gam = torch.maximum(gam_c, gam_p).clamp(max=1e8)    # eq. (13)-(14)

        tan_e = (c.err_s ** 2 * self.g_s2 * p_tot / self.n0
                 + c.err_b ** 2 * self.g_b2 * c.p_bs / self.n0 + 1.0)

        # ---- triple sum of (15): n_s, q, r -------------------------------- #
        base = self.be_s * gam * self.tan_b + self.xi_g    # (...,T,K)
        acc = torch.zeros_like(base)
        for n in range(c.m_s):
            tao = ((-1.0) ** n * _pochhammer_1m(c.m_s, n) * self.delta_s ** n
                   / math.factorial(n) ** 2)
            for q in range(n + 1):
                for r in range(q + 1):
                    coef = (math.factorial(n)
                            / (math.factorial(q) * self.be_s ** (n - q + 1))
                            * tao * math.comb(q, r)
                            * math.factorial(q - r + c.m_g - 1))
                    acc = acc + coef * gam ** q * self.tan_b ** (q - r) \
                        * tan_e ** r * base ** (-(q - r + c.m_g))

        p_out = 1.0 - (self.lambda_s
                       * torch.exp(-self.be_s * gam * tan_e)
                       * self.xi_g ** c.m_g / math.gamma(c.m_g) * acc)

        p_out = torch.where(ok, p_out, torch.ones_like(p_out))
        return p_out.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # throughput, power, AAoI
    # ------------------------------------------------------------------ #
    def _cumulative(self, p_out: Tensor) -> Tensor:
        """Cumulative outage P_out,k,t = prod_{tau<=t} P_s,k,tau."""
        return torch.cumprod(p_out, dim=-2)

    def _ee(self, a: Tensor, p_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Effective throughput (24), average power (25) and EE."""
        c = self.cfg
        T, K = c.T, c.K
        rc = a[..., :T].unsqueeze(-1)
        rp = a[..., T:2 * T].unsqueeze(-1)
        pc = a[..., 2 * T:3 * T]
        pk = a[..., 3 * T:3 * T + T * K].reshape(*a.shape[:-1], T, K)

        cum = self._cumulative(p_out)                       # (...,T,K)
        prev = torch.cat([torch.ones_like(cum[..., :1, :]),
                          cum[..., :-1, :]], dim=-2)        # P_out,k,0 = 1

        # eq. (24)
        thr = ((rc * self.c_k + rp) * (prev - cum)).sum(dim=(-1, -2))

        # eq. (25)
        all_done = torch.prod(1.0 - prev, dim=-1, keepdim=True)
        p_common = (pc.unsqueeze(-1) * (1.0 - all_done)).sum(dim=(-1, -2))
        p_private = (pk * prev).sum(dim=(-1, -2))
        power = p_common + p_private

        ee = c.bandwidth * thr / power.clamp_min(1e-9)
        return ee, thr, power

    def _aaoi(self, p_out: Tensor) -> Tensor:
        """Average AoI of the wait-for-all HARQ, eq. (23)."""
        c = self.cfg
        T, K = c.T, c.K
        cum = self._cumulative(p_out)                       # (...,T,K)
        prev = torch.cat([torch.ones_like(cum[..., :1, :]),
                          cum[..., :-1, :]], dim=-2)

        # E[X_j] = T_slot (1 + sum_{t=1}^{T-1} P_out,j,t)
        ex = c.slot * (1.0 + cum[..., :T - 1, :].sum(dim=-2))          # (...,K)

        # E[Y] and E[Y^2] from eq. (C.6)-(C.7) of the supplement
        done = torch.prod(1.0 - prev[..., 1:T, :], dim=-1)             # (...,T-1)
        idx = torch.arange(1, T, dtype=cum.dtype, device=cum.device)
        ey = c.slot * (T - done.sum(dim=-1))
        ey2 = c.slot ** 2 * (T ** 2 - ((2 * idx + 1) * done).sum(dim=-1))

        return ey2 / (2 * ey.clamp_min(1e-12)) + ex.mean(dim=-1)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def evaluate(self, a: Tensor) -> Dict[str, Tensor]:
        """All quantities of interest, differentiable with respect to ``a``.

        ``a`` has shape ``(..., T*(K+3))`` in physical units (bit/s/Hz and W)
        and may carry any number of leading batch dimensions.
        """
        a64 = a.to(torch.float64)
        p_out = self._outage(a64)
        ee, thr, power = self._ee(a64, p_out)
        aaoi = self._aaoi(p_out)
        op_final = p_out[..., -1, :]
        return {
            "ee": ee.to(self.dtype),
            "throughput": thr.to(self.dtype),
            "power": power.to(self.dtype),
            "p_out": p_out.to(self.dtype),
            "op_final": op_final.to(self.dtype),
            "aaoi": aaoi.to(self.dtype),
        }

    def violations(self, a: Tensor) -> Dict[str, Tensor]:
        """Violation degrees of eq. (27), differentiable in ``a``."""
        c = self.cfg
        T, K = c.T, c.K
        a64 = a.to(torch.float64)
        rc = a64[..., :T].unsqueeze(-1)
        rp = a64[..., T:2 * T].unsqueeze(-1)
        pc = a64[..., 2 * T:3 * T].unsqueeze(-1)
        pk = a64[..., 3 * T:3 * T + T * K].reshape(*a64.shape[:-1], T, K)

        g_c = torch.pow(2.0, rc) - 1.0
        g_p = torch.pow(2.0, rp) - 1.0
        p_priv = pk.sum(-1, keepdim=True)
        p_tot = pc + p_priv

        zeta_c = g_c - pc / p_priv.clamp_min(1e-9)
        ratio_p = pk / (p_priv - pk + c.sic_residual * pc).clamp_min(1e-9)
        zeta_p = g_p - ratio_p

        v_sinr = (zeta_c.clamp_min(0.0)
                  + zeta_p.clamp_min(0.0).sum(-1, keepdim=True)).sum(-2).squeeze(-1)

        out = self.evaluate(a64)
        v_op = (out["op_final"] - c.op_budget).clamp_min(0.0).sum(-1)
        v_aoi = (out["aaoi"] - c.aaoi_budget).clamp_min(0.0)
        v_pow = (p_tot - c.p_max).clamp_min(0.0).sum(-2).squeeze(-1)

        return {
            "sinr": v_sinr.to(self.dtype),
            "op": v_op.to(self.dtype),
            "aoi": v_aoi.to(self.dtype),
            "pow": v_pow.to(self.dtype),
        }

    def repair(self, a: Tensor, r_lo: float | None = None) -> Tensor:
        """Closed-form repair used by the projection layer.

        ``r_lo`` overrides the lower bound of the target rates.  The default is
        ``cfg.r_min``, the range in which the policy acts; the projection layer
        passes ``0.0``, because (26-f) only requires ``R >= 0`` and a rate below
        the policy range is sometimes needed to meet the outage budget.

        1. clip to the box ``[r_min, r_max] x [0, p_max]``;
        2. rescale every round's powers by a common factor so that the
           transmit power obeys the budget of (26-d);
        3. lower the target rates until the two SINR feasibility conditions
           hold with a 5 % power margin, i.e. until
           ``gamma_c,th <= 0.95 P_c / sum_k P_k`` and
           ``gamma_p,th <= 0.95 P_k / (sum_{i != k} P_i + Xi_k P_c)``.

        Step 3 without the margin would land exactly on the boundary, where
        the success probability of (15) is zero by construction.
        """
        c = self.cfg
        T, K = c.T, c.K
        lo = self.lo.to(a)
        if r_lo is not None:
            lo = lo.clone()
            lo[:2 * T] = r_lo
        a = torch.minimum(torch.maximum(a, lo), self.hi.to(a))
        rc = a[..., :T]
        rp = a[..., T:2 * T]
        pc = a[..., 2 * T:3 * T]
        pk = a[..., 3 * T:3 * T + T * K].reshape(*a.shape[:-1], T, K)

        # step 2: common rescaling of the powers of each round
        p_tot = pc + pk.sum(-1)
        scale = torch.where(p_tot > c.p_max, c.p_max / p_tot.clamp_min(1e-9),
                            torch.ones_like(p_tot))
        pc = pc * scale
        pk = pk * scale.unsqueeze(-1)

        # step 3: cap the rates at the SINR that the powers can support
        #
        # A relative power margin ``REPAIR_MARGIN`` keeps the repaired rate
        # strictly inside the feasible set.  Without it the cap would sit
        # exactly on the boundary gamma_c,th = P_c / sum_k P_k, where
        # den_c = tan_c1 - g_c * tan_c2 = 0 and the outage probability of (15)
        # is one by definition: the repair would then map every over-ambitious
        # rate onto the single worst feasible point of the SINR set.  The
        # margin costs a fraction of a bit/s/Hz and makes the map continuous.
        margin = 1.0 - 0.05
        p_priv = pk.sum(-1)
        cap_c = torch.log2(1.0 + margin * pc / p_priv.clamp_min(1e-9))
        cap_p = torch.log2(1.0 + margin * (pk / (p_priv.unsqueeze(-1) - pk
                                               + c.sic_residual * pc.unsqueeze(-1))
                                           .clamp_min(1e-9)).min(dim=-1).values)
        rc = torch.minimum(rc, cap_c)
        rp = torch.minimum(rp, cap_p)

        lo_r = c.r_min if r_lo is None else r_lo
        rc = rc.clamp(lo_r, c.r_max)
        rp = rp.clamp(lo_r, c.r_max)
        return torch.cat([rc, rp, pc, pk.reshape(*a.shape[:-1], T * K)], dim=-1)

    # ------------------------------------------------------------------ #
    # MDP interface
    # ------------------------------------------------------------------ #
    def _reset_internal(self) -> None:
        out = self.evaluate(self.a0)
        self.state = self._make_state(self.a0, out["op_final"], out["aaoi"])

    def _make_state(self, a: Tensor, op_final: Tensor, aaoi: Tensor) -> Tensor:
        """Normalised state: previous allocation + OP + AAoI (dimension 26)."""
        c = self.cfg
        norm = torch.cat([torch.full((2 * c.T,), c.r_max, dtype=a.dtype, device=a.device),
                          torch.full((c.T * (c.K + 1),), c.p_max, dtype=a.dtype,
                                     device=a.device)])
        return torch.cat([(a / norm).reshape(-1),
                          op_final.reshape(-1),
                          (aaoi / (c.T * c.slot)).reshape(-1)]).to(self.dtype)

    def reset(self) -> Tensor:
        """Start a new frame sequence; the previous allocation is nominal."""
        self._reset_internal()
        return self.state.clone()

    def step(self, a: Tensor) -> Tuple[Tensor, float, Dict[str, float]]:
        """Execute one frame.  ``a`` is the *projected* action ``a*``."""
        c = self.cfg
        a = a.reshape(-1).to(torch.float64)
        out = self.evaluate(a)
        reward = out["ee"] / c.reward_scale

        self.state = self._make_state(a, out["op_final"], out["aaoi"])
        info = {
            "ee": float(out["ee"]),
            "op_max": float(out["op_final"].max()),
            "aaoi": float(out["aaoi"]),
            "power": float(out["power"]),
        }
        return self.state, float(reward), info

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim

    @property
    def act_dim(self) -> int:
        return self.cfg.act_dim

    def describe(self) -> str:
        c = self.cfg
        # powers are in the normalised unit of the original simulator, not in W
        return (f"ESTNEnv(K={c.K}, T={c.T}, obs={c.obs_dim}, act={c.act_dim}, "
                f"P_max={c.p_max:g} (norm.), delta_s={c.op_budget:g}, "
                f"Delta_bar={c.aaoi_budget:g} s)")


if __name__ == "__main__":
    env = ESTNEnv()
    print(env.describe())
    s = env.reset()
    print("state:", tuple(s.shape), "range", float(s.min()), float(s.max()))
    a = env.a0
    print("nominal:", {k: round(v, 6) for k, v in env.step(a)[2].items()})
    print("violations:", {k: round(float(v), 6) for k, v in env.violations(a).items()})
