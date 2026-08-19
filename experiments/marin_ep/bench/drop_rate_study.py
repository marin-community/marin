# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MEP-H1: drop rates of capacity rules under skewed routing (hero shape).

Compares three static-capacity drop rules on synthetic trained-router-like
routing at the hero shape:

- `per_cell`: today's `fixed_all_to_all` — capacity per (source device,
  expert) cell, `ceil(cf * A / E)` rows; idle cells cannot lend to hot ones.
- `per_expert`: SPEC v1 pooled rule — capacity per expert across sources,
  `ceil(cf * A_global / E)` rows.
- `per_owner`: capacity pooled across each owner's `El` local experts
  (`El * ceil(cf * A_global / E)` rows per owner receive pool) — the
  natural bound for the kernel's per-owner receive buffer.

Routing model: global expert popularity `p ~ Dirichlet(alpha)` (symmetric,
small alpha = hot experts), optionally tilted per device by a lognormal
factor to model batch heterogeneity; tokens draw K distinct experts via
Gumbel top-K. `alpha` is calibrated so `per_cell` at cf=1.30 reproduces the
measured hero drop rate (~3.96%), then all rules are evaluated on the same
routing draws.

Run: `uv run python experiments/marin_ep/bench/drop_rate_study.py`
"""

import math
from dataclasses import dataclass

import numpy as np

from experiments.marin_ep.oracle import kept_rows_per_expert

HERO_DEVICES = 64
HERO_TOKENS = 65536
HERO_TOPK = 4
HERO_EXPERTS = 192
HERO_LOCAL_EXPERTS = 3
MEASURED_HERO_DROP = 0.0396  # per_cell at cf=1.30, .agents/logbooks/7279-moe-hero-ep.md


@dataclass(frozen=True)
class RoutingModel:
    alpha: float
    device_heterogeneity: float = 0.0  # stddev of per-device lognormal tilt


_INCLUSION_SAMPLE_TOKENS = 32768


def _topk_inclusion(rng: np.random.Generator, log_p: np.ndarray, topk: int) -> np.ndarray:
    """Monte-Carlo Gumbel top-K inclusion probability per expert."""
    num_experts = log_p.shape[0]
    gumbel = rng.gumbel(size=(_INCLUSION_SAMPLE_TOKENS, num_experts))
    chosen = np.argpartition(-(log_p + gumbel), topk, axis=1)[:, :topk]
    return np.bincount(chosen.reshape(-1), minlength=num_experts) / _INCLUSION_SAMPLE_TOKENS


def sample_expert_counts(
    rng: np.random.Generator,
    model: RoutingModel,
    *,
    devices: int = HERO_DEVICES,
    tokens: int = HERO_TOKENS,
    topk: int = HERO_TOPK,
    num_experts: int = HERO_EXPERTS,
) -> np.ndarray:
    """Sample a `[S, E]` count matrix of expert assignments.

    Tokens are iid, so device counts are Binomial(tokens, pi_g) with pi_g the
    Gumbel top-K inclusion probability of expert g — estimated once by Monte
    Carlo, then counts drawn directly (exact per-expert marginal noise without
    materializing every token). Per-device heterogeneity re-tilts pi per device.
    """
    popularity = rng.dirichlet(np.full(num_experts, model.alpha))
    log_p = np.log(popularity + 1e-30)
    counts = np.zeros((devices, num_experts), dtype=np.int64)
    if model.device_heterogeneity == 0.0:
        inclusion = _topk_inclusion(rng, log_p, topk)
        for d in range(devices):
            counts[d] = rng.binomial(tokens, inclusion)
        return counts
    for d in range(devices):
        tilt = model.device_heterogeneity * rng.standard_normal(num_experts)
        inclusion = _topk_inclusion(rng, log_p + tilt, topk)
        counts[d] = rng.binomial(tokens, inclusion)
    return counts


def drop_rate_per_cell(counts: np.ndarray, capacity_factor: float) -> float:
    devices, num_experts = counts.shape
    assignments = counts.sum() // devices
    cell_capacity = max(1, math.ceil(capacity_factor * assignments / num_experts))
    dropped = np.maximum(counts - cell_capacity, 0).sum()
    return float(dropped / counts.sum())


def drop_rate_grouped(counts: np.ndarray, capacity_factor: float, group_size: int) -> float:
    """Drop rate of the canonical group-pooled rule (SPEC S2); group_size=1 is per-expert."""
    num_experts = counts.shape[1]
    total = counts.sum(axis=0)
    capacity = max(1, math.ceil(capacity_factor * counts.sum() / num_experts))
    kept = kept_rows_per_expert(total, capacity=capacity, group_size=group_size)
    return float((total - kept).sum() / counts.sum())


def calibrate_alpha(rng: np.random.Generator, *, target_drop: float, capacity_factor: float) -> float:
    """Bisect alpha so per_cell drop at `capacity_factor` matches `target_drop`."""
    lo, hi = 0.05, 50.0  # smaller alpha = more skew = more drops
    for _ in range(24):
        mid = math.sqrt(lo * hi)
        counts = sample_expert_counts(rng, RoutingModel(alpha=mid))
        if drop_rate_per_cell(counts, capacity_factor) > target_drop:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main() -> None:
    rng = np.random.default_rng(seed=0)
    alpha = calibrate_alpha(rng, target_drop=MEASURED_HERO_DROP, capacity_factor=1.30)
    print(f"calibrated alpha = {alpha:.3f} (per_cell drop at cf 1.30 ~= {MEASURED_HERO_DROP:.2%})")
    for heterogeneity in (0.0, 0.1, 0.3):
        model = RoutingModel(alpha=alpha, device_heterogeneity=heterogeneity)
        counts = sample_expert_counts(rng, model)
        print(f"\ndevice_heterogeneity={heterogeneity}")
        print("  cf     per_cell  per_expert  per_owner(G=3)")
        for cf in (1.20, 1.30, 1.33, 1.50, 1.75, 2.00):
            row = (
                drop_rate_per_cell(counts, cf),
                drop_rate_grouped(counts, cf, 1),
                drop_rate_grouped(counts, cf, HERO_LOCAL_EXPERTS),
            )
            print(f"  {cf:.2f}   {row[0]:8.3%}  {row[1]:9.3%}  {row[2]:8.3%}")


if __name__ == "__main__":
    main()
