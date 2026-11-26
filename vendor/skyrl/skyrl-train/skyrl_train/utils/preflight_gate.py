"""Pre-flight reward gate for RLOO.

Probes the dataset before training and refuses to start when the reward
distribution leaves the band where RLOO has usable within-group variance.

The gate keys on **mean per-sample reward**, not pass@8.  RLOO learns from
variance *within* the group of ``n_samples_per_prompt``, which is maximal near
0.5.  Both extremes starve it — by different mechanisms:

* **Sparse** (mean near 0): nearly every group is uniform-zero, gets dropped by
  ``rloo_n_filter_zero_reward_groups``, and no gradient survives.
* **Saturated** (mean near 1): nearly every rollout succeeds, leaving almost no
  within-group variance.

Default band **0.25–0.75** on mean per-sample reward.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger


@dataclass
class PreflightResult:
    """Outcome of a pre-flight gate check."""

    passed: bool
    mean_reward: float
    non_zero_fraction: float
    num_samples: int
    reason: Optional[str] = None  # "sparse" | "saturated" | None
    bound: Optional[Tuple[float, float]] = None  # (measured, threshold)
    message: str = ""


def check_preflight_gate(
    rewards: List[float],
    min_reward: float = 0.25,
    max_reward: float = 0.75,
) -> PreflightResult:
    """Check whether a reward distribution is inside the learnable band.

    Args:
        rewards: Per-sample scalar rewards from the pre-flight probe.
        min_reward: Lower bound on mean per-sample reward.
        max_reward: Upper bound on mean per-sample reward.

    Returns:
        ``PreflightResult`` with the verdict and statistics.
    """
    n = len(rewards)
    if n == 0:
        return PreflightResult(
            passed=True,  # vacuously — let the trainer decide
            mean_reward=0.0,
            non_zero_fraction=0.0,
            num_samples=0,
            message="No pre-flight samples; gate skipped.",
        )

    mean_reward = sum(rewards) / n
    non_zero = sum(1 for r in rewards if r > 0.0) / n

    # Wilson-style interval for the non-zero fraction (rough confidence guide).
    # Not used for the decision — just reported.
    z = 1.96  # 95%
    ci_half = z * math.sqrt(max(mean_reward * (1 - mean_reward), 0) / n) if n > 0 else 0.0

    if mean_reward < min_reward:
        msg = (
            f"[preflight] {n} trials, mean per-sample reward {mean_reward:.3f}, "
            f"non-zero fraction {non_zero:.3f}\n"
            f"[preflight] FAIL: {mean_reward:.3f} is below min_reward {min_reward} (sparse regime).\n"
            f"            Nearly every group will be uniform and dropped, leaving no gradient.\n"
            f"            Set trainer.preflight_gate.on_failure=warn to train anyway."
        )
        logger.error(msg)
        return PreflightResult(
            passed=False,
            mean_reward=mean_reward,
            non_zero_fraction=non_zero,
            num_samples=n,
            reason="sparse",
            bound=(mean_reward, min_reward),
            message=msg,
        )

    if mean_reward > max_reward:
        msg = (
            f"[preflight] {n} trials, mean per-sample reward {mean_reward:.3f}, "
            f"non-zero fraction {non_zero:.3f}\n"
            f"[preflight] FAIL: {mean_reward:.3f} is above max_reward {max_reward} (saturated regime).\n"
            f"            Nearly every rollout succeeds, leaving almost no within-group variance.\n"
            f"            Set trainer.preflight_gate.on_failure=warn to train anyway."
        )
        logger.error(msg)
        return PreflightResult(
            passed=False,
            mean_reward=mean_reward,
            non_zero_fraction=non_zero,
            num_samples=n,
            reason="saturated",
            bound=(mean_reward, max_reward),
            message=msg,
        )

    msg = (
        f"[preflight] {n} trials, mean per-sample reward {mean_reward:.3f}, "
        f"non-zero fraction {non_zero:.3f} (±{ci_half:.3f}) — PASS "
        f"(band [{min_reward}, {max_reward}])."
    )
    logger.info(msg)
    return PreflightResult(
        passed=True,
        mean_reward=mean_reward,
        non_zero_fraction=non_zero,
        num_samples=n,
        message=msg,
    )
