"""Standalone truncation penalty logic.

Extracted from ``harbor/runner.py`` so it can be unit-tested without
pulling in the full runner / skyrl_gym import chain.

The penalty keys on PER-TURN truncation (a single generation that hit
``max_generate_length``), not on trajectory-level ``stop_reason``.  The
trajectory-level cap is almost never reached on real workloads, so the old
``stop_reason == "length"`` check was effectively dead code.
"""

from typing import List, Optional, Tuple


def detect_turn_truncation(
    per_turn_token_counts: Optional[List[int]],
    max_generate_length: int,
) -> bool:
    """Return True if any single generation terminated at the per-turn cap.

    A generation that emits exactly ``max_generate_length`` tokens was cut off
    by the engine's length limit rather than stopping naturally.  This is the
    population the penalty is meant to reach: the truncated output is
    mid-structure, typically unparseable as a tool call, and ends the trial at
    reward 0.

    Returns False when per-turn data is unavailable (safe default: no penalty).
    """
    if not per_turn_token_counts or max_generate_length <= 0:
        return False
    return any(n >= max_generate_length for n in per_turn_token_counts)


def apply_truncation_penalty(
    reward: float,
    original_reward: float,
    turn_truncated: bool,
    truncation_penalty: float,
) -> Tuple[float, bool]:
    """Penalize a trial with a cap-truncated generation.

    A trial where a single generation terminated at ``max_generate_length``
    (``turn_truncated == True``) and failed its verifier
    (``original_reward == 0.0``) has ``truncation_penalty`` SUBTRACTED from its
    reward, taking it below the zero floor so RLOO assigns it a negative
    advantage instead of lumping it with honest failures at zero.

    The penalty subtracts rather than overwrites, which matters as soon as a
    partial-credit shaper (``pass_ratio``) is enabled.  Overwriting would score
    every truncated trial at the same value regardless of how many tests it
    passed, flattening the partial-credit signal on exactly the population the
    penalty is aimed at.

    With no shaper the two are identical: ``reward == original_reward == 0.0``,
    so subtracting yields ``-truncation_penalty`` either way.

    Returns ``(new_reward, penalized)``.
    """
    if truncation_penalty > 0.0 and turn_truncated and original_reward == 0.0:
        return reward - truncation_penalty, True
    return reward, False
