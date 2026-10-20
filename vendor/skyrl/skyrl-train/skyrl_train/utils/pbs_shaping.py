"""Potential-based shaping (PBS) on edit tokens.

Each observed test run changes a potential based on the fraction of tests passing. The potential difference
is spread over the preceding turn's EDIT tokens. With gamma=1, the total telescopes to the final potential.
The total is clamped to ±max_total_shaping. This terminal bonus changes the trained objective; the trainer
adds it after the outcome estimator and batch normalization as token credit without a baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from skyrl_train.utils.span_tagger import SPAN_EDIT
from skyrl_train.utils.test_delta_parser import TestRunResult, extract_test_runs

# Default PBS hyperparameters (override via reward_shaping config).
DEFAULT_GAMMA: float = 1.0
DEFAULT_MAX_TOTAL_SHAPING: float = 0.3  # |sum of token_level_shaping| ceiling
DEFAULT_POTENTIAL_SHAPE: str = "linear"  # "linear" | "near_green"


def _potential(frac_passing: float, shape: str) -> float:
    """Potential Φ as a function of the fraction of tests passing (in [0, 1]).

    - "linear":     Φ = frac. Uniform credit per test fixed.
    - "near_green": Φ = frac**2. Convex → marginal potential rises toward 1.0,
                    so closing the LAST failing test yields the largest single
                    increment (targets the "stalls one test short" failure mode).
    """
    frac = max(0.0, min(1.0, frac_passing))
    if shape == "near_green":
        return frac * frac
    # default linear
    return frac


def _assistant_turn_token_ranges(response_span_tags: List[int]) -> List[List[int]]:
    """Group contiguous non-OTHER token positions into per-turn token ranges.

    The span tags are laid out turn-by-turn (OTHER for user/observation/prompt
    tokens, THINK/ACTION/EDIT for an assistant turn's generated tokens). A
    maximal run of consecutive non-OTHER positions therefore corresponds to ONE
    assistant turn's generated span. Returns a list (in trajectory order) of the
    token-index lists for each such assistant turn.
    """
    turns: List[List[int]] = []
    cur: List[int] = []
    for j, tag in enumerate(response_span_tags):
        if tag != 0:  # SPAN_OTHER == 0
            cur.append(j)
        elif cur:
            turns.append(cur)
            cur = []
    if cur:
        turns.append(cur)
    return turns


def _edit_positions_in_turn(turn_positions: List[int], response_span_tags: List[int]) -> List[int]:
    """The EDIT-tagged token positions within a single assistant turn's span."""
    return [j for j in turn_positions if response_span_tags[j] == SPAN_EDIT]


def compute_pbs_token_shaping(
    chat_history: Optional[List[Dict[str, Any]]],
    response_span_tags: List[int],
    *,
    gamma: float = DEFAULT_GAMMA,
    max_total_shaping: float = DEFAULT_MAX_TOTAL_SHAPING,
    potential_shape: str = DEFAULT_POTENTIAL_SHAPE,
    test_runs: Optional[List[TestRunResult]] = None,
) -> List[float]:
    """Compute the per-token PBS shaping vector for one trajectory.

    Args:
        chat_history: the full conversation (system + user + assistant + tool
            observations). Used by F2 to find real test-run stdout. Pass
            ``test_runs`` directly to bypass parsing (testing).
        response_span_tags: the F4 per-token span tags, 1:1 with ``response_ids``
            (SPAN_OTHER/THINK/ACTION/EDIT). Length defines the output length.
        gamma: PBS discount γ (default 1.0 → undiscounted telescoping).
        max_total_shaping: clamp on |Σ token_level_shaping| (default 0.3).
        potential_shape: "linear" or "near_green".
        test_runs: optional pre-parsed F2 runs (else parsed from chat_history).

    Returns:
        A list of floats len == len(response_span_tags). All-zero when there is
        no usable test signal or no edit turn to credit (→ no token credit for that
        trajectory). The non-zero entries land only on EDIT-tagged tokens.
    """
    n = len(response_span_tags)
    shaping = [0.0] * n
    if n == 0:
        return shaping

    if test_runs is None:
        test_runs = extract_test_runs(chat_history)
    if not test_runs:
        # No recognized in-trajectory test output → no-signal → outcome-only.
        return shaping

    # Per-turn token ranges, in trajectory order (assistant turns only).
    turn_ranges = _assistant_turn_token_ranges(response_span_tags)
    if not turn_ranges:
        return shaping

    # Map each assistant turn to the chat_history message index it corresponds
    # to. The span tags walk `response_messages` (== conversation[1:]); but the
    # turn ORDER is what we need to align the k-th edit turn with the k-th test
    # run. We use a simpler, robust correspondence: edit turns and test runs are
    # both in trajectory order, so the edit turn that produced test-run k is the
    # last EDIT turn occurring before test-run k. Since we only have token-span
    # ordering (not message indices) for turns, we credit each transition to the
    # MOST RECENT edit turn seen so far at the time of that test run.
    #
    # Identify which turn ranges are edit turns (contain >=1 EDIT token) and the
    # order in which edit turns appear.
    edit_turn_ranges: List[List[int]] = []
    for tr in turn_ranges:
        edit_toks = _edit_positions_in_turn(tr, response_span_tags)
        if edit_toks:
            edit_turn_ranges.append(edit_toks)

    if not edit_turn_ranges:
        # Tests moved but no EDIT turn to credit (e.g. config-only change tagged
        # ACTION). No edit tokens to scatter onto → outcome-only.
        return shaping

    # Walk test-run transitions and credit each to the next available edit turn.
    # We pair the k-th transition with the k-th edit turn in order; if there are
    # more transitions than edit turns, extra transitions fold onto the LAST edit
    # turn (its later effect). This keeps Σ F == γ·Φ_T − Φ_0 (telescoping intact).
    prev_potential = 0.0  # Φ(s_0): no test has run yet
    per_edit_turn_credit = [0.0] * len(edit_turn_ranges)
    n_edit_turns = len(edit_turn_ranges)
    for k, run in enumerate(test_runs):
        cur_potential = _potential(run.frac_passing, potential_shape)
        f_k = gamma * cur_potential - prev_potential
        # Credit transition k to edit turn min(k, last). (Edit precedes the run.)
        turn_idx = min(k, n_edit_turns - 1)
        per_edit_turn_credit[turn_idx] += f_k
        prev_potential = cur_potential

    # Bound the TOTAL shaping so the hidden-test outcome stays dominant.
    total = sum(per_edit_turn_credit)
    if abs(total) > max_total_shaping and total != 0.0:
        scale = max_total_shaping / abs(total)
        per_edit_turn_credit = [c * scale for c in per_edit_turn_credit]
        logger.debug(
            "pbs_shaping: total shaping {:.4f} exceeded ±{} → scaled by {:.4f}",
            total,
            max_total_shaping,
            scale,
        )

    # Scatter each edit turn's credit UNIFORMLY across its EDIT tokens.
    for credit, edit_toks in zip(per_edit_turn_credit, edit_turn_ranges):
        if not edit_toks or credit == 0.0:
            continue
        per_tok = credit / len(edit_toks)
        for j in edit_toks:
            shaping[j] = per_tok

    return shaping
