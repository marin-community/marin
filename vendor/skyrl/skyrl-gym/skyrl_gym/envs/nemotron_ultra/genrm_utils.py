# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for GenRM comparison server.

This module provides:
- Prompt key and conversation history extraction (for cohort grouping)
- Comparison pair generation strategies (circular, all_pairs)
- GenRM output parsing (JSON score extraction)
- Response API object text extraction
- Score aggregation with tiebreaker logic
- Length-based bonus/penalty computation
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_COMPARISON_STRATEGIES = frozenset({"circular", "all_pairs"})
SUPPORTED_AGGREGATOR_METHODS = frozenset({"simple_tiebreaker"})

# GenRM ranking midpoint - rankings are 1-6, midpoint is 3.5
# Used in tiebreaker: ranking < 3.5 means response_1 is better, > 3.5 means response_2 is better
RANKING_MIDPOINT = 3.5

# Default placeholder for empty/missing output
EMPTY_OUTPUT_PLACEHOLDER = "None"


# =============================================================================
# Exceptions
# =============================================================================


class GenRMOutputParseError(ValueError):
    """Raised when GenRM output cannot be parsed into expected JSON scores.

    Expected format: {"score_1": <1-5>, "score_2": <1-5>, "ranking": <1-6>}
    """

    pass


# =============================================================================
# Prompt key and conversation history (for cohort-based verify)
# =============================================================================


def _to_jsonable_data(value: Any) -> Any:
    """Recursively convert request payloads into JSON-serializable data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_jsonable_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_data(item) for item in value]
    return value


def get_prompt_key_from_input(input_messages: List[Any], principle: Optional[str] = None) -> str:
    """Stable key for grouping rollouts by prompt and principle.

    Used by cohort-based verify to group N rollouts per prompt.

    Args:
        input_messages: Conversation input (e.g. responses_create_params.input)
        principle: Optional principle string (principle-based GenRM)

    Returns:
        A stable string key for the prompt + principle combination
    """
    key_data = {"input": _to_jsonable_data(input_messages), "principle": principle}
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


def get_prompt_key(example: Dict) -> str:
    """Get stable key for grouping examples by prompt and principle.

    Examples with the same conversation history but different principles
    should be in separate groups. Accepts either prompt_id or hashes input + principle.

    Args:
        example: Dict with optional "prompt_id", or "responses_create_params" with "input"

    Returns:
        A stable string key
    """
    if "prompt_id" in example:
        prompt_id = str(example["prompt_id"])
        principle = example.get("principle")
        if principle is not None:
            return f"{prompt_id}::{principle}"
        return prompt_id
    conv = extract_conversation_history(example)
    principle = example.get("principle")
    return get_prompt_key_from_input(conv, principle)


def extract_conversation_history(example: Dict) -> List[Dict]:
    """Extract conversation history from example (responses_create_params.input).

    Args:
        example: Dict with responses_create_params.input (list of message dicts)

    Returns:
        List of message dicts with 'role' and 'content'

    Raises:
        ValueError: If required fields are missing
    """
    responses_create_params = example.get("responses_create_params")
    if responses_create_params is None:
        raise ValueError(f"Example missing 'responses_create_params': {list(example.keys())}")
    if "input" not in responses_create_params:
        raise ValueError(f"responses_create_params missing 'input': {list(responses_create_params.keys())}")
    return responses_create_params["input"]


# =============================================================================
# Comparison Strategy
# =============================================================================


def generate_comparison_pairs(strategy: str, num_responses: int) -> List[Tuple[int, int]]:
    """Generate pairs of response indices for pairwise comparison.

    Args:
        strategy: Comparison strategy - "circular" or "all_pairs"
            - "circular": Each response compared with next (N comparisons)
            - "all_pairs": Every pair compared (N*(N-1)/2 comparisons)
        num_responses: Number of responses to compare

    Returns:
        List of (i, j) tuples where i < j for all_pairs, or circular neighbors

    Raises:
        ValueError: If strategy is not supported or num_responses < 2
    """
    if strategy not in SUPPORTED_COMPARISON_STRATEGIES:
        raise ValueError(
            f"Unknown comparison strategy: '{strategy}'. Supported: {sorted(SUPPORTED_COMPARISON_STRATEGIES)}"
        )

    if num_responses < 2:
        raise ValueError(f"Need at least 2 responses for comparison, got {num_responses}")

    if strategy == "all_pairs":
        return list(itertools.combinations(range(num_responses), 2))
    else:  # circular
        return [(i, (i + 1) % num_responses) for i in range(num_responses)]


# =============================================================================
# GenRM Output Parsing
# =============================================================================


_JUDGE_ERROR_PREVIEW_CHARACTERS = 2000


def _unique_genrm_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GenRMOutputParseError("Duplicate GenRM fields")
        result[key] = value
    return result


def _finite_genrm_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise GenRMOutputParseError("Nonfinite GenRM JSON")
    return number


def parse_genrm_output(output: str, *, strict_json: bool = False) -> Tuple[float, float, float]:
    """Read the final complete GenRM object and require finite numeric scores."""
    decoder = (
        json.JSONDecoder(
            object_pairs_hook=_unique_genrm_object, parse_float=_finite_genrm_float, parse_constant=_finite_genrm_float
        )
        if strict_json
        else json.JSONDecoder()
    )
    cursor = 0
    parsed = None
    while (start := output.find("{", cursor)) >= 0:
        if strict_json and parsed is not None:
            raise GenRMOutputParseError("Multiple GenRM score objects")
        try:
            parsed, consumed = decoder.raw_decode(output[start:])
        except json.JSONDecodeError as error:
            raise GenRMOutputParseError(
                f"Incomplete GenRM JSON: {output[-_JUDGE_ERROR_PREVIEW_CHARACTERS:]}"
            ) from error
        cursor = start + consumed
    if isinstance(parsed, dict) and isinstance(parsed.get("overall"), dict):
        parsed = parsed["overall"]
    keys = ("score_1", "score_2", "ranking")
    if not isinstance(parsed, dict) or not all(key in parsed for key in keys):
        raise GenRMOutputParseError(f"Missing GenRM scores: {output[-_JUDGE_ERROR_PREVIEW_CHARACTERS:]}")
    values = tuple(parsed[key] for key in keys)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise GenRMOutputParseError(
            f"Nonfinite or nonnumeric GenRM scores: {output[-_JUDGE_ERROR_PREVIEW_CHARACTERS:]}"
        )
    first, second, ranking = map(float, values)
    if not (1 <= first <= 5 and 1 <= second <= 5 and 1 <= ranking <= 6):
        raise GenRMOutputParseError(f"GenRM scores outside their ranges: {output[-_JUDGE_ERROR_PREVIEW_CHARACTERS:]}")
    return first, second, ranking


# =============================================================================
# Response API Object Extraction
# =============================================================================


def extract_from_response_obj(response_obj: Dict[str, Any]) -> Tuple[str, str]:
    """Extract reasoning and output text from a Response API object.

    Parses the nested Response API structure to find:
    - Reasoning content from "reasoning" type items
    - Output text from "message" type items with "output_text" content

    Args:
        response_obj: Raw Response API object with "output" field

    Returns:
        Tuple of (reasoning_content, output_text)
    """
    reasoning_content = ""
    output_text = ""

    if not isinstance(response_obj, dict):
        return reasoning_content, output_text

    output = response_obj.get("output", [])
    if not isinstance(output, list):
        return reasoning_content, output_text

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "reasoning":
            # Extract from summary field
            summary = item.get("summary", [])
            if isinstance(summary, list):
                for s in summary:
                    if isinstance(s, dict) and "text" in s:
                        reasoning_content += s.get("text", "")

        elif item_type == "message":
            # Extract from content field
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        output_text += c.get("text", "")

    return reasoning_content, output_text


def extract_output_text(response_obj: Dict[str, Any]) -> str:
    """Extract only the output text (final answer) from Response API object.

    Args:
        response_obj: Raw Response API object

    Returns:
        The output text, or "None" if empty/missing
    """
    _, output = extract_from_response_obj(response_obj)

    if not output or not output.strip():
        return EMPTY_OUTPUT_PLACEHOLDER

    return output


# =============================================================================
# Length-Based Bonuses
# =============================================================================


def apply_length_bonuses(
    scores: List[float],
    response_objs: List[Dict[str, Any]],
    reasoning_bonus: float,
    answer_bonus: float,
    top_percentile: float,
    group_reasoning_length_penalty_coeff: float,
    group_answer_length_penalty_coeff: float,
) -> Tuple[List[float], List[float]]:
    """Apply length-based bonuses/penalties to scores.

    Two types of adjustments:
    1. Top-performer bonuses: Shortest reasoning/answer among top scorers gets bonus
    2. Group-relative penalties: Scores adjusted based on relative length within group

    Args:
        scores: Base scores to adjust (modified in place)
        response_objs: Response API objects to extract lengths from
        reasoning_bonus: Bonus for shortest reasoning among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Fraction of top scorers eligible for bonuses (0.0-1.0)
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty

    Returns:
        Tuple of (adjusted_scores, bonuses_per_response)
    """
    num_responses = len(response_objs)

    if num_responses < 2:
        return scores, [0.0] * len(scores)

    # Extract lengths from Response API objects
    bonuses = [0.0] * num_responses
    reasoning_lengths: List[int] = []
    answer_lengths: List[int] = []

    for obj in response_objs:
        reasoning, answer = extract_from_response_obj(obj)
        reasoning_lengths.append(len(reasoning.strip()))
        answer_lengths.append(len(answer.strip()))

    logger.debug(f"Reasoning lengths: {reasoning_lengths}")
    logger.debug(f"Answer lengths: {answer_lengths}")

    # Determine top percentile threshold
    sorted_scores = sorted(scores, reverse=True)
    threshold_idx = max(0, int(len(scores) * top_percentile) - 1)
    top_threshold = sorted_scores[threshold_idx]

    # Bonus for shortest non-empty reasoning among top performers
    if reasoning_bonus > 0:
        valid = [(i, length) for i, length in enumerate(reasoning_lengths) if length > 0]
        if valid:
            idx, length = min(valid, key=lambda x: x[1])
            if scores[idx] >= top_threshold:
                scores[idx] += reasoning_bonus
                bonuses[idx] += reasoning_bonus
                logger.debug(f"Reasoning bonus +{reasoning_bonus} to response {idx} (len={length})")

    # Bonus for shortest non-empty answer among top performers
    if answer_bonus > 0:
        valid = [(i, length) for i, length in enumerate(answer_lengths) if length > 0]
        if valid:
            idx, length = min(valid, key=lambda x: x[1])
            if scores[idx] >= top_threshold:
                scores[idx] += answer_bonus
                bonuses[idx] += answer_bonus
                logger.debug(f"Answer bonus +{answer_bonus} to response {idx} (len={length})")

    # Group-relative length adjustment (shorter = higher weight, zero-centered)
    if group_reasoning_length_penalty_coeff > 0 or group_answer_length_penalty_coeff > 0:
        reasoning_weights = _compute_length_weights(reasoning_lengths)
        answer_weights = _compute_length_weights(answer_lengths)

        for idx in range(num_responses):
            reasoning_adj = reasoning_weights[idx] * group_reasoning_length_penalty_coeff
            answer_adj = answer_weights[idx] * group_answer_length_penalty_coeff
            total_adj = reasoning_adj + answer_adj

            if total_adj != 0:
                scores[idx] += total_adj
                bonuses[idx] += total_adj
                logger.debug(
                    f"Length adjustment {total_adj:+.4f} to response {idx} "
                    f"(reasoning={reasoning_adj:+.4f}, answer={answer_adj:+.4f})"
                )

    return scores, bonuses


def _compute_length_weights(lengths: List[int]) -> List[float]:
    """Compute zero-centered weights where shorter = higher weight.

    Args:
        lengths: List of lengths

    Returns:
        List of weights, zero-centered (sum to ~0)
    """
    max_len, min_len = max(lengths), min(lengths)

    if max_len == min_len:
        return [0.0] * len(lengths)

    span = max_len - min_len
    # Shorter = weight closer to 1, longer = weight closer to 0
    raw_weights = [1.0 - ((length - min_len) / span) for length in lengths]
    # Zero-center
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [w - mean_weight for w in raw_weights]


# =============================================================================
# Score Aggregation
# =============================================================================


def aggregate_scores(
    comparison_results: List[Tuple[float, float, float]],
    comparison_metadata: List[Tuple[int, int, int]],
    response_objs: List[Dict[str, Any]],
    aggregator_method: str,
    default_score: float,
    reasoning_bonus: float,
    answer_bonus: float,
    top_percentile: float,
    group_reasoning_length_penalty_coeff: float,
    group_answer_length_penalty_coeff: float,
) -> Tuple[List[float], Dict[str, float], List[float], List[float]]:
    """Aggregate pairwise comparison results into per-response rewards.

    For "simple_tiebreaker" method:
    - When score_1 == score_2, use ranking to break the tie
    - ranking < 3.5 means response_1 is better → boost score_1, penalize score_2
    - ranking > 3.5 means response_2 is better → boost score_2, penalize score_1

    Args:
        comparison_results: List of (score_1, score_2, ranking) from pairwise comparisons
        comparison_metadata: List of (response_i, response_j, judge_idx) for each comparison
        response_objs: Raw Response API objects for length bonus computation
        aggregator_method: Only "simple_tiebreaker" is supported
        default_score: Default score when no comparisons exist for a response
        reasoning_bonus: Bonus for shortest reasoning among top performers
        answer_bonus: Bonus for shortest answer among top performers
        top_percentile: Percentile threshold for length bonuses
        group_reasoning_length_penalty_coeff: Coefficient for reasoning length penalty
        group_answer_length_penalty_coeff: Coefficient for answer length penalty

    Returns:
        Tuple of:
        - final_scores: Per-response rewards after all adjustments
        - metrics: Aggregation statistics (mean, std, tiebreak rate)
        - base_scores: Scores before length bonuses
        - bonuses: Length bonus applied to each response

    Raises:
        ValueError: If aggregator_method is not supported
    """
    if aggregator_method not in SUPPORTED_AGGREGATOR_METHODS:
        raise ValueError(
            f"Unsupported aggregator_method: '{aggregator_method}'. Supported: {sorted(SUPPORTED_AGGREGATOR_METHODS)}"
        )

    num_responses = len(response_objs)

    # Initialize accumulators
    accumulated_scores = [0.0] * num_responses
    comparison_counts = [0] * num_responses

    # Track metrics
    all_individual_scores: List[float] = []
    tiebreak_count = 0

    # Process each comparison
    for (score_1, score_2, ranking), (i, j, _judge_idx) in zip(comparison_results, comparison_metadata, strict=True):
        all_individual_scores.extend([score_1, score_2])

        # Apply tiebreaker when scores are equal
        if score_1 == score_2:
            tiebreak_count += 1
            # ranking < 3.5 → response_1 better, ranking > 3.5 → response_2 better
            adjustment = RANKING_MIDPOINT - ranking
            score_1 = score_1 + adjustment
            score_2 = score_2 - adjustment

        # Accumulate
        accumulated_scores[i] += score_1
        accumulated_scores[j] += score_2
        comparison_counts[i] += 1
        comparison_counts[j] += 1

    # Compute average scores
    final_scores = [
        accumulated_scores[idx] / comparison_counts[idx] if comparison_counts[idx] > 0 else default_score
        for idx in range(num_responses)
    ]

    # Store base scores before length adjustments
    base_scores = list(final_scores)
    bonuses = [0.0] * num_responses

    # Apply length bonuses if any are configured
    if any(
        [
            reasoning_bonus > 0,
            answer_bonus > 0,
            group_reasoning_length_penalty_coeff > 0,
            group_answer_length_penalty_coeff > 0,
        ]
    ):
        final_scores, bonuses = apply_length_bonuses(
            scores=final_scores,
            response_objs=response_objs,
            reasoning_bonus=reasoning_bonus,
            answer_bonus=answer_bonus,
            top_percentile=top_percentile,
            group_reasoning_length_penalty_coeff=group_reasoning_length_penalty_coeff,
            group_answer_length_penalty_coeff=group_answer_length_penalty_coeff,
        )

    # Compute metrics
    metrics: Dict[str, float] = {}

    if all_individual_scores:
        scores_array = np.array(all_individual_scores)
        metrics["mean_individual_score"] = float(np.mean(scores_array))
        metrics["std_individual_score"] = float(np.std(scores_array))

    if comparison_results:
        metrics["tiebreak_usage_rate"] = tiebreak_count / len(comparison_results)

    return final_scores, metrics, base_scores, bonuses
