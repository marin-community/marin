# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cohort-relative circular GenRM rewards from NVIDIA's Ultra recipe."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from skyrl_gym.envs.nemotron_ultra.genrm_utils import (
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
)
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


def response_object(assistant_message: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the Response-API fields consumed by NVIDIA's GenRM utilities."""
    output = []
    reasoning = assistant_message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        output.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})
    content = assistant_message.get("content")
    answer = content if isinstance(content, str) else ""
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": answer}],
        }
    )
    return {"output": output}


def grade_genrm_group(
    *,
    conversation_history: list[dict[str, Any]],
    response_objects: list[dict[str, Any]],
    principle: str,
    judge: OpenAIJudge,
    config: dict[str, Any],
) -> tuple[list[float], dict[str, float]]:
    if config.get("verifyit_enabled", False):
        from skyrl_gym.envs.nemotron_ultra.genrm_verifyit import grade_genrm_verifyit

        return grade_genrm_verifyit(
            conversation_history=conversation_history,
            response_objects=response_objects,
            principle=principle,
            judge=judge,
            config=config,
        )
    default_score = float(config.get("default_score", 3.0))
    pairs = generate_comparison_pairs("circular", len(response_objects))
    max_workers = int(config.get("max_concurrent_comparisons", len(pairs)))
    if max_workers < 1:
        raise ValueError("max_concurrent_comparisons must be at least 1")

    def compare(pair: tuple[int, int]) -> tuple[float, float, float]:
        first, second = pair
        metadata = {
            "response_1": extract_output_text(response_objects[first]),
            "response_2": extract_output_text(response_objects[second]),
            "principle": principle,
        }
        attempts = int(config.get("genrm_parse_retries", 1)) + 1
        last_error = None
        for attempt in range(attempts):
            try:
                output = judge.generate_response(
                    conversation_history,
                    metadata=metadata,
                    max_output_tokens=int(config.get("max_output_tokens", 24576)),
                    temperature=float(config.get("temperature", 1.0)),
                    top_p=float(config.get("top_p", 0.95)),
                )
                return parse_genrm_output(output, strict_json=bool(config.get("verifyit_strict_json", False)))
            except Exception as error:
                last_error = error
                logger.warning("GenRM comparison attempt {} failed: {}", attempt + 1, error)
                if attempt + 1 < attempts:
                    time.sleep(float(config.get("genrm_parse_retry_sleep_seconds", 0.2)))
        raise RuntimeError(f"GenRM comparison unavailable after bounded retries: {last_error}") from last_error

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pairs))) as executor:
        comparisons = list(executor.map(compare, pairs))
    metadata = [(first, second, 0) for first, second in pairs]
    rewards, metrics, _, _ = aggregate_scores(
        comparison_results=comparisons,
        comparison_metadata=metadata,
        response_objs=response_objects,
        aggregator_method="simple_tiebreaker",
        default_score=default_score,
        reasoning_bonus=float(config.get("reasoning_bonus", 0.5)),
        answer_bonus=float(config.get("answer_bonus", 0.5)),
        top_percentile=float(config.get("top_percentile", 0.2)),
        group_reasoning_length_penalty_coeff=float(config.get("group_reasoning_length_penalty_coeff", 0.1)),
        group_answer_length_penalty_coeff=float(config["group_answer_length_penalty_coeff"]),
    )
    return rewards, metrics
