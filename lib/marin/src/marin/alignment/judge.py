# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: Judge responses and build preference pairs.

Scores teacher (chosen) and rejected responses against rubrics using an LLM judge,
filters by quality thresholds, and outputs preference pairs in marin format
(sharded JSONL.GZ with chosen/rejected OpenAI chat messages).
"""

from __future__ import annotations

import concurrent.futures
import gzip
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from iris.marin_fs import url_to_fs

from marin.alignment.generate_prompts import _write_sharded_jsonl_gz
from marin.alignment.llm_client import llm_chat_single
from marin.alignment.prompts.judge import build_compliance_judge_prompt, build_judge_system_prompt
from marin.alignment.types import ComplianceResult, Statement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgePairConfig:
    """Configuration for judging and preference pair construction."""

    prompts_path: str
    chosen_responses_path: str
    rejected_responses_path: str
    spec_path: str
    output_path: str

    judge_model: str = "openai/gpt-4.1"
    min_chosen_score: float = 7.0
    min_gap: float = 2.0

    workers: int = 64
    judge_max_tokens: int = 1000


def _load_responses(path: str) -> dict[str, dict[str, Any]]:
    """Load responses keyed by prompt_id. Supports both local and GCS paths."""
    fs, base_path = url_to_fs(path)
    responses: dict[str, dict[str, Any]] = {}
    for shard_file in sorted(fs.glob(f"{base_path}/*.jsonl.gz")):
        with fs.open(shard_file, "rb") as raw_f:
            with gzip.open(raw_f, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    prompt_id = record.get("prompt_id", "")
                    responses[prompt_id] = record
    return responses


def _load_spec(spec_path: str) -> dict[str, Statement]:
    """Load statements from spec JSONL."""
    from marin.alignment.generate_prompts import load_spec

    return load_spec(spec_path)


def _parse_judge_response(content: str) -> dict[str, Any]:
    """Parse the JSON judgment from the judge response."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to extract raw JSON
    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    # Fallback: try parsing the entire content
    return json.loads(content)


def _judge_response(
    statement: Statement,
    user_message: str,
    model_response: str,
    rubric: str | None,
    judge_model: str,
    max_tokens: int,
) -> ComplianceResult:
    """Score a single response against a statement."""
    system_prompt = build_judge_system_prompt()
    user_prompt = build_compliance_judge_prompt(
        statement=statement,
        user_input=user_message,
        model_response=model_response,
        question_rubric=rubric,
    )

    response = llm_chat_single(
        config=judge_model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    try:
        parsed = _parse_judge_response(response.content)
        return ComplianceResult.from_judge_output(parsed, raw_response=response.content)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse judge response: %s", exc)
        return ComplianceResult(
            score=5,
            compliant=False,
            confidence=0.0,
            explanation=f"Parse failure: {exc}",
            raw_response=response.content,
        )


def _process_prompt_pair(
    prompt_id: str,
    chosen_record: dict[str, Any],
    rejected_record: dict[str, Any],
    statements: dict[str, Statement],
    config: JudgePairConfig,
) -> dict[str, Any] | None:
    """Judge and optionally build a preference pair for one prompt."""
    behavior_id = chosen_record.get("behavior_id", "")
    statement = statements.get(behavior_id)
    if not statement:
        logger.warning("No statement found for behavior_id '%s'", behavior_id)
        return None

    user_message = chosen_record.get("user_message", "")
    rubric = chosen_record.get("rubric")
    system_prompt = chosen_record.get("system_prompt", "")

    # Score the chosen (teacher) response
    chosen_responses = chosen_record.get("responses", [])
    if not chosen_responses:
        return None
    chosen_text = chosen_responses[0].get("content", "")

    chosen_result = _judge_response(
        statement=statement,
        user_message=user_message,
        model_response=chosen_text,
        rubric=rubric,
        judge_model=config.judge_model,
        max_tokens=config.judge_max_tokens,
    )

    if chosen_result.score < config.min_chosen_score:
        logger.debug(
            "Filtering out %s: chosen score %.1f < min %.1f",
            prompt_id,
            chosen_result.score,
            config.min_chosen_score,
        )
        return None

    # Score all rejected responses and pick the worst
    rejected_responses = rejected_record.get("responses", [])
    if not rejected_responses:
        return None

    worst_rejected_text = ""
    worst_rejected_score = float("inf")

    for resp in rejected_responses:
        resp_text = resp.get("content", "")
        if not resp_text:
            continue
        result = _judge_response(
            statement=statement,
            user_message=user_message,
            model_response=resp_text,
            rubric=rubric,
            judge_model=config.judge_model,
            max_tokens=config.judge_max_tokens,
        )
        if result.score < worst_rejected_score:
            worst_rejected_score = result.score
            worst_rejected_text = resp_text

    if not worst_rejected_text:
        return None

    # Check gap
    gap = chosen_result.score - worst_rejected_score
    if gap < config.min_gap:
        logger.debug(
            "Filtering out %s: gap %.1f < min %.1f (chosen=%.1f, rejected=%.1f)",
            prompt_id,
            gap,
            config.min_gap,
            chosen_result.score,
            worst_rejected_score,
        )
        return None

    # Build preference pair in marin format
    # Note: system_prompt is the scenario prompt only (no spec guidance)
    chosen_messages: list[dict[str, str]] = []
    rejected_messages: list[dict[str, str]] = []

    if system_prompt:
        chosen_messages.append({"role": "system", "content": system_prompt})
        rejected_messages.append({"role": "system", "content": system_prompt})

    chosen_messages.append({"role": "user", "content": user_message})
    chosen_messages.append({"role": "assistant", "content": chosen_text})

    rejected_messages.append({"role": "user", "content": user_message})
    rejected_messages.append({"role": "assistant", "content": worst_rejected_text})

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }


def judge_and_build_pairs(config: JudgePairConfig) -> None:
    """Judge responses and build preference pairs.

    Loads chosen and rejected responses, scores each against behavior statements
    using an LLM judge, filters by quality thresholds, and writes marin-format
    preference pairs as sharded JSONL.GZ.
    """
    statements = _load_spec(config.spec_path)
    chosen = _load_responses(config.chosen_responses_path)
    rejected = _load_responses(config.rejected_responses_path)
    logger.info("Loaded %d chosen, %d rejected responses", len(chosen), len(rejected))

    # Find prompt_ids present in both
    common_ids = sorted(set(chosen.keys()) & set(rejected.keys()))
    logger.info("Processing %d prompt pairs", len(common_ids))

    pairs: list[dict[str, Any]] = []
    filtered_count = 0
    failures: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as pool:
        future_map = {
            pool.submit(
                _process_prompt_pair,
                prompt_id,
                chosen[prompt_id],
                rejected[prompt_id],
                statements,
                config,
            ): prompt_id
            for prompt_id in common_ids
        }
        for future in concurrent.futures.as_completed(future_map):
            prompt_id = future_map[future]
            try:
                pair = future.result()
                if pair is not None:
                    pairs.append(pair)
                else:
                    filtered_count += 1
            except Exception as exc:
                failures.append(f"{prompt_id}: {exc}")
                logger.error("Judging failed for %s: %s", prompt_id, exc)

    logger.info(
        "Built %d preference pairs (%d filtered, %d failures)",
        len(pairs),
        filtered_count,
        len(failures),
    )

    # Write output as sharded JSONL.GZ (supports both local and GCS paths)
    _write_sharded_jsonl_gz(pairs, config.output_path, shard_size=5000)

    logger.info("Wrote %d preference pairs to %s", len(pairs), config.output_path)
