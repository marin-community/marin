# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve native rewards while scoring exact correctness and completed answers."""

import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any

from math_verify import parse, verify
from skyrl_gym.envs.data_contracts import get_data_contract
from skyrl_gym.envs.reasoning_gym.scoring import score_response
from tokenizers import Tokenizer

from experiments.post_training.async_rl_quality import AnswerStatus, extract_numeric_answer, normalize_numeric_answer
from experiments.post_training.async_rl_quality_audit import final_assistant_segment
from experiments.post_training.math_eval.lm_eval_regex import scores as lm_eval_scores

ACCEPTED_STOPS = frozenset({"complete", "end_turn", "eos", "stop"})
CONTRACT_RULES = {"gsm8k": "gsm8k_first_hash", "aime": "aime_last_answer_300", "reasoning_gym": "rg_last_answer"}
MATH_VERIFY_VERSION = "0.9.0"
SEMANTIC_DEPENDENCIES = {
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "sympy": "1.14.0",
    "mpmath": "1.3.0",
    "antlr4-python3-runtime": "4.11.0",
}


@dataclass(frozen=True)
class ScoredRow:
    prompt_sha256: str
    uid: str
    model: str
    score_contract: float
    contract_correct: bool
    score_contract_completed: float
    contract_rule: str
    score_semantic: float | None
    semantic_status: str
    semantic_engine: str
    truncated: bool
    thinking_closed: bool
    response_tokens: int
    stop_reason: str | None
    format_gap: float | None
    lm_eval_strict: float | None
    lm_eval_flexible: float | None


def semantic_answer(segment: str | None, boundary: str, gold: str) -> tuple[float | None, str, str]:
    """Use conservative scalar extraction, then parsed mathematical equivalence."""
    if segment is None:
        return None, boundary, "unresolved"
    extracted = extract_numeric_answer(segment)
    normalized_gold = normalize_numeric_answer(gold)
    if extracted.status == AnswerStatus.EXTRACTED and normalized_gold is not None:
        return float(extracted.value == normalized_gold), str(extracted.status), "fraction-exact"
    if extracted.status in {AnswerStatus.CONFLICTING, AnswerStatus.AMBIGUOUS, AnswerStatus.ROLE_CONTINUATION}:
        return None, str(extracted.status), "fraction-exact"
    for package, version in SEMANTIC_DEPENDENCIES.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Semantic dependency changed: {package}")

    reference = parse("$" + gold + "$", fallback_mode="no_fallback")
    prediction = parse(segment, fallback_mode="no_fallback")
    if not reference or not prediction:
        return None, "math_verify_no_parse", f"math-verify=={MATH_VERIFY_VERSION}"
    return float(verify(reference, prediction)), "parsed", f"math-verify=={MATH_VERIFY_VERSION}"


def score_row(row: dict[str, Any], decoder: Tokenizer, *, model: str, thinking: bool) -> ScoredRow:
    """Score a proven dump row; reject reward channels inconsistent with its verifier.

    The harness verifies token hashes and frozen prompt identity before this call.
    Native signed/fractional rewards are retained. Completion is a separate binary
    outcome, including the valid zero-correctness/zero-completion truncated case.
    """
    env = row["env_class"]
    if env not in CONTRACT_RULES:
        raise ValueError(f"Unsupported math contract: {env}")
    extras = row["env_extras"]
    gold = extras["reward_model"]["ground_truth"]
    if gold != extras["reward_spec"]["ground_truth"]:
        raise ValueError("Ground-truth channels disagree")
    response = row["output_response"]
    correct = get_data_contract(env).is_correct(response, gold)
    native = score_response(response, gold) if env == "reasoning_gym" else float(correct)
    if env == "reasoning_gym":
        correct = native >= 1.0
    if env == "aime":
        native = 1.0 if correct else -1.0
    rewards = row["score"]
    # Finalized trajectory dumps use token rewards; preserve the same sequence
    # reduction as audit_eval_dump / W&B avg_score, without binarizing it.
    raw = float(sum(rewards) if isinstance(rewards, list) else rewards)
    if not math.isfinite(raw) or not math.isclose(raw, native, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Dump reward does not match the frozen native verifier mapping")
    tokens = row["response_ids"]
    if len(tokens) != row["response_length"]:
        raise ValueError("Response token count changed")
    segment, boundary = final_assistant_segment(
        decoder, tokens, thinking=thinking, prompt_tokens=row["prompt_token_ids"]
    )
    semantic_gold = json.loads(gold)["entry"]["answer"] if env == "reasoning_gym" else gold
    semantic, status, engine = semantic_answer(segment, boundary, semantic_gold)
    stop = row["stop_reason"]
    completed = float(correct and stop in ACCEPTED_STOPS)
    strict, flexible = lm_eval_scores(response, gold) if env == "gsm8k" else (None, None)
    return ScoredRow(
        prompt_sha256=extras["extra_info"]["prompt_sha256"],
        uid=str(row["uid"]),
        model=model,
        score_contract=raw,
        contract_correct=correct,
        score_contract_completed=completed,
        contract_rule=CONTRACT_RULES[env],
        score_semantic=semantic,
        semantic_status=status,
        semantic_engine=engine,
        truncated=stop == "length",
        thinking_closed=not thinking or boundary in {"resolved", "role_or_thinking_continuation"},
        response_tokens=len(tokens),
        stop_reason=stop,
        format_gap=None if semantic is None else semantic - float(correct),
        lm_eval_strict=strict,
        lm_eval_flexible=flexible,
    )
