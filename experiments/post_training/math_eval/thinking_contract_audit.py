# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate, version-dispatched audit for post-thinking native rewards.

Callers bind source, tokenizer, input membership and the requested treatment before
using this module. Legacy KE4 scoring remains unchanged.
"""

import hashlib
import json
import math
from collections import Counter, deque

from skyrl_gym.envs.data_contracts import get_data_contract
from skyrl_gym.envs.reasoning_gym.scoring import score_response

from experiments.post_training.async_rl_quality_audit import final_assistant_segment
from experiments.post_training.math_eval.scoring import ACCEPTED_STOPS, semantic_answer

PARSER_VERSION = "post-thinking-native-v1"
INTERVENTION_VERSION = "non-agentic-token-intervention-v1"


def verifier_reward(env, answer, gold):
    if env == "reasoning_gym":
        return float(score_response(answer, gold))
    correct = get_data_contract(env).is_correct(answer, gold)
    if env == "aime":
        return 1.0 if correct else -1.0
    if env == "gsm8k":
        return float(correct)
    raise ValueError("Unsupported post-thinking task")


def forced_positions(tokens, config):
    """Independently check every intervention decision against the sampled prefix."""
    if config["protocol"] != INTERVENTION_VERSION or config["kind"] not in ("force_close", "repetition_stop"):
        raise ValueError("Unknown intervention protocol")
    sampled = []
    forced = []
    closed = False
    gram_counts = Counter()
    window = deque()
    n = config["repetition_ngram"]
    width = config["repetition_window"]
    if not 0 < n <= width or not 0 < config["repetition_fraction"] <= 1 or config["force_close_after"] <= 0:
        raise ValueError("Invalid intervention bounds")
    for position, token in enumerate(tokens):
        force_close = config["kind"] == "force_close" and not closed and len(sampled) >= config["force_close_after"]
        repetition = (
            config["kind"] == "repetition_stop"
            and len(sampled) >= width
            and 1 - len(gram_counts) / len(window) >= config["repetition_fraction"]
        )
        if force_close or repetition:
            expected = config["thinking_end_id"] if force_close else config["eos_id"]
            if token != expected or (repetition and position != len(tokens) - 1):
                raise ValueError("Native token sequence violates the prescribed intervention")
            forced.append(position)
        else:
            sampled.append(token)
            if len(sampled) >= n:
                gram = tuple(sampled[-n:])
                window.append(gram)
                gram_counts[gram] += 1
                if len(window) > width - n + 1:
                    old = window.popleft()
                    gram_counts[old] -= 1
                    if not gram_counts[old]:
                        del gram_counts[old]
        closed |= token == config["thinking_end_id"]
    return forced


def audit_post_thinking_row(row, decoder, *, expected_parser, intervention=None, overlong=None):
    """Recompute correctness and shaping; keep the exact dumped native reward.

    Semantic unresolved values remain explicit. A valid parser does not zero a
    correct length-stopped raw reward. Repetition termination is always incomplete.
    """
    if expected_parser != PARSER_VERSION or row.get("parser_protocol") != expected_parser:
        raise ValueError("Parser version differs from the frozen run")
    contract = row["non_agentic_contract"]
    if contract["parser_protocol"] != expected_parser or (intervention is not None and overlong is not None):
        raise ValueError("Mixed parser/treatment protocol")
    tokens = row["response_ids"]
    if row["response_length"] != len(tokens):
        raise ValueError("Response length differs from proven tokens")
    mask = row["policy_action_mask"]
    likelihoods = row["behavior_logprobs"]
    if len(mask) != len(tokens) or likelihoods is None or len(likelihoods) != len(tokens):
        raise ValueError("Missing native action/likelihood evidence")
    if any(not math.isfinite(value) or value > 1e-6 for value in likelihoods):
        raise ValueError("Invalid native behavior logprob")
    stop = row["stop_reason"]
    positions = []
    if intervention is not None:
        trace = contract["intervention"]
        if trace["configuration"] != intervention or trace["protocol"] != INTERVENTION_VERSION:
            raise ValueError("Intervention differs from frozen configuration")
        if decoder.id_to_token(intervention["thinking_end_id"]) != "<|end_think|>" or decoder.id_to_token(
            intervention["eos_id"]
        ) not in {"<|eot_id|>", "<|end_of_text|>"}:
            raise ValueError("Intervention token identity differs from the tokenizer")
        positions = forced_positions(tokens, intervention)
        if positions != trace["forced_positions"] or trace["sampled_token_count"] != len(tokens) - len(positions):
            raise ValueError("Forced/sample token accounting differs")
        if trace["sampled_positions"] != [i for i in range(len(tokens)) if i not in positions]:
            raise ValueError("Sample positions differ")
        repeated = bool(positions) and intervention["kind"] == "repetition_stop"
        if trace["repetition_stopped"] != repeated or stop != (
            "repetition" if repeated else trace["original_engine_stop_reason"]
        ):
            raise ValueError("Effective stop reason differs from native intervention evidence")
    elif "intervention" in contract:
        raise ValueError("Unexpected intervention in parser-only run")
    if mask != [int(i not in positions) for i in range(len(tokens))]:
        raise ValueError("Forced tokens entered policy loss or sampled tokens were removed")
    extras = row["env_extras"]
    gold = extras["reward_model"]["ground_truth"]
    if gold != extras["reward_spec"]["ground_truth"]:
        raise ValueError("Gold channels disagree")
    segment, boundary = final_assistant_segment(decoder, tokens, thinking=True, prompt_tokens=row["prompt_token_ids"])
    env = row["env_class"]
    reward = verifier_reward(env, "" if segment is None else segment, gold)
    legacy = verifier_reward(env, decoder.decode(tokens, skip_special_tokens=True), gold)
    correct = int(reward >= 1) if env == "reasoning_gym" else int(reward == 1)
    completed = correct * int(stop in ACCEPTED_STOPS)
    closed = tokens.count(decoder.token_to_id("<|end_think|>")) == 1
    expected = {
        "boundary_status": boundary,
        "verifier_reward": reward,
        "legacy_full_text_reward": legacy,
        "contract_correct": correct,
        "score_contract_completed": completed,
        "thinking_closed": closed,
    }
    if any(contract[key] != value for key, value in expected.items()):
        raise ValueError("Dumped parser result differs from independent token/verifier recomputation")
    penalty = 0.0
    if overlong is not None:
        maximum, cache, scale = overlong["l_max"], overlong["l_cache"], overlong["penalty_scale"]
        if not 0 < cache <= maximum or not math.isfinite(scale) or scale < 0:
            raise ValueError("Invalid frozen overlong parameters")
        penalty = -scale * min(1.0, max(0.0, (len(tokens) - (maximum - cache)) / cache))
        if row["reward_shaping_version"] != 2 or row["reward_shaping_components"] != {
            "passthrough": 0.0,
            "non_termination": 0.0,
            "overlong": penalty,
            "successful_length": 0.0,
        }:
            raise ValueError("Shaping components differ from the frozen single treatment")
    elif row["reward_shaping_components"] is not None or row["reward_shaping_version"] is not None:
        raise ValueError("Unexpected shaping in the frozen parser/intervention run")
    values = row["score"]
    raw = float(sum(values) if isinstance(values, list) else values)
    if not math.isfinite(raw) or not math.isclose(raw, reward + penalty, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Native optimization reward differs from verifier plus frozen shaping")
    semantic_gold = gold
    if env == "reasoning_gym":
        semantic_gold = json.loads(gold)["entry"]["answer"]
    semantic, status, engine = semantic_answer(segment, boundary, semantic_gold)
    return {
        "parser_protocol": expected_parser,
        "score_contract": raw,
        "contract_correct": correct,
        "score_contract_completed": completed,
        "corrected_verifier_reward": reward,
        "legacy_full_text_reward_diagnostic": legacy,
        "shaping_penalty": penalty,
        "score_semantic": semantic,
        "semantic_status": status,
        "semantic_engine": engine,
        "thinking_closed": closed,
        "stop_reason": stop,
        "response_tokens": len(tokens),
        "forced_positions": positions,
        "answer_segment_sha256": None if segment is None else hashlib.sha256(segment.encode()).hexdigest(),
    }
