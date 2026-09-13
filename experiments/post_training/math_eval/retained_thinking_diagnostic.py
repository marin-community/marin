# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-thinking diagnostics on byte-bound legacy responses, without relabelling them."""

import hashlib
import math

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.async_rl_quality_audit import final_assistant_segment
from experiments.post_training.math_eval.scoring import ACCEPTED_STOPS
from experiments.post_training.math_eval.thinking_contract_audit import verifier_reward

DIAGNOSTIC_VERSION = "legacy-retained-postthinking-diagnostic-v1"
STAGES = {"initial_k4_pilot", "initial_k4_complement", "fresh_k8_legacy_extremes"}


def _diagnostic(tokens, prompt_tokens, stop, env, gold, decoder):
    segment, boundary = final_assistant_segment(decoder, tokens, thinking=True, prompt_tokens=prompt_tokens)
    reward = verifier_reward(env, "" if segment is None else segment, gold)
    correct = int(reward >= 1) if env == "reasoning_gym" else int(reward == 1)
    return {
        "boundary_status": boundary,
        "verifier_reward_diagnostic": reward,
        "contract_correct_diagnostic": correct,
        "completed_correct_diagnostic": correct * int(stop in ACCEPTED_STOPS),
        "effective_stop_diagnostic": stop,
        "answer_segment_sha256": None if segment is None else hashlib.sha256(segment.encode()).hexdigest(),
    }


def diagnose_retained_row(native, record, item, decoder, *, stage):
    """Join an original audited record, then score its retained token sequence.

    The caller independently binds whole native/records/manifest/tokenizer bytes.
    Original semantic results are preserved, including unresolved values. The
    diagnostic does not claim new native rewards, action masks or likelihoods.
    """
    if stage not in STAGES or record["model"] != "snowball" or native.get("parser_protocol") is not None:
        raise ValueError("Expected a declared legacy Snowball rating stage")
    if native.get("non_agentic_contract") is not None:
        raise ValueError("New native parser evidence cannot be relabelled as legacy")
    tokens, prompt = native["response_ids"], native["prompt_token_ids"]
    rewards = native["score"]
    raw = float(sum(rewards) if isinstance(rewards, list) else rewards)
    extras = native["env_extras"]
    if (
        native["row_ordinal"] != record["row_ordinal"]
        or str(native["uid"]) != record["uid"]
        or extras["extra_info"]["prompt_sha256"] != record["prompt_sha256"]
        or record["prompt_sha256"] != item["prompt_sha256"]
        or item["split"] != "train"
        or native["env_class"] != item["env_class"]
        or native["data_source"] != record["bin"]
        or record["bin"] != item["bin"]
        or any(extras[key]["ground_truth"] != item["gold"] for key in ("reward_model", "reward_spec"))
        or audit.canonical_sha(tokens) != record["response_ids_sha256"]
        or audit.canonical_sha(prompt) != record["prompt_token_ids_sha256"]
        or len(tokens) != native["response_length"]
        or len(tokens) != record["response_tokens"]
        or not 0 < len(tokens) <= 8192
        or native["stop_reason"] != record["stop_reason"]
        or not math.isfinite(raw)
        or raw != record["score_contract"]
        or decoder.decode(tokens, skip_special_tokens=False) != native["output_response"]
    ):
        raise ValueError("Legacy response differs from its original audited identity")
    full = _diagnostic(tokens, prompt, native["stop_reason"], item["env_class"], item["gold"], decoder)
    cut = len(tokens) > 4096
    censored = _diagnostic(
        tokens[:4096],
        prompt,
        "synthetic_prefix_cutoff" if cut else native["stop_reason"],
        item["env_class"],
        item["gold"],
        decoder,
    )
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "stage": stage,
        "prompt_sha256": item["prompt_sha256"],
        "bin": item["bin"],
        "row_ordinal": record["row_ordinal"],
        "uid": record["uid"],
        "response_ids_sha256": record["response_ids_sha256"],
        "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
        "source_record_sha256": audit.canonical_sha(record),
        "score_contract": record["score_contract"],
        "original_contract_correct": record["contract_correct"],
        "original_score_contract_completed": record["score_contract_completed"],
        "original_score_semantic": record["score_semantic"],
        "original_semantic_status": record["semantic_status"],
        "original_semantic_engine": record["semantic_engine"],
        "original_stop_reason": record["stop_reason"],
        "original_response_tokens": len(tokens),
        "retained_full_response": full,
        "censored_prefix_4096": {"artificial_cutoff": cut, **censored},
        "actual_corrected_native_generation": False,
        "actual_4096_serving_equivalence": False,
    }
