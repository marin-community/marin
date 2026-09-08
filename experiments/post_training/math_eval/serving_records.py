# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-token requests and native-score records for inference-only difficulty ratings."""

import hashlib
import math
from collections import defaultdict

from skyrl_gym.envs.data_contracts import get_data_contract
from skyrl_gym.envs.reasoning_gym.scoring import score_response

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.contract import CONTRACT_IDS, render_prompt
from experiments.post_training.math_eval.pool import MODEL_TEMPLATES, prompt_hash
from experiments.post_training.math_eval.rate import MODEL_PROFILES


def completion_request(item, decoder, *, model, samples, api_model):
    """Send frozen token IDs; the endpoint must not apply a second chat template.

    The engine/global seed belongs to the separately audited serving configuration.
    No per-request seed is silently introduced here.
    """
    if model not in MODEL_PROFILES or type(samples) is not int or not 1 <= samples <= 8 or not api_model:
        raise ValueError("Invalid rating model, sample count, or served model identifier")
    if prompt_hash(item["problem"]) != item["prompt_sha256"]:
        raise ValueError("Rating question differs from its frozen identity")
    template = MODEL_TEMPLATES[model]
    if item[f"prompt_template_id_{model}"] != template.template_id:
        raise ValueError("Rating question has a different frozen template")
    prompt = decoder.encode(render_prompt(item["problem"], item["env_class"], template), add_special_tokens=False).ids
    if not prompt or len(prompt) > MODEL_PROFILES[model]["max_prompt_tokens"]:
        raise ValueError("Rating prompt exceeds its predeclared eligibility cap")
    return {
        "model": api_model,
        "prompt": prompt,
        "n": samples,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "min_tokens": 1,
        "ignore_eos": False,
        "max_tokens": MODEL_PROFILES[model]["max_response_tokens"],
        "return_token_ids": True,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "include_stop_str_in_output": True,
        "add_special_tokens": False,
        "truncate_prompt_tokens": None,
        "stop": None,
        "stream": False,
        "echo": False,
    }


def _native_score(text, env, gold):
    correct = get_data_contract(env).is_correct(text, gold)
    if env == "reasoning_gym":
        score = float(score_response(text, gold))
        correct = score >= 1.0
    elif env == "aime":
        score = 1.0 if correct else -1.0
    elif env == "gsm8k":
        score = float(correct)
    else:
        raise ValueError("Unsupported rating environment")
    if not math.isfinite(score):
        raise ValueError("Native rating score is not finite")
    return score, bool(correct)


def completion_rows(item, request, response, decoder, *, model, question_index):
    """Validate a complete serving response before producing any scored rows.

    These are raw engine tokens, with no trajectory padding or appended EOS.
    Original stop-token details and request/response identities are retained.
    """
    if type(question_index) is not int or question_index < 0:
        raise ValueError("Invalid global question ordinal")
    expected = completion_request(item, decoder, model=model, samples=request["n"], api_model=request["model"])
    if request != expected:
        raise ValueError("Issued request differs from the frozen rating protocol")
    if response.get("model") != request["model"] or not isinstance(response.get("id"), str) or not response["id"]:
        raise ValueError("Serving response model or request identity differs")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != request["n"]:
        raise ValueError("Serving response lacks the complete requested sample count")
    indices = [choice.get("index") for choice in choices]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(request["n"])):
        raise ValueError("Serving choices are duplicated or incomplete")
    rows = []
    for choice in sorted(choices, key=lambda choice: choice["index"]):
        tokens = choice.get("token_ids")
        if (
            choice.get("prompt_token_ids") != request["prompt"]
            or not isinstance(tokens, list)
            or any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) > request["max_tokens"]
            or choice.get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("Serving token identity, count or termination is invalid")
        native_text = decoder.decode(tokens, skip_special_tokens=True)
        if choice.get("text") != native_text:
            raise ValueError("Serving text differs from the qualified token renderer")
        score, correct = _native_score(native_text, item["env_class"], item["gold"])
        rows.append(
            {
                "uid": item["prompt_sha256"],
                "row_ordinal": question_index * request["n"] + choice["index"],
                "token_provenance": "raw_engine_response",
                "generator_engine_index": None,
                "prompt_token_ids": request["prompt"],
                "response_ids": tokens,
                "prompt_token_ids_sha256": audit.canonical_sha(request["prompt"]),
                "response_ids_sha256": audit.canonical_sha(tokens),
                "response_length": len(tokens),
                "input_prompt": decoder.decode(request["prompt"], skip_special_tokens=False),
                "output_response": decoder.decode(tokens, skip_special_tokens=False),
                "score": score,
                "native_contract_correct": correct,
                "stop_reason": choice["finish_reason"],
                "engine_stop_reason": choice.get("stop_reason"),
                "env_class": item["env_class"],
                "env_extras": {
                    "reward_model": {"ground_truth": item["gold"]},
                    "reward_spec": {"ground_truth": item["gold"]},
                    "extra_info": {
                        "prompt_sha256": item["prompt_sha256"],
                        "prompt_template_id": MODEL_TEMPLATES[model].template_id,
                        "contract": CONTRACT_IDS[item["env_class"]],
                    },
                },
                "data_source": item["bin"],
                "generation_request_sha256": audit.canonical_sha(request),
                "generation_response_id": response["id"],
                "engine_output_text_sha256": hashlib.sha256(native_text.encode()).hexdigest(),
            }
        )
    usage = response.get("usage", {})
    prompt_count, output_count = len(request["prompt"]), sum(row["response_length"] for row in rows)
    if (
        usage.get("prompt_tokens") != prompt_count
        or usage.get("completion_tokens") != output_count
        or usage.get("total_tokens") != prompt_count + output_count
    ):
        raise ValueError("Serving usage counters differ from returned token IDs")
    return rows


def serving_metrics(rows, *, samples):
    """Retain native score diagnostics and separate binary completion channels."""
    if not rows:
        raise ValueError("Cannot summarize an empty serving sample")
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["data_source"].replace("/", "_")].append(row)
    metrics = {}
    for name, items in groups.items():
        uids = defaultdict(list)
        for item in items:
            uids[item["uid"]].append(item)
        if any(len(values) != samples for values in uids.values()):
            raise ValueError("Serving metrics lack the prescribed per-question sample count")
        count = len(items)
        values = {
            "avg_score": sum(item["score"] for item in items) / count,
            f"pass_at_{samples}": sum(any(item["score"] > 0 for item in values) for values in uids.values()) / len(uids),
            "contract_correct": sum(item["native_contract_correct"] for item in items) / count,
            "contract_completed": (
                sum(item["native_contract_correct"] and item["stop_reason"] == "stop" for item in items) / count
            ),
        }
        values.update(
            audit.evaluation_response_metrics(
                [(len(item["response_ids"]), item["score"], item["stop_reason"]) for item in items]
            )
        )
        metrics.update({f"eval/{name}/{key}": value for key, value in values.items()})
    return metrics
