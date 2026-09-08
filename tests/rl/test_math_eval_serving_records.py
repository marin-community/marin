# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import reasoning_gym

from experiments.post_training.math_eval.contract import QWEN
from experiments.post_training.math_eval.pool import prompt_hash
from experiments.post_training.math_eval.serving import MAX_FAILURE_RECEIPT_BYTES, completion_rows_with_failure_receipt
from experiments.post_training.math_eval.serving_records import completion_request, completion_rows, serving_metrics


class Decoder:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(char) for char in text])

    def decode(self, tokens, *, skip_special_tokens):
        return "".join(chr(token) if token else ("" if skip_special_tokens else "<eos>") for token in tokens)


def fixture(env="aime"):
    decoder = Decoder()
    item = {
        "problem": "What is 2 + 3?",
        "prompt_sha256": prompt_hash("What is 2 + 3?"),
        "prompt_template_id_qwen": QWEN.template_id,
        "gold": "5",
        "env_class": env,
        "bin": "fixture/math",
    }
    request = completion_request(item, decoder, model="qwen", samples=8, api_model="frozen-rating-model")
    choices = []
    for index in range(8):
        text = ("#### " if env == "gsm8k" else "Answer: ") + ("5" if index < 4 else "7")
        choices.append(
            {
                "index": index,
                "text": text,
                "token_ids": [*decoder.encode(text).ids, 0],
                "prompt_token_ids": request["prompt"],
                "finish_reason": "length" if index in {2, 3} else "stop",
                "stop_reason": None,
            }
        )
    prompt_count = len(request["prompt"])
    response_count = sum(len(choice["token_ids"]) for choice in choices)
    response = {
        "id": "completion-fixture",
        "model": request["model"],
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_count,
            "completion_tokens": response_count,
            "total_tokens": prompt_count + response_count,
        },
    }
    return decoder, item, request, response


def test_renderer_mismatch_preserves_exact_failed_response_and_still_raises(tmp_path):
    decoder, item, request, response = fixture()
    response["choices"][3]["text"] += "\ufffd"
    original = deepcopy(response)
    path = tmp_path / "validation-failure.json"
    with pytest.raises(ValueError, match="qualified token renderer"):
        completion_rows_with_failure_receipt(
            item,
            request,
            response,
            decoder,
            model="qwen",
            question_index=13,
            failure_uri=str(path),
            provenance={"source_commit": "reviewed", "attempt_uid": "native-attempt"},
        )
    receipt = json.loads(path.read_text())
    assert receipt["rating_valid"] is False
    assert receipt["response"] == original
    assert receipt["request"] == request and receipt["item"] == item
    assert receipt["question_index"] == 13
    assert receipt["provenance"]["attempt_uid"] == "native-attempt"
    assert not (tmp_path / "dumped_evals").exists()
    with pytest.raises(RuntimeError, match="overwrite"):
        completion_rows_with_failure_receipt(
            item,
            request,
            response,
            decoder,
            model="qwen",
            question_index=13,
            failure_uri=str(path),
            provenance={},
        )
    assert json.loads(path.read_text()) == receipt


def test_valid_responses_keep_scores_without_writing_failure_receipt(tmp_path):
    decoder, item, request, response = fixture()
    path = tmp_path / "validation-failure.json"
    rows = completion_rows_with_failure_receipt(
        item,
        request,
        response,
        decoder,
        model="qwen",
        question_index=2,
        failure_uri=str(path),
        provenance={},
    )
    assert rows == completion_rows(item, request, response, decoder, model="qwen", question_index=2)
    assert not path.exists()


def test_oversized_failure_is_not_truncated_or_written_as_a_complete_receipt(tmp_path):
    decoder, item, request, response = fixture()
    response["choices"][0]["text"] = "x" * (MAX_FAILURE_RECEIPT_BYTES + 1)
    path = tmp_path / "validation-failure.json"
    with pytest.raises(RuntimeError, match="diagnostic bound"):
        completion_rows_with_failure_receipt(
            item,
            request,
            response,
            decoder,
            model="qwen",
            question_index=0,
            failure_uri=str(path),
            provenance={},
        )
    assert not path.exists()


@pytest.mark.parametrize("env", ["gsm8k", "aime"])
def test_serving_records_preserve_raw_tokens_and_signed_reward_while_gating_completion(env):
    decoder, item, request, response = fixture(env)
    rows = completion_rows(item, request, response, decoder, model="qwen", question_index=2)
    assert [row["row_ordinal"] for row in rows] == list(range(16, 24))
    assert all(row["token_provenance"] == "raw_engine_response" for row in rows)
    assert all(row["output_response"].endswith("<eos>") for row in rows)
    assert all(row["generator_engine_index"] is None for row in rows)
    assert rows[-1]["score"] == (-1 if env == "aime" else 0)
    metrics = serving_metrics(rows, samples=8)
    assert metrics["eval/all/contract_correct"] == 0.5
    assert metrics["eval/all/contract_completed"] == 0.25
    assert metrics["eval/all/pass_at_8"] == 1
    assert "seed" not in request and request["temperature"] == request["top_p"] == 1


@pytest.mark.parametrize(
    "poison", ["text", "prompt", "missing_tokens", "choice_index", "usage", "model", "stop", "protocol", "question"]
)
def test_serving_responses_fail_on_incomplete_or_mismatched_native_evidence(poison):
    decoder, item, request, response = fixture()
    if poison == "text":
        response["choices"][0]["text"] += "<eos>"
    elif poison == "prompt":
        response["choices"][0]["prompt_token_ids"] = [4, 5]
    elif poison == "missing_tokens":
        del response["choices"][0]["token_ids"]
    elif poison == "choice_index":
        response["choices"][1]["index"] = 0
    elif poison == "usage":
        response["usage"]["completion_tokens"] -= 1
    elif poison == "model":
        response["model"] = "foreign-model"
    elif poison == "stop":
        response["choices"][0]["finish_reason"] = "abort"
    elif poison == "protocol":
        request["temperature"] = 0
    else:
        item["problem"] = "What is 2 + 4?"
    with pytest.raises(ValueError):
        completion_rows(item, request, response, decoder, model="qwen", question_index=0)


def test_missing_sample_prevents_publishing_metrics():
    decoder, item, request, response = fixture()
    rows = completion_rows(item, request, response, decoder, model="qwen", question_index=0)
    with pytest.raises(ValueError, match="sample count"):
        serving_metrics(rows[:-1], samples=8)
    altered = deepcopy(response)
    altered["choices"].pop()
    with pytest.raises(ValueError, match="sample count"):
        completion_rows(item, request, altered, decoder, model="qwen", question_index=0)


def test_fractional_native_reasoning_gym_credit_does_not_become_binary_correctness():
    decoder, item, _request, _response = fixture()
    entry = reasoning_gym.create_dataset("chain_sum", size=1, seed=101)[0]
    item.update(
        problem=entry["question"],
        prompt_sha256=prompt_hash(entry["question"]),
        env_class="reasoning_gym",
        gold=json.dumps({"task": "chain_sum", "entry": entry}),
    )
    request = completion_request(item, decoder, model="qwen", samples=8, api_model="frozen-rating-model")
    choices = []
    for index in range(8):
        text = "Answer: " + entry["answer"] + (" junk" if index >= 4 else "")
        choices.append(
            {
                "index": index,
                "text": text,
                "token_ids": [*decoder.encode(text).ids, 0],
                "prompt_token_ids": request["prompt"],
                "finish_reason": "stop",
                "stop_reason": None,
            }
        )
    output_count = sum(len(choice["token_ids"]) for choice in choices)
    response = {
        "id": "rg-response",
        "model": request["model"],
        "choices": choices,
        "usage": {
            "prompt_tokens": len(request["prompt"]),
            "completion_tokens": output_count,
            "total_tokens": len(request["prompt"]) + output_count,
        },
    }
    rows = completion_rows(item, request, response, decoder, model="qwen", question_index=0)
    assert all(0 < row["score"] < 1 and not row["native_contract_correct"] for row in rows[4:])
    metrics = serving_metrics(rows, samples=8)
    assert metrics["eval/all/contract_correct"] == metrics["eval/all/contract_completed"] == 0.5


def test_prompt_overflow_fails_before_requesting_generation():
    decoder, item, _request, _response = fixture()
    item["problem"] = "x" * 4096
    item["prompt_sha256"] = prompt_hash(item["problem"])
    with pytest.raises(ValueError, match="eligibility cap"):
        completion_request(item, decoder, model="qwen", samples=8, api_model="frozen-rating-model")
