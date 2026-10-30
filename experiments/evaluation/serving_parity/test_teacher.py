# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.evaluation.serving_parity import teacher
from experiments.evaluation.serving_parity.capture import echo_contract
from experiments.evaluation.serving_parity.teacher import normalize_teacher_response


def test_echo_generated_suffix_is_not_mistaken_for_prompt_logprob():
    response = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["token_id:1", "token_id:2", "token_id:3"],
                    "token_logprobs": [None, -0.2, -0.4],
                    "top_logprobs": [None, {"token_id:2": -0.2}, {"token_id:3": -0.4}],
                }
            }
        ],
        "usage": {"completion_tokens": 1},
    }
    with pytest.raises(ValueError, match="Echo token IDs"):
        normalize_teacher_response(response, [1, 2])
    response["choices"][0]["logprobs"] = {k: v[:2] for k, v in response["choices"][0]["logprobs"].items()}
    response["usage"]["completion_tokens"] = 1
    rows = normalize_teacher_response(response, [1, 2])
    assert rows[1].logprob == -0.2
    assert rows[1].top1_token_id == 2


def test_loglikelihood_generation_cap_regression_is_rejected():
    response = {
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "choices": [
            {
                "logprobs": {
                    "tokens": ["a", "b", "c"],
                    "token_logprobs": [None, -0.2, -0.3],
                }
            }
        ],
    }
    echo_contract({"echo": True, "max_tokens": 1}, response)
    with pytest.raises(ValueError, match="unscored token"):
        echo_contract({"echo": True, "max_tokens": 256}, response)


def test_failed_teacher_contract_preserves_raw_response(tmp_path, monkeypatch):

    raw = {"choices": [{"logprobs": {"tokens": ["token_id:9"], "token_logprobs": [None], "top_logprobs": [None]}}]}
    monkeypatch.setattr(teacher.requests, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: raw))
    item = dict(prompt="x", completion="", input_sha256=hashlib.sha256(b"x").hexdigest(), token_ids=[1], id=303)
    output = tmp_path / "teacher"
    with pytest.raises(ValueError, match="Echo token IDs"):
        teacher.collect_teacher("http://localhost/v1", "model", item, output, concurrency=1)
    assert json.loads((output / "0.raw.json").read_text()) == raw
