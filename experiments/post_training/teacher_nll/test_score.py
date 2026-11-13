# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.post_training.teacher_nll import score


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_post(logprobs):
    def post(url, json, timeout, headers):
        ids = json["prompt"]
        prompt_logprobs = [None] + [
            {f"token_id:{t}": {"logprob": logprobs[i], "rank": 1}} for i, t in enumerate(ids[1:])
        ]
        return FakeResponse(
            {"choices": [{"prompt_token_ids": [f"token_id:{t}" for t in ids], "prompt_logprobs": prompt_logprobs}]}
        )

    return post


def test_weight_i_scores_token_i_plus_one(monkeypatch):
    # logprob of token j (j >= 1) is logprobs[j - 1].
    logprobs = [-0.1, -0.2, -0.3, -0.4]
    monkeypatch.setattr(score.requests, "post", fake_post(logprobs))
    row = {"cell": "c", "row": 0, "input_ids": [5, 6, 7, 8, 9], "loss_weights": [0, 1, 1, 0, 0]}
    result = score.score_row("url", "m", row)
    # Weights 1 and 2 score tokens 2 and 3: logprobs -0.2 and -0.3.
    assert math.isclose(result["nll_sum"], 0.5)
    assert result["loss_tokens"] == 2
    assert result["tokens"] == 5
    assert math.isclose(result["text_nll_sum"], 0.5)


def test_text_nll_skips_special_targets(monkeypatch):
    logprobs = [-0.1, -0.2, -0.3]
    monkeypatch.setattr(score.requests, "post", fake_post(logprobs))
    row = {"cell": "c", "row": 0, "input_ids": [5, 6, 128009, 7], "loss_weights": [1, 1, 1, 0]}
    result = score.score_row("url", "m", row)
    assert math.isclose(result["nll_sum"], 0.6)
    assert math.isclose(result["text_nll_sum"], 0.4)
    assert result["text_loss_tokens"] == 2


def test_last_weight_must_be_zero():
    row = {"cell": "c", "row": 0, "input_ids": [5, 6], "loss_weights": [0, 1]}
    with pytest.raises(ValueError, match="shifted"):
        score.score_row("url", "m", row)
