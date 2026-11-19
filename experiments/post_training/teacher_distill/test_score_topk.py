# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from experiments.post_training.teacher_distill import score_topk


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _entry(target: int, rank_of_target: int | None):
    """21 entries when the target is outside the top 20, else 20; logprob of token t is -t/10."""
    tokens = list(range(1, 21))
    if rank_of_target is None:
        tokens.append(target)
    return {f"token_id:{t}": {"logprob": -t / 10, "rank": t} for t in tokens}


def test_score_doc_keeps_sorted_top20_and_target(monkeypatch):
    ids = [128000, 5, 7, 99, 128001]
    weights = [0.0, 1.0, 0.0, 1.0, 0.0]  # score tokens at 2 (id 7) and 4 (EOS)
    prompt_logprobs = [None, _entry(5, 1), _entry(7, 7), _entry(99, None), _entry(128001, None)]
    payload = {"choices": [{"prompt_token_ids": [f"token_id:{t}" for t in ids], "prompt_logprobs": prompt_logprobs}]}
    monkeypatch.setattr(score_topk.requests, "post", lambda *a, **k: _Response(payload))
    out = score_topk.score_doc(("u", "m", {}, {"row": 3, "input_ids": ids, "loss_weights": weights}))
    assert out["positions"].tolist() == [1, 3]
    np.testing.assert_array_equal(out["ids"][0], np.arange(1, 21))
    assert np.all(np.diff(out["logprobs"], axis=1) <= 0)
    assert math.isclose(out["target"][0], -0.7, rel_tol=1e-6)
    assert math.isclose(out["target"][1], -12800.1, rel_tol=1e-6)


def test_score_doc_rejects_a_changed_prompt(monkeypatch):
    ids = [1, 2, 3]
    payload = {"choices": [{"prompt_token_ids": ["token_id:1", "token_id:2", "token_id:4"], "prompt_logprobs": []}]}
    monkeypatch.setattr(score_topk.requests, "post", lambda *a, **k: _Response(payload))
    with pytest.raises(ValueError, match="echoed prompt"):
        score_topk.score_doc(("u", "m", {}, {"row": 0, "input_ids": ids, "loss_weights": [1.0, 0.0, 0.0]}))
