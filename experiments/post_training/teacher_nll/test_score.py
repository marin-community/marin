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


def fake_post(logprobs, ranks=None, others=None):
    """``others[i]``: extra (token id, logprob) entries returned beside target i (the top-k list)."""

    def post(url, json, timeout, headers):
        ids = json["prompt"]
        prompt_logprobs = [None]
        for i, t in enumerate(ids[1:]):
            entry = {f"token_id:{t}": {"logprob": logprobs[i], "rank": ranks[i] if ranks else 1}}
            for k, lp in (others or {}).get(i, []):
                entry[f"token_id:{k}"] = {"logprob": lp, "rank": 0}
            prompt_logprobs.append(entry)
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


def test_rank_sums_cap_and_text_twin(monkeypatch):
    # Targets: token 6 (rank 3), token 128009 (special, rank 250), token 7 (rank 100).
    monkeypatch.setattr(score.requests, "post", fake_post([-0.1, -0.2, -0.3], ranks=[3, 250, 100]))
    row = {"cell": "c", "row": 0, "input_ids": [5, 6, 128009, 7], "loss_weights": [1, 1, 1, 0]}
    result = score.score_row("url", "m", row)
    assert math.isclose(result["nll_sum"], 0.6)
    assert result["rank_sum_capped"] == 3 + 100 + 100
    assert result["rank_sum_uncapped"] == 3 + 250 + 100
    assert result["rank_over_cap_tokens"] == 1
    assert result["text_rank_sum_capped"] == 3 + 100
    assert result["text_rank_over_cap_tokens"] == 0
    assert "strict_rank_sum_capped" not in result


def test_existing_fields_unchanged(monkeypatch):
    monkeypatch.setattr(score.requests, "post", fake_post([-0.1, -0.2, -0.3, -0.4]))
    row = {"cell": "c", "row": 0, "input_ids": [5, 6, 7, 8, 9], "loss_weights": [0, 1, 1, 0, 0]}
    result = score.score_row("url", "m", row)
    old_keys = [
        "cell",
        "source",
        "row",
        "family",
        "task_id",
        "trial",
        "kind",
        "tokens",
        "loss_tokens",
        "nll_sum",
        "text_loss_tokens",
        "text_nll_sum",
    ]
    assert list(result)[: len(old_keys)] == old_keys


def test_strict_rank_from_top_k_excludes_ties(monkeypatch):
    # Target 0 (token 6, lp -1.0): one strictly higher (-0.5) and one tie (-1.0); vLLM's >= rank is 3.
    others = {0: [(40, -0.5), (41, -1.0), (42, -2.0)]}
    monkeypatch.setattr(score.requests, "post", fake_post([-1.0, -0.1], ranks=[3, 1], others=others))
    seen = {}

    def spy(url, json, timeout, headers):
        seen["prompt_logprobs"] = json["prompt_logprobs"]
        return fake_post([-1.0, -0.1], ranks=[3, 1], others=others)(url, json, timeout, headers)

    monkeypatch.setattr(score.requests, "post", spy)
    row = {"cell": "c", "row": 0, "input_ids": [5, 6, 7], "loss_weights": [1, 1, 0]}
    result = score.score_row("url", "m", row, top=100)
    assert seen["prompt_logprobs"] == 100
    assert result["rank_sum_capped"] == 3 + 1
    assert result["strict_rank_sum_capped"] == 2 + 1


def test_strict_rank_caps_when_all_top_k_are_higher(monkeypatch):
    others = {0: [(1000 + k, -0.01 * k) for k in range(100)]}
    monkeypatch.setattr(score.requests, "post", fake_post([-5.0], ranks=[400], others=others))
    row = {"cell": "c", "row": 0, "input_ids": [5, 6], "loss_weights": [1, 0]}
    result = score.score_row("url", "m", row, top=100)
    assert result["strict_rank_sum_capped"] == 100
    assert result["rank_over_cap_tokens"] == 1


def test_missing_rank_is_an_error(monkeypatch):
    def post(url, json, timeout, headers):
        ids = json["prompt"]
        return FakeResponse(
            {
                "choices": [
                    {
                        "prompt_token_ids": [f"token_id:{t}" for t in ids],
                        "prompt_logprobs": [None, {f"token_id:{ids[1]}": {"logprob": -0.1}}],
                    }
                ]
            }
        )

    monkeypatch.setattr(score.requests, "post", post)
    with pytest.raises(KeyError):
        score.score_row("url", "m", {"cell": "c", "row": 0, "input_ids": [5, 6], "loss_weights": [1, 0]})
