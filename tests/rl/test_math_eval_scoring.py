# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native reward mappings, answer extraction and token boundary regressions."""

import json
from typing import ClassVar

import pytest
import reasoning_gym

import experiments.post_training.math_eval.scoring as scoring
from experiments.post_training.math_eval.lm_eval_regex import scores as lm_eval_scores
from experiments.post_training.math_eval.scoring import ACCEPTED_STOPS, score_row, semantic_answer


class Decoder:
    """Character-token I/O fixture with distinct real boundary token identities."""

    markers: ClassVar[dict[str, int]] = {"<|start_think|>": 100000, "<|end_think|>": 100001}

    def token_to_id(self, text):
        return self.markers.get(text)

    def id_to_token(self, token):
        return next((text for text, value in self.markers.items() if value == token), chr(token))

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(self.id_to_token(token) for token in tokens)


def row(response, *, env="aime", gold="5", raw=-1.0, stop="stop", tokens=None, prompt=None):
    tokens = list(map(ord, response)) if tokens is None else tokens
    return {
        "output_response": response,
        "env_class": env,
        "score": raw,
        "stop_reason": stop,
        "response_ids": tokens,
        "prompt_token_ids": prompt or [],
        "response_length": len(tokens),
        "uid": "0",
        "env_extras": {
            "reward_model": {"ground_truth": gold},
            "reward_spec": {"ground_truth": gold},
            "extra_info": {"prompt_sha256": "fixture"},
        },
    }


def test_gsm8k_first_hash_contract_conflicts_with_final_answer():
    result = score_row(row("#### 5\n#### 7", env="gsm8k", raw=1), Decoder(), model="qwen", thinking=False)
    assert result.score_contract == 1
    assert result.contract_correct
    assert result.score_semantic is None
    assert result.semantic_status == "conflicting"


@pytest.mark.parametrize("response", ["Answer: 5\n" + "x" * 310, r"\boxed{5}"])
def test_aime_contract_and_semantic_format_gap(response):
    result = score_row(row(response), Decoder(), model="qwen", thinking=False)
    assert result.score_contract == -1
    assert not result.contract_correct
    assert result.score_semantic == 1
    assert result.format_gap == 1


def test_math_verify_equivalence_is_independent_of_answer_format():
    value, status, engine = semantic_answer("The answer is 0.5", "resolved", r"\frac{1}{2}")
    assert (value, status, engine) == (1.0, "parsed", "math-verify==0.9.0")


def test_unclosed_thinking_is_unresolved_even_with_a_rewarded_answer():
    result = score_row(
        row("Answer: 5", raw=1, stop="length", prompt=[100000]), Decoder(), model="snowball", thinking=True
    )
    assert result.score_contract == 1 and result.contract_correct
    assert result.score_contract_completed == 0
    assert result.truncated and not result.thinking_closed
    assert result.score_semantic is None and result.semantic_status == "missing_thinking_end"


@pytest.mark.parametrize("stop", [*sorted(ACCEPTED_STOPS), "length", "unknown", None])
@pytest.mark.parametrize("correct", [True, False])
def test_completion_is_exact_correctness_times_accepted_stop(stop, correct):
    response = "Answer: 5" if correct else "Answer: 9"
    raw = 1 if correct else -1
    result = score_row(row(response, raw=raw, stop=stop), Decoder(), model="qwen", thinking=False)
    assert result.score_contract == raw
    assert result.score_contract_completed == int(correct and stop in ACCEPTED_STOPS)
    assert result.score_contract_completed <= result.contract_correct


def test_wrong_signed_reward_channel_is_rejected():
    with pytest.raises(ValueError, match="native verifier mapping"):
        score_row(row("Answer: 9", raw=0), Decoder(), model="qwen", thinking=False)


def test_fractional_procedural_reward_is_preserved_without_becoming_exact_correctness():
    entry = reasoning_gym.create_dataset(
        "chain_sum", size=1, seed=101, min_terms=2, max_terms=2, min_digits=1, max_digits=2
    )[0]
    gold = json.dumps({"task": "chain_sum", "entry": entry})
    result = score_row(
        row("Answer: 85 extra", env="reasoning_gym", gold=gold, raw=0.25), Decoder(), model="qwen", thinking=False
    )
    assert result.score_contract == 0.25
    assert not result.contract_correct
    assert result.score_contract_completed == 0


def test_procedural_correctness_uses_the_approved_at_least_one_threshold(monkeypatch):

    entry = reasoning_gym.create_dataset(
        "chain_sum", size=1, seed=101, min_terms=2, max_terms=2, min_digits=1, max_digits=2
    )[0]
    gold = json.dumps({"task": "chain_sum", "entry": entry})
    monkeypatch.setattr(scoring, "score_response", lambda *_args: 1.25)
    result = score_row(
        row("Answer: 85 extra", env="reasoning_gym", gold=gold, raw=1.25), Decoder(), model="qwen", thinking=False
    )
    assert result.contract_correct and result.score_contract_completed == 1
    assert result.score_contract == 1.25


@pytest.mark.parametrize(
    "response,gold,expected",
    [
        ("The answer is 5. Later it is 7.", "5", (1.0, 0.0)),
        ("The answer is $1,234.", "1234", (0.0, 1.0)),
        ("#### 5", "5", (0.0, 1.0)),
        ("The answer is 0.50.", "0.5", (0.0, 0.0)),
    ],
)
def test_lm_eval_channels_keep_their_own_regex_and_string_semantics(response, gold, expected):
    assert lm_eval_scores(response, gold) == expected


@pytest.mark.parametrize("end_count,closed", [(1, True), (2, False)])
def test_snowball_thinking_closure_uses_token_counts(end_count, closed):
    tokens = [*map(ord, "Reasoning."), *([100001] * end_count), *map(ord, "Answer: 5")]
    result = score_row(
        row(Decoder().decode(tokens), raw=1, prompt=[100000], tokens=tokens), Decoder(), model="snowball", thinking=True
    )
    assert result.thinking_closed is closed
    assert result.score_semantic == (1.0 if closed else None)


def test_closed_thinking_remains_closed_when_later_role_tokens_invalidate_semantics(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr(Decoder, "markers", {**decoder.markers, "<|im_start|>": 100002})
    tokens = [100001, *map(ord, "Answer: 5"), 100002]
    result = score_row(
        row(decoder.decode(tokens), raw=1, prompt=[100000], tokens=tokens), decoder, model="snowball", thinking=True
    )
    assert result.thinking_closed
    assert result.score_semantic is None
    assert result.semantic_status == "role_or_thinking_continuation"


@pytest.mark.parametrize("reward", [[0.0, 0.0, 1.0], [1.0]])
def test_production_token_reward_arrays_retain_native_sequence_score(reward):
    result = score_row(row("Answer: 5", raw=reward), Decoder(), model="qwen", thinking=False)
    assert result.score_contract == 1.0
    assert result.contract_correct and result.score_contract_completed == 1.0


def test_shaped_token_reward_sum_cannot_impersonate_native_contract_score():
    with pytest.raises(ValueError, match="native verifier mapping"):
        score_row(row("Answer: 5", raw=[0.25, 1.0]), Decoder(), model="qwen", thinking=False)
