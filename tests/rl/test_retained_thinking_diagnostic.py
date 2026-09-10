# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.retained_thinking_diagnostic import diagnose_retained_row


def example(*, stop="stop", long=False, closed=True):
    words = ["[UNK]", "<|start_think|>", "<|end_think|>", "<|eot_id|>", "####", "41", "42"]
    decoder = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]"))
    decoder.pre_tokenizer = WhitespaceSplit()
    decoder.add_special_tokens([AddedToken(s, special=True) for s in words[1:4]])
    response = "#### 41 " + ("41 " * 4100 if long else "")
    response += "<|end_think|> #### 42 <|eot_id|>" if closed else "41"
    tokens = decoder.encode(response).ids
    native = {
        "response_ids": tokens,
        "prompt_token_ids": [1],
        "response_length": len(tokens),
        "score": 0.0,
        "stop_reason": stop,
        "uid": "7",
        "row_ordinal": 0,
        "data_source": "g03-gsm8k",
        "env_class": "gsm8k",
        "env_extras": {
            "extra_info": {"prompt_sha256": "question"},
            "reward_model": {"ground_truth": "42"},
            "reward_spec": {"ground_truth": "42"},
        },
        "output_response": decoder.decode(tokens, skip_special_tokens=False),
    }
    record = {
        "model": "snowball",
        "row_ordinal": 0,
        "uid": "7",
        "prompt_sha256": "question",
        "bin": "g03-gsm8k",
        "response_ids_sha256": audit.canonical_sha(tokens),
        "prompt_token_ids_sha256": audit.canonical_sha([1]),
        "response_tokens": len(tokens),
        "stop_reason": stop,
        "score_contract": 0.0,
        "contract_correct": False,
        "score_contract_completed": 0.0,
        "score_semantic": None,
        "semantic_status": "semantic_worker_timeout",
        "semantic_engine": "unresolved",
    }
    item = {"prompt_sha256": "question", "split": "train", "env_class": "gsm8k", "bin": "g03-gsm8k", "gold": "42"}
    return native, record, item, decoder


@pytest.mark.parametrize("stage", ["initial_k4_pilot", "initial_k4_complement", "fresh_k8_legacy_extremes"])
@pytest.mark.parametrize("stop", ["stop", "length"])
def test_legacy_native_score_stays_verbatim_while_corrected_diagnostic_is_separate(stage, stop):
    rows = example(stop=stop)
    before = copy.deepcopy(rows[:3])
    result = diagnose_retained_row(*rows, stage=stage)
    assert result["stage"] == stage and result["score_contract"] == 0
    assert result["retained_full_response"]["contract_correct_diagnostic"] == 1
    assert result["retained_full_response"]["completed_correct_diagnostic"] == int(stop == "stop")
    assert result["original_score_semantic"] is None and result["original_semantic_status"] == "semantic_worker_timeout"
    assert result["actual_corrected_native_generation"] is False
    assert rows[:3] == before


def test_censored_prefix_is_incomplete_even_when_retained_full_response_completed():
    result = diagnose_retained_row(*example(long=True), stage="initial_k4_pilot")
    assert result["retained_full_response"]["completed_correct_diagnostic"] == 1
    prefix = result["censored_prefix_4096"]
    assert prefix["artificial_cutoff"] and prefix["boundary_status"] == "missing_thinking_end"
    assert (
        prefix["completed_correct_diagnostic"] == 0 and prefix["effective_stop_diagnostic"] == "synthetic_prefix_cutoff"
    )
    assert result["actual_4096_serving_equivalence"] is False


def test_missing_thinking_boundary_is_reported_without_native_reward_rewrite():
    result = diagnose_retained_row(*example(closed=False), stage="initial_k4_pilot")
    assert result["retained_full_response"]["boundary_status"] == "missing_thinking_end"
    assert result["retained_full_response"]["completed_correct_diagnostic"] == 0
    assert result["score_contract"] == 0


@pytest.mark.parametrize(
    "poison", ["tokens", "prompt", "gold", "ordinal", "raw_reward", "tiny_raw_reward", "stop", "split", "parser"]
)
def test_contradictory_original_evidence_is_rejected(poison):
    native, record, item, decoder = example()
    if poison == "tokens":
        native["response_ids"][0] = 0
    elif poison == "prompt":
        native["prompt_token_ids"] = [0]
    elif poison == "gold":
        item["gold"] = "41"
    elif poison == "ordinal":
        native["row_ordinal"] = 1
    elif poison == "raw_reward":
        native["score"] = 1
    elif poison == "tiny_raw_reward":
        native["score"] = 1e-13
    elif poison == "stop":
        native["stop_reason"] = "length"
    elif poison == "split":
        item["split"] = "heldout"
    else:
        native["parser_protocol"] = "post-thinking-native-v1"
    with pytest.raises(ValueError):
        diagnose_retained_row(native, record, item, decoder, stage="initial_k4_pilot")
