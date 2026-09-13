# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from experiments.post_training.math_eval.thinking_contract_audit import PARSER_VERSION, audit_post_thinking_row


def example(*, stop="stop", shaping=False):
    words = ["[UNK]", "<|start_think|>", "<|end_think|>", "<|eot_id|>", "####", "41", "42"]
    decoder = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]"))
    decoder.pre_tokenizer = WhitespaceSplit()
    decoder.add_special_tokens([AddedToken(s, special=True) for s in words[1:4]])
    tokens = decoder.encode("#### 41 <|end_think|> #### 42 <|eot_id|>").ids
    penalty = -0.5 if shaping else 0
    row = {
        "parser_protocol": PARSER_VERSION,
        "response_ids": tokens,
        "prompt_token_ids": [1],
        "response_length": len(tokens),
        "policy_action_mask": [1] * len(tokens),
        "behavior_logprobs": [-0.5] * len(tokens),
        "stop_reason": stop,
        "env_class": "gsm8k",
        "env_extras": {"reward_model": {"ground_truth": "42"}, "reward_spec": {"ground_truth": "42"}},
        "score": [0] * (len(tokens) - 1) + [1 + penalty],
        "reward_shaping_components": None,
        "reward_shaping_version": None,
        "non_agentic_contract": {
            "parser_protocol": PARSER_VERSION,
            "boundary_status": "resolved",
            "verifier_reward": 1,
            "legacy_full_text_reward": 0,
            "contract_correct": 1,
            "score_contract_completed": int(stop == "stop"),
            "thinking_closed": True,
        },
    }
    config = None
    if shaping:
        row["reward_shaping_components"] = {
            "passthrough": 0.0,
            "non_termination": 0.0,
            "overlong": penalty,
            "successful_length": 0.0,
        }
        row["reward_shaping_version"] = 2
        config = {"l_max": len(tokens) + 2, "l_cache": 4, "penalty_scale": 1}
    return row, decoder, config


@pytest.mark.parametrize("stop", ["stop", "length"])
@pytest.mark.parametrize("shaping", [False, True])
def test_correct_post_thinking_reward_and_completed_score_remain_separate(stop, shaping):
    row, decoder, config = example(stop=stop, shaping=shaping)
    result = audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION, overlong=config)
    assert result["score_contract"] == (0.5 if shaping else 1)
    assert result["contract_correct"] == 1
    assert result["score_contract_completed"] == int(stop == "stop")
    assert result["legacy_full_text_reward_diagnostic"] == 0
    assert result["score_semantic"] == 1


@pytest.mark.parametrize("poison", ["protocol", "correctness", "reward", "mask", "likelihood", "shaping"])
def test_changed_dump_contract_or_action_evidence_rejected(poison):
    row, decoder, config = example()
    if poison == "protocol":
        row["parser_protocol"] = "legacy"
    elif poison == "correctness":
        row["non_agentic_contract"]["contract_correct"] = 0
    elif poison == "reward":
        row["score"][-1] = 0
    elif poison == "mask":
        row["policy_action_mask"][0] = 0
    elif poison == "likelihood":
        row["behavior_logprobs"][0] = float("nan")
    else:
        row["reward_shaping_version"] = 2
    with pytest.raises(ValueError):
        audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION, overlong=config)


def test_forced_close_is_nonterminal_and_excluded_from_policy_actions():
    row, decoder, _ = example()
    config = {
        "protocol": "non-agentic-token-intervention-v1",
        "kind": "force_close",
        "thinking_end_id": 2,
        "eos_id": 3,
        "force_close_after": 2,
        "repetition_window": 256,
        "repetition_ngram": 16,
        "repetition_fraction": 0.5,
    }
    row["policy_action_mask"][2] = 0
    row["non_agentic_contract"]["intervention"] = {
        "configuration": config,
        "protocol": config["protocol"],
        "forced_positions": [2],
        "sampled_positions": [0, 1, 3, 4, 5],
        "sampled_token_count": 5,
        "repetition_stopped": False,
        "original_engine_stop_reason": "stop",
    }
    result = audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION, intervention=config)
    assert result["score_contract_completed"] == 1 and result["forced_positions"] == [2]
    changed = copy.deepcopy(row)
    changed["policy_action_mask"][2] = 1
    with pytest.raises(ValueError, match="policy loss"):
        audit_post_thinking_row(changed, decoder, expected_parser=PARSER_VERSION, intervention=config)


@pytest.mark.parametrize("tokens,status", [([4, 5], "missing_thinking_end"), ([2, 4, 6, 2], "multiple_thinking_ends")])
def test_unresolved_answer_boundary_preserves_explicit_semantic_status(tokens, status):
    row, decoder, _ = example()
    row.update(
        response_ids=tokens,
        response_length=len(tokens),
        policy_action_mask=[1] * len(tokens),
        behavior_logprobs=[-0.5] * len(tokens),
        score=[0] * len(tokens),
    )
    row["non_agentic_contract"].update(
        boundary_status=status,
        verifier_reward=0,
        contract_correct=0,
        score_contract_completed=0,
        thinking_closed=False,
        legacy_full_text_reward=int(status == "multiple_thinking_ends"),
    )
    result = audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION)
    assert result["score_semantic"] is None and result["semantic_status"] == status
    assert result["score_contract_completed"] == 0


@pytest.mark.parametrize("poison", [None, "premature", "missing_eos", "mask"])
def test_repetition_termination_replays_first_eligible_window(poison):
    row, decoder, _ = example(stop="repetition")
    config = dict(
        protocol="non-agentic-token-intervention-v1",
        kind="repetition_stop",
        thinking_end_id=2,
        eos_id=3,
        force_close_after=3072,
        repetition_window=256,
        repetition_ngram=16,
        repetition_fraction=0.5,
    )
    tokens = [6] * 256 + [3]
    if poison == "premature":
        tokens = [*tokens[:-2], 3]
    elif poison == "missing_eos":
        tokens[-1] = 6
    row.update(
        response_ids=tokens,
        response_length=len(tokens),
        policy_action_mask=[1] * (len(tokens) - 1) + [int(poison == "mask")],
        behavior_logprobs=[-0.5] * len(tokens),
        score=[0] * len(tokens),
    )
    row["non_agentic_contract"].update(
        boundary_status="missing_thinking_end",
        verifier_reward=0,
        legacy_full_text_reward=0,
        contract_correct=0,
        score_contract_completed=0,
        thinking_closed=False,
        intervention=dict(
            configuration=config,
            protocol=config["protocol"],
            forced_positions=[len(tokens) - 1],
            sampled_positions=list(range(len(tokens) - 1)),
            sampled_token_count=len(tokens) - 1,
            repetition_stopped=True,
            original_engine_stop_reason="stop",
        ),
    )
    if poison is not None:
        with pytest.raises(ValueError):
            audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION, intervention=config)
    else:
        result = audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION, intervention=config)
        assert result["stop_reason"] == "repetition" and result["score_contract_completed"] == 0
        assert result["forced_positions"] == [256] and result["score_semantic"] is None


@pytest.mark.parametrize(
    "env,answer,expected",
    [("aime", "41", -1.0), ("reasoning_gym", "42 reasoning", 2 / 12), ("reasoning_gym", "42", 1.0)],
)
def test_task_native_signed_fractional_and_full_credit_dispatch(env, answer, expected):
    row, decoder, _ = example()
    decoder.add_tokens(["Answer:", "reasoning"])
    tokens = decoder.encode("<|end_think|> Answer: " + answer + " <|eot_id|>").ids
    gold = (
        "42"
        if env == "aime"
        else json.dumps(
            dict(task="chain_sum", entry=dict(question="41 + 1", answer="42", metadata=dict(source_dataset="chain_sum")))
        )
    )
    row.update(
        env_class=env,
        response_ids=tokens,
        response_length=len(tokens),
        policy_action_mask=[1] * len(tokens),
        behavior_logprobs=[-0.5] * len(tokens),
        score=[0] * (len(tokens) - 1) + [expected],
    )
    row["env_extras"] = {"reward_model": {"ground_truth": gold}, "reward_spec": {"ground_truth": gold}}
    row["non_agentic_contract"].update(
        verifier_reward=expected,
        legacy_full_text_reward=expected,
        contract_correct=int(expected == 1),
        score_contract_completed=int(expected == 1),
    )
    result = audit_post_thinking_row(row, decoder, expected_parser=PARSER_VERSION)
    assert result["score_contract"] == expected and result["corrected_verifier_reward"] == expected
    assert result["contract_correct"] == int(expected == 1) and result["score_contract_completed"] == int(expected == 1)
