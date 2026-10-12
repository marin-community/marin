# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rolloutengine.closure_repair import ClosureTargetKind, verified_closure_target
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.structure_closure.benchmark import (
    ClosureCase,
    Family,
    score_case,
    syntax_state,
    task_cases,
)
from experiments.post_training.structure_closure.token_data import tokenized_case_record


def test_conditional_target_keeps_prompt_tokens_when_joint_bpe_would_merge():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['"abcdef" instructions ASSISTANT:'] * 10,
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = "{{ messages[0]['content'] }}\nASSISTANT:"
    case = ClosureCase(
        "case", "task", "train", Family.STRING, "continue_required", "instructions", '"abc', '"abcdef"', "source"
    )
    record = tokenized_case_record(case, tokenizer)
    prompt = "instructions\nASSISTANT:" + case.prefix
    assert tokenizer.decode(record["prompt_token_ids"], clean_up_tokenization_spaces=False) == prompt
    assert (
        tokenizer.decode(record["prompt_token_ids"] + record["target_token_ids"], clean_up_tokenization_spaces=False)
        == "instructions\nASSISTANT:" + case.expected
    )
    assert not record["canonical_joint_encoding_matches"]


@pytest.mark.parametrize("family", [family for family in Family if family != Family.REPEATED_OPENER])
def test_verified_repair_uses_exact_conditional_tokens_across_formats(family):
    cases = task_cases("sample_task", [{"search": {"query": 'quote=" slash=\\'}}], "train", "source")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [case.expected for case in cases] * 10,
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    case = next(case for case in cases if case.family == family and case.control == "close_required")

    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    prompt = encode("ASSISTANT:\n" + case.prefix)
    target = verified_closure_target(
        prompt,
        [],
        structure_prefix=case.prefix,
        inspect=lambda value: syntax_state(value, family),
        verify_complete=lambda value: score_case(case, value[len(case.prefix) :]).success,
        encode=encode,
        decode=decode,
    )
    assert target is not None
    assert target.kind == ClosureTargetKind.MISSING_SUFFIX
    assert decode(prompt + list(target.repair_token_ids)) == "ASSISTANT:\n" + case.expected
