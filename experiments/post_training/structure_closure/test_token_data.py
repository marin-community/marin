# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat, score_closure

from experiments.post_training.structure_closure.benchmark import (
    ClosureCase,
    Family,
    task_cases,
)
from experiments.post_training.structure_closure.token_data import ToolPrompt, tokenized_case_record


def test_native_tool_prompts_preserve_content_and_continuation_controls():
    cases = task_cases("sample_task", [{"search": {"query": 'quote=" slash=\\'}}], "train", "source")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [case.expected for case in cases],
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = (
        "{% if tools %}TOOLS={{ tools | tojson }}\n{% endif %}"
        "{% for message in messages %}"
        "{% if message.tool_calls is defined %}"
        "<tool_call>\n{{ message.tool_calls[0].function | tojson }}\n</tool_call>"
        "{% else %}{{ message.content }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}\nASSISTANT:\n{% endif %}"
    )
    for case in cases:
        native = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.NATIVE_DECLARED)
        compact = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
        if case.family not in (Family.TOOL, Family.REPEATED_OPENER):
            assert native == compact
            continue
        prompt = tokenizer.decode(native["prompt_token_ids"], clean_up_tokenization_spaces=False)
        target = tokenizer.decode(native["target_token_ids"], clean_up_tokenization_spaces=False)
        assert prompt.startswith('TOOLS=[{"type": "function"')
        assert native["prefix"].startswith("<tool_call>\n")
        assert native["prefix"] + target == native["expected"]
        assert json.loads(native["expected"].removeprefix("<tool_call>").removesuffix("</tool_call>")) == json.loads(
            case.expected.removeprefix("<tool_call>").removesuffix("</tool_call>")
        )
        assert score_closure(native["prefix"] + target, native["expected"], ClosureFormat.TOOL).success
        if case.control == "close_required":
            assert target == "</tool_call>"
        else:
            assert not score_closure(native["prefix"] + "</tool_call>", native["expected"], ClosureFormat.TOOL).success
            assert native["instruction"].endswith(native["expected"])


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
    record = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
    prompt = "instructions\nASSISTANT:" + case.prefix
    assert tokenizer.decode(record["prompt_token_ids"], clean_up_tokenization_spaces=False) == prompt
    assert (
        tokenizer.decode(record["prompt_token_ids"] + record["target_token_ids"], clean_up_tokenization_spaces=False)
        == "instructions\nASSISTANT:" + case.expected
    )
    assert not record["canonical_joint_encoding_matches"]
