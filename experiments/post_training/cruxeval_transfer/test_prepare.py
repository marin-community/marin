# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from experiments.post_training.cruxeval_transfer import rendering
from experiments.post_training.cruxeval_transfer.rendering import token_counts, training_row


def test_training_row_preserves_reasoning_inside_native_assistant_content():
    source = {
        "id": "test/algebra/1.json::sample-0",
        "problem_id": "test/algebra/1.json",
        "messages": [
            {"role": "user", "content": "Problem: Compute two plus three.\nAnswer:"},
            {"role": "assistant", "content": r"Therefore \boxed{5}.", "reasoning_content": " Two plus three is five. "},
        ],
    }
    original = json.dumps(source)
    prepared = training_row(source)
    assert prepared["messages"][1] == {
        "role": "assistant",
        "content": "Two plus three is five.\n<|end_think|>\n\nTherefore \\boxed{5}.",
    }
    assert prepared["messages"][0] == source["messages"][0]
    assert prepared["chat_template_kwargs"] == {"enable_thinking": True}
    assert json.dumps(source) == original


def test_native_mask_excludes_prompt_and_covers_unopened_reasoning(monkeypatch):
    vocab = {
        word: index
        for index, word in enumerate(
            [
                "[UNK]",
                "<|eot_id|>",
                "<|end_think|>",
                "<|start_think|>",
                "user",
                "assistant",
                "Question",
                "Reasoning",
                "Answer",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|eot_id|>", "<|end_think|>", "<|start_think|>"],
        chat_template=(
            "{% for message in messages %}{{ message.role }} "
            "{% if message.role == 'assistant' %}{% generation %}"
            "{{ message.content }}<|eot_id|>{% endgeneration %}"
            "{% else %}{{ message.content }}<|eot_id|> {% endif %}"
            "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )
    monkeypatch.setattr(rendering, "START_THINK_ID", vocab["<|start_think|>"])
    monkeypatch.setattr(rendering, "END_THINK_ID", vocab["<|end_think|>"])
    monkeypatch.setattr(rendering, "END_TURN_ID", vocab["<|eot_id|>"])
    row = training_row(
        {
            "id": "mask-probe",
            "problem_id": "mask-probe",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "reasoning_content": "Reasoning", "content": "Answer"},
            ],
        }
    )
    assert token_counts(row, tokenizer) == (8, 4)
