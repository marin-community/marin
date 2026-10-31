# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rendered-byte parity and assistant loss masks against the pinned Qwen tokenizers."""

import hashlib
import json
import re

import pytest
from transformers import AutoTokenizer

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.stage0 import qwen3_native_template, qwen35_native_template
from experiments.post_training.stage0.qwen_native_lengths import audit_row
from experiments.post_training.stage0.serve_check import checked_templates


@pytest.fixture(params=[("qwen3_moe", qwen3_native_template), ("qwen3_5_moe", qwen35_native_template)])
def native_template(request):
    family, module = request.param
    repo, revision = FAMILY_TOKENIZER_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == module.SOURCE_SHA256
    return tokenizer, module.CHAT_TEMPLATE


@pytest.mark.parametrize("followup", [False, True])
def test_native_template_preserves_tool_history_and_masks_only_assistants(native_template, followup):
    tokenizer, template = native_template
    messages = [
        {"role": "system", "content": "Use the shell."},
        {"role": "user", "content": "List files."},
        {
            "role": "assistant",
            "content": "Listing now.",
            "reasoning_content": "Inspect the working directory.",
            "tool_calls": [{"function": {"name": "shell", "arguments": {"command": "ls"}}}],
        },
        {"role": "tool", "content": "file.txt"},
        {"role": "assistant", "content": "One file."},
    ]
    if followup:
        messages += [
            {"role": "user", "content": "What was its name?"},
            {"role": "assistant", "content": "file.txt", "reasoning_content": "Read the earlier output."},
        ]
    original = tokenizer.apply_chat_template(messages, tokenize=False)
    assert tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False) == original
    encoded = tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    assert encoded["input_ids"] == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
    offsets = tokenizer(original, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    spans = [
        (m.start(1), m.end(1)) for m in re.finditer(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>\n)", original, re.S)
    ]
    expected = [int(any(start <= left < right <= end for start, end in spans)) for left, right in offsets]
    assert encoded["assistant_masks"] == expected
    trained = tokenizer.decode([token for token, mask in zip(encoded["input_ids"], expected, strict=True) if mask])
    assert "</think>" in trained and "<|im_end|>" in trained and "<tool_call>" in trained
    assert "<tool_response>" not in trained and "<|im_start|>" not in trained
    assert ("Inspect the working directory." in original) is not followup
    family = next(key for key, (repo, _) in FAMILY_TOKENIZER_REVISIONS.items() if repo == tokenizer.name_or_path)
    audit = audit_row(family, {"row_id": "tool-history", "messages": messages})
    assert audit.tokens == len(encoded["input_ids"])
    assert audit.reasoning_turns_dropped == int(followup)
    assert audit.empty_think_blocks == int(not followup)
    assert audit.structured_tool_calls == 1


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_native_generation_prompt_is_unchanged(native_template, enable_thinking):
    tokenizer, template = native_template
    messages = [{"role": "user", "content": "What is two plus two?"}]
    options = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    assert tokenizer.apply_chat_template(messages, chat_template=template, **options) == tokenizer.apply_chat_template(
        messages, **options
    )


def test_native_audit_separates_final_turn_reasoning_from_dropped_history(native_template):
    family = "qwen3_moe" if native_template[1] == qwen3_native_template.CHAT_TEMPLATE else "qwen3_5_moe"
    messages = [
        {"role": "user", "content": "What is two plus two?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Add two and two."},
        {"role": "user", "content": "And times three?"},
        {"role": "assistant", "content": "12", "reasoning_content": "Multiply four by three."},
    ]
    audit = audit_row(family, {"row_id": "two-turn-thinking", "messages": messages})
    assert audit.reasoning_turns_dropped == 1
    assert audit.final_reasoning
    assert not audit.final_reasoning_dropped


def test_exported_native_template_rejects_stale_processor_override(tmp_path):
    expected = qwen35_native_template.CHAT_TEMPLATE
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": expected}))
    processor_template = tmp_path / "chat_template.jinja"
    processor_template.write_text(expected)
    assert checked_templates(str(tmp_path), expected) == {
        "tokenizer_config.json": expected,
        "chat_template.jinja": expected,
    }
    processor_template.write_text("{{ messages[0]['content'] }}")
    with pytest.raises(ValueError, match=r"chat_template\.jinja"):
        checked_templates(str(tmp_path), expected)
