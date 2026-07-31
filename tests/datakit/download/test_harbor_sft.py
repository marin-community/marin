# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
from datasets import load_dataset
from marin.datakit.download.harbor_sft import (
    HarborSftHarness,
    RejectionReason,
    convert_harbor_row,
    load_harbor_sft_manifest,
    resolve_teacher_tokenizer,
)
from transformers import AutoTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    }
]


class MappingTokenizer:
    def __init__(self, decoded: dict[tuple[int, ...], str]):
        self.decoded = decoded

    def decode(self, token_ids, *, skip_special_tokens: bool):
        assert skip_special_tokens is False
        return self.decoded[tuple(token_ids)]


def _literal_row() -> tuple[dict, MappingTokenizer]:
    prompt = (
        "<|im_start|>system\n"
        "# Tools\n<tools>\n"
        f"{json.dumps(TOOLS[0])}\n"
        "</tools>\n<IMPORTANT>Render tool calls with the active template.</IMPORTANT>\n"
        "You are opencode. Solve the user's task."
        "<|im_end|>\n"
        "<|im_start|>user\nFix the parser.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    first_completion = (
        "I will inspect it.</think>\n"
        "<tool_call><function=bash>"
        "<parameter=command>\npytest -q\n</parameter>"
        "<parameter=timeout>\n30\n</parameter>"
        "</function></tool_call><|im_end|>"
    )
    second_completion = (
        "<think>\nThe failure is fixed.</think>\nDone.\n"
        '<tool_call>\n{"name": "bash", "arguments": {"command": "git diff"}}\n</tool_call>'
        "<|im_end|>"
    )
    tokenizer = MappingTokenizer(
        {
            (1,): prompt,
            (2,): first_completion,
            (3,): second_completion,
        }
    )
    row = {
        "task": "parser-regression",
        "prompt_token_ids": [[1], [10]],
        "completion_token_ids": [[2], [3]],
        "conversations": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "lossy first completion"},
            {"role": "user", "content": "<tool_response>\n1 failed\n</tool_response>"},
            {"role": "assistant", "content": "lossy second completion"},
        ],
    }
    return row, tokenizer


def test_opencode_literals_reconstruct_tools_aware_sft_record():
    row, tokenizer = _literal_row()

    result = convert_harbor_row(row, HarborSftHarness.OPENCODE_LITERALS, tokenizer)

    assert result.rejection is None
    assert result.record == {
        "messages": [
            {
                "role": "system",
                "content": "You are opencode. Solve the user's task.",
                "tool_calls": [],
            },
            {"role": "user", "content": "Fix the parser.", "tool_calls": []},
            {
                "role": "assistant",
                "content": "<think>\nI will inspect it.</think>",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "pytest -q", "timeout": 30}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "1 failed", "tool_calls": []},
            {
                "role": "assistant",
                "content": "<think>\nThe failure is fixed.</think>\nDone.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "git diff"}',
                        },
                    }
                ],
            },
        ],
        "tools": json.dumps(TOOLS, ensure_ascii=False),
        "task": "parser-regression",
        "num_turns": 5,
        "num_tool_calls": 2,
    }


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {"prompt_token_ids": [], "completion_token_ids": []},
            RejectionReason.MISSING_LITERALS,
        ),
        (
            {"completion_token_ids": [[2]]},
            RejectionReason.ASSISTANT_COMPLETION_MISMATCH,
        ),
    ],
)
def test_opencode_literals_reject_lossy_or_misaligned_rows(change, reason):
    row, tokenizer = _literal_row()
    row.update(change)

    result = convert_harbor_row(row, HarborSftHarness.OPENCODE_LITERALS, tokenizer)

    assert result.record is None
    assert result.rejection is reason


def test_structured_harbor_rows_do_not_require_literal_reconstruction():
    row = {
        "task": "structured-trial",
        "tools": TOOLS,
        "messages": [
            {"role": "system", "content": "Use the available tools."},
            {"role": "user", "content": "List the files."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": {"command": "ls"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "README.md"},
            {"role": "assistant", "content": "README.md is present."},
        ],
    }

    result = convert_harbor_row(row, HarborSftHarness.STRUCTURED, tokenizer=None)

    assert result.rejection is None
    assert result.record["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
    assert result.record["messages"][3]["role"] == "tool"
    assert result.record["tools"] == json.dumps(TOOLS, ensure_ascii=False)
    assert result.record["num_turns"] == 5
    assert result.record["num_tool_calls"] == 1


def test_manifest_applies_adapter_defaults_and_preserves_reproduction_gates(tmp_path):
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "historical-run",
                "harness": "opencode_literals",
                "teacher_tokenizer": "teacher/tokenizer",
                "teacher_tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
                "sources": [
                    {
                        "name": "tasks",
                        "hf_dataset_id": "org/tasks-traces",
                        "revision": "abc1234",
                        "expected_rows": 22,
                    }
                ],
            }
        )
    )

    manifest = load_harbor_sft_manifest(manifest_path)

    assert manifest.name == "historical-run"
    assert manifest.sources[0].harness is HarborSftHarness.OPENCODE_LITERALS
    assert manifest.sources[0].teacher_tokenizer == "teacher/tokenizer"
    assert manifest.sources[0].teacher_tokenizer_revision == "0123456789abcdef0123456789abcdef01234567"
    assert manifest.sources[0].expected_rows == 22


def test_literal_tokenizer_provenance_must_pin_the_model_revision(tmp_path):
    (tmp_path / "tokenizer_provenance.json").write_text(json.dumps({"served_model": "teacher/tokenizer"}))

    with pytest.raises(ValueError, match="does not pin served_model_revision"):
        resolve_teacher_tokenizer(
            str(tmp_path),
            HarborSftHarness.OPENCODE_LITERALS,
            override=None,
            override_revision=None,
        )


def test_grug_reproduction_manifest_covers_the_exact_training_mixture():
    manifest = load_harbor_sft_manifest(
        Path(__file__).parents[3] / "experiments/datakit/manifests/grug_67b_a2b_agentic_sft.json"
    )

    assert len(manifest.sources) == 29
    assert sum(source.expected_rows or 0 for source in manifest.sources) == 77_012
    assert all(len(source.revision) == 40 for source in manifest.sources)
    assert {source.teacher_tokenizer_revision for source in manifest.sources} == {
        "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
    }
    assert {source.name for source in manifest.sources}.isdisjoint(
        {
            "exp_rpt_methods2test-large-v3",
            "exp_rpt_pymethods2test-large",
            "exp_rpt_pymethods2test-v3",
        }
    )


@pytest.mark.data_integration
def test_reproduces_grug_curriculum_hard_training_records():
    """The compact historical oracle must match all 22 records used by the Grug SFT pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-122B-A10B-FP8",
        revision="a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
        trust_remote_code=True,
    )
    rows = load_dataset(
        "penfever/exp_rpt_curriculum-hard-qwen3.5-122b-131k-opencode-traces",
        split="train",
        revision="404f3fc1ff0aad3818fcbd07a4d573892e316ffe",
        streaming=True,
    )
    converted = [
        result.record
        for row in rows
        if (
            result := convert_harbor_row(
                row,
                HarborSftHarness.OPENCODE_LITERALS,
                tokenizer,
            )
        ).record
        is not None
    ]
    canonical = [json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for record in converted]

    assert len(canonical) == 22
    assert hashlib.sha256("\n".join(canonical).encode()).hexdigest() == (
        "f7edd569a88c8563b15cb14643908c5e9aff37ad40742446f8342d52cd391a29"
    )
