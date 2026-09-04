# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import load_dataset
from marin.datakit.download.harbor_sft import (
    HarborSftHarness,
    RejectionReason,
    convert_harbor_row,
    detect_harbor_harness,
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
        "agent": "opencode",
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

    result = convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer)

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

    result = convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer)

    assert result.record is None
    assert result.rejection is reason


def test_opencode_literals_reject_malformed_tool_schema():
    row, tokenizer = _literal_row()
    tokenizer.decoded[(1,)] = tokenizer.decoded[(1,)].replace(json.dumps(TOOLS[0]), "{not-json")

    result = convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer)

    assert result.record is None
    assert result.rejection is RejectionReason.INVALID_TOOLS


def test_opencode_literals_reject_malformed_typed_tool_argument():
    row, tokenizer = _literal_row()
    tokenizer.decoded[(2,)] = tokenizer.decoded[(2,)].replace(">\n30\n</parameter>", ">\nthirty\n</parameter>")

    result = convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer)

    assert result.record is None
    assert result.rejection is RejectionReason.INVALID_TOOL_CALLS


def test_terminus_2_conversations_are_literal_sft_ground_truth():
    row = {
        "agent": "terminus-2",
        "task_id": "terminus-trial",
        "messages": [
            {"role": "user", "content": "Lossy projection."},
            {"role": "assistant", "content": "Do not train on this."},
        ],
        "conversations": [
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": '{"cmd":"ls"}'},
            {"role": "user", "content": "Chunk ID: abc123\nProcess exited with code 0\nFinal output:\nREADME.md"},
            {"role": "assistant", "content": "README.md is present."},
        ],
    }

    result = convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer=None)

    assert result.rejection is None
    assert result.record == {
        "messages": [
            {"role": "user", "content": "List the files.", "tool_calls": []},
            {"role": "assistant", "content": '{"cmd":"ls"}', "tool_calls": []},
            {
                "role": "user",
                "content": "Chunk ID: abc123\nProcess exited with code 0\nFinal output:\nREADME.md",
                "tool_calls": [],
            },
            {"role": "assistant", "content": "README.md is present.", "tool_calls": []},
        ],
        "tools": "[]",
        "task": "terminus-trial",
        "num_turns": 4,
        "num_tool_calls": 0,
    }


@pytest.mark.parametrize(
    ("row_factory", "asserted_harness"),
    [
        (_literal_row, HarborSftHarness.TERMINUS_2),
        (
            lambda: (
                {
                    "agent": "terminus-2",
                    "conversations": [
                        {"role": "user", "content": "Do the task."},
                        {"role": "assistant", "content": "Done."},
                    ],
                },
                None,
            ),
            HarborSftHarness.OPENCODE,
        ),
    ],
)
def test_row_harness_assertion_rejects_mismatch(row_factory, asserted_harness):
    row, tokenizer = row_factory()

    with pytest.raises(ValueError, match="harness mismatch"):
        convert_harbor_row(row, asserted_harness, tokenizer)


@pytest.mark.parametrize("agent", [None, "", "unknown-agent"])
def test_row_harness_detection_rejects_missing_or_unknown_agent(agent):
    row, tokenizer = _literal_row()
    row["agent"] = agent

    with pytest.raises(ValueError, match="agent"):
        convert_harbor_row(row, HarborSftHarness.AUTO, tokenizer)


def _write_agent_shard(path: Path, agents: list[str]) -> None:
    pq.write_table(pa.table({"agent": agents}), path)


def test_dataset_harness_detection_requires_one_uniform_known_agent(tmp_path):
    uniform = tmp_path / "uniform"
    uniform.mkdir()
    _write_agent_shard(uniform / "part-0.parquet", ["terminus-2", "terminus-2"])
    _write_agent_shard(uniform / "part-1.parquet", ["terminus-2"])

    assert detect_harbor_harness(str(uniform), HarborSftHarness.AUTO) is HarborSftHarness.TERMINUS_2
    assert detect_harbor_harness(str(uniform), HarborSftHarness.TERMINUS_2) is HarborSftHarness.TERMINUS_2

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    _write_agent_shard(mixed / "part.parquet", ["terminus-2", "opencode"])
    with pytest.raises(ValueError, match="mixed Harbor harnesses"):
        detect_harbor_harness(str(mixed), HarborSftHarness.AUTO)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    _write_agent_shard(unknown / "part.parquet", ["custom-agent"])
    with pytest.raises(ValueError, match="unknown Harbor agent"):
        detect_harbor_harness(str(unknown), HarborSftHarness.AUTO)


def test_dataset_harness_assertion_rejects_mismatch(tmp_path):
    _write_agent_shard(tmp_path / "part.parquet", ["opencode"])

    with pytest.raises(ValueError, match="harness mismatch"):
        detect_harbor_harness(str(tmp_path), HarborSftHarness.TERMINUS_2)


def test_manifest_applies_adapter_defaults_and_preserves_reproduction_gates(tmp_path):
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "historical-run",
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
    assert manifest.sources[0].harness is HarborSftHarness.AUTO
    assert manifest.sources[0].teacher_tokenizer == "teacher/tokenizer"
    assert manifest.sources[0].teacher_tokenizer_revision == "0123456789abcdef0123456789abcdef01234567"
    assert manifest.sources[0].expected_rows == 22


def test_literal_tokenizer_provenance_must_pin_the_model_revision(tmp_path):
    (tmp_path / "tokenizer_provenance.json").write_text(json.dumps({"served_model": "teacher/tokenizer"}))

    with pytest.raises(ValueError, match="does not pin served_model_revision"):
        resolve_teacher_tokenizer(
            str(tmp_path),
            HarborSftHarness.OPENCODE,
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
    by_name = {source.name: source for source in manifest.sources}
    assert by_name["exp_rpt_curriculum-medium"].revision == "0421a462d1d540ad6a48239cc12c0d1147307b34"
    assert {source.name for source in manifest.sources}.isdisjoint(
        {
            "exp_rpt_methods2test-large-v3",
            "exp_rpt_pymethods2test-large",
            "exp_rpt_pymethods2test-v3",
        }
    )


@pytest.mark.data_integration
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("dataset_id", "revision", "expected_rows", "expected_sha256"),
    [
        (
            "penfever/exp_rpt_curriculum-hard-qwen3.5-122b-131k-opencode-traces",
            "404f3fc1ff0aad3818fcbd07a4d573892e316ffe",
            22,
            "f7edd569a88c8563b15cb14643908c5e9aff37ad40742446f8342d52cd391a29",
        ),
        (
            "penfever/exp_rpt_curriculum-medium-qwen3.5-122b-131k-opencode-traces",
            "0421a462d1d540ad6a48239cc12c0d1147307b34",
            451,
            "320c396af2783f36bdb9d6da67f1b9fa6a14b8744ee084aa21b8cb0e77926b20",
        ),
    ],
)
def test_reproduces_grug_curriculum_training_records(
    dataset_id,
    revision,
    expected_rows,
    expected_sha256,
):
    """Historical source revisions must reproduce the records consumed by the Grug SFT pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-122B-A10B-FP8",
        revision="a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
        trust_remote_code=True,
    )
    rows = load_dataset(
        dataset_id,
        split="train",
        revision=revision,
        streaming=True,
    )
    converted = [
        result.record
        for row in rows
        if (
            result := convert_harbor_row(
                row,
                HarborSftHarness.AUTO,
                tokenizer,
            )
        ).record
        is not None
    ]
    canonical = [json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for record in converted]

    assert len(canonical) == expected_rows
    assert hashlib.sha256("\n".join(canonical).encode()).hexdigest() == expected_sha256
