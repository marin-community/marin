# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import zipfile

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.post_training.skill_acquisition.data import build_diagnostic, diagnostic_task_split, replay_prefixes


@pytest.fixture
def tokenizer():
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    result = Tokenizer(models.BPE(vocab, []))
    result.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    result.decoder = decoders.ByteLevel()
    result.add_special_tokens(["<tool_call>", "</tool_call>", "<|start_think|>", "<|end_think|>", "<|start_header_id|>"])
    return result


def test_replay_keeps_exact_history_and_excludes_thinking_and_observation_calls(tokenizer):
    prompt = "User: café <tool_call><tool_call><|start_header_id|>assistant<|start_think|>"
    assistant = (
        "<tool_call><tool_call><|end_think|>"
        '<tool_call>{"name":"read","arguments":{"x":"<tool_call><tool_call>"}}</tool_call>'
        "<tool_call><tool_call>"
    )
    prompt_ids = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(assistant).ids
    results = replay_prefixes(prompt_ids + response_ids, [0] * len(prompt_ids) + [1] * len(response_ids), tokenizer)
    assert len(results) == 1
    prefix, status = results[0]
    assert list(prefix) == prompt_ids + response_ids[:-1]
    assert tokenizer.decode(list(prefix), skip_special_tokens=False) == prompt + assistant[: -len("<tool_call>")]
    assert status == "repeated_opener"


def test_build_replay_audits_partition_and_preserves_immutable_output(tmp_path, tokenizer):
    approved = ["a", "b", "c"]
    train, reserved = diagnostic_task_split(approved, 1, "test-v1")
    assert set(train).isdisjoint(reserved)
    assert set(train) | set(reserved) == set(approved)
    assert diagnostic_task_split(list(reversed(approved)), 1, "test-v1") == (train, reserved)
    partition = {
        "dataset_commit": "test-source",
        "complement": [{"name": task, "source_id": task, "digest": task} for task in approved],
        "parity": [{"name": "heldout", "source_id": "heldout", "digest": "heldout"}],
    }
    partition_file = tmp_path / "partition.json"
    partition_file.write_text(json.dumps(partition))
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    prompt = tokenizer.encode("<|start_header_id|>assistant<|end_think|>").ids
    response = tokenizer.encode("<tool_call><tool_call>").ids
    row = {
        "record_id": "record-a",
        "phase": "eval",
        "trajectory": {"instance_id": reserved[0]},
        "prompt": {"token_ids": prompt},
        "response": {"token_ids": response, "loss_mask": [1] * len(response)},
        "provenance": {"model_source_identity": "historical-model"},
    }
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("records/a.json.gz", gzip.compress(json.dumps(row).encode()))
    output = tmp_path / "version-1"
    arguments = dict(version="test-v1", validation_tasks=1, max_prefix_tokens=100)
    manifest = build_diagnostic([archive], partition_file, tokenizer_file, output, **arguments)
    case_bytes = (output / "cases.jsonl").read_bytes()
    case = json.loads(case_bytes)
    assert case["prompt_token_ids"] == prompt + response[:1]
    assert case["source_phase"] == "eval"
    assert case["task_source_id"] not in manifest["train_task_ids"]
    assert case["task_source_id"] not in manifest["parity_task_ids"]
    with pytest.raises(FileExistsError):
        build_diagnostic([archive], partition_file, tokenizer_file, output, **arguments)
    assert (output / "cases.jsonl").read_bytes() == case_bytes

    forbidden = tmp_path / "forbidden.zip"
    with zipfile.ZipFile(forbidden, "w") as stream:
        for task in ("heldout", "unknown"):
            bad_row = {**row, "record_id": task, "trajectory": {"instance_id": task}}
            stream.writestr(f"records/{task}.json.gz", gzip.compress(json.dumps(bad_row).encode()))
    with pytest.raises(ValueError) as failure:
        build_diagnostic([archive, forbidden], partition_file, tokenizer_file, tmp_path / "invalid", **arguments)
    errors = json.loads(str(failure.value))["conversion_failures"]
    assert {error["record"] for error in errors} == {"heldout", "unknown"}
    assert not (tmp_path / "invalid").exists()
