# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json
import tomllib
import zipfile
from dataclasses import asdict

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.post_training.bfcl_rl.data import task_identity
from experiments.post_training.skill_acquisition.data import build_diagnostic, diagnostic_task_split, replay_prefixes
from experiments.post_training.skill_acquisition.native_tasks import materialize_native_tasks
from experiments.post_training.skill_acquisition.training_data import materialize_training_tasks


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


def test_native_conversion_preserves_task_content_and_reports_all_unknown_images(tmp_path):
    source = tmp_path / "source"
    identities = []
    dockerfile = b"FROM python:3.10-slim\nWORKDIR /app\n"
    for name in ("a", "b"):
        task = source / name
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "environment/Dockerfile").write_bytes(dockerfile)
        (task / "instruction.md").write_text("Write the answer to /app/result.json.\n")
        (task / "tests/test.sh").write_text("python3 /tests/evaluate.py\n")
        (task / "task.toml").write_text(
            f'version = "1.0"\n[metadata]\nsource_id = "{name}"\n'
            '[verifier]\ntimeout_sec = 300\n[environment]\ntype = "docker"\n'
        )
        identities.append(task_identity(task))
    output = tmp_path / "native-v1"
    image = "example.test/bfcl@sha256:" + "a" * 64
    manifest = materialize_native_tasks(
        source, tuple(identities), {hashlib.sha256(dockerfile).hexdigest(): image}, output
    )
    assert manifest["source_tasks"] == [asdict(task) for task in identities]
    for identity in identities:
        original = source / identity.name
        converted = output / identity.name
        assert task_identity(original) == identity
        assert task_identity(converted).digest != identity.digest
        for file in original.rglob("*"):
            if file.is_file() and file.name != "task.toml":
                assert (converted / file.relative_to(original)).read_bytes() == file.read_bytes()
        config = tomllib.loads((converted / "task.toml").read_text())
        assert config["artifacts"] == ["/app/result.json"]
        assert config["verifier"] == {"timeout_sec": 300, "environment_mode": "separate"}
        assert config["environment"] == {"type": "docker", "docker_image": image, "workdir": "/app"}
    with pytest.raises(ValueError) as failure:
        materialize_native_tasks(source, tuple(identities), {}, tmp_path / "unknown")
    assert {item["task"] for item in json.loads(str(failure.value))["conversion_failures"]} == {"a", "b"}
    assert not (tmp_path / "unknown").exists()


def test_training_subset_preserves_files_and_audits_all_changed_tasks(tmp_path):
    source = tmp_path / "source"
    identities = []
    for name in ("train-a", "train-b", "diagnostic"):
        task = source / name
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'[metadata]\nsource_id = "{name}"\n')
        (task / "instruction.md").write_text(f"Original task {name}\n")
        identities.append(asdict(task_identity(task)))
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"dataset_commit": "pinned", "complement": identities, "parity": []}))
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "version": "split-v1",
                "source_partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
                "train_task_ids": ["train-a", "train-b"],
                "diagnostic_reserved_task_ids": ["diagnostic"],
                "parity_task_ids": [],
            }
        )
    )
    output = tmp_path / "train-v1"
    manifest = materialize_training_tasks(source, partition, diagnostic, output)
    assert manifest["training_tasks"] == 2
    assert {path.name for path in (output / "tasks").iterdir()} == {"train-a", "train-b"}
    for name in ("train-a", "train-b"):
        assert task_identity(output / "tasks" / name) == task_identity(source / name)
        (source / name / "instruction.md").write_text("Changed task\n")
    with pytest.raises(FileExistsError):
        materialize_training_tasks(source, partition, diagnostic, output)
    with pytest.raises(ValueError) as failure:
        materialize_training_tasks(source, partition, diagnostic, tmp_path / "invalid")
    assert {item["task"] for item in json.loads(str(failure.value))["conversion_failures"]} == {"train-a", "train-b"}
    assert not (tmp_path / "invalid").exists()
