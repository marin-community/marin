# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
from pathlib import Path

import marin.training.speculators as speculators
import pytest
import torch
from finestore.eval import ParticipantType
from marin.training.speculators import (
    HiddenStateCaptureConfig,
    _make_checkpoint_portable,
    _preferred_checkpoint,
    _publish_directory,
    _restore_directory,
    _validate_draft_checkpoint,
    capture_hidden_states,
    interleave_conversation_streams,
    rollout_conversations,
    verifier_config_for_transformers,
)
from safetensors.torch import save_file


def _row(doc_id: str, participant: str, content: str) -> dict:
    return {
        "task": "aime24",
        "doc_id": doc_id,
        "trial_id": "",
        "participant_type": participant,
        "content": content,
    }


class _CaptureStoragePath:
    def __init__(self, path: str):
        self.path = path

    def download_to(self, local_path: str, *, recursive: bool = False, batch_size: int | None = None):
        del batch_size
        destination = Path(local_path)
        if recursive:
            destination.mkdir()
        else:
            destination.write_text(self.path)

    def exists(self):
        return Path(self.path).exists()

    def size(self):
        return Path(self.path).stat().st_size

    @property
    def is_local(self):
        return True

    @property
    def parent(self):
        return _CaptureStoragePath(str(Path(self.path).parent))

    def mkdirs(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def upload_from(self, local_path: str):
        shutil.copy2(local_path, self.path)

    def __truediv__(self, relative_path: str):
        return _CaptureStoragePath(str(Path(self.path) / relative_path))


class _CaptureEnvironment:
    server_url = "http://127.0.0.1:8000/v1"

    def wait_until_ready(self):
        return None


def _capture_config(*, output_path: str = "published") -> HiddenStateCaptureConfig:
    return HiddenStateCaptureConfig(
        dataset_path="dataset.jsonl",
        target_model="target",
        processor_model="tokenizer",
        output_path=output_path,
        target_layer_ids=(2, 13, 23),
        verifier_num_hidden_layers=26,
        sequence_length=32768,
        data_parallel_size=8,
        concurrency=64,
        max_samples=None,
        minimum_valid_tokens=None,
        gpu_memory_utilization=0.9,
    )


def test_rollout_conversations_groups_complete_rollouts():
    rows = [
        _row("1", ParticipantType.USER, "first prompt"),
        _row("1", ParticipantType.ASSISTANT, "first response"),
        _row("2", ParticipantType.USER, "unfinished prompt"),
        _row("3", ParticipantType.SYSTEM, "system"),
        _row("3", ParticipantType.USER, "third prompt"),
        _row("3", ParticipantType.ASSISTANT, "third response"),
    ]

    assert list(rollout_conversations(rows)) == [
        {
            "conversations": [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first response"},
            ]
        },
        {
            "conversations": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "third prompt"},
                {"role": "assistant", "content": "third response"},
            ]
        },
    ]


def test_interleave_conversation_streams_preserves_short_sources() -> None:
    long_source = iter([{"conversations": [str(index)]} for index in range(4)])
    short_source = iter([{"conversations": ["agent-0"]}, {"conversations": ["agent-1"]}])

    assert list(interleave_conversation_streams((long_source, short_source))) == [
        {"conversations": ["0"]},
        {"conversations": ["agent-0"]},
        {"conversations": ["1"]},
        {"conversations": ["agent-1"]},
        {"conversations": ["2"]},
        {"conversations": ["3"]},
    ]


def test_verifier_config_keeps_architecture_when_rewriting_model_type():
    source = {
        "architectures": ["GrugMoeForCausalLM"],
        "model_type": "grug_moe",
        "hidden_size": 2560,
    }

    rewritten = verifier_config_for_transformers(source, "llama")

    assert rewritten == {
        "architectures": ["GrugMoeForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2560,
    }
    assert source["model_type"] == "grug_moe"


def test_checkpoint_selection_and_portable_verifier_reference(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"speculators_config": {"verifier": {"name_or_path": "/tmp/verifier"}}})
    )

    assert _preferred_checkpoint(checkpoint.parent) == checkpoint
    _make_checkpoint_portable(checkpoint)

    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["speculators_config"]["verifier"]["name_or_path"] is None


def test_validate_draft_checkpoint_rejects_serialized_target_owned_embedding(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"embed_requires_grad": False}))
    save_file({"embed_tokens.weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="Frozen EAGLE embeddings must be omitted"):
        _validate_draft_checkpoint(tmp_path)


def test_publish_directory_skips_same_size_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    destination.mkdir()
    (source / "complete.txt").write_text("complete")
    (destination / "complete.txt").write_text("existing")
    (source / "nested" / "new.txt").write_text("new")

    _publish_directory(source, str(destination))

    assert (destination / "complete.txt").read_text() == "existing"
    assert (destination / "nested" / "new.txt").read_text() == "new"


def test_restore_directory_recovers_published_capture_progress(tmp_path: Path):
    source = tmp_path / "published"
    destination = tmp_path / "restored"
    (source / "hidden_states").mkdir(parents=True)
    (source / "hidden_states" / "hs_7.safetensors").write_text("captured")

    _restore_directory(str(source), destination)

    assert (destination / "hidden_states" / "hs_7.safetensors").read_text() == "captured"


def test_capture_publishes_after_vllm_teardown_failure(monkeypatch, tmp_path: Path):
    class FailingExit:
        def __enter__(self):
            return _CaptureEnvironment()

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            raise RuntimeError("expected shutdown failure")

    class FakeBackend:
        def __init__(self, config):
            del config

        def start(self, model, *, extra_args, subprocess_env):
            del model, extra_args, subprocess_env
            return FailingExit()

    def fake_command(command, *, environment=None):
        del environment
        if "prepare-data" in command:
            prepared_data = Path(command[command.index("--output") + 1])
            prepared_data.mkdir()
            (prepared_data / "dataset_info.json").write_text("{}")

    monkeypatch.setattr(speculators, "StoragePath", _CaptureStoragePath)
    monkeypatch.setattr(speculators, "VllmBackend", FakeBackend)
    monkeypatch.setattr(speculators, "_run_command", fake_command)
    output_path = tmp_path / "published"

    capture_hidden_states(_capture_config(output_path=str(output_path)))

    assert (output_path / "dataset_info.json").read_text() == "{}"


def test_capture_publishes_progress_before_propagating_generation_failure(monkeypatch, tmp_path: Path):
    class FakeBackend:
        def __init__(self, config):
            del config

        def start(self, model, *, extra_args, subprocess_env):
            del model, extra_args, subprocess_env
            return self

        def __enter__(self):
            return _CaptureEnvironment()

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

    def fake_command(command, *, environment=None):
        del environment
        if "prepare-data" in command:
            prepared_data = Path(command[command.index("--output") + 1])
            prepared_data.mkdir()
            (prepared_data / "dataset_info.json").write_text("{}")
            return
        hidden_states = Path(command[command.index("--output") + 1])
        (hidden_states / "hs_0.safetensors").write_text("captured")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(speculators, "StoragePath", _CaptureStoragePath)
    monkeypatch.setattr(speculators, "VllmBackend", FakeBackend)
    monkeypatch.setattr(speculators, "_run_command", fake_command)
    output_path = tmp_path / "published"

    with pytest.raises(subprocess.CalledProcessError):
        capture_hidden_states(_capture_config(output_path=str(output_path)))

    assert (output_path / "dataset_info.json").read_text() == "{}"
    assert (output_path / "hidden_states" / "hs_0.safetensors").read_text() == "captured"
