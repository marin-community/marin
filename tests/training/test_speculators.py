# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import marin.training.speculators as speculators
import pytest
from finestore.eval import ParticipantType
from marin.training.speculators import (
    HiddenStateCaptureConfig,
    _best_checkpoint,
    _capture_vllm_args,
    _make_checkpoint_portable,
    _publish_directory,
    _restore_directory,
    capture_hidden_states,
    rollout_conversations,
    verifier_config_for_transformers,
)


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
        return False


class _CaptureEnvironment:
    server_url = "http://127.0.0.1:8000/v1"

    def wait_until_ready(self):
        return None


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


def test_capture_args_request_auxiliary_and_final_hidden_states(tmp_path: Path):
    config = HiddenStateCaptureConfig(
        dataset_path="dataset.jsonl",
        target_model="target",
        processor_model="tokenizer",
        output_path="output",
        target_layer_ids=(2, 13, 23),
        verifier_num_hidden_layers=26,
        sequence_length=16384,
        data_parallel_size=8,
        concurrency=64,
        max_samples=None,
        gpu_memory_utilization=0.9,
    )

    args = _capture_vllm_args(config, tmp_path)
    speculative_config = json.loads(args[args.index("--speculative-config") + 1])
    transfer_config = json.loads(args[args.index("--kv-transfer-config") + 1])

    assert speculative_config["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [
        2,
        13,
        23,
        26,
    ]
    assert transfer_config["kv_connector_extra_config"]["shared_storage_path"] == str(tmp_path)
    assert args[:5] == [
        "--enforce-eager",
        "--no-enable-flashinfer-autotune",
        "--data-parallel-size",
        "8",
        "--enable-expert-parallel",
    ]
    assert args[args.index("--data-parallel-size") + 1] == "8"


def test_checkpoint_selection_and_portable_verifier_reference(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"speculators_config": {"verifier": {"name_or_path": "/tmp/verifier"}}})
    )

    assert _best_checkpoint(checkpoint.parent) == checkpoint
    _make_checkpoint_portable(checkpoint)

    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["speculators_config"]["verifier"]["name_or_path"] is None


def test_publish_directory_resumes_complete_files(tmp_path: Path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    destination.mkdir()
    (source / "complete.txt").write_text("complete")
    (destination / "complete.txt").write_text("complete")
    (source / "nested" / "new.txt").write_text("new")

    _publish_directory(source, str(destination))

    assert (destination / "complete.txt").read_text() == "complete"
    assert (destination / "nested" / "new.txt").read_text() == "new"


def test_restore_directory_recovers_published_capture_progress(tmp_path: Path):
    source = tmp_path / "published"
    destination = tmp_path / "restored"
    (source / "hidden_states").mkdir(parents=True)
    (source / "hidden_states" / "hs_7.safetensors").write_text("captured")

    _restore_directory(str(source), destination)

    assert (destination / "hidden_states" / "hs_7.safetensors").read_text() == "captured"


def test_capture_publishes_after_vllm_teardown_failure(monkeypatch):
    published = []

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
            Path(command[command.index("--output") + 1]).mkdir()

    monkeypatch.setattr(speculators, "StoragePath", _CaptureStoragePath)
    monkeypatch.setattr(speculators, "VllmBackend", FakeBackend)
    monkeypatch.setattr(speculators, "_run_command", fake_command)
    monkeypatch.setattr(speculators, "_publish_directory", lambda source, destination: published.append(destination))

    capture_hidden_states(
        HiddenStateCaptureConfig(
            dataset_path="dataset.jsonl",
            target_model="target",
            processor_model="tokenizer",
            output_path="published",
            target_layer_ids=(2, 13, 23),
            verifier_num_hidden_layers=26,
            sequence_length=32768,
            data_parallel_size=8,
            concurrency=64,
            max_samples=None,
            gpu_memory_utilization=0.9,
        )
    )

    assert published == ["published"]


def test_capture_publishes_progress_before_propagating_generation_failure(monkeypatch):
    published = []

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
    monkeypatch.setattr(
        speculators,
        "_publish_directory",
        lambda source, destination: published.append(
            (destination, (source / "hidden_states" / "hs_0.safetensors").exists())
        ),
    )

    with pytest.raises(subprocess.CalledProcessError):
        capture_hidden_states(
            HiddenStateCaptureConfig(
                dataset_path="dataset.jsonl",
                target_model="target",
                processor_model="tokenizer",
                output_path="published",
                target_layer_ids=(2, 13, 23),
                verifier_num_hidden_layers=26,
                sequence_length=32768,
                data_parallel_size=8,
                concurrency=64,
                max_samples=None,
                gpu_memory_utilization=0.9,
            )
        )

    assert published == [("published", True)]
