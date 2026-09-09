# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io

import fsspec
import pytest
import torch

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.calibration_protocol import checkpoint_serving_configuration
from experiments.post_training.math_eval.checkpoint_state import checkpoint_progress
from experiments.post_training.math_eval.export_binding import (
    bind_checkpoint_export,
    hash_file_inventory,
    validate_inventory,
)
from experiments.post_training.math_eval.rate import MODEL_PROFILES


def test_export_inventory_hashes_weight_bytes_and_detects_mutation(tmp_path):
    for name, data in {"config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"weights"}.items():
        (tmp_path / name).write_bytes(data)
    filesystem = fsspec.filesystem("file")
    first = hash_file_inventory(filesystem, str(tmp_path))
    assert first["files"]["model.safetensors"] == {"bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}
    (tmp_path / "model.safetensors").write_bytes(b"WEIGHTS")
    second = hash_file_inventory(filesystem, str(tmp_path))
    assert first["total_bytes"] == second["total_bytes"] == 11
    assert first["files_sha256"] != second["files_sha256"]


def tokenizer_source():
    files = {
        name: {"bytes": 1, "sha256": "c" * 64} for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    }
    files["tokenizer.json"]["sha256"] = MODEL_PROFILES["qwen"]["tokenizer_sha256"]
    return {
        "uri": "s3://marin-us-east-02a/marin/original/hf",
        "files": files,
        "total_bytes": 3,
        "files_sha256": audit.canonical_sha(files),
    }


def evidence():
    prefix = "s3://marin-us-east-02a/marin/users/ahmad/checkpoints/fixture"
    runtime = {"commit": "a" * 40}
    training = {
        "request": {
            "seed": 17,
            "model": {"uri": tokenizer_source()["uri"]},
            "completion_mode": "checkpoint",
            "runtime": runtime,
            "run_id": "fixture-run",
            "attempt_id": "train-attempt",
            "output": {"terminal_manifest_uri": prefix + "/terminal.json", "checkpoint_root": prefix + "/native"},
        },
        "response": {
            "state": "succeeded",
            "runtime": runtime,
            "iris_job_id": "fixture-job",
            "run_id": "fixture-run",
            "attempt_id": "train-attempt",
            "training": {"global_step": 96, "checkpoint": {"global_step": 96}},
        },
    }
    exported = {
        "training_manifest_sha256": audit.canonical_sha(training),
        "request": {
            "training_manifest_uri": prefix + "/terminal.json",
            "attempt_id": "export-attempt",
            "output": {"terminal_manifest_uri": prefix + "/export-terminal.json", "export_root": prefix},
        },
        "response": {
            "state": "succeeded",
            "runtime": runtime,
            "run_id": "fixture-run",
            "attempt_id": "export-attempt",
            "training_iris_job_id": "fixture-job",
            "model": {
                "global_step": 96,
                "checkpoint_root": prefix + "/native",
                "policy_export_uri": prefix + "/hf",
                "terminal_manifest_uri": prefix + "/export-terminal.json",
                "tokenizer_uri": "Qwen/Qwen3-0.6B",
                "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            },
        },
        "export_receipt_uri": prefix + "/receipts/fingerprint.json",
    }
    completion = {
        "global_step": 96,
        "attempt_id": "export-attempt",
        "export_path": prefix + "/hf",
        "request_fingerprint": "fingerprint",
    }
    files = {
        "tokenizer.json": {"sha256": MODEL_PROFILES["qwen"]["tokenizer_sha256"], "bytes": 1},
        "config.json": {"sha256": "c" * 64, "bytes": 2},
        "model.safetensors": {"sha256": "d" * 64, "bytes": 7},
    }
    return (
        training,
        exported,
        completion,
        {"files": files, "files_sha256": audit.canonical_sha(files), "total_bytes": 10},
    )


def bind(rows):
    return bind_checkpoint_export(
        *rows,
        training_sha256=audit.canonical_sha(rows[0]),
        export_sha256=audit.canonical_sha(rows[1]),
        completion_sha256=audit.canonical_sha(rows[2]),
        seed=17,
        runtime_commit="a" * 40,
        tokenizer_source=tokenizer_source(),
    )


def test_checkpoint_binding_rejects_self_consistent_but_wrong_export_links():
    original = bind(evidence())
    assert original["global_step"] == 96 and original["training_seed"] == 17
    for field, value in [("global_step", 95), ("checkpoint_root", "foreign"), ("policy_export_uri", "foreign")]:
        rows = evidence()
        rows[1]["response"]["model"][field] = value
        with pytest.raises(ValueError):
            bind(rows)


@pytest.mark.parametrize("poison", ["seed", "runtime", "attempt", "training_hash", "tokenizer", "inventory"])
def test_checkpoint_binding_rejects_wrong_provenance_after_receipt_rehash(poison):
    rows = evidence()
    if poison == "seed":
        rows[0]["request"]["seed"] = 29
        rows[1]["training_manifest_sha256"] = audit.canonical_sha(rows[0])
    elif poison == "runtime":
        rows[1]["response"]["runtime"] = {"commit": "b" * 40}
    elif poison == "attempt":
        rows[2]["attempt_id"] = "another-export"
    elif poison == "training_hash":
        rows[1]["training_manifest_sha256"] = "b" * 64
    elif poison == "tokenizer":
        rows[1]["response"]["model"]["tokenizer_revision"] = "another-revision"
    else:
        rows[3]["files_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        bind(rows)


@pytest.mark.parametrize("poison", ["missing_weight", "missing_config", "size", "bound", "digest", "path"])
def test_inventory_rejects_rehashed_incomplete_or_invalid_content(poison):
    inventory = evidence()[3]
    files = inventory["files"]
    if poison == "missing_weight":
        del files["model.safetensors"]
    elif poison == "missing_config":
        del files["config.json"]
    elif poison == "size":
        files["model.safetensors"]["bytes"] = -1
    elif poison == "bound":
        files["model.safetensors"]["bytes"] = 3 * 1024**3
    elif poison == "digest":
        files["model.safetensors"]["sha256"] = "not-a-digest"
    else:
        files["../foreign"] = {"sha256": "a" * 64, "bytes": 1}
    inventory["total_bytes"] = sum(row["bytes"] for row in files.values())
    inventory["files_sha256"] = audit.canonical_sha(files)
    with pytest.raises(ValueError):
        validate_inventory(inventory)


@pytest.mark.parametrize("field", ["attempt_id", "run_id"])
def test_training_response_must_match_request_after_rehash(field):
    rows = evidence()
    rows[0]["response"][field] = "foreign"
    rows[1]["training_manifest_sha256"] = audit.canonical_sha(rows[0])
    with pytest.raises(ValueError):
        bind(rows)


def test_binding_preserves_exported_tokenizer_without_using_it_for_calibration():
    rows = evidence()
    rows[3]["files"]["tokenizer.json"]["sha256"] = "b" * 64
    rows[3]["files_sha256"] = audit.canonical_sha(rows[3]["files"])
    result = bind(rows)
    assert result["content"]["files"]["tokenizer.json"]["sha256"] == "b" * 64
    assert result["tokenizer_source"] == tokenizer_source()


def test_binding_rejects_another_original_tokenizer_source():
    rows = evidence()
    rows[0]["request"]["model"]["uri"] = "s3://marin-us-east-02a/marin/another/hf"
    rows[1]["training_manifest_sha256"] = audit.canonical_sha(rows[0])
    with pytest.raises(ValueError, match="original training model"):
        bind(rows)


def ladder_evidence(minibatches, *, saved_update_count=96, saved_batch_size=None):
    rows = evidence()
    native_step = 96 // minibatches
    state = {
        "global_step": native_step,
        "successful_policy_updates": saved_update_count,
        "config": {
            "trainer": {
                "seed": 17,
                "max_steps": native_step,
                "policy_mini_batch_size": 64,
                "train_batch_size": 64 * minibatches if saved_batch_size is None else saved_batch_size,
            }
        },
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    raw = buffer.getvalue()
    training, exported, completion, _ = rows
    training["response"]["training"].update(global_step=native_step)
    training["response"]["training"]["checkpoint"].update(
        global_step=native_step, trainer_state_sha256=hashlib.sha256(raw).hexdigest()
    )
    exported["training_manifest_sha256"] = audit.canonical_sha(training)
    exported["response"]["model"]["global_step"] = native_step
    completion["global_step"] = native_step
    return rows, raw


def ladder_bind(rows, progress, expected_progress_sha256):
    return bind_checkpoint_export(
        *rows,
        training_sha256=audit.canonical_sha(rows[0]),
        export_sha256=audit.canonical_sha(rows[1]),
        completion_sha256=audit.canonical_sha(rows[2]),
        seed=17,
        runtime_commit="a" * 40,
        tokenizer_source=tokenizer_source(),
        progress=progress,
        progress_sha256=expected_progress_sha256,
    )


@pytest.mark.parametrize("minibatches,native_step", [(1, 96), (2, 48)])
def test_ladder_export_proves_saved_updates_separately_from_checkpoint_step(minibatches, native_step):
    rows, raw = ladder_evidence(minibatches)
    progress = checkpoint_progress(
        rows[0], raw, training_sha256=audit.canonical_sha(rows[0]), optimizer_updates=96, minibatches=minibatches
    )
    binding = ladder_bind(rows, progress, progress["progress_sha256"])
    assert binding["global_step"] == native_step and binding["optimizer_updates"] == 96
    assert binding["progress"]["trainer_state_sha256"] == hashlib.sha256(raw).hexdigest()
    model, _ = checkpoint_serving_configuration(binding, expected_binding_sha256=binding["binding_sha256"])
    assert model.weights == rows[1]["response"]["model"]["policy_export_uri"]
    rows[2]["global_step"] = 47
    with pytest.raises(ValueError, match="training attempt"):
        ladder_bind(rows, progress, progress["progress_sha256"])


@pytest.mark.parametrize("saved_updates,saved_batch", [(95, 128), (None, 128), (96, 64)])
def test_ladder_checkpoint_rejects_incomplete_updates_or_wrong_saved_geometry(saved_updates, saved_batch):
    rows, raw = ladder_evidence(2, saved_update_count=saved_updates, saved_batch_size=saved_batch)
    with pytest.raises(ValueError, match="successful updates"):
        checkpoint_progress(
            rows[0], raw, training_sha256=audit.canonical_sha(rows[0]), optimizer_updates=96, minibatches=2
        )


def test_ladder_progress_rejects_mutated_state_bytes_before_deserialization():
    rows, raw = ladder_evidence(2)
    with pytest.raises(ValueError, match="trainer bytes"):
        checkpoint_progress(
            rows[0], raw + b"changed", training_sha256=audit.canonical_sha(rows[0]), optimizer_updates=96, minibatches=2
        )


def test_rehashed_progress_cannot_replace_the_independently_qualified_receipt():
    rows, raw = ladder_evidence(2)
    progress = checkpoint_progress(
        rows[0], raw, training_sha256=audit.canonical_sha(rows[0]), optimizer_updates=96, minibatches=2
    )
    expected = progress["progress_sha256"]
    progress["saved_config_sha256"] = "b" * 64
    progress["progress_sha256"] = audit.canonical_sha({k: v for k, v in progress.items() if k != "progress_sha256"})
    with pytest.raises(ValueError, match="independently qualified digest"):
        ladder_bind(rows, progress, expected)
