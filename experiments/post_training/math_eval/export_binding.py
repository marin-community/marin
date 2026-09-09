# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind calibration weights to a successful export and exact file contents."""

import hashlib
import os
import posixpath
import re

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.checkpoint_progress import validate_progress
from experiments.post_training.math_eval.checkpoint_tokenizer import validate_tokenizer_source

EAST_PREFIX = "s3://marin-us-east-02a/marin/"
MAX_EXPORT_BYTES = 2 * 1024**3
MAX_EXPORT_FILES = 128
CHUNK_BYTES = 16 * 1024**2


def hash_export_files(export_uri):
    """Stream east HF bytes inside Iris; caller's audited envelope binds region."""
    if not export_uri.startswith(EAST_PREFIX) or not os.environ.get("IRIS_TASK_ID"):
        raise ValueError("Hash calibration weights only inside an east Iris task")
    filesystem, root = audit.fs_path(export_uri)
    return hash_file_inventory(filesystem, root)


def hash_file_inventory(filesystem, root):
    """Hash actual returned bytes, refusing changed sizes or oversized exports."""
    inventory = filesystem.find(root, detail=True, withdirs=False)
    if not 1 <= len(inventory) <= MAX_EXPORT_FILES:
        raise ValueError("Export file count is outside its bound")
    expected_total = sum(item["size"] for item in inventory.values())
    if not 0 < expected_total <= MAX_EXPORT_BYTES:
        raise ValueError("Export bytes are outside the Qwen bound")
    files = {}
    for path, item in sorted(inventory.items()):
        name = posixpath.relpath(path, root)
        if name.startswith("../") or name in {".", ".."} or item["size"] <= 0:
            raise ValueError("Invalid export file path or size")
        digest, size = hashlib.sha256(), 0
        with filesystem.open(path, "rb") as stream:
            while chunk := stream.read(CHUNK_BYTES):
                size += len(chunk)
                if size > item["size"]:
                    raise ValueError("Export file grew after its inventory")
                digest.update(chunk)
        if size != item["size"]:
            raise ValueError("Export file size changed after its inventory")
        files[name] = {"bytes": size, "sha256": digest.hexdigest()}
    result = {"files": files, "total_bytes": expected_total, "files_sha256": audit.canonical_sha(files)}
    validate_inventory(result)
    return result


def validate_inventory(inventory):
    """Require a complete bounded hash inventory even when supplied by a caller."""
    files = inventory["files"]
    if not 1 <= len(files) <= MAX_EXPORT_FILES:
        raise ValueError("Export file count is outside its bound")
    if not {"config.json", "tokenizer.json"} <= files.keys() or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Export lacks required HF model content")
    for name, item in files.items():
        if (
            not name
            or name.startswith("/")
            or posixpath.normpath(name) != name
            or name.startswith("../")
            or name in {".", ".."}
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise ValueError("Invalid export file path, size or digest")
    total = sum(item["bytes"] for item in files.values())
    if not 0 < total <= MAX_EXPORT_BYTES or inventory["total_bytes"] != total:
        raise ValueError("Export content size changed or exceeded its bound")
    if inventory["files_sha256"] != audit.canonical_sha(files):
        raise ValueError("Export content inventory changed")


def bind_checkpoint_export(
    training,
    exported,
    completion,
    inventory,
    *,
    training_sha256,
    export_sha256,
    completion_sha256,
    seed,
    runtime_commit,
    tokenizer_source,
    progress=None,
    progress_sha256=None,
):
    """Bind content to preaudited terminal receipts, update96 and its training seed.

    The supplied expected receipt hashes come from independent native terminal
    audits. This function proves the content/export chain; it does not replace
    controller task-state, cost, or training-telemetry qualification.
    """
    for payload, expected in ((training, training_sha256), (exported, export_sha256), (completion, completion_sha256)):
        if audit.canonical_sha(payload) != expected:
            raise ValueError("Checkpoint receipt differs from its audited identity")
    train_request, train_response = training["request"], training["response"]
    export_request, export_response = exported["request"], exported["response"]
    model = export_response["model"]
    checkpoint = train_response["training"]["checkpoint"]
    native_step = 96
    if (progress is None) != (progress_sha256 is None):
        raise ValueError("Checkpoint progress requires its independently qualified digest")
    if progress is not None:
        if progress["progress_sha256"] != progress_sha256:
            raise ValueError("Checkpoint progress differs from its independently qualified digest")
        validate_progress(
            progress, training_sha256=training_sha256, trainer_state_sha256=checkpoint["trainer_state_sha256"]
        )
        if progress["optimizer_updates"] != 96 or progress["training_seed"] != seed:
            raise ValueError("Calibration requires the intended seed at 96 successful optimizer updates")
        native_step = progress["global_step"]
    if (
        seed not in {17, 29}
        or train_request["seed"] != seed
        or train_request["completion_mode"] != "checkpoint"
        or train_response["state"] != "succeeded"
        or train_response["run_id"] != train_request["run_id"]
        or train_response["attempt_id"] != train_request["attempt_id"]
        or export_response["state"] != "succeeded"
        or train_response["training"]["global_step"] != native_step
        or checkpoint["global_step"] != native_step
        or model["global_step"] != native_step
        or completion["global_step"] != native_step
        or exported["training_manifest_sha256"] != training_sha256
        or export_request["training_manifest_uri"] != train_request["output"]["terminal_manifest_uri"]
        or model["checkpoint_root"] != train_request["output"]["checkpoint_root"]
        or not model["policy_export_uri"].startswith(export_request["output"]["export_root"].rstrip("/") + "/")
        or export_response["training_iris_job_id"] != train_response["iris_job_id"]
        or export_response["run_id"] != train_request["run_id"]
        or export_response["attempt_id"] != export_request["attempt_id"]
        or completion["attempt_id"] != export_request["attempt_id"]
        or completion["export_path"] != model["policy_export_uri"]
        or model["terminal_manifest_uri"] != export_request["output"]["terminal_manifest_uri"]
        or any(
            value["commit"] != runtime_commit
            for value in (train_request["runtime"], train_response["runtime"], export_response["runtime"])
        )
    ):
        raise ValueError("Export does not bind the intended training attempt, seed, runtime and update")
    receipt_name = posixpath.basename(exported["export_receipt_uri"])
    if receipt_name != completion["request_fingerprint"] + ".json":
        raise ValueError("Export completion receipt fingerprint differs")
    if not model["policy_export_uri"].startswith(EAST_PREFIX):
        raise ValueError("Calibration checkpoint must remain east")
    if (
        model["tokenizer_uri"] != "Qwen/Qwen3-0.6B"
        or model["tokenizer_revision"] != "c1899de289a04d12100db370d81485cdf75e47ca"
    ):
        raise ValueError("Calibration tokenizer provenance changed")
    validate_inventory(inventory)
    validate_tokenizer_source(tokenizer_source)
    if tokenizer_source["uri"] != train_request["model"]["uri"]:
        raise ValueError("Calibration tokenizer source differs from the original training model")
    binding = {
        "schema": "math_eval_checkpoint_content_v1",
        "training_seed": seed,
        "global_step": native_step,
        "training_manifest_sha256": training_sha256,
        "export_manifest_sha256": export_sha256,
        "completion_receipt_sha256": completion_sha256,
        "export_attempt_id": export_request["attempt_id"],
        "training_attempt_id": train_request["attempt_id"],
        "runtime_commit": runtime_commit,
        "model_uri": model["policy_export_uri"],
        "content": inventory,
        "tokenizer_source": tokenizer_source,
    }
    if progress is not None:
        binding.update(
            schema="math_eval_checkpoint_content_v2",
            optimizer_updates=96,
            trainer_state_sha256=checkpoint["trainer_state_sha256"],
            progress=progress,
        )
    return binding | {"binding_sha256": audit.canonical_sha(binding)}
