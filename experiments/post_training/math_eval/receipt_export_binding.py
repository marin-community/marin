# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind converted checkpoint bytes without claiming a successful training lifecycle."""

import posixpath

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.checkpoint_tokenizer import validate_tokenizer_source
from experiments.post_training.math_eval.export_binding import EAST_PREFIX, validate_inventory

SCHEMA = "math_eval_receipt_checkpoint_content_v1"


def bind_receipt_checkpoint_export(
    request, mechanical, scientific, snapshot, exported, completion, inventory, *, expected_hashes, tokenizer_source
):
    """Join independently audited receipts to a conversion of the exact snapshot.

    Expected hashes must come from independent receipt qualification. Controller
    history and allocation auditing remain separate; no training terminal is made.
    """
    receipts = dict(
        request=request,
        mechanical=mechanical,
        scientific=scientific,
        snapshot=snapshot,
        exported=exported,
        completion=completion,
    )
    if set(expected_hashes) != set(receipts) or any(
        audit.canonical_sha(value) != expected_hashes[name] for name, value in receipts.items()
    ):
        raise ValueError("Receipt-only export differs from independently audited inputs")
    seed = request["seed"]
    checkpoint = mechanical["training_result"]["checkpoint"]
    source = exported["source_binding"]
    proof = scientific["receipt_proof"]
    original_files = {entry["path"]: entry["size"] for entry in checkpoint["files"]}
    snapshot_files = {entry["path"]: entry["size"] for entry in snapshot["files"]}
    snapshot_unsigned = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if (
        seed not in {17, 29}
        or request["completion_mode"] != "checkpoint"
        or mechanical["schema"] != "math_eval_checkpoint_receipt_audit_v1"
        or scientific["schema"] != "math_eval_type_c_scientific_receipt_only_v1"
        or not scientific["scientific_measurement_qualified"]
        or scientific["lifecycle_clean"]
        or scientific["clean_end_to_end"]
        or proof["terminal_sha256"] is not None
        or any(value["seed"] != seed for value in (mechanical, scientific, snapshot))
        or any(value["optimizer_updates"] != 96 for value in (mechanical, scientific, snapshot))
        or mechanical["minibatches"] != 4
        or scientific["minibatches"] != 4
        or any(value["native_global_step"] != 24 for value in (mechanical, scientific, snapshot))
        or checkpoint["global_step"] != 24
        or proof["saved_successful_updates"] != 96
        or proof["trainer_state_sha256"] != checkpoint["trainer_state_sha256"]
        or proof["independent_checkpoint_audit_sha256"] != expected_hashes["mechanical"]
        or proof["receipt_sha256"] != mechanical["training_receipt_sha256"]
        or mechanical["original_request_sha256"] != expected_hashes["request"]
        or snapshot["original_checkpoint"] != checkpoint
        or snapshot["original_audit_bytes_sha256"] != expected_hashes["mechanical"]
        or snapshot["snapshot_sha256"] != audit.canonical_sha(snapshot_unsigned)
        or not snapshot["source_and_readback_sha256_equal"]
        or snapshot["source_metadata_before"] != snapshot["source_metadata_after"]
        or snapshot["native_before"] != snapshot["native_after"]
        or len(snapshot_files) != len(snapshot["files"])
        or original_files != snapshot_files
    ):
        raise ValueError("Snapshot does not bind the audited receipt-only optimizer checkpoint")
    if any(len(entry["sha256"]) != 64 for entry in snapshot["files"]):
        raise ValueError("Snapshot requires every file's byte digest")
    if (
        exported["schema"] != "math_eval_receipt_only_policy_export_v1"
        or exported["state"] != "succeeded"
        or exported["training_lifecycle_clean"]
        or exported["training_manifest_sha256"] is not None
        or exported["source_binding_sha256"] != audit.canonical_sha(source)
        or source["schema"] != "math_eval_receipt_only_export_source_v1"
        or source["original_request_sha256"] != expected_hashes["request"]
        or source["original_training_envelope_sha256"] != mechanical["original_envelope_bytes_sha256"]
        or source["effective_yaml_sha256"] != mechanical["effective_export_yaml_sha256"]
        or source["mechanical_sha256"] != expected_hashes["mechanical"]
        or source["scientific_bytes_sha256"] != expected_hashes["scientific"]
        or source["snapshot_bytes_sha256"] != expected_hashes["snapshot"]
        or source["snapshot_sha256"] != snapshot["snapshot_sha256"]
        or source["snapshot_root"] != snapshot["snapshot_root"]
        or source["snapshot_checkpoint_path"] != snapshot["snapshot_checkpoint_path"]
        or source["training_result"] != mechanical["training_result"]
        or source["training_runtime"] != request["runtime"]
        or source["training_runtime"] != proof["runtime"]
        or source["exporter_runtime"] != exported["exporter_runtime"]
        or exported["global_step"] != completion["global_step"]
        or completion["global_step"] != 24
        or completion["request_fingerprint"] != exported["source_binding_sha256"]
        or completion["attempt_id"] != exported["export_attempt_id"]
        or completion["export_path"] != exported["model_uri"]
        or posixpath.basename(exported["completion_receipt_uri"]) != completion["request_fingerprint"] + ".json"
        or not exported["model_uri"].startswith(source["output"]["exports"].rstrip("/") + "/")
        or not exported["model_uri"].startswith(EAST_PREFIX)
    ):
        raise ValueError("Conversion does not bind the exact snapshot and native completion receipt")
    validate_inventory(inventory)
    validate_tokenizer_source(tokenizer_source)
    if tokenizer_source["uri"] != request["model"]["uri"]:
        raise ValueError("Receipt-only calibration requires the original tokenizer source")
    binding = {
        "schema": SCHEMA,
        "training_seed": seed,
        "global_step": 24,
        "optimizer_updates": 96,
        "minibatches": 4,
        "training_manifest_sha256": None,
        "training_lifecycle_clean": False,
        "scientific_measurement_qualified": True,
        "receipt_hashes": expected_hashes,
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "trainer_state_sha256": checkpoint["trainer_state_sha256"],
        "training_receipt_sha256": mechanical["training_receipt_sha256"],
        "training_attempt_id": request["attempt_id"],
        "export_attempt_id": exported["export_attempt_id"],
        "runtime_commit": request["runtime"]["commit"],
        "exporter_runtime": exported["exporter_runtime"],
        "model_uri": exported["model_uri"],
        "content": inventory,
        "tokenizer_source": tokenizer_source,
    }
    return binding | {"binding_sha256": audit.canonical_sha(binding)}


def validate_receipt_content_binding(binding):
    """Validate receipt-only progress without reclassifying controller failure."""
    if (
        binding["schema"] != SCHEMA
        or binding["training_manifest_sha256"] is not None
        or binding["training_lifecycle_clean"]
        or not binding["scientific_measurement_qualified"]
        or binding["optimizer_updates"] != 96
        or binding["minibatches"] != 4
        or binding["global_step"] != 24
        or set(binding["receipt_hashes"])
        != {"request", "mechanical", "scientific", "snapshot", "exported", "completion"}
        or any(len(value) != 64 for value in binding["receipt_hashes"].values())
    ):
        raise ValueError("Receipt-only calibration checkpoint or lifecycle scope changed")
