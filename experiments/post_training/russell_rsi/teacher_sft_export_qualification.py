# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check saved counters and finite weights when early optimizer norm metrics are absent."""

import math

import numpy as np
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi.launch_teacher_sft import SFT_LEARNING_RATE

PROTOCOL = "teacher-sft-four-update-export-recovery-v1"
FLOAT32_LEARNING_RATE = float(np.float32(SFT_LEARNING_RATE))
RECOVERY_SHARDS = 39
RECOVERY_TENSORS = 502


def validate_numeric_proof(proof: dict, amendment: dict) -> None:
    requirements = amendment["required_numeric_validation"]
    comparisons = proof["fixed_tensor_comparisons"]
    shards = proof["shards"]
    if (
        proof["status"] != "passed"
        or proof["all_saved_bf16_values_finite"] is not True
        or proof["fixed_tensor_net_change"] is not True
        or proof["four_positive_updates_proved"] is not False
        or proof["missing_metric_steps"] != [0, 1, 2]
        or proof["generation_requests"] != 0
        or proof["source"] != prefix_join(amendment["producer"]["root"], "hf/step-3")
        or any(proof[key] != requirements[key] for key in ("shard_count", "tensor_count", "parameter_count"))
        or len(shards) != RECOVERY_SHARDS
        or len({item["name"] for item in shards}) != RECOVERY_SHARDS
        or any(
            item["bytes"] <= 0
            or item["nonfinite_count"] != 0
            or item["before_after_identity_equal"] is not True
            or item["conditional_reads"] is not True
            or item["bf16_values_checked"] <= 0
            or len(item["sha256"]) != 64
            for item in shards
        )
        or sum(item["bf16_values_checked"] for item in shards) != requirements["parameter_count"]
        or [item["key"] for item in comparisons] != requirements["predetermined_comparison_tensors"]
        or any(
            item["both_finite"] is not True or item["before_after_identity_equal"] is not True for item in comparisons
        )
        or not any(item["changed_bf16_word_count"] > 0 for item in comparisons)
    ):
        raise ValueError("Recovery requires exact finite saved weights and the declared parent net change")


def qualified_recovered_four_update_sft(
    record: dict,
    *,
    producer_identity: str,
    producer_root: str,
    recovery_identity: str,
    recovery_root: str,
    source_config_sha256: str,
) -> str:
    """Require the explicit amended export, without claiming missing per-step norms."""
    recovered = record["recovery"]
    amendment = recovered["amendment"]
    export = prefix_join(recovery_root, "hf/step-3")
    producer = recovered["producer"]
    training = recovered["training_evidence"]
    reload = record["serving_reload"]
    shards = recovered["hf_shards"]
    if (
        record["protocol"] != PROTOCOL
        or recovered["protocol"] != PROTOCOL
        or producer != amendment["producer"]
        or producer["identity"] != producer_identity
        or producer["root"] != producer_root
        or producer["status"] != "FAILED"
        or producer["config_sha256"] != source_config_sha256
        or record["source_config_sha256"] != source_config_sha256
        or record["recovery_identity"] != recovery_identity
        or record["recovery_root"] != recovery_root
        or record["recovery_status"] != "SUCCESS"
        or recovered["hf_export_uri"] != export
        or recovered["original_telemetry_gate"] != "UNMET"
        or recovered["missing_metric_steps"] != [0, 1, 2]
        or amendment["protocol"] != PROTOCOL
        or amendment["original_telemetry_gate"] != "UNMET"
        or amendment["missing_metric_steps"] != [0, 1, 2]
        or recovered["amendment_sha256"] != record["amendment_sha256"]
        or recovered["numeric_proof_sha256"] != record["numeric_proof_sha256"]
        or recovered["numeric_manifest_sha256"] != recovered["numeric_proof"]["input_manifest_sha256"]
        or training != amendment["training_evidence"]
        or any(
            training[key] != 4
            for key in ("native_trainer_counter", "optimizer_counter", "inner_optimizer_counter", "schedule_counter")
        )
        or training["learning_rate"] != FLOAT32_LEARNING_RATE
        or training["skip_bad_steps"] is not False
        or training["resume"] is not False
        or training["observed_step_metrics"]["step"] != 3
        or any(
            not math.isfinite(training["observed_step_metrics"][key]) or training["observed_step_metrics"][key] <= 0
            for key in ("grad_norm", "update_norm")
        )
        or not math.isfinite(training["observed_step_metrics"]["loss"])
        or reload["verified"] is not True
        or reload["status"] != "SUCCESS"
        or reload["model_identity"] != recovery_identity
        or reload["model_uri"] != export
        or reload["suite"] != "mmlu-smoke"
        or reload["limit"] != 1
        or not reload["evidence_uri"]
        or len(reload["evidence_sha256"]) != 64
        or len(shards) != RECOVERY_SHARDS
        or len({item["path"] for item in shards}) != RECOVERY_SHARDS
        or set(recovered["hf_weight_map"].values()) != {item["path"] for item in shards}
        or len(recovered["hf_weight_map"]) != RECOVERY_TENSORS
        or {(item["path"], item["size"], item["sha256"]) for item in shards}
        != {(item["name"], item["bytes"], item["sha256"]) for item in recovered["numeric_proof"]["shards"]}
        or any(item["size"] <= 0 or len(item["sha256"]) != 64 for item in shards)
        or not recovered["hf_files"]
        or any(len(item["sha256"]) != 64 for item in recovered["hf_files"])
        or not all(record["hf_verified"][key] is True for key in ("shards", "config", "tokenizer", "eos"))
    ):
        raise ValueError("Post-SFT requires the amended recovered export and its own serving reload")
    validate_numeric_proof(recovered["numeric_proof"], amendment)
    return export
