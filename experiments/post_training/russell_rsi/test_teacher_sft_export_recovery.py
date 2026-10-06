# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check counter-based recovery binding without storage or model calls."""

from copy import deepcopy

import pytest

from experiments.post_training.russell_rsi.launch_post_teacher_sft import qualified_four_update_sft
from experiments.post_training.russell_rsi.teacher_sft_export_qualification import (
    FLOAT32_LEARNING_RATE,
    PROTOCOL,
    qualified_recovered_four_update_sft,
)
from experiments.post_training.russell_rsi.teacher_sft_export_recovery import copy_saved_shard

PRODUCER = "checkpoints/producer@v1:12345678"
ROOT = "s3://region/bank/producer"
RECOVERY = "checkpoints/recovery@v2:87654321"
RECOVERY_ROOT = "s3://region/bank/recovery"
SHA = "a" * 64


@pytest.fixture
def recovered_record():
    producer = {"identity": PRODUCER, "root": ROOT, "config_sha256": SHA, "status": "FAILED"}
    training = {
        "native_trainer_counter": 4,
        "optimizer_counter": 4,
        "inner_optimizer_counter": 4,
        "schedule_counter": 4,
        "learning_rate": FLOAT32_LEARNING_RATE,
        "skip_bad_steps": False,
        "resume": False,
        "observed_step_metrics": {"step": 3, "loss": 0.5, "grad_norm": 4.2, "update_norm": 0.17},
    }
    amendment = {
        "protocol": PROTOCOL,
        "producer": producer,
        "training_evidence": training,
        "original_telemetry_gate": "UNMET",
        "missing_metric_steps": [0, 1, 2],
        "required_numeric_validation": {
            "shard_count": 39,
            "tensor_count": 502,
            "parameter_count": 67078882816,
            "predetermined_comparison_tensors": ["a", "b", "c"],
        },
    }
    shards = [
        {
            "name": f"model-{i}.safetensors",
            "bytes": 2,
            "sha256": SHA,
            "nonfinite_count": 0,
            "bf16_values_checked": 67078882816 - 38 if i == 0 else 1,
            "conditional_reads": True,
            "before_after_identity_equal": True,
        }
        for i in range(39)
    ]
    proof = {
        "status": "passed",
        "input_manifest_sha256": SHA,
        "all_saved_bf16_values_finite": True,
        "fixed_tensor_net_change": True,
        "four_positive_updates_proved": False,
        "missing_metric_steps": [0, 1, 2],
        "generation_requests": 0,
        "source": f"{ROOT}/hf/step-3",
        "shard_count": 39,
        "tensor_count": 502,
        "parameter_count": 67078882816,
        "shards": shards,
        "fixed_tensor_comparisons": [
            {
                "key": k,
                "both_finite": True,
                "before_after_identity_equal": True,
                "changed_bf16_word_count": int(k == "a"),
            }
            for k in ["a", "b", "c"]
        ],
    }
    recovered = {
        "protocol": PROTOCOL,
        "producer": producer,
        "amendment": amendment,
        "amendment_sha256": SHA,
        "numeric_proof_sha256": SHA,
        "numeric_manifest_sha256": SHA,
        "training_evidence": training,
        "numeric_proof": proof,
        "original_telemetry_gate": "UNMET",
        "missing_metric_steps": [0, 1, 2],
        "hf_export_uri": f"{RECOVERY_ROOT}/hf/step-3",
        "hf_shards": [{"path": x["name"], "size": 2, "sha256": SHA} for x in shards],
        "hf_weight_map": {f"tensor{i}": shards[i % 39]["name"] for i in range(502)},
        "hf_files": [{"path": "config.json", "sha256": SHA}],
    }
    return {
        "protocol": PROTOCOL,
        "recovery": recovered,
        "source_config_sha256": SHA,
        "recovery_identity": RECOVERY,
        "recovery_root": RECOVERY_ROOT,
        "recovery_status": "SUCCESS",
        "amendment_sha256": SHA,
        "numeric_proof_sha256": SHA,
        "serving_reload": {
            "verified": True,
            "status": "SUCCESS",
            "model_identity": RECOVERY,
            "model_uri": f"{RECOVERY_ROOT}/hf/step-3",
            "suite": "mmlu-smoke",
            "limit": 1,
            "evidence_uri": "saved-reload",
            "evidence_sha256": SHA,
        },
        "hf_verified": dict.fromkeys(["shards", "config", "tokenizer", "eos"], True),
    }


def qualify(record):
    return qualified_recovered_four_update_sft(
        record,
        producer_identity=PRODUCER,
        producer_root=ROOT,
        recovery_identity=RECOVERY,
        recovery_root=RECOVERY_ROOT,
        source_config_sha256=SHA,
    )


def test_amended_saved_export_qualifies_without_missing_step_claim(recovered_record):
    assert qualify(recovered_record) == f"{RECOVERY_ROOT}/hf/step-3"


@pytest.mark.parametrize(
    "path,value",
    [
        (("recovery", "producer", "identity"), "other"),
        (("source_config_sha256",), "b" * 64),
        (("recovery_status",), "FAILED"),
        (("numeric_proof_sha256",), "b" * 64),
        (("serving_reload", "model_identity"), PRODUCER),
        (("serving_reload", "model_uri"), f"{ROOT}/hf/step-3"),
        (("recovery", "numeric_proof", "shards", 0, "nonfinite_count"), 1),
        (("recovery", "numeric_proof", "fixed_tensor_net_change"), False),
    ],
)
def test_recovery_rejects_changed_binding_or_weight_proof(recovered_record, path, value):
    changed = deepcopy(recovered_record)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        qualify(changed)


def test_strict_four_update_gate_still_rejects_missing_early_metrics(recovered_record):
    record = {
        **recovered_record["recovery"],
        "protocol": "teacher-sft-four-update-qualification-v1",
        "sft_identity": RECOVERY,
        "sft_root": RECOVERY_ROOT,
        "optimizer_updates": 4,
        "learning_rate": 1e-6,
        "loss": 0.5,
        "gradient_norm": 4.2,
        "update_norm": 0.17,
        "serving_reload": recovered_record["serving_reload"],
        "hf_verified": recovered_record["hf_verified"],
        "optimizer_steps": [
            {"step": 3, "skipped": False, "learning_rate": 1e-6, "loss": 0.5, "gradient_norm": 4.2, "update_norm": 0.17}
        ],
    }
    with pytest.raises(ValueError, match="four complete"):
        qualified_four_update_sft(record, identity=RECOVERY, root=RECOVERY_ROOT)


class CopyStorage:
    def __init__(self, changed=False):
        self.changed = changed
        self.calls = []

    def split_path(self, path):
        bucket, key = path.removeprefix("s3://").split("/", 1)
        return bucket, key, None

    def call_s3(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method == "copy_object":
            return {"CopyObjectResult": {"ETag": '"copy"'}}
        if kwargs["Key"] == "recovery/shard":
            return {"ContentLength": 2, "ETag": '"copy"'}
        return {"ContentLength": 2, "ETag": '"changed"' if self.changed else '"source"'}


def test_copy_uses_conditional_server_side_request_and_preserves_source_sha():
    storage = CopyStorage()
    result = copy_saved_shard(
        storage,
        "s3://region/producer/shard",
        "s3://region/recovery/shard",
        {"name": "shard", "etag": '"source"', "bytes": 2, "sha256": SHA},
    )
    assert result["sha256"] == SHA
    copies = [kwargs for method, kwargs in storage.calls if method == "copy_object"]
    assert copies == [
        {
            "Bucket": "region",
            "Key": "recovery/shard",
            "CopySource": {"Bucket": "region", "Key": "producer/shard"},
            "CopySourceIfMatch": '"source"',
        }
    ]


def test_changed_source_rejected_before_copy():
    storage = CopyStorage(changed=True)
    with pytest.raises(ValueError, match="changed before"):
        copy_saved_shard(
            storage,
            "s3://region/producer/shard",
            "s3://region/recovery/shard",
            {"name": "shard", "etag": '"source"', "bytes": 2, "sha256": SHA},
        )
    assert all(method != "copy_object" for method, _ in storage.calls)
