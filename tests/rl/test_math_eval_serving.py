# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.rate import MODEL_PROFILES
from experiments.post_training.math_eval.scoring import SEMANTIC_DEPENDENCIES
from experiments.post_training.math_eval.serving import (
    EAST_PREFIX,
    SERVING_MODELS,
    RatingServingConfig,
    load_rating_inputs,
    serving_configuration,
    serving_specification,
    verify_wheel_receipt,
)
from experiments.post_training.math_eval.serving_audit import validate_serving_generation


def config():
    return RatingServingConfig(
        model="qwen",
        pool_uri=EAST_PREFIX + "fixture/pool",
        overlay_uri=EAST_PREFIX + "fixture/overlay.json",
        overlay_sha256="a" * 64,
        manifest_sha256="b" * 64,
        expected_ids=("c" * 64,),
        samples=8,
        output_uri=EAST_PREFIX + "fixture/ratings",
        source_commit="d" * 40,
        tensor_parallel_size=1,
    )


def test_rating_configuration_is_roundtrip_stable_and_has_no_request_seed():
    specification = config()
    restored = RatingServingConfig(**json.loads(json.dumps(asdict(specification))))
    model, engine = serving_configuration(restored)
    assert model.max_model_len == 3072
    assert engine.max_num_seqs == 64
    assert engine.extra_args == ("--seed", "17", "--generation-config", "vllm")
    snowball = replace(specification, model="snowball", samples=4, data_parallel_size=4)
    assert serving_configuration(snowball)[0].max_model_len == 12288


@pytest.mark.parametrize(
    "changes",
    [
        {"samples": 4},
        {"samples": 0},
        {"expected_ids": ()},
        {"expected_ids": ("c" * 64,) * 2},
        {"pool_uri": "s3://foreign/pool"},
        {"source_commit": "main"},
        {"concurrency": 0},
        {"request_timeout_seconds": 0},
        {"model": "snowball", "tensor_parallel_size": 1},
        {"model": "snowball", "tensor_parallel_size": 4, "data_parallel_size": 4},
        {"model": "snowball", "tensor_parallel_size": 8, "data_parallel_size": 4},
        {"data_parallel_size": 4},
    ],
)
def test_rating_configuration_rejects_changed_membership_region_or_unbounded_protocol(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def wheel_evidence():
    return {
        "release_tag": "marin-vllm-gpu-20260827-f0d7cc7f5874",
        "source_commit": "f0d7cc7f587482e0ab771e3c9715e726eb914e60",
        "version": "0.0.0.dev20260827+marin.f0d7cc7f5874",
        "wheel_sha256": "400cb816aea3d46841da6cf9f8e3f4e40bcb7b6a60cf66ecf59e7afd7da4c7c7",
        "wheel_url": (
            "https://github.com/marin-community/vllm/releases/download/"
            "marin-vllm-gpu-20260827-f0d7cc7f5874/vllm-0.0.0.dev20260827%2Bmarin.f0d7cc7f5874-cp38-"
            "abi3-manylinux_2_28_x86_64.whl"
        ),
        "sm_targets": ["9.0"],
        "compute_capability": "9.0",
        "extension_path": "/fixture/vllm/_C.so",
    }


def test_native_wheel_proof_preserves_limit_of_digest_claim():
    evidence = wheel_evidence()
    proof = verify_wheel_receipt("MARIN_VLLM_WHEEL_VERIFIED=" + json.dumps(evidence), evidence["version"])
    assert proof["wheel_digest_verification"] == "promoted_asset_provenance_only"


@pytest.mark.parametrize(
    "poison", ["source_commit", "wheel_url", "compute_capability", "version", "absent", "duplicate"]
)
def test_native_runtime_requires_installed_wheel_proof_and_matching_api_version(poison):
    evidence = wheel_evidence()
    version = evidence["version"]
    if poison in evidence:
        evidence[poison] = "wrong"
    line = "MARIN_VLLM_WHEEL_VERIFIED=" + json.dumps(evidence)
    log = "" if poison == "absent" else (line + "\n" + line if poison == "duplicate" else line)
    with pytest.raises(ValueError):
        verify_wheel_receipt(log, version)


def generation_fixture():
    cfg = config()
    model, _engine = serving_configuration(cfg)
    specification = serving_specification(cfg)
    runtime = wheel_evidence()
    generation = {
        "schema": "math_eval_serving_generation_v1",
        "score_dependency_versions": SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"},
        "specification": specification,
        "specification_sha256": audit.canonical_sha(specification),
        "task_id": "/atqamar/fixture/0",
        "attempt_id": 0,
        "worker_region_hint": None,
        "model_identity": "users/ahmad/models/async-rl-qwen3-0.6b@2026.09.08.83:8a30d2b5",
        "model_config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "tokenizer_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "prompt_template_id": "qwen3-c1899de-nothink-antibox-v1",
        "engine_global_seed": 17,
        "request_sampling_seed": None,
        "expected_ids_sha256": audit.canonical_sha(sorted(cfg.expected_ids)),
        "rows": 8,
        "runtime": runtime | {"api_version": runtime["version"]},
        "native_command": [
            "vllm",
            "serve",
            model.weights,
            "--seed",
            "17",
            "--generation-config",
            "vllm",
            "--max-model-len",
            "3072",
            "--tensor-parallel-size",
            "1",
            "--served-model-name",
            "rating-qwen",
            "--dtype",
            "bfloat16",
        ],
    }
    tasks = {
        "tasks": [
            {
                "task_id": generation["task_id"],
                "state": "TASK_STATE_SUCCEEDED",
                "exit_code": 0,
                "current_attempt_id": 0,
                "started_at": {"epoch_ms": "1000"},
                "finished_at": {"epoch_ms": "3601000"},
                "attempts": [{"attempt_id": 0, "state": "TASK_STATE_SUCCEEDED", "exit_code": 0}],
            }
        ]
    }
    generation["attempt_uid"] = "fixture-uid"
    generation["controller_allocation"] = {
        "controller_scope": "local",
        "resources": {"device": {"kind": "gpu", "variant": "H100", "count": 1}},
        "attempt_uid": "fixture-uid",
        "started_at_ms": 1000,
    }
    task = tasks["tasks"][0]
    task["cluster"] = "cw-us-east-02a"
    task["attempts"][0].update(started_at=task["started_at"], finished_at=task["finished_at"], attempt_uid="fixture-uid")
    native_job = {
        "job": {
            "job_id": "/atqamar/fixture",
            "state": "JOB_STATE_SUCCEEDED",
            "exit_code": 0,
            "cluster": "cw-us-east-02a",
            "task_count": 1,
            "completed_count": 1,
            "resources": {"device": {"gpu": {"variant": "H100", "count": 1}}},
        }
    }
    return cfg, generation, tasks, native_job


def test_generation_audit_binds_actual_terminal_task_and_native_command():
    cfg, generation, tasks, native_job = generation_fixture()
    protocol = validate_serving_generation(cfg, generation, tasks, native_job)
    assert protocol["task_gpu_hours"] == 1
    assert protocol["model_label"] == "qwen" and protocol["samples"] == 8


@pytest.mark.parametrize(
    "poison",
    [
        "model_identity",
        "tokenizer_sha256",
        "engine_global_seed",
        "request_sampling_seed",
        "rows",
        "native_seed",
        "failed_task",
        "retried_task",
        "missing_task",
        "foreign_task",
        "changed_specification",
    ],
)
def test_generation_audit_refuses_caller_labels_that_disagree_with_native_evidence(poison):
    cfg, generation, tasks, native_job = generation_fixture()
    if poison in generation:
        generation[poison] = "wrong"
    elif poison == "native_seed":
        generation["native_command"][generation["native_command"].index("--seed") + 1] = "29"
    elif poison == "failed_task":
        tasks["tasks"][0]["state"] = "TASK_STATE_FAILED"
    elif poison == "retried_task":
        tasks["tasks"][0]["attempts"].append(deepcopy(tasks["tasks"][0]["attempts"][0]))
    elif poison == "missing_task":
        tasks["tasks"] = []
    elif poison == "foreign_task":
        tasks["tasks"][0]["task_id"] = "/atqamar/foreign/0"
    else:
        generation["specification"]["engine"]["extra_args"] = ["--seed", "29"]
    with pytest.raises(ValueError):
        validate_serving_generation(cfg, generation, tasks, native_job)


def test_unicode_manifest_readback_uses_the_existing_pool_hash_representation(tmp_path):
    manifest = [{"prompt_sha256": "a" * 64, "problem": "How many café tables?"}]
    pool_hash = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    pq.write_table(pa.Table.from_pylist(manifest), tmp_path / "manifest.parquet")
    (tmp_path / "selection.json").write_text(json.dumps({"manifest_sha256": pool_hash}))
    overlay = {
        "manifest_sha256": pool_hash,
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": "c" * 64,
        "statuses": {"a" * 64: "accept"},
    }
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(overlay))
    cfg = SimpleNamespace(
        pool_uri=str(tmp_path),
        overlay_uri=str(overlay_path),
        manifest_sha256=pool_hash,
        overlay_sha256=audit.canonical_sha(overlay),
        expected_ids=("a" * 64,),
    )
    assert load_rating_inputs(cfg) == manifest
    cfg.manifest_sha256 = audit.canonical_sha(manifest)
    assert cfg.manifest_sha256 != pool_hash
    with pytest.raises(ValueError, match="manifest changed"):
        load_rating_inputs(cfg)


@pytest.mark.parametrize("poison", ["gpu_count", "gpu_variant", "region", "attempt_uid", "attempt_timestamps"])
def test_cost_and_region_require_actual_controller_allocation(poison):
    cfg, generation, tasks, native_job = generation_fixture()
    if poison == "gpu_count":
        native_job["job"]["resources"]["device"]["gpu"]["count"] = 8
    elif poison == "gpu_variant":
        native_job["job"]["resources"]["device"]["gpu"]["variant"] = "B200"
    elif poison == "region":
        tasks["tasks"][0]["cluster"] = "foreign-region"
    elif poison == "attempt_uid":
        tasks["tasks"][0]["attempts"][0]["attempt_uid"] = "other-attempt"
    else:
        tasks["tasks"][0]["attempts"][0]["started_at"] = {"epoch_ms": "2000"}
    with pytest.raises(ValueError):
        validate_serving_generation(cfg, generation, tasks, native_job)


def test_producer_interval_is_separate_from_allocation_and_rejects_impossible_duration():
    cfg, generation, tasks, native_job = generation_fixture()
    start = int(tasks["tasks"][0]["started_at"]["epoch_ms"])
    timing = {"started_at_ms": start + 100, "finished_at_ms": start + 1100, "monotonic_seconds": 1.0}
    generation["generation_and_native_scoring"] = timing
    protocol = validate_serving_generation(cfg, generation, tasks, native_job)
    assert protocol["generation_and_native_scoring"] == timing
    assert protocol["task_gpu_hours"] > timing["monotonic_seconds"] / 3600
    generation["generation_and_native_scoring"]["monotonic_seconds"] = 99999
    with pytest.raises(ValueError, match="Producer generation interval"):
        validate_serving_generation(cfg, generation, tasks, native_job)


def test_snowball_audit_counts_physical_gpus_and_binds_native_expert_parallel_flags():
    cfg, generation, tasks, native_job = generation_fixture()
    cfg = replace(cfg, model="snowball", samples=4, data_parallel_size=4)
    model, engine = serving_configuration(cfg)
    profile = MODEL_PROFILES["snowball"]
    generation.update(
        specification=serving_specification(cfg),
        model_identity=SERVING_MODELS["snowball"]["identity"],
        model_config_sha256=SERVING_MODELS["snowball"]["config_sha256"],
        tokenizer_sha256=profile["tokenizer_sha256"],
        prompt_template_id=profile["prompt_template_id"],
        rows=4,
        native_command=[
            "vllm",
            "serve",
            model.weights,
            "--max-model-len",
            "12288",
            "--tensor-parallel-size",
            "1",
            "--served-model-name",
            model.model_id,
            "--dtype",
            "bfloat16",
            "--max-num-seqs",
            "32",
            *engine.extra_args,
        ],
    )
    generation["specification_sha256"] = audit.canonical_sha(generation["specification"])
    native_job["job"]["resources"]["device"]["gpu"]["count"] = 4
    generation["controller_allocation"]["resources"]["device"]["count"] = 4
    protocol = validate_serving_generation(cfg, generation, tasks, native_job)
    assert protocol["task_gpu_hours"] == 4
    assert cfg.allocated_gpus == 4 and model.tensor_parallel_size == 1
    for flag in (
        "--enable-expert-parallel",
        "--data-parallel-size",
        "--max-num-seqs",
        "--kv-cache-dtype",
        "--model-loader-extra-config",
    ):
        poisoned = deepcopy(generation)
        poisoned["native_command"].remove(flag)
        with pytest.raises(ValueError, match="Native"):
            validate_serving_generation(cfg, poisoned, tasks, native_job)
    native_job["job"]["resources"]["device"]["gpu"]["count"] = 1
    with pytest.raises(ValueError):
        validate_serving_generation(cfg, generation, tasks, native_job)


def test_qwen_specification_preserves_frozen_single_gpu_wire_shape():
    cfg = config()
    specification = serving_specification(cfg)
    legacy = asdict(cfg)
    del legacy["data_parallel_size"]
    assert specification["config"] == json.loads(json.dumps(legacy))
    assert specification["engine"]["extra_args"] == ["--seed", "17", "--generation-config", "vllm"]
    assert cfg.allocated_gpus == 1
