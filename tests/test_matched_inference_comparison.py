# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import pytest
from levanter.inference.benchmark import BatchMeasurement, TokenWorkload, measure_batches

from experiments.benchmarks.matched_comparison import compare_fixture


@pytest.fixture
def paired_reports(tmp_path):
    workload = TokenWorkload([[1, 2], [3, 4]], 3)
    config = {"type": "hero", "num_layers": 2, "inference_attention_implementation": None}
    manifest = {"checkpoint": "sha256:shared-weights", "dtype": "bfloat16", "model_config": config}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "workload.json").write_text(json.dumps(dataclasses.asdict(workload)))
    for backend, elapsed in (("native", 2), ("vllm", 4)):
        report = dataclasses.asdict(
            measure_batches(
                workload,
                lambda tokens, elapsed=elapsed: BatchMeasurement(elapsed, [0.5, 0.5], [[5, 6, 7], [8, 9, 10]]),
                warmup_batches=1,
                measured_batches=2,
            )
        )
        common = {**manifest, "devices": ["NVIDIA H100"] * 2, "hardware_label": "H100x2-single-host"}
        if backend == "native":
            report["provenance"] = {
                **common,
                "backend": "levanter",
                "checkpoint": {"identity": manifest["checkpoint"], "hf_config": {"hidden_size": 256}},
                "device_kind": common["devices"],
                "mesh": {"replica_dcn": 1, "data": 1, "context": 1, "expert": 2, "model": 1},
                "model_config": {**config, "inference_attention_implementation": "gpu_pallas_bf16_3x"},
                "effective_execution": {"runtime": "jax_jit"},
            }
        else:
            report["provenance"] = {
                **common,
                "backend": "vllm",
                "hf_config": {"hidden_size": 256},
                "effective_dtype": "bfloat16",
                "engine_args": {
                    "tensor_parallel_size": 1,
                    "data_parallel_size": 2,
                    "data_parallel_size_local": 2,
                    "enable_expert_parallel": True,
                    "enable_prefix_caching": False,
                },
                "effective_execution": {"cudagraph_mode": "NONE", "enforce_eager": True},
            }
        (tmp_path / f"{backend}-result.json").write_text(json.dumps(report))
    return tmp_path


def test_paired_serialized_reports_compare_execution_choices_and_throughput(paired_reports):
    compare_fixture(paired_reports)
    saved = json.loads((paired_reports / "comparison.json").read_text())
    assert saved["native_over_vllm_throughput"] == 2
    assert saved["validation_tokens_agree"]
    assert saved["first_divergent_token"] is None
    assert saved["execution_config_differences"]["native"] == {
        "inference_attention_implementation": {"fixture": None, "actual": "gpu_pallas_bf16_3x"}
    }
    assert saved["effective_execution"]["vllm"]["cudagraph_mode"] == "NONE"


def test_paired_report_preserves_first_divergence_and_withholds_speed_ratio(paired_reports):
    path = paired_reports / "vllm-result.json"
    report = json.loads(path.read_text())
    workload = TokenWorkload(**report["workload"])
    changed = measure_batches(
        workload,
        lambda tokens: BatchMeasurement(4, [0.5, 0.5], [[5, 6, 7], [8, 99, 10]]),
        warmup_batches=1,
        measured_batches=2,
    )
    path.write_text(json.dumps({**dataclasses.asdict(changed), "provenance": report["provenance"]}))
    result = compare_fixture(paired_reports)
    assert result["native_over_vllm_throughput"] is None
    assert result["first_divergent_token"] == {
        "request_index": 1,
        "output_token_index": 1,
        "native_token": 9,
        "vllm_token": 99,
    }
    assert result["all_batch_hashes_agree_with_validation"] == {"native": True, "vllm": True}


@pytest.mark.parametrize("mismatch", ["checkpoint", "architecture", "workload", "topology"])
def test_incompatible_paired_reports_cannot_publish_a_comparison(paired_reports, mismatch):
    path = paired_reports / "vllm-result.json"
    report = json.loads(path.read_text())
    if mismatch == "checkpoint":
        report["provenance"]["checkpoint"] = "different-weights"
    elif mismatch == "architecture":
        report["provenance"]["model_config"]["num_layers"] = 4
    elif mismatch == "workload":
        report["workload"]["prompts"][0][0] = 99
    else:
        report["provenance"]["engine_args"]["enable_expert_parallel"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare_fixture(paired_reports)
    assert not (paired_reports / "comparison.json").exists()


def test_paired_report_withholds_ratio_when_a_timed_batch_differs(paired_reports):
    path = paired_reports / "native-result.json"
    report = json.loads(path.read_text())
    report["samples"][0]["output_sha256"] = "different-timed-generation"
    path.write_text(json.dumps(report))
    result = compare_fixture(paired_reports)
    assert result["validation_tokens_agree"]
    assert result["native_over_vllm_throughput"] is None
    assert not result["all_batch_hashes_agree_with_validation"]["native"]
