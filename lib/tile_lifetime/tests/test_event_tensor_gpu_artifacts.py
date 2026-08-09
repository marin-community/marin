# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

from tile_lifetime.event_dataflow_adapters import streaming_contract_fold_event_descriptor
from tile_lifetime.ir import DType
from tile_lifetime.sm100_grouped_contract_event_codegen import sm100_bf16_grouped_contract_event_schedule
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.streaming_event_schedule import derive_streaming_physical_event_schedule

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "artifacts"


def _verify_manifest(artifact: Path) -> None:
    for line in (artifact / "SHA256SUMS").read_text().splitlines():
        expected, relative_path = line.split("  ", maxsplit=1)
        digest = hashlib.sha256((artifact / relative_path).read_bytes()).hexdigest()
        assert digest == expected


def test_sm90_streaming_attachment_artifact_matches_the_derived_schedule_boundary() -> None:
    artifact = ARTIFACT_ROOT / "event_tensor_sm90_fold_state_replay_h100_v1"
    _verify_manifest(artifact)
    result = json.loads((artifact / "result.json").read_text())
    semantic = build_attention_tensor_program(
        batch_size=1,
        query_length=result["shape"]["sequence"],
        key_length=result["shape"]["sequence"],
        query_heads=result["shape"]["query_heads"],
        key_value_heads=result["shape"]["key_value_heads"],
        key_dimension=result["shape"]["head_dimension"],
        value_dimension=result["shape"]["head_dimension"],
        score_map=apply_causal_score_mask(scaled_score_map(result["score_map"]["scale"])),
        input_dtype=DType.BF16,
    )
    program = derive_streaming_attention(
        semantic,
        schedule=StreamingTileSchedule(
            query_tile_size=result["schedule"]["tile_m"],
            key_value_tile_size=result["schedule"]["tile_n"],
            pipeline_depth=result["schedule"]["stages"],
        ),
    )
    schedule = derive_streaming_physical_event_schedule(streaming_contract_fold_event_descriptor(program))

    assert result["environment"]["hardware"] == "NVIDIA H100 80GB HBM3"
    assert schedule.first_streamed_input_stages == schedule.second_streamed_input_stages == result["schedule"]["stages"]
    assert schedule.workers.cta_threads == result["schedule"]["threads"]
    assert result["event_to_canonical_ratio"] < 1.01
    assert {capture["maximum_absolute_error"] for capture in result["event_tensor"]["captures"]} == {0.015625}
    assert all(capture["deterministic"] for capture in result["event_tensor"]["captures"])
    assert {
        capture["deterministic_output_hash"]
        for implementation in (result["canonical"], result["event_tensor"])
        for capture in implementation["captures"]
    } == {"36626c594342bc25b794afd7457a02eefd839c6abd192eb69c24f0bd165bd6c7"}


def test_sm100_grouped_contract_artifact_matches_the_generated_wrapper_abi() -> None:
    artifact = ARTIFACT_ROOT / "event_tensor_grouped_contract_sm100_gb200_v0"
    _verify_manifest(artifact)
    result = json.loads((artifact / "result.json").read_text())
    schedule = sm100_bf16_grouped_contract_event_schedule()

    assert result["environment"]["gpu_name"] == "NVIDIA GB200"
    assert result["scope"] == "generated_sync_abi_wrapper_proof_not_internal_barrier_codegen"
    assert result["status"] == "ok"
    assert result["correctness"]["passed"]
    assert result["correctness"]["nan_count"] == result["correctness"]["infinity_count"] == 0
    assert result["event_tensor_schedule"] == {
        "cluster_ctas": schedule.descriptor.workers.cluster_ctas,
        "fingerprint": schedule.fingerprint,
        "load_pipeline_stages": schedule.descriptor.load_pipeline_stages,
        "logical_event_counts": {
            "operand_ready": schedule.operand_ready_count,
            "operand_release": schedule.operand_release_count,
            "output_ready": schedule.output_ready_count,
            "output_release": schedule.output_release_count,
        },
        "operand_release_point": schedule.descriptor.operand_release_point.value,
        "operand_transaction_bytes": schedule.operand_transaction_bytes,
        "output_release_point": schedule.descriptor.output_release_point.value,
        "transaction_completion_enabled": schedule.transaction_completion_enabled,
    }
    assert result["kernel_resources"] == {
        "barriers": 5,
        "registers": 255,
        "spill_load_bytes": 0,
        "spill_store_bytes": 0,
        "static_shared_memory_bytes": 224,
    }
