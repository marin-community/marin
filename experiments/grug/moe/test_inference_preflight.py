# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest

from experiments.grug.moe.inference_preflight import (
    BRANCH_COUNT,
    CASES,
    MARIN_BASE_SHA,
    VLLM_SHA,
    decode_routed_experts,
    deterministic_workload,
    frozen_manifest,
    layer_types,
    materialize_prompt,
    metric_delta,
    parse_prometheus,
    predict_kv_bytes,
    routing_histogram,
    write_case,
)


def test_frozen_shas_are_full_and_distinct() -> None:
    assert len(MARIN_BASE_SHA) == 40
    assert len(VLLM_SHA) == 40
    assert MARIN_BASE_SHA != VLLM_SHA


def test_pinned_attention_schedule_is_every_four_plus_final() -> None:
    assert layer_types(8) == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert layer_types(6)[-1] == "full_attention"
    assert layer_types(6).count("full_attention") == 2


def test_reference_case_is_the_loadable_uniform_kv_approximation() -> None:
    case = CASES["reference-ep8"]
    assert case.hidden_size == 6144
    assert case.num_hidden_layers == 48
    assert case.num_attention_heads == 48
    assert case.num_key_value_heads == 12
    assert case.num_experts_per_tok == 4
    assert case.num_experts == 128
    assert case.moe_intermediate_size == 3072
    assert case.sliding_window == 512
    assert case.data_parallel_size == 8
    assert case.node_count == 2


def test_granular_case_keeps_matched_active_width() -> None:
    reference = CASES["reference-ep8"]
    granular = CASES["granular-ep16"]
    assert reference.num_experts_per_tok * reference.moe_intermediate_size == (
        granular.num_experts_per_tok * granular.moe_intermediate_size
    )
    assert granular.num_experts == 2 * reference.num_experts
    assert granular.node_count == 4


def test_kv_prediction_reproduces_reference_estimates() -> None:
    exact = predict_kv_bytes(
        sequence_length=65_536,
        local_layers=40,
        global_layers=8,
        local_kv_heads=12,
        global_kv_heads=6,
        head_dim=128,
        sliding_window=512,
    )
    uniform = predict_kv_bytes(
        sequence_length=65_536,
        local_layers=40,
        global_layers=8,
        local_kv_heads=12,
        global_kv_heads=12,
        head_dim=128,
        sliding_window=512,
    )
    assert exact / 2**30 == pytest.approx(1.6171875)
    assert uniform / 2**30 == pytest.approx(3.1171875)
    assert (uniform - exact) / 2**30 == pytest.approx(1.5)


def test_workload_has_18_roots_144_branches_and_required_boundaries() -> None:
    workload = deterministic_workload(max_prefix_tokens=65_536)
    assert workload["root_count"] == 18
    assert workload["request_count"] == BRANCH_COUNT == 144
    assert 17 in workload["lengths"]
    assert 513 in workload["lengths"]
    assert 65_536 in workload["lengths"]
    assert len({request["request_id"] for request in workload["requests"]}) == BRANCH_COUNT
    assert len(workload["roots"]) == 18
    assert all(
        materialize_prompt(workload, request) != materialize_prompt(workload, request, mutated=True)
        for request in workload["requests"]
    )
    for root in workload["roots"]:
        prefix = root["prefix_token_ids"]
        mutated = root["mutated_prefix_token_ids"]
        assert prefix[0] == mutated[0]
        assert prefix[1] != mutated[1]
        assert prefix[2:] == mutated[2:]


def test_workload_is_deterministic() -> None:
    assert deterministic_workload(max_prefix_tokens=2048) == deterministic_workload(max_prefix_tokens=2048)
    assert deterministic_workload(max_prefix_tokens=2048, seed=1) != deterministic_workload(
        max_prefix_tokens=2048, seed=2
    )


def test_prometheus_parser_sums_labeled_ranks_and_computes_delta() -> None:
    before = parse_prometheus(
        """
        # HELP vllm:prefix_cache_hits Prefix hits
        vllm:prefix_cache_hits{model_name="x",rank="0"} 4
        vllm:prefix_cache_hits{model_name="x",rank="1"} 7
        """
    )
    after = parse_prometheus(
        """
        vllm:prefix_cache_hits{model_name="x",rank="0"} 9
        vllm:prefix_cache_hits{model_name="x",rank="1"} 12
        """
    )
    assert before["vllm:prefix_cache_hits"] == 11
    assert metric_delta(before, after, "vllm:prefix_cache_hits") == 10
    assert (
        metric_delta(
            {"vllm:generation_tokens_total": 8},
            {"vllm:generation_tokens_total": 21},
            "vllm:generation_tokens",
        )
        == 13
    )


def test_routed_expert_transport_and_histogram() -> None:
    routed = np.array([[[0, 2], [1, 2]], [[3, 2], [0, 1]]], dtype=np.int32)
    buffer = io.BytesIO()
    np.save(buffer, routed)
    decoded = decode_routed_experts(base64.b64encode(buffer.getvalue()).decode("ascii"))
    np.testing.assert_array_equal(decoded, routed)
    assert routing_histogram(decoded, num_experts=4) == [2, 2, 3, 1]


def test_write_case_freezes_config_workload_and_manifest(tmp_path) -> None:
    write_case(tmp_path, case=CASES["tiny"], run_id="unit", git_sha="f" * 40)
    config = json.loads((tmp_path / "config.json").read_text())
    workload = json.loads((tmp_path / "workload.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert config["architectures"] == ["GrugMoeForCausalLM"]
    assert config["model_type"] == "grug_moe"
    assert workload["request_count"] == BRANCH_COUNT
    assert manifest == frozen_manifest(CASES["tiny"], run_id="unit", git_sha="f" * 40)
