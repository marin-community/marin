# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import io
import json

import numpy as np
import pytest
from tokenizers import Tokenizer

from experiments.grug.moe.inference_preflight import (
    BRANCH_COUNT,
    CASES,
    IDENTITY_CHAT_TOKENS,
    MARIN_BASE_SHA,
    P0_SMOKE_CASES,
    VLLM_SHA,
    aggregate_preflight_status,
    decode_routed_experts,
    deterministic_balanced_routing_fixture,
    deterministic_boundary_workload,
    deterministic_capacity_stress_workload,
    deterministic_trajectory_workload,
    deterministic_workload,
    expert_parallel_rank_histogram,
    frozen_manifest,
    hybrid_kv_cache_hit_alignment,
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


def test_pinned_attention_schedule_is_every_six_plus_final() -> None:
    assert layer_types(12) == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert layer_types(7)[-1] == "full_attention"
    assert layer_types(7).count("full_attention") == 2
    assert layer_types(8, global_interval=4).count("full_attention") == 2


def test_reference_case_is_the_exact_two_node_architecture() -> None:
    case = CASES["reference-ep8"]
    assert case.hidden_size == 6144
    assert case.num_hidden_layers == 48
    assert case.num_attention_heads == 48
    assert case.num_key_value_heads == 12
    assert case.local_kv_heads == 12
    assert case.global_kv_heads == 6
    assert case.global_every == 6
    assert case.num_experts_per_tok == 4
    assert case.num_experts == 128
    assert case.moe_intermediate_size == 3072
    assert case.shared_expert_intermediate_size == 6144
    assert case.num_shared_experts == 2
    assert case.sliding_window == 512
    assert case.rope_fraction == 0.5
    assert case.rope_fused
    assert case.sconv
    assert case.sconv_sites == ("k", "v", "attn", "mlp")
    assert case.data_parallel_size == 8
    assert case.node_count == 2


def test_granular_smoke_keeps_matched_active_width() -> None:
    reference = CASES["one-node-ep4"]
    granular = CASES["granular-ep16"]
    assert reference.num_experts_per_tok * reference.moe_intermediate_size == (
        granular.num_experts_per_tok * granular.moe_intermediate_size
    )
    assert granular.num_experts == 2 * reference.num_experts
    assert granular.node_count == 4


def test_final_case_is_exact_reference_not_granular_variant() -> None:
    ep8 = CASES["reference-ep8"]
    ep16 = CASES["exact-reference-ep16"]
    assert ep16.data_parallel_size == 16
    assert ep16.node_count == 4
    assert {key: value for key, value in ep16.hf_config().items() if key not in {"vocab_size"}} == {
        key: value for key, value in ep8.hf_config().items() if key not in {"vocab_size"}
    }
    assert ep16.num_experts_per_tok == 4
    assert ep16.num_experts == 128
    assert ep16.moe_intermediate_size == 3072


def test_p0_family_manifest_names_every_distinct_path() -> None:
    assert set(P0_SMOKE_CASES) == {
        "uniform-kv_every4_sconv-off",
        "heterogeneous-kv_every6_sconv-on",
        "global-kv-2_window-2048",
        "top8-256_ep16",
        "exact-ep16",
    }
    assert {case for cases in P0_SMOKE_CASES.values() for case in cases} <= CASES.keys()


def test_top_level_status_is_the_literal_required_conjunction() -> None:
    passing = {
        "placement": {"passed": True},
        "all_rank_health": True,
        "correctness": {"passed": True},
        "duration": True,
        "token_count": True,
        "repeatability": {"passed": True},
        "artifact_readback": True,
    }
    assert aggregate_preflight_status(passing)["status"] == "passed"
    for failed_check in passing:
        components = {**passing, failed_check: False}
        result = aggregate_preflight_status(components)
        assert result["status"] == "failed"
        assert not result["checks"][failed_check]
    with pytest.raises(ValueError, match="missing required"):
        aggregate_preflight_status({key: True for key in passing if key != "placement"})


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
    exact_schedule_uniform_heads = predict_kv_bytes(
        sequence_length=65_536,
        local_layers=40,
        global_layers=8,
        local_kv_heads=12,
        global_kv_heads=12,
        head_dim=128,
        sliding_window=512,
    )
    loadable_semantics = predict_kv_bytes(
        sequence_length=65_536,
        local_layers=36,
        global_layers=12,
        local_kv_heads=12,
        global_kv_heads=12,
        head_dim=128,
        sliding_window=512,
    )
    full_allocation = predict_kv_bytes(
        sequence_length=65_536,
        local_layers=0,
        global_layers=48,
        local_kv_heads=12,
        global_kv_heads=12,
        head_dim=128,
        sliding_window=512,
    )
    assert exact / 2**30 == pytest.approx(1.6171875)
    assert exact_schedule_uniform_heads / 2**30 == pytest.approx(3.1171875)
    assert loadable_semantics / 2**30 == pytest.approx(4.60546875)
    assert full_allocation / 2**30 == pytest.approx(18.0)


def test_workload_is_the_exact_18_root_144_branch_acceptance_shape() -> None:
    workload = deterministic_workload()
    assert workload["root_count"] == 18
    assert workload["request_count"] == BRANCH_COUNT == 144
    assert workload["history_lengths"] == [10_240, 30_720, 62_464]
    assert workload["append_tokens"] == 1_024
    assert workload["response_tokens"] == 2_048
    assert workload["final_lengths"] == [13_312, 33_792, 65_536]
    assert len({request["request_id"] for request in workload["requests"]}) == BRANCH_COUNT
    assert len(workload["roots"]) == 18
    assert [sum(root["cohort"] == cohort for root in workload["roots"]) for cohort in ("short", "medium", "long")] == [
        6,
        6,
        6,
    ]
    for request in workload["requests"]:
        assert len(materialize_prompt(workload, request)) == request["prefix_token_count"] + 1_024
        assert request["max_tokens"] == 2_048
        assert request["final_token_count"] in {13_312, 33_792, 65_536}


@pytest.mark.parametrize(
    ("case_name", "expected_alignment", "expected_lengths"),
    [
        ("one-node-ep4", 32, [33, 513]),
        ("legacy-control-ep4", 16, [17, 513]),
        ("kv2-window2048-ep4", 64, [65, 513]),
    ],
)
def test_boundary_workload_crosses_hybrid_alignment_and_window(
    case_name: str,
    expected_alignment: int,
    expected_lengths: list[int],
) -> None:
    workload = deterministic_boundary_workload(CASES[case_name])
    assert workload["cache_hit_alignment"] == expected_alignment
    assert workload["lengths"] == expected_lengths
    for request in workload["requests"]:
        assert materialize_prompt(workload, request) != materialize_prompt(workload, request, mutated=True)
        root = workload["roots"][request["root"]]
        assert root["prefix_token_ids"][0] == root["mutated_prefix_token_ids"][0]
        assert root["prefix_token_ids"][1] != root["mutated_prefix_token_ids"][1]
        assert root["prefix_token_ids"][2:] == root["mutated_prefix_token_ids"][2:]


def test_hybrid_cache_hit_alignment_matches_every_frozen_case() -> None:
    assert {name: hybrid_kv_cache_hit_alignment(case) for name, case in CASES.items()} == {
        "tiny": 32,
        "one-node-ep4": 32,
        "legacy-control-ep4": 16,
        "kv2-window2048-ep4": 64,
        "reference-ep8": 32,
        "granular-ep16": 32,
        "exact-reference-ep16": 32,
        "window1024-ep16": 32,
        "window2048-ep16": 32,
        "global-every4-ep16": 32,
        "exact-reference-131k-ep16": 32,
        "window1024-131k-ep16": 32,
        "window2048-131k-ep16": 32,
        "global-every4-131k-ep16": 32,
    }


def test_attention_screen_cases_change_only_the_declared_property() -> None:
    reference = CASES["exact-reference-ep16"]
    for name, field, expected in (
        ("window1024-ep16", "sliding_window", 1024),
        ("window2048-ep16", "sliding_window", 2048),
        ("global-every4-ep16", "global_every", 4),
    ):
        candidate = CASES[name]
        changed = {
            key
            for key, value in dataclasses.asdict(reference).items()
            if key != "name" and dataclasses.asdict(candidate)[key] != value
        }
        assert changed == {field}
        assert getattr(candidate, field) == expected


def test_131k_attention_cases_change_only_context_ceiling_from_screen_cases() -> None:
    for base_name in (
        "exact-reference-ep16",
        "window1024-ep16",
        "window2048-ep16",
        "global-every4-ep16",
    ):
        extended_name = f"{base_name.removesuffix('-ep16')}-131k-ep16"
        base = dataclasses.asdict(CASES[base_name])
        extended = dataclasses.asdict(CASES[extended_name])
        changed = {key for key, value in base.items() if key != "name" and extended[key] != value}
        assert changed == {"max_model_len"}
        assert extended["max_model_len"] == 131_072


def test_trajectory_and_capacity_workloads_are_exact_and_deterministic() -> None:
    trajectory = deterministic_trajectory_workload()
    assert trajectory == deterministic_trajectory_workload()
    assert trajectory != deterministic_trajectory_workload(seed=4321)
    assert trajectory["request_count"] == 144
    assert trajectory["turn_count"] == 4
    assert trajectory["final_lengths"] == [22_528, 43_008, 65_536]
    assert {tuple(turn["final_token_count"] for turn in request["turns"]) for request in trajectory["requests"]} == {
        (13_312, 16_384, 19_456, 22_528),
        (33_792, 36_864, 39_936, 43_008),
        (56_320, 59_392, 62_464, 65_536),
    }

    capacity = deterministic_capacity_stress_workload()
    assert capacity == deterministic_capacity_stress_workload()
    assert capacity != deterministic_capacity_stress_workload(seed=4321)
    assert capacity["root_count"] == 6
    assert capacity["request_count"] == 48
    assert {request["prompt_token_count"] for request in capacity["requests"]} == {122_880}
    assert {request["final_token_count"] for request in capacity["requests"]} == {131_072}


def test_workload_is_deterministic() -> None:
    assert deterministic_workload() == deterministic_workload()
    assert deterministic_workload(seed=1) != deterministic_workload(seed=2)


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


def test_linear_expert_placement_and_balanced_control() -> None:
    assert expert_parallel_rank_histogram([2, 2, 3, 1], ep_size=2) == [4, 4]
    fixture = deterministic_balanced_routing_fixture(num_experts=128, top_k=4, ep_size=8)
    assert fixture["expert_histogram"] == [4] * 128
    assert fixture["ep_rank_histogram"] == [64] * 8
    assert fixture["all_experts_equal"]
    assert fixture["all_ep_ranks_equal"]


def test_linear_expert_placement_rejects_partial_ranks() -> None:
    with pytest.raises(ValueError, match="divisible"):
        expert_parallel_rank_histogram([1, 2, 3], ep_size=2)


def test_write_case_freezes_config_workload_and_manifest(tmp_path) -> None:
    write_case(tmp_path, case=CASES["tiny"], run_id="unit", git_sha="f" * 40)
    config = json.loads((tmp_path / "config.json").read_text())
    workload = json.loads((tmp_path / "workload.json").read_text())
    correctness_workload = json.loads((tmp_path / "correctness-workload.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    tokenizer = json.loads((tmp_path / "tokenizer.json").read_text())
    assert config["architectures"] == ["GrugMoeForCausalLM"]
    assert config["model_type"] == "grug_moe"
    assert workload["request_count"] == BRANCH_COUNT
    assert correctness_workload["cache_hit_alignment"] == 32
    assert correctness_workload["lengths"] == [33, 513]
    assert tokenizer["model"]["vocab"][IDENTITY_CHAT_TOKENS[0]] == 0
    assert tokenizer["model"]["vocab"][IDENTITY_CHAT_TOKENS[255]] == 255
    assert len(set(IDENTITY_CHAT_TOKENS)) == 256
    assert all(len(token) == 1 and not token.isspace() for token in IDENTITY_CHAT_TOKENS)
    prompt_ids = [index % 256 for index in range(65_535)]
    content = "".join(IDENTITY_CHAT_TOKENS[token_id] for token_id in prompt_ids)
    assert len(content) == len(prompt_ids) == 65_535
    assert Tokenizer.from_file(str(tmp_path / "tokenizer.json")).encode(content).ids == prompt_ids
    expected_manifest = json.loads(json.dumps(frozen_manifest(CASES["tiny"], run_id="unit", git_sha="f" * 40)))
    assert manifest == expected_manifest
