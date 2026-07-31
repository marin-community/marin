# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.grug.moe.inference_preflight import (
    CASES,
    FROZEN_FIXTURE_PATH,
    SNOWBALL_EXPORT,
    VLLM_SHA,
    deterministic_boundary_workload,
)
from scripts.iris.dev_gpu import CoreweaveTarget, DevGpuState, PodRef, Priority
from scripts.iris.grugmoe_inference_preflight import (
    LOCAL_DP_SIZE,
    _acceptance_components,
    _free_port,
    _immutable_image,
    _record_completion,
    _unattended_worker_argv,
    _validate_unattended_mode,
    boundary_requests,
    parse_args,
    parse_kv_group_snapshots,
    remote_process_probe,
    summarize_kv_snapshot,
    validate_acceptance_thresholds,
    validate_session,
    vllm_args,
    vllm_command,
)


def _state(*, nodes: int, gpu_variant: str = "GB200", priority: Priority = Priority.INTERACTIVE) -> DevGpuState:
    return DevGpuState(
        session_name="unit",
        config_file="/tmp/config.yaml",
        job_id="unit-job",
        gpus_per_node=LOCAL_DP_SIZE,
        gpu_variant=gpu_variant,
        priority=priority,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/tmp/kubeconfig"),
        pods=[PodRef(namespace="iris", pod_name=f"pod-{index}") for index in range(nodes)],
    )


def test_reference_ep8_requires_two_interactive_gb200_nodes() -> None:
    validate_session(_state(nodes=2), CASES["reference-ep8"])
    with pytest.raises(ValueError, match="requires 2 nodes"):
        validate_session(_state(nodes=1), CASES["reference-ep8"])
    with pytest.raises(ValueError, match="requires GB200"):
        validate_session(_state(nodes=2, gpu_variant="H100"), CASES["reference-ep8"])
    with pytest.raises(ValueError, match="interactive"):
        validate_session(_state(nodes=2, priority=Priority.BATCH), CASES["reference-ep8"])


def test_two_node_commands_use_one_global_ep8_group() -> None:
    case = CASES["reference-ep8"]
    leader = vllm_args(case, model_dir="/model", model_source="dummy", leader_ip="10.0.0.1", node_index=0, smoke=False)
    follower = vllm_args(case, model_dir="/model", model_source="dummy", leader_ip="10.0.0.1", node_index=1, smoke=False)
    joined_leader = " ".join(leader)
    joined_follower = " ".join(follower)
    for joined in (joined_leader, joined_follower):
        assert "--pipeline-parallel-size 1" in joined
        assert "--tensor-parallel-size 1" in joined
        assert "--data-parallel-size 8" in joined
        assert "--data-parallel-size-local 4" in joined
        assert "--enable-expert-parallel" in joined
        assert "--data-parallel-address 10.0.0.1" in joined
        assert "--load-format dummy" in joined
        assert "--enable-prefix-caching" in joined
        assert "--enable-return-routed-experts" in joined
        assert "--enable-prompt-tokens-details" in joined
        assert "--max-logprobs 64" in joined
    assert "--data-parallel-start-rank 0" in joined_leader
    assert "--data-parallel-start-rank 4" in joined_follower
    assert "--headless" not in leader
    assert "--api-server-count 1" in joined_leader
    assert "--headless" in follower
    assert "--api-server-count" not in joined_follower


def test_smoke_bounds_context_without_changing_model_config() -> None:
    case = CASES["reference-ep8"]
    args = vllm_args(case, model_dir="/model", model_source="dummy", leader_ip="10.0.0.1", node_index=0, smoke=True)
    joined = " ".join(args)
    assert "--max-model-len 2048" in joined
    assert case.max_model_len == 65_536


def test_four_node_case_has_contiguous_rank_starts() -> None:
    case = CASES["granular-ep16"]
    rank_starts = []
    for node_index in range(4):
        args = vllm_args(
            case,
            model_dir="/model",
            model_source="dummy",
            leader_ip="10.0.0.1",
            node_index=node_index,
            smoke=False,
        )
        start_index = args.index("--data-parallel-start-rank")
        rank_starts.append(int(args[start_index + 1]))
    assert rank_starts == [0, 4, 8, 12]


def test_command_pins_vllm_git_sha_and_cuda_backend() -> None:
    command = vllm_command(["serve", "/model"])
    joined = " ".join(command)
    assert VLLM_SHA in joined
    assert "--torch-backend cu130" in joined
    assert "runai-model-streamer[s3]==0.16.1" in joined


def test_validate_session_does_not_mutate_state() -> None:
    state = _state(nodes=1)
    original = dataclasses.asdict(state)
    validate_session(state, CASES["one-node-ep4"])
    assert dataclasses.asdict(state) == original


def test_snowball_uses_pinned_export_and_streaming_loader() -> None:
    args = vllm_args(
        CASES["reference-ep8"],
        model_dir=SNOWBALL_EXPORT,
        model_source="snowball",
        leader_ip="10.0.0.1",
        node_index=0,
        smoke=True,
    )
    joined = " ".join(args)
    assert SNOWBALL_EXPORT in joined
    assert "--load-format runai_streamer" in joined
    assert "--skip-tokenizer-init" not in joined


def test_frozen_fixture_uses_half_safetensors_and_every_custom_path() -> None:
    fixture_config = json.loads((Path(FROZEN_FIXTURE_PATH) / "config.json").read_text())
    assert fixture_config["head_dim"] % 16 == 0
    args = vllm_args(
        CASES["tiny"],
        model_dir="/fixture",
        model_source="fixture",
        leader_ip="10.0.0.1",
        node_index=0,
        smoke=True,
    )
    joined = " ".join(args)
    assert "--dtype half" in joined
    assert "--kv-cache-dtype auto" in joined
    assert "--load-format safetensors" in joined
    assert "--max-model-len 1024" in joined
    assert "--max-logprobs 64" in joined


def test_correctness_selects_both_required_boundaries_once() -> None:
    selected = boundary_requests(deterministic_boundary_workload())
    assert [request["prefix_token_count"] for request in selected] == [17, 513]


def test_completion_evidence_keeps_routes_out_of_compact_result(tmp_path) -> None:
    routes = np.array([[[1, 3, 5, 7]]], dtype=np.uint8)
    encoded = io.BytesIO()
    np.save(encoded, routes, allow_pickle=False)
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "logprobs": {"token_logprobs": [-1.25]},
                "routed_experts": base64.b64encode(encoded.getvalue()).decode(),
                "token_ids": [11],
            }
        ],
        "usage": {"completion_tokens": 1, "prompt_tokens": 17},
    }

    summary = _record_completion(payload, artifact_dir=tmp_path, stem="boundary-cold")

    assert summary["routed_experts_shape"] == [1, 1, 4]
    assert payload["choices"][0]["routed_experts"] not in json.dumps(summary)
    np.testing.assert_array_equal(
        np.load(tmp_path / summary["routed_experts_path"], allow_pickle=False),
        routes,
    )
    assert json.loads((tmp_path / summary["response_path"]).read_text()) == payload


def test_remote_process_probe_rejects_zombies() -> None:
    probe = remote_process_probe("/tmp/run.pid")
    assert 'test "$state" != Z' in probe
    assert 'test "$state" != X' in probe
    assert 'kill -0 "$pid"' in probe


def test_free_port_returns_a_boundable_port() -> None:
    assert 0 < _free_port() < 65_536


def test_acceptance_thresholds_cannot_be_weakened() -> None:
    validate_acceptance_thresholds(minimum_seconds=600, minimum_generated_tokens=250_000)
    with pytest.raises(ValueError, match="600 seconds"):
        validate_acceptance_thresholds(minimum_seconds=599, minimum_generated_tokens=250_000)
    with pytest.raises(ValueError, match="250000 generated"):
        validate_acceptance_thresholds(minimum_seconds=600, minimum_generated_tokens=249_999)


def test_unattended_cli_forbids_a_second_snowball_attempt_and_wrong_exact_modes() -> None:
    with pytest.raises(ValueError, match="must not be retried"):
        _validate_unattended_mode(
            CASES["reference-ep8"],
            mode="smoke",
            model_source="snowball",
        )
    with pytest.raises(ValueError, match="exact-reference-ep16"):
        _validate_unattended_mode(
            CASES["reference-ep8"],
            mode="acceptance",
            model_source="dummy",
        )
    with pytest.raises(ValueError, match="reference-ep8"):
        _validate_unattended_mode(
            CASES["one-node-ep4"],
            mode="kv",
            model_source="dummy",
        )
    parsed = parse_args(
        [
            "submit",
            "--case",
            "reference-ep8",
            "--mode",
            "kv",
            "--task-image",
            "example.invalid/task@sha256:" + "a" * 64,
        ]
    )
    assert parsed.command == "submit"
    assert not parsed.wait
    assert _immutable_image(parsed.task_image) == parsed.task_image


def test_unattended_worker_runs_as_a_repo_module() -> None:
    image = "example.invalid/task@sha256:" + "a" * 64
    args = parse_args(
        [
            "submit",
            "--case",
            "tiny",
            "--model-source",
            "fixture",
            "--task-image",
            image,
        ]
    )
    command = _unattended_worker_argv(
        args,
        case=CASES["tiny"],
        run_id="unit",
        image=image,
        marin_commit="b" * 40,
    )
    assert command[:4] == [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "worker",
    ]
    assert command[command.index("--marin-commit") + 1] == "b" * 40


def test_kv_snapshot_separates_semantic_padded_physical_and_reserved_bytes() -> None:
    groups = [
        {
            "engine_idx": 0,
            "role": "attention",
            "kind": "sliding_window",
            "active_requests": 1,
            "active_blocks": 33,
            "active_payload_bytes": 100,
            "active_padded_bytes": 120,
            "active_physical_bytes": 500,
            "reserved_physical_bytes": 1_000,
        },
        {
            "engine_idx": 0,
            "role": "attention",
            "kind": "full_attention",
            "active_requests": 1,
            "active_blocks": 385,
            "active_payload_bytes": 200,
            "active_padded_bytes": 240,
            "active_physical_bytes": 600,
            "reserved_physical_bytes": 1_000,
        },
        {
            "engine_idx": 1,
            "role": "attention",
            "kind": "full_attention",
            "active_requests": 0,
            "active_blocks": 0,
            "active_payload_bytes": 0,
            "active_padded_bytes": 0,
            "active_physical_bytes": 0,
            "reserved_physical_bytes": 1_000,
        },
        {
            "engine_idx": 0,
            "role": "sconv",
            "kind": "sconv",
            "active_requests": 1,
            "active_blocks": 34,
            "active_payload_bytes": 50,
            "active_padded_bytes": 60,
            "active_physical_bytes": 550,
            "reserved_physical_bytes": 1_000,
        },
    ]
    text = "INFO GrugMoE KV group usage: " + json.dumps(groups)
    (parsed,) = parse_kv_group_snapshots(text)
    summary = summarize_kv_snapshot(parsed)
    assert summary["active_requests"] == 1
    assert summary["semantic_active_bytes"] == 350
    assert summary["semantic_attention_active_bytes"] == 300
    assert summary["semantic_sconv_active_bytes"] == 50
    assert summary["padded_group_active_bytes"] == 420
    assert summary["padded_attention_active_bytes"] == 360
    assert summary["padded_sconv_active_bytes"] == 60
    assert summary["padding_active_bytes"] == 70
    assert summary["physical_active_bytes"] == 1_650
    assert summary["reserved_physical_bytes_global"] == 2_000
    assert summary["local_attention_active_blocks"] == 33
    assert summary["global_attention_active_blocks"] == 385
    assert summary["sconv_active_blocks"] == 34


def test_acceptance_components_require_ten_minutes_tokens_and_all_branches() -> None:
    arm = {
        "elapsed_seconds": 601,
        "stable_minutes_passed": True,
        "stable_full_minutes": 10,
        "generated_tokens": 250_000,
        "branch_coverage": {"passed": True, "observed": 144},
    }
    components = _acceptance_components(
        {
            "arms": [arm, arm],
            "repeatability": {"passed": True},
        }
    )
    assert all(component["passed"] for component in components.values())
    failed = _acceptance_components(
        {
            "arms": [
                {**arm, "stable_minutes_passed": False},
                {**arm, "generated_tokens": 249_999},
            ],
            "repeatability": {"passed": False},
        }
    )
    assert not failed["duration"]["passed"]
    assert not failed["token_count"]["passed"]
    assert not failed["repeatability"]["passed"]
