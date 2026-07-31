# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import scripts.iris.grugmoe_inference_preflight as grug_preflight
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
    _attended_result_passed,
    _free_port,
    _immutable_image,
    _record_completion,
    _unattended_placement_component,
    _unattended_worker_argv,
    _validate_submitted_coscheduling,
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
    assert command[:2] == ["uvx", "--no-config"]
    assert VLLM_SHA in joined
    assert "--torch-backend cu130" in joined
    assert "runai-model-streamer[s3]==0.16.1" in joined


def test_fixture_parity_allows_prereleases_and_preserves_resolver_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["uv", "run", "--no-config", "--prerelease=allow"]
        assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8000"
        assert command[command.index("--model") + 1] == "/fixture"
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["VLLM_TARGET_DEVICE"] == "cuda"
        assert environment["UV_CACHE_DIR"] == str(tmp_path.with_name(f"{tmp_path.name}-cuda-uv-cache"))
        raise subprocess.CalledProcessError(
            1,
            command,
            output="resolver stdout\n",
            stderr="resolver stderr\n",
        )

    monkeypatch.setattr(grug_preflight, "_run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        grug_preflight.run_fixture_parity("http://127.0.0.1:8000", "/fixture", artifact_dir=tmp_path)

    assert (tmp_path / "fixture-tensor-parity.stdout").read_text() == "resolver stdout\n"
    assert (tmp_path / "fixture-tensor-parity.stderr").read_text() == "resolver stderr\n"


def test_fixture_server_parity_runs_inside_isolated_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tensor = {"passed": True}
    server = {"passed": True, "boundary": {"reused_prompt_tokens": 512}}

    def succeed(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"tensor": tensor, "server": server}))
        return subprocess.CompletedProcess(command, 0, stdout="parity stdout\n", stderr="")

    monkeypatch.setattr(grug_preflight, "_run", succeed)

    result = grug_preflight.run_fixture_parity(
        "http://127.0.0.1:8000",
        "/fixture",
        artifact_dir=tmp_path,
    )

    assert result["passed"]
    assert result["tensor"] == tensor
    assert result["server"] == server
    assert json.loads((tmp_path / "fixture-server-parity.json").read_text()) == server


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
    selected = boundary_requests(
        deterministic_boundary_workload(CASES["one-node-ep4"]),
        expected_cache_hit_alignment=32,
    )
    assert [request["prefix_token_count"] for request in selected] == [33, 513]


def test_completion_can_pin_a_data_parallel_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        ok = True

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{}]}

    def post(url: str, **kwargs: object) -> Response:
        observed.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(grug_preflight.requests, "post", post)

    grug_preflight._completion(
        "http://server",
        "model",
        [1, 2, 3],
        data_parallel_rank=2,
    )

    assert observed["headers"] == {"X-data-parallel-rank": "2"}


def test_prefix_metric_wait_observes_delayed_request_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = iter(
        [
            ("first", {"vllm:prefix_cache_queries_total": 10.0}),
            ("second", {"vllm:prefix_cache_queries_total": 28.0}),
        ]
    )
    monkeypatch.setattr(grug_preflight, "_metrics", lambda _: next(snapshots))

    text, metrics, evidence = grug_preflight._wait_for_metric_delta(
        "http://server",
        {"vllm:prefix_cache_queries_total": 10.0},
        "vllm:prefix_cache_queries",
        minimum_delta=18,
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert text == "second"
    assert metrics["vllm:prefix_cache_queries_total"] == 28
    assert evidence["synchronized"]
    assert evidence["observed_delta"] == 18
    assert evidence["poll_attempts"] == 2


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
        coscheduling=None,
    )
    assert command[:4] == [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "worker",
    ]
    assert command[command.index("--marin-commit") + 1] == "b" * 40
    assert "--submitted-coscheduling" not in command


def test_unattended_worker_records_submitted_coscheduling() -> None:
    image = "example.invalid/task@sha256:" + "a" * 64
    args = parse_args(
        [
            "submit",
            "--case",
            "reference-ep8",
            "--task-image",
            image,
        ]
    )
    submitted = grug_preflight.CoschedulingConfig(group_by="nvlink.domain")

    command = _unattended_worker_argv(
        args,
        case=CASES["reference-ep8"],
        run_id="unit",
        image=image,
        marin_commit="b" * 40,
        coscheduling=submitted,
    )

    assert command[command.index("--submitted-coscheduling") + 1] == submitted.group_by


def test_unattended_worker_rejects_missing_submitted_coscheduling() -> None:
    _validate_submitted_coscheduling(expected_tasks=2, submitted="nvlink.domain")
    _validate_submitted_coscheduling(expected_tasks=1, submitted=None)

    with pytest.raises(RuntimeError, match=r"expected 'nvlink\.domain'"):
        _validate_submitted_coscheduling(expected_tasks=2, submitted=None)


def test_unattended_placement_uses_host_network_node_identity_without_worker_ids() -> None:
    rendezvous = [
        {
            "task_index": str(index),
            "num_tasks": "2",
            "advertise_host": host,
            "url": f"tcp://{host}:13345",
        }
        for index, host in enumerate(("10.0.0.1", "10.0.0.2"))
    ]
    rank_records = [{"rank": index, "coscheduling": "nvlink.domain"} for index in range(2)]

    placement = _unattended_placement_component(
        rendezvous,
        rank_records,
        expected_tasks=2,
    )

    assert placement["passed"]
    assert placement["distinct_advertise_hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert placement["distinct_worker_ids"] == []
    assert placement["topology_enforcement"] == "Kueue hard podset-required-topology"


@pytest.mark.parametrize(
    ("rendezvous", "rank_records"),
    [
        (
            [
                {"task_index": "0", "advertise_host": "10.0.0.1"},
                {"task_index": "1", "advertise_host": "10.0.0.1"},
            ],
            [
                {"coscheduling": "nvlink.domain"},
                {"coscheduling": "nvlink.domain"},
            ],
        ),
        (
            [
                {"task_index": "0", "advertise_host": "10.0.0.1"},
                {"task_index": "1", "advertise_host": "10.0.0.2"},
            ],
            [
                {"coscheduling": "nvlink.domain"},
                {"coscheduling": None},
            ],
        ),
    ],
)
def test_unattended_placement_rejects_shared_node_or_missing_topology(
    rendezvous: list[dict[str, str]],
    rank_records: list[dict[str, object]],
) -> None:
    placement = _unattended_placement_component(
        rendezvous,
        rank_records,
        expected_tasks=2,
    )

    assert not placement["passed"]


@pytest.mark.parametrize(
    ("mode", "correctness", "load", "expected"),
    [
        ("smoke", {"passed": True}, None, True),
        ("smoke", {"passed": False}, None, False),
        ("acceptance", {"passed": True}, {"passed": True}, True),
        ("acceptance", {"passed": True}, {"passed": False}, False),
        ("acceptance", {"passed": False}, {"passed": True}, False),
    ],
)
def test_attended_result_is_the_component_conjunction(
    mode: str,
    correctness: dict[str, bool],
    load: dict[str, bool] | None,
    expected: bool,
) -> None:
    assert _attended_result_passed(mode, correctness=correctness, load=load) is expected


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


def test_load_arm_samples_live_counter_before_slow_request_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request spanning minute boundaries must not create false zero buckets."""
    sample_seconds = 0.01
    metric_started = time.monotonic()

    def metrics(_: str) -> tuple[str, dict[str, float]]:
        # Scale continuously generated tokens so every 10 ms test interval is
        # analogous to one live minute in the acceptance run.
        generated = int((time.monotonic() - metric_started) * 1_000_000)
        return "", {"vllm:generation_tokens_total": float(generated)}

    def completion(*_: object, **__: object) -> dict[str, object]:
        time.sleep(0.025)
        return {"usage": {"completion_tokens": 1}}

    monkeypatch.setattr(grug_preflight, "_metrics", metrics)
    monkeypatch.setattr(grug_preflight, "_completion", completion)
    workload = {
        "roots": [{"root": 0, "prefix_token_ids": [1]}],
        "requests": [
            {
                "request_id": "slow",
                "root": 0,
                "append_token_ids": [],
                "prefix_token_count": 1,
                "max_tokens": 1,
                "final_token_count": 2,
            }
        ],
    }

    arm = grug_preflight._run_load_arm(
        "http://server",
        "model",
        workload,
        max_model_len=2,
        minimum_seconds=0.12,
        minimum_generated_tokens=1,
        concurrency=1,
        counter_sample_seconds=sample_seconds,
    )

    assert arm["latency_seconds"]["min"] > sample_seconds
    assert arm["stable_full_minutes"] == 10
    assert arm["stable_minutes_passed"]
    assert all(tokens > 0 for tokens in arm["last_ten_stable_minute_generated_tokens"])
    assert arm["generation_counter"]["stable_window_seconds"] == pytest.approx(
        10 * sample_seconds,
        abs=sample_seconds,
    )
