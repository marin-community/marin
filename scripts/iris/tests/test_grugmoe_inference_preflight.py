# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import io
import json
import subprocess
import threading
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
from experiments.grug.moe.rolling_benchmark import (
    PlateauRequirements,
    PlateauWindow,
    frozen_cohort_slots,
    histogram_quantile_delta,
    parse_labeled_prometheus,
    prometheus_value,
    prometheus_values_by_label,
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


def test_health_server_knobs_toggle_r3_and_batch_budget() -> None:
    enabled = vllm_args(
        CASES["exact-reference-ep16"],
        model_dir="/model",
        model_source="dummy",
        leader_ip="10.0.0.1",
        node_index=0,
        smoke=False,
        r3_enabled=True,
        max_num_batched_tokens=4096,
    )
    disabled = vllm_args(
        CASES["exact-reference-ep16"],
        model_dir="/model",
        model_source="dummy",
        leader_ip="10.0.0.1",
        node_index=0,
        smoke=False,
        r3_enabled=False,
        max_num_batched_tokens=8192,
    )

    assert "--enable-return-routed-experts" in enabled
    assert "--enable-return-routed-experts" not in disabled
    assert enabled[enabled.index("--max-num-batched-tokens") + 1] == "4096"
    assert disabled[disabled.index("--max-num-batched-tokens") + 1] == "8192"


def test_health_server_enables_control_routes_and_aggregated_engine_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pass

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(grug_preflight, "GLOO_CONTROL_INTERFACE", "lo")
    monkeypatch.setattr(grug_preflight, "get_job_info", lambda: type("Info", (), {"advertise_host": "10.0.0.1"})())
    monkeypatch.setattr(grug_preflight.subprocess, "Popen", fake_popen)
    server = grug_preflight._start_local_vllm(
        case=CASES["exact-reference-ep16"],
        model_source="dummy",
        model_dir="/model",
        leader_ip="10.0.0.1",
        node_index=0,
        smoke=False,
        local_dir=tmp_path,
        enable_dev_endpoints=True,
        aggregate_engine_logging=True,
    )
    server.log_stream.close()

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["VLLM_SERVER_DEV_MODE"] == "1"
    assert server.provenance_environment == {"VLLM_SERVER_DEV_MODE": "1"}
    assert server.command.count("--aggregate-engine-logging") == 1


def test_health_cli_freezes_worker_settings_and_three_way_concurrency() -> None:
    image = "example.invalid/task@sha256:" + "a" * 64
    args = parse_args(
        [
            "submit",
            "--case",
            "exact-reference-ep16",
            "--mode",
            "health",
            "--task-image",
            image,
            "--r3",
            "off",
            "--max-num-batched-tokens",
            "4096",
            "--max-num-seqs",
            "96",
            "--concurrency",
            "48",
            "--concurrency",
            "72",
            "--minimum-seconds",
            "120",
        ]
    )
    command = _unattended_worker_argv(
        args,
        case=CASES["exact-reference-ep16"],
        run_id="unit",
        image=image,
        marin_commit="b" * 40,
        coscheduling=grug_preflight.CoschedulingConfig(group_by="nvlink.domain"),
    )

    assert command[command.index("--r3") + 1] == "off"
    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert command[command.index("--max-num-seqs") + 1] == "96"
    assert [command[index + 1] for index, value in enumerate(command) if value == "--concurrency"] == ["48", "72"]


def test_health_manifest_freezes_representative_disjoint_warmup_inputs() -> None:
    cohorts = ("short", "medium", "long")
    workload = {
        "schema_version": 2,
        "kind": "unit",
        "seed": 7,
        "branches_per_root": 8,
        "history_lengths": [3, 4, 5],
        "append_tokens": 2,
        "response_tokens": 2,
        "final_lengths": [7, 8, 9],
        "roots": [
            {"root": index, "cohort": cohort, "prefix_token_ids": [250] * (index + 3)}
            for index, cohort in enumerate(cohorts)
        ],
        "requests": [
            {
                "request_id": f"request-{index}-{branch}",
                "root": index,
                "branch": branch,
                "cohort": cohort,
                "prefix_token_count": index + 3,
                "append_token_count": 2,
                "append_token_ids": [index + 3, branch + 10],
                "max_tokens": 2,
                "final_token_count": index + 7,
            }
            for index, cohort in enumerate(cohorts)
            for branch in range(8)
        ],
    }

    manifest = grug_preflight._health_workload_manifest(workload, concurrencies=[24])

    warm_up = manifest["warm_up"]
    roots = warm_up["root_copies"]
    records = warm_up["by_concurrency"]["24"]
    assert warm_up["rolling_passes"] == grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert len(roots) == 3 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert len(records) == 24 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert {record["pass"] for record in records} == set(range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES))
    assert {record["slot_id"] for record in records} == set(range(24))
    assert {record["data_parallel_rank"] for record in records} == {0, 1, 2}
    assert {record["max_tokens"] for record in records} == {2}
    assert {record["token_count"] for record in records} == {5, 6, 7}
    assert all(record["disjoint_from_measured_roots"] for record in records)
    assert all(record["shared_root_graph"] for record in records)
    assert len({record["token_ids_sha256"] for record in records}) == len(records)
    assert len({record["token_ids_sha256"] for record in roots}) == len(roots)
    assert all(record["disjoint_from_measured_roots"] for record in roots)
    assert all(record["disjoint_from_other_passes"] for record in roots)
    for pass_index in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES):
        pass_records = [record for record in records if record["pass"] == pass_index]
        sharing = {root: sum(record["root"] == root for record in pass_records) for root in range(3)}
        assert sharing == {0: 8, 1: 8, 2: 8}
        warm_workload = grug_preflight._health_warm_workload(workload, pass_index=pass_index)
        requests_by_id = {request["request_id"]: request for request in workload["requests"]}
        assert all(
            record["token_ids_sha256"]
            == grug_preflight._sha256_json(
                grug_preflight.materialize_prompt(
                    warm_workload,
                    requests_by_id[record["manifest_request_id"]],
                )
            )
            for record in pass_records
        )


def test_health_warmup_populates_shared_roots_and_barriers_each_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cohorts = ("short", "medium", "long")
    workload = {
        "roots": [
            {"root": root, "cohort": cohort, "prefix_token_ids": [1, 20 + root]} for root, cohort in enumerate(cohorts)
        ],
        "requests": [
            {
                "request_id": f"request-{root}",
                "root": root,
                "branch": 0,
                "cohort": cohort,
                "append_token_ids": [40 + root],
                "max_tokens": 2,
            }
            for root, cohort in enumerate(cohorts)
        ],
    }
    barriers = [threading.Barrier(3) for _ in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES)]
    events: list[str] = []
    lock = threading.Lock()

    def completion(*_: object, request_id: str, data_parallel_rank: int, **__: object) -> dict[str, object]:
        with lock:
            events.append(f"start:{request_id}")
        if "-slot-" in request_id:
            pass_index = next(
                index
                for index in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES)
                if f"-warm-pass-{index:02d}-slot-" in request_id
            )
            barriers[pass_index].wait(timeout=2)
        with lock:
            events.append(f"finish:{request_id}")
        return {"data_parallel_rank": data_parallel_rank, "route_array": None}

    monkeypatch.setattr(grug_preflight, "_health_completion", completion)
    branches, roots = grug_preflight._health_warm_up(
        "http://server",
        "model",
        workload,
        case=CASES["exact-reference-ep16"],
        r3_enabled=False,
        arm_id="arm",
        target_concurrency=3,
    )

    passes = grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert len(branches) == 3 * passes
    assert len(roots) == 3 * passes
    for pass_index in range(passes):
        root_marker = f"-warm-pass-{pass_index:02d}-populate-root-"
        branch_marker = f"-warm-pass-{pass_index:02d}-slot-"
        root_finishes = [
            index for index, event in enumerate(events) if event.startswith("finish:") and root_marker in event
        ]
        branch_starts = [
            index for index, event in enumerate(events) if event.startswith("start:") and branch_marker in event
        ]
        branch_finishes = [
            index for index, event in enumerate(events) if event.startswith("finish:") and branch_marker in event
        ]
        assert len(root_finishes) == len(branch_starts) == len(branch_finishes) == 3
        assert max(root_finishes) < min(branch_starts)
        if pass_index + 1 < passes:
            next_root_marker = f"-warm-pass-{pass_index + 1:02d}-populate-root-"
            next_root_starts = [
                index for index, event in enumerate(events) if event.startswith("start:") and next_root_marker in event
            ]
            assert max(branch_finishes) < min(next_root_starts)


def test_health_repeatability_gates_only_identical_settings() -> None:
    def arm(arm_id: str, concurrency: int, rate: float) -> dict[str, object]:
        return {
            "arm_id": arm_id,
            "settings": {
                "r3_enabled": True,
                "target_concurrency": concurrency,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 48,
            },
            "headline": {"generation_tokens_per_second_per_gpu": rate},
        }

    passing = grug_preflight._health_repeatability([arm("a", 48, 100), arm("b", 48, 101)])
    failing = grug_preflight._health_repeatability([arm("a", 48, 100), arm("b", 48, 103)])
    calibration = grug_preflight._health_repeatability([arm("a", 48, 100), arm("b", 72, 103)])

    assert passing == {
        "applicable": True,
        "comparisons": [
            {
                "left_arm": "a",
                "right_arm": "b",
                "left_generation_tokens_per_second_per_gpu": 100.0,
                "right_generation_tokens_per_second_per_gpu": 101.0,
                "delta_percent": pytest.approx(0.9950248756),
                "limit_percent": 2.0,
                "passed": True,
            }
        ],
        "passed": True,
    }
    assert not failing["passed"]
    assert calibration == {"applicable": False, "comparisons": [], "passed": True}


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


def test_timed_completion_freezes_seed_identity_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        ok = True
        content = b'{"choices":[{}]}'

        def __init__(self) -> None:
            self.headers = {"content-length": "16", "content-encoding": "identity"}

    def post(url: str, **kwargs: object) -> Response:
        observed.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(grug_preflight.requests, "post", post)

    _, timing = grug_preflight._timed_completion(
        "http://server",
        "model",
        [1, 2, 3],
        data_parallel_rank=7,
        sampling_seed=123,
        request_id="slot-7-attempt-9",
    )

    assert observed["headers"] == {
        "X-data-parallel-rank": "7",
        "X-Request-Id": "slot-7-attempt-9",
    }
    body = observed["json"]
    assert isinstance(body, dict)
    assert body["seed"] == 123
    assert observed["stream"] is True
    assert timing["response_bytes"] == len(Response.content)
    assert timing["client_e2e_seconds"] >= timing["seconds_to_response_body"]
    assert timing["response_body_transfer_seconds"] >= 0


def test_labeled_prometheus_preserves_engines_and_window_histograms() -> None:
    before = parse_labeled_prometheus(
        """
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0"} 100
vllm:generation_tokens_total{engine="1"} 200
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_running{engine="1"} 3
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{engine="0",le="2.0"} 4
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 4
"""
    )
    after = parse_labeled_prometheus(
        """
vllm:generation_tokens_total{engine="0"} 110
vllm:generation_tokens_total{engine="1"} 220
vllm:num_requests_running{engine="0"} 4
vllm:num_requests_running{engine="1"} 5
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0"} 3
vllm:time_to_first_token_seconds_bucket{engine="0",le="2.0"} 8
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 8
"""
    )

    assert prometheus_value(after, "vllm:generation_tokens") == 330
    assert prometheus_values_by_label(after, "vllm:num_requests_running", label="engine") == {
        "0": 4,
        "1": 5,
    }
    assert histogram_quantile_delta(before, after, "vllm:time_to_first_token_seconds", 0.5) == pytest.approx(4 / 3)


def test_r3_summary_checks_every_axis_and_cached_root_prefix() -> None:
    case = CASES["exact-reference-ep16"]
    root = np.zeros((2, case.num_hidden_layers, case.num_experts_per_tok), dtype=np.uint8)
    routes = np.concatenate(
        [root, np.ones((3, case.num_hidden_layers, case.num_experts_per_tok), dtype=np.uint8)], axis=0
    )
    encoded = io.BytesIO()
    np.save(encoded, routes, allow_pickle=False)
    payload = {"choices": [{"routed_experts": base64.b64encode(encoded.getvalue()).decode()}]}

    summary, retained = grug_preflight._health_route_summary(
        payload,
        case=case,
        expected_positions=5,
        r3_enabled=True,
        expected_prefix_routes=root,
        keep_routes=True,
    )

    assert summary["shape"] == [5, 48, 4]
    assert summary["root_prefix_positions_compared"] == 2
    assert summary["all_expected_positions_layers_topk_aligned"]
    assert sum(summary["expert_histogram"]) == routes.size
    np.testing.assert_array_equal(retained, routes)

    with pytest.raises(AssertionError, match="shape"):
        grug_preflight._health_route_summary(
            payload,
            case=case,
            expected_positions=6,
            r3_enabled=True,
        )


def test_fake_rolling_arm_replenishes_slots_and_excludes_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = CASES["exact-reference-ep16"]
    roots = [
        {"root": root, "cohort": ("short", "medium", "long")[root // 6], "prefix_token_ids": [250]} for root in range(18)
    ]
    requests = []
    for root in range(18):
        for branch in range(8):
            requests.append(
                {
                    "request_id": f"root-{root:02d}-branch-{branch:02d}",
                    "root": root,
                    "branch": branch,
                    "cohort": ("short", "medium", "long")[root // 6],
                    "prefix_token_count": 1,
                    "append_token_count": 0,
                    "append_token_ids": [],
                    "max_tokens": 1,
                    "final_token_count": 2,
                }
            )
    workload = {
        "history_lengths": [1, 1, 1],
        "roots": roots,
        "requests": requests,
    }
    lock = threading.Lock()
    completed = 0
    started = 0
    warm_requests = 0
    successor_started = threading.Event()
    opening_snapshot_taken = threading.Event()
    post_boundary_completion = threading.Event()

    def completion(*_: object, max_tokens: int, request_id: str, **kwargs: object) -> dict[str, object]:
        nonlocal completed, started, warm_requests
        with lock:
            if "-warm-pass-" in request_id:
                warm_requests += 1
            else:
                started += 1
                if started > 3:
                    successor_started.set()
            completed += 1
            if opening_snapshot_taken.is_set():
                post_boundary_completion.set()
        return {
            "request_id": request_id,
            "data_parallel_rank": kwargs["data_parallel_rank"],
            "sampling_seed": kwargs["sampling_seed"],
            "prompt_tokens": 1,
            "completion_tokens": max_tokens,
            "prompt_token_ids_sha256": "a" * 64,
            "generated_token_ids_sha256": "b" * 64,
            "final_prefix_token_ids_sha256": "c" * 64,
            "sampled_token_logprobs": {
                "count": max_tokens,
                "minimum": -1.0,
                "maximum": -1.0,
                "sha256": "d" * 64,
            },
            "timing": {
                "started_at_monotonic_seconds": time.monotonic() - 0.001,
                "completed_at_monotonic_seconds": time.monotonic(),
                "client_e2e_seconds": 0.001,
                "response_bytes": 10,
                "seconds_to_response_headers": 0.0004,
                "response_body_transfer_seconds": 0.0005,
                "seconds_to_decode": 0.0001,
            },
            "routes": {"enabled": False, "transport": "absent", "carrier_payload_bytes": 0},
            "route_array": None,
        }

    def capture(
        _: str,
        *,
        metrics_map: list[dict[str, object]],
        arm_id: str,
        phase: str,
        **__: object,
    ) -> grug_preflight.HealthMetricSnapshot:
        # Hold the opening scrape until the controller has replenished a slot,
        # then hold its response until another request finishes after the
        # snapshot. This proves both sides of the boundary without wall-clock
        # timing assumptions.
        if phase == "plateau-open":
            assert successor_started.wait(timeout=2)
        with lock:
            count = float(completed)
        captured_at = time.monotonic()
        if phase == "plateau-open":
            opening_snapshot_taken.set()
            assert post_boundary_completion.wait(timeout=2)
        index = len(metrics_map)
        totals = {metric: 0.0 for metric in grug_preflight.HEALTH_COUNTER_METRICS}
        totals["vllm:generation_tokens"] = count
        totals["vllm:prompt_tokens"] = count
        totals["vllm:request_success"] = count
        totals["vllm:prefix_cache_queries"] = count
        totals["vllm:prefix_cache_hits"] = count
        by_engine = {
            metric: {str(engine): 3.0 for engine in range(case.data_parallel_size)}
            for metric in grug_preflight.HEALTH_ENGINE_METRICS
        }
        samples = parse_labeled_prometheus(
            "\n".join(
                f'{metric}_bucket{{engine="0",le="{bound}"}} {count}'
                for metric in grug_preflight.HEALTH_HISTOGRAM_METRICS
                for bound in ("1.0", "+Inf")
            )
            + "\n"
        )
        metrics_map.append(
            {
                "index": index,
                "path": f"metrics/raw-{index:06d}.prom",
                "monotonic_seconds": captured_at,
                "leader_log_bytes": 0,
                "arm_id": arm_id,
                "phase": phase,
                "totals": totals,
                "by_engine": by_engine,
            }
        )
        return grug_preflight.HealthMetricSnapshot(
            index,
            f"metrics/raw-{index:06d}.prom",
            captured_at,
            samples,
            totals,
            by_engine,
            0,
        )

    monkeypatch.setattr(grug_preflight, "_reset_health_prefix_cache", lambda _: {"status_code": 200})
    monkeypatch.setattr(
        grug_preflight,
        "_populate_health_roots",
        lambda *args, **kwargs: ({root: None for root in range(18)}, [{} for _ in range(18)]),
    )
    monkeypatch.setattr(grug_preflight, "_capture_health_metrics", capture)
    monkeypatch.setattr(
        grug_preflight,
        "_wait_for_health_counter",
        lambda base_url, **kwargs: capture(
            base_url,
            artifact_dir=kwargs["artifact_dir"],
            metrics_map=kwargs["metrics_map"],
            arm_id=kwargs["arm_id"],
            phase="counter-sync",
        ),
    )
    monkeypatch.setattr(grug_preflight, "_health_completion", completion)
    monkeypatch.setattr(grug_preflight, "_health_kv_summary", lambda *args, **kwargs: {"passed": True})
    events = grug_preflight.HealthEventWriter(tmp_path / "events.jsonl")
    metrics_map: list[dict[str, object]] = []

    arm = grug_preflight._run_rolling_health_arm(
        "http://server",
        "model",
        workload,
        case=case,
        artifact_dir=tmp_path,
        metrics_map=metrics_map,
        events=events,
        log_path=tmp_path / "server.log",
        arm_id="fake",
        target_concurrency=3,
        minimum_seconds=0,
        minimum_generated_tokens=0,
        r3_enabled=False,
        max_num_batched_tokens=8,
        max_num_seqs=3,
        metric_sample_seconds=0.001,
    )
    events.close()

    assert warm_requests == 3 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert arm["warm_up"] == {
        "requests": warm_requests + 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "branch_requests": warm_requests,
        "root_population_requests": 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "root_copies": 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "rolling_passes": grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "target_concurrency": 3,
        "max_tokens_per_request": 1,
        "data_parallel_ranks_covered": list(range(16)),
        "shared_root_graph": True,
        "full_pass_barrier": True,
        "distinct_root_copy_per_pass": True,
        "disjoint_from_measured_roots": True,
        "excluded_from_measurement": True,
        "prefix_cache_reset_after": True,
    }
    assert successor_started.is_set()
    assert opening_snapshot_taken.is_set()
    assert post_boundary_completion.is_set()
    assert arm["passed"]
    assert arm["requests"]["plateau_successes"] >= 144
    assert arm["requests"]["excluded_before_valid_plateau_successes"] > 0
    assert arm["requests"]["drain_successes"] >= 3
    assert (
        arm["requests"]["whole_run_successes"]
        == arm["requests"]["excluded_before_valid_plateau_successes"]
        + arm["requests"]["plateau_successes"]
        + arm["requests"]["drain_successes"]
    )
    assert arm["drain"]["excluded_from_plateau"]
    assert arm["plateau"]["discarded_windows"] == []
    assert arm["whole_run_token_reconciliation"]["passed"]
    assert arm["requests"]["branch_coverage"] == {"expected": 144, "observed": 144, "passed": True}
    assert arm["metrics"]["rolling_start"] != arm["metrics"]["boundary_start"]
    boundary_start = next(
        float(entry["monotonic_seconds"]) for entry in metrics_map if entry["path"] == arm["metrics"]["boundary_start"]
    )
    boundary_end = next(
        float(entry["monotonic_seconds"]) for entry in metrics_map if entry["path"] == arm["metrics"]["boundary_end"]
    )
    event_records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    completion_events = [record for record in event_records if record["event"] == "request_completed"]
    assert (
        sum(
            boundary_start <= float(record["completed_at_monotonic_seconds"]) <= boundary_end
            for record in completion_events
        )
        == arm["requests"]["plateau_successes"]
    )


def test_independent_reader_separates_health_from_integrity_and_rejects_tampering(tmp_path: Path) -> None:
    run_id = "readback-unit"

    class MemoryFilesystem:
        def __init__(self) -> None:
            self.data: dict[str, bytes] = {}

        def open(self, key: str, _: str) -> object:
            filesystem = self

            class Sink:
                def __init__(self) -> None:
                    self.buffer = io.BytesIO()

                def __enter__(self) -> object:
                    return self

                def write(self, payload: bytes) -> int:
                    return self.buffer.write(payload)

                def __exit__(self, *_: object) -> None:
                    filesystem.data[key] = self.buffer.getvalue()

            return Sink()

        def cat_file(self, key: str) -> bytes:
            return self.data[key]

        def find(self, prefix: str) -> list[str]:
            return sorted(key for key in self.data if key.startswith(prefix))

    def raw_metrics(generation: int, prompt: int, successes: int, observations: int) -> str:
        lines = [
            f'vllm:generation_tokens_total{{engine="0"}} {generation}',
            f'vllm:prompt_tokens_total{{engine="0"}} {prompt}',
            f'vllm:request_success_total{{engine="0",finished_reason="length"}} {successes}',
            'vllm:num_preemptions_total{engine="0"} 0',
            f'vllm:prefix_cache_queries_total{{engine="0"}} {generation}',
            f'vllm:prefix_cache_hits_total{{engine="0"}} {generation}',
        ]
        for metric in grug_preflight.HEALTH_ENGINE_METRICS:
            lines.extend(f'{metric}{{engine="{engine}"}} 1' for engine in range(16))
        for metric in grug_preflight.HEALTH_HISTOGRAM_METRICS:
            lines.extend(
                [
                    f'{metric}_bucket{{engine="0",le="1.0"}} {observations}',
                    f'{metric}_bucket{{engine="0",le="+Inf"}} {observations}',
                ]
            )
        return "\n".join(lines) + "\n"

    raw_before = raw_metrics(100, 200, 0, 0)
    raw_after = raw_metrics(295_012, 500_200, 144, 144)
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "raw-000000.prom").write_text(raw_before)
    (metrics_dir / "raw-000001.prom").write_text(raw_after)
    metrics_map = []
    for index, text in enumerate((raw_before, raw_after)):
        parsed = parse_labeled_prometheus(text)
        metrics_map.append(
            {
                "index": index,
                "path": f"metrics/raw-{index:06d}.prom",
                "monotonic_seconds": 120.0 * index,
                "arm_id": "arm-00-c3",
                "phase": "plateau-open" if index == 0 else "plateau-close",
                "bytes": len(text.encode()),
                "sha256": grug_preflight.hashlib.sha256(text.encode()).hexdigest(),
                "totals": {
                    metric: grug_preflight.prometheus_value(parsed, metric)
                    for metric in grug_preflight.HEALTH_COUNTER_METRICS
                },
                "by_engine": {
                    metric: grug_preflight.prometheus_values_by_label(parsed, metric, label="engine")
                    for metric in grug_preflight.HEALTH_ENGINE_METRICS
                },
            }
        )
    kv_groups = [
        {
            "engine_idx": engine,
            "active_requests": 1 if engine < 3 else 0,
            "reserved_physical_bytes": 1_000,
            "active_physical_bytes": 100 if engine < 3 else 0,
            "active_payload_bytes": 80 if engine < 3 else 0,
            "active_padded_bytes": 100 if engine < 3 else 0,
            "role": "attention",
            "kind": "sliding_window",
            "active_blocks": 1 if engine < 3 else 0,
        }
        for engine in range(16)
    ]
    kv_text = f"test {grug_preflight.KV_LOG_MARKER}{json.dumps(kv_groups, separators=(',', ':'))}\n"
    kv_path = metrics_dir / "arm-00-c3-kv.log"
    kv_path.write_text(kv_text)
    server_latency = {
        metric: {"p50_seconds": 0.5, "p99_seconds": 0.99} for metric in grug_preflight.HEALTH_HISTOGRAM_METRICS
    }
    workload_manifest = grug_preflight._health_workload_manifest(
        grug_preflight.deterministic_workload(),
        concurrencies=[3],
    )
    final_prefix_provenance = [
        {
            "manifest_request_id": request["request_id"],
            "prompt_token_ids_sha256": request["prompt_token_ids_sha256"],
            "occurrences": 1,
            "outcomes": [
                {
                    "generated_token_ids_sha256": grug_preflight._sha256_json([request["request_id"], "generated"]),
                    "final_prefix_token_ids_sha256": grug_preflight._sha256_json([request["request_id"], "final"]),
                    "occurrences": 1,
                }
            ],
        }
        for request in workload_manifest["requests"]
    ]
    event_records = [
        {"event": event, "arm_id": "arm-00-c3"} for event in ("plateau_opened", "plateau_closed", "arm_completed")
    ]
    total_route_assignments = 0
    total_route_npy_bytes = 0
    total_route_base64_bytes = 0
    total_response_bytes = 0
    for index, entry in enumerate(final_prefix_provenance):
        request = workload_manifest["requests"][index]
        route_assignments = (int(request["final_token_count"]) - 1) * 48 * 4
        route_npy_bytes = route_assignments + 128
        route_base64_bytes = route_npy_bytes + 128
        response_bytes = route_base64_bytes + 256
        total_route_assignments += route_assignments
        total_route_npy_bytes += route_npy_bytes
        total_route_base64_bytes += route_base64_bytes
        total_response_bytes += response_bytes
        timing = {
            "started_at_monotonic_seconds": index,
            "completed_at_monotonic_seconds": (index + 1) * 120 / 145,
            "client_e2e_seconds": 1.0,
            "seconds_to_response_headers": 0.25,
            "response_body_transfer_seconds": 0.5,
            "seconds_to_decode": 0.25,
            "response_bytes": response_bytes,
        }
        event_records.append(
            {
                "event": "request_completed",
                "arm_id": "arm-00-c3",
                "manifest_request_id": entry["manifest_request_id"],
                "cohort": request["cohort"],
                "completion_tokens": 2_048,
                "completed_at_monotonic_seconds": timing["completed_at_monotonic_seconds"],
                "client_e2e_seconds": timing["client_e2e_seconds"],
                "response_bytes": response_bytes,
                "timing": timing,
                "prompt_token_ids_sha256": entry["prompt_token_ids_sha256"],
                "generated_token_ids_sha256": entry["outcomes"][0]["generated_token_ids_sha256"],
                "final_prefix_token_ids_sha256": entry["outcomes"][0]["final_prefix_token_ids_sha256"],
                "sampled_token_logprobs_count": 2_048,
                "sampled_token_logprobs_sha256": "d" * 64,
                "route_sha256": "e" * 64,
                "route_summary": {
                    "enabled": True,
                    "shape": [int(request["final_token_count"]) - 1, 48, 4],
                    "dtype": "uint8",
                    "minimum_expert": 0,
                    "maximum_expert": 0,
                    "all_expected_positions_layers_topk_aligned": True,
                    "root_prefix_positions_compared": int(request["prefix_token_count"]),
                    "root_prefix_aligned": True,
                    "expert_histogram": [route_assignments, *([0] * 127)],
                    "ep_rank_histogram": [route_assignments, *([0] * 15)],
                    "route_sha256": "e" * 64,
                    "carrier_array_bytes": route_assignments,
                    "carrier_npy_bytes": route_npy_bytes,
                    "carrier_base64_bytes": route_base64_bytes,
                    "transport": "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy",
                },
            },
        )
    event_records.append({"event": "worker_completed"})
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in event_records) + "\n")
    arm = {
        "arm_id": "arm-00-c3",
        "passed": True,
        "gates": {"all": True},
        "settings": {
            "r3_enabled": True,
            "target_concurrency": 3,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 3,
            "settings_drift": False,
        },
        "headline": {
            "generation_tokens_per_second_per_gpu": 294_912 / 120 / 16,
        },
        "plateau": {
            "elapsed_seconds": 120,
            "generated_tokens": 294_912,
            "processed_prompt_tokens": 500_000,
            "target_concurrency": 3,
            "minimum_required_in_flight": 3,
            "in_flight": {
                "samples": 2,
                "min": 3,
                "mean": 3.0,
                "max": 3,
                "at_close": 3,
            },
            "successful_requests": 144,
            "failed_requests": 0,
            "client_generated_tokens": 294_912,
            "cohort_completions": {"long": 48, "medium": 48, "short": 48},
            "manifest": {"expected": 144, "observed": 144, "passed": True},
        },
        "requests": {
            "branch_coverage": {"observed": 144, "passed": True},
            "whole_run_successes": 144,
            "engine_success_counter_delta": 144,
            "plateau_successes": 144,
            "excluded_before_valid_plateau_successes": 0,
            "drain_successes": 0,
            "final_prefix_provenance": final_prefix_provenance,
            "sampled_token_logprobs": {
                "validated_requests": 144,
                "validated_generated_tokens": 294_912,
                "all_completion_tokens_covered": True,
            },
        },
        "whole_run_token_reconciliation": {
            "client_generated_tokens": 294_912,
            "engine_generated_tokens": 294_912,
            "delta": 0,
            "passed": True,
        },
        "preemptions": 0,
        "moe_routing": {
            "r3_enabled": True,
            "expert_histogram": [total_route_assignments, *([0] * 127)],
            "ep_rank_histogram": [total_route_assignments, *([0] * 15)],
            "ep_rank_load": {
                "mean_assignments": total_route_assignments / 16,
                "max_assignments": total_route_assignments,
                "max_over_mean": 16.0,
            },
            "alignment_passed": True,
            "carrier": {
                "array_bytes": total_route_assignments,
                "npy_bytes": total_route_npy_bytes,
                "base64_bytes": total_route_base64_bytes,
                "full_response_bytes": total_response_bytes,
                "transport": "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy",
            },
        },
        "metrics": {
            "rolling_start": "metrics/raw-000000.prom",
            "boundary_start": "metrics/raw-000000.prom",
            "boundary_end": "metrics/raw-000001.prom",
            "final_after_drain": "metrics/raw-000001.prom",
            "counter_deltas": {
                "vllm:generation_tokens": 294_912,
                "vllm:prompt_tokens": 500_000,
                "vllm:prompt_tokens_cached": 0,
                "vllm:request_success": 144,
                "vllm:num_preemptions": 0,
                "vllm:prefix_cache_queries": 294_912,
                "vllm:prefix_cache_hits": 294_912,
            },
            "prefix_cache": {
                "query_tokens": 294_912,
                "hit_tokens": 294_912,
                "hit_ratio": 1.0,
            },
            "per_engine_plateau_series": grug_preflight._health_engine_series(
                metrics_map,
                first_index=0,
                last_index=1,
            ),
        },
        "latency": {
            "client_e2e_seconds": grug_preflight._health_percentiles([1.0] * 144),
            "client_e2e_seconds_by_cohort": {
                cohort: grug_preflight._health_percentiles([1.0] * 48) for cohort in ("short", "medium", "long")
            },
            "server_histogram_window": server_latency,
            "client_transport_window": {
                "seconds_to_response_headers": grug_preflight._health_percentiles([0.25] * 144),
                "response_body_transfer_seconds": grug_preflight._health_percentiles([0.5] * 144),
                "seconds_to_decode": grug_preflight._health_percentiles([0.25] * 144),
            },
        },
    }
    arm["kv_cache"] = grug_preflight._health_kv_summary_from_text(
        kv_text,
        case=CASES["exact-reference-ep16"],
        target_concurrency=3,
    )
    arm["kv_cache"]["source"] = {
        "path": "metrics/arm-00-c3-kv.log",
        "bytes": len(kv_text.encode()),
        "sha256": grug_preflight.hashlib.sha256(kv_text.encode()).hexdigest(),
        "boundary": "vLLM leader log bytes captured only during the accepted plateau",
    }
    placement = {
        "passed": True,
        "distinct_advertise_hosts": [f"10.0.0.{index}" for index in range(4)],
        "required_coscheduling": grug_preflight.UNATTENDED_COSCHEDULING,
        "topology_enforcement": "Kueue hard podset-required-topology",
    }
    image = "example.invalid/task@sha256:" + "a" * 64
    marin_commit = "b" * 40
    rank_commands = {
        str(rank): [
            *grug_preflight.vllm_command(
                grug_preflight.vllm_args(
                    CASES["exact-reference-ep16"],
                    model_dir="/tmp/model",
                    model_source="dummy",
                    leader_ip="10.0.0.0",
                    node_index=rank,
                    smoke=False,
                    r3_enabled=True,
                    max_num_batched_tokens=8192,
                    max_num_seqs=3,
                )
            ),
            "--aggregate-engine-logging",
        ]
        for rank in range(4)
    }
    ranks = [
        {
            "rank": rank,
            "gpu_inventory": [{"name": "NVIDIA GB200 NVL", "uuid": f"GPU-{rank}-{gpu}"} for gpu in range(4)],
            "vllm_command": rank_commands[str(rank)],
            "vllm_environment": dict(grug_preflight.VLLM_SERVER_DEV_MODE_ENVIRONMENT),
            "marin_commit": marin_commit,
            "vllm_commit": grug_preflight.VLLM_SHA,
            "task_image": image,
            "coscheduling": grug_preflight.UNATTENDED_COSCHEDULING,
        }
        for rank in range(4)
    ]
    result = {
        "run_id": run_id,
        "case": "exact-reference-ep16",
        "model": "exact-reference-ep16",
        "status": "passed",
        "passed": True,
        "arms": [arm],
        "repeatability": grug_preflight._health_repeatability([arm]),
        "placement": placement,
        "all_rank_health": {"passed": True, "ranks": ranks},
    }
    manifest = {
        "experiment": "experiment-0",
        "run_id": run_id,
        "protocol": {
            "minimum_plateau_seconds": 120,
            "minimum_plateau_engine_generation_tokens": 250_000,
            "minimum_in_flight_fraction": 0.95,
            "drain_excluded": True,
            "headline_counter": "vllm:generation_tokens",
        },
        "model_config": dataclasses.asdict(CASES["exact-reference-ep16"]),
        "model_fixture": {
            "source": "dummy",
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "seed": grug_preflight.DUMMY_SEED,
        },
        "server_settings": {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "data_parallel_size": 16,
            "expert_parallel_size": 16,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 3,
            "r3_enabled": True,
            "concurrencies": [3],
            "prefix_caching": True,
            "chunked_prefill": True,
            "cuda_graphs": True,
            "aggregate_engine_logging": True,
            "vllm_environment": dict(grug_preflight.VLLM_SERVER_DEV_MODE_ENVIRONMENT),
        },
        "workload": workload_manifest,
        "routing_fixture": {
            "kind": "canonical seeded vLLM dummy routing",
            "seed": grug_preflight.DUMMY_SEED,
            "expert_placement": "linear contiguous experts per EP rank",
            "capacity_factor": None,
            "balanced_control": {"applicable": False},
        },
        "implementation_controls": {
            "new_hot_path_family": False,
            "no_op_control": {"applicable": False},
        },
        "r3": {
            "enabled": True,
            "carrier": "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy",
            "expected_layers": 48,
            "expected_top_k": 4,
        },
        "final_prefix_provenance": {"arm-00-c3": final_prefix_provenance},
        "train_to_serve_parity": {"status": "inherited from reviewed exact-anchor preflight"},
        "placement": placement,
        "rank_commands": rank_commands,
        "provenance": {
            "marin_commit": marin_commit,
            "marin_commit_url": f"https://github.com/marin-community/marin/commit/{marin_commit}",
            "vllm_commit": grug_preflight.VLLM_SHA,
            "vllm_commit_url": f"https://github.com/marin-community/vllm/commit/{grug_preflight.VLLM_SHA}",
            "task_image": image,
            "dependency_lock_sha256": "c" * 64,
            "cluster_config": grug_preflight.DEFAULT_CLUSTER_CONFIG,
            "iris_task_count": 4,
            "iris_priority": "interactive",
            "iris_coscheduling": grug_preflight.UNATTENDED_COSCHEDULING,
            "iris_retry_policy": {
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
            },
        },
    }
    filesystem = MemoryFilesystem()
    grug_preflight._write_and_upload_health_artifacts(
        filesystem,
        artifact_dir=tmp_path,
        artifact_prefix=f"{grug_preflight.HEALTH_ARTIFACT_ROOT}/{run_id}/",
        result=result,
        manifest=manifest,
        metrics_map=metrics_map,
    )

    receipt = grug_preflight.readback_health_artifacts(filesystem, run_id=run_id)

    assert receipt["passed"], json.dumps(receipt["checks"], indent=2)
    assert receipt["benchmark_health"]["passed"]

    repeated_arm = json.loads(json.dumps(arm))
    repeated_arm_id = "arm-01-c3"
    repeated_arm["arm_id"] = repeated_arm_id
    repeated_arm["plateau"]["elapsed_seconds"] = 130
    repeated_arm["headline"]["generation_tokens_per_second_per_gpu"] = 294_912 / 130 / 16
    repeated_kv_path = metrics_dir / f"{repeated_arm_id}-kv.log"
    repeated_kv_path.write_text(kv_text)
    repeated_arm["kv_cache"]["source"]["path"] = f"metrics/{repeated_kv_path.name}"

    first_repeated_metric_index = len(metrics_map)
    repeated_metrics = []
    for offset, source in enumerate(metrics_map):
        repeated = json.loads(json.dumps(source))
        repeated["index"] = first_repeated_metric_index + offset
        repeated["path"] = f"metrics/raw-{repeated['index']:06d}.prom"
        repeated["monotonic_seconds"] = 200.0 + 130.0 * offset
        repeated["arm_id"] = repeated_arm_id
        (tmp_path / repeated["path"]).write_bytes((tmp_path / source["path"]).read_bytes())
        repeated_metrics.append(repeated)
    metrics_map.extend(repeated_metrics)
    repeated_arm["metrics"].update(
        {
            "rolling_start": repeated_metrics[0]["path"],
            "boundary_start": repeated_metrics[0]["path"],
            "boundary_end": repeated_metrics[1]["path"],
            "final_after_drain": repeated_metrics[1]["path"],
            "per_engine_plateau_series": grug_preflight._health_engine_series(
                metrics_map,
                first_index=first_repeated_metric_index,
                last_index=first_repeated_metric_index + 1,
            ),
        }
    )

    repeated_events = [
        {"event": event, "arm_id": repeated_arm_id} for event in ("plateau_opened", "plateau_closed", "arm_completed")
    ]
    for event in event_records:
        if event.get("event") != "request_completed":
            continue
        repeated = json.loads(json.dumps(event))
        repeated["arm_id"] = repeated_arm_id
        completed = 200.0 + float(event["completed_at_monotonic_seconds"]) * 130.0 / 120.0
        repeated["completed_at_monotonic_seconds"] = completed
        repeated["timing"]["started_at_monotonic_seconds"] += 200.0
        repeated["timing"]["completed_at_monotonic_seconds"] = completed
        repeated_events.append(repeated)
    event_records[-1:-1] = repeated_events
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(event) for event in event_records) + "\n")

    result["arms"].append(repeated_arm)
    result["repeatability"] = grug_preflight._health_repeatability(result["arms"])
    result["passed"] = False
    result["status"] = "failed"
    manifest["server_settings"]["concurrencies"].append(3)
    manifest["final_prefix_provenance"][repeated_arm_id] = final_prefix_provenance
    grug_preflight._write_and_upload_health_artifacts(
        filesystem,
        artifact_dir=tmp_path,
        artifact_prefix=f"{grug_preflight.HEALTH_ARTIFACT_ROOT}/{run_id}/",
        result=result,
        manifest=manifest,
        metrics_map=metrics_map,
    )

    failed_health = grug_preflight.readback_health_artifacts(filesystem, run_id=run_id)
    assert failed_health["passed"], json.dumps(failed_health["checks"], indent=2)
    assert not failed_health["benchmark_health"]["passed"]
    assert failed_health["benchmark_health"]["status"] == "failed"
    assert not failed_health["benchmark_health"]["repeatability"]["passed"]
    assert failed_health["checks"]["repeatability"]["passed"]
    assert not failed_health["checks"]["repeatability"]["benchmark_passed"]

    root_key = grug_preflight._s3_key(f"{grug_preflight.HEALTH_ARTIFACT_ROOT}/{run_id}")
    filesystem.data[f"{root_key}/metrics/raw-000001.prom"] += b"\n"
    tampered = grug_preflight.readback_health_artifacts(filesystem, run_id=run_id)
    assert not tampered["passed"]
    assert not tampered["checks"]["claimed_file_hashes"]["passed"]


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


def test_health_completion_requires_one_sampled_logprob_per_token(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "length",
                "logprobs": {"token_logprobs": [-1.25, -0.75]},
                "token_ids": [11, 12],
            }
        ],
        "usage": {"completion_tokens": 2, "prompt_tokens": 2},
    }
    monkeypatch.setattr(grug_preflight, "_timed_completion", lambda *args, **kwargs: (payload, {"elapsed": 1.0}))

    result = grug_preflight._health_completion(
        "http://server",
        "model",
        [1, 2],
        case=CASES["exact-reference-ep16"],
        max_tokens=2,
        data_parallel_rank=0,
        request_id="request-0",
        sampling_seed=7,
        r3_enabled=False,
    )

    assert result["sampled_token_logprobs"]["count"] == 2
    assert len(result["sampled_token_logprobs"]["sha256"]) == 64

    payload["choices"][0]["logprobs"]["token_logprobs"] = [-1.25]
    with pytest.raises(AssertionError, match="one finite sampled-token logprob"):
        grug_preflight._health_completion(
            "http://server",
            "model",
            [1, 2],
            case=CASES["exact-reference-ep16"],
            max_tokens=2,
            data_parallel_rank=0,
            request_id="request-1",
            sampling_seed=8,
            r3_enabled=False,
        )


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


@pytest.mark.parametrize(
    ("summary", "expected_exit_code"),
    [
        ({"terminal_job_succeeded": True}, 0),
        ({"terminal_job_succeeded": False}, 1),
        ({"terminal_job_state": 5}, 1),
    ],
)
def test_submit_wait_exit_code_requires_successful_terminal_iris_state(
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, object],
    expected_exit_code: int,
) -> None:
    """The waited CLI must not turn failed or incomplete Iris state into success."""
    monkeypatch.setattr(grug_preflight, "submit_unattended", lambda _: summary)

    exit_code = grug_preflight.main(
        [
            "submit",
            "--case",
            "exact-reference-ep16",
            "--task-image",
            "example.invalid/image@sha256:" + "a" * 64,
            "--wait",
        ]
    )

    assert exit_code == expected_exit_code


def test_frozen_cohort_slots_keep_equal_load_and_cover_each_manifest() -> None:
    requests = [
        {"request_id": f"{cohort}-{index}", "cohort": cohort}
        for cohort in ("short", "medium", "long")
        for index in range(6)
    ]

    slots = frozen_cohort_slots(requests, target_concurrency=6)
    observed = {cohort: set() for cohort in ("short", "medium", "long")}
    for _ in range(3):
        for slot in slots:
            request = slot.next_request()
            observed[slot.cohort].add(request["request_id"])

    assert {cohort: sum(slot.cohort == cohort for slot in slots) for cohort in observed} == {
        "short": 2,
        "medium": 2,
        "long": 2,
    }
    assert all(len(request_ids) == 6 for request_ids in observed.values())


def test_plateau_discards_a_load_dip_and_closes_only_after_all_floors() -> None:
    requirements = PlateauRequirements(
        target_concurrency=20,
        minimum_seconds=120,
        minimum_generated_tokens=250_000,
        required_request_ids=frozenset({"short", "medium", "long"}),
    )
    plateau = PlateauWindow(requirements)

    assert (
        plateau.observe_in_flight(
            now=10,
            in_flight=19,
            generation_counter=1_000,
            prompt_counter=2_000,
        )
        == "opened"
    )
    plateau.record_completion(request_id="short", cohort="short", completion_tokens=2_048, succeeded=True)
    assert plateau.observe_in_flight(now=20, in_flight=18) == "discarded"

    assert (
        plateau.observe_in_flight(
            now=30,
            in_flight=20,
            generation_counter=3_000,
            prompt_counter=4_000,
        )
        == "opened"
    )
    for request_id in ("short", "medium", "long"):
        plateau.record_completion(
            request_id=request_id,
            cohort=request_id,
            completion_tokens=2_048,
            succeeded=True,
        )

    assert not plateau.ready_to_close(now=150, in_flight=19, generation_counter=253_000)
    assert plateau.ready_to_close(now=150, in_flight=20, generation_counter=253_000)
    result = plateau.close(
        now=150,
        in_flight=20,
        generation_counter=253_000,
        prompt_counter=104_000,
    )

    assert result["elapsed_seconds"] == 120
    assert result["generated_tokens"] == 250_000
    assert result["in_flight"]["at_close"] == 20
    assert result["manifest"] == {"expected": 3, "observed": 3, "passed": True}
    assert result["discarded_windows"][0]["reason"] == "in_flight_below_95_percent"


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
