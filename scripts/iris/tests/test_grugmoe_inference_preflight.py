# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import dataclasses
import hashlib
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


def test_matrix_submission_forwards_explicit_priority_and_keeps_zero_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeJob:
        job_id = "unit-job"

    class FakeClient:
        def submit(self, **kwargs: object) -> FakeJob:
            captured.update(kwargs)
            return FakeJob()

    class FakeControllerContext:
        def __enter__(self) -> FakeClient:
            return FakeClient()

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        grug_preflight,
        "_clean_pushed_checkout",
        lambda: {"commit": "b" * 40, "branch": "unit", "origin": "example.invalid/repo"},
    )
    monkeypatch.setattr(grug_preflight, "controller_client", lambda _: FakeControllerContext())
    image = "example.invalid/task@sha256:" + "a" * 64
    args = parse_args(
        [
            "submit-matrix",
            "--plan",
            "instrument-v1",
            "--task-image",
            image,
            "--priority",
            "production",
        ]
    )
    default_args = parse_args(
        [
            "submit-matrix",
            "--plan",
            "instrument-v1",
            "--task-image",
            image,
        ]
    )

    summary = grug_preflight.submit_matrix(args)

    assert default_args.priority == "interactive"
    entrypoint = captured["entrypoint"]
    assert isinstance(entrypoint, grug_preflight.Entrypoint)
    assert entrypoint.command[entrypoint.command.index("--iris-priority") + 1] == "production"
    assert captured["priority_band"] == grug_preflight.PRIORITY_BANDS[Priority.PRODUCTION]
    assert captured["max_retries_failure"] == 0
    assert captured["max_retries_preemption"] == 0
    assert captured["max_task_failures"] == 0
    assert summary["priority"] == "production"


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
    assert warm_up["successor_waves_after_final_pass"] == grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES
    assert len(roots) == 3 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    assert len(records) == 24 * (
        grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
        + grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES
    )
    assert {record["pass"] for record in records} == set(range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES))
    final_pass = grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES - 1
    assert {record["wave"] for record in records if record["pass"] < final_pass} == {0}
    assert {record["wave"] for record in records if record["pass"] == final_pass} == {0, 1}
    assert {record["slot_id"] for record in records} == set(range(24))
    assert {record["data_parallel_rank"] for record in records} == {0, 1, 2}
    assert {record["max_tokens"] for record in records} == {2}
    assert {record["token_count"] for record in records} == {5, 6, 7}
    assert all(record["disjoint_from_measured_roots"] for record in records)
    assert all(record["shared_root_graph"] for record in records)
    assert len({record["token_ids_sha256"] for record in records}) == (
        24 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
    )
    assert len({record["token_ids_sha256"] for record in roots}) == len(roots)
    assert all(record["disjoint_from_measured_roots"] for record in roots)
    assert all(record["disjoint_from_other_passes"] for record in roots)
    for pass_index in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES):
        pass_records = [record for record in records if record["pass"] == pass_index]
        for wave_index in {record["wave"] for record in pass_records}:
            wave_records = [record for record in pass_records if record["wave"] == wave_index]
            sharing = {root: sum(record["root"] == root for record in wave_records) for root in range(3)}
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
    first_wave = {
        record["slot_id"]: record for record in records if record["pass"] == final_pass and record["wave"] == 0
    }
    successor_wave = {
        record["slot_id"]: record for record in records if record["pass"] == final_pass and record["wave"] == 1
    }
    assert first_wave.keys() == successor_wave.keys()
    assert all(
        (
            first_wave[slot_id]["manifest_request_id"],
            first_wave[slot_id]["token_ids_sha256"],
        )
        == (
            successor_wave[slot_id]["manifest_request_id"],
            successor_wave[slot_id]["token_ids_sha256"],
        )
        for slot_id in first_wave
    )


def test_health_warmup_populates_shared_roots_and_barriers_each_wave_including_successor(
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
    final_pass = grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES - 1
    barriers = {
        (pass_index, wave_index): threading.Barrier(3)
        for pass_index in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES)
        for wave_index in range(
            1 + (grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES if pass_index == final_pass else 0)
        )
    }
    events: list[str] = []
    lock = threading.Lock()

    def completion(*_: object, request_id: str, data_parallel_rank: int, **__: object) -> dict[str, object]:
        with lock:
            events.append(f"start:{request_id}")
        if "-wave-" in request_id:
            pass_index = next(
                index
                for index in range(grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES)
                if f"-warm-pass-{index:02d}-wave-" in request_id
            )
            wave_index = next(index for index in range(2) if f"-wave-{index:02d}-slot-" in request_id)
            barriers[pass_index, wave_index].wait(timeout=2)
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
    assert len(branches) == 3 * (passes + grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES)
    assert len(roots) == 3 * passes
    for pass_index in range(passes):
        root_marker = f"-warm-pass-{pass_index:02d}-populate-root-"
        root_finishes = [
            index for index, event in enumerate(events) if event.startswith("finish:") and root_marker in event
        ]
        previous_finishes = root_finishes
        wave_count = 1 + (
            grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES if pass_index == final_pass else 0
        )
        for wave_index in range(wave_count):
            branch_marker = f"-warm-pass-{pass_index:02d}-wave-{wave_index:02d}-slot-"
            branch_starts = [
                index for index, event in enumerate(events) if event.startswith("start:") and branch_marker in event
            ]
            branch_finishes = [
                index for index, event in enumerate(events) if event.startswith("finish:") and branch_marker in event
            ]
            assert len(previous_finishes) == len(branch_starts) == len(branch_finishes) == 3
            assert max(previous_finishes) < min(branch_starts)
            previous_finishes = branch_finishes
        if pass_index + 1 < passes:
            next_root_marker = f"-warm-pass-{pass_index + 1:02d}-populate-root-"
            next_root_starts = [
                index for index, event in enumerate(events) if event.startswith("start:") and next_root_marker in event
            ]
            assert max(previous_finishes) < min(next_root_starts)


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


def test_timed_chat_completion_preserves_frozen_ids_in_identity_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Response:
        ok = True
        content = b'{"choices":[{}]}'

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

    def post(url: str, **kwargs: object) -> Response:
        observed.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(grug_preflight.requests, "post", post)
    _, timing = grug_preflight._timed_completion(
        "http://server",
        "model",
        [1, 37, 9, 255],
        request_transport="chat",
    )

    assert observed["url"] == "http://server/v1/chat/completions"
    body = observed["json"]
    assert isinstance(body, dict)
    assert body["messages"] == [
        {
            "role": "user",
            "content": "".join(grug_preflight.IDENTITY_CHAT_TOKENS[token_id] for token_id in [1, 37, 9, 255]),
        }
    ]
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 1
    assert timing["request_transport"] == "chat"


def test_timed_chat_completion_fits_vllm_character_guard_at_65k(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        ok = True
        content = b'{"choices":[{}]}'

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

    def post(url: str, **kwargs: object) -> Response:
        observed.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(grug_preflight.requests, "post", post)
    prompt_token_ids = [255] * 65_535
    grug_preflight._timed_completion(
        "http://server",
        "model",
        prompt_token_ids,
        max_tokens=1,
        request_transport="chat",
    )

    body = observed["json"]
    assert isinstance(body, dict)
    content = body["messages"][0]["content"]
    assert len(content) == len(prompt_token_ids)
    assert content == "".join(grug_preflight.IDENTITY_CHAT_TOKENS[token_id] for token_id in prompt_token_ids)


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


def test_chat_r3_summary_accepts_only_generated_token_nested_lists() -> None:
    case = CASES["exact-reference-ep16"]
    routes = np.arange(3 * case.num_hidden_layers * case.num_experts_per_tok, dtype=np.int64)
    routes = (routes % case.num_experts).reshape(3, case.num_hidden_layers, case.num_experts_per_tok)
    payload = {"choices": [{"routed_experts": routes.tolist()}]}

    summary, retained = grug_preflight._health_route_summary(
        payload,
        case=case,
        expected_positions=3,
        r3_enabled=True,
        request_transport="chat",
        keep_routes=True,
    )

    assert summary["shape"] == [3, 48, 4]
    assert summary["carrier_npy_bytes"] == 0
    assert summary["carrier_base64_bytes"] == 0
    assert summary["carrier_json_bytes"] > routes.size
    assert summary["root_prefix_positions_compared"] == 0
    np.testing.assert_array_equal(retained, routes)


def test_route_audit_reconciles_owned_experts_without_drops() -> None:
    case = CASES["tiny"]
    counts = [[1] * case.num_experts for _ in range(case.num_hidden_layers)]
    snapshot = {
        "mode": "record",
        "num_layers": case.num_hidden_layers,
        "num_experts": case.num_experts,
        "assignment_count": case.num_hidden_layers * case.num_experts,
        "counts": counts,
        "local_expert_mask": counts,
        "worker": {"global_rank": 0, "local_rank": 0, "dp_rank": 0, "ep_rank": 0},
    }

    summary = grug_preflight._health_route_audit_summary(
        [snapshot],
        case=case,
        mode="record",
        routing_regime="balanced",
        expected_assignment_count=case.num_hidden_layers * case.num_experts,
    )

    assert summary["passed"]
    assert summary["assignment_count"] == summary["expected_assignment_count"]
    assert summary["counts_outside_ownership"] == 0


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

    assert warm_requests == 3 * (
        grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES
        + grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES
    )
    assert arm["warm_up"] == {
        "requests": warm_requests + 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "branch_requests": warm_requests,
        "root_population_requests": 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "root_copies": 18 * grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "rolling_passes": grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "successor_waves_after_final_pass": grug_preflight.HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES,
        "target_concurrency": 3,
        "max_tokens_per_request": 1,
        "data_parallel_ranks_covered": list(range(16)),
        "shared_root_graph": True,
        "full_pass_barrier": True,
        "full_wave_barrier": True,
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
    assert arm["requests"]["branch_coverage"] == {
        "expected": 144,
        "observed": 144,
        "passed": True,
        "required": True,
    }
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
    assert result["manifest"] == {"expected": 3, "observed": 3, "passed": True, "required": True}
    assert result["discarded_windows"][0]["reason"] == "in_flight_below_95_percent"


def test_provisional_plateau_can_close_without_full_manifest_coverage() -> None:
    plateau = PlateauWindow(
        PlateauRequirements(
            target_concurrency=3,
            minimum_seconds=30,
            minimum_generated_tokens=32_768,
            required_request_ids=frozenset({"short", "medium", "long"}),
            require_manifest_coverage=False,
        )
    )

    assert plateau.observe_in_flight(
        now=0,
        in_flight=3,
        generation_counter=1_000,
        prompt_counter=2_000,
    ) == "opened"
    plateau.record_completion(request_id="short", cohort="short", completion_tokens=8_192, succeeded=True)
    assert plateau.ready_to_close(now=30, in_flight=3, generation_counter=33_768)
    result = plateau.close(now=30, in_flight=3, generation_counter=33_768, prompt_counter=4_000)

    assert result["manifest"] == {"expected": 3, "observed": 1, "passed": False, "required": False}


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


def test_matrix_calibration_uses_frozen_95_percent_lowest_concurrency_rule() -> None:
    arms: list[dict[str, object]] = []
    for concurrency in grug_preflight.CALIBRATION_CONCURRENCIES:
        for mbt in grug_preflight.CALIBRATION_MAX_NUM_BATCHED_TOKENS:
            rate = float(concurrency) / 2
            if concurrency == 144 and mbt == 8192:
                rate = 100.0
            if concurrency == 96:
                rate = 96.0 if mbt == 8192 else 97.0
            arms.append(
                {
                    "arm_id": f"c{concurrency}-m{mbt}",
                    "passed": True,
                    "matrix": {"case": "exact-reference-ep16", "role": "calibration"},
                    "settings": {
                        "target_concurrency": concurrency,
                        "max_num_batched_tokens": mbt,
                        "max_num_seqs": 144,
                    },
                    "headline": {"generation_tokens_per_second_per_gpu": rate},
                }
            )

    selection = grug_preflight._calibration_selection(arms, case_name="exact-reference-ep16")

    assert selection["passed"]
    assert selection["threshold_generation_tokens_per_second_per_gpu"] == 95.0
    assert selection["selected"]["target_concurrency"] == 96
    assert selection["selected"]["max_num_batched_tokens"] == 16384
    assert selection["selected"]["max_num_seqs"] == 144
    followups = grug_preflight._instrument_followup_phases(selection)
    assert [phase["phase_id"] for phase in followups] == [
        "ep16-r3off-aa",
        "ep16-chat-r3off",
        "ep16-chat-r3on",
    ]
    assert followups[0]["concurrencies"] == [96, 96]
    assert followups[1]["request_transport"] == followups[2]["request_transport"] == "chat"
    assert followups[1]["r3_enabled"] is False
    assert followups[2]["r3_enabled"] is True


def test_artifact_writer_uploads_nested_matrix_kv_evidence(tmp_path: Path) -> None:
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

    kv_paths = [
        "metrics/arm-kv.log",
        "metrics/coarse-medium-kv.log",
        "metrics/coarse-long-kv.log",
        "metrics/trajectory-kv.log",
        "metrics/capacity-kv.log",
    ]
    (tmp_path / "events.jsonl").write_text("{}\n")
    for path in kv_paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"evidence for {path}\n")
    result = {
        "arms": [
            {
                "kv_cache": {"source": {"path": kv_paths[0]}},
                "coarse_curve": [
                    {"kv_cache": {"source": {"path": kv_paths[1]}}},
                    {"kv_cache": {"source": {"path": kv_paths[2]}}},
                ],
                "trajectory_65k": {"kv_cache": {"source": {"path": kv_paths[3]}}},
                "capacity_stress_131k": {"kv_cache": {"source": {"path": kv_paths[4]}}},
            }
        ]
    }
    filesystem = MemoryFilesystem()

    records = grug_preflight._write_and_upload_health_artifacts(
        filesystem,
        artifact_dir=tmp_path,
        artifact_prefix="s3://unit-bucket/matrix/run/",
        result=result,
        manifest={},
        metrics_map=[],
        result_markdown="# unit\n",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(kv_paths) <= set(manifest["claimed_files"])
    assert {record["path"] for record in records} >= {f"s3://unit-bucket/matrix/run/{path}" for path in kv_paths}
    assert {f"unit-bucket/matrix/run/{path}" for path in kv_paths} <= set(filesystem.data)


def test_matrix_model_configs_use_the_artifact_json_domain() -> None:
    configs = grug_preflight._matrix_model_configs(["exact-reference-ep16"])

    assert configs == json.loads(json.dumps({"exact-reference-ep16": dataclasses.asdict(CASES["exact-reference-ep16"])}))
    assert configs["exact-reference-ep16"]["sconv_sites"] == ["k", "v", "attn", "mlp"]


def test_topology_calibration_sources_require_independent_frozen_selections() -> None:
    class MemoryFilesystem:
        def __init__(self) -> None:
            self.data: dict[str, bytes] = {}

        def cat_file(self, key: str) -> bytes:
            return self.data[key]

    filesystem = MemoryFilesystem()
    marin_commit = "b" * 40
    reader_marin_commit = "d" * 40
    task_image = "example.invalid/image@sha256:" + "a" * 64

    def add_source(*, plan: str, run_id: str, case: str, concurrency: int, mbt: int) -> None:
        artifact_prefix = grug_preflight._matrix_artifact_prefix(plan, run_id)
        manifest = {
            "plan": plan,
            "run_id": run_id,
            "provenance": {
                "marin_commit": marin_commit,
                "vllm_commit": VLLM_SHA,
                "task_image": task_image,
            },
        }
        result = {
            "plan": plan,
            "run_id": run_id,
            "passed": True,
            "analysis": {
                "calibration": {
                    "selected": {
                        "case": case,
                        "target_concurrency": concurrency,
                        "max_num_batched_tokens": mbt,
                        "max_num_seqs": 144,
                    }
                }
            },
        }
        manifest_bytes = json.dumps(manifest).encode()
        result_bytes = json.dumps(result).encode()
        filesystem.data[grug_preflight._s3_key(f"{artifact_prefix}manifest.json")] = manifest_bytes
        filesystem.data[grug_preflight._s3_key(f"{artifact_prefix}result.json")] = result_bytes
        receipt = {
            "plan": plan,
            "run_id": run_id,
            "passed": True,
            "benchmark_passed": True,
            "reader_marin_commit": reader_marin_commit,
            "task_image": task_image,
            "source_object_sha256": {
                "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
                "result.json": hashlib.sha256(result_bytes).hexdigest(),
            },
        }
        receipt_uri = f"{grug_preflight._matrix_control_prefix(plan, run_id)}independent-readback.json"
        filesystem.data[grug_preflight._s3_key(receipt_uri)] = json.dumps(receipt).encode()

    add_source(plan="ep8-calibration", run_id="ep8-run", case="reference-ep8", concurrency=96, mbt=16384)
    add_source(
        plan="instrument-v1",
        run_id="ep16-run",
        case="exact-reference-ep16",
        concurrency=144,
        mbt=8192,
    )

    sources = grug_preflight._verified_topology_calibration_sources(
        filesystem,
        ep8_run_id="ep8-run",
        ep16_run_id="ep16-run",
        ep8_concurrency=96,
        ep8_max_num_batched_tokens=16384,
        ep16_concurrency=144,
        ep16_max_num_batched_tokens=8192,
        max_num_seqs=144,
        marin_commit=marin_commit,
        task_image=task_image,
    )

    assert sources["ep8"]["selection"]["target_concurrency"] == 96
    assert sources["ep16"]["selection"]["max_num_batched_tokens"] == 8192
    assert sources["ep16"]["provenance"]["marin_commit"] == marin_commit
    assert sources["ep16"]["provenance"]["reader_marin_commit"] == reader_marin_commit
    with pytest.raises(RuntimeError, match="selection"):
        grug_preflight._verified_topology_calibration_sources(
            filesystem,
            ep8_run_id="ep8-run",
            ep16_run_id="ep16-run",
            ep8_concurrency=144,
            ep8_max_num_batched_tokens=16384,
            ep16_concurrency=144,
            ep16_max_num_batched_tokens=8192,
            max_num_seqs=144,
            marin_commit=marin_commit,
            task_image=task_image,
        )


def test_topology_plan_reverses_fresh_ep8_ep16_arms_and_adds_noop_control() -> None:
    args = parse_args(
        [
            "matrix-worker",
            "--plan",
            "topology-v1",
            "--run-id",
            "unit",
            "--task-image",
            "example.invalid/image@sha256:" + "a" * 64,
            "--marin-commit",
            "b" * 40,
            "--iris-priority",
            "interactive",
            "--submitted-coscheduling",
            "nvlink.domain",
            "--ep8-concurrency",
            "96",
            "--ep8-max-num-batched-tokens",
            "16384",
            "--ep16-concurrency",
            "144",
            "--ep16-max-num-batched-tokens",
            "8192",
            "--ep8-calibration-run-id",
            "ep8-run",
            "--ep16-instrument-run-id",
            "ep16-run",
        ]
    )

    grug_preflight._validate_matrix_args(args)

    phases = grug_preflight._matrix_initial_phases(args)

    assert len(phases) == 10
    assert [phase["case"] for phase in phases[:4]] == [
        "reference-ep8",
        "exact-reference-ep16",
        "exact-reference-ep16",
        "reference-ep8",
    ]
    assert [phase["case"] for phase in phases[4:8]] == [
        "reference-ep8",
        "exact-reference-ep16",
        "exact-reference-ep16",
        "reference-ep8",
    ]
    assert [phase["routing_regime"] for phase in phases[:8]] == ["canonical"] * 4 + ["balanced"] * 4
    assert [phase["route_audit_mode"] for phase in phases[-2:]] == ["noop", "record"]
    assert all(phase["r3_enabled"] is False for phase in phases)
    assert all(phase["request_transport"] == "completion" for phase in phases)
    followups = grug_preflight._topology_followup_phases(args)
    assert [phase["phase_id"] for phase in followups] == [
        "targeted-ep8-chat-r3off",
        "targeted-ep8-chat-r3on",
    ]
    assert all(phase["case"] == "reference-ep8" for phase in followups)
    assert all(phase["request_transport"] == "chat" for phase in followups)
    assert [phase["r3_enabled"] for phase in followups] == [False, True]


@pytest.mark.parametrize(
    ("candidate", "order", "expected_cases"),
    [
        ("window1024-ep16", "ab", ["exact-reference-ep16", "window1024-ep16"]),
        ("global-every4-ep16", "ba", ["global-every4-ep16", "exact-reference-ep16"]),
    ],
)
def test_attention_pair_plan_is_a_fresh_reversed_pair_with_homogeneous_slices(
    candidate: str,
    order: str,
    expected_cases: list[str],
) -> None:
    args = parse_args(
        [
            "submit-matrix",
            "--plan",
            "attention-pair-v1",
            "--run-id",
            f"exp4-{candidate}-{order}-unit",
            "--task-image",
            "example.invalid/image@sha256:" + "a" * 64,
            "--priority",
            "production",
            "--ep16-instrument-run-id",
            "instrument-run",
            "--attention-candidate",
            candidate,
            "--attention-order",
            order,
        ]
    )

    grug_preflight._validate_matrix_args(args)
    phases = grug_preflight._matrix_initial_phases(args)
    command = grug_preflight._matrix_worker_argv(
        args,
        run_id=args.run_id,
        image=args.task_image,
        marin_commit="b" * 40,
    )

    assert [phase["case"] for phase in phases] == expected_cases
    assert all(phase["homogeneous_slices"] is True for phase in phases)
    assert all(phase["role"] == "attention-comparison" for phase in phases)
    assert all(phase["r3_enabled"] is False for phase in phases)
    assert all(phase["request_transport"] == "completion" for phase in phases)
    assert command[command.index("--attention-candidate") + 1] == candidate
    assert command[command.index("--attention-order") + 1] == order
    assert command[command.index("--ep16-instrument-run-id") + 1] == "instrument-run"
    assert command[command.index("--iris-priority") + 1] == "production"


def test_attention_finalist_plan_runs_65k_trajectory_and_131k_capacity_on_fresh_servers() -> None:
    args = parse_args(
        [
            "submit-matrix",
            "--plan",
            "attention-finalist-v1",
            "--run-id",
            "exp4-window1024-ep16-validation-unit",
            "--task-image",
            "example.invalid/image@sha256:" + "a" * 64,
            "--priority",
            "production",
            "--ep16-instrument-run-id",
            "instrument-run",
            "--attention-finalist",
            "window1024-ep16",
        ]
    )

    grug_preflight._validate_matrix_args(args)
    phases = grug_preflight._matrix_initial_phases(args)
    command = grug_preflight._matrix_worker_argv(
        args,
        run_id=args.run_id,
        image=args.task_image,
        marin_commit="b" * 40,
    )

    assert [phase["case"] for phase in phases] == [
        "exact-reference-131k-ep16",
        "window1024-131k-ep16",
    ]
    assert all(phase["trajectory_65k"] is True for phase in phases)
    assert all(phase["capacity_stress_131k"] is True for phase in phases)
    assert all(phase["homogeneous_slices"] is False for phase in phases)
    assert all(phase["r3_enabled"] is False for phase in phases)
    assert command[command.index("--attention-finalist") + 1] == "window1024-ep16"
    assert command[command.index("--ep16-instrument-run-id") + 1] == "instrument-run"
    assert command[command.index("--iris-priority") + 1] == "production"


@pytest.mark.parametrize("target", grug_preflight.ATTENTION_CAPACITY_CONCURRENCIES)
def test_attention_capacity_workload_has_unique_live_inputs_and_balanced_dp16_roots(target: int) -> None:
    workload = grug_preflight._attention_capacity_workload(target)
    frozen = grug_preflight.deterministic_workload(seed=grug_preflight.DUMMY_SEED)
    schedule = grug_preflight._health_slot_schedule(workload, target_concurrency=target)
    first_live_request_ids = [slot["cyclic_request_ids"][0] for slot in schedule]
    requests = workload["requests"]
    capacity_dp_size = CASES["exact-reference-ep16"].data_parallel_size
    roots_per_cohort_per_rank = target // 3 // 8 // capacity_dp_size

    assert workload["request_count"] == target
    assert workload["roots"][: frozen["root_count"]] == frozen["roots"]
    assert workload["requests"][: frozen["request_count"]] == frozen["requests"]
    assert len(requests) == target
    assert len({request["request_id"] for request in requests}) == target
    assert len(
        {
            hashlib.sha256(bytes(grug_preflight.materialize_prompt(workload, request))).hexdigest()
            for request in requests
        }
    ) == target
    assert len(schedule) == target
    assert len(set(first_live_request_ids)) == target
    assert set(first_live_request_ids) == {request["request_id"] for request in requests}
    assert {
        sum(
            root["cohort"] == cohort and int(root["root"]) % capacity_dp_size == rank
            for root in workload["roots"]
        )
        for cohort in ("short", "medium", "long")
        for rank in range(capacity_dp_size)
    } == {roots_per_cohort_per_rank}


def test_attention_capacity_c144_keeps_the_original_frozen_manifest_hash() -> None:
    original = grug_preflight._health_workload_manifest(
        grug_preflight.deterministic_workload(seed=grug_preflight.DUMMY_SEED),
        concurrencies=[144],
    )
    capacity_compatible = grug_preflight._health_workload_manifest(
        grug_preflight._attention_capacity_workload(144, seed=grug_preflight.DUMMY_SEED),
        concurrencies=[144],
    )

    assert capacity_compatible == original
    assert capacity_compatible["frozen_inputs_sha256"] == original["frozen_inputs_sha256"]


def test_attention_capacity_scout_freezes_one_matched_point_and_relaxes_only_scout_floors() -> None:
    args = parse_args(
        [
            "submit-matrix",
            "--plan",
            "attention-capacity-v1",
            "--run-id",
            "capacity-scout-c384-matched-ab-unit",
            "--task-image",
            "example.invalid/image@sha256:" + "a" * 64,
            "--capacity-mode",
            "scout",
            "--capacity-concurrency",
            "384",
            "--capacity-variant",
            "matched",
            "--attention-order",
            "ab",
            "--capacity-c144-ab-run-id",
            "global-every4-ep16-ab-source",
            "--capacity-c144-ba-run-id",
            "global-every4-ep16-ba-source",
        ]
    )

    grug_preflight._validate_matrix_args(args)
    phases = grug_preflight._matrix_initial_phases(args)
    command = grug_preflight._matrix_worker_argv(
        args,
        run_id=args.run_id,
        image=args.task_image,
        marin_commit="b" * 40,
    )

    assert args.minimum_seconds == grug_preflight.ATTENTION_CAPACITY_SCOUT_MINIMUM_SECONDS
    assert args.minimum_generated_tokens == grug_preflight.ATTENTION_CAPACITY_SCOUT_MINIMUM_GENERATED_TOKENS
    assert [phase["case"] for phase in phases] == ["exact-reference-ep16", "global-every4-ep16"]
    assert all(phase["concurrencies"] == [384] for phase in phases)
    assert all(phase["max_num_batched_tokens"] == 8192 for phase in phases)
    assert all(phase["max_num_seqs"] == 384 for phase in phases)
    assert all(phase["require_manifest_coverage"] is False for phase in phases)
    assert all(phase["r3_enabled"] is False for phase in phases)
    assert command[command.index("--minimum-seconds") + 1] == "30.0"
    assert command[command.index("--capacity-concurrency") + 1] == "384"


def test_attention_capacity_confirmation_reverses_pair_and_single_continuation_has_no_order() -> None:
    common = [
        "--plan",
        "attention-capacity-v1",
        "--task-image",
        "example.invalid/image@sha256:" + "a" * 64,
        "--capacity-mode",
        "confirm",
        "--capacity-c144-ab-run-id",
        "global-every4-ep16-ab-source",
        "--capacity-c144-ba-run-id",
        "global-every4-ep16-ba-source",
    ]
    matched = parse_args(
        [
            "matrix-worker",
            *common,
            "--run-id",
            "capacity-confirm-c768-matched-ba-unit",
            "--marin-commit",
            "b" * 40,
            "--iris-priority",
            "interactive",
            "--submitted-coscheduling",
            "nvlink.domain",
            "--capacity-concurrency",
            "768",
            "--capacity-variant",
            "matched",
            "--attention-order",
            "ba",
        ]
    )
    grug_preflight._validate_matrix_args(matched)
    matched_phases = grug_preflight._matrix_initial_phases(matched)
    assert [phase["case"] for phase in matched_phases] == [
        "global-every4-ep16",
        "exact-reference-ep16",
    ]
    assert all(phase["require_manifest_coverage"] is True for phase in matched_phases)

    single = parse_args(
        [
            "matrix-worker",
            *common,
            "--run-id",
            "capacity-confirm-c1536-reference-single-unit",
            "--marin-commit",
            "b" * 40,
            "--iris-priority",
            "interactive",
            "--submitted-coscheduling",
            "nvlink.domain",
            "--capacity-concurrency",
            "1536",
            "--capacity-variant",
            "reference",
        ]
    )
    grug_preflight._validate_matrix_args(single)
    single_phases = grug_preflight._matrix_initial_phases(single)
    assert len(single_phases) == 1
    assert single_phases[0]["case"] == "exact-reference-ep16"
    assert single_phases[0]["order"] is None


def test_attention_capacity_curve_uses_only_stable_points_for_knee_and_maximum_safe() -> None:
    curve = grug_preflight._attention_capacity_curve(
        [
            {
                "variant": "reference",
                "concurrency": 144,
                "qualification": "stable",
                "generation_tokens_per_second_per_gpu": 300.0,
            },
            {
                "variant": "reference",
                "concurrency": 384,
                "qualification": "stable",
                "generation_tokens_per_second_per_gpu": 315.0,
            },
            {
                "variant": "reference",
                "concurrency": 1536,
                "qualification": "stable",
                "generation_tokens_per_second_per_gpu": 310.0,
            },
            {
                "variant": "global-every4",
                "concurrency": 144,
                "qualification": "stable",
                "generation_tokens_per_second_per_gpu": 280.0,
            },
            {
                "variant": "global-every4",
                "concurrency": 384,
                "qualification": "provisional-pass",
                "generation_tokens_per_second_per_gpu": 300.0,
            },
            {
                "variant": "global-every4",
                "concurrency": 768,
                "qualification": "failed",
                "generation_tokens_per_second_per_gpu": 250.0,
            },
        ]
    )

    assert curve["variants"]["reference"] == {
        "best_stable_generation_tokens_per_second_per_gpu": 315.0,
        "throughput_knee": 144,
        "maximum_safe_concurrency": 1536,
        "maximum_safe_is_lower_bound": True,
        "first_failed_boundary": None,
    }
    assert curve["variants"]["global-every4"] == {
        "best_stable_generation_tokens_per_second_per_gpu": 280.0,
        "throughput_knee": 144,
        "maximum_safe_concurrency": 144,
        "maximum_safe_is_lower_bound": False,
        "first_failed_boundary": 768,
    }


def test_homogeneous_slice_uses_all_branches_and_keeps_each_root_on_one_dp_rank() -> None:
    workload = grug_preflight.deterministic_workload(seed=grug_preflight.DUMMY_SEED)
    case = CASES["exact-reference-ep16"]

    schedule = grug_preflight._homogeneous_cohort_schedule(workload, case=case, cohort="long")

    assert len(schedule) == 48
    assert [item["request"]["request_id"] for item in schedule] == [
        request["request_id"] for request in workload["requests"] if request["cohort"] == "long"
    ]
    assert all(item["data_parallel_rank"] == item["request"]["root"] % case.data_parallel_size for item in schedule)
    assert len({item["request"]["root"] for item in schedule}) == 6


def test_coarse_curve_reader_recomputes_raw_inputs_and_rejects_duplicate_events() -> None:
    case = CASES["exact-reference-ep16"]
    raw_workload = grug_preflight.deterministic_workload(seed=grug_preflight.DUMMY_SEED)
    workload = grug_preflight._health_workload_manifest(raw_workload, case=case, concurrencies=[144])
    manifest_requests = {request["request_id"]: request for request in workload["requests"]}
    kv_groups = [
        {
            "engine_idx": engine,
            "role": "attention",
            "kind": "full_attention",
            "active_requests": 1,
            "active_blocks": 1,
            "active_payload_bytes": 2,
            "active_padded_bytes": 2,
            "active_physical_bytes": 2,
            "reserved_physical_bytes": 100,
        }
        for engine in range(case.data_parallel_size)
    ]
    kv_payload = ("INFO GrugMoE KV group usage: " + json.dumps(kv_groups)).encode()
    curve = []
    parsed_by_path = {}
    snapshot_by_path = {}
    kv_sources = {}
    events = []
    for cohort_index, cohort in enumerate(("short", "medium", "long")):
        raw_schedule = grug_preflight._homogeneous_cohort_schedule(raw_workload, case=case, cohort=cohort)
        schedule = [
            {
                "slot": item["slot"],
                "data_parallel_rank": item["data_parallel_rank"],
                "request_id": item["request"]["request_id"],
                "prompt_token_ids_sha256": manifest_requests[item["request"]["request_id"]]["prompt_token_ids_sha256"],
            }
            for item in raw_schedule
        ]
        expected_generation = sum(manifest_requests[item["request_id"]]["max_tokens"] for item in schedule)
        before_path = f"metrics/{cohort}-before.prom"
        after_path = f"metrics/{cohort}-after.prom"
        parsed_by_path[before_path] = parse_labeled_prometheus(
            "\n".join(
                (
                    'vllm:generation_tokens_total{engine="0"} 0',
                    'vllm:request_success_total{engine="0",finished_reason="length"} 0',
                    'vllm:num_preemptions_total{engine="0"} 0',
                )
            )
        )
        parsed_by_path[after_path] = parse_labeled_prometheus(
            "\n".join(
                (
                    f'vllm:generation_tokens_total{{engine="0"}} {expected_generation}',
                    'vllm:request_success_total{engine="0",finished_reason="length"} 48',
                    'vllm:num_preemptions_total{engine="0"} 0',
                )
            )
        )
        snapshot_by_path[before_path] = {"monotonic_seconds": float(cohort_index * 10 + 1)}
        snapshot_by_path[after_path] = {"monotonic_seconds": float(cohort_index * 10 + 3)}
        kv_path = f"metrics/arm-{cohort}-kv.log"
        kv = grug_preflight._health_kv_summary_from_text(
            kv_payload.decode(),
            case=case,
            target_concurrency=48,
        )
        final_context_tokens = manifest_requests[schedule[0]["request_id"]]["final_token_count"]
        layer_schedule = grug_preflight.layer_types(case.num_hidden_layers, global_interval=case.global_every)
        kv["attention_prediction"] = {
            "final_context_tokens": final_context_tokens,
            "local_layers": layer_schedule.count("sliding_attention"),
            "global_layers": layer_schedule.count("full_attention"),
            "per_live_sequence_bytes": grug_preflight.predict_kv_bytes(
                sequence_length=final_context_tokens,
                local_layers=layer_schedule.count("sliding_attention"),
                global_layers=layer_schedule.count("full_attention"),
                local_kv_heads=case.local_kv_heads,
                global_kv_heads=case.global_kv_heads,
                head_dim=case.head_dim,
                sliding_window=case.sliding_window,
            ),
            "scope": "semantic K and V payload before block rounding",
        }
        kv["source"] = {
            "path": kv_path,
            "bytes": len(kv_payload),
            "sha256": hashlib.sha256(kv_payload).hexdigest(),
        }
        kv_sources[kv_path] = kv_payload
        rate = expected_generation / 2 / case.data_parallel_size
        curve.append(
            {
                "cohort": cohort,
                "passed": True,
                "gates": {"synthetic": True},
                "final_context_tokens": final_context_tokens,
                "slice_concurrency": 48,
                "population_requests": 6,
                "measured_requests": 48,
                "schedule": schedule,
                "elapsed_seconds": 2.0,
                "engine_generation_tokens": expected_generation,
                "generation_tokens_per_second_per_gpu": rate,
                "gpu_seconds_per_generated_token": case.data_parallel_size * 2 / expected_generation,
                "slowdown_from_short_percent": 0.0,
                "preemptions": 0,
                "kv_cache": kv,
                "metrics": {"boundary_start": before_path, "boundary_end": after_path},
            }
        )
        events.extend(
            {
                "event": "cohort_slice_request_completed",
                "arm_id": "arm",
                "cohort": cohort,
                "request_id": f"arm-{cohort}-slot-{item['slot']:03d}-{item['request_id']}",
                "manifest_request_id": item["request_id"],
                "data_parallel_rank": item["data_parallel_rank"],
                "completion_tokens": manifest_requests[item["request_id"]]["max_tokens"],
                "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
                "generated_token_ids_sha256": "b" * 64,
                "final_prefix_token_ids_sha256": "c" * 64,
                "sampled_token_logprobs_sha256": "d" * 64,
            }
            for item in schedule
        )

    arm = {"arm_id": "arm", "matrix": {"case": case.name}, "coarse_curve": curve}
    contract = grug_preflight._matrix_coarse_curve_contract(
        arm,
        workload=workload,
        parsed_by_path=parsed_by_path,
        snapshot_by_path=snapshot_by_path,
        event_records=events,
        kv_source_by_path=kv_sources,
    )
    duplicate_contract = grug_preflight._matrix_coarse_curve_contract(
        arm,
        workload=workload,
        parsed_by_path=parsed_by_path,
        snapshot_by_path=snapshot_by_path,
        event_records=[*events, events[0]],
        kv_source_by_path=kv_sources,
    )

    assert contract["passed"]
    assert all(point["kv_recomputed"] for point in contract["points"])
    assert not duplicate_contract["passed"]


def test_capacity_probe_reader_recomputes_raw_counters_events_and_kv_log() -> None:
    case = CASES["exact-reference-131k-ep16"]
    before_text = "\n".join(
        (
            'vllm:generation_tokens_total{engine="0"} 10',
            'vllm:request_success_total{engine="0",finished_reason="length"} 3',
            'vllm:num_preemptions_total{engine="0"} 0',
        )
    )
    after_text = "\n".join(
        (
            'vllm:generation_tokens_total{engine="0"} 12',
            'vllm:request_success_total{engine="0",finished_reason="length"} 4',
            'vllm:num_preemptions_total{engine="0"} 0',
        )
    )
    kv_groups = [
        {
            "engine_idx": engine,
            "role": "attention",
            "kind": "full_attention",
            "active_requests": 1 if engine == 0 else 0,
            "active_blocks": 1 if engine == 0 else 0,
            "active_payload_bytes": 2 if engine == 0 else 0,
            "active_padded_bytes": 2 if engine == 0 else 0,
            "active_physical_bytes": 2 if engine == 0 else 0,
            "reserved_physical_bytes": 100,
        }
        for engine in range(case.data_parallel_size)
    ]
    kv_text = "INFO GrugMoE KV group usage: " + json.dumps(kv_groups)
    kv = grug_preflight._health_kv_summary_from_text(kv_text, case=case, target_concurrency=1)
    kv["source"] = {
        "path": "metrics/capacity-kv.log",
        "bytes": len(kv_text.encode()),
        "sha256": hashlib.sha256(kv_text.encode()).hexdigest(),
    }
    record = {
        "request_id": "capacity-root-00-branch-00",
        "root": 0,
        "branch": 0,
        "data_parallel_rank": 0,
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "final_token_count": 5,
        "prompt_token_ids_sha256": "a" * 64,
        "generated_token_ids_sha256": "b" * 64,
        "final_prefix_token_ids_sha256": "c" * 64,
        "sampled_token_logprobs_sha256": "d" * 64,
        "sampled_token_logprobs_count": 2,
    }
    arm = {
        "arm_id": "arm-capacity",
        "matrix": {"case": case.name},
        "capacity_stress_131k": {
            "passed": True,
            "gates": {"synthetic": True},
            "engine_generation_tokens": 2,
            "request_successes": 1,
            "preemptions": 0,
            "elapsed_seconds": 2.0,
            "generation_tokens_per_second_per_gpu": 2 / 2 / case.data_parallel_size,
            "request_provenance": [record],
            "kv_cache": kv,
            "metrics": {"boundary_start": "before.prom", "boundary_end": "after.prom"},
        },
    }
    manifest = {
        "request_count": 1,
        "response_tokens": 2,
        "requests": [
            {
                "request_id": record["request_id"],
                "root": 0,
                "branch": 0,
                "data_parallel_rank": 0,
                "prompt_token_count": 3,
                "max_tokens": 2,
                "final_token_count": 5,
                "prompt_token_ids_sha256": "a" * 64,
            }
        ],
        "roots": [],
    }
    event = {
        "timestamp": "unit",
        "monotonic_seconds": 2.0,
        "event": "capacity_131k_request_completed",
        "arm_id": arm["arm_id"],
        **record,
    }
    contract = grug_preflight._matrix_sequence_probe_contract(
        arm,
        field="capacity_stress_131k",
        manifest=manifest,
        parsed_by_path={
            "before.prom": parse_labeled_prometheus(before_text),
            "after.prom": parse_labeled_prometheus(after_text),
        },
        snapshot_by_path={
            "before.prom": {"monotonic_seconds": 1.0},
            "after.prom": {"monotonic_seconds": 3.0},
        },
        event_records=[event],
        kv_source_by_path={"metrics/capacity-kv.log": kv_text.encode()},
    )

    assert contract["passed"]
    assert contract["kv_recomputed"]
    missing_raw_kv = grug_preflight._matrix_sequence_probe_contract(
        arm,
        field="capacity_stress_131k",
        manifest=manifest,
        parsed_by_path={
            "before.prom": parse_labeled_prometheus(before_text),
            "after.prom": parse_labeled_prometheus(after_text),
        },
        snapshot_by_path={
            "before.prom": {"monotonic_seconds": 1.0},
            "after.prom": {"monotonic_seconds": 3.0},
        },
        event_records=[event],
        kv_source_by_path={},
    )
    assert not missing_raw_kv["passed"]


def test_matrix_reader_reconstructs_phase_commands_and_matched_chat_diff() -> None:
    provenance = {
        "iris_task_count": 4,
        "iris_job_id": "job-1",
        "iris_coscheduling": "nvlink.domain",
        "marin_commit": "a" * 40,
        "vllm_commit": VLLM_SHA,
        "task_image": "example.invalid/image@sha256:" + "b" * 64,
    }

    def build(phase: dict[str, object]) -> dict[str, object]:
        case = CASES[str(phase["case"])]
        leader = "10.0.0.1"
        environment = dict(grug_preflight.VLLM_SERVER_DEV_MODE_ENVIRONMENT)
        ranks: list[dict[str, object]] = []
        startup: list[dict[str, object]] = []
        for rank in range(4):
            active = rank < case.node_count
            command = None
            if active:
                command = vllm_command(
                    vllm_args(
                        case,
                        model_dir=f"/tmp/{phase['phase_id']}/rank-{rank}",
                        model_source="dummy",
                        leader_ip=leader,
                        node_index=rank,
                        smoke=False,
                        r3_enabled=bool(phase["r3_enabled"]),
                        max_num_batched_tokens=int(phase["max_num_batched_tokens"]),
                        max_num_seqs=int(phase["max_num_seqs"]),
                        chat_transport=True,
                    )
                )
                command.append("--aggregate-engine-logging")
                startup.append(
                    {
                        "rank": rank,
                        "command": command,
                        "environment": environment,
                        "alive": True,
                        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
                    }
                )
            ranks.append(
                {
                    "rank": rank,
                    "active": active,
                    "active_task_count": case.node_count,
                    "case": case.name,
                    "job_id": "job-1",
                    "marin_commit": provenance["marin_commit"],
                    "vllm_commit": provenance["vllm_commit"],
                    "task_image": provenance["task_image"],
                    "coscheduling": provenance["iris_coscheduling"],
                    "vllm_command": command,
                    "vllm_environment": environment if active else None,
                    "vllm_alive_before_stop": active,
                    "error": None,
                }
            )
        arm = {
            "passed": True,
            "matrix": {
                "phase_id": phase["phase_id"],
                "case": case.name,
                "role": phase["role"],
                "active_tasks": case.node_count,
                "routing_regime": phase["routing_regime"],
                "order": phase["order"],
                "replicate": phase["replicate"],
                "fresh_server": True,
                "same_iris_allocation": "job-1",
            },
            "settings": {
                "target_concurrency": phase["concurrencies"][0],
                "max_num_batched_tokens": phase["max_num_batched_tokens"],
                "max_num_seqs": phase["max_num_seqs"],
                "r3_enabled": phase["r3_enabled"],
                "request_transport": phase["request_transport"],
                "routing_regime": phase["routing_regime"],
                "route_audit_mode": phase["route_audit_mode"],
                "settings_drift": False,
            },
        }
        return {
            "phase": phase,
            "phase_id": phase["phase_id"],
            "model": case.name,
            "passed": True,
            "error": None,
            "placement": {
                "passed": True,
                "endpoints": [
                    {"task_index": str(rank), "advertise_host": leader if rank == 0 else f"10.0.0.{rank + 1}"}
                    for rank in range(case.node_count)
                ],
            },
            "startup": startup,
            "arms": [arm],
            "all_rank_health": {"passed": True, "ranks": ranks},
        }

    off_plan = grug_preflight._matrix_phase(
        "targeted-ep8-chat-r3off",
        case="reference-ep8",
        role="targeted-chat-r3",
        concurrencies=[96],
        max_num_batched_tokens=16384,
        request_transport="chat",
        r3_enabled=False,
    )
    on_plan = {**off_plan, "phase_id": "targeted-ep8-chat-r3on", "r3_enabled": True}
    off = build(off_plan)
    on = build(on_plan)

    assert grug_preflight._matrix_phase_evidence_contract(off, off_plan, provenance)["passed"]
    assert grug_preflight._matrix_phase_evidence_contract(on, on_plan, provenance)["passed"]
    assert grug_preflight._matrix_matched_chat_pair_contract(
        [off, on],
        off_phase_id="targeted-ep8-chat-r3off",
        on_phase_id="targeted-ep8-chat-r3on",
    )["passed"]

    on["all_rank_health"]["ranks"][0]["vllm_environment"] = {"unexpected": "drift"}
    assert not grug_preflight._matrix_matched_chat_pair_contract(
        [off, on],
        off_phase_id="targeted-ep8-chat-r3off",
        on_phase_id="targeted-ep8-chat-r3on",
    )["passed"]


def test_topology_summary_requires_stable_canonical_and_balanced_ep8_wins() -> None:
    def arm(
        phase_id: str,
        *,
        case: str,
        role: str,
        routing: str,
        order: str | None,
        rate: float,
        r3_enabled: bool = False,
    ) -> dict[str, object]:
        return {
            "arm_id": phase_id,
            "passed": True,
            "headline": {"generation_tokens_per_second_per_gpu": rate},
            "settings": {"r3_enabled": r3_enabled},
            "moe_routing": {
                "carrier": {
                    "json_bytes_per_engine_generation_token": 321.0 if r3_enabled else 0.0,
                }
            },
            "matrix": {
                "phase_id": phase_id,
                "case": case,
                "role": role,
                "routing_regime": routing,
                "order": order,
            },
        }

    arms = [
        arm(
            "canonical-ab-ep8",
            case="reference-ep8",
            role="topology-comparison",
            routing="canonical",
            order="ab",
            rate=110,
        ),
        arm(
            "canonical-ab-ep16",
            case="exact-reference-ep16",
            role="topology-comparison",
            routing="canonical",
            order="ab",
            rate=100,
        ),
        arm(
            "canonical-ba-ep16",
            case="exact-reference-ep16",
            role="topology-comparison",
            routing="canonical",
            order="ba",
            rate=101,
        ),
        arm(
            "canonical-ba-ep8",
            case="reference-ep8",
            role="topology-comparison",
            routing="canonical",
            order="ba",
            rate=111,
        ),
        arm(
            "balanced-ab-ep8", case="reference-ep8", role="topology-comparison", routing="balanced", order="ab", rate=120
        ),
        arm(
            "balanced-ab-ep16",
            case="exact-reference-ep16",
            role="topology-comparison",
            routing="balanced",
            order="ab",
            rate=105,
        ),
        arm(
            "balanced-ba-ep16",
            case="exact-reference-ep16",
            role="topology-comparison",
            routing="balanced",
            order="ba",
            rate=106,
        ),
        arm(
            "balanced-ba-ep8", case="reference-ep8", role="topology-comparison", routing="balanced", order="ba", rate=121
        ),
        arm(
            "audit-control-noop-ep16",
            case="exact-reference-ep16",
            role="audit-control",
            routing="canonical",
            order=None,
            rate=101,
        ),
        arm(
            "audit-control-record-ep16",
            case="exact-reference-ep16",
            role="audit-control",
            routing="canonical",
            order=None,
            rate=100.5,
        ),
        arm(
            "targeted-ep8-chat-r3off",
            case="reference-ep8",
            role="targeted-chat-r3",
            routing="canonical",
            order=None,
            rate=110,
        ),
        arm(
            "targeted-ep8-chat-r3on",
            case="reference-ep8",
            role="targeted-chat-r3",
            routing="canonical",
            order=None,
            rate=100,
            r3_enabled=True,
        ),
    ]

    summary = grug_preflight._matrix_topology_summary(arms)

    assert summary["passed"]
    assert summary["repeatability"]["passed"]
    assert summary["ep8_is_targeted_chat_r3_finalist"]
    assert summary["targeted_chat_r3"]["passed"]
    assert summary["targeted_chat_r3"]["r3_on_over_off_percent"] == pytest.approx(-9.0909090909)
    assert "Advance EP8" in summary["recommendation"]

    incomplete = grug_preflight._matrix_topology_summary(arms[:-2])
    assert incomplete["ep8_is_targeted_chat_r3_finalist"]
    assert not incomplete["targeted_chat_r3"]["passed"]
    assert not incomplete["passed"]
