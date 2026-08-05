#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen GrugMoE preflight on an allocated Iris GB200 gang.

Allocation and execution are separate on purpose: the expensive holder remains
visible to ``dev_gpu.py status`` and can always be released by
``dev_gpu.py release``. Once allocated, this is the intended one-command run:

    uv run scripts/iris/grugmoe_inference_preflight.py run \
      --session grugmoe-preflight --case reference-ep8 --run-id <id>
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import requests
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2

from experiments.grug.moe.inference_preflight import (
    ARTIFACT_ROOT,
    CAPACITY_FINAL_TOKENS,
    CASES,
    DUMMY_SEED,
    FROZEN_FIXTURE_PATH,
    IDENTITY_CHAT_TOKENS,
    SNOWBALL_EXPORT,
    VLLM_SHA,
    ModelCase,
    aggregate_preflight_status,
    decode_routed_experts,
    deterministic_balanced_routing_fixture,
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
from experiments.grug.moe.rolling_benchmark import (
    PlateauRequirements,
    PlateauWindow,
    frozen_cohort_slots,
    histogram_quantile_delta,
    parse_labeled_prometheus,
    prometheus_value,
    prometheus_values_by_label,
)
from scripts.iris.dev_gpu import (
    PRIORITY_BANDS,
    STATE_DIR,
    DevGpuState,
    PodRef,
    Priority,
    controller_client,
    kubectl_base,
    load_state,
    state_path,
)

VLLM_FROM_SPEC = f"vllm @ git+https://github.com/marin-community/vllm.git@{VLLM_SHA}"
RUNAI_STREAMER = "runai-model-streamer[s3]==0.16.1"
PYTHON_VERSION = "3.12"
RPC_PORT = 13345
SERVER_PORT = 8000
LOCAL_DP_SIZE = 4
PINNED_PROBE_DP_RANK = 0
SERVER_TIMEOUT_SECONDS = 3600
REQUEST_TIMEOUT_SECONDS = 3600
PREFIX_METRIC_TIMEOUT_SECONDS = 30
PREFIX_METRIC_POLL_SECONDS = 0.25
ACCEPTANCE_MINIMUM_SECONDS = 600
ACCEPTANCE_MINIMUM_GENERATED_TOKENS = 250_000
ACCEPTANCE_STABLE_MINUTES = 10
ACCEPTANCE_COUNTER_SAMPLE_SECONDS = 60.0
HEALTH_MINIMUM_SECONDS = 120
HEALTH_MINIMUM_GENERATED_TOKENS = 250_000
HEALTH_REPRESENTATIVE_WARM_UP_PASSES = 4
HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES = 1
CALIBRATION_CONCURRENCIES = (24, 48, 72, 96, 144)
CALIBRATION_MAX_NUM_BATCHED_TOKENS = (8192, 16384)
CALIBRATION_MAX_NUM_SEQS = 144
ATTENTION_CANDIDATES = ("window1024-ep16", "window2048-ep16", "global-every4-ep16")
ATTENTION_FINALISTS = ("exact-reference-ep16", *ATTENTION_CANDIDATES)
ATTENTION_ORDERS = ("ab", "ba")
MATRIX_PLANS = (
    "instrument-v1",
    "ep8-calibration",
    "topology-v1",
    "attention-pair-v1",
    "attention-finalist-v1",
)
HEALTH_ARTIFACT_ROOT = "s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-0"
TOPOLOGY_ARTIFACT_ROOT = "s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-1"
GLOBAL_CADENCE_ARTIFACT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-3"
)
SLIDING_WINDOW_ARTIFACT_ROOT = (
    "s3://marin-us-east-02a/marin/users/romain/inference-bench/grugmoe-architecture/experiment-4"
)
MARINSKYRL_CONSUMER_SHA = "96a026abda71c1b9fc53b06a7ff3c9f90a122d78"
HEALTH_SAMPLING_PARAMETERS = {
    "temperature": 0.0,
    "ignore_eos": True,
    "logprobs": 1,
    "return_token_ids": True,
    "return_tokens_as_token_ids": True,
}
CHAT_HEALTH_SAMPLING_PARAMETERS = {
    "temperature": 0.0,
    "ignore_eos": True,
    "logprobs": True,
    "top_logprobs": 1,
    "return_token_ids": True,
    "return_tokens_as_token_ids": True,
}
VLLM_SERVER_DEV_MODE_ENVIRONMENT = {"VLLM_SERVER_DEV_MODE": "1"}
REMOTE_ROOT = "/tmp/grugmoe-inference-preflight"
LOG_TAIL_LINES = 400
GLOO_CONTROL_INTERFACE = "enP6p3s0np0"
UNATTENDED_COSCHEDULING = "nvlink.domain"
DEFAULT_CLUSTER_CONFIG = "lib/iris/config/cw-us-east-08a.yaml"
FIXTURE_DIR = Path(FROZEN_FIXTURE_PATH)
AWS_CONFIG_CONTENT = "[default]\ns3 =\n    addressing_style = virtual\n"
REMOTE_UPLOAD_PROGRAM = """
import json
import pathlib
import sys

import s3fs

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2].removeprefix("s3://").rstrip("/")
relative_files = json.loads(sys.argv[3])
filesystem = s3fs.S3FileSystem()
records = []
for relative in relative_files:
    local_path = root / relative
    destination = f"{prefix}/{relative}"
    expected = local_path.read_bytes()
    filesystem.put_file(str(local_path), destination)
    readback = filesystem.cat_file(destination)
    if readback != expected:
        raise OSError(f"artifact readback mismatch for s3://{destination}")
    records.append(
        {
            "path": f"s3://{destination}",
            "bytes": len(expected),
            "readback": "identical",
        }
    )
print(json.dumps(records, sort_keys=True))
""".strip()


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=capture_output,
        check=check,
        env=env,
        timeout=timeout,
    )


def validate_session(state: DevGpuState, case: ModelCase) -> None:
    if state.gpu_variant != "GB200":
        raise ValueError(f"preflight requires GB200, got {state.gpu_variant}")
    if state.gpus_per_node != LOCAL_DP_SIZE:
        raise ValueError(f"preflight requires four GPUs per GB200 node, got {state.gpus_per_node}")
    if len(state.pods) != case.node_count:
        raise ValueError(
            f"{case.name} requires {case.node_count} nodes for EP{case.data_parallel_size}; "
            f"session {state.session_name!r} has {len(state.pods)}"
        )
    if state.priority.value != "interactive":
        raise ValueError(f"preflight requires Iris interactive priority, got {state.priority.value}")


def validate_acceptance_thresholds(*, minimum_seconds: float, minimum_generated_tokens: int) -> None:
    if minimum_seconds < ACCEPTANCE_MINIMUM_SECONDS:
        raise ValueError(f"acceptance requires at least {ACCEPTANCE_MINIMUM_SECONDS} seconds per arm")
    if minimum_generated_tokens < ACCEPTANCE_MINIMUM_GENERATED_TOKENS:
        raise ValueError(f"acceptance requires at least {ACCEPTANCE_MINIMUM_GENERATED_TOKENS} generated tokens per arm")


def validate_health_thresholds(*, minimum_seconds: float, minimum_generated_tokens: int) -> None:
    if minimum_seconds < HEALTH_MINIMUM_SECONDS:
        raise ValueError(f"rolling health requires at least {HEALTH_MINIMUM_SECONDS} plateau seconds per arm")
    if minimum_generated_tokens < HEALTH_MINIMUM_GENERATED_TOKENS:
        raise ValueError(
            f"rolling health requires at least {HEALTH_MINIMUM_GENERATED_TOKENS} engine generation tokens per arm"
        )


def vllm_args(
    case: ModelCase,
    *,
    model_dir: str,
    model_source: str,
    leader_ip: str,
    node_index: int,
    smoke: bool,
    r3_enabled: bool = True,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 64,
    chat_transport: bool = False,
) -> list[str]:
    if not 0 <= node_index < case.node_count:
        raise ValueError(f"node_index {node_index} is outside case node count {case.node_count}")
    if max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    fixture = model_source == "fixture"
    args = [
        "serve",
        model_dir,
        "--trust-remote-code",
        "--dtype",
        "half" if fixture else "bfloat16",
        "--kv-cache-dtype",
        "auto" if fixture else "bfloat16",
        "--seed",
        str(DUMMY_SEED),
        "--served-model-name",
        case.name,
        "--pipeline-parallel-size",
        "1",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        str(case.data_parallel_size),
        "--data-parallel-size-local",
        str(LOCAL_DP_SIZE if case.data_parallel_size > 1 else 1),
        "--data-parallel-start-rank",
        str(node_index * LOCAL_DP_SIZE),
        "--data-parallel-backend",
        "mp",
        "--data-parallel-address",
        leader_ip,
        "--data-parallel-rpc-port",
        str(RPC_PORT),
        "--enable-expert-parallel",
        "--expert-placement-strategy",
        "linear",
        "--moe-backend",
        "triton",
        "--attention-backend",
        "FLASH_ATTN",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-prompt-tokens-details",
        "--max-logprobs",
        "64",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        str(min(case.max_model_len, 2048) if smoke else case.max_model_len),
        "--max-num-batched-tokens",
        str(2048 if smoke else max_num_batched_tokens),
        "--max-num-seqs",
        str(8 if smoke else max_num_seqs),
    ]
    if r3_enabled:
        args.append("--enable-return-routed-experts")
    if model_source == "dummy":
        args.extend(["--load-format", "dummy"])
        if not chat_transport:
            args.append("--skip-tokenizer-init")
    elif model_source == "fixture":
        args.extend(["--load-format", "safetensors"])
        if not chat_transport:
            args.append("--skip-tokenizer-init")
    elif model_source == "snowball":
        args.extend(["--load-format", "runai_streamer"])
    else:
        raise ValueError(f"unknown model source: {model_source}")
    if node_index == 0:
        # A single API process keeps Prometheus counters coherent across the
        # cold/reuse/mutation sequence. The default is one process per local
        # DP rank, whose process-local metrics can be sampled inconsistently.
        args.extend(["--api-server-count", "1", "--host", "0.0.0.0", "--port", str(SERVER_PORT)])
    else:
        args.append("--headless")
    return args


def vllm_command(args: list[str]) -> list[str]:
    return [
        "uvx",
        "--no-config",
        "--from",
        VLLM_FROM_SPEC,
        "--with",
        RUNAI_STREAMER,
        "--python",
        PYTHON_VERSION,
        "--torch-backend",
        "cu130",
        "vllm",
        *args,
    ]


def _cuda_uv_environment(cache_dir: Path) -> dict[str, str]:
    """Keep CUDA source builds separate from workspace TPU wheels at the same Git SHA."""
    return {
        "UV_CACHE_DIR": str(cache_dir),
        "VLLM_TARGET_DEVICE": "cuda",
    }


@dataclasses.dataclass(frozen=True)
class PodRuntime:
    node_index: int
    pod: PodRef
    pod_ip: str
    image: str
    image_id: str
    environment: dict[str, str]
    command: list[str]
    remote_dir: str


class Gang:
    def __init__(
        self,
        state: DevGpuState,
        case: ModelCase,
        run_id: str,
        local_dir: Path,
        *,
        model_source: str,
        smoke: bool,
    ) -> None:
        self.state = state
        self.case = case
        self.run_id = run_id
        self.local_dir = local_dir
        self.model_source = model_source
        self.smoke = smoke
        self.remote_dir = f"{REMOTE_ROOT}/{run_id}"
        self.runtimes: list[PodRuntime] = []

    def _kubectl(self, *args: str) -> list[str]:
        return [*kubectl_base(self.state.target), *args]

    def _pod_json(self, pod: PodRef) -> dict[str, Any]:
        completed = _run(self._kubectl("get", "pod", pod.pod_name, "-o", "json"), capture_output=True)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise TypeError(f"unexpected pod payload for {pod.pod_name}")
        return payload

    def inspect(self) -> None:
        pod_payloads = [self._pod_json(pod) for pod in self.state.pods]
        leader_ip = str(pod_payloads[0]["status"]["podIP"])
        runtimes: list[PodRuntime] = []
        for node_index, (pod, payload) in enumerate(zip(self.state.pods, pod_payloads, strict=True)):
            statuses = payload["status"].get("containerStatuses", [])
            task_status = next((status for status in statuses if status.get("name") == pod.container), None)
            if task_status is None:
                raise RuntimeError(f"task container status missing for {pod.pod_name}")
            model_dir = {
                "dummy": self.remote_dir,
                "fixture": str(FIXTURE_DIR.resolve()),
                "snowball": SNOWBALL_EXPORT,
            }[self.model_source]
            args = vllm_args(
                self.case,
                model_dir=model_dir,
                model_source=self.model_source,
                leader_ip=leader_ip,
                node_index=node_index,
                smoke=self.smoke,
            )
            runtimes.append(
                PodRuntime(
                    node_index=node_index,
                    pod=pod,
                    pod_ip=str(payload["status"]["podIP"]),
                    image=str(task_status.get("image", "")),
                    image_id=str(task_status.get("imageID", "")),
                    environment={
                        # The Iris GB200 hostname resolves to 127.0.0.1.
                        # Gloo otherwise advertises loopback to remote ranks.
                        "AWS_CONFIG_FILE": f"{self.remote_dir}/aws-config",
                        "GLOO_SOCKET_IFNAME": GLOO_CONTROL_INTERFACE,
                        "VLLM_HOST_IP": str(payload["status"]["podIP"]),
                    },
                    command=vllm_command(args),
                    remote_dir=self.remote_dir,
                )
            )
        self.runtimes = runtimes

    def _exec(self, runtime: PodRuntime, *command: str) -> subprocess.CompletedProcess[str]:
        args = ["exec"]
        args.extend([runtime.pod.pod_name, "-c", runtime.pod.container, "--", *command])
        return _run(self._kubectl(*args), capture_output=True)

    def stage(self) -> None:
        for runtime in self.runtimes:
            self._exec(runtime, "test", "-e", f"/sys/class/net/{GLOO_CONTROL_INTERFACE}")
            self._exec(runtime, "mkdir", "-p", runtime.remote_dir)
            for filename in (
                "aws-config",
                "config.json",
                "correctness-workload.json",
                "workload.json",
                "manifest.json",
            ):
                source = str(self.local_dir / filename)
                destination = f"{runtime.pod.pod_name}:{runtime.remote_dir}/{filename}"
                _run(
                    self._kubectl("cp", source, destination, "-c", runtime.pod.container),
                    capture_output=True,
                )

    def start(self) -> None:
        def start_one(runtime: PodRuntime) -> None:
            log_path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.log"
            pid_path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.pid"
            command_text = shlex.join(runtime.command)
            environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(runtime.environment.items()))
            shell = (
                "export VLLM_USE_PRECOMPILED=1 VLLM_USE_FLASHINFER_SAMPLER=0 "
                f"PYTHONUNBUFFERED=1 {environment}; "
                f"setsid {command_text} </dev/null > {shlex.quote(log_path)} 2>&1 & "
                f"echo $! > {shlex.quote(pid_path)}"
            )
            self._exec(runtime, "bash", "-lc", shell)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.runtimes)) as executor:
            futures = [executor.submit(start_one, runtime) for runtime in self.runtimes]
            for future in futures:
                future.result()

    def stop(self) -> None:
        for runtime in self.runtimes:
            pid_path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.pid"
            shell = (
                f"if test -s {shlex.quote(pid_path)}; then "
                f"kill -TERM -- -$(cat {shlex.quote(pid_path)}) 2>/dev/null || true; fi"
            )
            try:
                self._exec(runtime, "bash", "-lc", shell)
            except Exception:
                pass

    def wait_for_leader_port(self, timeout_seconds: float) -> None:
        if not self.runtimes:
            raise RuntimeError("gang must be inspected before waiting for the server")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            states: list[str] = []
            for runtime in self.runtimes:
                pid_path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.pid"
                process_probe = remote_process_probe(pid_path)
                if runtime.node_index == 0:
                    shell = (
                        f"if (exec 3<>/dev/tcp/127.0.0.1/{SERVER_PORT}) 2>/dev/null; "
                        f"then echo ready; elif {process_probe}; "
                        "then echo running; else echo failed; fi"
                    )
                else:
                    shell = f"if {process_probe}; then echo running; else echo failed; fi"
                states.append(self._exec(runtime, "bash", "-lc", shell).stdout.strip())
            if states[0] == "ready":
                return
            failed_nodes = [index for index, state in enumerate(states) if state == "failed"]
            if failed_nodes:
                raise RuntimeError(f"vLLM exited before opening the API port on nodes {failed_nodes}")
            time.sleep(5)
        raise TimeoutError(f"vLLM did not open port {SERVER_PORT} within {timeout_seconds}s")

    def collect_logs(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for runtime in self.runtimes:
            log_path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.log"
            local_path = self.local_dir / f"vllm-node-{runtime.node_index}.log"
            try:
                completed = self._exec(runtime, "cat", log_path)
                local_path.write_text(completed.stdout)
                status = "collected"
            except Exception as exc:
                local_path.write_text(f"log collection failed: {exc!r}\n")
                status = "failed"
            records.append(
                {
                    **dataclasses.asdict(runtime),
                    "pod": dataclasses.asdict(runtime.pod),
                    "log_path": str(local_path),
                    "log_status": status,
                }
            )
        return records

    def failure_tail(self) -> str:
        tails: list[str] = []
        for runtime in self.runtimes:
            path = f"{runtime.remote_dir}/vllm-node-{runtime.node_index}.log"
            try:
                completed = self._exec(runtime, "tail", "-n", str(LOG_TAIL_LINES), path)
                tails.append(f"=== node {runtime.node_index} {runtime.pod.pod_name} ===\n{completed.stdout}")
            except Exception as exc:
                tails.append(f"=== node {runtime.node_index} ===\nfailed to read log: {exc!r}")
        return "\n".join(tails)

    def upload_and_readback(self, prefix: str, relative_files: list[str]) -> list[dict[str, Any]]:
        if not self.runtimes:
            raise RuntimeError("gang must be inspected before artifact upload")
        runtime = self.runtimes[0]
        artifact_root = f"{runtime.remote_dir}/artifacts"
        self._exec(runtime, "mkdir", "-p", artifact_root)
        for relative in relative_files:
            local_path = self.local_dir / relative
            if not local_path.is_file():
                raise FileNotFoundError(local_path)
            remote_path = f"{artifact_root}/{relative}"
            self._exec(runtime, "mkdir", "-p", str(Path(remote_path).parent))
            _run(
                self._kubectl(
                    "cp",
                    str(local_path),
                    f"{runtime.pod.pod_name}:{remote_path}",
                    "-c",
                    runtime.pod.container,
                ),
                capture_output=True,
            )
        completed = self._exec(
            runtime,
            "uv",
            "run",
            "--no-project",
            "--with",
            "s3fs",
            "python",
            "-c",
            REMOTE_UPLOAD_PROGRAM,
            artifact_root,
            prefix,
            json.dumps(relative_files),
        )
        records = json.loads(completed.stdout.strip().splitlines()[-1])
        if not isinstance(records, list) or len(records) != len(relative_files):
            raise RuntimeError(f"unexpected remote upload response: {completed.stdout!r}")
        return records


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def remote_process_probe(pid_path: str) -> str:
    """Shell predicate that rejects dead processes, including unreaped zombies."""
    quoted_path = shlex.quote(pid_path)
    return (
        f'test -s {quoted_path} && pid="$(cat {quoted_path} 2>/dev/null)" && test -n "$pid" '
        '&& test -r "/proc/$pid/stat" && read -r _ _ state _ < "/proc/$pid/stat" '
        '&& test "$state" != Z && test "$state" != X && kill -0 "$pid" 2>/dev/null'
    )


@contextmanager
def port_forward(state: DevGpuState, pod: PodRef):
    local_port = _free_port()
    command = [
        *kubectl_base(state.target),
        "port-forward",
        pod.pod_name,
        f"{local_port}:{SERVER_PORT}",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        yield local_port, process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def wait_for_server(base_url: str, process: subprocess.Popen[str], timeout_seconds: float) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"kubectl port-forward exited with {process.returncode}: {stderr}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.ok:
                models = response.json().get("data", [])
                if models and models[0].get("id"):
                    return str(models[0]["id"])
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise TimeoutError(f"server did not become ready within {timeout_seconds}s; last error: {last_error}")


def _metrics(base_url: str) -> tuple[str, dict[str, float]]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    return response.text, parse_prometheus(response.text)


def _wait_for_metric_delta(
    base_url: str,
    baseline: dict[str, float],
    metric: str,
    *,
    minimum_delta: float,
    timeout_seconds: float = PREFIX_METRIC_TIMEOUT_SECONDS,
    poll_seconds: float = PREFIX_METRIC_POLL_SECONDS,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Wait until Prometheus has exported the stats for one completed request."""
    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    while True:
        attempts += 1
        text, samples = _metrics(base_url)
        observed_delta = metric_delta(baseline, samples, metric)
        now = time.monotonic()
        synchronized = observed_delta >= minimum_delta
        if synchronized or now >= deadline:
            return (
                text,
                samples,
                {
                    "metric": metric,
                    "minimum_delta": minimum_delta,
                    "observed_delta": observed_delta,
                    "synchronized": synchronized,
                    "poll_attempts": attempts,
                    "elapsed_seconds": now - started,
                },
            )
        time.sleep(min(poll_seconds, deadline - now))


def _completion(
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    *,
    max_tokens: int = 4,
    data_parallel_rank: int | None = None,
    sampling_seed: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    payload, _ = _timed_completion(
        base_url,
        model,
        prompt_token_ids,
        max_tokens=max_tokens,
        data_parallel_rank=data_parallel_rank,
        sampling_seed=sampling_seed,
        request_id=request_id,
    )
    return payload


def _timed_completion(
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    *,
    max_tokens: int = 4,
    data_parallel_rank: int | None = None,
    sampling_seed: int | None = None,
    request_id: str | None = None,
    request_transport: str = "completion",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request_transport not in {"completion", "chat"}:
        raise ValueError(f"unknown request transport: {request_transport}")
    headers: dict[str, str] = {}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    if request_transport == "completion":
        endpoint = "/v1/completions"
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            **HEALTH_SAMPLING_PARAMETERS,
        }
    else:
        if any(type(token_id) is not int or not 0 <= token_id < 256 for token_id in prompt_token_ids):
            raise ValueError("the frozen identity chat tokenizer only accepts token IDs in [0, 256)")
        endpoint = "/v1/chat/completions"
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": " ".join(IDENTITY_CHAT_TOKENS[token_id] for token_id in prompt_token_ids)}
            ],
            "max_tokens": max_tokens,
            **CHAT_HEALTH_SAMPLING_PARAMETERS,
        }
    if sampling_seed is not None:
        body["seed"] = sampling_seed
    started = time.monotonic()
    response = requests.post(
        f"{base_url}{endpoint}",
        headers=headers or None,
        json=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    headers_received = time.monotonic()
    if not response.ok:
        raise RuntimeError(f"{request_transport} completion failed with {response.status_code}: {response.text[:4000]}")
    if hasattr(response, "content"):
        raw_response = response.content
        body_received = time.monotonic()
        payload = json.loads(raw_response)
    else:
        payload = response.json()
        body_received = time.monotonic()
        raw_response = json.dumps(payload, separators=(",", ":")).encode()
    decoded = time.monotonic()
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"expected one completion choice, got {payload!r}")
    response_headers = getattr(response, "headers", {})
    return payload, {
        "request_id": request_id,
        "request_transport": request_transport,
        "endpoint": endpoint,
        "data_parallel_rank": data_parallel_rank,
        "sampling_seed": sampling_seed,
        "started_at_monotonic_seconds": started,
        "completed_at_monotonic_seconds": decoded,
        "request_bytes": len(json.dumps(body, separators=(",", ":")).encode()),
        "response_bytes": len(raw_response),
        "seconds_to_response_headers": headers_received - started,
        "seconds_to_response_body": body_received - started,
        "response_body_transfer_seconds": body_received - headers_received,
        "seconds_to_decode": decoded - body_received,
        "client_e2e_seconds": decoded - started,
        "content_length": response_headers.get("content-length") if response_headers else None,
        "content_encoding": response_headers.get("content-encoding") if response_headers else None,
    }


def _choice(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["choices"][0]


def _cached_prompt_tokens(payload: dict[str, Any]) -> int:
    details = payload.get("usage", {}).get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        raise AssertionError("completion response omitted prompt cache details")
    return int(details["cached_tokens"])


def _assert_same_reuse(cold: dict[str, Any], reused: dict[str, Any]) -> None:
    cold_choice = _choice(cold)
    reused_choice = _choice(reused)
    if cold_choice.get("token_ids") != reused_choice.get("token_ids"):
        raise AssertionError("cold and reused requests generated different token IDs")
    cold_logprobs = (cold_choice.get("logprobs") or {}).get("token_logprobs")
    reused_logprobs = (reused_choice.get("logprobs") or {}).get("token_logprobs")
    if cold_logprobs != reused_logprobs:
        raise AssertionError("cold and reused requests returned different next-token logprobs")
    cold_routes = cold_choice.get("routed_experts")
    reused_routes = reused_choice.get("routed_experts")
    if not cold_routes or not reused_routes:
        raise AssertionError("routed experts were not returned")
    import numpy as np  # noqa: PLC0415

    np.testing.assert_array_equal(decode_routed_experts(cold_routes), decode_routed_experts(reused_routes))


def boundary_requests(
    workload: dict[str, Any],
    *,
    expected_cache_hit_alignment: int | None = None,
) -> list[dict[str, Any]]:
    """Choose one request at each required cache boundary."""
    cache_hit_alignment = int(workload.get("cache_hit_alignment", 0))
    if cache_hit_alignment <= 0:
        raise AssertionError("workload omits a positive hybrid cache-hit alignment")
    if expected_cache_hit_alignment is not None and cache_hit_alignment != expected_cache_hit_alignment:
        raise AssertionError(
            f"workload cache-hit alignment is {cache_hit_alignment}, expected {expected_cache_hit_alignment}"
        )
    selected: list[dict[str, Any]] = []
    for boundary in (cache_hit_alignment + 1, 513):
        request = next(
            (candidate for candidate in workload["requests"] if candidate["prefix_token_count"] == boundary),
            None,
        )
        if request is None:
            raise AssertionError(f"workload lacks required {boundary}-token prefix")
        selected.append(request)
    return selected


def _record_completion(
    payload: dict[str, Any],
    *,
    artifact_dir: Path,
    stem: str,
) -> dict[str, Any]:
    """Persist a full response and return the compact evidence index."""
    import numpy as np  # noqa: PLC0415

    evidence_dir = artifact_dir / "correctness"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    response_path = evidence_dir / f"{stem}.json"
    response_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    choice = _choice(payload)
    encoded_routes = choice.get("routed_experts")
    if not encoded_routes:
        raise AssertionError("completion response omitted routed experts")
    routes = decode_routed_experts(encoded_routes)
    routes_path = evidence_dir / f"{stem}-routed-experts.npy"
    np.save(routes_path, routes, allow_pickle=False)
    return {
        "response_path": response_path.relative_to(artifact_dir).as_posix(),
        "routed_experts_path": routes_path.relative_to(artifact_dir).as_posix(),
        "routed_experts_shape": list(routes.shape),
        "routed_experts_sha256": hashlib.sha256(routes.tobytes()).hexdigest(),
        "token_ids": choice.get("token_ids"),
        "token_logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
        "usage": payload.get("usage"),
        "finish_reason": choice.get("finish_reason"),
    }


def _record_metrics(
    text: str,
    *,
    artifact_dir: Path,
    stem: str,
    evidence_subdir: str = "correctness",
) -> str:
    evidence_dir = artifact_dir / evidence_subdir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evidence_dir / f"{stem}.prom"
    metrics_path.write_text(text)
    return metrics_path.relative_to(artifact_dir).as_posix()


def run_correctness(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    case: ModelCase,
    artifact_dir: Path,
) -> dict[str, Any]:
    cache_hit_alignment = hybrid_kv_cache_hit_alignment(case)
    candidates = boundary_requests(
        workload,
        expected_cache_hit_alignment=cache_hit_alignment,
    )
    before_text, overall_before = _metrics(base_url)
    before_path = _record_metrics(before_text, artifact_dir=artifact_dir, stem="metrics-before")
    probe_baseline = overall_before
    boundary_results: list[dict[str, Any]] = []
    histogram = [0] * case.num_experts
    for request in candidates:
        prompt_token_ids = materialize_prompt(workload, request)
        mutated_prompt_token_ids = materialize_prompt(workload, request, mutated=True)
        # Prefix caches belong to individual DP engines. Keep this triplet on
        # one engine so it measures reuse instead of the load balancer.
        cold = _completion(
            base_url,
            model,
            prompt_token_ids,
            data_parallel_rank=PINNED_PROBE_DP_RANK,
        )
        cold_evidence = _record_completion(cold, artifact_dir=artifact_dir, stem=f"{request['request_id']}-cold")
        after_cold_text, after_cold, cold_metric_export = _wait_for_metric_delta(
            base_url,
            probe_baseline,
            "vllm:prefix_cache_queries",
            minimum_delta=len(prompt_token_ids),
        )
        after_cold_path = _record_metrics(
            after_cold_text,
            artifact_dir=artifact_dir,
            stem=f"{request['request_id']}-metrics-after-cold",
        )
        reused = _completion(
            base_url,
            model,
            prompt_token_ids,
            data_parallel_rank=PINNED_PROBE_DP_RANK,
        )
        reused_evidence = _record_completion(
            reused,
            artifact_dir=artifact_dir,
            stem=f"{request['request_id']}-reused",
        )
        after_reuse_text, after_reuse, reuse_metric_export = _wait_for_metric_delta(
            base_url,
            after_cold,
            "vllm:prefix_cache_queries",
            minimum_delta=len(prompt_token_ids),
        )
        after_reuse_path = _record_metrics(
            after_reuse_text,
            artifact_dir=artifact_dir,
            stem=f"{request['request_id']}-metrics-after-reuse",
        )
        mutated = _completion(
            base_url,
            model,
            mutated_prompt_token_ids,
            data_parallel_rank=PINNED_PROBE_DP_RANK,
        )
        mutated_evidence = _record_completion(
            mutated,
            artifact_dir=artifact_dir,
            stem=f"{request['request_id']}-mutated",
        )
        after_mutated_text, after_mutated, mutation_metric_export = _wait_for_metric_delta(
            base_url,
            after_reuse,
            "vllm:prefix_cache_queries",
            minimum_delta=len(mutated_prompt_token_ids),
        )
        after_mutated_path = _record_metrics(
            after_mutated_text,
            artifact_dir=artifact_dir,
            stem=f"{request['request_id']}-metrics-after-mutated",
        )
        reuse_hits = metric_delta(after_cold, after_reuse, "vllm:prefix_cache_hits")
        mutation_hits = metric_delta(after_reuse, after_mutated, "vllm:prefix_cache_hits")
        reuse_cached_tokens = _cached_prompt_tokens(reused)
        mutated_cached_tokens = _cached_prompt_tokens(mutated)
        expected_cached_tokens = len(prompt_token_ids) // cache_hit_alignment * cache_hit_alignment
        routes = decode_routed_experts(_choice(reused)["routed_experts"])
        route_histogram = routing_histogram(routes, num_experts=case.num_experts)
        histogram = [left + right for left, right in zip(histogram, route_histogram, strict=True)]
        request_id = request["request_id"]
        boundary_result = {
            "request_id": request_id,
            "prefix_token_count": request["prefix_token_count"],
            "prompt_token_count": len(prompt_token_ids),
            "data_parallel_rank": PINNED_PROBE_DP_RANK,
            "cache_hit_alignment": cache_hit_alignment,
            "cold": cold_evidence,
            "reused": reused_evidence,
            "mutated": mutated_evidence,
            "reuse_prefix_hits": reuse_hits,
            "mutated_prefix_hits": mutation_hits,
            "expected_cached_prompt_tokens": expected_cached_tokens,
            "reuse_cached_prompt_tokens": reuse_cached_tokens,
            "mutated_cached_prompt_tokens": mutated_cached_tokens,
            "route_histogram": route_histogram,
            "metrics": {
                "after_cold": after_cold_path,
                "after_reuse": after_reuse_path,
                "after_mutated": after_mutated_path,
            },
            "metric_export": {
                "cold": cold_metric_export,
                "reused": reuse_metric_export,
                "mutated": mutation_metric_export,
            },
        }
        boundary_results.append(boundary_result)
        diagnostic_path = artifact_dir / "correctness" / f"{request_id}-checks.json"
        diagnostic_path.write_text(json.dumps(boundary_result, indent=2, sort_keys=True) + "\n")

        _assert_same_reuse(cold, reused)
        for phase, export in boundary_result["metric_export"].items():
            if not export["synchronized"]:
                raise AssertionError(
                    f"{phase} request metrics did not synchronize within {PREFIX_METRIC_TIMEOUT_SECONDS}s: {export}"
                )
        if reuse_hits != expected_cached_tokens:
            raise AssertionError(
                f"identical request produced {reuse_hits} prefix-cache hit tokens, expected {expected_cached_tokens}"
            )
        if mutation_hits != 0:
            raise AssertionError(f"mutated request unexpectedly reused prefix cache blocks: {mutation_hits}")
        if reuse_cached_tokens != expected_cached_tokens:
            raise AssertionError(
                f"identical request reported {reuse_cached_tokens} cached prompt tokens, "
                f"expected {expected_cached_tokens}"
            )
        if mutated_cached_tokens != 0:
            raise AssertionError(f"mutated request reported cached prompt tokens: {mutated_cached_tokens}")
        probe_baseline = after_mutated
    after_text, after = _metrics(base_url)
    after_path = _record_metrics(after_text, artifact_dir=artifact_dir, stem="metrics-after")
    ep_rank_histogram = expert_parallel_rank_histogram(histogram, ep_size=case.data_parallel_size)
    mean_rank_assignments = sum(ep_rank_histogram) / len(ep_rank_histogram)
    imbalance_triggered = any(count == 0 for count in histogram) or any(
        count > 2 * mean_rank_assignments for count in ep_rank_histogram
    )
    return {
        "passed": True,
        "cache_hit_alignment": cache_hit_alignment,
        "boundaries": boundary_results,
        "route_histogram": histogram,
        "ep_rank_histogram": ep_rank_histogram,
        "routing_balance": {
            "unused_experts": sum(count == 0 for count in histogram),
            "mean_ep_rank_assignments": mean_rank_assignments,
            "max_ep_rank_assignments": max(ep_rank_histogram),
            "balanced_control_triggered": imbalance_triggered,
            "balanced_control": (
                deterministic_balanced_routing_fixture(
                    num_experts=case.num_experts,
                    top_k=case.num_experts_per_tok,
                    ep_size=case.data_parallel_size,
                )
                if imbalance_triggered
                else None
            ),
        },
        "metrics_before": before_path,
        "metrics_after": after_path,
        "metric_deltas": {
            metric: metric_delta(overall_before, after, metric)
            for metric in (
                "vllm:prefix_cache_queries",
                "vllm:prefix_cache_hits",
                "vllm:prompt_tokens",
                "vllm:prompt_tokens_cached",
                "vllm:generation_tokens",
                "vllm:num_preemptions",
            )
        },
    }


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        raise ValueError("cannot summarize an empty latency sample")
    ordered = sorted(latencies)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _run_load_arm(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    max_model_len: int,
    minimum_seconds: float,
    minimum_generated_tokens: int,
    concurrency: int,
    counter_sample_seconds: float = ACCEPTANCE_COUNTER_SAMPLE_SECONDS,
) -> dict[str, Any]:
    if counter_sample_seconds <= 0:
        raise ValueError("counter sample interval must be positive")

    _, baseline_metrics = _metrics(base_url)
    started = time.monotonic()
    generated = 0
    request_count = 0
    latencies: list[float] = []
    cohort_tokens: dict[str, int] = {}
    workload_requests = workload["requests"]
    covered_request_ids: set[str] = set()
    generation_counter_samples = [
        {
            "elapsed_seconds": 0.0,
            "generation_tokens": metric_delta({}, baseline_metrics, "vllm:generation_tokens"),
        }
    ]
    next_counter_sample_at = started + counter_sample_seconds

    def one(request: dict[str, Any]) -> tuple[float, dict[str, Any], int, str]:
        prompt = materialize_prompt(workload, request)
        max_tokens = int(request["max_tokens"])
        if len(prompt) + max_tokens != int(request["final_token_count"]):
            raise AssertionError(f"{request['request_id']} does not have the frozen final length")
        if len(prompt) + max_tokens > max_model_len:
            raise AssertionError(f"{request['request_id']} exceeds max model length")
        request_started = time.monotonic()
        payload = _completion(base_url, model, prompt, max_tokens=max_tokens)
        return (
            time.monotonic() - request_started,
            payload,
            int(request["prefix_token_count"]),
            str(request["request_id"]),
        )

    def record(future: concurrent.futures.Future[tuple[float, dict[str, Any], int, str]]) -> None:
        nonlocal generated, request_count
        latency, payload, prefix_tokens, request_id = future.result()
        completion_tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
        if completion_tokens != int(workload_requests[0]["max_tokens"]):
            raise AssertionError(
                f"{request_id} generated {completion_tokens} tokens, expected {workload_requests[0]['max_tokens']}"
            )
        generated += completion_tokens
        request_count += 1
        covered_request_ids.add(request_id)
        latencies.append(latency)
        cohort = str(prefix_tokens)
        cohort_tokens[cohort] = cohort_tokens.get(cohort, 0) + completion_tokens

    def sample_generation_counter() -> None:
        nonlocal next_counter_sample_at
        while time.monotonic() >= next_counter_sample_at:
            _, samples = _metrics(base_url)
            generation_counter_samples.append(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "generation_tokens": metric_delta({}, samples, "vllm:generation_tokens"),
                }
            )
            next_counter_sample_at += counter_sample_seconds

    def drain(
        futures: list[concurrent.futures.Future[tuple[float, dict[str, Any], int, str]]],
    ) -> None:
        pending = set(futures)
        while pending:
            timeout = max(0.0, next_counter_sample_at - time.monotonic())
            done, pending = concurrent.futures.wait(
                pending,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                record(future)
            sample_generation_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # The first wave is a literal traversal of all 144 frozen branches.
        coverage_futures = [executor.submit(one, request) for request in workload_requests]
        drain(coverage_futures)

        request_index = 0
        while (
            time.monotonic() - started < minimum_seconds
            or generated < minimum_generated_tokens
            or len(covered_request_ids) != len(workload_requests)
        ):
            batch = [
                workload_requests[(request_index + offset) % len(workload_requests)] for offset in range(concurrency)
            ]
            request_index += len(batch)
            futures = [executor.submit(one, request) for request in batch]
            drain(futures)

    elapsed = time.monotonic() - started
    counter_intervals: list[dict[str, float | int]] = []
    for before, after in itertools.pairwise(generation_counter_samples):
        interval_seconds = float(after["elapsed_seconds"]) - float(before["elapsed_seconds"])
        interval_tokens = round(float(after["generation_tokens"]) - float(before["generation_tokens"]))
        if interval_seconds <= 0:
            raise AssertionError("generation counter sample times must increase")
        if interval_tokens < 0:
            raise AssertionError("generation counter reset during a load arm")
        counter_intervals.append(
            {
                "start_elapsed_seconds": float(before["elapsed_seconds"]),
                "end_elapsed_seconds": float(after["elapsed_seconds"]),
                "elapsed_seconds": interval_seconds,
                "generated_tokens": interval_tokens,
            }
        )
    stable_intervals = counter_intervals[-ACCEPTANCE_STABLE_MINUTES:]
    stable_minute_tokens = [int(interval["generated_tokens"]) for interval in stable_intervals]
    stable_minutes_passed = len(stable_minute_tokens) == ACCEPTANCE_STABLE_MINUTES and all(
        tokens > 0 for tokens in stable_minute_tokens
    )
    stable_window_seconds = sum(float(interval["elapsed_seconds"]) for interval in stable_intervals)
    stable_mean = sum(stable_minute_tokens) / stable_window_seconds if stable_window_seconds else 0.0
    coverage_passed = len(covered_request_ids) == len(workload_requests)
    return {
        "passed": (
            elapsed >= minimum_seconds
            and generated >= minimum_generated_tokens
            and coverage_passed
            and stable_minutes_passed
        ),
        "elapsed_seconds": elapsed,
        "stable_full_minutes": len(stable_minute_tokens),
        "stable_minutes_passed": stable_minutes_passed,
        "generated_tokens": generated,
        "requests": request_count,
        "concurrency": concurrency,
        "branch_coverage": {
            "expected": len(workload_requests),
            "observed": len(covered_request_ids),
            "request_ids": sorted(covered_request_ids),
            "passed": coverage_passed,
        },
        "throughput_tokens_per_second": {
            "full_mean": generated / elapsed,
            "last_ten_stable_minute_mean": stable_mean,
        },
        "full_minute_generated_tokens": [int(interval["generated_tokens"]) for interval in counter_intervals],
        "last_ten_stable_minute_generated_tokens": stable_minute_tokens,
        "generation_counter": {
            "metric": "vllm:generation_tokens",
            "sample_interval_seconds": counter_sample_seconds,
            "samples": generation_counter_samples,
            "full_intervals": counter_intervals,
            "stable_window_seconds": stable_window_seconds,
        },
        "generated_tokens_by_prefix_length": cohort_tokens,
        "latency_seconds": _latency_summary(latencies),
    }


def run_acceptance_load(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    artifact_dir: Path,
    max_model_len: int,
    minimum_seconds: float,
    minimum_generated_tokens: int,
    concurrency: int = 64,
) -> dict[str, Any]:
    workload_requests = workload["requests"]
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    metric_names = (
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_hits",
        "vllm:prompt_tokens",
        "vllm:prompt_tokens_cached",
        "vllm:generation_tokens",
        "vllm:num_preemptions",
    )
    before_text, before = _metrics(base_url)
    metric_paths = [
        _record_metrics(
            before_text,
            artifact_dir=artifact_dir,
            stem="metrics-before-warm",
            evidence_subdir="load",
        )
    ]
    warm_started = time.monotonic()
    warm_generated = 0

    def warm_one(request: dict[str, Any]) -> int:
        prompt = materialize_prompt(workload, request)
        max_tokens = min(int(request["max_tokens"]), max_model_len - len(prompt))
        if max_tokens <= 0:
            raise AssertionError(f"{request['request_id']} leaves no warmup generation capacity")
        payload = _completion(base_url, model, prompt, max_tokens=max_tokens)
        return int(payload.get("usage", {}).get("completion_tokens", 0))

    # Populate one branch from every root before the concurrent warm wave. If
    # all sibling branches start together, none is guaranteed to find a
    # completed parent prefix in the cache.
    root_leaders = [request for request in workload_requests if int(request["branch"]) == 0]
    for request in root_leaders:
        warm_generated += warm_one(request)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(warm_one, request) for request in workload_requests]
        for future in concurrent.futures.as_completed(futures):
            warm_generated += future.result()
    warm_elapsed = time.monotonic() - warm_started
    after_warm_text, after_warm = _metrics(base_url)
    metric_paths.append(
        _record_metrics(
            after_warm_text,
            artifact_dir=artifact_dir,
            stem="metrics-after-warm",
            evidence_subdir="load",
        )
    )

    arms: list[dict[str, Any]] = []
    previous_metrics = after_warm
    for arm_index in range(2):
        arm = _run_load_arm(
            base_url,
            model,
            workload,
            max_model_len=max_model_len,
            minimum_seconds=minimum_seconds,
            minimum_generated_tokens=minimum_generated_tokens,
            concurrency=concurrency,
        )
        after_arm_text, after_arm = _metrics(base_url)
        metric_paths.append(
            _record_metrics(
                after_arm_text,
                artifact_dir=artifact_dir,
                stem=f"metrics-after-arm-{arm_index + 1}",
                evidence_subdir="load",
            )
        )
        arm["metric_deltas"] = {metric: metric_delta(previous_metrics, after_arm, metric) for metric in metric_names}
        arms.append(arm)
        previous_metrics = after_arm

    arm_throughputs = [arm["throughput_tokens_per_second"]["last_ten_stable_minute_mean"] for arm in arms]
    throughput_mean = sum(arm_throughputs) / len(arm_throughputs)
    repeatability_delta_percent = (
        100 * abs(arm_throughputs[0] - arm_throughputs[1]) / throughput_mean if throughput_mean else math.inf
    )
    return {
        "passed": warm_generated > 0 and all(arm["passed"] for arm in arms) and repeatability_delta_percent <= 2.0,
        "warm_populate": {
            "passed": warm_generated > 0,
            "elapsed_seconds": warm_elapsed,
            "requests": len(root_leaders) + len(workload_requests),
            "generated_tokens": warm_generated,
            "covered_prefix_lengths": sorted({int(request["prefix_token_count"]) for request in workload_requests}),
        },
        "metrics": {
            "snapshots": metric_paths,
            "warm_deltas": {metric: metric_delta(before, after_warm, metric) for metric in metric_names},
        },
        "arms": arms,
        "repeatability": {
            "throughput_delta_percent": repeatability_delta_percent,
            "limit_percent": 2.0,
            "passed": repeatability_delta_percent <= 2.0,
        },
    }


KV_LOG_MARKER = "GrugMoE KV group usage: "


def run_fixture_parity(base_url: str, model: str, *, artifact_dir: Path) -> dict[str, Any]:
    """Run the frozen tensor oracle in the pinned vLLM env and server parity live."""
    tensor_path = artifact_dir / "fixture-tensor-parity.json"
    command = [
        "uv",
        "run",
        "--no-config",
        "--prerelease=allow",
        "--no-project",
        "--with",
        VLLM_FROM_SPEC,
        "--with",
        RUNAI_STREAMER,
        "python",
        "tests/cluster/vllm/grug_exact_reference_check.py",
        "--fixture",
        str(FIXTURE_DIR),
        "--base-url",
        base_url,
        "--model",
        model,
        "--output",
        str(tensor_path),
    ]
    environment = dict(os.environ)
    environment.update(_cuda_uv_environment(artifact_dir.with_name(f"{artifact_dir.name}-cuda-uv-cache")))
    environment["UV_TORCH_BACKEND"] = "cu130"
    stdout_path = artifact_dir / "fixture-tensor-parity.stdout"
    stderr_path = artifact_dir / "fixture-tensor-parity.stderr"
    try:
        completed = _run(
            command,
            capture_output=True,
            env=environment,
            timeout=SERVER_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        stdout_path.write_text(exc.stdout or "")
        stderr_path.write_text(exc.stderr or "")
        raise
    tensor_payload = json.loads(tensor_path.read_text())
    if completed.stdout:
        stdout_path.write_text(completed.stdout)
    if completed.stderr:
        stderr_path.write_text(completed.stderr)

    server = tensor_payload["server"]
    server_path = artifact_dir / "fixture-server-parity.json"
    server_path.write_text(json.dumps(server, indent=2, sort_keys=True) + "\n")
    tensor = tensor_payload["tensor"]
    return {
        "passed": bool(tensor["passed"] and server["passed"]),
        "tensor": tensor,
        "server": server,
        "evidence": [
            tensor_path.relative_to(artifact_dir).as_posix(),
            server_path.relative_to(artifact_dir).as_posix(),
        ],
    }


def parse_kv_group_snapshots(text: str) -> list[list[dict[str, Any]]]:
    """Extract complete compact JSON snapshots from a vLLM log."""
    snapshots: list[list[dict[str, Any]]] = []
    for line in text.splitlines():
        if KV_LOG_MARKER not in line:
            continue
        raw = line.split(KV_LOG_MARKER, 1)[1].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list) and all(isinstance(group, dict) for group in payload):
            snapshots.append(payload)
    return snapshots


def summarize_kv_snapshot(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one multi-engine snapshot into semantic and physical occupancy."""
    active_requests_by_engine: dict[int, int] = {}
    reserved_by_engine: dict[int, int] = {}
    for group in groups:
        engine = int(group.get("engine_idx", 0))
        active_requests_by_engine[engine] = max(
            active_requests_by_engine.get(engine, 0),
            int(group["active_requests"]),
        )
        reserved_by_engine[engine] = max(
            reserved_by_engine.get(engine, 0),
            int(group["reserved_physical_bytes"]),
        )
    active_groups = [group for group in groups if int(group["active_requests"]) > 0]
    # Managers share one BlockPool, but each group owns distinct allocated
    # block IDs. Each such ID occupies a full physical tuple spanning all
    # layer tensors, so group physical occupancies add.
    active_physical_bytes = sum(int(group["active_physical_bytes"]) for group in active_groups)
    semantic_bytes = sum(int(group["active_payload_bytes"]) for group in active_groups)
    padded_group_bytes = sum(int(group["active_padded_bytes"]) for group in active_groups)
    attention_groups = [group for group in active_groups if group["role"] == "attention"]
    sconv_groups = [group for group in active_groups if group["role"] == "sconv"]

    def max_blocks(*, role: str, kind: str | None = None) -> int:
        return max(
            (
                int(group["active_blocks"])
                for group in active_groups
                if group["role"] == role and (kind is None or group["kind"] == kind)
            ),
            default=0,
        )

    return {
        "active_requests": sum(active_requests_by_engine.values()),
        "active_requests_by_engine": {
            str(engine): active_requests_by_engine[engine] for engine in sorted(active_requests_by_engine)
        },
        "semantic_active_bytes": semantic_bytes,
        "semantic_attention_active_bytes": sum(int(group["active_payload_bytes"]) for group in attention_groups),
        "semantic_sconv_active_bytes": sum(int(group["active_payload_bytes"]) for group in sconv_groups),
        "padded_group_active_bytes": padded_group_bytes,
        "padded_attention_active_bytes": sum(int(group["active_padded_bytes"]) for group in attention_groups),
        "padded_sconv_active_bytes": sum(int(group["active_padded_bytes"]) for group in sconv_groups),
        "physical_active_bytes": active_physical_bytes,
        "padding_active_bytes": padded_group_bytes - semantic_bytes,
        "reserved_physical_bytes_per_engine": {
            str(engine): reserved_by_engine[engine] for engine in sorted(reserved_by_engine)
        },
        "reserved_physical_bytes_global": sum(reserved_by_engine.values()),
        "local_attention_active_blocks": max_blocks(
            role="attention",
            kind="sliding_window",
        ),
        "global_attention_active_blocks": max_blocks(
            role="attention",
            kind="full_attention",
        ),
        "sconv_active_blocks": max_blocks(role="sconv"),
        "groups": groups,
    }


def _metric_value(samples: dict[str, float], metric: str) -> float:
    if metric in samples:
        return samples[metric]
    return samples.get(f"{metric}_total", 0.0)


def run_kv_measurement(
    base_url: str,
    model: str,
    *,
    case: ModelCase,
    log_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Measure one live request at 6,144 and 65,536 tokens with caching on."""
    if case.hidden_size != 6144 or case.num_hidden_layers != 48:
        raise ValueError("live KV measurement requires the exact reference architecture")
    targets = (6_144, 65_536)
    response_tokens = 2_048
    observations: list[dict[str, Any]] = []
    for probe_index, final_length in enumerate(targets):
        prompt_length = final_length - response_tokens
        prompt = [3 + ((probe_index * 97 + token_index * 31) % 252) for token_index in range(prompt_length)]
        log_start = log_path.stat().st_size if log_path.exists() else 0
        running_samples: list[float] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Keep both context probes on one cache-owning engine.
            future = executor.submit(
                _completion,
                base_url,
                model,
                prompt,
                max_tokens=response_tokens,
                data_parallel_rank=PINNED_PROBE_DP_RANK,
            )
            while not future.done():
                try:
                    _, metrics = _metrics(base_url)
                    running = _metric_value(metrics, "vllm:num_requests_running")
                    if running > 0:
                        running_samples.append(running)
                except requests.RequestException:
                    pass
                time.sleep(2)
            payload = future.result()
        completion_tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
        if completion_tokens != response_tokens:
            raise AssertionError(f"KV probe at {final_length} ended with {completion_tokens} generated tokens")
        if not running_samples or set(running_samples) != {1.0}:
            raise AssertionError(f"KV probe at {final_length} did not hold one active request: {running_samples}")
        log_text = log_path.read_bytes()[log_start:].decode(errors="replace")
        raw_path = artifact_dir / "kv" / f"kv-{final_length}.log"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(log_text)
        summaries = [summarize_kv_snapshot(snapshot) for snapshot in parse_kv_group_snapshots(log_text)]
        active = [summary for summary in summaries if summary["active_requests"] == 1]
        if not active:
            raise AssertionError(f"KV probe at {final_length} produced no one-request group snapshot")
        peak = max(
            active,
            key=lambda summary: (
                summary["global_attention_active_blocks"],
                summary["semantic_active_bytes"],
            ),
        )
        layer_schedule = layer_types(
            case.num_hidden_layers,
            global_interval=case.global_every,
        )
        global_layers = layer_schedule.count("full_attention")
        semantic_prediction = predict_kv_bytes(
            sequence_length=final_length,
            local_layers=case.num_hidden_layers - global_layers,
            global_layers=global_layers,
            local_kv_heads=case.local_kv_heads,
            global_kv_heads=case.global_kv_heads,
            head_dim=case.head_dim,
            sliding_window=case.sliding_window,
        )
        attention_prediction_delta_fraction = (
            abs(peak["semantic_attention_active_bytes"] - semantic_prediction) / semantic_prediction
            if semantic_prediction
            else math.inf
        )
        gap_fraction = (
            (peak["physical_active_bytes"] - peak["semantic_active_bytes"]) / peak["semantic_active_bytes"]
            if peak["semantic_active_bytes"]
            else math.inf
        )
        observations.append(
            {
                "final_sequence_tokens": final_length,
                "data_parallel_rank": PINNED_PROBE_DP_RANK,
                "prompt_tokens": prompt_length,
                "generated_tokens": completion_tokens,
                "fixed_active_request_count": 1,
                "semantic_attention_prediction_bytes": semantic_prediction,
                "semantic_attention_prediction_delta_fraction": attention_prediction_delta_fraction,
                "peak": peak,
                "physical_minus_semantic_fraction": gap_fraction,
                "gap_explanation": (
                    "The physical block is a unified page spanning every KV/SConv "
                    "group. A request occupies the largest group block index across "
                    "that tuple; per-group head padding is reported separately."
                    if gap_fraction > 0.10
                    else "Physical occupancy is within 10% of semantic group payload."
                ),
                "raw_log": raw_path.relative_to(artifact_dir).as_posix(),
                "snapshot_count": len(active),
            }
        )

    short, long = observations
    local_plateau = short["peak"]["local_attention_active_blocks"] == long["peak"]["local_attention_active_blocks"] > 0
    global_growth = long["peak"]["global_attention_active_blocks"] > short["peak"]["global_attention_active_blocks"] > 0
    semantic_predictions_match = all(
        observation["semantic_attention_prediction_delta_fraction"] <= 0.10 for observation in observations
    )
    result = {
        "passed": local_plateau and global_growth and semantic_predictions_match,
        "prefix_caching_enabled": True,
        "fixed_active_request_count": 1,
        "observations": observations,
        "local_layer_active_kv_plateaus": local_plateau,
        "global_layer_active_kv_grows": global_growth,
        "semantic_attention_predictions_within_10_percent": semantic_predictions_match,
    }
    output_path = artifact_dir / "kv" / "summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _s3_key(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri}")
    return uri.removeprefix("s3://")


def _s3_filesystem():
    import s3fs  # noqa: PLC0415

    return s3fs.S3FileSystem()


def _put_bytes_readback(filesystem: Any, uri: str, payload: bytes) -> dict[str, Any]:
    key = _s3_key(uri)
    with filesystem.open(key, "wb") as stream:
        stream.write(payload)
    readback = filesystem.cat_file(key)
    if readback != payload:
        raise OSError(f"artifact readback mismatch for {uri}")
    return {
        "path": uri,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "readback": "identical",
    }


def _put_json_readback(filesystem: Any, uri: str, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return _put_bytes_readback(filesystem, uri, encoded)


def _read_s3_json(filesystem: Any, uri: str) -> dict[str, Any]:
    payload = json.loads(filesystem.cat_file(_s3_key(uri)))
    if not isinstance(payload, dict):
        raise TypeError(f"{uri} did not contain a JSON object")
    return payload


def _wait_for_s3_jsons(
    filesystem: Any,
    uris: list[str],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    missing = list(uris)
    while time.monotonic() < deadline:
        missing = [uri for uri in uris if not filesystem.exists(_s3_key(uri))]
        if not missing:
            return [_read_s3_json(filesystem, uri) for uri in uris]
        time.sleep(5)
    raise TimeoutError(f"timed out waiting for S3 rendezvous files: {missing}")


def _immutable_image(image: str) -> str:
    if "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[1]) != 64:
        raise ValueError("--task-image must be an immutable image@sha256:<64 hex> reference")
    return image


def _clean_pushed_checkout() -> dict[str, str]:
    status = _run(["git", "status", "--porcelain"], capture_output=True).stdout
    if status:
        raise RuntimeError("unattended evidence may only be submitted from a clean checkout")
    head = _git_sha()
    upstream = _run(["git", "rev-parse", "@{upstream}"], capture_output=True).stdout.strip()
    if upstream != head:
        raise RuntimeError(f"HEAD {head} is not the pushed upstream commit {upstream}")
    return {
        "commit": head,
        "branch": (
            _run(
                ["git", "branch", "--show-current"],
                capture_output=True,
            ).stdout.strip()
        ),
        "origin": (
            _run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
            ).stdout.strip()
        ),
    }


def _validate_unattended_mode(
    case: ModelCase,
    *,
    mode: str,
    model_source: str,
) -> None:
    if mode == "acceptance" and case.name != "exact-reference-ep16":
        raise ValueError("acceptance is frozen to exact-reference-ep16")
    if mode == "kv" and case.name != "reference-ep8":
        raise ValueError("the exact live KV measurement uses reference-ep8")
    if mode == "health" and case.name != "exact-reference-ep16":
        raise ValueError("rolling health is frozen to exact-reference-ep16")
    if mode == "health" and model_source != "dummy":
        raise ValueError("experiment 0 rolling health requires dummy weights")
    if model_source == "fixture" and case.name != "tiny":
        raise ValueError("the frozen tensor fixture uses the tiny exact case")
    if model_source == "snowball":
        raise ValueError("the one allowed Snowball attempt has already completed; it must not be retried")


def _unattended_worker_argv(
    args: argparse.Namespace,
    *,
    case: ModelCase,
    run_id: str,
    image: str,
    marin_commit: str,
    coscheduling: CoschedulingConfig | None,
) -> list[str]:
    argv = [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "worker",
        "--case",
        case.name,
        "--model-source",
        args.model_source,
        "--mode",
        args.mode,
        "--run-id",
        run_id,
        "--task-image",
        image,
        "--marin-commit",
        marin_commit,
        "--iris-priority",
        args.priority,
        "--server-timeout",
        str(args.server_timeout),
        "--minimum-seconds",
        str(args.minimum_seconds),
        "--minimum-generated-tokens",
        str(args.minimum_generated_tokens),
    ]
    if args.mode == "health":
        concurrencies = args.concurrency or [48]
        max_num_seqs = args.max_num_seqs or max(concurrencies)
        argv.extend(
            [
                "--r3",
                args.r3,
                "--request-transport",
                args.request_transport,
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--max-num-seqs",
                str(max_num_seqs),
            ]
        )
        for concurrency in concurrencies:
            argv.extend(["--concurrency", str(concurrency)])
    if coscheduling is not None:
        argv.extend(["--submitted-coscheduling", coscheduling.group_by])
    return argv


def submit_unattended(args: argparse.Namespace) -> dict[str, Any]:
    case = CASES[args.case]
    _validate_unattended_mode(
        case,
        mode=args.mode,
        model_source=args.model_source,
    )
    if args.mode == "acceptance":
        validate_acceptance_thresholds(
            minimum_seconds=args.minimum_seconds,
            minimum_generated_tokens=args.minimum_generated_tokens,
        )
    if args.mode == "health":
        validate_health_thresholds(
            minimum_seconds=args.minimum_seconds,
            minimum_generated_tokens=args.minimum_generated_tokens,
        )
        concurrencies = args.concurrency or [48]
        if any(concurrency <= 0 or concurrency % 3 for concurrency in concurrencies):
            raise ValueError("every health concurrency must be positive and divisible by three")
        if args.max_num_batched_tokens <= 0:
            raise ValueError("health max_num_batched_tokens must be positive")
        if args.max_num_seqs is not None and args.max_num_seqs <= 0:
            raise ValueError("health max_num_seqs must be positive")
    checkout = _clean_pushed_checkout()
    image = _immutable_image(args.task_image)
    priority = Priority(args.priority)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    coscheduling = CoschedulingConfig(group_by=UNATTENDED_COSCHEDULING) if case.node_count > 1 else None
    worker_argv = _unattended_worker_argv(
        args,
        case=case,
        run_id=run_id,
        image=image,
        marin_commit=checkout["commit"],
        coscheduling=coscheduling,
    )
    resources = ResourceSpec(
        cpu=64,
        memory="512GB",
        disk="100GB",
        device=gpu_device("GB200", LOCAL_DP_SIZE),
    )
    with controller_client(args.config) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*worker_argv),
            name=f"grugmoe-{args.mode}-{case.name}-{run_id}".lower().replace("_", "-"),
            resources=resources,
            environment=EnvironmentSpec(
                sync_packages=["marin-iris", "marin-core"],
                env_vars={"PYTHONUNBUFFERED": "1"},
            ),
            replicas=case.node_count,
            coscheduling=coscheduling,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            task_image=image,
            priority_band=PRIORITY_BANDS[priority],
        )
        summary: dict[str, Any] = {
            "status": "submitted",
            "job_id": str(job.job_id),
            "run_id": run_id,
            "case": case.name,
            "mode": args.mode,
            "replicas": case.node_count,
            "coscheduling": coscheduling.group_by if coscheduling is not None else None,
            "task_image": image,
            "priority": priority.value,
            "checkout": checkout,
            "artifact_prefix": (
                f"{HEALTH_ARTIFACT_ROOT}/{run_id}/"
                if args.mode == "health"
                else f"{ARTIFACT_ROOT}/{case.name}/{run_id}/"
            ),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.wait:
            status = job.wait(
                timeout=args.wait_timeout,
                poll_interval=10,
                raise_on_failure=False,
                stream_logs=True,
            )
            summary["terminal_job_state"] = int(status.state)
            summary["terminal_job_state_name"] = job_pb2.JobState.Name(status.state)
            summary["terminal_job_succeeded"] = status.state == job_pb2.JOB_STATE_SUCCEEDED
            summary["terminal_error"] = status.error
            summary["status"] = summary["terminal_job_state_name"].removeprefix("JOB_STATE_").lower()
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def _wait_for_rank_endpoints(
    name: str,
    *,
    expected: int,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    deadline = time.monotonic() + timeout_seconds
    latest: list[dict[str, str]] = []
    while time.monotonic() < deadline:
        resolved = iris_ctx().resolver.resolve(name)
        latest = [
            {
                "url": endpoint.url,
                "actor_id": endpoint.actor_id,
                **endpoint.metadata,
            }
            for endpoint in resolved.endpoints
        ]
        indexes = {int(endpoint["task_index"]) for endpoint in latest}
        if len(latest) == expected and indexes == set(range(expected)):
            return sorted(latest, key=lambda endpoint: int(endpoint["task_index"]))
        time.sleep(2)
    raise TimeoutError(f"Iris rendezvous {name!r} found {len(latest)}/{expected} endpoints: {latest}")


@dataclasses.dataclass
class LocalVllm:
    process: subprocess.Popen[str]
    log_stream: TextIO
    log_path: Path
    command: list[str]
    provenance_environment: dict[str, str]


def _start_local_vllm(
    *,
    case: ModelCase,
    model_source: str,
    model_dir: str,
    leader_ip: str,
    node_index: int,
    smoke: bool,
    local_dir: Path,
    r3_enabled: bool = True,
    max_num_batched_tokens: int = 8192,
    max_num_seqs: int = 64,
    enable_dev_endpoints: bool = False,
    aggregate_engine_logging: bool = False,
    request_transport: str = "completion",
    routing_regime: str = "canonical",
    route_audit_mode: str | None = None,
) -> LocalVllm:
    if request_transport not in {"completion", "chat"}:
        raise ValueError(f"unknown request transport: {request_transport}")
    if routing_regime not in {"canonical", "balanced"}:
        raise ValueError(f"unknown routing regime: {routing_regime}")
    if route_audit_mode not in {None, "noop", "record"}:
        raise ValueError(f"unknown route-audit mode: {route_audit_mode}")
    if r3_enabled and route_audit_mode is not None:
        raise ValueError("R3 response capture and aggregate route audit are mutually exclusive")
    interface = Path(f"/sys/class/net/{GLOO_CONTROL_INTERFACE}")
    if not interface.exists():
        raise RuntimeError(f"required Gloo interface is absent: {interface}")
    command = vllm_command(
        vllm_args(
            case,
            model_dir=model_dir,
            model_source=model_source,
            leader_ip=leader_ip,
            node_index=node_index,
            smoke=smoke,
            r3_enabled=r3_enabled,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            chat_transport=request_transport == "chat",
        )
    )
    if aggregate_engine_logging:
        command.append("--aggregate-engine-logging")
    log_path = local_dir / f"vllm-node-{node_index}.log"
    log_stream = log_path.open("w")
    # vLLM exposes prefix-cache reset through its development router. Health
    # workers run in isolated jobs and need that route between frozen arms.
    provenance_environment = dict(VLLM_SERVER_DEV_MODE_ENVIRONMENT) if enable_dev_endpoints else {}
    if routing_regime == "balanced":
        provenance_environment["VLLM_GRUGMOE_ROUTING_FIXTURE"] = "balanced"
    if route_audit_mode is not None:
        provenance_environment["VLLM_GRUGMOE_ROUTE_AUDIT"] = route_audit_mode
    environment = {
        **os.environ,
        **_cuda_uv_environment(local_dir.with_name(f"{local_dir.name}-cuda-uv-cache")),
        **provenance_environment,
        "AWS_CONFIG_FILE": str(local_dir / "aws-config"),
        "GLOO_SOCKET_IFNAME": GLOO_CONTROL_INTERFACE,
        "PYTHONUNBUFFERED": "1",
        "VLLM_HOST_IP": get_job_info().advertise_host,
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_USE_PRECOMPILED": "1",
    }
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    return LocalVllm(
        process=process,
        log_stream=log_stream,
        log_path=log_path,
        command=command,
        provenance_environment=provenance_environment,
    )


def _stop_local_vllm(server: LocalVllm) -> None:
    if server.process.poll() is None:
        os.killpg(server.process.pid, signal.SIGTERM)
        try:
            server.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.process.pid, signal.SIGKILL)
            server.process.wait(timeout=30)
    server.log_stream.close()


def _wait_for_local_server(
    base_url: str,
    server: LocalVllm,
    *,
    timeout_seconds: float,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            tail = server.log_path.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
            raise RuntimeError(f"vLLM exited with {server.process.returncode} before readiness\n" + "\n".join(tail))
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.ok:
                models = response.json().get("data", [])
                if models and models[0].get("id"):
                    return str(models[0]["id"])
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s; last error: {last_error}")


class HealthEventWriter:
    """Append compact events immediately so a failed live run remains diagnosable."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a")

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "monotonic_seconds": time.monotonic(),
            "event": event,
            **fields,
        }
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


@dataclasses.dataclass(frozen=True)
class HealthMetricSnapshot:
    index: int
    relative_path: str
    monotonic_seconds: float
    samples: list[Any]
    totals: dict[str, float]
    by_engine: dict[str, dict[str, float]]
    leader_log_bytes: int | None = None


HEALTH_COUNTER_METRICS = (
    "vllm:generation_tokens",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:request_success",
    "vllm:num_preemptions",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
)
HEALTH_ENGINE_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
HEALTH_HISTOGRAM_METRICS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
)


def _capture_health_metrics(
    base_url: str,
    *,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    arm_id: str,
    phase: str,
    log_path: Path | None = None,
) -> HealthMetricSnapshot:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    text = response.text
    captured_at = time.monotonic()
    leader_log_bytes = log_path.stat().st_size if log_path is not None and log_path.exists() else None
    samples = parse_labeled_prometheus(text)
    index = len(metrics_map)
    relative_path = f"metrics/raw-{index:06d}.prom"
    raw_path = artifact_dir / relative_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)
    totals = {metric: prometheus_value(samples, metric) for metric in HEALTH_COUNTER_METRICS}
    by_engine = {metric: prometheus_values_by_label(samples, metric, label="engine") for metric in HEALTH_ENGINE_METRICS}
    metrics_map.append(
        {
            "index": index,
            "path": relative_path,
            "timestamp": datetime.now(UTC).isoformat(),
            "monotonic_seconds": captured_at,
            "leader_log_bytes": leader_log_bytes,
            "arm_id": arm_id,
            "phase": phase,
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "totals": totals,
            "by_engine": by_engine,
        }
    )
    return HealthMetricSnapshot(
        index=index,
        relative_path=relative_path,
        monotonic_seconds=captured_at,
        samples=samples,
        totals=totals,
        by_engine=by_engine,
        leader_log_bytes=leader_log_bytes,
    )


def _health_metric_delta(before: HealthMetricSnapshot, after: HealthMetricSnapshot, metric: str) -> float:
    return after.totals.get(metric, 0.0) - before.totals.get(metric, 0.0)


def _reset_health_prefix_cache(base_url: str) -> dict[str, Any]:
    started = time.monotonic()
    response = requests.post(f"{base_url}/reset_prefix_cache", timeout=60)
    elapsed = time.monotonic() - started
    if not response.ok:
        raise RuntimeError(f"prefix-cache reset failed with {response.status_code}: {response.text[:1000]}")
    return {"status_code": response.status_code, "elapsed_seconds": elapsed}


def _frozen_health_seed(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{DUMMY_SEED}:{identity}".encode()).digest()[:4], "big") & 0x7FFF_FFFF


def _health_route_summary(
    payload: dict[str, Any],
    *,
    case: ModelCase,
    expected_positions: int,
    r3_enabled: bool,
    request_transport: str = "completion",
    expected_prefix_routes: Any | None = None,
    keep_routes: bool = False,
) -> tuple[dict[str, Any], Any | None]:
    if request_transport not in {"completion", "chat"}:
        raise ValueError(f"unknown request transport: {request_transport}")
    choice = _choice(payload)
    encoded = choice.get("routed_experts")
    if not r3_enabled:
        if encoded is not None:
            raise AssertionError("R3-off response unexpectedly carried routed experts")
        return {
            "enabled": False,
            "transport": "absent",
            "carrier_payload_bytes": 0,
        }, None
    import numpy as np  # noqa: PLC0415

    if request_transport == "completion":
        if not isinstance(encoded, str) or not encoded:
            raise AssertionError("R3-on completion response omitted routed experts")
        routes = decode_routed_experts(encoded)
        carrier_array_bytes = int(routes.nbytes)
        carrier_npy_bytes = len(base64.b64decode(encoded))
        carrier_base64_bytes = len(encoded.encode())
        carrier_json_bytes = 0
        transport = "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy"
    else:
        if not isinstance(encoded, list) or not encoded:
            raise AssertionError("R3-on chat response omitted generated-token routed experts")
        routes = np.asarray(encoded)
        if routes.dtype.kind not in {"i", "u"}:
            raise AssertionError(f"chat routed experts have non-integer dtype {routes.dtype}")
        carrier_array_bytes = int(routes.nbytes)
        carrier_npy_bytes = 0
        carrier_base64_bytes = 0
        carrier_json_bytes = len(json.dumps(encoded, separators=(",", ":")).encode())
        transport = "OpenAI chat JSON choice.routed_experts; generated-token-only nested integer lists"
    # Completion emits the full prompt routing followed by generated-prefix
    # routing. Chat deliberately emits one row per generated token. The caller
    # supplies the exact transport-specific position count.
    expected_shape = (expected_positions, case.num_hidden_layers, case.num_experts_per_tok)
    if routes.shape != expected_shape:
        raise AssertionError(f"routed experts shape {routes.shape} does not match {expected_shape}")
    if routes.size and (int(routes.min()) < 0 or int(routes.max()) >= case.num_experts):
        raise AssertionError(f"routed expert IDs are outside [0, {case.num_experts})")
    prefix_positions = 0
    if expected_prefix_routes is not None:
        if request_transport != "completion":
            raise AssertionError("chat generated-token routes cannot be compared with full-prompt root routes")
        prefix_positions = int(expected_prefix_routes.shape[0])
        if routes.shape[1:] != expected_prefix_routes.shape[1:]:
            raise AssertionError("root and branch route layer/top-k dimensions differ")
        np.testing.assert_array_equal(routes[:prefix_positions], expected_prefix_routes)
    expert_histogram = np.bincount(routes.reshape(-1).astype(np.int64), minlength=case.num_experts)
    experts_per_rank = case.num_experts // case.data_parallel_size
    if experts_per_rank * case.data_parallel_size != case.num_experts:
        raise AssertionError("experts do not divide evenly across EP ranks")
    ep_rank_histogram = expert_histogram.reshape(case.data_parallel_size, experts_per_rank).sum(axis=1)
    route_hash = hashlib.sha256()
    route_hash.update(routes.dtype.str.encode())
    route_hash.update(json.dumps(list(routes.shape), separators=(",", ":")).encode())
    route_hash.update(routes.tobytes())
    summary = {
        "enabled": True,
        "shape": list(routes.shape),
        "dtype": str(routes.dtype),
        "minimum_expert": int(routes.min()) if routes.size else None,
        "maximum_expert": int(routes.max()) if routes.size else None,
        "all_expected_positions_layers_topk_aligned": True,
        "root_prefix_positions_compared": prefix_positions,
        "root_prefix_aligned": expected_prefix_routes is None or prefix_positions > 0,
        "root_prefix_alignment_applicable": request_transport == "completion" and expected_prefix_routes is not None,
        "expert_histogram": expert_histogram.astype(int).tolist(),
        "ep_rank_histogram": ep_rank_histogram.astype(int).tolist(),
        "route_sha256": route_hash.hexdigest(),
        "carrier_array_bytes": carrier_array_bytes,
        "carrier_npy_bytes": carrier_npy_bytes,
        "carrier_base64_bytes": carrier_base64_bytes,
        "carrier_json_bytes": carrier_json_bytes,
        "transport": transport,
    }
    return summary, routes if keep_routes else None


def _health_completion(
    base_url: str,
    model: str,
    prompt: list[int],
    *,
    case: ModelCase,
    max_tokens: int,
    data_parallel_rank: int,
    request_id: str,
    sampling_seed: int,
    r3_enabled: bool,
    request_transport: str = "completion",
    expected_prefix_routes: Any | None = None,
    keep_routes: bool = False,
    keep_generated_tokens: bool = False,
) -> dict[str, Any]:
    payload, timing = _timed_completion(
        base_url,
        model,
        prompt,
        max_tokens=max_tokens,
        data_parallel_rank=data_parallel_rank,
        sampling_seed=sampling_seed,
        request_id=request_id,
        request_transport=request_transport,
    )
    usage = payload.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    choice = _choice(payload)
    if prompt_tokens != len(prompt):
        raise AssertionError(f"{request_id} reported {prompt_tokens} prompt tokens, expected {len(prompt)}")
    if completion_tokens != max_tokens:
        raise AssertionError(f"{request_id} generated {completion_tokens} tokens, expected {max_tokens}")
    if choice.get("finish_reason") != "length":
        raise AssertionError(f"{request_id} finish reason was {choice.get('finish_reason')!r}, expected 'length'")
    token_ids = choice.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != max_tokens
        or any(type(token_id) is not int for token_id in token_ids)
    ):
        raise AssertionError(f"{request_id} did not return exactly {max_tokens} token IDs")
    if request_transport == "chat":
        prompt_token_ids = payload.get("prompt_token_ids")
        if prompt_token_ids != prompt:
            raise AssertionError(f"{request_id} chat template changed the frozen prompt token IDs")
    logprobs = choice.get("logprobs")
    if request_transport == "completion":
        sampled_logprobs = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
    else:
        content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None
        sampled_logprobs = (
            [entry.get("logprob") for entry in content_logprobs]
            if isinstance(content_logprobs, list) and all(isinstance(entry, dict) for entry in content_logprobs)
            else None
        )
    if (
        not isinstance(sampled_logprobs, list)
        or len(sampled_logprobs) != max_tokens
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in sampled_logprobs)
    ):
        raise AssertionError(f"{request_id} did not return one finite sampled-token logprob per generated token")
    route_summary, routes = _health_route_summary(
        payload,
        case=case,
        expected_positions=max_tokens if request_transport == "chat" else len(prompt) + max_tokens - 1,
        r3_enabled=r3_enabled,
        request_transport=request_transport,
        expected_prefix_routes=expected_prefix_routes,
        keep_routes=keep_routes,
    )
    result = {
        "request_id": request_id,
        "request_transport": request_transport,
        "data_parallel_rank": data_parallel_rank,
        "sampling_seed": sampling_seed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_token_ids_sha256": _sha256_json(prompt),
        "generated_token_ids_sha256": _sha256_json(token_ids),
        "final_prefix_token_ids_sha256": _sha256_json([*prompt, *token_ids]),
        "sampled_token_logprobs": {
            "count": len(sampled_logprobs),
            "minimum": min(sampled_logprobs),
            "maximum": max(sampled_logprobs),
            "sha256": _sha256_json(sampled_logprobs),
        },
        "timing": timing,
        "routes": route_summary,
        "route_array": routes,
    }
    if keep_generated_tokens:
        result["generated_token_ids"] = token_ids
    return result


def _health_warm_prompt(index: int, prompt_length: int) -> list[int]:
    prompt = [1, *(3 + ((index * 101 + position * 37 + 17) % 252) for position in range(prompt_length - 1))]
    # The prior arithmetic repeats every 252 indices. Encode the index in the
    # first two payload tokens so all experiment-0 warm-up prompts are unique.
    if prompt_length > 1:
        prompt[1] = 3 + index % 252
    if prompt_length > 2:
        prompt[2] = 3 + (index // 252) % 252
    return prompt


def _health_warm_workload(workload: dict[str, Any], *, pass_index: int) -> dict[str, Any]:
    """Copy the measured root graph onto inputs unique to one excluded pass."""
    roots: list[dict[str, Any]] = []
    measured_roots = {tuple(root["prefix_token_ids"]) for root in workload["roots"]}
    root_count = len(workload["roots"])
    for root_position, root in enumerate(workload["roots"]):
        root_index = int(root["root"])
        if root_index != root_position:
            raise ValueError(f"workload root index mismatch at {root_position}")
        prompt = _health_warm_prompt(
            pass_index * root_count + root_index,
            len(root["prefix_token_ids"]),
        )
        if tuple(prompt) in measured_roots:
            raise AssertionError("warm-up root collided with a measured root")
        roots.append({**root, "prefix_token_ids": prompt})
    return {**workload, "roots": roots}


def _health_warm_root_schedule(workload: dict[str, Any], *, case: ModelCase) -> list[dict[str, Any]]:
    """Freeze all copied roots populated before each excluded branch wave."""
    schedule: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for pass_index in range(HEALTH_REPRESENTATIVE_WARM_UP_PASSES):
        warm_workload = _health_warm_workload(workload, pass_index=pass_index)
        for root in warm_workload["roots"]:
            root_index = int(root["root"])
            prompt = list(root["prefix_token_ids"])
            prompt_key = tuple(prompt)
            if prompt_key in seen:
                raise AssertionError("warm-up root was reused across excluded passes")
            seen.add(prompt_key)
            schedule.append(
                {
                    "pass": pass_index,
                    "root": root_index,
                    "cohort": str(root["cohort"]),
                    "token_count": len(prompt),
                    "token_ids_sha256": _sha256_json(prompt),
                    "sampling_seed": _frozen_health_seed(f"warm-populate:{pass_index}:{root_index}"),
                    "data_parallel_rank": root_index % case.data_parallel_size,
                    "disjoint_from_measured_roots": True,
                    "disjoint_from_other_passes": True,
                }
            )
    return schedule


def _health_warm_up_schedule(
    workload: dict[str, Any],
    *,
    case: ModelCase,
    target_concurrency: int,
) -> list[dict[str, Any]]:
    """Freeze measured-shape branch waves over copied shared-root graphs."""
    schedule: list[dict[str, Any]] = []
    slots = frozen_cohort_slots(workload["requests"], target_concurrency=target_concurrency)
    for pass_index in range(HEALTH_REPRESENTATIVE_WARM_UP_PASSES):
        warm_workload = _health_warm_workload(workload, pass_index=pass_index)
        wave_count = 1 + (
            HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES
            if pass_index == HEALTH_REPRESENTATIVE_WARM_UP_PASSES - 1
            else 0
        )
        for wave_index in range(wave_count):
            for slot in slots:
                request = slot.next_request()
                prompt = materialize_prompt(warm_workload, request)
                root_index = int(request["root"])
                root_prompt = list(warm_workload["roots"][root_index]["prefix_token_ids"])
                schedule.append(
                    {
                        "pass": pass_index,
                        "wave": wave_index,
                        "slot_id": int(slot.slot_id),
                        "cohort": str(slot.cohort),
                        "manifest_request_id": str(request["request_id"]),
                        "root": root_index,
                        "branch": int(request["branch"]),
                        "root_token_count": len(root_prompt),
                        "root_token_ids_sha256": _sha256_json(root_prompt),
                        "token_count": len(prompt),
                        "token_ids_sha256": _sha256_json(prompt),
                        "max_tokens": int(request["max_tokens"]),
                        "sampling_seed": _frozen_health_seed(
                            f"warm:{target_concurrency}:{pass_index}:{wave_index}:{slot.slot_id}"
                        ),
                        "data_parallel_rank": root_index % case.data_parallel_size,
                        "disjoint_from_measured_roots": True,
                        "shared_root_graph": True,
                    }
                )
    return schedule


def _health_warm_up(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    case: ModelCase,
    r3_enabled: bool,
    request_transport: str = "completion",
    arm_id: str,
    target_concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Warm copied root graphs, then exercise a same-slot successor wave."""
    schedule = _health_warm_up_schedule(workload, case=case, target_concurrency=target_concurrency)
    records_by_wave: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in schedule:
        records_by_wave.setdefault((int(record["pass"]), int(record["wave"])), []).append(record)
    requests_by_id = {str(request["request_id"]): request for request in workload["requests"]}
    branch_results: list[dict[str, Any]] = []
    root_results: list[dict[str, Any]] = []

    for pass_index in range(HEALTH_REPRESENTATIVE_WARM_UP_PASSES):
        warm_workload = _health_warm_workload(workload, pass_index=pass_index)
        _, populated = _populate_health_roots(
            base_url,
            model,
            warm_workload,
            case=case,
            r3_enabled=r3_enabled,
            request_transport=request_transport,
            arm_id=f"{arm_id}-warm-pass-{pass_index:02d}",
            sampling_seed_prefix=f"warm-populate:{pass_index}",
            retain_routes_for_alignment=False,
        )
        root_results.extend(populated)
        wave_count = 1 + (
            HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES
            if pass_index == HEALTH_REPRESENTATIVE_WARM_UP_PASSES - 1
            else 0
        )
        for wave_index in range(wave_count):
            wave_records = records_by_wave[pass_index, wave_index]
            if len(wave_records) != target_concurrency:
                raise AssertionError(f"warm-up pass {pass_index} wave {wave_index} has {len(wave_records)} branches")

            def one(
                record: dict[str, Any],
                pass_workload: dict[str, Any] = warm_workload,
            ) -> dict[str, Any]:
                request = requests_by_id[str(record["manifest_request_id"])]
                prompt = materialize_prompt(pass_workload, request)
                if _sha256_json(prompt) != record["token_ids_sha256"]:
                    raise AssertionError("warm-up branch changed after its manifest was frozen")
                result = _health_completion(
                    base_url,
                    model,
                    prompt,
                    case=case,
                    max_tokens=int(record["max_tokens"]),
                    data_parallel_rank=int(record["data_parallel_rank"]),
                    request_id=(
                        f"{arm_id}-warm-pass-{record['pass']:02d}-wave-{record['wave']:02d}-"
                        f"slot-{record['slot_id']:03d}-{record['manifest_request_id']}"
                    ),
                    sampling_seed=int(record["sampling_seed"]),
                    r3_enabled=r3_enabled,
                    request_transport=request_transport,
                )
                result.pop("route_array", None)
                return result

            # Recreate the pool only after the whole branch wave completes. This
            # is the wave barrier; no fast slot can submit its successor early.
            with concurrent.futures.ThreadPoolExecutor(max_workers=target_concurrency) as executor:
                branch_results.extend(executor.map(one, wave_records))
    return branch_results, root_results


def _populate_health_roots(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    case: ModelCase,
    r3_enabled: bool,
    request_transport: str = "completion",
    arm_id: str,
    sampling_seed_prefix: str = "populate",
    retain_routes_for_alignment: bool = True,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Populate every shared root through ordinary completion requests."""

    def one(root: dict[str, Any]) -> dict[str, Any]:
        root_index = int(root["root"])
        return _health_completion(
            base_url,
            model,
            list(root["prefix_token_ids"]),
            case=case,
            max_tokens=1,
            data_parallel_rank=root_index % case.data_parallel_size,
            request_id=f"{arm_id}-populate-root-{root_index:02d}",
            sampling_seed=_frozen_health_seed(f"{sampling_seed_prefix}:{root_index}"),
            r3_enabled=r3_enabled,
            request_transport=request_transport,
            keep_routes=r3_enabled and retain_routes_for_alignment and request_transport == "completion",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workload["roots"])) as executor:
        populated = list(executor.map(one, workload["roots"]))
    root_routes: dict[int, Any] = {}
    summaries: list[dict[str, Any]] = []
    for root, result in zip(workload["roots"], populated, strict=True):
        root_index = int(root["root"])
        routes = result.pop("route_array")
        if r3_enabled and retain_routes_for_alignment and request_transport == "completion" and routes is None:
            raise AssertionError(f"root {root_index} did not retain R3 alignment evidence")
        root_routes[root_index] = routes
        summaries.append(result)
    return root_routes, summaries


def _wait_for_health_counter(
    base_url: str,
    *,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    arm_id: str,
    baseline: HealthMetricSnapshot,
    metric: str,
    minimum_delta: int,
    timeout_seconds: float = 120,
) -> HealthMetricSnapshot:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        snapshot = _capture_health_metrics(
            base_url,
            artifact_dir=artifact_dir,
            metrics_map=metrics_map,
            arm_id=arm_id,
            phase=f"counter-sync-{metric}-{attempt}",
        )
        if _health_metric_delta(baseline, snapshot, metric) >= minimum_delta:
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{metric} did not advance by {minimum_delta} within {timeout_seconds}s")
        attempt += 1
        time.sleep(1)


def _health_engine_series(
    metrics_map: list[dict[str, Any]],
    *,
    first_index: int,
    last_index: int,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for metric in HEALTH_ENGINE_METRICS:
        by_engine: dict[str, list[float]] = {}
        for entry in metrics_map[first_index : last_index + 1]:
            for engine, value in entry["by_engine"].get(metric, {}).items():
                by_engine.setdefault(engine, []).append(float(value))
        result[metric] = {
            engine: {
                "samples": len(values),
                "minimum": min(values),
                "mean": sum(values) / len(values),
                "maximum": max(values),
            }
            for engine, values in sorted(by_engine.items())
        }
    return result


def _health_kv_summary_from_text(
    log_slice: str,
    *,
    case: ModelCase,
    target_concurrency: int,
) -> dict[str, Any]:
    summaries = [summarize_kv_snapshot(snapshot) for snapshot in parse_kv_group_snapshots(log_slice)]
    if not summaries:
        return {"passed": False, "snapshot_count": 0, "reason": "no GrugMoE KV group snapshots in arm log"}
    peak = max(summaries, key=lambda item: int(item["physical_active_bytes"]))
    compact_peak = {key: value for key, value in peak.items() if key != "groups"}
    active_requests = int(peak["active_requests"])
    compact_peak["semantic_active_bytes_per_live_sequence"] = (
        int(peak["semantic_active_bytes"]) / active_requests if active_requests else None
    )
    compact_peak["physical_active_bytes_per_live_sequence"] = (
        int(peak["physical_active_bytes"]) / active_requests if active_requests else None
    )
    schedule = layer_types(case.num_hidden_layers, global_interval=case.global_every)
    global_layers = schedule.count("full_attention")
    local_layers = len(schedule) - global_layers
    per_cohort_attention = {
        name: predict_kv_bytes(
            sequence_length=length,
            local_layers=local_layers,
            global_layers=global_layers,
            local_kv_heads=case.local_kv_heads,
            global_kv_heads=case.global_kv_heads,
            head_dim=case.head_dim,
            sliding_window=case.sliding_window,
        )
        for name, length in zip(("short", "medium", "long"), (13_312, 33_792, 65_536), strict=True)
    }
    slots_per_cohort = target_concurrency // 3
    predicted_attention = slots_per_cohort * sum(per_cohort_attention.values())
    measured_attention = int(peak["semantic_attention_active_bytes"])
    prediction_delta_fraction = (
        abs(measured_attention - predicted_attention) / predicted_attention if predicted_attention else math.inf
    )
    semantic = int(peak["semantic_active_bytes"])
    physical = int(peak["physical_active_bytes"])
    physical_gap_fraction = (physical - semantic) / semantic if semantic else math.inf
    engine_count = len(peak["active_requests_by_engine"])
    reserved_engine_count = len(peak["reserved_physical_bytes_per_engine"])
    return {
        "passed": (
            active_requests > 0
            and engine_count == case.data_parallel_size
            and reserved_engine_count == case.data_parallel_size
        ),
        "snapshot_count": len(summaries),
        "engine_coverage": {
            "expected": case.data_parallel_size,
            "active_request_series_observed": engine_count,
            "reserved_byte_series_observed": reserved_engine_count,
        },
        "peak": compact_peak,
        "attention_prediction": {
            "per_live_sequence_bytes_by_cohort": per_cohort_attention,
            "equal_slots_per_cohort": slots_per_cohort,
            "predicted_active_bytes_without_prefix_page_sharing": predicted_attention,
            "measured_peak_semantic_attention_bytes": measured_attention,
            "absolute_gap_from_unshared_final_context_prediction_fraction": prediction_delta_fraction,
            "gap_explanation": (
                "the prediction charges every live sequence its complete final-context KV; the measured rolling peak "
                "contains mixed decode positions and counts shared cached root pages once"
                if prediction_delta_fraction > 0.10
                else "the measured rolling peak is within 10% of the unshared final-context prediction"
            ),
        },
        "physical_to_semantic_gap_fraction": physical_gap_fraction,
        "physical_gap_explanation": (
            "physical accounting includes hybrid group page/block padding and SConv state; "
            "semantic accounting counts payload"
            if physical_gap_fraction > 0.10
            else "physical and semantic active-byte accounting differ by at most 10%"
        ),
    }


def _health_kv_summary(
    log_path: Path,
    *,
    start_offset: int,
    case: ModelCase,
    target_concurrency: int,
    end_offset: int | None = None,
) -> dict[str, Any]:
    if not log_path.exists():
        log_slice = ""
    else:
        payload = log_path.read_bytes()
        log_slice = payload[start_offset:end_offset].decode(errors="replace")
    return _health_kv_summary_from_text(
        log_slice,
        case=case,
        target_concurrency=target_concurrency,
    )


def _health_percentiles(values: list[float]) -> dict[str, float]:
    summary = _latency_summary(values)
    ordered = sorted(values)
    summary["p99"] = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.99))]
    return summary


def _health_final_prefix_provenance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact per-instance output hashes into one auditable record per frozen branch."""
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        request_id = str(record["manifest_request_id"])
        group = grouped.setdefault(
            request_id,
            {
                "manifest_request_id": request_id,
                "prompt_token_ids_sha256": str(record["prompt_token_ids_sha256"]),
                "occurrences": 0,
                "outcomes": {},
            },
        )
        if group["prompt_token_ids_sha256"] != record["prompt_token_ids_sha256"]:
            raise AssertionError(f"{request_id} changed its frozen prompt")
        outcome_key = (
            str(record["generated_token_ids_sha256"]),
            str(record["final_prefix_token_ids_sha256"]),
        )
        group["occurrences"] += 1
        group["outcomes"][outcome_key] = group["outcomes"].get(outcome_key, 0) + 1
    compact: list[dict[str, Any]] = []
    for request_id, group in sorted(grouped.items()):
        compact.append(
            {
                "manifest_request_id": request_id,
                "prompt_token_ids_sha256": group["prompt_token_ids_sha256"],
                "occurrences": group["occurrences"],
                "outcomes": [
                    {
                        "generated_token_ids_sha256": generated,
                        "final_prefix_token_ids_sha256": final,
                        "occurrences": occurrences,
                    }
                    for (generated, final), occurrences in sorted(group["outcomes"].items())
                ],
            }
        )
    return compact


def _health_collective_rpc(base_url: str, method: str) -> list[Any]:
    started = time.monotonic()
    response = requests.post(
        f"{base_url}/collective_rpc",
        json={"method": method, "args": [], "kwargs": {}},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"collective RPC {method} failed with {response.status_code}: {response.text[:4000]}")
    payload = response.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise AssertionError(f"collective RPC {method} returned no worker result list")
    if time.monotonic() - started > REQUEST_TIMEOUT_SECONDS:
        raise TimeoutError(f"collective RPC {method} exceeded its request timeout")
    return results


def _reset_health_route_audit(base_url: str, *, expected_workers: int) -> dict[str, Any]:
    started = time.monotonic()
    results = _health_collective_rpc(base_url, "grugmoe_route_audit_reset")
    if len(results) != expected_workers or any(result is not None for result in results):
        raise AssertionError(f"route-audit reset returned unexpected worker results: {results!r}")
    return {
        "workers": len(results),
        "elapsed_seconds": time.monotonic() - started,
        "completed_before_measured_requests": True,
    }


def _health_route_audit_summary(
    snapshots: list[Any],
    *,
    case: ModelCase,
    mode: str,
    routing_regime: str,
    expected_assignment_count: int,
) -> dict[str, Any]:
    if mode not in {"noop", "record"}:
        raise ValueError(f"unknown route-audit mode: {mode}")
    if routing_regime not in {"canonical", "balanced"}:
        raise ValueError(f"unknown routing regime: {routing_regime}")
    import numpy as np  # noqa: PLC0415

    shape = (case.num_hidden_layers, case.num_experts)
    valid_workers: list[dict[str, Any]] = []
    worker_shapes_valid = len(snapshots) == case.data_parallel_size
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            worker_shapes_valid = False
            continue
        counts = np.asarray(snapshot.get("counts"))
        mask = np.asarray(snapshot.get("local_expert_mask"))
        worker = snapshot.get("worker")
        valid = (
            snapshot.get("mode") == mode
            and int(snapshot.get("num_layers", -1)) == case.num_hidden_layers
            and int(snapshot.get("num_experts", -1)) == case.num_experts
            and counts.shape == shape
            and mask.shape == shape
            and counts.dtype.kind in {"i", "u"}
            and mask.dtype.kind in {"i", "u"}
            and bool(np.all(counts >= 0))
            and bool(np.all((mask == 0) | (mask == 1)))
            and isinstance(worker, dict)
            and all(type(worker.get(rank)) is int for rank in ("global_rank", "local_rank", "dp_rank", "ep_rank"))
        )
        worker_shapes_valid = worker_shapes_valid and valid
        if valid:
            valid_workers.append(
                {
                    "snapshot": snapshot,
                    "counts": counts.astype(np.int64),
                    "mask": mask.astype(np.int64),
                    "worker": worker,
                }
            )

    if len(valid_workers) == case.data_parallel_size:
        aggregate_counts = sum((worker["counts"] for worker in valid_workers), np.zeros(shape, dtype=np.int64))
        aggregate_mask = sum((worker["mask"] for worker in valid_workers), np.zeros(shape, dtype=np.int64))
    else:
        aggregate_counts = np.zeros(shape, dtype=np.int64)
        aggregate_mask = np.zeros(shape, dtype=np.int64)
    assignment_count = int(aggregate_counts.sum())
    layer_assignment_counts = aggregate_counts.sum(axis=1)
    expert_assignment_counts = aggregate_counts.sum(axis=0)
    worker_assignment_counts = [int(worker["counts"].sum()) for worker in valid_workers]
    reported_worker_assignment_counts = [int(worker["snapshot"].get("assignment_count", -1)) for worker in valid_workers]
    dp_ranks = [int(worker["worker"]["dp_rank"]) for worker in valid_workers]
    ep_ranks = [int(worker["worker"]["ep_rank"]) for worker in valid_workers]
    global_ranks = [int(worker["worker"]["global_rank"]) for worker in valid_workers]
    counts_outside_ownership = sum(int((worker["counts"] * (1 - worker["mask"])).sum()) for worker in valid_workers)
    expected_per_layer = expected_assignment_count // case.num_hidden_layers if case.num_hidden_layers else 0
    ownership_per_expert = aggregate_mask.sum(axis=0) if aggregate_mask.size else np.array([], dtype=np.int64)
    expert_mean = float(expert_assignment_counts.mean()) if assignment_count else 0.0
    expert_max = int(expert_assignment_counts.max()) if expert_assignment_counts.size else 0
    expert_min = int(expert_assignment_counts.min()) if expert_assignment_counts.size else 0
    gates = {
        "worker_schema": worker_shapes_valid and len(valid_workers) == case.data_parallel_size,
        "rank_coverage": (
            sorted(dp_ranks) == list(range(case.data_parallel_size))
            and sorted(ep_ranks) == list(range(case.data_parallel_size))
            and len(set(global_ranks)) == case.data_parallel_size
        ),
        "single_owner_per_expert_layer": bool(aggregate_mask.size and np.all(aggregate_mask == 1)),
        "single_owner_per_expert": bool(
            ownership_per_expert.size and np.all(ownership_per_expert == case.num_hidden_layers)
        ),
        "counts_only_on_owner": counts_outside_ownership == 0,
        "worker_assignment_reconciliation": worker_assignment_counts == reported_worker_assignment_counts,
        "mode_assignment_count": assignment_count == (expected_assignment_count if mode == "record" else 0),
        "per_layer_no_drops": (
            bool(np.all(layer_assignment_counts == expected_per_layer))
            if mode == "record"
            else bool(np.all(layer_assignment_counts == 0))
        ),
        "balanced_fixture_load": (
            expert_min > 0 and expert_max / expert_mean <= 1.05
            if mode == "record" and routing_regime == "balanced" and expert_mean
            else True
        ),
    }
    return {
        "enabled": True,
        "mode": mode,
        "routing_regime": routing_regime,
        "source": (
            "fixed GPU int64 counters in the existing GrugMoE router capture hook; one worker D2H snapshot after the arm"
        ),
        "timing_boundary": "reset before first measured request; snapshot after drain; both outside headline plateau",
        "passed": all(gates.values()),
        "gates": gates,
        "expected_assignment_count": expected_assignment_count,
        "assignment_count": assignment_count,
        "counts_outside_ownership": counts_outside_ownership,
        "layer_assignment_counts": layer_assignment_counts.astype(int).tolist(),
        "expert_assignment_counts": expert_assignment_counts.astype(int).tolist(),
        "worker_assignment_counts": worker_assignment_counts,
        "workers": [worker["snapshot"] for worker in valid_workers],
        "expert_load": {
            "minimum": expert_min,
            "mean": expert_mean,
            "maximum": expert_max,
            "maximum_over_mean": expert_max / expert_mean if expert_mean else None,
        },
    }


def _run_rolling_health_arm(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    case: ModelCase,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    events: HealthEventWriter,
    log_path: Path,
    arm_id: str,
    target_concurrency: int,
    minimum_seconds: float,
    minimum_generated_tokens: int,
    r3_enabled: bool,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    request_transport: str = "completion",
    routing_regime: str = "canonical",
    route_audit_mode: str | None = None,
    metric_sample_seconds: float = 5.0,
) -> dict[str, Any]:
    if request_transport not in {"completion", "chat"}:
        raise ValueError(f"unknown request transport: {request_transport}")
    if routing_regime not in {"canonical", "balanced"}:
        raise ValueError(f"unknown routing regime: {routing_regime}")
    if route_audit_mode not in {None, "noop", "record"}:
        raise ValueError(f"unknown route-audit mode: {route_audit_mode}")
    if r3_enabled and route_audit_mode is not None:
        raise ValueError("R3 response capture and aggregate route audit are mutually exclusive")
    if target_concurrency <= 0 or target_concurrency % 3:
        raise ValueError("health concurrency must be positive and divisible by three")
    if metric_sample_seconds <= 0:
        raise ValueError("health metric sample interval must be positive")
    required_request_ids = frozenset(str(request["request_id"]) for request in workload["requests"])
    if len(required_request_ids) != 144:
        raise ValueError("health workload must contain 144 unique frozen branches")
    events.emit(
        "arm_started",
        arm_id=arm_id,
        target_concurrency=target_concurrency,
        r3_enabled=r3_enabled,
        request_transport=request_transport,
        routing_regime=routing_regime,
        route_audit_mode=route_audit_mode,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    warm_branches, warm_roots = _health_warm_up(
        base_url,
        model,
        workload,
        case=case,
        r3_enabled=r3_enabled,
        request_transport=request_transport,
        arm_id=arm_id,
        target_concurrency=target_concurrency,
    )
    events.emit(
        "warm_up_completed",
        arm_id=arm_id,
        requests=len(warm_branches) + len(warm_roots),
        branch_requests=len(warm_branches),
        root_population_requests=len(warm_roots),
    )
    cache_reset = _reset_health_prefix_cache(base_url)
    events.emit("prefix_cache_reset", arm_id=arm_id, **cache_reset)
    population_before = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase="root-population-before",
    )
    root_routes, population = _populate_health_roots(
        base_url,
        model,
        workload,
        case=case,
        r3_enabled=r3_enabled,
        request_transport=request_transport,
        arm_id=arm_id,
    )
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=population_before,
        metric="vllm:generation_tokens",
        minimum_delta=len(workload["roots"]),
    )
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=population_before,
        metric="vllm:request_success",
        minimum_delta=len(workload["roots"]),
    )
    events.emit("root_population_completed", arm_id=arm_id, roots=len(population))

    slots = frozen_cohort_slots(workload["requests"], target_concurrency=target_concurrency)
    requirements = PlateauRequirements(
        target_concurrency=target_concurrency,
        minimum_seconds=minimum_seconds,
        minimum_generated_tokens=minimum_generated_tokens,
        required_request_ids=required_request_ids,
    )
    plateau = PlateauWindow(requirements)
    gate = threading.Event()
    initial_barrier = threading.Barrier(target_concurrency + 1)
    sequence_by_slot = {slot.slot_id: 0 for slot in slots}
    failures: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []

    def one(slot: Any, request: dict[str, Any], sequence: int, initial_gate: threading.Event | None) -> dict[str, Any]:
        if initial_gate is not None:
            initial_gate.wait()
            initial_barrier.wait(timeout=60)
        prompt = materialize_prompt(workload, request)
        max_tokens = int(request["max_tokens"])
        if len(prompt) + max_tokens != int(request["final_token_count"]):
            raise AssertionError(f"{request['request_id']} does not have its frozen final length")
        root_index = int(request["root"])
        rank = root_index % case.data_parallel_size
        instance_id = f"{arm_id}-slot-{slot.slot_id:03d}-sequence-{sequence:06d}-{request['request_id']}"
        result = _health_completion(
            base_url,
            model,
            prompt,
            case=case,
            max_tokens=max_tokens,
            data_parallel_rank=rank,
            request_id=instance_id,
            sampling_seed=_frozen_health_seed(f"branch:{request['request_id']}"),
            r3_enabled=r3_enabled,
            request_transport=request_transport,
            expected_prefix_routes=root_routes[root_index] if request_transport == "completion" else None,
        )
        result.pop("route_array", None)
        result.update(
            {
                "manifest_request_id": str(request["request_id"]),
                "root": root_index,
                "branch": int(request["branch"]),
                "cohort": str(request["cohort"]),
                "slot_id": int(slot.slot_id),
                "sequence": sequence,
            }
        )
        # This is the completion boundary for the closed-loop controller. The
        # transport timing above stops after JSON decode; R3 validation and
        # hashing can continue after that. A slot is not available for refill
        # until this point, so plateau membership must use this timestamp.
        result["controller_completed_at_monotonic_seconds"] = time.monotonic()
        return result

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=target_concurrency)
    metric_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending: dict[concurrent.futures.Future[dict[str, Any]], tuple[Any, dict[str, Any], int]] = {}

    def capture_periodic_metrics(phase: str = "plateau-periodic") -> HealthMetricSnapshot:
        return _capture_health_metrics(
            base_url,
            artifact_dir=artifact_dir,
            metrics_map=metrics_map,
            arm_id=arm_id,
            phase=phase,
            log_path=log_path,
        )

    try:
        for slot in slots:
            request = slot.next_request()
            sequence = sequence_by_slot[slot.slot_id]
            sequence_by_slot[slot.slot_id] += 1
            pending[executor.submit(one, slot, request, sequence, gate)] = (slot, request, sequence)
        rolling_before = _capture_health_metrics(
            base_url,
            artifact_dir=artifact_dir,
            metrics_map=metrics_map,
            arm_id=arm_id,
            phase="rolling-start",
        )
        route_audit_reset = None
        if route_audit_mode is not None:
            route_audit_reset = _reset_health_route_audit(
                base_url,
                expected_workers=case.data_parallel_size,
            )
            events.emit(
                "route_audit_reset",
                arm_id=arm_id,
                mode=route_audit_mode,
                **route_audit_reset,
            )
        # The rolling baseline precedes every measured request. Release the
        # initial slots together, then scrape the opening boundary on the
        # metrics thread so completed slots can still be refilled immediately.
        plateau_log_end: int | None = None
        gate.set()
        initial_barrier.wait(timeout=60)
        next_metric_at = math.inf
        metric_future: concurrent.futures.Future[HealthMetricSnapshot] | None = metric_executor.submit(
            capture_periodic_metrics,
            "plateau-open",
        )
        metric_boundary_records: list[dict[str, Any]] = []
        plateau_before: HealthMetricSnapshot | None = None
        plateau_log_start: int | None = None
        plateau_after: HealthMetricSnapshot | None = None

        def record(result: dict[str, Any], *, in_window: bool) -> None:
            all_records.append(result)
            if in_window:
                window_records.append(result)
                plateau.record_completion(
                    request_id=result["manifest_request_id"],
                    cohort=result["cohort"],
                    completion_tokens=int(result["completion_tokens"]),
                    succeeded=True,
                )
            events.emit(
                "request_completed",
                arm_id=arm_id,
                request_id=result["request_id"],
                manifest_request_id=result["manifest_request_id"],
                cohort=result["cohort"],
                slot_id=result["slot_id"],
                completion_tokens=result["completion_tokens"],
                completed_at_monotonic_seconds=result["controller_completed_at_monotonic_seconds"],
                client_e2e_seconds=result["timing"]["client_e2e_seconds"],
                response_bytes=result["timing"]["response_bytes"],
                timing=result["timing"],
                route_sha256=result["routes"].get("route_sha256"),
                route_summary=result["routes"],
                prompt_token_ids_sha256=result["prompt_token_ids_sha256"],
                generated_token_ids_sha256=result["generated_token_ids_sha256"],
                final_prefix_token_ids_sha256=result["final_prefix_token_ids_sha256"],
                sampled_token_logprobs_count=result["sampled_token_logprobs"]["count"],
                sampled_token_logprobs_sha256=result["sampled_token_logprobs"]["sha256"],
            )

        excluded_before_valid_plateau = 0
        drain_records: list[dict[str, Any]] = []
        drain_started: float | None = None
        while plateau_after is None:
            now = time.monotonic()
            if plateau_before is not None and metric_future is None and now >= next_metric_at:
                metric_future = metric_executor.submit(capture_periodic_metrics)
                next_metric_at = now + metric_sample_seconds
            wait_for = set(pending)
            if metric_future is not None:
                wait_for.add(metric_future)
            timeout = None if metric_future is not None else max(0.0, next_metric_at - now)
            done, _ = concurrent.futures.wait(
                wait_for,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            metric_boundary_ready = metric_future is not None and metric_future in done
            if metric_boundary_ready:
                # The scrape records its boundary before returning. Harvest
                # every request future already complete now, not only the set
                # that FIRST_COMPLETED happened to return. Any controller
                # completion at or before the boundary must be classified in
                # this iteration; otherwise it could be mislabeled as drain.
                done.update(future for future in pending if future.done())
            completed_results: list[dict[str, Any]] = []
            for future in [item for item in done if item in pending]:
                slot, _, _ = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    failure = {"type": type(exc).__name__, "message": str(exc), "slot_id": slot.slot_id}
                    failures.append(failure)
                    plateau.record_completion(
                        request_id="failed",
                        cohort=slot.cohort,
                        completion_tokens=0,
                        succeeded=False,
                    )
                    events.emit("request_failed", arm_id=arm_id, **failure)
                    raise
                successor = slot.next_request()
                sequence = sequence_by_slot[slot.slot_id]
                sequence_by_slot[slot.slot_id] += 1
                pending[executor.submit(one, slot, successor, sequence, None)] = (slot, successor, sequence)
                completed_results.append(result)

            if not metric_boundary_ready:
                if metric_future is None:
                    for result in completed_results:
                        record(result, in_window=plateau.is_open)
                else:
                    # Refills above remain immediate, but the scrape's exact
                    # timestamp is not known until it returns. Defer only the
                    # accounting so responses that finish during the scrape
                    # can be partitioned against its real boundary.
                    metric_boundary_records.extend(completed_results)
                continue
            assert metric_future is not None
            snapshot = metric_future.result()
            metric_future = None
            completed_results = [*metric_boundary_records, *completed_results]
            metric_boundary_records.clear()
            before_boundary: list[dict[str, Any]] = []
            after_boundary: list[dict[str, Any]] = []
            for result in completed_results:
                completed_at = float(result["controller_completed_at_monotonic_seconds"])
                (before_boundary if completed_at <= snapshot.monotonic_seconds else after_boundary).append(result)
            was_open = plateau.is_open
            for result in before_boundary:
                record(result, in_window=was_open)
            # A closed-loop slot stays occupied until its response is consumed
            # and its successor is submitted. Refill happens above before any
            # accounting or event work, so this is the client concurrency at
            # the metrics boundary even if a future completed nanoseconds ago.
            sample_in_flight = len(pending)
            transition = plateau.observe_in_flight(
                now=snapshot.monotonic_seconds,
                in_flight=sample_in_flight,
                generation_counter=snapshot.totals["vllm:generation_tokens"],
                prompt_counter=snapshot.totals["vllm:prompt_tokens"],
            )
            if transition == "discarded":
                window_records.clear()
                events.emit("plateau_discarded", arm_id=arm_id, in_flight=sample_in_flight)
            elif transition == "opened":
                first_open = plateau_before is None
                window_records.clear()
                plateau_before = snapshot
                plateau_log_start = snapshot.leader_log_bytes
                if plateau_log_start is None:
                    raise AssertionError("plateau metric boundary omitted the leader log byte offset")
                next_metric_at = time.monotonic() + metric_sample_seconds
                events.emit(
                    "plateau_opened" if first_open else "plateau_reopened",
                    arm_id=arm_id,
                    target_concurrency=target_concurrency,
                    in_flight=sample_in_flight,
                    metric_snapshot=snapshot.relative_path,
                )
            if plateau.ready_to_close(
                now=snapshot.monotonic_seconds,
                in_flight=sample_in_flight,
                generation_counter=snapshot.totals["vllm:generation_tokens"],
            ):
                plateau_result = plateau.close(
                    now=snapshot.monotonic_seconds,
                    in_flight=sample_in_flight,
                    generation_counter=snapshot.totals["vllm:generation_tokens"],
                    prompt_counter=snapshot.totals["vllm:prompt_tokens"],
                )
                plateau_after = snapshot
                plateau_log_end = snapshot.leader_log_bytes
                if plateau_log_end is None:
                    raise AssertionError("closing metric boundary omitted the leader log byte offset")
                metrics_map[snapshot.index]["phase"] = "plateau-close"
                excluded_before_valid_plateau = len(all_records) - len(window_records)
                drain_started = time.monotonic()
                for result in after_boundary:
                    drain_records.append(result)
                    record(result, in_window=False)
                events.emit(
                    "plateau_closed",
                    arm_id=arm_id,
                    metric_snapshot=snapshot.relative_path,
                    elapsed_seconds=plateau_result["elapsed_seconds"],
                    generated_tokens=plateau_result["generated_tokens"],
                )
            else:
                for result in after_boundary:
                    record(result, in_window=plateau.is_open)

        # Closing freezes the headline boundary. Already-live work is drained
        # without replacement and is intentionally excluded from the plateau.
        if metric_boundary_records:
            raise AssertionError("unclassified request completions remained after the plateau boundary")
        assert drain_started is not None
        for future in concurrent.futures.as_completed(list(pending)):
            result = future.result()
            drain_records.append(result)
            record(result, in_window=False)
        pending.clear()
        drain_seconds = time.monotonic() - drain_started
    finally:
        gate.set()
        initial_barrier.abort()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        metric_executor.shutdown(wait=True, cancel_futures=True)

    route_audit_snapshots: list[Any] | None = None
    if route_audit_mode is not None:
        route_audit_snapshots = _health_collective_rpc(base_url, "grugmoe_route_audit_snapshot")
        events.emit(
            "route_audit_snapshot",
            arm_id=arm_id,
            mode=route_audit_mode,
            workers=len(route_audit_snapshots),
        )

    client_generated_tokens = sum(int(record["completion_tokens"]) for record in all_records)
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=rolling_before,
        metric="vllm:generation_tokens",
        minimum_delta=client_generated_tokens,
    )
    final_snapshot = _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=rolling_before,
        metric="vllm:request_success",
        minimum_delta=len(all_records),
    )
    engine_generated_tokens = round(_health_metric_delta(rolling_before, final_snapshot, "vllm:generation_tokens"))
    engine_successes = round(_health_metric_delta(rolling_before, final_snapshot, "vllm:request_success"))
    preemptions = round(_health_metric_delta(rolling_before, final_snapshot, "vllm:num_preemptions"))
    route_audit: dict[str, Any] | None = None
    if route_audit_mode is not None:
        processed_positions = round(
            _health_metric_delta(rolling_before, final_snapshot, "vllm:prompt_tokens")
            - _health_metric_delta(rolling_before, final_snapshot, "vllm:prompt_tokens_cached")
            + engine_generated_tokens
            - engine_successes
        )
        if processed_positions < 0:
            raise AssertionError(f"route-audit processed-position count is negative: {processed_positions}")
        assert route_audit_snapshots is not None
        route_audit = _health_route_audit_summary(
            route_audit_snapshots,
            case=case,
            mode=route_audit_mode,
            routing_regime=routing_regime,
            expected_assignment_count=processed_positions * case.num_hidden_layers * case.num_experts_per_tok,
        )
        route_audit["reset"] = route_audit_reset
        route_audit["processed_positions"] = processed_positions
        route_audit["position_reconciliation"] = {
            "prompt_tokens": round(_health_metric_delta(rolling_before, final_snapshot, "vllm:prompt_tokens")),
            "prompt_tokens_cached": round(
                _health_metric_delta(rolling_before, final_snapshot, "vllm:prompt_tokens_cached")
            ),
            "generation_tokens": engine_generated_tokens,
            "request_successes": engine_successes,
            "formula": "prompt - cached + generation - successes",
        }
    reconciliation = {
        "client_generated_tokens": client_generated_tokens,
        "engine_generated_tokens": engine_generated_tokens,
        "delta": engine_generated_tokens - client_generated_tokens,
        "passed": engine_generated_tokens == client_generated_tokens,
    }
    cohort_latencies = {
        cohort: _health_percentiles(
            [record["timing"]["client_e2e_seconds"] for record in window_records if record["cohort"] == cohort]
        )
        for cohort in ("short", "medium", "long")
    }
    assert plateau_before is not None
    assert plateau_after is not None
    server_latency = {
        metric: {
            "p50_seconds": histogram_quantile_delta(plateau_before.samples, plateau_after.samples, metric, 0.50),
            "p99_seconds": histogram_quantile_delta(plateau_before.samples, plateau_after.samples, metric, 0.99),
        }
        for metric in HEALTH_HISTOGRAM_METRICS
    }
    import numpy as np  # noqa: PLC0415

    expert_histogram = np.zeros(case.num_experts, dtype=np.int64)
    ep_rank_histogram = np.zeros(case.data_parallel_size, dtype=np.int64)
    carrier_array_bytes = 0
    carrier_npy_bytes = 0
    carrier_base64_bytes = 0
    carrier_json_bytes = 0
    response_bytes = 0
    for record in window_records:
        routes = record["routes"]
        if r3_enabled:
            expert_histogram += np.asarray(routes["expert_histogram"], dtype=np.int64)
            ep_rank_histogram += np.asarray(routes["ep_rank_histogram"], dtype=np.int64)
            carrier_array_bytes += int(routes["carrier_array_bytes"])
            carrier_npy_bytes += int(routes["carrier_npy_bytes"])
            carrier_base64_bytes += int(routes["carrier_base64_bytes"])
            carrier_json_bytes += int(routes["carrier_json_bytes"])
        response_bytes += int(record["timing"]["response_bytes"])
    transport_timing = {
        key: _health_percentiles([float(record["timing"][key]) for record in window_records])
        for key in (
            "seconds_to_response_headers",
            "response_body_transfer_seconds",
            "seconds_to_decode",
        )
    }
    plateau_seconds = float(plateau_result["elapsed_seconds"])
    generated_tokens = int(plateau_result["generated_tokens"])
    processed_prompt_tokens = int(plateau_result["processed_prompt_tokens"])
    branch_coverage = {record["manifest_request_id"] for record in window_records}
    final_prefix_provenance = _health_final_prefix_provenance(all_records)
    sampled_logprob_count = sum(int(record["sampled_token_logprobs"]["count"]) for record in all_records)
    route_alignment_passed = all(
        not r3_enabled
        or (record["routes"]["all_expected_positions_layers_topk_aligned"] and record["routes"]["root_prefix_aligned"])
        for record in window_records
    )
    assert plateau_log_start is not None
    assert plateau_log_end is not None
    kv_source_path = f"metrics/{arm_id}-kv.log"
    full_log = log_path.read_bytes() if log_path.exists() else b""
    kv_source_bytes = full_log[plateau_log_start:plateau_log_end]
    kv_source_file = artifact_dir / kv_source_path
    kv_source_file.parent.mkdir(parents=True, exist_ok=True)
    kv_source_file.write_bytes(kv_source_bytes)
    kv = _health_kv_summary(
        kv_source_file,
        start_offset=0,
        case=case,
        target_concurrency=target_concurrency,
    )
    kv["source"] = {
        "path": kv_source_path,
        "bytes": len(kv_source_bytes),
        "sha256": hashlib.sha256(kv_source_bytes).hexdigest(),
        "boundary": "vLLM leader log bytes captured only during the accepted plateau",
    }
    counter_deltas = {
        metric: _health_metric_delta(plateau_before, plateau_after, metric) for metric in HEALTH_COUNTER_METRICS
    }
    prefix_queries = counter_deltas["vllm:prefix_cache_queries"]
    prefix_hits = counter_deltas["vllm:prefix_cache_hits"]
    prefix_hit_ratio = prefix_hits / prefix_queries if prefix_queries else None
    per_engine_metrics_complete = all(
        len(plateau_before.by_engine.get(metric, {})) == case.data_parallel_size
        and len(plateau_after.by_engine.get(metric, {})) == case.data_parallel_size
        for metric in HEALTH_ENGINE_METRICS
    )
    latency_histograms_complete = all(
        value is not None for summary in server_latency.values() for value in summary.values()
    )
    gates = {
        "plateau_duration": plateau_seconds >= minimum_seconds,
        "plateau_generation_tokens": generated_tokens >= minimum_generated_tokens,
        "queue_floor_and_full_close": (
            plateau_result["in_flight"]["min"] >= requirements.minimum_in_flight
            and plateau_result["in_flight"]["max"] <= target_concurrency
            and plateau_result["in_flight"]["at_close"] == target_concurrency
        ),
        "manifest_coverage": branch_coverage == set(required_request_ids),
        "final_prefix_provenance": (
            {entry["manifest_request_id"] for entry in final_prefix_provenance} == set(required_request_ids)
        ),
        "sampled_token_logprobs": sampled_logprob_count == client_generated_tokens,
        "all_requests_succeeded": not failures and len(all_records) == engine_successes,
        "zero_preemptions": preemptions == 0,
        "whole_run_token_reconciliation": reconciliation["passed"],
        "r3_alignment": route_alignment_passed,
        "route_audit": route_audit is None or route_audit["passed"],
        "kv_instrumentation": bool(kv["passed"]),
        "per_engine_metrics_complete": per_engine_metrics_complete,
        "server_latency_histograms_complete": latency_histograms_complete,
        "prefix_cache_metrics_observed": prefix_queries > 0 and prefix_hits > 0,
    }
    rank_assignment_mean = float(ep_rank_histogram.mean()) if r3_enabled else None
    rank_assignment_max = int(ep_rank_histogram.max()) if r3_enabled else None
    per_engine_plateau_series = _health_engine_series(
        metrics_map,
        first_index=plateau_before.index,
        last_index=plateau_after.index,
    )
    result = {
        "arm_id": arm_id,
        "passed": all(gates.values()),
        "gates": gates,
        "settings": {
            "target_concurrency": target_concurrency,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "r3_enabled": r3_enabled,
            "request_transport": request_transport,
            "routing_regime": routing_regime,
            "route_audit_mode": route_audit_mode,
            "queue_definition": (
                "occupied frozen client slots; a completed response retains its slot until "
                "the controller consumes it and submits the same-cohort successor"
            ),
            "settings_drift": False,
        },
        "warm_up": {
            "requests": len(warm_branches) + len(warm_roots),
            "branch_requests": len(warm_branches),
            "root_population_requests": len(warm_roots),
            "root_copies": len(workload["roots"]) * HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
            "rolling_passes": HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
            "successor_waves_after_final_pass": HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES,
            "target_concurrency": target_concurrency,
            "max_tokens_per_request": max(int(request["max_tokens"]) for request in workload["requests"]),
            "data_parallel_ranks_covered": sorted(
                {int(root["root"]) % case.data_parallel_size for root in workload["roots"]}
                | {
                    int(record["data_parallel_rank"])
                    for record in _health_warm_up_schedule(
                        workload,
                        case=case,
                        target_concurrency=target_concurrency,
                    )
                }
            ),
            "shared_root_graph": True,
            "full_pass_barrier": True,
            "full_wave_barrier": True,
            "distinct_root_copy_per_pass": True,
            "disjoint_from_measured_roots": True,
            "excluded_from_measurement": True,
            "prefix_cache_reset_after": True,
        },
        "prefix_cache_reset": cache_reset,
        "root_population": {
            "requests": len(population),
            "normal_completion_requests": True,
            "dp_rank_rule": f"root modulo {case.data_parallel_size}",
            "r3_root_routes_retained_for_branch_alignment": r3_enabled and request_transport == "completion",
        },
        "plateau": plateau_result,
        "drain": {"elapsed_seconds": drain_seconds, "excluded_from_plateau": True},
        "headline": {
            "counter": "vllm:generation_tokens boundary delta",
            "aggregate_generation_tokens_per_second": generated_tokens / plateau_seconds,
            "generation_tokens_per_second_per_gpu": generated_tokens / plateau_seconds / case.data_parallel_size,
            "prompt_tokens_per_second_per_gpu": processed_prompt_tokens / plateau_seconds / case.data_parallel_size,
            "gpu_count": case.data_parallel_size,
        },
        "requests": {
            "whole_run_successes": len(all_records),
            "engine_success_counter_delta": engine_successes,
            "plateau_successes": len(window_records),
            "excluded_before_valid_plateau_successes": excluded_before_valid_plateau,
            "drain_successes": len(drain_records),
            "failures": failures,
            "branch_coverage": {
                "expected": len(required_request_ids),
                "observed": len(branch_coverage),
                "passed": branch_coverage == set(required_request_ids),
            },
            "cohort_plateau_completions": plateau_result["cohort_completions"],
            "final_prefix_provenance": final_prefix_provenance,
            "sampled_token_logprobs": {
                "validated_requests": len(all_records),
                "validated_generated_tokens": sampled_logprob_count,
                "all_completion_tokens_covered": sampled_logprob_count == client_generated_tokens,
            },
        },
        "latency": {
            "client_e2e_seconds": _health_percentiles(
                [record["timing"]["client_e2e_seconds"] for record in window_records]
            ),
            "client_e2e_seconds_by_cohort": cohort_latencies,
            "server_histogram_window": server_latency,
            "client_transport_window": transport_timing,
        },
        "metrics": {
            "rolling_start": rolling_before.relative_path,
            "boundary_start": plateau_before.relative_path,
            "boundary_end": plateau_after.relative_path,
            "final_after_drain": final_snapshot.relative_path,
            "counter_deltas": counter_deltas,
            "prefix_cache": {
                "query_tokens": prefix_queries,
                "hit_tokens": prefix_hits,
                "hit_ratio": prefix_hit_ratio,
            },
            "per_engine_plateau_series": per_engine_plateau_series,
        },
        "kv_cache": kv,
        "moe_routing": {
            "r3_enabled": r3_enabled,
            "fixture": {
                "kind": "vLLM dummy BF16 weights and KV cache",
                "dummy_seed": DUMMY_SEED,
                "experts": case.num_experts,
                "top_k": case.num_experts_per_tok,
                "expert_parallel_size": case.data_parallel_size,
                "placement": "linear contiguous experts per EP rank",
            },
            "expert_histogram": expert_histogram.astype(int).tolist() if r3_enabled else None,
            "ep_rank_histogram": ep_rank_histogram.astype(int).tolist() if r3_enabled else None,
            "ep_rank_load": {
                "mean_assignments": rank_assignment_mean,
                "max_assignments": rank_assignment_max,
                "max_over_mean": rank_assignment_max / rank_assignment_mean if rank_assignment_mean else None,
            },
            "alignment_passed": route_alignment_passed,
            "carrier": {
                "attribution": "R3 response carrier only; generation counter excludes transport bytes",
                "array_bytes": carrier_array_bytes,
                "npy_bytes": carrier_npy_bytes,
                "base64_bytes": carrier_base64_bytes,
                "json_bytes": carrier_json_bytes,
                "full_response_bytes": response_bytes,
                "client_transport_window": transport_timing,
                "base64_bytes_per_engine_generation_token": (
                    carrier_base64_bytes / generated_tokens if generated_tokens else None
                ),
                "json_bytes_per_engine_generation_token": (
                    carrier_json_bytes / generated_tokens if generated_tokens else None
                ),
                "transport": (
                    "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy"
                    if r3_enabled and request_transport == "completion"
                    else (
                        "OpenAI chat JSON choice.routed_experts; generated-token-only nested integer lists"
                        if r3_enabled
                        else "absent"
                    )
                ),
            },
            "balanced_control": {
                "applicable": routing_regime == "balanced",
                "reason": (
                    "deterministic balanced router fixture enabled"
                    if routing_regime == "balanced"
                    else "canonical seeded dummy routing"
                ),
            },
            "aggregate_route_audit": route_audit,
        },
        "resident_capacity": {
            "zero_preemptions": preemptions == 0,
            "by_data_parallel_rank": {
                engine: {
                    "peak_running_requests": values["maximum"],
                    "peak_kv_cache_usage_fraction": (
                        per_engine_plateau_series["vllm:kv_cache_usage_perc"].get(engine, {}).get("maximum")
                    ),
                }
                for engine, values in per_engine_plateau_series["vllm:num_requests_running"].items()
            },
        },
        "whole_run_token_reconciliation": reconciliation,
        "preemptions": preemptions,
    }
    events.emit("arm_completed", arm_id=arm_id, passed=result["passed"], gates=gates)
    return result


def _homogeneous_cohort_schedule(
    workload: dict[str, Any],
    *,
    case: ModelCase,
    cohort: str,
) -> list[dict[str, Any]]:
    cohort_requests = [request for request in workload["requests"] if request["cohort"] == cohort]
    if cohort not in {"short", "medium", "long"} or len(cohort_requests) != 48:
        raise ValueError(f"{cohort}: expected 48 frozen cohort requests")
    return [
        {
            "slot": slot,
            "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
            "request": request,
        }
        for slot, request in enumerate(cohort_requests)
    ]


def _run_homogeneous_cohort_slice(
    base_url: str,
    model: str,
    workload: dict[str, Any],
    *,
    case: ModelCase,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    events: HealthEventWriter,
    log_path: Path,
    arm_id: str,
    cohort: str,
) -> dict[str, Any]:
    """Measure one context point with an engine-counter-bounded homogeneous wave."""
    schedule = _homogeneous_cohort_schedule(workload, case=case, cohort=cohort)
    cohort_requests = [item["request"] for item in schedule]
    slice_concurrency = len(schedule)
    roots = {int(root["root"]): root for root in workload["roots"]}
    population_keys = sorted({(int(item["data_parallel_rank"]), int(item["request"]["root"])) for item in schedule})
    events.emit(
        "cohort_slice_started",
        arm_id=arm_id,
        cohort=cohort,
        slice_concurrency=slice_concurrency,
        schedule_sha256=_sha256_json(
            [
                {
                    "slot": item["slot"],
                    "data_parallel_rank": item["data_parallel_rank"],
                    "request_id": item["request"]["request_id"],
                }
                for item in schedule
            ]
        ),
    )

    def populate(item: tuple[int, int]) -> dict[str, Any]:
        rank, root_index = item
        result = _health_completion(
            base_url,
            model,
            list(roots[root_index]["prefix_token_ids"]),
            case=case,
            max_tokens=1,
            data_parallel_rank=rank,
            request_id=f"{arm_id}-{cohort}-populate-r{rank:02d}-root-{root_index:02d}",
            sampling_seed=_frozen_health_seed(f"slice-populate:{cohort}:{rank}:{root_index}"),
            r3_enabled=False,
        )
        result.pop("route_array", None)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(population_keys)) as executor:
        population = list(executor.map(populate, population_keys))

    gate = threading.Event()
    barrier = threading.Barrier(slice_concurrency + 1)

    event_lock = threading.Lock()

    def one(item: dict[str, Any]) -> dict[str, Any]:
        request = item["request"]
        gate.wait()
        barrier.wait(timeout=60)
        result = _health_completion(
            base_url,
            model,
            materialize_prompt(workload, request),
            case=case,
            max_tokens=int(request["max_tokens"]),
            data_parallel_rank=int(item["data_parallel_rank"]),
            request_id=f"{arm_id}-{cohort}-slot-{int(item['slot']):03d}-{request['request_id']}",
            sampling_seed=_frozen_health_seed(f"slice:{cohort}:{int(item['slot'])}:{request['request_id']}"),
            r3_enabled=False,
        )
        result.pop("route_array", None)
        result.update(
            {
                "manifest_request_id": str(request["request_id"]),
                "cohort": cohort,
                "slot": int(item["slot"]),
                "data_parallel_rank": int(item["data_parallel_rank"]),
            }
        )
        with event_lock:
            events.emit(
                "cohort_slice_request_completed",
                arm_id=arm_id,
                cohort=cohort,
                request_id=result["request_id"],
                manifest_request_id=result["manifest_request_id"],
                data_parallel_rank=result["data_parallel_rank"],
                completion_tokens=result["completion_tokens"],
                prompt_token_ids_sha256=result["prompt_token_ids_sha256"],
                generated_token_ids_sha256=result["generated_token_ids_sha256"],
                final_prefix_token_ids_sha256=result["final_prefix_token_ids_sha256"],
                sampled_token_logprobs_sha256=result["sampled_token_logprobs"]["sha256"],
            )
        return result

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=slice_concurrency)
    futures = [executor.submit(one, item) for item in schedule]
    before = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase=f"cohort-{cohort}-open",
        log_path=log_path,
    )
    try:
        gate.set()
        barrier.wait(timeout=60)
        records = [future.result() for future in futures]
    finally:
        gate.set()
        barrier.abort()
        executor.shutdown(wait=True, cancel_futures=True)

    client_generated_tokens = sum(int(record["completion_tokens"]) for record in records)
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:generation_tokens",
        minimum_delta=client_generated_tokens,
        timeout_seconds=600,
    )
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:request_success",
        minimum_delta=len(records),
        timeout_seconds=600,
    )
    after = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase=f"cohort-{cohort}-close",
        log_path=log_path,
    )
    elapsed_seconds = after.monotonic_seconds - before.monotonic_seconds
    generation_tokens = round(_health_metric_delta(before, after, "vllm:generation_tokens"))
    request_successes = round(_health_metric_delta(before, after, "vllm:request_success"))
    preemptions = round(_health_metric_delta(before, after, "vllm:num_preemptions"))
    if before.leader_log_bytes is None or after.leader_log_bytes is None:
        raise AssertionError("homogeneous slice omitted leader log boundaries")
    kv_source_path = f"metrics/{arm_id}-{cohort}-kv.log"
    log_payload = log_path.read_bytes() if log_path.exists() else b""
    kv_payload = log_payload[before.leader_log_bytes : after.leader_log_bytes]
    kv_source_file = artifact_dir / kv_source_path
    kv_source_file.parent.mkdir(parents=True, exist_ok=True)
    kv_source_file.write_bytes(kv_payload)
    kv = _health_kv_summary(
        kv_source_file,
        start_offset=0,
        case=case,
        target_concurrency=slice_concurrency,
    )
    final_length = int(cohort_requests[0]["final_token_count"])
    schedule_types = layer_types(case.num_hidden_layers, global_interval=case.global_every)
    global_layers = schedule_types.count("full_attention")
    local_layers = len(schedule_types) - global_layers
    predicted_semantic_bytes = predict_kv_bytes(
        sequence_length=final_length,
        local_layers=local_layers,
        global_layers=global_layers,
        local_kv_heads=case.local_kv_heads,
        global_kv_heads=case.global_kv_heads,
        head_dim=case.head_dim,
        sliding_window=case.sliding_window,
    )
    kv["attention_prediction"] = {
        "final_context_tokens": final_length,
        "local_layers": local_layers,
        "global_layers": global_layers,
        "per_live_sequence_bytes": predicted_semantic_bytes,
        "scope": "semantic K and V payload before block rounding",
    }
    kv["source"] = {
        "path": kv_source_path,
        "bytes": len(kv_payload),
        "sha256": hashlib.sha256(kv_payload).hexdigest(),
        "boundary": "vLLM leader log bytes between homogeneous engine-counter snapshots",
    }
    sampled_logprobs = sum(int(record["sampled_token_logprobs"]["count"]) for record in records)
    gates = {
        "engine_client_token_reconciliation": generation_tokens == client_generated_tokens,
        "request_success_reconciliation": request_successes == len(records),
        "sampled_token_logprobs": sampled_logprobs == client_generated_tokens,
        "zero_preemptions": preemptions == 0,
        "kv_instrumentation": kv.get("passed") is True,
        "positive_boundary": elapsed_seconds > 0 and generation_tokens > 0,
    }
    rate = generation_tokens / elapsed_seconds / case.data_parallel_size
    result = {
        "cohort": cohort,
        "passed": all(gates.values()),
        "gates": gates,
        "final_context_tokens": final_length,
        "slice_concurrency": slice_concurrency,
        "population_requests": len(population),
        "measured_requests": len(records),
        "schedule": [
            {
                "slot": int(item["slot"]),
                "data_parallel_rank": int(item["data_parallel_rank"]),
                "request_id": str(item["request"]["request_id"]),
                "prompt_token_ids_sha256": _sha256_json(materialize_prompt(workload, item["request"])),
            }
            for item in schedule
        ],
        "elapsed_seconds": elapsed_seconds,
        "engine_generation_tokens": generation_tokens,
        "generation_tokens_per_second_per_gpu": rate,
        "gpu_seconds_per_generated_token": case.data_parallel_size * elapsed_seconds / generation_tokens,
        "slowdown_from_short_percent": None,
        "preemptions": preemptions,
        "kv_cache": kv,
        "metrics": {
            "boundary_start": before.relative_path,
            "boundary_end": after.relative_path,
            "counter_deltas": {metric: _health_metric_delta(before, after, metric) for metric in HEALTH_COUNTER_METRICS},
            "per_engine_series": _health_engine_series(
                metrics_map,
                first_index=before.index,
                last_index=after.index,
            ),
        },
        "latency": _health_percentiles([float(record["timing"]["client_e2e_seconds"]) for record in records]),
    }
    events.emit(
        "cohort_slice_completed",
        arm_id=arm_id,
        cohort=cohort,
        passed=result["passed"],
        elapsed_seconds=elapsed_seconds,
        generation_tokens=generation_tokens,
        generation_tokens_per_second_per_gpu=rate,
    )
    return result


def _sequence_probe_kv(
    *,
    artifact_dir: Path,
    log_path: Path,
    before: HealthMetricSnapshot,
    after: HealthMetricSnapshot,
    case: ModelCase,
    target_concurrency: int,
    source_path: str,
    boundary: str,
) -> dict[str, Any]:
    if before.leader_log_bytes is None or after.leader_log_bytes is None:
        raise AssertionError("sequence probe omitted leader log boundaries")
    payload = log_path.read_bytes() if log_path.exists() else b""
    log_slice = payload[before.leader_log_bytes : after.leader_log_bytes]
    local_path = artifact_dir / source_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(log_slice)
    kv = _health_kv_summary(
        local_path,
        start_offset=0,
        case=case,
        target_concurrency=target_concurrency,
    )
    kv["source"] = {
        "path": source_path,
        "bytes": len(log_slice),
        "sha256": hashlib.sha256(log_slice).hexdigest(),
        "boundary": boundary,
    }
    return kv


def _run_four_turn_trajectory(
    base_url: str,
    model: str,
    *,
    case: ModelCase,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    events: HealthEventWriter,
    log_path: Path,
    arm_id: str,
) -> dict[str, Any]:
    """Run 144 sequential four-turn branches and carry every answer forward."""
    workload = deterministic_trajectory_workload(seed=DUMMY_SEED)
    roots = {int(root["root"]): root for root in workload["roots"]}

    def populate(root: dict[str, Any]) -> None:
        result = _health_completion(
            base_url,
            model,
            list(root["prefix_token_ids"]),
            case=case,
            max_tokens=1,
            data_parallel_rank=int(root["root"]) % case.data_parallel_size,
            request_id=f"{arm_id}-trajectory-populate-root-{int(root['root']):02d}",
            sampling_seed=_frozen_health_seed(f"trajectory-populate:{int(root['root'])}"),
            r3_enabled=False,
        )
        result.pop("route_array", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(roots)) as executor:
        list(executor.map(populate, roots.values()))

    gate = threading.Event()
    barrier = threading.Barrier(len(workload["requests"]) + 1)
    event_lock = threading.Lock()

    def branch(request: dict[str, Any]) -> list[dict[str, Any]]:
        carried = list(roots[int(request["root"])]["prefix_token_ids"])
        records: list[dict[str, Any]] = []
        gate.wait()
        barrier.wait(timeout=60)
        for turn in request["turns"]:
            append = list(turn["append_token_ids"])
            carried_prefix_sha256 = _sha256_json(carried)
            prompt = [*carried, *append]
            if len(prompt) != int(turn["prompt_token_count"]):
                raise AssertionError(f"{request['request_id']} turn {turn['turn']} prompt length drifted")
            completion = _health_completion(
                base_url,
                model,
                prompt,
                case=case,
                max_tokens=int(turn["max_tokens"]),
                data_parallel_rank=int(request["root"]) % case.data_parallel_size,
                request_id=f"{arm_id}-{request['request_id']}-turn-{int(turn['turn'])}",
                sampling_seed=_frozen_health_seed(f"trajectory:{request['request_id']}:{int(turn['turn'])}"),
                r3_enabled=False,
                keep_generated_tokens=True,
            )
            generated = completion.pop("generated_token_ids")
            completion.pop("route_array", None)
            carried = [*prompt, *generated]
            if len(carried) != int(turn["final_token_count"]):
                raise AssertionError(f"{request['request_id']} turn {turn['turn']} final length drifted")
            record = {
                "request_id": str(request["request_id"]),
                "root": int(request["root"]),
                "branch": int(request["branch"]),
                "cohort": str(request["cohort"]),
                "turn": int(turn["turn"]),
                "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
                "carried_prefix_token_ids_sha256": carried_prefix_sha256,
                "append_token_ids_sha256": _sha256_json(append),
                "prompt_tokens": int(completion["prompt_tokens"]),
                "completion_tokens": int(completion["completion_tokens"]),
                "final_token_count": len(carried),
                "prompt_token_ids_sha256": completion["prompt_token_ids_sha256"],
                "generated_token_ids_sha256": completion["generated_token_ids_sha256"],
                "final_prefix_token_ids_sha256": completion["final_prefix_token_ids_sha256"],
                "sampled_token_logprobs_sha256": completion["sampled_token_logprobs"]["sha256"],
                "sampled_token_logprobs_count": completion["sampled_token_logprobs"]["count"],
            }
            records.append(record)
            with event_lock:
                events.emit("trajectory_turn_completed", arm_id=arm_id, **record)
        return records

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(workload["requests"]))
    futures = [executor.submit(branch, request) for request in workload["requests"]]
    before = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase="trajectory-open",
        log_path=log_path,
    )
    try:
        gate.set()
        barrier.wait(timeout=60)
        records = [record for future in futures for record in future.result()]
    finally:
        gate.set()
        barrier.abort()
        executor.shutdown(wait=True, cancel_futures=True)

    expected_generation = sum(int(record["completion_tokens"]) for record in records)
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:generation_tokens",
        minimum_delta=expected_generation,
        timeout_seconds=1800,
    )
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:request_success",
        minimum_delta=len(records),
        timeout_seconds=1800,
    )
    after = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase="trajectory-close",
        log_path=log_path,
    )
    generation_delta = round(_health_metric_delta(before, after, "vllm:generation_tokens"))
    success_delta = round(_health_metric_delta(before, after, "vllm:request_success"))
    preemptions = round(_health_metric_delta(before, after, "vllm:num_preemptions"))
    kv = _sequence_probe_kv(
        artifact_dir=artifact_dir,
        log_path=log_path,
        before=before,
        after=after,
        case=case,
        target_concurrency=len(workload["requests"]),
        source_path=f"metrics/{arm_id}-trajectory-kv.log",
        boundary="vLLM leader log bytes between four-turn trajectory counter snapshots",
    )
    final_by_cohort = {
        cohort: sorted(
            {
                int(record["final_token_count"])
                for record in records
                if record["cohort"] == cohort and int(record["turn"]) == int(workload["turn_count"])
            }
        )
        for cohort in ("short", "medium", "long")
    }
    gates = {
        "exact_request_turn_count": len(records) == int(workload["request_count"]) * int(workload["turn_count"]),
        "engine_client_token_reconciliation": generation_delta == expected_generation,
        "request_success_reconciliation": success_delta == len(records),
        "sampled_token_logprobs": (
            sum(int(record["sampled_token_logprobs_count"]) for record in records) == expected_generation
        ),
        "answer_carry_chain": all(
            turn == 1
            or record["carried_prefix_token_ids_sha256"]
            == next(
                prior["final_prefix_token_ids_sha256"]
                for prior in records
                if prior["request_id"] == record["request_id"] and int(prior["turn"]) == turn - 1
            )
            for record in records
            for turn in [int(record["turn"])]
        ),
        "final_context_lengths": final_by_cohort == {"short": [22_528], "medium": [43_008], "long": [65_536]},
        "same_root_dp_rank": all(
            int(record["data_parallel_rank"]) == int(record["root"]) % case.data_parallel_size for record in records
        ),
        "zero_preemptions": preemptions == 0,
        "kv_instrumentation": kv.get("passed") is True,
    }
    elapsed = after.monotonic_seconds - before.monotonic_seconds
    return {
        "kind": workload["kind"],
        "passed": all(gates.values()),
        "gates": gates,
        "request_count": int(workload["request_count"]),
        "turn_count": int(workload["turn_count"]),
        "completion_count": len(records),
        "final_lengths_by_cohort": final_by_cohort,
        "engine_generation_tokens": generation_delta,
        "request_successes": success_delta,
        "preemptions": preemptions,
        "elapsed_seconds": elapsed,
        "generation_tokens_per_second_per_gpu": generation_delta / elapsed / case.data_parallel_size,
        "request_provenance": records,
        "kv_cache": kv,
        "metrics": {
            "boundary_start": before.relative_path,
            "boundary_end": after.relative_path,
            "counter_deltas": {metric: _health_metric_delta(before, after, metric) for metric in HEALTH_COUNTER_METRICS},
        },
    }


def _run_capacity_stress_131k(
    base_url: str,
    model: str,
    *,
    case: ModelCase,
    artifact_dir: Path,
    metrics_map: list[dict[str, Any]],
    events: HealthEventWriter,
    log_path: Path,
    arm_id: str,
) -> dict[str, Any]:
    """Attempt one synchronized 48-branch wave ending at 131,072 tokens."""
    workload = deterministic_capacity_stress_workload(seed=DUMMY_SEED)
    roots = {int(root["root"]): root for root in workload["roots"]}

    def populate(root: dict[str, Any]) -> None:
        result = _health_completion(
            base_url,
            model,
            list(root["prefix_token_ids"]),
            case=case,
            max_tokens=1,
            data_parallel_rank=int(root["root"]) % case.data_parallel_size,
            request_id=f"{arm_id}-capacity-populate-root-{int(root['root']):02d}",
            sampling_seed=_frozen_health_seed(f"capacity-populate:{int(root['root'])}"),
            r3_enabled=False,
        )
        result.pop("route_array", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(roots)) as executor:
        list(executor.map(populate, roots.values()))

    gate = threading.Event()
    barrier = threading.Barrier(len(workload["requests"]) + 1)
    event_lock = threading.Lock()

    def one(request: dict[str, Any]) -> dict[str, Any]:
        gate.wait()
        barrier.wait(timeout=60)
        prompt = [*roots[int(request["root"])]["prefix_token_ids"], *request["append_token_ids"]]
        completion = _health_completion(
            base_url,
            model,
            prompt,
            case=case,
            max_tokens=int(request["max_tokens"]),
            data_parallel_rank=int(request["root"]) % case.data_parallel_size,
            request_id=f"{arm_id}-{request['request_id']}",
            sampling_seed=_frozen_health_seed(f"capacity:{request['request_id']}"),
            r3_enabled=False,
        )
        completion.pop("route_array", None)
        record = {
            "request_id": str(request["request_id"]),
            "root": int(request["root"]),
            "branch": int(request["branch"]),
            "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
            "prompt_tokens": int(completion["prompt_tokens"]),
            "completion_tokens": int(completion["completion_tokens"]),
            "final_token_count": int(completion["prompt_tokens"]) + int(completion["completion_tokens"]),
            "prompt_token_ids_sha256": completion["prompt_token_ids_sha256"],
            "generated_token_ids_sha256": completion["generated_token_ids_sha256"],
            "final_prefix_token_ids_sha256": completion["final_prefix_token_ids_sha256"],
            "sampled_token_logprobs_sha256": completion["sampled_token_logprobs"]["sha256"],
            "sampled_token_logprobs_count": completion["sampled_token_logprobs"]["count"],
        }
        with event_lock:
            events.emit("capacity_131k_request_completed", arm_id=arm_id, **record)
        return record

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(workload["requests"]))
    futures = [executor.submit(one, request) for request in workload["requests"]]
    before = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase="capacity-131k-open",
        log_path=log_path,
    )
    try:
        gate.set()
        barrier.wait(timeout=60)
        records = [future.result() for future in futures]
    finally:
        gate.set()
        barrier.abort()
        executor.shutdown(wait=True, cancel_futures=True)

    expected_generation = sum(int(record["completion_tokens"]) for record in records)
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:generation_tokens",
        minimum_delta=expected_generation,
        timeout_seconds=1800,
    )
    _wait_for_health_counter(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        baseline=before,
        metric="vllm:request_success",
        minimum_delta=len(records),
        timeout_seconds=1800,
    )
    after = _capture_health_metrics(
        base_url,
        artifact_dir=artifact_dir,
        metrics_map=metrics_map,
        arm_id=arm_id,
        phase="capacity-131k-close",
        log_path=log_path,
    )
    generation_delta = round(_health_metric_delta(before, after, "vllm:generation_tokens"))
    success_delta = round(_health_metric_delta(before, after, "vllm:request_success"))
    preemptions = round(_health_metric_delta(before, after, "vllm:num_preemptions"))
    kv = _sequence_probe_kv(
        artifact_dir=artifact_dir,
        log_path=log_path,
        before=before,
        after=after,
        case=case,
        target_concurrency=len(records),
        source_path=f"metrics/{arm_id}-capacity-131k-kv.log",
        boundary="vLLM leader log bytes between 131K capacity counter snapshots",
    )
    final_lengths = sorted({int(record["final_token_count"]) for record in records})
    gates = {
        "exact_request_count": len(records) == int(workload["request_count"]),
        "exact_final_context": final_lengths == [CAPACITY_FINAL_TOKENS],
        "engine_client_token_reconciliation": generation_delta == expected_generation,
        "request_success_reconciliation": success_delta == len(records),
        "sampled_token_logprobs": (
            sum(int(record["sampled_token_logprobs_count"]) for record in records) == expected_generation
        ),
        "same_root_dp_rank": all(
            int(record["data_parallel_rank"]) == int(record["root"]) % case.data_parallel_size for record in records
        ),
        "zero_preemptions": preemptions == 0,
        "kv_instrumentation": kv.get("passed") is True,
    }
    elapsed = after.monotonic_seconds - before.monotonic_seconds
    return {
        "kind": workload["kind"],
        "passed": all(gates.values()),
        "gates": gates,
        "request_count": len(records),
        "final_context_tokens": CAPACITY_FINAL_TOKENS,
        "engine_generation_tokens": generation_delta,
        "request_successes": success_delta,
        "preemptions": preemptions,
        "elapsed_seconds": elapsed,
        "generation_tokens_per_second_per_gpu": generation_delta / elapsed / case.data_parallel_size,
        "request_provenance": records,
        "kv_cache": kv,
        "metrics": {
            "boundary_start": before.relative_path,
            "boundary_end": after.relative_path,
            "counter_deltas": {metric: _health_metric_delta(before, after, metric) for metric in HEALTH_COUNTER_METRICS},
        },
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _health_slot_schedule(workload: dict[str, Any], *, target_concurrency: int) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for slot in frozen_cohort_slots(workload["requests"], target_concurrency=target_concurrency):
        cycle_length = len(slot.requests) // math.gcd(len(slot.requests), slot.stride)
        request_ids = [str(slot.next_request()["request_id"]) for _ in range(cycle_length)]
        schedule.append(
            {
                "slot_id": slot.slot_id,
                "cohort": slot.cohort,
                "stride": slot.stride,
                "cyclic_request_ids": request_ids,
            }
        )
    return schedule


def _health_workload_manifest(
    workload: dict[str, Any],
    *,
    case: ModelCase = CASES["exact-reference-ep16"],
    concurrencies: list[int],
    request_transport: str = "completion",
) -> dict[str, Any]:
    warm_up = {
        "kind": (
            "excluded measured-shape shared-root preconditioning with full pass and wave barriers, "
            "including a same-slot successor wave before the prefix-cache reset"
        ),
        "rolling_passes": HEALTH_REPRESENTATIVE_WARM_UP_PASSES,
        "successor_waves_after_final_pass": HEALTH_REPRESENTATIVE_WARM_UP_SUCCESSOR_WAVES,
        "root_copies": _health_warm_root_schedule(workload, case=case),
        "by_concurrency": {
            str(concurrency): _health_warm_up_schedule(workload, case=case, target_concurrency=concurrency)
            for concurrency in sorted(set(concurrencies))
        },
    }
    roots = [
        {
            "root": int(root["root"]),
            "cohort": str(root["cohort"]),
            "token_count": len(root["prefix_token_ids"]),
            "token_ids_sha256": _sha256_json(root["prefix_token_ids"]),
            "data_parallel_rank": int(root["root"]) % case.data_parallel_size,
            "populate_seed": _frozen_health_seed(f"populate:{int(root['root'])}"),
        }
        for root in workload["roots"]
    ]
    requests = [
        {
            "request_id": str(request["request_id"]),
            "root": int(request["root"]),
            "branch": int(request["branch"]),
            "cohort": str(request["cohort"]),
            "prefix_token_count": int(request["prefix_token_count"]),
            "append_token_count": int(request["append_token_count"]),
            "append_token_ids_sha256": _sha256_json(request["append_token_ids"]),
            "prompt_token_ids_sha256": _sha256_json(materialize_prompt(workload, request)),
            "max_tokens": int(request["max_tokens"]),
            "final_token_count": int(request["final_token_count"]),
            "sampling_seed": _frozen_health_seed(f"branch:{request['request_id']}"),
            "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
        }
        for request in workload["requests"]
    ]
    manifest = {
        "schema_version": int(workload["schema_version"]),
        "kind": workload["kind"],
        "generator_seed": int(workload["seed"]),
        "root_count": len(roots),
        "branches_per_root": int(workload["branches_per_root"]),
        "request_count": len(requests),
        "history_lengths": workload["history_lengths"],
        "append_tokens": int(workload["append_tokens"]),
        "response_tokens": int(workload["response_tokens"]),
        "final_lengths": workload["final_lengths"],
        "request_transport": request_transport,
        "sampling_parameters": (
            CHAT_HEALTH_SAMPLING_PARAMETERS if request_transport == "chat" else HEALTH_SAMPLING_PARAMETERS
        ),
        "history_policy": "append-only frozen token history; one response turn in experiment 0",
        "warm_up": warm_up,
        "roots": roots,
        "requests": requests,
        "slot_schedules": {
            str(concurrency): _health_slot_schedule(workload, target_concurrency=concurrency)
            for concurrency in sorted(set(concurrencies))
        },
        "replacement_rule": "each completion is immediately replaced in the same slot from that slot's frozen cycle",
    }
    manifest["frozen_inputs_sha256"] = _sha256_json(
        {
            "generator_seed": manifest["generator_seed"],
            "sampling_parameters": manifest["sampling_parameters"],
            "history_policy": manifest["history_policy"],
            "warm_up": warm_up,
            "roots": roots,
            "requests": requests,
            "slot_schedules": manifest["slot_schedules"],
        }
    )
    return manifest


def _trajectory_workload_manifest(workload: dict[str, Any], *, case: ModelCase) -> dict[str, Any]:
    roots = [
        {
            "root": int(root["root"]),
            "cohort": str(root["cohort"]),
            "token_count": len(root["prefix_token_ids"]),
            "token_ids_sha256": _sha256_json(root["prefix_token_ids"]),
            "data_parallel_rank": int(root["root"]) % case.data_parallel_size,
        }
        for root in workload["roots"]
    ]
    requests = []
    for request in workload["requests"]:
        turns = []
        for turn in request["turns"]:
            turns.append(
                {
                    "turn": int(turn["turn"]),
                    "append_token_count": int(turn["append_token_count"]),
                    "append_token_ids_sha256": _sha256_json(turn["append_token_ids"]),
                    "prompt_token_count": int(turn["prompt_token_count"]),
                    "max_tokens": int(turn["max_tokens"]),
                    "final_token_count": int(turn["final_token_count"]),
                }
            )
        requests.append(
            {
                "request_id": str(request["request_id"]),
                "root": int(request["root"]),
                "branch": int(request["branch"]),
                "cohort": str(request["cohort"]),
                "initial_history_tokens": int(request["initial_history_tokens"]),
                "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
                "turns": turns,
                "final_token_count": int(request["final_token_count"]),
            }
        )
    manifest = {
        "schema_version": int(workload["schema_version"]),
        "kind": workload["kind"],
        "generator_seed": int(workload["seed"]),
        "root_count": int(workload["root_count"]),
        "branches_per_root": int(workload["branches_per_root"]),
        "request_count": int(workload["request_count"]),
        "turn_count": int(workload["turn_count"]),
        "initial_history_lengths": workload["initial_history_lengths"],
        "append_tokens_per_turn": int(workload["append_tokens_per_turn"]),
        "response_tokens_per_turn": int(workload["response_tokens_per_turn"]),
        "final_lengths": workload["final_lengths"],
        "history_policy": "four sequential turns per branch; each generated answer is carried into the next prompt",
        "roots": roots,
        "requests": requests,
    }
    manifest["frozen_inputs_sha256"] = _sha256_json(manifest)
    return manifest


def _capacity_workload_manifest(workload: dict[str, Any], *, case: ModelCase) -> dict[str, Any]:
    roots = [
        {
            "root": int(root["root"]),
            "token_count": len(root["prefix_token_ids"]),
            "token_ids_sha256": _sha256_json(root["prefix_token_ids"]),
            "data_parallel_rank": int(root["root"]) % case.data_parallel_size,
        }
        for root in workload["roots"]
    ]
    requests = [
        {
            "request_id": str(request["request_id"]),
            "root": int(request["root"]),
            "branch": int(request["branch"]),
            "prefix_token_count": int(request["prefix_token_count"]),
            "append_token_count": int(request["append_token_count"]),
            "append_token_ids_sha256": _sha256_json(request["append_token_ids"]),
            "prompt_token_count": int(request["prompt_token_count"]),
            "prompt_token_ids_sha256": _sha256_json(
                [*workload["roots"][int(request["root"])]["prefix_token_ids"], *request["append_token_ids"]]
            ),
            "max_tokens": int(request["max_tokens"]),
            "final_token_count": int(request["final_token_count"]),
            "data_parallel_rank": int(request["root"]) % case.data_parallel_size,
        }
        for request in workload["requests"]
    ]
    manifest = {
        "schema_version": int(workload["schema_version"]),
        "kind": workload["kind"],
        "generator_seed": int(workload["seed"]),
        "root_count": int(workload["root_count"]),
        "branches_per_root": int(workload["branches_per_root"]),
        "request_count": int(workload["request_count"]),
        "history_tokens": int(workload["history_tokens"]),
        "append_tokens": int(workload["append_tokens"]),
        "response_tokens": int(workload["response_tokens"]),
        "final_tokens": int(workload["final_tokens"]),
        "roots": roots,
        "requests": requests,
    }
    manifest["frozen_inputs_sha256"] = _sha256_json(manifest)
    return manifest


def _health_result_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GrugMoE rolling benchmark health",
        "",
        f"Run: `{result['run_id']}`",
        "",
        f"Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "| Arm | R3 | Concurrency | MBT | gen tok/s/GPU | seconds | engine tokens | preemptions | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in result.get("arms", []):
        lines.append(
            (
                "| {arm_id} | {r3} | {concurrency} | {mbt} | {throughput:.3f} | "
                "{seconds:.3f} | {tokens} | {preemptions} | {passed} |"
            ).format(
                arm_id=arm["arm_id"],
                r3="on" if arm["settings"]["r3_enabled"] else "off",
                concurrency=arm["settings"]["target_concurrency"],
                mbt=arm["settings"]["max_num_batched_tokens"],
                throughput=arm["headline"]["generation_tokens_per_second_per_gpu"],
                seconds=arm["plateau"]["elapsed_seconds"],
                tokens=arm["plateau"]["generated_tokens"],
                preemptions=arm["preemptions"],
                passed="yes" if arm["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "The headline rate is the unique `vllm:generation_tokens` counter delta between the recorded "
            "plateau boundaries, divided by boundary time and 16 GPUs. Drain work is excluded.",
            "",
            "See `manifest.json` for frozen tokens, seeds, slot order, provenance, topology, and hashes. "
            "See `metrics/map.json` for every raw Prometheus snapshot.",
            "",
        ]
    )
    return "\n".join(lines)


def _health_repeatability(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate repeated arms whose complete runtime settings are identical."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for arm in arms:
        settings = arm["settings"]
        key = (
            bool(settings["r3_enabled"]),
            str(settings.get("request_transport", "completion")),
            int(settings["target_concurrency"]),
            int(settings["max_num_batched_tokens"]),
            int(settings["max_num_seqs"]),
        )
        groups.setdefault(key, []).append(arm)
    comparisons: list[dict[str, Any]] = []
    for grouped_arms in groups.values():
        if len(grouped_arms) < 2:
            continue
        for left, right in itertools.combinations(grouped_arms, 2):
            left_rate = float(left["headline"]["generation_tokens_per_second_per_gpu"])
            right_rate = float(right["headline"]["generation_tokens_per_second_per_gpu"])
            mean_rate = (left_rate + right_rate) / 2
            delta_percent = 100 * abs(left_rate - right_rate) / mean_rate if mean_rate else math.inf
            comparisons.append(
                {
                    "left_arm": left["arm_id"],
                    "right_arm": right["arm_id"],
                    "left_generation_tokens_per_second_per_gpu": left_rate,
                    "right_generation_tokens_per_second_per_gpu": right_rate,
                    "delta_percent": delta_percent,
                    "limit_percent": 2.0,
                    "passed": delta_percent <= 2.0,
                }
            )
    return {
        "applicable": bool(comparisons),
        "comparisons": comparisons,
        "passed": all(comparison["passed"] for comparison in comparisons),
    }


def _matrix_artifact_prefix(plan: str, run_id: str) -> str:
    if plan == "instrument-v1":
        return f"{HEALTH_ARTIFACT_ROOT}/instrument-v1/{run_id}/"
    if plan == "ep8-calibration":
        return f"{TOPOLOGY_ARTIFACT_ROOT}/calibration/{run_id}/"
    if plan == "topology-v1":
        return f"{TOPOLOGY_ARTIFACT_ROOT}/topology-v1/{run_id}/"
    if plan in {"attention-pair-v1", "attention-finalist-v1"}:
        root = (
            GLOBAL_CADENCE_ARTIFACT_ROOT
            if "global-every4-ep16" in run_id
            else (
                SLIDING_WINDOW_ARTIFACT_ROOT
                if any(candidate in run_id for candidate in ("window1024-ep16", "window2048-ep16"))
                else HEALTH_ARTIFACT_ROOT if "exact-reference-ep16" in run_id else None
            )
        )
        if root is None:
            raise ValueError("attention run ID must include its candidate or finalist case")
        return f"{root}/{plan}/{run_id}/"
    raise ValueError(f"unknown matrix plan: {plan}")


def _matrix_control_prefix(plan: str, run_id: str) -> str:
    return f"{ARTIFACT_ROOT}/matrix-control/{plan}/{run_id}/"


def _verified_calibration_source(
    filesystem: Any,
    *,
    plan: str,
    run_id: str,
    expected_case: str,
    expected_concurrency: int,
    expected_max_num_batched_tokens: int,
    expected_max_num_seqs: int,
    marin_commit: str | None,
    task_image: str,
) -> dict[str, Any]:
    """Verify one frozen calibration result and its independent receipt."""
    artifact_prefix = _matrix_artifact_prefix(plan, run_id)
    manifest_bytes = filesystem.cat_file(_s3_key(f"{artifact_prefix}manifest.json"))
    result_bytes = filesystem.cat_file(_s3_key(f"{artifact_prefix}result.json"))
    receipt_uri = f"{_matrix_control_prefix(plan, run_id)}independent-readback.json"
    receipt_bytes = filesystem.cat_file(_s3_key(receipt_uri))
    manifest = json.loads(manifest_bytes)
    result = json.loads(result_bytes)
    receipt = json.loads(receipt_bytes)
    selected = result.get("analysis", {}).get("calibration", {}).get("selected", {})
    provenance = manifest.get("provenance", {})
    source_sha256 = {
        "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
        "result.json": hashlib.sha256(result_bytes).hexdigest(),
    }
    checks = {
        "identity": (
            manifest.get("plan") == plan
            and result.get("plan") == plan
            and receipt.get("plan") == plan
            and manifest.get("run_id") == run_id
            and result.get("run_id") == run_id
            and receipt.get("run_id") == run_id
        ),
        "passed": (
            result.get("passed") is True and receipt.get("passed") is True and receipt.get("benchmark_passed") is True
        ),
        "selection": (
            selected.get("case") == expected_case
            and selected.get("target_concurrency") == expected_concurrency
            and selected.get("max_num_batched_tokens") == expected_max_num_batched_tokens
            and selected.get("max_num_seqs") == expected_max_num_seqs
        ),
        "source_hashes": all(
            receipt.get("source_object_sha256", {}).get(relative) == digest for relative, digest in source_sha256.items()
        ),
        "provenance": (
            (marin_commit is None or provenance.get("marin_commit") == marin_commit)
            and isinstance(provenance.get("marin_commit"), str)
            and len(provenance["marin_commit"]) == 40
            and provenance.get("vllm_commit") == VLLM_SHA
            and provenance.get("task_image") == task_image
            and receipt.get("reader_marin_commit") == provenance.get("marin_commit")
            and receipt.get("task_image") == task_image
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{plan} calibration source {run_id} failed: {', '.join(failed)}")
    return {
        "plan": plan,
        "run_id": run_id,
        "artifact_prefix": artifact_prefix,
        "independent_readback_uri": receipt_uri,
        "selection": {
            "case": selected["case"],
            "target_concurrency": selected["target_concurrency"],
            "max_num_batched_tokens": selected["max_num_batched_tokens"],
            "max_num_seqs": selected["max_num_seqs"],
        },
        "provenance": {
            "marin_commit": provenance["marin_commit"],
            "vllm_commit": provenance["vllm_commit"],
            "task_image": provenance["task_image"],
        },
        "source_object_sha256": source_sha256,
        "independent_readback_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


def _verified_topology_calibration_sources(
    filesystem: Any,
    *,
    ep8_run_id: str,
    ep16_run_id: str,
    ep8_concurrency: int,
    ep8_max_num_batched_tokens: int,
    ep16_concurrency: int,
    ep16_max_num_batched_tokens: int,
    max_num_seqs: int,
    marin_commit: str,
    task_image: str,
) -> dict[str, Any]:
    return {
        "ep8": _verified_calibration_source(
            filesystem,
            plan="ep8-calibration",
            run_id=ep8_run_id,
            expected_case="reference-ep8",
            expected_concurrency=ep8_concurrency,
            expected_max_num_batched_tokens=ep8_max_num_batched_tokens,
            expected_max_num_seqs=max_num_seqs,
            marin_commit=marin_commit,
            task_image=task_image,
        ),
        "ep16": _verified_calibration_source(
            filesystem,
            plan="instrument-v1",
            run_id=ep16_run_id,
            expected_case="exact-reference-ep16",
            expected_concurrency=ep16_concurrency,
            expected_max_num_batched_tokens=ep16_max_num_batched_tokens,
            expected_max_num_seqs=max_num_seqs,
            marin_commit=marin_commit,
            task_image=task_image,
        ),
    }


def _calibration_selection(arms: list[dict[str, Any]], *, case_name: str) -> dict[str, Any]:
    """Apply the frozen 95%-of-best selection rule without a favorable reroll."""
    candidates = [
        {
            "arm_id": arm["arm_id"],
            "case": arm["matrix"]["case"],
            "target_concurrency": int(arm["settings"]["target_concurrency"]),
            "max_num_batched_tokens": int(arm["settings"]["max_num_batched_tokens"]),
            "max_num_seqs": int(arm["settings"]["max_num_seqs"]),
            "generation_tokens_per_second_per_gpu": float(arm["headline"]["generation_tokens_per_second_per_gpu"]),
            "passed": arm.get("passed") is True,
        }
        for arm in arms
        if arm.get("matrix", {}).get("case") == case_name and arm.get("matrix", {}).get("role") == "calibration"
    ]
    expected_grid = {
        (concurrency, max_num_batched_tokens)
        for concurrency in CALIBRATION_CONCURRENCIES
        for max_num_batched_tokens in CALIBRATION_MAX_NUM_BATCHED_TOKENS
    }
    observed_grid = {(candidate["target_concurrency"], candidate["max_num_batched_tokens"]) for candidate in candidates}
    valid = [candidate for candidate in candidates if candidate["passed"]]
    if observed_grid != expected_grid or len(valid) != len(expected_grid):
        return {
            "passed": False,
            "case": case_name,
            "rule": (
                "among passing candidates at or above 95% of the best tok/s/GPU, choose the lowest "
                "concurrency, then the higher-throughput MBT at that concurrency"
            ),
            "expected_grid": sorted([list(item) for item in expected_grid]),
            "observed_grid": sorted([list(item) for item in observed_grid]),
            "candidates": candidates,
            "error": "the bounded calibration grid was incomplete or had a failing arm",
        }
    best_rate = max(candidate["generation_tokens_per_second_per_gpu"] for candidate in valid)
    threshold = 0.95 * best_rate
    eligible = [candidate for candidate in valid if candidate["generation_tokens_per_second_per_gpu"] >= threshold]
    lowest_concurrency = min(candidate["target_concurrency"] for candidate in eligible)
    at_lowest_concurrency = [
        candidate for candidate in eligible if candidate["target_concurrency"] == lowest_concurrency
    ]
    selected = max(
        at_lowest_concurrency,
        key=lambda candidate: (
            candidate["generation_tokens_per_second_per_gpu"],
            -candidate["max_num_batched_tokens"],
        ),
    )
    return {
        "passed": True,
        "case": case_name,
        "rule": (
            "among passing candidates at or above 95% of the best tok/s/GPU, choose the lowest "
            "concurrency, then the higher-throughput MBT at that concurrency"
        ),
        "best_generation_tokens_per_second_per_gpu": best_rate,
        "threshold_generation_tokens_per_second_per_gpu": threshold,
        "candidates": candidates,
        "eligible_arm_ids": [candidate["arm_id"] for candidate in eligible],
        "selected": selected,
    }


def _matrix_phase(
    phase_id: str,
    *,
    case: str,
    role: str,
    concurrencies: list[int],
    max_num_batched_tokens: int,
    max_num_seqs: int = CALIBRATION_MAX_NUM_SEQS,
    r3_enabled: bool = False,
    request_transport: str = "completion",
    routing_regime: str = "canonical",
    route_audit_mode: str | None = None,
    order: str | None = None,
    replicate: int | None = None,
    homogeneous_slices: bool = False,
    trajectory_65k: bool = False,
    capacity_stress_131k: bool = False,
) -> dict[str, Any]:
    model_case = CASES[case]
    return {
        "phase_id": phase_id,
        "case": case,
        "active_tasks": model_case.node_count,
        "role": role,
        "concurrencies": concurrencies,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "r3_enabled": r3_enabled,
        "request_transport": request_transport,
        "routing_regime": routing_regime,
        "route_audit_mode": route_audit_mode,
        "order": order,
        "replicate": replicate,
        "homogeneous_slices": homogeneous_slices,
        "trajectory_65k": trajectory_65k,
        "capacity_stress_131k": capacity_stress_131k,
    }


def _matrix_initial_phases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.plan == "instrument-v1":
        return [
            _matrix_phase(
                f"ep16-calibration-mbt{max_num_batched_tokens}",
                case="exact-reference-ep16",
                role="calibration",
                concurrencies=list(CALIBRATION_CONCURRENCIES),
                max_num_batched_tokens=max_num_batched_tokens,
            )
            for max_num_batched_tokens in CALIBRATION_MAX_NUM_BATCHED_TOKENS
        ]
    if args.plan == "ep8-calibration":
        return [
            _matrix_phase(
                f"ep8-calibration-mbt{max_num_batched_tokens}",
                case="reference-ep8",
                role="calibration",
                concurrencies=list(CALIBRATION_CONCURRENCIES),
                max_num_batched_tokens=max_num_batched_tokens,
            )
            for max_num_batched_tokens in CALIBRATION_MAX_NUM_BATCHED_TOKENS
        ]
    if args.plan == "attention-pair-v1":
        cases = ("exact-reference-ep16", args.attention_candidate)
        if args.attention_order == "ba":
            cases = tuple(reversed(cases))
        return [
            _matrix_phase(
                f"{args.attention_order}-{position + 1}-{case}",
                case=case,
                role="attention-comparison",
                concurrencies=[args.ep16_concurrency],
                max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                max_num_seqs=args.topology_max_num_seqs,
                order=args.attention_order,
                replicate=1 if args.attention_order == "ab" else 2,
                homogeneous_slices=True,
            )
            for position, case in enumerate(cases)
        ]
    if args.plan == "attention-finalist-v1":
        base_cases = ["exact-reference-ep16"]
        if args.attention_finalist != "exact-reference-ep16":
            base_cases.append(args.attention_finalist)
        return [
            _matrix_phase(
                f"finalist-{position + 1}-{base_case}",
                case=f"{base_case.removesuffix('-ep16')}-131k-ep16",
                role="attention-finalist-validation",
                concurrencies=[args.ep16_concurrency],
                max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                max_num_seqs=args.topology_max_num_seqs,
                order="reference-then-finalist",
                replicate=position + 1,
                trajectory_65k=True,
                capacity_stress_131k=True,
            )
            for position, base_case in enumerate(base_cases)
        ]
    if args.plan != "topology-v1":
        raise ValueError(f"unknown matrix plan: {args.plan}")

    settings = {
        "ep8": {
            "case": "reference-ep8",
            "concurrency": args.ep8_concurrency,
            "max_num_batched_tokens": args.ep8_max_num_batched_tokens,
        },
        "ep16": {
            "case": "exact-reference-ep16",
            "concurrency": args.ep16_concurrency,
            "max_num_batched_tokens": args.ep16_max_num_batched_tokens,
        },
    }
    phases: list[dict[str, Any]] = []
    for routing_regime in ("canonical", "balanced"):
        for order, sequence in (("ab", ("ep8", "ep16")), ("ba", ("ep16", "ep8"))):
            for position, topology in enumerate(sequence):
                selected = settings[topology]
                phases.append(
                    _matrix_phase(
                        f"{routing_regime}-{order}-{position + 1}-{topology}",
                        case=selected["case"],
                        role="topology-comparison",
                        concurrencies=[selected["concurrency"]],
                        max_num_batched_tokens=selected["max_num_batched_tokens"],
                        max_num_seqs=args.topology_max_num_seqs,
                        routing_regime=routing_regime,
                        route_audit_mode="record",
                        order=order,
                        replicate=1 if order == "ab" else 2,
                    )
                )
    phases.extend(
        [
            _matrix_phase(
                "audit-control-noop-ep16",
                case="exact-reference-ep16",
                role="audit-control",
                concurrencies=[args.ep16_concurrency],
                max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                max_num_seqs=args.topology_max_num_seqs,
                route_audit_mode="noop",
            ),
            _matrix_phase(
                "audit-control-record-ep16",
                case="exact-reference-ep16",
                role="audit-control",
                concurrencies=[args.ep16_concurrency],
                max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                max_num_seqs=args.topology_max_num_seqs,
                route_audit_mode="record",
            ),
        ]
    )
    return phases


def _instrument_followup_phases(selection: dict[str, Any]) -> list[dict[str, Any]]:
    if selection.get("passed") is not True:
        return []
    selected = selection["selected"]
    common = {
        "case": "exact-reference-ep16",
        "concurrencies": [int(selected["target_concurrency"])],
        "max_num_batched_tokens": int(selected["max_num_batched_tokens"]),
        "max_num_seqs": int(selected["max_num_seqs"]),
    }
    return [
        _matrix_phase(
            "ep16-r3off-aa",
            role="instrument-aa",
            **{**common, "concurrencies": common["concurrencies"] * 2},
        ),
        _matrix_phase(
            "ep16-chat-r3off",
            role="chat-carrier",
            request_transport="chat",
            r3_enabled=False,
            **common,
        ),
        _matrix_phase(
            "ep16-chat-r3on",
            role="chat-carrier",
            request_transport="chat",
            r3_enabled=True,
            **common,
        ),
    ]


def _topology_followup_phases(args: argparse.Namespace) -> list[dict[str, Any]]:
    common = {
        "case": "reference-ep8",
        "role": "targeted-chat-r3",
        "concurrencies": [int(args.ep8_concurrency)],
        "max_num_batched_tokens": int(args.ep8_max_num_batched_tokens),
        "max_num_seqs": int(args.topology_max_num_seqs),
        "request_transport": "chat",
    }
    return [
        _matrix_phase("targeted-ep8-chat-r3off", r3_enabled=False, **common),
        _matrix_phase("targeted-ep8-chat-r3on", r3_enabled=True, **common),
    ]


def _percent_delta(left: float, right: float) -> float:
    mean = (left + right) / 2
    return 100 * abs(left - right) / mean if mean else math.inf


def _matrix_topology_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_arms = [arm for arm in arms if arm["matrix"]["role"] == "topology-comparison"]
    comparisons: list[dict[str, Any]] = []
    for routing_regime in ("canonical", "balanced"):
        for order in ("ab", "ba"):
            grouped = [
                arm
                for arm in comparison_arms
                if arm["matrix"]["routing_regime"] == routing_regime and arm["matrix"]["order"] == order
            ]
            by_case = {arm["matrix"]["case"]: arm for arm in grouped}
            ep8 = by_case.get("reference-ep8")
            ep16 = by_case.get("exact-reference-ep16")
            if ep8 is None or ep16 is None:
                comparisons.append(
                    {
                        "routing_regime": routing_regime,
                        "order": order,
                        "passed": False,
                        "error": "matched EP8/EP16 arms are incomplete",
                    }
                )
                continue
            ep8_rate = float(ep8["headline"]["generation_tokens_per_second_per_gpu"])
            ep16_rate = float(ep16["headline"]["generation_tokens_per_second_per_gpu"])
            comparisons.append(
                {
                    "routing_regime": routing_regime,
                    "order": order,
                    "ep8_arm": ep8["arm_id"],
                    "ep16_arm": ep16["arm_id"],
                    "ep8_generation_tokens_per_second_per_gpu": ep8_rate,
                    "ep16_generation_tokens_per_second_per_gpu": ep16_rate,
                    "ep8_over_ep16_percent": 100 * (ep8_rate / ep16_rate - 1) if ep16_rate else math.inf,
                    "passed": ep8.get("passed") is True and ep16.get("passed") is True,
                }
            )

    aggregates: list[dict[str, Any]] = []
    repeatability: list[dict[str, Any]] = []
    for routing_regime in ("canonical", "balanced"):
        by_topology: dict[str, list[dict[str, Any]]] = {}
        for arm in comparison_arms:
            if arm["matrix"]["routing_regime"] == routing_regime:
                by_topology.setdefault(arm["matrix"]["case"], []).append(arm)
        means: dict[str, float] = {}
        for case_name in ("reference-ep8", "exact-reference-ep16"):
            grouped = by_topology.get(case_name, [])
            rates = [float(arm["headline"]["generation_tokens_per_second_per_gpu"]) for arm in grouped]
            delta = _percent_delta(rates[0], rates[1]) if len(rates) == 2 else math.inf
            repeatability.append(
                {
                    "routing_regime": routing_regime,
                    "case": case_name,
                    "arm_ids": [arm["arm_id"] for arm in grouped],
                    "rates": rates,
                    "delta_percent": delta,
                    "limit_percent": 2.0,
                    "passed": len(rates) == 2 and delta <= 2.0,
                }
            )
            if rates:
                means[case_name] = sum(rates) / len(rates)
        ep8_mean = means.get("reference-ep8")
        ep16_mean = means.get("exact-reference-ep16")
        aggregates.append(
            {
                "routing_regime": routing_regime,
                "ep8_mean_generation_tokens_per_second_per_gpu": ep8_mean,
                "ep16_mean_generation_tokens_per_second_per_gpu": ep16_mean,
                "ep8_over_ep16_percent": (
                    100 * (ep8_mean / ep16_mean - 1) if ep8_mean is not None and ep16_mean not in {None, 0} else None
                ),
                "winner": (
                    "EP8"
                    if ep8_mean is not None and ep16_mean is not None and ep8_mean > ep16_mean
                    else (
                        "EP16"
                        if ep8_mean is not None and ep16_mean is not None and ep16_mean > ep8_mean
                        else "tie-or-incomplete"
                    )
                ),
            }
        )

    controls = {arm["matrix"]["phase_id"]: arm for arm in arms if arm["matrix"]["role"] == "audit-control"}
    noop = controls.get("audit-control-noop-ep16")
    record = controls.get("audit-control-record-ep16")
    audit_control: dict[str, Any]
    if noop is None or record is None:
        audit_control = {"passed": False, "error": "no-op and record control arms are incomplete"}
    else:
        noop_rate = float(noop["headline"]["generation_tokens_per_second_per_gpu"])
        record_rate = float(record["headline"]["generation_tokens_per_second_per_gpu"])
        audit_control = {
            "passed": noop.get("passed") is True and record.get("passed") is True,
            "noop_arm": noop["arm_id"],
            "record_arm": record["arm_id"],
            "noop_generation_tokens_per_second_per_gpu": noop_rate,
            "record_generation_tokens_per_second_per_gpu": record_rate,
            "record_over_noop_percent": 100 * (record_rate / noop_rate - 1) if noop_rate else math.inf,
            "interpretation": "isolates the fixed aggregate route-counter update from the surrounding router hook",
        }
    wins = {aggregate["routing_regime"]: aggregate["winner"] for aggregate in aggregates}
    ep8_is_finalist = wins == {"canonical": "EP8", "balanced": "EP8"}
    targeted_arms = [arm for arm in arms if arm["matrix"]["role"] == "targeted-chat-r3"]
    if ep8_is_finalist:
        by_r3 = {bool(arm["settings"]["r3_enabled"]): arm for arm in targeted_arms}
        off = by_r3.get(False)
        on = by_r3.get(True)
        if off is None or on is None:
            targeted_chat_r3 = {
                "applicable": True,
                "passed": False,
                "error": "EP8 won both topology controls, but its matched chat R3 pair is incomplete",
            }
        else:
            off_rate = float(off["headline"]["generation_tokens_per_second_per_gpu"])
            on_rate = float(on["headline"]["generation_tokens_per_second_per_gpu"])
            targeted_chat_r3 = {
                "applicable": True,
                "passed": off.get("passed") is True and on.get("passed") is True,
                "name": "production-like chat-R3 carrier overhead",
                "scope": (
                    "timed OpenAI /v1/chat/completions route-return carrier only; this is not a full "
                    "MarinSkyRL rollout or training benchmark"
                ),
                "consumer_commit": MARINSKYRL_CONSUMER_SHA,
                "r3_off_arm": off["arm_id"],
                "r3_on_arm": on["arm_id"],
                "r3_off_generation_tokens_per_second_per_gpu": off_rate,
                "r3_on_generation_tokens_per_second_per_gpu": on_rate,
                "r3_on_over_off_percent": 100 * (on_rate / off_rate - 1) if off_rate else math.inf,
                "r3_on_json_bytes_per_engine_generation_token": on["moe_routing"]["carrier"][
                    "json_bytes_per_engine_generation_token"
                ],
                "only_server_setting_difference": "--enable-return-routed-experts",
                "contract": "generated-token-only nested integer lists, one row per generated token",
            }
    else:
        targeted_chat_r3 = {
            "applicable": False,
            "passed": not targeted_arms,
            "reason": "EP8 did not win both canonical and balanced topology controls",
        }
    if ep8_is_finalist and targeted_chat_r3["passed"]:
        recommendation = (
            "Advance EP8 as the targeted chat-R3 finalist; it wins both topology controls and its matched "
            "chat carrier check passed."
        )
    elif ep8_is_finalist:
        recommendation = "EP8 wins both topology controls, but do not advance it until its matched chat-R3 check passes."
    else:
        recommendation = "Do not advance EP8; the canonical and balanced topology controls do not both favor EP8."
    return {
        "comparisons": comparisons,
        "aggregates": aggregates,
        "repeatability": {
            "comparisons": repeatability,
            "limit_percent": 2.0,
            "passed": bool(repeatability) and all(item["passed"] for item in repeatability),
        },
        "audit_control": audit_control,
        "targeted_chat_r3": targeted_chat_r3,
        "recommendation": recommendation,
        "ep8_is_targeted_chat_r3_finalist": ep8_is_finalist,
        "passed": (
            len(comparisons) == 4
            and all(item["passed"] for item in comparisons)
            and bool(repeatability)
            and all(item["passed"] for item in repeatability)
            and audit_control["passed"] is True
            and targeted_chat_r3["passed"] is True
        ),
    }


def _matrix_attention_pair_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = [arm for arm in arms if arm.get("matrix", {}).get("role") == "attention-comparison"]
    candidates = {arm["matrix"]["case"]: arm for arm in comparison if arm["matrix"]["case"] != "exact-reference-ep16"}
    reference = next(
        (arm for arm in comparison if arm["matrix"]["case"] == "exact-reference-ep16"),
        None,
    )
    if reference is None or len(candidates) != 1:
        return {"passed": False, "error": "attention reference/candidate pair is incomplete"}
    candidate_name, candidate = next(iter(candidates.items()))
    reference_rate = float(reference["headline"]["generation_tokens_per_second_per_gpu"])
    candidate_rate = float(candidate["headline"]["generation_tokens_per_second_per_gpu"])
    reference_curve = {point["cohort"]: point for point in reference.get("coarse_curve", [])}
    candidate_curve = {point["cohort"]: point for point in candidate.get("coarse_curve", [])}
    curves = []
    for cohort in ("short", "medium", "long"):
        ref_point = reference_curve.get(cohort)
        candidate_point = candidate_curve.get(cohort)
        if ref_point is None or candidate_point is None:
            curves.append({"cohort": cohort, "passed": False, "error": "coarse point is missing"})
            continue
        ref_rate = float(ref_point["generation_tokens_per_second_per_gpu"])
        cand_rate = float(candidate_point["generation_tokens_per_second_per_gpu"])
        curves.append(
            {
                "cohort": cohort,
                "final_context_tokens": ref_point["final_context_tokens"],
                "reference_generation_tokens_per_second_per_gpu": ref_rate,
                "candidate_generation_tokens_per_second_per_gpu": cand_rate,
                "candidate_over_reference_percent": 100 * (cand_rate / ref_rate - 1) if ref_rate else math.inf,
                "reference_gpu_seconds_per_generated_token": ref_point["gpu_seconds_per_generated_token"],
                "candidate_gpu_seconds_per_generated_token": candidate_point["gpu_seconds_per_generated_token"],
                "reference_slowdown_from_short_percent": ref_point["slowdown_from_short_percent"],
                "candidate_slowdown_from_short_percent": candidate_point["slowdown_from_short_percent"],
                "reference_kv": ref_point["kv_cache"],
                "candidate_kv": candidate_point["kv_cache"],
                "passed": ref_point.get("passed") is True and candidate_point.get("passed") is True,
            }
        )
    headline_delta = 100 * (candidate_rate / reference_rate - 1) if reference_rate else math.inf
    return {
        "passed": (
            len(comparison) == 2
            and reference.get("passed") is True
            and candidate.get("passed") is True
            and len(curves) == 3
            and all(point.get("passed") is True for point in curves)
        ),
        "candidate": candidate_name,
        "order": reference["matrix"]["order"],
        "reference_arm": reference["arm_id"],
        "candidate_arm": candidate["arm_id"],
        "reference_generation_tokens_per_second_per_gpu": reference_rate,
        "candidate_generation_tokens_per_second_per_gpu": candidate_rate,
        "candidate_over_reference_percent": headline_delta,
        "material_threshold_percent": 5.0,
        "material_in_this_pair": abs(headline_delta) >= 5.0,
        "curve": curves,
        "interpretation": (
            "one same-allocation order only; combine with the reversed fresh-allocation pair before a recommendation"
        ),
    }


def _matrix_attention_finalist_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [arm for arm in arms if arm.get("matrix", {}).get("role") == "attention-finalist-validation"]
    stretch: list[dict[str, Any]] = []
    for arm in validation:
        capacity = arm.get("capacity_stress_131k", {})
        peak = capacity.get("kv_cache", {}).get("peak", {})
        case = CASES[str(arm["matrix"]["case"])]
        physical_per_sequence = peak.get("physical_active_bytes_per_live_sequence")
        pools = peak.get("reserved_physical_bytes_per_engine", {})
        minimum_pool = min((int(value) for value in pools.values()), default=0)
        dense_stored_131k = (
            2 * case.num_hidden_layers * case.num_key_value_heads * case.head_dim * CAPACITY_FINAL_TOKENS * 2
        )
        extrapolated = dense_stored_131k * 2
        impractical = minimum_pool > 0 and extrapolated > minimum_pool
        stretch.append(
            {
                "arm_id": arm["arm_id"],
                "case": arm["matrix"]["case"],
                "measured_131k_physical_bytes_per_live_sequence": physical_per_sequence,
                "current_vllm_dense_stored_kv_bytes_per_sequence_131k": dense_stored_131k,
                "minimum_reserved_physical_bytes_per_engine": minimum_pool,
                "extrapolated_262k_physical_bytes_per_live_sequence": extrapolated,
                "scale_factor": 2.0,
                "status": "impractical-without-serving-or-sharding-change" if impractical else "not-proved-impractical",
                "passed": physical_per_sequence is not None and minimum_pool > 0 and impractical,
            }
        )
    return {
        "passed": (
            bool(validation)
            and len(validation) <= 2
            and all(arm.get("passed") is True for arm in validation)
            and all(arm.get("trajectory_65k", {}).get("passed") is True for arm in validation)
            and all(arm.get("capacity_stress_131k", {}).get("passed") is True for arm in validation)
            and len(stretch) == len(validation)
            and all(item.get("passed") is True for item in stretch)
        ),
        "arms": [arm["arm_id"] for arm in validation],
        "trajectory_65k": [arm.get("trajectory_65k") for arm in validation],
        "capacity_stress_131k": [arm.get("capacity_stress_131k") for arm in validation],
        "stretch_262k": stretch,
        "stretch_interpretation": (
            "262K is not run when the measured 131K physical bytes extrapolate beyond one engine's fixed KV pool; "
            "doing so would require a new serving or sharding topology"
        ),
    }


def _matrix_result(
    *,
    plan: str,
    run_id: str,
    phase_results: list[dict[str, Any]],
    placement: dict[str, Any],
    all_rank_health: dict[str, Any],
    elapsed_seconds: float,
    error: dict[str, str] | None,
) -> dict[str, Any]:
    arms = [arm for phase in phase_results for arm in phase.get("arms", [])]
    phase_health = bool(phase_results) and all(phase.get("passed") is True for phase in phase_results)
    analysis: dict[str, Any]
    expected_phase_count: int
    if plan == "instrument-v1":
        expected_phase_count = 5
        selection = _calibration_selection(arms, case_name="exact-reference-ep16")
        aa_arms = [arm for arm in arms if arm["matrix"]["role"] == "instrument-aa"]
        aa = _health_repeatability(aa_arms)
        carrier_arms = [arm for arm in arms if arm["matrix"]["role"] == "chat-carrier"]
        by_r3 = {bool(arm["settings"]["r3_enabled"]): arm for arm in carrier_arms}
        off = by_r3.get(False)
        on = by_r3.get(True)
        if off is None or on is None:
            carrier = {"passed": False, "error": "matched chat R3-off/R3-on arms are incomplete"}
        else:
            off_rate = float(off["headline"]["generation_tokens_per_second_per_gpu"])
            on_rate = float(on["headline"]["generation_tokens_per_second_per_gpu"])
            carrier = {
                "passed": off.get("passed") is True and on.get("passed") is True,
                "name": "production-like chat-R3 carrier overhead",
                "scope": (
                    "timed OpenAI /v1/chat/completions route-return carrier only; this is not a full "
                    "MarinSkyRL rollout or training benchmark"
                ),
                "consumer_commit": MARINSKYRL_CONSUMER_SHA,
                "r3_off_arm": off["arm_id"],
                "r3_on_arm": on["arm_id"],
                "r3_off_generation_tokens_per_second_per_gpu": off_rate,
                "r3_on_generation_tokens_per_second_per_gpu": on_rate,
                "r3_on_over_off_percent": 100 * (on_rate / off_rate - 1) if off_rate else math.inf,
                "r3_on_json_bytes_per_engine_generation_token": on["moe_routing"]["carrier"][
                    "json_bytes_per_engine_generation_token"
                ],
                "only_server_setting_difference": "--enable-return-routed-experts",
                "contract": "generated-token-only nested integer lists, one row per generated token",
            }
        analysis = {
            "calibration": selection,
            "r3_off_aa": aa,
            "chat_r3_carrier": carrier,
            "passed": selection.get("passed") is True and aa.get("passed") is True and carrier.get("passed") is True,
        }
    elif plan == "ep8-calibration":
        expected_phase_count = 2
        selection = _calibration_selection(arms, case_name="reference-ep8")
        analysis = {"calibration": selection, "passed": selection.get("passed") is True}
    elif plan == "topology-v1":
        analysis = _matrix_topology_summary(arms)
        expected_phase_count = 10 + (2 if analysis["ep8_is_targeted_chat_r3_finalist"] else 0)
    elif plan == "attention-pair-v1":
        analysis = _matrix_attention_pair_summary(arms)
        expected_phase_count = 2
    elif plan == "attention-finalist-v1":
        analysis = _matrix_attention_finalist_summary(arms)
        finalist_cases = {
            str(phase.get("phase", {}).get("case"))
            for phase in phase_results
            if phase.get("phase", {}).get("role") == "attention-finalist-validation"
        }
        expected_phase_count = 1 if finalist_cases == {"exact-reference-131k-ep16"} else 2
    else:
        raise ValueError(f"unknown matrix plan: {plan}")
    passed = (
        error is None
        and placement.get("passed") is True
        and all_rank_health.get("passed") is True
        and phase_health
        and len(phase_results) == expected_phase_count
        and analysis.get("passed") is True
    )
    return {
        "schema_version": 1,
        "kind": "grugmoe-benchmark-matrix",
        "plan": plan,
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "analysis": analysis,
        "arms": arms,
        "phases": phase_results,
        "placement": placement,
        "all_rank_health": all_rank_health,
        "error": error,
        "elapsed_seconds": elapsed_seconds,
    }


def _matrix_result_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# GrugMoE {result['plan']}",
        "",
        f"Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        f"Run: `{result['run_id']}`",
        "",
    ]
    analysis = result["analysis"]
    if result["plan"] in {"instrument-v1", "ep8-calibration"}:
        selection = analysis["calibration"]
        selected = selection.get("selected")
        lines.extend(["## Result", ""])
        if selected:
            lines.append(
                "Selected C{target_concurrency}, MBT {max_num_batched_tokens}, max-num-seqs "
                "{max_num_seqs} at {generation_tokens_per_second_per_gpu:.3f} generated tok/s/GPU.".format(**selected)
            )
        else:
            lines.append("The bounded calibration did not produce a valid selection.")
        if result["plan"] == "instrument-v1":
            carrier = analysis["chat_r3_carrier"]
            if carrier.get("r3_on_over_off_percent") is not None:
                lines.append(
                    "The production-like chat-R3 carrier changed throughput by "
                    f"{carrier['r3_on_over_off_percent']:.3f}% versus matched R3-off."
                )
            lines.append("This times the chat response carrier only. It is not a full MarinSkyRL benchmark.")
    elif result["plan"] == "topology-v1":
        lines.extend(["## Decision", "", analysis["recommendation"], ""])
        lines.extend(
            [
                "| Routing | EP8 mean gen tok/s/GPU | EP16 mean gen tok/s/GPU | EP8 over EP16 | Winner |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for aggregate in analysis["aggregates"]:
            lines.append(
                "| {routing_regime} | {ep8_mean_generation_tokens_per_second_per_gpu:.3f} | "
                "{ep16_mean_generation_tokens_per_second_per_gpu:.3f} | {ep8_over_ep16_percent:.3f}% | "
                "{winner} |".format(**aggregate)
            )
        targeted = analysis["targeted_chat_r3"]
        if targeted.get("applicable"):
            if targeted.get("r3_on_over_off_percent") is not None:
                lines.extend(
                    [
                        "",
                        "The targeted production-like chat-R3 carrier changed EP8 throughput by "
                        f"{targeted['r3_on_over_off_percent']:.3f}% versus matched R3-off.",
                        "This times the chat response carrier only. It is not a full MarinSkyRL benchmark.",
                    ]
                )
    elif result["plan"] == "attention-pair-v1":
        lines.extend(
            [
                "## Same-allocation pair",
                "",
                f"Candidate: `{analysis.get('candidate', 'incomplete')}`; order: "
                f"`{analysis.get('order', 'incomplete')}`.",
                "",
            ]
        )
        if analysis.get("candidate_over_reference_percent") is not None:
            lines.append(f"Headline candidate over reference: {analysis['candidate_over_reference_percent']:.3f}%.")
        lines.extend(
            [
                "This is one pair, not a recommendation. Combine it with the reversed fresh-allocation pair.",
                "",
                "| Context | Ref gen tok/s/GPU | Candidate gen tok/s/GPU | Candidate over ref | "
                "Ref slowdown | Candidate slowdown |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in analysis.get("curve", []):
            if point.get("passed") is not True:
                continue
            lines.append(
                "| {final_context_tokens} | {reference_generation_tokens_per_second_per_gpu:.3f} | "
                "{candidate_generation_tokens_per_second_per_gpu:.3f} | "
                "{candidate_over_reference_percent:.3f}% | "
                "{reference_slowdown_from_short_percent:.3f}% | "
                "{candidate_slowdown_from_short_percent:.3f}% |".format(**point)
            )
    elif result["plan"] == "attention-finalist-v1":
        lines.extend(
            [
                "## Finalist validation",
                "",
                "The reference and finalist each use a fresh server for the four-turn trajectory and 131K stress.",
                "",
                "262K is recorded as impractical only when the 131K physical-KV result extrapolates beyond the "
                "fixed per-engine KV pool.",
            ]
        )
    else:
        raise ValueError(f"unknown matrix plan: {result['plan']}")
    lines.extend(
        [
            "",
            "## Measured arms",
            "",
            "| Arm | Case | Routing | Audit | R3 | C | MBT | gen tok/s/GPU | Preemptions | Pass |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in result["arms"]:
        settings = arm["settings"]
        lines.append(
            "| {arm} | {case} | {routing} | {audit} | {r3} | {concurrency} | {mbt} | {rate:.3f} | "
            "{preemptions} | {passed} |".format(
                arm=arm["arm_id"],
                case=arm["matrix"]["case"],
                routing=settings["routing_regime"],
                audit=settings["route_audit_mode"] or "off",
                r3="on" if settings["r3_enabled"] else "off",
                concurrency=settings["target_concurrency"],
                mbt=settings["max_num_batched_tokens"],
                rate=arm["headline"]["generation_tokens_per_second_per_gpu"],
                preemptions=arm["preemptions"],
                passed="yes" if arm["passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "The headline rate is the engine generation-token counter delta divided by the exact plateau "
            "time and by the arm's GPU count. Drain work and route-audit snapshots are outside that boundary.",
            "",
            "`manifest.json`, `events.jsonl`, and `metrics/map.json` contain the frozen protocol and raw evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _health_result_contract(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    recomputed_repeatability: dict[str, Any],
    result_markdown: str,
) -> dict[str, Any]:
    """Check that the stored benchmark verdict matches the recomputed verdict."""
    arms = result.get("arms", [])
    expected_passed = (
        result.get("error") is None
        and result.get("all_rank_health", {}).get("passed") is True
        and result.get("placement", {}).get("passed") is True
        and all(arm.get("passed") is True for arm in arms)
        and len(arms) == len(manifest["server_settings"]["concurrencies"])
        and recomputed_repeatability.get("passed") is True
    )
    expected_status = "passed" if expected_passed else "failed"
    expected_markdown_status = "PASS" if expected_passed else "FAIL"
    return {
        "passed": (
            result.get("repeatability") == recomputed_repeatability
            and result.get("passed") is expected_passed
            and result.get("status") == expected_status
            and manifest.get("result_aggregate_sha256") == _sha256_json(arms)
            and result.get("run_id") in result_markdown
            and f"Status: **{expected_markdown_status}**" in result_markdown
        ),
        "benchmark_passed": expected_passed,
        "expected_status": expected_status,
        "stored_passed": result.get("passed"),
        "stored_status": result.get("status"),
    }


def _write_and_upload_health_artifacts(
    filesystem: Any,
    *,
    artifact_dir: Path,
    artifact_prefix: str,
    result: dict[str, Any],
    manifest: dict[str, Any],
    metrics_map: list[dict[str, Any]],
    result_markdown: str | None = None,
) -> list[dict[str, Any]]:
    metrics_map_path = artifact_dir / "metrics" / "map.json"
    metrics_map_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_map_path.write_text(
        json.dumps({"schema_version": 1, "snapshots": metrics_map}, indent=2, sort_keys=True) + "\n"
    )
    result_md_path = artifact_dir / "result.md"
    result_md_path.write_text(result_markdown if result_markdown is not None else _health_result_markdown(result))
    kv_source_paths = [str(arm["kv_cache"]["source"]["path"]) for arm in result.get("arms", [])]
    if len(kv_source_paths) != len(set(kv_source_paths)) or any(
        not path.startswith("metrics/") or ".." in Path(path).parts for path in kv_source_paths
    ):
        raise ValueError("health KV evidence paths must be unique files below metrics/")
    claimed_paths = [
        "events.jsonl",
        "metrics/map.json",
        "result.md",
        *[str(entry["path"]) for entry in metrics_map],
        *kv_source_paths,
    ]
    manifest["claimed_files"] = {
        relative: {
            "bytes": (artifact_dir / relative).stat().st_size,
            "sha256": _sha256_path(artifact_dir / relative),
        }
        for relative in claimed_paths
    }
    manifest["result_aggregate_sha256"] = _sha256_json(result.get("arms", []))
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest_sha256"] = _sha256_path(manifest_path)
    result["artifact_contract"] = {
        "prefix": artifact_prefix,
        "required_files": [
            "manifest.json",
            "events.jsonl",
            "metrics/map.json",
            "result.json",
            "result.md",
        ],
        "raw_metrics_glob": "metrics/raw-*.prom",
        "writer_requires_byte_identical_readback": True,
        "independent_reader_required": True,
    }
    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    relative_files = [
        "manifest.json",
        "events.jsonl",
        "metrics/map.json",
        "result.json",
        "result.md",
        *[str(entry["path"]) for entry in metrics_map],
        *kv_source_paths,
    ]
    records = [
        _put_bytes_readback(
            filesystem,
            f"{artifact_prefix.rstrip('/')}/{relative}",
            (artifact_dir / relative).read_bytes(),
        )
        for relative in relative_files
    ]
    if not records or not all(record["readback"] == "identical" for record in records):
        raise OSError("health artifact writer readback was not byte-identical")
    return records


def _upload_rank_evidence(
    filesystem: Any,
    *,
    prefix: str,
    rank: int,
    local_dir: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    status_path = local_dir / f"rank-{rank}.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    records = [
        _put_bytes_readback(
            filesystem,
            f"{prefix}ranks/rank-{rank}.json",
            status_path.read_bytes(),
        )
    ]
    log_path = local_dir / f"vllm-node-{rank}.log"
    if log_path.exists():
        records.append(
            _put_bytes_readback(
                filesystem,
                f"{prefix}ranks/vllm-node-{rank}.log",
                log_path.read_bytes(),
            )
        )
    receipt = {
        "rank": rank,
        "passed": all(record["readback"] == "identical" for record in records),
        "files": records,
    }
    _put_json_readback(
        filesystem,
        f"{prefix}ranks/rank-{rank}-upload.json",
        receipt,
    )
    return receipt


def _smoke_components(
    correctness: dict[str, Any],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "correctness": correctness,
        "duration": {
            "passed": elapsed_seconds > 0,
            "elapsed_seconds": elapsed_seconds,
            "contract": "smoke completed",
        },
        "token_count": {
            "passed": bool(correctness.get("passed")),
            "contract": "correctness requests generated tokens",
        },
        "repeatability": {
            "passed": bool(correctness.get("passed")),
            "contract": "cold and prefix-reused outputs were identical",
        },
    }


def _unattended_placement_component(
    rendezvous: list[dict[str, str]],
    rank_records: list[dict[str, Any]],
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    """Prove distinct node-saturating tasks under the required Kueue topology.

    The CoreWeave cluster runs task pods with ``host_network: true``, so Iris'
    downward-API ``advertise_host`` is the Kubernetes node IP. Each task requests
    all four GPUs on a four-GPU GB200 node. Distinct advertise hosts therefore
    prove distinct nodes even though the K8s backend does not set
    ``IRIS_WORKER_ID``. The coscheduling record is passed to each worker from
    the same ``CoschedulingConfig`` object submitted to Iris, so it proves the
    job requested Kueue's hard ``nvlink.domain`` topology.
    """
    task_indexes = {int(endpoint["task_index"]) for endpoint in rendezvous if endpoint.get("task_index", "").isdigit()}
    advertise_hosts = sorted({endpoint["advertise_host"] for endpoint in rendezvous if endpoint.get("advertise_host")})
    worker_ids = sorted({endpoint["worker_id"] for endpoint in rendezvous if endpoint.get("worker_id")})
    topology_required = expected_tasks == 1 or (
        len(rank_records) == expected_tasks
        and all(record.get("coscheduling") == UNATTENDED_COSCHEDULING for record in rank_records)
    )
    passed = (
        len(rendezvous) == expected_tasks
        and task_indexes == set(range(expected_tasks))
        and len(advertise_hosts) == expected_tasks
        and topology_required
    )
    return {
        "passed": passed,
        "required_coscheduling": UNATTENDED_COSCHEDULING if expected_tasks > 1 else None,
        "topology_enforcement": "Kueue hard podset-required-topology" if expected_tasks > 1 else None,
        "coscheduling_evidence": (
            "worker argument derived from submitted Iris CoschedulingConfig" if expected_tasks > 1 else None
        ),
        "node_identity": "IRIS_ADVERTISE_HOST=status.podIP=node IP because cw-us-east-08a uses host_network=true",
        "node_shape": "one task requests all 4 GPUs on one 4-GPU GB200 node",
        "endpoints": rendezvous,
        "distinct_advertise_hosts": advertise_hosts,
        "distinct_worker_ids": worker_ids,
    }


def _acceptance_components(load: dict[str, Any]) -> dict[str, Any]:
    arms = load["arms"]
    return {
        "duration": {
            "passed": all(
                arm["elapsed_seconds"] >= ACCEPTANCE_MINIMUM_SECONDS and arm["stable_minutes_passed"] for arm in arms
            ),
            "arms": [
                {
                    "elapsed_seconds": arm["elapsed_seconds"],
                    "stable_full_minutes": arm["stable_full_minutes"],
                }
                for arm in arms
            ],
        },
        "token_count": {
            "passed": all(
                arm["generated_tokens"] >= ACCEPTANCE_MINIMUM_GENERATED_TOKENS and arm["branch_coverage"]["passed"]
                for arm in arms
            ),
            "arms": [
                {
                    "generated_tokens": arm["generated_tokens"],
                    "branch_coverage": arm["branch_coverage"]["observed"],
                }
                for arm in arms
            ],
        },
        "repeatability": load["repeatability"],
    }


def _validate_submitted_coscheduling(*, expected_tasks: int, submitted: str | None) -> None:
    required = UNATTENDED_COSCHEDULING if expected_tasks > 1 else None
    if submitted != required:
        raise RuntimeError(f"Iris submitted coscheduling {submitted!r}; expected {required!r}")


def _local_gpu_inventory() -> list[dict[str, str]]:
    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
    )
    inventory: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi inventory line: {line!r}")
        inventory.append(dict(zip(("index", "name", "uuid", "memory_mib"), fields, strict=True)))
    return inventory


def _run_matrix_phase(
    args: argparse.Namespace,
    *,
    info: Any,
    filesystem: Any,
    local_root: Path,
    phase: dict[str, Any],
    phase_index: int,
    events: HealthEventWriter | None,
    metrics_map: list[dict[str, Any]],
) -> dict[str, Any]:
    rank = int(info.task_index)
    phase_id = str(phase["phase_id"])
    case = CASES[str(phase["case"])]
    active = rank < int(phase["active_tasks"])
    phase_dir = local_root / f"phase-{phase_index:02d}-{phase_id}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    write_case(phase_dir, case=case, run_id=f"{args.run_id}-{phase_id}", git_sha=args.marin_commit)
    (phase_dir / "aws-config").write_text(AWS_CONFIG_CONTENT)
    phase_prefix = f"{_matrix_control_prefix(args.plan, args.run_id)}phases/{phase_index:02d}-{phase_id}/"
    stop_uri = f"{phase_prefix}stop.json"
    done_uri = f"{phase_prefix}done.json"
    endpoint_name = f"grugmoe-matrix-{hashlib.sha256(f'{args.run_id}:{phase_index}'.encode()).hexdigest()[:16]}"
    metadata = {
        "task_index": str(rank),
        "num_tasks": str(info.num_tasks),
        "advertise_host": info.advertise_host,
        "phase": phase_id,
    }
    if info.worker_id:
        metadata["worker_id"] = info.worker_id
    server: LocalVllm | None = None
    server_rendezvous: list[dict[str, str]] = []
    startup_records: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    model = ""
    worker_error: dict[str, str] | None = None
    alive_before_stop = False
    started = time.monotonic()
    try:
        if active:
            with iris_ctx().registry.registered(
                endpoint_name,
                f"tcp://{info.advertise_host}:{RPC_PORT}",
                metadata,
            ):
                server_rendezvous = _wait_for_rank_endpoints(
                    endpoint_name,
                    expected=case.node_count,
                    timeout_seconds=args.server_timeout,
                )
                leader = next(endpoint for endpoint in server_rendezvous if int(endpoint["task_index"]) == 0)
                server = _start_local_vllm(
                    case=case,
                    model_source="dummy",
                    model_dir=str(phase_dir),
                    leader_ip=leader["advertise_host"],
                    node_index=rank,
                    smoke=False,
                    local_dir=phase_dir,
                    r3_enabled=bool(phase["r3_enabled"]),
                    max_num_batched_tokens=int(phase["max_num_batched_tokens"]),
                    max_num_seqs=int(phase["max_num_seqs"]),
                    enable_dev_endpoints=True,
                    aggregate_engine_logging=True,
                    request_transport=str(phase["request_transport"]),
                    routing_regime=str(phase["routing_regime"]),
                    route_audit_mode=phase["route_audit_mode"],
                )
                startup = {
                    "rank": rank,
                    "pid": server.process.pid,
                    "worker_id": info.worker_id,
                    "advertise_host": info.advertise_host,
                    "command": server.command,
                    "command_sha256": hashlib.sha256("\0".join(server.command).encode()).hexdigest(),
                    "environment": server.provenance_environment,
                    "alive": server.process.poll() is None,
                }
                _put_json_readback(filesystem, f"{phase_prefix}startup/rank-{rank}.json", startup)
                if rank == 0:
                    startup_records = _wait_for_s3_jsons(
                        filesystem,
                        [f"{phase_prefix}startup/rank-{index}.json" for index in range(case.node_count)],
                        timeout_seconds=args.server_timeout,
                    )
                    if not all(record.get("alive") is True for record in startup_records):
                        raise RuntimeError(f"not every vLLM rank started for {phase_id}: {startup_records}")
                    base_url = f"http://127.0.0.1:{SERVER_PORT}"
                    model = _wait_for_local_server(base_url, server, timeout_seconds=args.server_timeout)
                    if events is None:
                        raise AssertionError("matrix leader omitted its event writer")
                    events.emit(
                        "matrix_phase_started",
                        phase_index=phase_index,
                        phase=phase,
                        model=model,
                    )
                    workload = json.loads((phase_dir / "workload.json").read_text())
                    for arm_index, concurrency in enumerate(phase["concurrencies"]):
                        arm_id = f"p{phase_index:02d}-{phase_id}-a{arm_index:02d}-c{int(concurrency)}"
                        arm = _run_rolling_health_arm(
                            base_url,
                            model,
                            workload,
                            case=case,
                            artifact_dir=local_root,
                            metrics_map=metrics_map,
                            events=events,
                            log_path=server.log_path,
                            arm_id=arm_id,
                            target_concurrency=int(concurrency),
                            minimum_seconds=args.minimum_seconds,
                            minimum_generated_tokens=args.minimum_generated_tokens,
                            r3_enabled=bool(phase["r3_enabled"]),
                            max_num_batched_tokens=int(phase["max_num_batched_tokens"]),
                            max_num_seqs=int(phase["max_num_seqs"]),
                            request_transport=str(phase["request_transport"]),
                            routing_regime=str(phase["routing_regime"]),
                            route_audit_mode=phase["route_audit_mode"],
                        )
                        arm["settings"]["case"] = case.name
                        arm["matrix"] = {
                            "phase_index": phase_index,
                            "phase_id": phase_id,
                            "case": case.name,
                            "role": phase["role"],
                            "active_tasks": case.node_count,
                            "routing_regime": phase["routing_regime"],
                            "order": phase.get("order"),
                            "replicate": phase.get("replicate"),
                            "fresh_server": True,
                            "same_iris_allocation": str(info.job_id),
                        }
                        validation_base_passed = arm.get("passed") is True
                        # Retain the already-bounded rolling result even if a
                        # later diagnostic probe fails or kills the server.
                        arms.append(arm)
                        if phase.get("homogeneous_slices") and arm.get("passed") is True:
                            curve = [
                                _run_homogeneous_cohort_slice(
                                    base_url,
                                    model,
                                    workload,
                                    case=case,
                                    artifact_dir=local_root,
                                    metrics_map=metrics_map,
                                    events=events,
                                    log_path=server.log_path,
                                    arm_id=arm_id,
                                    cohort=cohort,
                                )
                                for cohort in ("short", "medium", "long")
                            ]
                            short_rate = float(curve[0]["generation_tokens_per_second_per_gpu"])
                            for point in curve:
                                point["slowdown_from_short_percent"] = (
                                    100 * (1 - float(point["generation_tokens_per_second_per_gpu"]) / short_rate)
                                    if short_rate
                                    else math.inf
                                )
                            arm["coarse_curve"] = curve
                            arm["gates"]["homogeneous_slices"] = all(point.get("passed") is True for point in curve)
                            arm["passed"] = arm["passed"] and arm["gates"]["homogeneous_slices"]
                        if phase.get("trajectory_65k") and validation_base_passed:
                            trajectory = _run_four_turn_trajectory(
                                base_url,
                                model,
                                case=case,
                                artifact_dir=local_root,
                                metrics_map=metrics_map,
                                events=events,
                                log_path=server.log_path,
                                arm_id=arm_id,
                            )
                            arm["trajectory_65k"] = trajectory
                            arm["gates"]["trajectory_65k"] = trajectory.get("passed") is True
                            arm["passed"] = arm["passed"] and arm["gates"]["trajectory_65k"]
                        if phase.get("capacity_stress_131k") and validation_base_passed:
                            capacity = _run_capacity_stress_131k(
                                base_url,
                                model,
                                case=case,
                                artifact_dir=local_root,
                                metrics_map=metrics_map,
                                events=events,
                                log_path=server.log_path,
                                arm_id=arm_id,
                            )
                            arm["capacity_stress_131k"] = capacity
                            arm["gates"]["capacity_stress_131k"] = capacity.get("passed") is True
                            arm["passed"] = arm["passed"] and arm["gates"]["capacity_stress_131k"]
                        if arm.get("passed") is not True:
                            break
                    alive_before_stop = server.process.poll() is None
                else:
                    _wait_for_s3_jsons(filesystem, [stop_uri], timeout_seconds=args.server_timeout * 2)
                    alive_before_stop = server.process.poll() is None
        else:
            _wait_for_s3_jsons(filesystem, [stop_uri], timeout_seconds=args.server_timeout * 2)
    except Exception as exc:
        worker_error = {"type": type(exc).__name__, "message": str(exc)}
        if events is not None:
            events.emit("matrix_phase_failed", phase_index=phase_index, phase_id=phase_id, **worker_error)
    finally:
        if rank == 0:
            try:
                _put_json_readback(
                    filesystem,
                    stop_uri,
                    {"rank": rank, "run_id": args.run_id, "phase_id": phase_id, "error": worker_error},
                )
            except Exception as exc:
                worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}
        if server is not None:
            alive_before_stop = alive_before_stop or server.process.poll() is None
            _stop_local_vllm(server)
        rank_status = {
            "rank": rank,
            "active": active,
            "task_count": info.num_tasks,
            "active_task_count": case.node_count,
            "phase_index": phase_index,
            "phase_id": phase_id,
            "case": case.name,
            "worker_id": info.worker_id,
            "advertise_host": info.advertise_host,
            "job_id": str(info.job_id),
            "task_id": str(info.task_id),
            "task_image": args.task_image,
            "marin_commit": args.marin_commit,
            "vllm_commit": VLLM_SHA,
            "coscheduling": args.submitted_coscheduling,
            "vllm_command": server.command if server is not None else None,
            "vllm_environment": server.provenance_environment if server is not None else None,
            "vllm_alive_before_stop": alive_before_stop,
            "vllm_returncode_after_stop": server.process.returncode if server is not None else None,
            "error": worker_error,
            "elapsed_seconds": time.monotonic() - started,
        }
        try:
            _upload_rank_evidence(
                filesystem,
                prefix=phase_prefix,
                rank=rank,
                local_dir=phase_dir,
                status=rank_status,
            )
        except Exception as exc:
            worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}

    if rank == 0:
        try:
            rank_records = _wait_for_s3_jsons(
                filesystem,
                [f"{phase_prefix}ranks/rank-{index}.json" for index in range(info.num_tasks)],
                timeout_seconds=300,
            )
            rank_receipts = _wait_for_s3_jsons(
                filesystem,
                [f"{phase_prefix}ranks/rank-{index}-upload.json" for index in range(info.num_tasks)],
                timeout_seconds=300,
            )
        except Exception as exc:
            rank_records = [rank_status]
            rank_receipts = []
            worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}
        active_rank_records = [record for record in rank_records if record.get("active") is True]
        placement = _unattended_placement_component(
            server_rendezvous,
            active_rank_records,
            expected_tasks=case.node_count,
        )
        all_rank_health = {
            "passed": (
                len(rank_records) == info.num_tasks
                and len(rank_receipts) == info.num_tasks
                and all(receipt.get("passed") is True for receipt in rank_receipts)
                and all(
                    record.get("error") is None
                    and (record.get("active") is False or record.get("vllm_alive_before_stop") is True)
                    for record in rank_records
                )
            ),
            "ranks": rank_records,
            "upload_receipts": rank_receipts,
        }
        phase_passed = (
            worker_error is None
            and placement.get("passed") is True
            and all_rank_health["passed"] is True
            and len(arms) == len(phase["concurrencies"])
            and all(arm.get("passed") is True for arm in arms)
        )
        phase_result = {
            "schema_version": 1,
            "phase": phase,
            "phase_index": phase_index,
            "phase_id": phase_id,
            "model": model,
            "passed": phase_passed,
            "arms": arms,
            "startup": startup_records,
            "placement": placement,
            "all_rank_health": all_rank_health,
            "error": worker_error,
            "elapsed_seconds": time.monotonic() - started,
        }
        _put_json_readback(filesystem, done_uri, phase_result)
        if events is not None:
            events.emit(
                "matrix_phase_completed",
                phase_index=phase_index,
                phase_id=phase_id,
                passed=phase_passed,
            )
    return _wait_for_s3_jsons(filesystem, [done_uri], timeout_seconds=300)[0]


def run_matrix_worker(args: argparse.Namespace) -> dict[str, Any]:
    _validate_matrix_args(args)
    info = get_job_info()
    if info is None:
        raise RuntimeError("matrix worker must run inside an Iris job")
    expected_tasks = 2 if args.plan == "ep8-calibration" else 4
    if info.num_tasks != expected_tasks:
        raise RuntimeError(f"{args.plan} requires {expected_tasks} Iris tasks, got {info.num_tasks}")
    _validate_submitted_coscheduling(expected_tasks=expected_tasks, submitted=args.submitted_coscheduling)
    rank = int(info.task_index)
    local_root = Path(REMOTE_ROOT) / f"matrix-{args.plan}-{args.run_id}" / f"rank-{rank}"
    local_root.mkdir(parents=True, exist_ok=True)
    (local_root / "aws-config").write_text(AWS_CONFIG_CONTENT)
    os.environ["AWS_CONFIG_FILE"] = str(local_root / "aws-config")
    filesystem = _s3_filesystem()
    control_prefix = _matrix_control_prefix(args.plan, args.run_id)
    artifact_prefix = _matrix_artifact_prefix(args.plan, args.run_id)
    endpoint_name = f"grugmoe-matrix-job-{hashlib.sha256(args.run_id.encode()).hexdigest()[:16]}"
    metadata = {
        "task_index": str(rank),
        "num_tasks": str(info.num_tasks),
        "advertise_host": info.advertise_host,
    }
    if info.worker_id:
        metadata["worker_id"] = info.worker_id
    started = time.monotonic()
    events = HealthEventWriter(local_root / "events.jsonl") if rank == 0 else None
    metrics_map: list[dict[str, Any]] = []
    phase_results: list[dict[str, Any]] = []
    phases = _matrix_initial_phases(args)
    executed_phases: list[dict[str, Any]] = []
    calibration_sources: dict[str, Any] = {}
    gpu_inventory: list[dict[str, str]] = []
    job_rendezvous: list[dict[str, str]] = []
    worker_error: dict[str, str] | None = None
    try:
        if args.plan == "topology-v1":
            calibration_sources = _verified_topology_calibration_sources(
                filesystem,
                ep8_run_id=args.ep8_calibration_run_id,
                ep16_run_id=args.ep16_instrument_run_id,
                ep8_concurrency=args.ep8_concurrency,
                ep8_max_num_batched_tokens=args.ep8_max_num_batched_tokens,
                ep16_concurrency=args.ep16_concurrency,
                ep16_max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                max_num_seqs=args.topology_max_num_seqs,
                marin_commit=args.marin_commit,
                task_image=args.task_image,
            )
        elif args.plan in {"attention-pair-v1", "attention-finalist-v1"}:
            calibration_sources = {
                "ep16": _verified_calibration_source(
                    filesystem,
                    plan="instrument-v1",
                    run_id=args.ep16_instrument_run_id,
                    expected_case="exact-reference-ep16",
                    expected_concurrency=args.ep16_concurrency,
                    expected_max_num_batched_tokens=args.ep16_max_num_batched_tokens,
                    expected_max_num_seqs=args.topology_max_num_seqs,
                    marin_commit=None,
                    task_image=args.task_image,
                )
            }
        gpu_inventory = _local_gpu_inventory()
        if len(gpu_inventory) != LOCAL_DP_SIZE or not all("GB200" in gpu["name"] for gpu in gpu_inventory):
            raise RuntimeError(f"rank {rank} did not receive four GB200 GPUs: {gpu_inventory}")
        with iris_ctx().registry.registered(
            endpoint_name,
            f"tcp://{info.advertise_host}:{RPC_PORT}",
            metadata,
        ):
            job_rendezvous = _wait_for_rank_endpoints(
                endpoint_name,
                expected=info.num_tasks,
                timeout_seconds=args.server_timeout,
            )
            phase_index = 0
            while phase_index < len(phases):
                phase = phases[phase_index]
                executed_phases.append(phase)
                phase_result = _run_matrix_phase(
                    args,
                    info=info,
                    filesystem=filesystem,
                    local_root=local_root,
                    phase=phase,
                    phase_index=phase_index,
                    events=events,
                    metrics_map=metrics_map,
                )
                phase_results.append(phase_result)
                if phase_result.get("passed") is not True:
                    break
                phase_index += 1
                if args.plan == "instrument-v1" and phase_index == 2 and len(phases) == 2:
                    calibration_arms = [arm for result in phase_results for arm in result.get("arms", [])]
                    selection = _calibration_selection(calibration_arms, case_name="exact-reference-ep16")
                    phases.extend(_instrument_followup_phases(selection))
                    if selection.get("passed") is not True:
                        break
                if args.plan == "topology-v1" and phase_index == 10 and len(phases) == 10:
                    topology_arms = [arm for result in phase_results for arm in result.get("arms", [])]
                    summary = _matrix_topology_summary(topology_arms)
                    if summary["ep8_is_targeted_chat_r3_finalist"]:
                        phases.extend(_topology_followup_phases(args))
    except Exception as exc:
        worker_error = {"type": type(exc).__name__, "message": str(exc)}
        if events is not None:
            events.emit("matrix_worker_failed", rank=rank, **worker_error)

    rank_status = {
        "rank": rank,
        "task_count": info.num_tasks,
        "worker_id": info.worker_id,
        "advertise_host": info.advertise_host,
        "job_id": str(info.job_id),
        "task_id": str(info.task_id),
        "task_image": args.task_image,
        "marin_commit": args.marin_commit,
        "vllm_commit": VLLM_SHA,
        "coscheduling": args.submitted_coscheduling,
        "gpu_inventory": gpu_inventory,
        "phase_ids": [result["phase_id"] for result in phase_results],
        "phase_passes": [result.get("passed") for result in phase_results],
        "error": worker_error,
        "elapsed_seconds": time.monotonic() - started,
    }
    try:
        _upload_rank_evidence(
            filesystem,
            prefix=control_prefix,
            rank=rank,
            local_dir=local_root,
            status=rank_status,
        )
    except Exception as exc:
        worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}

    if rank != 0:
        if worker_error:
            raise RuntimeError(worker_error["message"])
        return rank_status

    try:
        rank_records = _wait_for_s3_jsons(
            filesystem,
            [f"{control_prefix}ranks/rank-{index}.json" for index in range(info.num_tasks)],
            timeout_seconds=300,
        )
        rank_receipts = _wait_for_s3_jsons(
            filesystem,
            [f"{control_prefix}ranks/rank-{index}-upload.json" for index in range(info.num_tasks)],
            timeout_seconds=300,
        )
    except Exception as exc:
        rank_records = [rank_status]
        rank_receipts = []
        worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}
    placement = _unattended_placement_component(job_rendezvous, rank_records, expected_tasks=info.num_tasks)
    all_rank_health = {
        "passed": (
            len(rank_records) == info.num_tasks
            and len(rank_receipts) == info.num_tasks
            and all(record.get("error") is None for record in rank_records)
            and all(receipt.get("passed") is True for receipt in rank_receipts)
            and len({record.get("job_id") for record in rank_records}) == 1
        ),
        "ranks": rank_records,
        "upload_receipts": rank_receipts,
    }
    result = _matrix_result(
        plan=args.plan,
        run_id=args.run_id,
        phase_results=phase_results,
        placement=placement,
        all_rank_health=all_rank_health,
        elapsed_seconds=time.monotonic() - started,
        error=worker_error,
    )
    if events is None:
        events = HealthEventWriter(local_root / "events.jsonl")
    events.emit(
        "matrix_worker_completed",
        plan=args.plan,
        passed=result["passed"],
        phases=len(phase_results),
        arms=len(result["arms"]),
    )
    events.close()
    workload_cases = sorted({phase["case"] for phase in executed_phases})
    manifest = {
        "schema_version": 1,
        "kind": "grugmoe-benchmark-matrix-manifest",
        "plan": args.plan,
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_prefix": artifact_prefix,
        "protocol": {
            "minimum_plateau_seconds": args.minimum_seconds,
            "minimum_plateau_engine_generation_tokens": args.minimum_generated_tokens,
            "minimum_in_flight_fraction": 0.95,
            "drain_excluded": True,
            "headline_counter": "vllm:generation_tokens",
            "calibration_concurrencies": list(CALIBRATION_CONCURRENCIES),
            "calibration_max_num_batched_tokens": list(CALIBRATION_MAX_NUM_BATCHED_TOKENS),
            "calibration_selection": (
                "among passing candidates at or above 95% of the best tok/s/GPU, choose the lowest "
                "concurrency, then the higher-throughput MBT at that concurrency"
            ),
            "failure_policy": "zero retry; preserve the first bounded result; no favorable reroll",
            "attention_curve": {
                "applicable": args.plan == "attention-pair-v1",
                "cohorts": ["short", "medium", "long"] if args.plan == "attention-pair-v1" else [],
                "requests_per_cohort": 48 if args.plan == "attention-pair-v1" else 0,
                "root_affinity": "all eight branches of a root use root modulo 16 as their DP rank",
                "boundary": "one 48-request homogeneous wave bounded by vllm:generation_tokens snapshots",
                "headline_unchanged": True,
            },
            "attention_finalist": {
                "applicable": args.plan == "attention-finalist-v1",
                "trajectory": "18 roots, 8 branches per root, four sequential 1024+2048 turns with answer carry",
                "capacity": "48 branches, 121856 cached + 1024 append + 8192 generation = 131072",
                "stretch_262k": "run only without a new serving or sharding topology; otherwise extrapolate from 131K",
            },
        },
        "phase_plan": executed_phases,
        "calibration_sources": calibration_sources,
        "model_configs": {name: dataclasses.asdict(CASES[name]) for name in workload_cases},
        "workloads": {
            f"{name}:{request_transport}": _health_workload_manifest(
                deterministic_workload(seed=DUMMY_SEED),
                case=CASES[name],
                concurrencies=sorted(
                    {
                        int(concurrency)
                        for phase in executed_phases
                        if phase["case"] == name and phase["request_transport"] == request_transport
                        for concurrency in phase["concurrencies"]
                    }
                ),
                request_transport=request_transport,
            )
            for name in workload_cases
            for request_transport in sorted(
                {phase["request_transport"] for phase in executed_phases if phase["case"] == name}
            )
        },
        "trajectory_workloads": (
            {
                name: _trajectory_workload_manifest(
                    deterministic_trajectory_workload(seed=DUMMY_SEED),
                    case=CASES[name],
                )
                for name in workload_cases
            }
            if args.plan == "attention-finalist-v1"
            else {}
        ),
        "capacity_workloads": (
            {
                name: _capacity_workload_manifest(
                    deterministic_capacity_stress_workload(seed=DUMMY_SEED),
                    case=CASES[name],
                )
                for name in workload_cases
            }
            if args.plan == "attention-finalist-v1"
            else {}
        ),
        "model_fixture": {
            "source": "dummy",
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "seed": DUMMY_SEED,
        },
        "routing": {
            "canonical": "seeded vLLM dummy router with linear contiguous expert placement",
            "balanced": (
                "deterministic token-major round-robin expert fixture with unchanged normalized weights and a "
                "layer-rotated integer remainder"
            ),
            "response_routes_in_topology_headlines": False,
            "aggregate_audit_source": (
                "fixed GPU int64 counters in the existing GrugMoE router capture hook; one worker D2H "
                "snapshot after each arm"
            ),
            "aggregate_audit_boundary": "reset before first measured request; snapshot after drain",
            "capacity_factor": None,
        },
        "implementation_controls": {
            "new_hot_path_family": args.plan == "topology-v1",
            "no_op_control": {
                "applicable": args.plan == "topology-v1",
                "phase": "audit-control-noop-ep16" if args.plan == "topology-v1" else None,
            },
        },
        "chat_r3": {
            "consumer_commit": MARINSKYRL_CONSUMER_SHA,
            "endpoint": "/v1/chat/completions",
            "contract": "generated-token-only nested integer lists, one row per generated token",
            "scope_name": "production-like chat-R3 carrier overhead",
            "not_full_marinskyrl": True,
        },
        "placement": placement,
        "provenance": {
            "marin_commit": args.marin_commit,
            "marin_commit_url": f"https://github.com/marin-community/marin/commit/{args.marin_commit}",
            "vllm_commit": VLLM_SHA,
            "vllm_commit_url": f"https://github.com/marin-community/vllm/commit/{VLLM_SHA}",
            "task_image": args.task_image,
            "dependency_lock_sha256": _sha256_path(Path("uv.lock")),
            "cluster_config": DEFAULT_CLUSTER_CONFIG,
            "iris_job_id": str(info.job_id),
            "iris_task_count": info.num_tasks,
            "iris_priority": args.iris_priority,
            "iris_coscheduling": args.submitted_coscheduling,
            "iris_retry_policy": {
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
            },
        },
        "phase_result_sha256": _sha256_json(phase_results),
    }
    try:
        records = _write_and_upload_health_artifacts(
            filesystem,
            artifact_dir=local_root,
            artifact_prefix=artifact_prefix,
            result=result,
            manifest=manifest,
            metrics_map=metrics_map,
            result_markdown=_matrix_result_markdown(result),
        )
        _put_json_readback(
            filesystem,
            f"{control_prefix}writer-readback.json",
            {"passed": True, "artifact_prefix": artifact_prefix, "files": records},
        )
    except Exception as exc:
        _put_json_readback(
            filesystem,
            f"{control_prefix}writer-readback.json",
            {
                "passed": False,
                "artifact_prefix": artifact_prefix,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    if not result["passed"]:
        raise RuntimeError(json.dumps(result.get("error") or result["analysis"]))
    return result


def run_health_unattended_worker(args: argparse.Namespace) -> dict[str, Any]:
    info = get_job_info()
    if info is None:
        raise RuntimeError("health worker must run inside an Iris job")
    case = CASES[args.case]
    if case.name != "exact-reference-ep16" or info.num_tasks != 4:
        raise RuntimeError("rolling health requires exact-reference-ep16 on exactly four Iris tasks")
    if args.model_source != "dummy":
        raise RuntimeError("experiment 0 rolling health is frozen to dummy BF16 weights and KV")
    _validate_submitted_coscheduling(expected_tasks=info.num_tasks, submitted=args.submitted_coscheduling)
    rank = info.task_index
    run_id = args.run_id
    r3_enabled = args.r3 == "on"
    request_transport = args.request_transport
    concurrencies = list(args.concurrency or [48])
    max_num_seqs = args.max_num_seqs or max(concurrencies)
    artifact_prefix = f"{HEALTH_ARTIFACT_ROOT}/{run_id}/"
    control_prefix = f"{ARTIFACT_ROOT}/health-control/{run_id}/"
    local_dir = Path(REMOTE_ROOT) / f"health-{run_id}" / f"rank-{rank}"
    local_dir.mkdir(parents=True, exist_ok=True)
    write_case(local_dir, case=case, run_id=run_id, git_sha=args.marin_commit)
    (local_dir / "aws-config").write_text(AWS_CONFIG_CONTENT)
    os.environ["AWS_CONFIG_FILE"] = str(local_dir / "aws-config")
    filesystem = _s3_filesystem()
    endpoint_name = f"grugmoe-health-ranks-{run_id}"
    metadata = {
        "task_index": str(rank),
        "num_tasks": str(info.num_tasks),
        "advertise_host": info.advertise_host,
    }
    if info.worker_id:
        metadata["worker_id"] = info.worker_id
    server: LocalVllm | None = None
    rendezvous: list[dict[str, str]] = []
    events: HealthEventWriter | None = None
    metrics_map: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    worker_error: dict[str, str] | None = None
    alive_before_stop = False
    stop_uri = f"{control_prefix}stop.json"
    started = time.monotonic()
    gpu_inventory: list[dict[str, str]] = []
    model = ""
    try:
        gpu_inventory = _local_gpu_inventory()
        if len(gpu_inventory) != LOCAL_DP_SIZE or not all("GB200" in gpu["name"] for gpu in gpu_inventory):
            raise RuntimeError(f"rank {rank} did not receive four GB200 GPUs: {gpu_inventory}")
        with iris_ctx().registry.registered(
            endpoint_name,
            f"tcp://{info.advertise_host}:{RPC_PORT}",
            metadata,
        ):
            rendezvous = _wait_for_rank_endpoints(
                endpoint_name,
                expected=info.num_tasks,
                timeout_seconds=args.server_timeout,
            )
            leader = next(endpoint for endpoint in rendezvous if int(endpoint["task_index"]) == 0)
            server = _start_local_vllm(
                case=case,
                model_source=args.model_source,
                model_dir=str(local_dir),
                leader_ip=leader["advertise_host"],
                node_index=rank,
                smoke=False,
                local_dir=local_dir,
                r3_enabled=r3_enabled,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                enable_dev_endpoints=True,
                aggregate_engine_logging=True,
                request_transport=request_transport,
            )
            startup = {
                "rank": rank,
                "pid": server.process.pid,
                "worker_id": info.worker_id,
                "advertise_host": info.advertise_host,
                "command": server.command,
                "command_sha256": hashlib.sha256("\0".join(server.command).encode()).hexdigest(),
                "gpu_inventory": gpu_inventory,
                "alive": server.process.poll() is None,
            }
            _put_json_readback(filesystem, f"{control_prefix}startup/rank-{rank}.json", startup)
            if rank != 0:
                while not filesystem.exists(_s3_key(stop_uri)):
                    if server.process.poll() is not None:
                        raise RuntimeError(f"rank {rank} vLLM exited early with {server.process.returncode}")
                    time.sleep(5)
                alive_before_stop = server.process.poll() is None
            else:
                startup_records = _wait_for_s3_jsons(
                    filesystem,
                    [f"{control_prefix}startup/rank-{index}.json" for index in range(info.num_tasks)],
                    timeout_seconds=args.server_timeout,
                )
                if not all(record["alive"] for record in startup_records):
                    raise RuntimeError(f"not every vLLM rank started: {startup_records}")
                base_url = f"http://127.0.0.1:{SERVER_PORT}"
                model = _wait_for_local_server(base_url, server, timeout_seconds=args.server_timeout)
                events = HealthEventWriter(local_dir / "events.jsonl")
                events.emit(
                    "worker_ready",
                    run_id=run_id,
                    model=model,
                    r3_enabled=r3_enabled,
                    request_transport=request_transport,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    concurrencies=concurrencies,
                )
                workload = json.loads((local_dir / "workload.json").read_text())
                for arm_index, concurrency in enumerate(concurrencies):
                    arm = _run_rolling_health_arm(
                        base_url,
                        model,
                        workload,
                        case=case,
                        artifact_dir=local_dir,
                        metrics_map=metrics_map,
                        events=events,
                        log_path=server.log_path,
                        arm_id=f"arm-{arm_index:02d}-c{concurrency}",
                        target_concurrency=concurrency,
                        minimum_seconds=args.minimum_seconds,
                        minimum_generated_tokens=args.minimum_generated_tokens,
                        r3_enabled=r3_enabled,
                        max_num_batched_tokens=args.max_num_batched_tokens,
                        max_num_seqs=max_num_seqs,
                        request_transport=request_transport,
                    )
                    arms.append(arm)
                    if not arm["passed"]:
                        break
                alive_before_stop = server.process.poll() is None
    except Exception as exc:
        worker_error = {"type": type(exc).__name__, "message": str(exc)}
        if events is not None:
            events.emit("worker_failed", rank=rank, **worker_error)
    finally:
        if rank == 0:
            try:
                _put_json_readback(
                    filesystem,
                    stop_uri,
                    {"rank": rank, "run_id": run_id, "error": worker_error},
                )
            except Exception as exc:
                worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}
        if server is not None:
            alive_before_stop = alive_before_stop or server.process.poll() is None
            _stop_local_vllm(server)
        rank_status = {
            "rank": rank,
            "task_count": info.num_tasks,
            "worker_id": info.worker_id,
            "advertise_host": info.advertise_host,
            "job_id": str(info.job_id),
            "task_id": str(info.task_id),
            "task_image": args.task_image,
            "marin_commit": args.marin_commit,
            "vllm_commit": VLLM_SHA,
            "coscheduling": args.submitted_coscheduling,
            "gpu_inventory": gpu_inventory,
            "rendezvous": rendezvous,
            "vllm_command": server.command if server is not None else None,
            "vllm_environment": server.provenance_environment if server is not None else None,
            "vllm_alive_before_stop": alive_before_stop,
            "vllm_returncode_after_stop": server.process.returncode if server is not None else None,
            "error": worker_error,
            "elapsed_seconds": time.monotonic() - started,
        }
        try:
            _upload_rank_evidence(
                filesystem,
                prefix=control_prefix,
                rank=rank,
                local_dir=local_dir,
                status=rank_status,
            )
        except Exception as exc:
            worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}

    if rank != 0:
        if worker_error:
            raise RuntimeError(worker_error["message"])
        return rank_status

    try:
        rank_records = _wait_for_s3_jsons(
            filesystem,
            [f"{control_prefix}ranks/rank-{index}.json" for index in range(info.num_tasks)],
            timeout_seconds=300,
        )
        rank_receipts = _wait_for_s3_jsons(
            filesystem,
            [f"{control_prefix}ranks/rank-{index}-upload.json" for index in range(info.num_tasks)],
            timeout_seconds=300,
        )
    except Exception as exc:
        rank_records = [rank_status]
        rank_receipts = []
        worker_error = worker_error or {"type": type(exc).__name__, "message": str(exc)}
    placement = _unattended_placement_component(rendezvous, rank_records, expected_tasks=info.num_tasks)
    all_rank_health = (
        len(rank_records) == info.num_tasks
        and len(rank_receipts) == info.num_tasks
        and all(record.get("vllm_alive_before_stop") and record.get("error") is None for record in rank_records)
        and all(receipt.get("passed") for receipt in rank_receipts)
    )
    repeatability = _health_repeatability(arms)
    passed = (
        worker_error is None
        and all_rank_health
        and placement["passed"]
        and all(arm["passed"] for arm in arms)
        and len(arms) == len(concurrencies)
        and repeatability["passed"]
    )
    result = {
        "schema_version": 1,
        "experiment": "experiment-0",
        "run_id": run_id,
        "case": case.name,
        "model": model,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "arms": arms,
        "repeatability": repeatability,
        "placement": placement,
        "all_rank_health": {"passed": all_rank_health, "ranks": rank_records},
        "error": worker_error,
        "elapsed_seconds": time.monotonic() - started,
    }
    if events is None:
        events = HealthEventWriter(local_dir / "events.jsonl")
    events.emit(
        "worker_completed",
        passed=result["passed"],
        placement_passed=placement["passed"],
        all_rank_health=all_rank_health,
    )
    events.close()
    workload = json.loads((local_dir / "workload.json").read_text())
    manifest = {
        "schema_version": 1,
        "experiment": "experiment-0",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_prefix": artifact_prefix,
        "protocol": {
            "kind": "rolling closed-loop three-cohort health benchmark",
            "minimum_plateau_seconds": args.minimum_seconds,
            "minimum_plateau_engine_generation_tokens": args.minimum_generated_tokens,
            "minimum_in_flight_fraction": 0.95,
            "drain_excluded": True,
            "headline_counter": "vllm:generation_tokens",
        },
        "model_config": dataclasses.asdict(case),
        "model_fixture": {
            "source": "dummy",
            "weight_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "seed": DUMMY_SEED,
        },
        "server_settings": {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "data_parallel_size": 16,
            "expert_parallel_size": 16,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "r3_enabled": r3_enabled,
            "request_transport": request_transport,
            "concurrencies": concurrencies,
            "prefix_caching": True,
            "chunked_prefill": True,
            "cuda_graphs": True,
            "aggregate_engine_logging": True,
            "vllm_environment": dict(VLLM_SERVER_DEV_MODE_ENVIRONMENT),
        },
        "workload": _health_workload_manifest(
            workload,
            case=case,
            concurrencies=concurrencies,
            request_transport=request_transport,
        ),
        "routing_fixture": {
            "kind": "canonical seeded vLLM dummy routing",
            "seed": DUMMY_SEED,
            "expert_placement": "linear contiguous experts per EP rank",
            "capacity_factor": None,
            "balanced_control": {
                "applicable": False,
                "reason": "experiment 0 does not compare MoE geometry or EP topology",
            },
        },
        "implementation_controls": {
            "new_hot_path_family": False,
            "no_op_control": {
                "applicable": False,
                "reason": "experiment 0 adds a client dispatcher and evidence path, not a serving hot path",
            },
        },
        "r3": {
            "enabled": r3_enabled,
            "carrier": (
                "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy"
                if r3_enabled and request_transport == "completion"
                else (
                    "OpenAI chat JSON choice.routed_experts; generated-token-only nested integer lists"
                    if r3_enabled
                    else "absent"
                )
            ),
            "request_transport": request_transport,
            "consumer_commit": MARINSKYRL_CONSUMER_SHA if request_transport == "chat" else None,
            "expected_layers": case.num_hidden_layers,
            "expected_top_k": case.num_experts_per_tok,
        },
        "final_prefix_provenance": {arm["arm_id"]: arm["requests"]["final_prefix_provenance"] for arm in arms},
        "train_to_serve_parity": {
            "status": "inherited from reviewed exact-anchor preflight",
            "evidence": (
                "https://github.com/marin-community/marin/blob/"
                "12b4a6f0d8a1ef1e4021e1fa5b0ba0893fb087ed/notes/preflight/findings.md"
            ),
        },
        "provenance": {
            "marin_commit": args.marin_commit,
            "marin_commit_url": f"https://github.com/marin-community/marin/commit/{args.marin_commit}",
            "vllm_commit": VLLM_SHA,
            "vllm_commit_url": f"https://github.com/marin-community/vllm/commit/{VLLM_SHA}",
            "task_image": args.task_image,
            "dependency_lock_sha256": _sha256_path(Path("uv.lock")),
            "cluster_config": DEFAULT_CLUSTER_CONFIG,
            "iris_job_id": str(info.job_id),
            "iris_task_count": info.num_tasks,
            "iris_priority": args.iris_priority,
            "iris_coscheduling": args.submitted_coscheduling,
            "iris_retry_policy": {
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
            },
        },
        "placement": placement,
        "rank_commands": {str(record["rank"]): record.get("vllm_command") for record in rank_records},
    }
    artifact_records: list[dict[str, Any]] = []
    try:
        artifact_records = _write_and_upload_health_artifacts(
            filesystem,
            artifact_dir=local_dir,
            artifact_prefix=artifact_prefix,
            result=result,
            manifest=manifest,
            metrics_map=metrics_map,
        )
        _put_json_readback(
            filesystem,
            f"{control_prefix}writer-readback.json",
            {"passed": True, "artifact_prefix": artifact_prefix, "files": artifact_records},
        )
    except Exception as exc:
        result["passed"] = False
        result["status"] = "failed"
        result["artifact_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _put_json_readback(
            filesystem,
            f"{control_prefix}writer-readback.json",
            {"passed": False, "artifact_prefix": artifact_prefix, "error": result["artifact_error"]},
        )
        raise
    if not result["passed"]:
        raise RuntimeError(json.dumps(result.get("error") or {"placement": placement, "arms": arms}))
    return result


def _attended_result_passed(
    mode: str,
    *,
    correctness: dict[str, Any],
    load: dict[str, Any] | None,
) -> bool:
    return bool(correctness.get("passed")) and (mode != "acceptance" or bool(load and load.get("passed")))


def run_unattended_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "health":
        return run_health_unattended_worker(args)
    info = get_job_info()
    if info is None:
        raise RuntimeError("worker must run inside an Iris job")
    case = CASES[args.case]
    _validate_unattended_mode(
        case,
        mode=args.mode,
        model_source=args.model_source,
    )
    if info.num_tasks != case.node_count:
        raise RuntimeError(f"{case.name} requires {case.node_count} tasks, Iris supplied {info.num_tasks}")
    _validate_submitted_coscheduling(
        expected_tasks=info.num_tasks,
        submitted=args.submitted_coscheduling,
    )
    rank = info.task_index
    run_id = args.run_id
    prefix = f"{ARTIFACT_ROOT}/{case.name}/{run_id}/"
    local_dir = Path(REMOTE_ROOT) / run_id / f"rank-{rank}"
    local_dir.mkdir(parents=True, exist_ok=True)
    write_case(local_dir, case=case, run_id=run_id, git_sha=args.marin_commit)
    (local_dir / "aws-config").write_text(AWS_CONFIG_CONTENT)
    os.environ["AWS_CONFIG_FILE"] = str(local_dir / "aws-config")
    filesystem = _s3_filesystem()
    endpoint_name = f"grugmoe-ranks-{run_id}"
    metadata = {
        "task_index": str(rank),
        "num_tasks": str(info.num_tasks),
        "advertise_host": info.advertise_host,
    }
    if info.worker_id:
        metadata["worker_id"] = info.worker_id
    server: LocalVllm | None = None
    rendezvous: list[dict[str, str]] = []
    test_components: dict[str, Any] = {}
    result: dict[str, Any] = {
        "status": "starting",
        "case": case.name,
        "mode": args.mode,
        "model_source": args.model_source,
        "run_id": run_id,
    }
    started = time.monotonic()
    worker_error: dict[str, str] | None = None
    alive_before_stop = False
    stop_uri = f"{prefix}control/stop.json"
    try:
        with iris_ctx().registry.registered(
            endpoint_name,
            f"tcp://{info.advertise_host}:{RPC_PORT}",
            metadata,
        ):
            rendezvous = _wait_for_rank_endpoints(
                endpoint_name,
                expected=info.num_tasks,
                timeout_seconds=args.server_timeout,
            )
            leader_endpoint = next(endpoint for endpoint in rendezvous if int(endpoint["task_index"]) == 0)
            leader_ip = leader_endpoint["advertise_host"]
            model_dir = {
                "dummy": str(local_dir),
                "fixture": str(FIXTURE_DIR.resolve()),
                "snowball": SNOWBALL_EXPORT,
            }[args.model_source]
            server = _start_local_vllm(
                case=case,
                model_source=args.model_source,
                model_dir=model_dir,
                leader_ip=leader_ip,
                node_index=rank,
                smoke=args.mode == "smoke",
                local_dir=local_dir,
            )
            startup = {
                "rank": rank,
                "pid": server.process.pid,
                "worker_id": info.worker_id,
                "advertise_host": info.advertise_host,
                "command_sha256": hashlib.sha256("\0".join(server.command).encode()).hexdigest(),
                "alive": server.process.poll() is None,
            }
            _put_json_readback(
                filesystem,
                f"{prefix}startup/rank-{rank}.json",
                startup,
            )
            if rank != 0:
                while not filesystem.exists(_s3_key(stop_uri)):
                    if server.process.poll() is not None:
                        raise RuntimeError(f"rank {rank} vLLM exited early with {server.process.returncode}")
                    time.sleep(5)
                alive_before_stop = server.process.poll() is None
            else:
                startup_uris = [f"{prefix}startup/rank-{index}.json" for index in range(info.num_tasks)]
                startup_records = _wait_for_s3_jsons(
                    filesystem,
                    startup_uris,
                    timeout_seconds=args.server_timeout,
                )
                if not all(record["alive"] for record in startup_records):
                    raise RuntimeError(f"not every vLLM rank started: {startup_records}")
                base_url = f"http://127.0.0.1:{SERVER_PORT}"
                model = _wait_for_local_server(
                    base_url,
                    server,
                    timeout_seconds=args.server_timeout,
                )
                result["model"] = model
                correctness_started = time.monotonic()
                if args.model_source == "fixture":
                    correctness = run_fixture_parity(
                        base_url,
                        model,
                        artifact_dir=local_dir,
                    )
                else:
                    correctness_workload = json.loads((local_dir / "correctness-workload.json").read_text())
                    correctness = run_correctness(
                        base_url,
                        model,
                        correctness_workload,
                        case=case,
                        artifact_dir=local_dir,
                    )
                test_components = _smoke_components(
                    correctness,
                    elapsed_seconds=time.monotonic() - correctness_started,
                )
                if args.mode == "kv":
                    kv = run_kv_measurement(
                        base_url,
                        model,
                        case=case,
                        log_path=server.log_path,
                        artifact_dir=local_dir,
                    )
                    result["kv"] = kv
                    test_components["duration"] = {
                        "passed": kv["passed"],
                        "contexts": [observation["final_sequence_tokens"] for observation in kv["observations"]],
                    }
                    test_components["token_count"] = {
                        "passed": all(observation["generated_tokens"] == 2_048 for observation in kv["observations"]),
                        "generated_tokens": [observation["generated_tokens"] for observation in kv["observations"]],
                    }
                elif args.mode == "acceptance":
                    workload = json.loads((local_dir / "workload.json").read_text())
                    load = run_acceptance_load(
                        base_url,
                        model,
                        workload,
                        artifact_dir=local_dir,
                        max_model_len=case.max_model_len,
                        minimum_seconds=args.minimum_seconds,
                        minimum_generated_tokens=args.minimum_generated_tokens,
                    )
                    result["load"] = load
                    test_components.update(_acceptance_components(load))
                result["correctness"] = correctness
                alive_before_stop = server.process.poll() is None
    except Exception as exc:
        worker_error = {"type": type(exc).__name__, "message": str(exc)}
        if rank == 0:
            result["error"] = worker_error
    finally:
        if rank == 0:
            try:
                _put_json_readback(
                    filesystem,
                    stop_uri,
                    {
                        "rank": rank,
                        "run_id": run_id,
                        "error": worker_error,
                    },
                )
            except Exception as exc:
                worker_error = worker_error or {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        if server is not None:
            alive_before_stop = alive_before_stop or server.process.poll() is None
            _stop_local_vllm(server)
        rank_status = {
            "rank": rank,
            "task_count": info.num_tasks,
            "worker_id": info.worker_id,
            "advertise_host": info.advertise_host,
            "job_id": str(info.job_id),
            "task_id": str(info.task_id),
            "task_image": args.task_image,
            "marin_commit": args.marin_commit,
            "vllm_commit": VLLM_SHA,
            "coscheduling": args.submitted_coscheduling,
            "rendezvous": rendezvous,
            "vllm_alive_before_stop": alive_before_stop,
            "vllm_returncode_after_stop": server.process.returncode if server is not None else None,
            "error": worker_error,
            "elapsed_seconds": time.monotonic() - started,
        }
        try:
            _upload_rank_evidence(
                filesystem,
                prefix=prefix,
                rank=rank,
                local_dir=local_dir,
                status=rank_status,
            )
        except Exception as exc:
            worker_error = worker_error or {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    if rank != 0:
        if worker_error:
            raise RuntimeError(worker_error["message"])
        return rank_status

    rank_uris = [f"{prefix}ranks/rank-{index}.json" for index in range(info.num_tasks)]
    receipt_uris = [f"{prefix}ranks/rank-{index}-upload.json" for index in range(info.num_tasks)]
    try:
        rank_records = _wait_for_s3_jsons(
            filesystem,
            rank_uris,
            timeout_seconds=300,
        )
        receipts = _wait_for_s3_jsons(
            filesystem,
            receipt_uris,
            timeout_seconds=300,
        )
    except Exception as exc:
        rank_records = [rank_status]
        receipts = []
        worker_error = worker_error or {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    placement = _unattended_placement_component(
        rendezvous,
        rank_records,
        expected_tasks=info.num_tasks,
    )
    all_rank_health = {
        "passed": (
            len(rank_records) == info.num_tasks
            and all(record.get("vllm_alive_before_stop") and record.get("error") is None for record in rank_records)
        ),
        "ranks": rank_records,
    }
    components = {
        "placement": placement,
        "all_rank_health": all_rank_health,
        "correctness": test_components.get("correctness", False),
        "duration": test_components.get("duration", False),
        "token_count": test_components.get("token_count", False),
        "repeatability": test_components.get("repeatability", False),
        "artifact_readback": False,
    }
    manifest = frozen_manifest(
        case,
        run_id=run_id,
        git_sha=args.marin_commit,
        model_source=args.model_source,
    )
    manifest.update(
        {
            "task_image": args.task_image,
            "iris": {
                "job_id": str(info.job_id),
                "task_count": info.num_tasks,
                "coscheduling": args.submitted_coscheduling,
                "priority": Priority.INTERACTIVE.value,
            },
            "files": {
                name: _sha256_path(local_dir / name)
                for name in (
                    "config.json",
                    "correctness-workload.json",
                    "workload.json",
                )
            },
            "dependency_lock_sha256": _sha256_path(Path("uv.lock")),
        }
    )
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["components"] = components
    result["rank_uploads"] = receipts
    result["elapsed_seconds"] = time.monotonic() - started
    if worker_error:
        result["error"] = worker_error

    upload_records: list[dict[str, Any]] = []
    for path in sorted(local_dir.rglob("*")):
        if path.is_file() and path.name != "result.json":
            relative = path.relative_to(local_dir).as_posix()
            upload_records.append(
                _put_bytes_readback(
                    filesystem,
                    f"{prefix}{relative}",
                    path.read_bytes(),
                )
            )
    components["artifact_readback"] = {
        "passed": (
            bool(upload_records)
            and all(record["readback"] == "identical" for record in upload_records)
            and len(receipts) == info.num_tasks
            and all(receipt["passed"] for receipt in receipts)
        ),
        "files": upload_records,
        "rank_receipts": receipts,
    }
    aggregate = aggregate_preflight_status(components)
    result.update(aggregate)
    if worker_error:
        result["status"] = "failed"
        result["passed"] = False
    result["result_readback"] = {
        "path": f"{prefix}result.json",
        "readback": "identical",
    }
    result_path = local_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _put_bytes_readback(
        filesystem,
        f"{prefix}result.json",
        result_path.read_bytes(),
    )
    if not result["passed"]:
        raise RuntimeError(json.dumps(result.get("error") or result["components"]))
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    case = CASES[args.case]
    if args.mode == "acceptance":
        validate_acceptance_thresholds(
            minimum_seconds=args.minimum_seconds,
            minimum_generated_tokens=args.minimum_generated_tokens,
        )
    state = load_state(state_path(Path(args.state_dir), args.session))
    validate_session(state, case)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    local_dir = Path(args.local_output or tempfile.mkdtemp(prefix=f"grugmoe-preflight-{run_id}-"))
    local_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    write_case(local_dir, case=case, run_id=run_id, git_sha=git_sha)
    (local_dir / "aws-config").write_text(AWS_CONFIG_CONTENT)
    workload = json.loads((local_dir / "workload.json").read_text())
    correctness_workload = json.loads((local_dir / "correctness-workload.json").read_text())
    manifest = frozen_manifest(case, run_id=run_id, git_sha=git_sha, model_source=args.model_source)
    gang = Gang(
        state,
        case,
        run_id,
        local_dir,
        model_source=args.model_source,
        smoke=args.mode == "smoke",
    )
    result: dict[str, Any] = {
        "status": "starting",
        "mode": args.mode,
        "case": case.name,
        "model_source": args.model_source,
        "run_id": run_id,
        "local_output": str(local_dir),
        "manifest": manifest,
    }
    gang.inspect()
    result["pods"] = [dataclasses.asdict(runtime) for runtime in gang.runtimes]
    manifest["files"] = {
        filename: _sha256_path(local_dir / filename)
        for filename in (
            "aws-config",
            "config.json",
            "correctness-workload.json",
            "workload.json",
        )
    }
    manifest["dependency_lock_sha256"] = _sha256_path(Path("uv.lock"))
    manifest["iris"] = {
        "session_name": state.session_name,
        "job_id": state.job_id,
        "priority": state.priority.value,
        "gpu_variant": state.gpu_variant,
        "gpus_per_node": state.gpus_per_node,
        "node_count": len(state.pods),
        "config_file": state.config_file,
    }
    manifest["pods"] = result["pods"]
    (local_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        gang.stage()
        gang.start()
        try:
            gang.wait_for_leader_port(args.server_timeout)
        except Exception as exc:
            raise RuntimeError(f"{exc}\n{gang.failure_tail()}") from exc
        with port_forward(state, state.pods[0]) as (local_port, forward):
            base_url = f"http://127.0.0.1:{local_port}"
            try:
                model = wait_for_server(base_url, forward, args.server_timeout)
            except Exception as exc:
                raise RuntimeError(f"{exc}\n{gang.failure_tail()}") from exc
            result["model"] = model
            if args.model_source == "fixture":
                result["correctness"] = run_fixture_parity(
                    base_url,
                    model,
                    artifact_dir=local_dir,
                )
            else:
                result["correctness"] = run_correctness(
                    base_url,
                    model,
                    correctness_workload,
                    case=case,
                    artifact_dir=local_dir,
                )
            if args.mode == "acceptance":
                result["load"] = run_acceptance_load(
                    base_url,
                    model,
                    workload,
                    artifact_dir=local_dir,
                    max_model_len=case.max_model_len,
                    minimum_seconds=args.minimum_seconds,
                    minimum_generated_tokens=args.minimum_generated_tokens,
                )
        result["passed"] = _attended_result_passed(
            args.mode,
            correctness=result["correctness"],
            load=result.get("load"),
        )
        if not result["passed"]:
            raise RuntimeError(
                "attended preflight component failed: "
                + json.dumps(
                    {
                        "correctness": result["correctness"].get("passed"),
                        "load": (result.get("load") or {}).get("passed"),
                    },
                    sort_keys=True,
                )
            )
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["passed"] = False
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result["pod_logs"] = gang.collect_logs()
        gang.stop()
        result_path = local_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not args.no_upload:
            prefix = f"{ARTIFACT_ROOT}/{case.name}/{run_id}/"
            try:
                relative_files = sorted(
                    path.relative_to(local_dir).as_posix()
                    for path in local_dir.rglob("*")
                    if path.is_file() and path != result_path
                )
                uploaded = gang.upload_and_readback(prefix, relative_files)
                result["upload"] = {
                    "prefix": prefix,
                    "files": uploaded,
                    "passed": bool(uploaded),
                    "result": {
                        "path": f"{prefix.rstrip('/')}/result.json",
                        "readback": "identical",
                    },
                }
                result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                gang.upload_and_readback(prefix, ["result.json"])
            except Exception as upload_exc:
                result["upload"] = {
                    "prefix": prefix,
                    "passed": False,
                    "error": {
                        "type": type(upload_exc).__name__,
                        "message": str(upload_exc),
                    },
                }
                if result["status"] == "passed":
                    result["status"] = "failed"
                    result["passed"] = False
                    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                    raise
                result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def readback_health_artifacts(filesystem: Any, *, run_id: str) -> dict[str, Any]:
    """Independently re-read and recompute the experiment-0 artifact contract."""
    artifact_prefix = f"{HEALTH_ARTIFACT_ROOT}/{run_id}/"
    root_key = _s3_key(artifact_prefix).rstrip("/")

    def read(relative: str) -> bytes:
        return filesystem.cat_file(f"{root_key}/{relative}")

    manifest_bytes = read("manifest.json")
    result_bytes = read("result.json")
    events_bytes = read("events.jsonl")
    metrics_map_bytes = read("metrics/map.json")
    result_md_bytes = read("result.md")
    manifest = json.loads(manifest_bytes)
    result = json.loads(result_bytes)
    metrics_map = json.loads(metrics_map_bytes)
    checks: dict[str, Any] = {}
    checks["manifest_identity"] = {
        "passed": (
            manifest.get("experiment") == "experiment-0"
            and manifest.get("run_id") == run_id
            and result.get("run_id") == run_id
            and result.get("manifest_sha256") == hashlib.sha256(manifest_bytes).hexdigest()
        )
    }
    claimed_checks: dict[str, Any] = {}
    for relative, claim in manifest.get("claimed_files", {}).items():
        payload = read(relative)
        claimed_checks[relative] = {
            "passed": len(payload) == int(claim["bytes"]) and hashlib.sha256(payload).hexdigest() == claim["sha256"],
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    checks["claimed_file_hashes"] = {
        "passed": bool(claimed_checks) and all(item["passed"] for item in claimed_checks.values()),
        "files": claimed_checks,
    }
    snapshots = metrics_map.get("snapshots", [])
    snapshot_by_path = {entry["path"]: entry for entry in snapshots}
    raw_paths = {entry["path"] for entry in snapshots}
    raw_checks: dict[str, Any] = {}
    parsed_raw: dict[str, list[Any]] = {}
    recomputed_metrics_map: list[dict[str, Any]] = []
    for entry in snapshots:
        payload = read(entry["path"])
        parsed = parse_labeled_prometheus(payload.decode())
        recomputed_totals = {metric: prometheus_value(parsed, metric) for metric in HEALTH_COUNTER_METRICS}
        recomputed_by_engine = {
            metric: prometheus_values_by_label(parsed, metric, label="engine") for metric in HEALTH_ENGINE_METRICS
        }
        totals_match = set(entry.get("totals", {})) == set(HEALTH_COUNTER_METRICS) and all(
            math.isclose(
                float(entry["totals"][metric]),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for metric, value in recomputed_totals.items()
        )
        by_engine_match = entry.get("by_engine") == recomputed_by_engine
        raw_checks[entry["path"]] = {
            "passed": (
                len(payload) == int(entry["bytes"])
                and hashlib.sha256(payload).hexdigest() == entry["sha256"]
                and totals_match
                and by_engine_match
            ),
            "totals_match": totals_match,
            "by_engine_match": by_engine_match,
        }
        parsed_raw[entry["path"]] = parsed
        recomputed_metrics_map.append(
            {
                **entry,
                "totals": recomputed_totals,
                "by_engine": recomputed_by_engine,
            }
        )
    checks["metrics_map"] = {
        "passed": (
            bool(snapshots)
            and len(snapshot_by_path) == len(snapshots)
            and [int(entry["index"]) for entry in snapshots] == list(range(len(snapshots)))
            and all(item["passed"] for item in raw_checks.values())
            and raw_paths
            == {relative for relative in manifest.get("claimed_files", {}) if relative.startswith("metrics/raw-")}
        ),
        "snapshots": len(snapshots),
        "raw_files": raw_checks,
    }
    found = {
        path.removeprefix(f"{root_key}/")
        for path in filesystem.find(root_key)
        if path != root_key and not path.endswith("/")
    }
    expected = {
        "manifest.json",
        "events.jsonl",
        "metrics/map.json",
        "result.json",
        "result.md",
        *raw_paths,
        *[str(arm["kv_cache"]["source"]["path"]) for arm in result.get("arms", [])],
    }
    unexpected = sorted(path for path in found - expected if not path.startswith("profiles/"))
    checks["exact_layout"] = {
        "passed": expected <= found and not unexpected,
        "expected": sorted(expected),
        "missing": sorted(expected - found),
        "unexpected_non_profile_files": unexpected,
    }
    event_records = [json.loads(line) for line in events_bytes.decode().splitlines() if line.strip()]
    workload_requests_for_reader = {request["request_id"]: request for request in manifest["workload"]["requests"]}
    arm_checks: dict[str, Any] = {}
    protocol = manifest["protocol"]
    for arm in result.get("arms", []):
        start_path = arm["metrics"]["boundary_start"]
        end_path = arm["metrics"]["boundary_end"]
        start_samples = parsed_raw[start_path]
        end_samples = parsed_raw[end_path]
        generated = round(
            prometheus_value(end_samples, "vllm:generation_tokens")
            - prometheus_value(start_samples, "vllm:generation_tokens")
        )
        prompt = round(
            prometheus_value(end_samples, "vllm:prompt_tokens") - prometheus_value(start_samples, "vllm:prompt_tokens")
        )
        recomputed_counter_deltas = {
            metric: prometheus_value(end_samples, metric) - prometheus_value(start_samples, metric)
            for metric in HEALTH_COUNTER_METRICS
        }
        reported_counter_deltas = arm["metrics"]["counter_deltas"]
        counter_deltas_match = all(
            math.isclose(float(reported_counter_deltas[metric]), recomputed, rel_tol=1e-12, abs_tol=1e-12)
            for metric, recomputed in recomputed_counter_deltas.items()
        )
        prefix_queries = recomputed_counter_deltas["vllm:prefix_cache_queries"]
        prefix_hits = recomputed_counter_deltas["vllm:prefix_cache_hits"]
        expected_prefix_ratio = prefix_hits / prefix_queries if prefix_queries else None
        reported_prefix = arm["metrics"]["prefix_cache"]
        prefix_metrics_match = (
            float(reported_prefix["query_tokens"]) == prefix_queries
            and float(reported_prefix["hit_tokens"]) == prefix_hits
            and (
                reported_prefix["hit_ratio"] is None
                if expected_prefix_ratio is None
                else math.isclose(
                    float(reported_prefix["hit_ratio"]),
                    expected_prefix_ratio,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        )
        per_engine_metrics_complete = all(
            len(prometheus_values_by_label(start_samples, metric, label="engine")) == 16
            and len(prometheus_values_by_label(end_samples, metric, label="engine")) == 16
            for metric in HEALTH_ENGINE_METRICS
        )
        target_concurrency = int(arm["settings"]["target_concurrency"])
        recomputed_engine_series = _health_engine_series(
            recomputed_metrics_map,
            first_index=int(snapshot_by_path[start_path]["index"]),
            last_index=int(snapshot_by_path[end_path]["index"]),
        )
        per_engine_series_match = arm["metrics"]["per_engine_plateau_series"] == recomputed_engine_series
        kv_source = arm["kv_cache"]["source"]
        kv_source_path = str(kv_source["path"])
        kv_source_bytes = read(kv_source_path)
        kv_source_match = (
            kv_source_path in manifest.get("claimed_files", {})
            and int(kv_source["bytes"]) == len(kv_source_bytes)
            and kv_source["sha256"] == hashlib.sha256(kv_source_bytes).hexdigest()
        )
        recomputed_kv = _health_kv_summary_from_text(
            kv_source_bytes.decode(errors="replace"),
            case=CASES["exact-reference-ep16"],
            target_concurrency=target_concurrency,
        )
        reported_kv = {key: value for key, value in arm["kv_cache"].items() if key != "source"}
        kv_aggregate_match = kv_source_match and reported_kv == recomputed_kv and recomputed_kv["passed"] is True
        seconds = float(arm["plateau"]["elapsed_seconds"])
        plateau = arm["plateau"]
        minimum_in_flight = math.ceil(float(protocol["minimum_in_flight_fraction"]) * target_concurrency)
        in_flight = plateau["in_flight"]
        plateau_contract_match = (
            int(plateau["target_concurrency"]) == target_concurrency
            and int(plateau["minimum_required_in_flight"]) == minimum_in_flight
            and int(in_flight["min"]) >= minimum_in_flight
            and int(in_flight["max"]) <= target_concurrency
            and int(in_flight["at_close"]) == target_concurrency
            and int(plateau["failed_requests"]) == 0
            and plateau["manifest"]
            == {
                "expected": 144,
                "observed": 144,
                "passed": True,
            }
        )
        boundary_seconds = float(snapshot_by_path[end_path]["monotonic_seconds"]) - float(
            snapshot_by_path[start_path]["monotonic_seconds"]
        )
        boundary_seconds_match = math.isclose(seconds, boundary_seconds, rel_tol=1e-12, abs_tol=1e-12)
        boundary_start_seconds = float(snapshot_by_path[start_path]["monotonic_seconds"])
        boundary_end_seconds = float(snapshot_by_path[end_path]["monotonic_seconds"])
        completion_events = [
            record
            for record in event_records
            if record.get("event") == "request_completed" and record.get("arm_id") == arm["arm_id"]
        ]
        timestamped_events = [
            record
            for record in completion_events
            if isinstance(record.get("completed_at_monotonic_seconds"), (int, float))
        ]
        window_events = [
            record
            for record in timestamped_events
            if boundary_start_seconds <= float(record["completed_at_monotonic_seconds"]) <= boundary_end_seconds
        ]
        before_window_events = [
            record
            for record in timestamped_events
            if float(record["completed_at_monotonic_seconds"]) < boundary_start_seconds
        ]
        drain_events = [
            record
            for record in timestamped_events
            if float(record["completed_at_monotonic_seconds"]) > boundary_end_seconds
        ]
        window_request_ids = {str(record.get("manifest_request_id")) for record in window_events}
        window_cohorts: dict[str, int] = {}
        for record in window_events:
            cohort = str(record.get("cohort"))
            window_cohorts[cohort] = window_cohorts.get(cohort, 0) + 1
        event_window_match = (
            len(timestamped_events) == len(completion_events)
            and len(window_events) == int(arm["requests"]["plateau_successes"])
            and len(window_events) == int(plateau["successful_requests"])
            and len(before_window_events) == int(arm["requests"]["excluded_before_valid_plateau_successes"])
            and len(drain_events) == int(arm["requests"]["drain_successes"])
            and len(completion_events) == int(arm["requests"]["whole_run_successes"])
            and window_request_ids == set(workload_requests_for_reader)
            and sum(int(record["completion_tokens"]) for record in window_events)
            == int(plateau["client_generated_tokens"])
            and window_cohorts == plateau["cohort_completions"]
            and all(
                str(record.get("manifest_request_id")) in workload_requests_for_reader
                and int(record["completion_tokens"])
                == int(workload_requests_for_reader[str(record["manifest_request_id"])]["max_tokens"])
                and record.get("cohort") == workload_requests_for_reader[str(record["manifest_request_id"])]["cohort"]
                for record in completion_events
            )
        )
        rolling_start_path = arm["metrics"]["rolling_start"]
        final_path = arm["metrics"]["final_after_drain"]
        rolling_start_samples = parsed_raw[rolling_start_path]
        final_samples = parsed_raw[final_path]
        whole_engine_generated = round(
            prometheus_value(final_samples, "vllm:generation_tokens")
            - prometheus_value(rolling_start_samples, "vllm:generation_tokens")
        )
        whole_engine_successes = round(
            prometheus_value(final_samples, "vllm:request_success")
            - prometheus_value(rolling_start_samples, "vllm:request_success")
        )
        whole_preemptions = round(
            prometheus_value(final_samples, "vllm:num_preemptions")
            - prometheus_value(rolling_start_samples, "vllm:num_preemptions")
        )
        whole_client_generated = sum(int(record["completion_tokens"]) for record in completion_events)
        reported_reconciliation = arm["whole_run_token_reconciliation"]
        whole_run_match = (
            whole_engine_generated == whole_client_generated
            and whole_engine_successes == len(completion_events)
            and whole_preemptions == 0
            and int(arm["preemptions"]) == whole_preemptions
            and int(arm["requests"]["engine_success_counter_delta"]) == whole_engine_successes
            and reported_reconciliation
            == {
                "client_generated_tokens": whole_client_generated,
                "engine_generated_tokens": whole_engine_generated,
                "delta": whole_engine_generated - whole_client_generated,
                "passed": whole_engine_generated == whole_client_generated,
            }
        )
        expected_headline = generated / seconds / 16
        server_histograms_match = True
        for metric in HEALTH_HISTOGRAM_METRICS:
            reported = arm["latency"]["server_histogram_window"][metric]
            for label, quantile in (("p50_seconds", 0.50), ("p99_seconds", 0.99)):
                recomputed = histogram_quantile_delta(start_samples, end_samples, metric, quantile)
                value = reported[label]
                if recomputed is None or value is None:
                    server_histograms_match = server_histograms_match and recomputed is None and value is None
                else:
                    server_histograms_match = server_histograms_match and math.isclose(
                        float(value), recomputed, rel_tol=1e-12, abs_tol=1e-12
                    )
        arm_passed = (
            arm.get("passed") is True
            and all(arm.get("gates", {}).values())
            and generated == int(arm["plateau"]["generated_tokens"])
            and prompt == int(arm["plateau"]["processed_prompt_tokens"])
            and seconds >= float(protocol["minimum_plateau_seconds"])
            and generated >= int(protocol["minimum_plateau_engine_generation_tokens"])
            and plateau_contract_match
            and boundary_seconds_match
            and event_window_match
            and whole_run_match
            and int(arm["requests"]["branch_coverage"]["observed"]) == 144
            and arm["requests"]["branch_coverage"]["passed"] is True
            and int(arm["preemptions"]) == 0
            and arm["whole_run_token_reconciliation"]["passed"] is True
            and math.isclose(
                float(arm["headline"]["generation_tokens_per_second_per_gpu"]),
                expected_headline,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and server_histograms_match
            and counter_deltas_match
            and prefix_metrics_match
            and per_engine_metrics_complete
            and per_engine_series_match
            and kv_aggregate_match
            and start_path in snapshot_by_path
            and end_path in snapshot_by_path
        )
        arm_checks[arm["arm_id"]] = {
            "passed": arm_passed,
            "recomputed_generation_tokens": generated,
            "recomputed_prompt_tokens": prompt,
            "recomputed_generation_tokens_per_second_per_gpu": expected_headline,
            "recomputed_boundary_seconds": boundary_seconds,
            "boundary_seconds_match": boundary_seconds_match,
            "event_window_match": event_window_match,
            "whole_run_match": whole_run_match,
            "server_histograms_match": server_histograms_match,
            "counter_deltas_match": counter_deltas_match,
            "prefix_metrics_match": prefix_metrics_match,
            "per_engine_metrics_complete": per_engine_metrics_complete,
            "per_engine_series_match": per_engine_series_match,
            "kv_aggregate_match": kv_aggregate_match,
            "plateau_contract_match": plateau_contract_match,
        }
    checks["arm_aggregates"] = {
        "passed": bool(arm_checks) and all(item["passed"] for item in arm_checks.values()),
        "arms": arm_checks,
    }
    recomputed_repeatability = _health_repeatability(result.get("arms", []))
    checks["repeatability"] = {
        "passed": result.get("repeatability") == recomputed_repeatability,
        "benchmark_passed": recomputed_repeatability["passed"],
        "recomputed": recomputed_repeatability,
    }
    event_names = [record.get("event") for record in event_records]
    arm_ids = {arm["arm_id"] for arm in result.get("arms", [])}
    checks["events"] = {
        "passed": (
            bool(event_records)
            and "worker_completed" in event_names
            and all(
                any(record.get("event") == event and record.get("arm_id") == arm_id for record in event_records)
                for arm_id in arm_ids
                for event in ("plateau_opened", "plateau_closed", "arm_completed")
            )
        ),
        "records": len(event_records),
    }
    workload = manifest["workload"]
    placement = manifest["placement"]
    rank_records = result["all_rank_health"]["ranks"]
    gpu_inventory = [gpu for rank in rank_records for gpu in rank.get("gpu_inventory", [])]
    server_settings = manifest["server_settings"]
    provenance = manifest["provenance"]
    fixture = manifest["model_fixture"]
    rank_commands = manifest["rank_commands"]
    request_transport = str(server_settings.get("request_transport", "completion"))
    workload_requests = {request["request_id"]: request for request in workload["requests"]}

    def is_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    prefix_checks: dict[str, Any] = {}
    for arm in result.get("arms", []):
        arm_id = arm["arm_id"]
        event_provenance = _health_final_prefix_provenance(
            [
                record
                for record in event_records
                if record.get("event") == "request_completed" and record.get("arm_id") == arm_id
            ]
        )
        reported_provenance = arm["requests"]["final_prefix_provenance"]
        ids = {entry["manifest_request_id"] for entry in reported_provenance}
        hashes_valid = all(
            entry["manifest_request_id"] in workload_requests
            and entry["prompt_token_ids_sha256"]
            == workload_requests[entry["manifest_request_id"]]["prompt_token_ids_sha256"]
            and is_sha256(entry["prompt_token_ids_sha256"])
            and entry["occurrences"] > 0
            and sum(outcome["occurrences"] for outcome in entry["outcomes"]) == entry["occurrences"]
            and all(
                is_sha256(outcome["generated_token_ids_sha256"])
                and is_sha256(outcome["final_prefix_token_ids_sha256"])
                and outcome["occurrences"] > 0
                for outcome in entry["outcomes"]
            )
            for entry in reported_provenance
        )
        prefix_checks[arm_id] = {
            "passed": (
                len(reported_provenance) == 144
                and ids == set(workload_requests)
                and hashes_valid
                and event_provenance == reported_provenance
                and manifest["final_prefix_provenance"].get(arm_id) == reported_provenance
                and sum(entry["occurrences"] for entry in reported_provenance) == arm["requests"]["whole_run_successes"]
            ),
            "branches": len(ids),
            "occurrences": sum(entry["occurrences"] for entry in reported_provenance),
        }
    checks["final_prefix_provenance"] = {
        "passed": bool(prefix_checks) and all(item["passed"] for item in prefix_checks.values()),
        "arms": prefix_checks,
    }
    logprob_checks: dict[str, Any] = {}
    for arm in result.get("arms", []):
        arm_id = arm["arm_id"]
        completion_events = [
            record
            for record in event_records
            if record.get("event") == "request_completed" and record.get("arm_id") == arm_id
        ]
        reported = arm["requests"]["sampled_token_logprobs"]
        event_token_count = sum(int(record.get("sampled_token_logprobs_count", -1)) for record in completion_events)
        logprob_checks[arm_id] = {
            "passed": (
                len(completion_events) == int(arm["requests"]["whole_run_successes"])
                and all(
                    int(record.get("sampled_token_logprobs_count", -1)) > 0
                    and int(record["sampled_token_logprobs_count"]) == int(record.get("completion_tokens", -1))
                    and is_sha256(record.get("sampled_token_logprobs_sha256"))
                    for record in completion_events
                )
                and int(reported["validated_requests"]) == len(completion_events)
                and int(reported["validated_generated_tokens"]) == event_token_count
                and reported["all_completion_tokens_covered"] is True
            ),
            "requests": len(completion_events),
            "sampled_token_logprobs": event_token_count,
        }
    checks["sampled_token_logprobs"] = {
        "passed": bool(logprob_checks) and all(item["passed"] for item in logprob_checks.values()),
        "arms": logprob_checks,
    }
    route_checks: dict[str, Any] = {}
    latency_checks: dict[str, Any] = {}
    for arm in result.get("arms", []):
        arm_id = arm["arm_id"]
        start_seconds = float(snapshot_by_path[arm["metrics"]["boundary_start"]]["monotonic_seconds"])
        end_seconds = float(snapshot_by_path[arm["metrics"]["boundary_end"]]["monotonic_seconds"])
        completion_events = [
            record
            for record in event_records
            if record.get("event") == "request_completed" and record.get("arm_id") == arm_id
        ]
        window_events = [
            record
            for record in completion_events
            if start_seconds <= float(record["completed_at_monotonic_seconds"]) <= end_seconds
        ]
        r3_enabled = bool(arm["settings"]["r3_enabled"])
        route_summaries = [record.get("route_summary") for record in window_events]
        route_records_valid = True
        aggregate_experts = [0] * 128
        aggregate_ep_ranks = [0] * 16
        aggregate_array_bytes = 0
        aggregate_npy_bytes = 0
        aggregate_base64_bytes = 0
        aggregate_json_bytes = 0
        for event, summary in zip(window_events, route_summaries, strict=True):
            request = workload_requests_for_reader[str(event["manifest_request_id"])]
            if r3_enabled:
                expected_positions = (
                    int(request["max_tokens"]) if request_transport == "chat" else int(request["final_token_count"]) - 1
                )
                expert_histogram = summary.get("expert_histogram", []) if isinstance(summary, dict) else []
                ep_rank_histogram = summary.get("ep_rank_histogram", []) if isinstance(summary, dict) else []
                assignments = expected_positions * 48 * 4
                record_valid = (
                    isinstance(summary, dict)
                    and summary.get("enabled") is True
                    and summary.get("shape") == [expected_positions, 48, 4]
                    and (
                        summary.get("dtype") == "uint8"
                        if request_transport == "completion"
                        else summary.get("dtype") in {"int32", "int64"}
                    )
                    and 0 <= int(summary.get("minimum_expert", -1)) <= int(summary.get("maximum_expert", 128)) < 128
                    and summary.get("all_expected_positions_layers_topk_aligned") is True
                    and int(summary.get("root_prefix_positions_compared", -1))
                    == (int(request["prefix_token_count"]) if request_transport == "completion" else 0)
                    and summary.get("root_prefix_aligned") is True
                    and is_sha256(summary.get("route_sha256"))
                    and event.get("route_sha256") == summary.get("route_sha256")
                    and len(expert_histogram) == 128
                    and len(ep_rank_histogram) == 16
                    and sum(int(value) for value in expert_histogram) == assignments
                    and [sum(int(value) for value in expert_histogram[index : index + 8]) for index in range(0, 128, 8)]
                    == [int(value) for value in ep_rank_histogram]
                    and int(summary.get("carrier_array_bytes", -1))
                    == assignments * (1 if request_transport == "completion" else 8)
                    and (
                        int(summary.get("carrier_npy_bytes", -1)) >= assignments
                        and int(summary.get("carrier_base64_bytes", -1)) >= int(summary.get("carrier_npy_bytes", 0))
                        and int(summary.get("carrier_json_bytes", 0)) == 0
                        if request_transport == "completion"
                        else int(summary.get("carrier_npy_bytes", -1)) == 0
                        and int(summary.get("carrier_base64_bytes", -1)) == 0
                        and int(summary.get("carrier_json_bytes", -1)) > assignments
                    )
                )
                route_records_valid = route_records_valid and record_valid
                if record_valid:
                    aggregate_experts = [
                        current + int(value) for current, value in zip(aggregate_experts, expert_histogram, strict=True)
                    ]
                    aggregate_ep_ranks = [
                        current + int(value)
                        for current, value in zip(aggregate_ep_ranks, ep_rank_histogram, strict=True)
                    ]
                    aggregate_array_bytes += int(summary["carrier_array_bytes"])
                    aggregate_npy_bytes += int(summary["carrier_npy_bytes"])
                    aggregate_base64_bytes += int(summary["carrier_base64_bytes"])
                    aggregate_json_bytes += int(summary.get("carrier_json_bytes", 0))
            else:
                route_records_valid = route_records_valid and (
                    isinstance(summary, dict)
                    and summary.get("enabled") is False
                    and summary.get("transport") == "absent"
                    and event.get("route_sha256") is None
                )
        routing = arm["moe_routing"]
        carrier = routing["carrier"]
        if r3_enabled:
            rank_mean = sum(aggregate_ep_ranks) / len(aggregate_ep_ranks)
            rank_max = max(aggregate_ep_ranks)
            route_aggregate_match = (
                routing["expert_histogram"] == aggregate_experts
                and routing["ep_rank_histogram"] == aggregate_ep_ranks
                and float(routing["ep_rank_load"]["mean_assignments"]) == rank_mean
                and int(routing["ep_rank_load"]["max_assignments"]) == rank_max
                and math.isclose(
                    float(routing["ep_rank_load"]["max_over_mean"]),
                    rank_max / rank_mean,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and int(carrier["array_bytes"]) == aggregate_array_bytes
                and int(carrier["npy_bytes"]) == aggregate_npy_bytes
                and int(carrier["base64_bytes"]) == aggregate_base64_bytes
                and int(carrier.get("json_bytes", 0)) == aggregate_json_bytes
                and int(carrier["full_response_bytes"]) == sum(int(record["response_bytes"]) for record in window_events)
                and carrier["transport"]
                == (
                    "OpenAI chat JSON choice.routed_experts; generated-token-only nested integer lists"
                    if request_transport == "chat"
                    else "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy"
                )
                and routing["alignment_passed"] is True
            )
        else:
            route_aggregate_match = (
                routing["expert_histogram"] is None
                and routing["ep_rank_histogram"] is None
                and carrier["transport"] == "absent"
                and int(carrier["array_bytes"]) == 0
                and int(carrier["npy_bytes"]) == 0
                and int(carrier["base64_bytes"]) == 0
                and int(carrier.get("json_bytes", 0)) == 0
                and routing["alignment_passed"] is True
            )
        route_checks[arm_id] = {
            "passed": bool(window_events) and route_records_valid and route_aggregate_match,
            "window_requests": len(window_events),
            "assignments": sum(aggregate_experts),
        }

        timing_records = [record.get("timing") for record in window_events]
        timing_records_valid = all(
            isinstance(timing, dict)
            and all(
                isinstance(timing.get(key), (int, float)) and float(timing[key]) >= 0
                for key in (
                    "client_e2e_seconds",
                    "seconds_to_response_headers",
                    "response_body_transfer_seconds",
                    "seconds_to_decode",
                )
            )
            for timing in timing_records
        )
        latency_match = False
        if timing_records_valid and timing_records:
            client = _health_percentiles([float(timing["client_e2e_seconds"]) for timing in timing_records])
            by_cohort = {
                cohort: _health_percentiles(
                    [
                        float(record["timing"]["client_e2e_seconds"])
                        for record in window_events
                        if record["cohort"] == cohort
                    ]
                )
                for cohort in ("short", "medium", "long")
            }
            transport = {
                key: _health_percentiles([float(timing[key]) for timing in timing_records])
                for key in (
                    "seconds_to_response_headers",
                    "response_body_transfer_seconds",
                    "seconds_to_decode",
                )
            }
            latency_match = (
                arm["latency"]["client_e2e_seconds"] == client
                and arm["latency"]["client_e2e_seconds_by_cohort"] == by_cohort
                and arm["latency"]["client_transport_window"] == transport
            )
        latency_checks[arm_id] = {
            "passed": timing_records_valid and latency_match,
            "window_requests": len(timing_records),
        }
    checks["r3_and_moe_aggregates"] = {
        "passed": bool(route_checks) and all(item["passed"] for item in route_checks.values()),
        "arms": route_checks,
    }
    checks["client_latency_aggregates"] = {
        "passed": bool(latency_checks) and all(item["passed"] for item in latency_checks.values()),
        "arms": latency_checks,
    }

    def command_flag(command: list[str], flag: str) -> str | None:
        return (
            command[command.index(flag) + 1]
            if command.count(flag) == 1 and command.index(flag) + 1 < len(command)
            else None
        )

    def command_matches(rank_text: str, command: Any) -> bool:
        if not isinstance(command, list):
            return False
        rank = int(rank_text)
        return (
            command[:3] == ["uvx", "--no-config", "--from"]
            and command_flag(command, "--from") == VLLM_FROM_SPEC
            and command_flag(command, "--with") == RUNAI_STREAMER
            and command_flag(command, "--python") == PYTHON_VERSION
            and command_flag(command, "--torch-backend") == "cu130"
            and "vllm" in command
            and "serve" in command
            and command.count("--trust-remote-code") == 1
            and command_flag(command, "--served-model-name") == "exact-reference-ep16"
            and command_flag(command, "--dtype") == "bfloat16"
            and command_flag(command, "--kv-cache-dtype") == "bfloat16"
            and command_flag(command, "--seed") == str(DUMMY_SEED)
            and command_flag(command, "--pipeline-parallel-size") == "1"
            and command_flag(command, "--tensor-parallel-size") == "1"
            and command_flag(command, "--data-parallel-size") == "16"
            and command_flag(command, "--data-parallel-size-local") == "4"
            and command_flag(command, "--data-parallel-start-rank") == str(rank * 4)
            and command_flag(command, "--data-parallel-backend") == "mp"
            and command_flag(command, "--data-parallel-address") in placement["distinct_advertise_hosts"]
            and command_flag(command, "--data-parallel-rpc-port") == str(RPC_PORT)
            and command_flag(command, "--expert-placement-strategy") == "linear"
            and command_flag(command, "--moe-backend") == "triton"
            and command_flag(command, "--attention-backend") == "FLASH_ATTN"
            and command_flag(command, "--max-model-len") == "65536"
            and command_flag(command, "--max-logprobs") == "64"
            and command_flag(command, "--gpu-memory-utilization") == "0.90"
            and command_flag(command, "--max-num-batched-tokens") == str(server_settings["max_num_batched_tokens"])
            and command_flag(command, "--max-num-seqs") == str(server_settings["max_num_seqs"])
            and command_flag(command, "--load-format") == "dummy"
            and command.count("--skip-tokenizer-init") == (0 if request_transport == "chat" else 1)
            and command.count("--enable-expert-parallel") == 1
            and command.count("--enable-prefix-caching") == 1
            and command.count("--enable-chunked-prefill") == 1
            and command.count("--enable-prompt-tokens-details") == 1
            and command.count("--aggregate-engine-logging") == 1
            and "--enforce-eager" not in command
            and (command.count("--enable-return-routed-experts") == 1) is bool(server_settings["r3_enabled"])
            and (
                (
                    rank == 0
                    and command_flag(command, "--api-server-count") == "1"
                    and command_flag(command, "--host") == "0.0.0.0"
                    and command_flag(command, "--port") == str(SERVER_PORT)
                    and "--headless" not in command
                )
                or (rank > 0 and command.count("--headless") == 1 and "--api-server-count" not in command)
            )
        )

    command_settings_match = (
        set(rank_commands) == {"0", "1", "2", "3"}
        and all(command_matches(rank, command) for rank, command in rank_commands.items())
        and len(
            {
                command_flag(command, "--data-parallel-address")
                for command in rank_commands.values()
                if isinstance(command, list)
            }
        )
        == 1
    )
    arm_settings_match = len(result.get("arms", [])) == len(server_settings["concurrencies"]) and all(
        arm["settings"]["target_concurrency"] == concurrency
        and arm["settings"]["max_num_batched_tokens"] == server_settings["max_num_batched_tokens"]
        and arm["settings"]["max_num_seqs"] == server_settings["max_num_seqs"]
        and arm["settings"]["r3_enabled"] is server_settings["r3_enabled"]
        and arm["settings"].get("request_transport", "completion") == request_transport
        and arm["settings"]["settings_drift"] is False
        for arm, concurrency in zip(result.get("arms", []), server_settings["concurrencies"], strict=True)
    )
    rank_provenance_matches = len(rank_records) == 4 and all(
        rank.get("vllm_command") == rank_commands.get(str(rank["rank"]))
        and rank.get("vllm_environment") == server_settings.get("vllm_environment")
        and rank.get("marin_commit") == provenance["marin_commit"]
        and rank.get("vllm_commit") == provenance["vllm_commit"]
        and rank.get("task_image") == provenance["task_image"]
        and rank.get("coscheduling") == UNATTENDED_COSCHEDULING
        for rank in rank_records
    )
    regenerated_workload = _health_workload_manifest(
        deterministic_workload(seed=int(workload["generator_seed"])),
        case=CASES["exact-reference-ep16"],
        concurrencies=[int(value) for value in server_settings["concurrencies"]],
        request_transport=request_transport,
    )
    checks["frozen_protocol"] = {
        "passed": (
            manifest["model_config"] == json.loads(json.dumps(dataclasses.asdict(CASES["exact-reference-ep16"])))
            and result["case"] == "exact-reference-ep16"
            and result["model"] == "exact-reference-ep16"
            and fixture
            == {
                "source": "dummy",
                "weight_dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "seed": DUMMY_SEED,
            }
            and server_settings["pipeline_parallel_size"] == 1
            and server_settings["tensor_parallel_size"] == 1
            and server_settings["data_parallel_size"] == 16
            and server_settings["expert_parallel_size"] == 16
            and server_settings["prefix_caching"] is True
            and server_settings["chunked_prefill"] is True
            and server_settings["cuda_graphs"] is True
            and server_settings.get("aggregate_engine_logging") is True
            and server_settings.get("vllm_environment") == VLLM_SERVER_DEV_MODE_ENVIRONMENT
            and int(server_settings["max_num_seqs"]) > 0
            and all(
                int(concurrency) > 0 and int(concurrency) % 3 == 0 for concurrency in server_settings["concurrencies"]
            )
            and float(protocol["minimum_plateau_seconds"]) >= HEALTH_MINIMUM_SECONDS
            and int(protocol["minimum_plateau_engine_generation_tokens"]) >= HEALTH_MINIMUM_GENERATED_TOKENS
            and float(protocol["minimum_in_flight_fraction"]) == 0.95
            and protocol["drain_excluded"] is True
            and protocol["headline_counter"] == "vllm:generation_tokens"
            and workload["history_lengths"] == [10_240, 30_720, 62_464]
            and workload["append_tokens"] == 1_024
            and workload["response_tokens"] == 2_048
            and workload["final_lengths"] == [13_312, 33_792, 65_536]
            and workload["sampling_parameters"]
            == (HEALTH_SAMPLING_PARAMETERS if request_transport == "completion" else CHAT_HEALTH_SAMPLING_PARAMETERS)
            and workload.get("request_transport", "completion") == request_transport
            and workload["history_policy"] == "append-only frozen token history; one response turn in experiment 0"
            and workload["root_count"] == 18
            and workload["request_count"] == 144
            and workload == regenerated_workload
            and manifest["routing_fixture"]["kind"] == "canonical seeded vLLM dummy routing"
            and manifest["routing_fixture"]["seed"] == DUMMY_SEED
            and manifest["routing_fixture"]["expert_placement"] == "linear contiguous experts per EP rank"
            and manifest["routing_fixture"]["capacity_factor"] is None
            and manifest["routing_fixture"]["balanced_control"]["applicable"] is False
            and manifest["implementation_controls"]["new_hot_path_family"] is False
            and manifest["implementation_controls"]["no_op_control"]["applicable"] is False
            and manifest["r3"]["enabled"] is server_settings["r3_enabled"]
            and manifest["r3"]["expected_layers"] == 48
            and manifest["r3"]["expected_top_k"] == 4
            and manifest["r3"]["carrier"]
            == (
                "OpenAI completion JSON choice.routed_experts; base64-encoded NumPy .npy"
                if server_settings["r3_enabled"] and request_transport == "completion"
                else (
                    "OpenAI chat JSON choice.routed_experts; generated-token-only nested integer lists"
                    if server_settings["r3_enabled"]
                    else "absent"
                )
            )
            and manifest["r3"].get("request_transport", "completion") == request_transport
            and manifest["r3"].get("consumer_commit")
            == (MARINSKYRL_CONSUMER_SHA if request_transport == "chat" else None)
            and manifest["train_to_serve_parity"]["status"] == "inherited from reviewed exact-anchor preflight"
            and command_settings_match
            and arm_settings_match
            and rank_provenance_matches
            and provenance["iris_task_count"] == 4
            and provenance["iris_priority"] in {priority.value for priority in Priority}
            and provenance["iris_coscheduling"] == UNATTENDED_COSCHEDULING
            and provenance["cluster_config"] == DEFAULT_CLUSTER_CONFIG
            and provenance["iris_retry_policy"]
            == {
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
            }
            and provenance["task_image"] == _immutable_image(provenance["task_image"])
            and provenance["vllm_commit"] == VLLM_SHA
            and provenance["marin_commit_url"].endswith(provenance["marin_commit"])
            and provenance["vllm_commit_url"].endswith(provenance["vllm_commit"])
            and len(provenance["dependency_lock_sha256"]) == 64
            and placement["passed"] is True
            and result["placement"] == placement
            and result["all_rank_health"]["passed"] is True
            and len(placement["distinct_advertise_hosts"]) == 4
            and placement["required_coscheduling"] == UNATTENDED_COSCHEDULING
            and placement["topology_enforcement"] == "Kueue hard podset-required-topology"
            and len(gpu_inventory) == 16
            and all("GB200" in gpu["name"] for gpu in gpu_inventory)
            and len({gpu["uuid"] for gpu in gpu_inventory}) == 16
            and all(len(rank.get("gpu_inventory", [])) == 4 for rank in rank_records)
        ),
        "gpu_count": len(gpu_inventory),
        "command_settings_match": command_settings_match,
        "arm_settings_match": arm_settings_match,
        "rank_provenance_matches": rank_provenance_matches,
    }
    result_contract = _health_result_contract(
        result,
        manifest,
        recomputed_repeatability=recomputed_repeatability,
        result_markdown=result_md_bytes.decode(),
    )
    checks["result_contract"] = result_contract
    passed = all(check.get("passed") is True for check in checks.values())
    return {
        "schema_version": 1,
        "kind": "independent-health-artifact-readback",
        "run_id": run_id,
        "artifact_prefix": artifact_prefix,
        "read_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "benchmark_health": {
            "passed": result_contract["benchmark_passed"],
            "status": result_contract["expected_status"],
            "repeatability": recomputed_repeatability,
        },
        "checks": checks,
        "source_object_sha256": {
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            "result.json": hashlib.sha256(result_bytes).hexdigest(),
            "events.jsonl": hashlib.sha256(events_bytes).hexdigest(),
            "metrics/map.json": hashlib.sha256(metrics_map_bytes).hexdigest(),
            "result.md": hashlib.sha256(result_md_bytes).hexdigest(),
        },
    }


def _matrix_phase_evidence_contract(
    phase_result: dict[str, Any],
    planned_phase: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct one phase's exact server commands from its frozen plan."""
    case = CASES[str(planned_phase["case"])]
    rank_records = phase_result.get("all_rank_health", {}).get("ranks", [])
    records_by_rank = {int(record["rank"]): record for record in rank_records}
    endpoints = phase_result.get("placement", {}).get("endpoints", [])
    leader_hosts = [endpoint.get("advertise_host") for endpoint in endpoints if str(endpoint.get("task_index")) == "0"]
    leader_host = leader_hosts[0] if len(leader_hosts) == 1 else None
    expected_environment = dict(VLLM_SERVER_DEV_MODE_ENVIRONMENT)
    if planned_phase["routing_regime"] == "balanced":
        expected_environment["VLLM_GRUGMOE_ROUTING_FIXTURE"] = "balanced"
    if planned_phase["route_audit_mode"] is not None:
        expected_environment["VLLM_GRUGMOE_ROUTE_AUDIT"] = str(planned_phase["route_audit_mode"])

    command_checks: dict[str, bool] = {}
    for rank in range(int(provenance["iris_task_count"])):
        record = records_by_rank.get(rank)
        active = rank < case.node_count
        if record is None:
            command_checks[str(rank)] = False
            continue
        command = record.get("vllm_command")
        if active and isinstance(command, list) and "serve" in command and leader_host is not None:
            serve_index = command.index("serve")
            model_dir = command[serve_index + 1] if serve_index + 1 < len(command) else ""
            expected_command = vllm_command(
                vllm_args(
                    case,
                    model_dir=model_dir,
                    model_source="dummy",
                    leader_ip=leader_host,
                    node_index=rank,
                    smoke=False,
                    r3_enabled=bool(planned_phase["r3_enabled"]),
                    max_num_batched_tokens=int(planned_phase["max_num_batched_tokens"]),
                    max_num_seqs=int(planned_phase["max_num_seqs"]),
                    chat_transport=planned_phase["request_transport"] == "chat",
                )
            )
            expected_command.append("--aggregate-engine-logging")
            command_valid = command == expected_command
        else:
            command_valid = not active and command is None
        command_checks[str(rank)] = (
            command_valid
            and record.get("active") is active
            and int(record.get("active_task_count", -1)) == case.node_count
            and record.get("case") == case.name
            and record.get("vllm_environment") == (expected_environment if active else None)
            and record.get("marin_commit") == provenance["marin_commit"]
            and record.get("vllm_commit") == provenance["vllm_commit"]
            and record.get("task_image") == provenance["task_image"]
            and record.get("coscheduling") == provenance["iris_coscheduling"]
            and record.get("error") is None
            and (not active or record.get("vllm_alive_before_stop") is True)
        )

    startup_by_rank = {int(record["rank"]): record for record in phase_result.get("startup", [])}
    startup_checks = {
        str(rank): (
            rank in records_by_rank
            and startup_by_rank.get(rank, {}).get("command") == records_by_rank[rank].get("vllm_command")
            and startup_by_rank.get(rank, {}).get("environment") == expected_environment
            and startup_by_rank.get(rank, {}).get("alive") is True
            and startup_by_rank.get(rank, {}).get("command_sha256")
            == hashlib.sha256("\0".join(records_by_rank[rank]["vllm_command"]).encode()).hexdigest()
        )
        for rank in range(case.node_count)
    }
    arms = phase_result.get("arms", [])
    arm_checks = [
        arm.get("passed") is True
        and arm.get("matrix", {}).get("phase_id") == planned_phase["phase_id"]
        and arm.get("matrix", {}).get("case") == case.name
        and arm.get("matrix", {}).get("role") == planned_phase["role"]
        and arm.get("matrix", {}).get("active_tasks") == case.node_count
        and arm.get("matrix", {}).get("routing_regime") == planned_phase["routing_regime"]
        and arm.get("matrix", {}).get("order") == planned_phase.get("order")
        and arm.get("matrix", {}).get("replicate") == planned_phase.get("replicate")
        and arm.get("matrix", {}).get("fresh_server") is True
        and arm.get("matrix", {}).get("same_iris_allocation") == provenance["iris_job_id"]
        and arm.get("settings", {}).get("target_concurrency") == concurrency
        and arm.get("settings", {}).get("max_num_batched_tokens") == planned_phase["max_num_batched_tokens"]
        and arm.get("settings", {}).get("max_num_seqs") == planned_phase["max_num_seqs"]
        and arm.get("settings", {}).get("r3_enabled") is planned_phase["r3_enabled"]
        and arm.get("settings", {}).get("request_transport") == planned_phase["request_transport"]
        and arm.get("settings", {}).get("routing_regime") == planned_phase["routing_regime"]
        and arm.get("settings", {}).get("route_audit_mode") == planned_phase["route_audit_mode"]
        and arm.get("settings", {}).get("settings_drift") is False
        for arm, concurrency in zip(arms, planned_phase["concurrencies"], strict=False)
    ]
    passed = (
        phase_result.get("phase") == planned_phase
        and phase_result.get("phase_id") == planned_phase["phase_id"]
        and phase_result.get("model") == case.name
        and phase_result.get("passed") is True
        and phase_result.get("error") is None
        and phase_result.get("placement", {}).get("passed") is True
        and phase_result.get("all_rank_health", {}).get("passed") is True
        and len(rank_records) == int(provenance["iris_task_count"])
        and len(records_by_rank) == len(rank_records)
        and len(leader_hosts) == 1
        and bool(command_checks)
        and all(command_checks.values())
        and len(startup_by_rank) == case.node_count
        and all(startup_checks.values())
        and len(arms) == len(planned_phase["concurrencies"])
        and len(arm_checks) == len(arms)
        and all(arm_checks)
    )
    return {
        "passed": passed,
        "phase_id": planned_phase["phase_id"],
        "command_checks": command_checks,
        "startup_checks": startup_checks,
        "arm_checks": arm_checks,
    }


def _matrix_matched_chat_pair_contract(
    phase_results: list[dict[str, Any]],
    *,
    off_phase_id: str,
    on_phase_id: str,
) -> dict[str, Any]:
    """Prove a fresh chat pair changes only the R3 server flag."""
    by_id = {phase.get("phase_id"): phase for phase in phase_results}
    off = by_id.get(off_phase_id)
    on = by_id.get(on_phase_id)
    if off is None or on is None:
        return {"passed": False, "error": "matched chat phases are incomplete"}

    def normalize(command: Any) -> list[str] | None:
        if not isinstance(command, list) or "serve" not in command:
            return None
        normalized = list(command)
        serve_index = normalized.index("serve")
        if serve_index + 1 >= len(normalized):
            return None
        normalized[serve_index + 1] = "<fresh-model-directory>"
        if "--enable-return-routed-experts" in normalized:
            normalized.remove("--enable-return-routed-experts")
        return normalized

    off_plan = dict(off["phase"])
    on_plan = dict(on["phase"])
    for plan in (off_plan, on_plan):
        plan.pop("phase_id", None)
        plan.pop("r3_enabled", None)
    off_ranks = {int(record["rank"]): record for record in off["all_rank_health"]["ranks"] if record["active"]}
    on_ranks = {int(record["rank"]): record for record in on["all_rank_health"]["ranks"] if record["active"]}
    command_checks = {
        str(rank): (
            off_ranks[rank]["vllm_command"].count("--enable-return-routed-experts") == 0
            and on_ranks[rank]["vllm_command"].count("--enable-return-routed-experts") == 1
            and normalize(off_ranks[rank]["vllm_command"]) == normalize(on_ranks[rank]["vllm_command"])
            and off_ranks[rank]["vllm_environment"] == on_ranks[rank]["vllm_environment"]
        )
        for rank in sorted(set(off_ranks) & set(on_ranks))
    }
    passed = (
        off_plan == on_plan and set(off_ranks) == set(on_ranks) and bool(command_checks) and all(command_checks.values())
    )
    return {
        "passed": passed,
        "off_phase_id": off_phase_id,
        "on_phase_id": on_phase_id,
        "command_checks": command_checks,
    }


def _matrix_coarse_curve_contract(
    arm: dict[str, Any],
    *,
    workload: dict[str, Any],
    parsed_by_path: dict[str, list[Any]],
    snapshot_by_path: dict[str, dict[str, Any]],
    event_records: list[dict[str, Any]],
    kv_source_by_path: dict[str, bytes],
) -> dict[str, Any]:
    """Recompute each homogeneous context slice from raw counters and the frozen workload."""
    case = CASES[str(arm["matrix"]["case"])]
    requests = {str(request["request_id"]): request for request in workload["requests"]}
    points = arm.get("coarse_curve", [])
    point_checks: list[dict[str, Any]] = []
    for point, cohort in zip(points, ("short", "medium", "long"), strict=False):
        start_path = str(point["metrics"]["boundary_start"])
        end_path = str(point["metrics"]["boundary_end"])
        start = parsed_by_path.get(start_path, [])
        end = parsed_by_path.get(end_path, [])
        start_snapshot = snapshot_by_path.get(start_path, {})
        end_snapshot = snapshot_by_path.get(end_path, {})
        generation_delta = round(
            prometheus_value(end, "vllm:generation_tokens") - prometheus_value(start, "vllm:generation_tokens")
        )
        success_delta = round(
            prometheus_value(end, "vllm:request_success") - prometheus_value(start, "vllm:request_success")
        )
        preemption_delta = round(
            prometheus_value(end, "vllm:num_preemptions") - prometheus_value(start, "vllm:num_preemptions")
        )
        elapsed = float(end_snapshot.get("monotonic_seconds", 0)) - float(start_snapshot.get("monotonic_seconds", 0))
        expected_rate = generation_delta / elapsed / case.data_parallel_size if elapsed > 0 else math.inf
        schedule = point.get("schedule", [])
        scheduled_requests = [requests.get(str(item.get("request_id"))) for item in schedule]
        expected_cohort_requests = [request for request in workload["requests"] if str(request.get("cohort")) == cohort]
        expected_concurrency = len(expected_cohort_requests)
        expected_generation = sum(int(request["max_tokens"]) for request in scheduled_requests if request is not None)
        expected_population = len(
            {
                (int(item["data_parallel_rank"]), int(request["root"]))
                for item, request in zip(schedule, scheduled_requests, strict=False)
                if request is not None
            }
        )
        cohort_requests = [request for request in scheduled_requests if request is not None]
        final_lengths = {int(request["final_token_count"]) for request in cohort_requests}
        schedule_valid = (
            len(schedule) == expected_concurrency
            and [int(item["slot"]) for item in schedule] == list(range(expected_concurrency))
            and all(
                request is not None
                and request["cohort"] == cohort
                and int(item["data_parallel_rank"]) == int(request["root"]) % case.data_parallel_size
                and item["prompt_token_ids_sha256"] == request["prompt_token_ids_sha256"]
                for item, request in zip(schedule, scheduled_requests, strict=False)
            )
            and [str(item["request_id"]) for item in schedule]
            == [str(request["request_id"]) for request in expected_cohort_requests]
            and {int(item["data_parallel_rank"]) for item in schedule}
            == {int(request["root"]) % case.data_parallel_size for request in expected_cohort_requests}
        )
        expected_events = {
            f"{arm['arm_id']}-{cohort}-slot-{int(item['slot']):03d}-{item['request_id']}": (
                item,
                request,
            )
            for item, request in zip(schedule, scheduled_requests, strict=False)
            if request is not None
        }
        observed_event_records = [
            event
            for event in event_records
            if event.get("event") == "cohort_slice_request_completed"
            and event.get("arm_id") == arm["arm_id"]
            and event.get("cohort") == cohort
        ]
        observed_events = {str(event.get("request_id")): event for event in observed_event_records}
        events_valid = (
            len(observed_event_records) == len(expected_events)
            and set(observed_events) == set(expected_events)
            and all(
                event.get("manifest_request_id") == item["request_id"]
                and int(event.get("data_parallel_rank", -1)) == int(item["data_parallel_rank"])
                and int(event.get("completion_tokens", -1)) == int(request["max_tokens"])
                and event.get("prompt_token_ids_sha256") == request["prompt_token_ids_sha256"]
                and all(
                    isinstance(event.get(field), str) and len(event[field]) == 64
                    for field in (
                        "prompt_token_ids_sha256",
                        "generated_token_ids_sha256",
                        "final_prefix_token_ids_sha256",
                        "sampled_token_logprobs_sha256",
                    )
                )
                for request_id, (item, request) in expected_events.items()
                for event in [observed_events[request_id]]
            )
        )
        schedule_types = layer_types(case.num_hidden_layers, global_interval=case.global_every)
        expected_prediction = (
            {
                "final_context_tokens": next(iter(final_lengths)),
                "local_layers": schedule_types.count("sliding_attention"),
                "global_layers": schedule_types.count("full_attention"),
                "per_live_sequence_bytes": predict_kv_bytes(
                    sequence_length=next(iter(final_lengths)),
                    local_layers=schedule_types.count("sliding_attention"),
                    global_layers=schedule_types.count("full_attention"),
                    local_kv_heads=case.local_kv_heads,
                    global_kv_heads=case.global_kv_heads,
                    head_dim=case.head_dim,
                    sliding_window=case.sliding_window,
                ),
                "scope": "semantic K and V payload before block rounding",
            }
            if len(final_lengths) == 1
            else None
        )
        kv = point.get("kv_cache", {})
        kv_source = kv.get("source", {})
        kv_source_path = str(kv_source.get("path", ""))
        kv_source_payload = kv_source_by_path.get(kv_source_path, b"")
        recomputed_kv = _health_kv_summary_from_text(
            kv_source_payload.decode(errors="replace"),
            case=case,
            target_concurrency=expected_concurrency,
        )
        recomputed_kv["attention_prediction"] = expected_prediction
        stored_kv_without_source = {key: value for key, value in kv.items() if key != "source"}
        kv_recomputed = (
            kv_source_path in kv_source_by_path
            and int(kv_source.get("bytes", -1)) == len(kv_source_payload)
            and kv_source.get("sha256") == hashlib.sha256(kv_source_payload).hexdigest()
            and stored_kv_without_source == recomputed_kv
        )
        point_passed = (
            point.get("cohort") == cohort
            and bool(start)
            and bool(end)
            and point.get("passed") is True
            and all(point.get("gates", {}).values())
            and schedule_valid
            and events_valid
            and int(point.get("slice_concurrency", -1)) == expected_concurrency
            and int(point.get("measured_requests", -1)) == expected_concurrency
            and int(point.get("population_requests", -1)) == expected_population
            and generation_delta == expected_generation == int(point.get("engine_generation_tokens", -1))
            and success_delta == expected_concurrency
            and preemption_delta == int(point.get("preemptions", -1)) == 0
            and math.isclose(elapsed, float(point.get("elapsed_seconds", 0)), rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(
                expected_rate,
                float(point.get("generation_tokens_per_second_per_gpu", math.inf)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                case.data_parallel_size * elapsed / generation_delta,
                float(point.get("gpu_seconds_per_generated_token", math.inf)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and len(final_lengths) == 1
            and int(point.get("final_context_tokens", -1)) == next(iter(final_lengths))
            and kv.get("passed") is True
            and all(kv.get("gates", {}).values())
            and kv.get("attention_prediction") == expected_prediction
            and kv_recomputed
        )
        point_checks.append(
            {
                "cohort": cohort,
                "passed": point_passed,
                "schedule_valid": schedule_valid,
                "events_valid": events_valid,
                "kv_recomputed": kv_recomputed,
                "generation_delta": generation_delta,
                "success_delta": success_delta,
                "preemption_delta": preemption_delta,
                "elapsed_seconds": elapsed,
                "expected_rate": expected_rate,
            }
        )
    short_rate = float(points[0]["generation_tokens_per_second_per_gpu"]) if len(points) == 3 else math.inf
    slowdown_valid = len(points) == 3 and all(
        math.isclose(
            float(point["slowdown_from_short_percent"]),
            100 * (1 - float(point["generation_tokens_per_second_per_gpu"]) / short_rate),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for point in points
    )
    return {
        "applicable": True,
        "passed": (
            [point.get("cohort") for point in points] == ["short", "medium", "long"]
            and len(point_checks) == 3
            and all(item["passed"] for item in point_checks)
            and slowdown_valid
        ),
        "slowdown_valid": slowdown_valid,
        "points": point_checks,
    }


def _matrix_sequence_probe_contract(
    arm: dict[str, Any],
    *,
    field: str,
    manifest: dict[str, Any],
    parsed_by_path: dict[str, list[Any]],
    snapshot_by_path: dict[str, dict[str, Any]],
    event_records: list[dict[str, Any]],
    kv_source_by_path: dict[str, bytes],
) -> dict[str, Any]:
    """Recompute a trajectory or capacity probe from raw counters and compact hashes."""
    probe = arm.get(field, {})
    event_name = "trajectory_turn_completed" if field == "trajectory_65k" else "capacity_131k_request_completed"
    start_path = str(probe.get("metrics", {}).get("boundary_start", ""))
    end_path = str(probe.get("metrics", {}).get("boundary_end", ""))
    start = parsed_by_path.get(start_path, [])
    end = parsed_by_path.get(end_path, [])
    start_snapshot = snapshot_by_path.get(start_path, {})
    end_snapshot = snapshot_by_path.get(end_path, {})
    generation_delta = round(
        prometheus_value(end, "vllm:generation_tokens") - prometheus_value(start, "vllm:generation_tokens")
    )
    success_delta = round(
        prometheus_value(end, "vllm:request_success") - prometheus_value(start, "vllm:request_success")
    )
    preemption_delta = round(
        prometheus_value(end, "vllm:num_preemptions") - prometheus_value(start, "vllm:num_preemptions")
    )
    elapsed = float(end_snapshot.get("monotonic_seconds", 0)) - float(start_snapshot.get("monotonic_seconds", 0))
    provenance = probe.get("request_provenance", [])
    observed_events = [
        event for event in event_records if event.get("event") == event_name and event.get("arm_id") == arm.get("arm_id")
    ]
    compact_events = [
        {key: value for key, value in event.items() if key not in {"timestamp", "monotonic_seconds", "event", "arm_id"}}
        for event in observed_events
    ]
    event_sort_key = (
        (lambda record: (str(record.get("request_id")), int(record.get("turn", -1))))
        if field == "trajectory_65k"
        else (lambda record: str(record.get("request_id")))
    )
    events_match_result = sorted(compact_events, key=event_sort_key) == sorted(provenance, key=event_sort_key)
    request_by_id = {str(request["request_id"]): request for request in manifest.get("requests", [])}
    root_by_id = {int(root["root"]): root for root in manifest.get("roots", [])}
    event_hashes_valid = all(
        all(
            isinstance(record.get(key), str) and len(record[key]) == 64
            for key in (
                "prompt_token_ids_sha256",
                "generated_token_ids_sha256",
                "final_prefix_token_ids_sha256",
                "sampled_token_logprobs_sha256",
            )
        )
        for record in provenance
    )
    if field == "trajectory_65k":
        by_key = {(str(record["request_id"]), int(record["turn"])): record for record in provenance}
        expected_keys = {
            (str(request["request_id"]), int(turn["turn"]))
            for request in manifest.get("requests", [])
            for turn in request.get("turns", [])
        }
        schedule_valid = (
            len(provenance) == len(expected_keys)
            and set(by_key) == expected_keys
            and all(
                bool(record)
                and record["cohort"] == request["cohort"]
                and int(record["root"]) == int(request["root"])
                and int(record["branch"]) == int(request["branch"])
                and int(record["data_parallel_rank"]) == int(request["data_parallel_rank"])
                and int(record["prompt_tokens"]) == int(turn["prompt_token_count"])
                and int(record["completion_tokens"]) == int(turn["max_tokens"])
                and int(record["final_token_count"]) == int(turn["final_token_count"])
                and int(record["sampled_token_logprobs_count"]) == int(turn["max_tokens"])
                and record["append_token_ids_sha256"] == turn["append_token_ids_sha256"]
                and (
                    record["carried_prefix_token_ids_sha256"] == root_by_id[int(request["root"])]["token_ids_sha256"]
                    if int(turn["turn"]) == 1
                    else record["carried_prefix_token_ids_sha256"]
                    == by_key[(str(request["request_id"]), int(turn["turn"]) - 1)]["final_prefix_token_ids_sha256"]
                )
                for request in manifest.get("requests", [])
                for turn in request.get("turns", [])
                for record in [by_key.get((str(request["request_id"]), int(turn["turn"])), {})]
            )
        )
        expected_generation = (
            int(manifest.get("request_count", 0))
            * int(manifest.get("turn_count", 0))
            * int(manifest.get("response_tokens_per_turn", 0))
        )
        expected_successes = int(manifest.get("request_count", 0)) * int(manifest.get("turn_count", 0))
    else:
        by_key = {str(record["request_id"]): record for record in provenance}
        expected_keys = set(request_by_id)
        schedule_valid = (
            len(provenance) == len(expected_keys)
            and set(by_key) == expected_keys
            and all(
                bool(record)
                and int(record["root"]) == int(request["root"])
                and int(record["branch"]) == int(request["branch"])
                and int(record["data_parallel_rank"]) == int(request["data_parallel_rank"])
                and int(record["prompt_tokens"]) == int(request["prompt_token_count"])
                and int(record["completion_tokens"]) == int(request["max_tokens"])
                and int(record["final_token_count"]) == int(request["final_token_count"])
                and int(record["sampled_token_logprobs_count"]) == int(request["max_tokens"])
                and record["prompt_token_ids_sha256"] == request["prompt_token_ids_sha256"]
                for request_id, request in request_by_id.items()
                for record in [by_key.get(request_id, {})]
            )
        )
        expected_generation = int(manifest.get("request_count", 0)) * int(manifest.get("response_tokens", 0))
        expected_successes = int(manifest.get("request_count", 0))
    case = CASES[str(arm["matrix"]["case"])]
    kv = probe.get("kv_cache", {})
    kv_source = kv.get("source", {})
    kv_source_path = str(kv_source.get("path", ""))
    kv_source_payload = kv_source_by_path.get(kv_source_path, b"")
    target_concurrency = int(manifest.get("request_count", 0))
    recomputed_kv = _health_kv_summary_from_text(
        kv_source_payload.decode(errors="replace"),
        case=case,
        target_concurrency=target_concurrency,
    )
    stored_kv_without_source = {key: value for key, value in kv.items() if key != "source"}
    kv_recomputed = stored_kv_without_source == recomputed_kv
    kv_source_valid = (
        kv_source_path in kv_source_by_path
        and int(kv_source.get("bytes", -1)) == len(kv_source_payload)
        and kv_source.get("sha256") == hashlib.sha256(kv_source_payload).hexdigest()
    )
    expected_rate = generation_delta / elapsed / case.data_parallel_size if elapsed > 0 else math.inf
    passed = (
        bool(start)
        and bool(end)
        and probe.get("passed") is True
        and all(probe.get("gates", {}).values())
        and schedule_valid
        and event_hashes_valid
        and events_match_result
        and generation_delta == expected_generation == int(probe.get("engine_generation_tokens", -1))
        and success_delta == expected_successes == int(probe.get("request_successes", -1))
        and preemption_delta == int(probe.get("preemptions", -1)) == 0
        and math.isclose(elapsed, float(probe.get("elapsed_seconds", 0)), rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(
            expected_rate,
            float(probe.get("generation_tokens_per_second_per_gpu", math.inf)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and kv.get("passed") is True
        and kv_recomputed
        and kv_source_valid
    )
    return {
        "applicable": True,
        "passed": passed,
        "field": field,
        "schedule_valid": schedule_valid,
        "events_match_result": events_match_result,
        "generation_delta": generation_delta,
        "success_delta": success_delta,
        "preemption_delta": preemption_delta,
        "kv_recomputed": kv_recomputed,
        "kv_source_valid": kv_source_valid,
    }


def readback_matrix_artifacts(filesystem: Any, *, plan: str, run_id: str) -> dict[str, Any]:
    """Independently re-read the matrix and recompute every decision field."""
    artifact_prefix = _matrix_artifact_prefix(plan, run_id)
    root_key = _s3_key(artifact_prefix).rstrip("/")

    def read(relative: str) -> bytes:
        return filesystem.cat_file(f"{root_key}/{relative}")

    manifest_bytes = read("manifest.json")
    result_bytes = read("result.json")
    events_bytes = read("events.jsonl")
    metrics_map_bytes = read("metrics/map.json")
    result_md_bytes = read("result.md")
    manifest = json.loads(manifest_bytes)
    result = json.loads(result_bytes)
    metrics_map = json.loads(metrics_map_bytes)
    checks: dict[str, Any] = {}
    checks["identity"] = {
        "passed": (
            manifest.get("kind") == "grugmoe-benchmark-matrix-manifest"
            and result.get("kind") == "grugmoe-benchmark-matrix"
            and manifest.get("plan") == plan
            and result.get("plan") == plan
            and manifest.get("run_id") == run_id
            and result.get("run_id") == run_id
            and result.get("manifest_sha256") == hashlib.sha256(manifest_bytes).hexdigest()
        )
    }
    claimed_checks: dict[str, Any] = {}
    for relative, claim in manifest.get("claimed_files", {}).items():
        payload = read(relative)
        claimed_checks[relative] = {
            "passed": len(payload) == int(claim["bytes"]) and hashlib.sha256(payload).hexdigest() == claim["sha256"],
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    checks["claimed_file_hashes"] = {
        "passed": bool(claimed_checks) and all(item["passed"] for item in claimed_checks.values()),
        "files": claimed_checks,
    }
    snapshots = metrics_map.get("snapshots", [])
    parsed_by_path: dict[str, list[Any]] = {}
    raw_checks: dict[str, Any] = {}
    for index, entry in enumerate(snapshots):
        payload = read(entry["path"])
        parsed = parse_labeled_prometheus(payload.decode())
        totals = {metric: prometheus_value(parsed, metric) for metric in HEALTH_COUNTER_METRICS}
        by_engine = {
            metric: prometheus_values_by_label(parsed, metric, label="engine") for metric in HEALTH_ENGINE_METRICS
        }
        totals_match = set(entry.get("totals", {})) == set(HEALTH_COUNTER_METRICS) and all(
            math.isclose(float(entry["totals"][metric]), value, rel_tol=1e-12, abs_tol=1e-12)
            for metric, value in totals.items()
        )
        raw_checks[entry["path"]] = {
            "passed": (
                int(entry["index"]) == index
                and len(payload) == int(entry["bytes"])
                and hashlib.sha256(payload).hexdigest() == entry["sha256"]
                and totals_match
                and entry.get("by_engine") == by_engine
            ),
            "totals_match": totals_match,
            "by_engine_match": entry.get("by_engine") == by_engine,
        }
        parsed_by_path[entry["path"]] = parsed
    checks["raw_metrics"] = {
        "passed": (
            bool(snapshots)
            and len({entry["path"] for entry in snapshots}) == len(snapshots)
            and all(item["passed"] for item in raw_checks.values())
        ),
        "snapshots": len(snapshots),
        "files": raw_checks,
    }
    snapshot_by_path = {entry["path"]: entry for entry in snapshots}
    event_records = [json.loads(line) for line in events_bytes.decode().splitlines() if line.strip()]
    kv_source_paths = {
        str(arm[field]["kv_cache"]["source"]["path"])
        for arm in result.get("arms", [])
        for field in ("trajectory_65k", "capacity_stress_131k")
        if field in arm
    }
    kv_source_paths.update(
        str(point["kv_cache"]["source"]["path"])
        for arm in result.get("arms", [])
        for point in arm.get("coarse_curve", [])
        if point.get("kv_cache", {}).get("source", {}).get("path")
    )
    kv_sources = {path: read(path) for path in kv_source_paths}
    arm_checks: dict[str, Any] = {}
    for arm in result.get("arms", []):
        start_path = arm["metrics"]["boundary_start"]
        end_path = arm["metrics"]["boundary_end"]
        start = parsed_by_path.get(start_path, [])
        end = parsed_by_path.get(end_path, [])
        generation_delta = prometheus_value(end, "vllm:generation_tokens") - prometheus_value(
            start, "vllm:generation_tokens"
        )
        prompt_delta = prometheus_value(end, "vllm:prompt_tokens") - prometheus_value(start, "vllm:prompt_tokens")
        preemption_delta = prometheus_value(end, "vllm:num_preemptions") - prometheus_value(
            start, "vllm:num_preemptions"
        )
        elapsed = float(arm["plateau"]["elapsed_seconds"])
        gpu_count = int(arm["headline"]["gpu_count"])
        expected_rate = generation_delta / elapsed / gpu_count if elapsed and gpu_count else math.inf
        audit = arm["moe_routing"].get("aggregate_route_audit")
        audit_valid = True
        if audit is not None:
            mode = arm["settings"]["route_audit_mode"]
            audit_valid = (
                audit.get("passed") is True
                and all(audit.get("gates", {}).values())
                and int(audit["assignment_count"])
                == (int(audit["expected_assignment_count"]) if mode == "record" else 0)
                and int(audit["counts_outside_ownership"]) == 0
                and len(audit["workers"]) == gpu_count
                and len(audit["expert_assignment_counts"]) == 128
                and len(audit["layer_assignment_counts"]) == 48
            )
        carrier = arm["moe_routing"]["carrier"]
        case = CASES[arm["matrix"]["case"]]
        request_transport = arm["settings"]["request_transport"]
        workload = manifest["workloads"][f"{case.name}:{request_transport}"]
        request_by_id = {request["request_id"]: request for request in workload["requests"]}
        curve_contract = (
            _matrix_coarse_curve_contract(
                arm,
                workload=workload,
                parsed_by_path=parsed_by_path,
                snapshot_by_path=snapshot_by_path,
                event_records=event_records,
                kv_source_by_path=kv_sources,
            )
            if plan == "attention-pair-v1"
            else {"applicable": False, "passed": "coarse_curve" not in arm}
        )
        trajectory_contract = (
            _matrix_sequence_probe_contract(
                arm,
                field="trajectory_65k",
                manifest=manifest.get("trajectory_workloads", {}).get(case.name, {}),
                parsed_by_path=parsed_by_path,
                snapshot_by_path=snapshot_by_path,
                event_records=event_records,
                kv_source_by_path=kv_sources,
            )
            if plan == "attention-finalist-v1"
            else {"applicable": False, "passed": "trajectory_65k" not in arm}
        )
        capacity_contract = (
            _matrix_sequence_probe_contract(
                arm,
                field="capacity_stress_131k",
                manifest=manifest.get("capacity_workloads", {}).get(case.name, {}),
                parsed_by_path=parsed_by_path,
                snapshot_by_path=snapshot_by_path,
                event_records=event_records,
                kv_source_by_path=kv_sources,
            )
            if plan == "attention-finalist-v1"
            else {"applicable": False, "passed": "capacity_stress_131k" not in arm}
        )
        completion_events = [
            record
            for record in event_records
            if record.get("event") == "request_completed" and record.get("arm_id") == arm["arm_id"]
        ]
        boundary_start = float(snapshot_by_path[start_path]["monotonic_seconds"])
        boundary_end = float(snapshot_by_path[end_path]["monotonic_seconds"])
        window_events = [
            record
            for record in completion_events
            if boundary_start <= float(record["completed_at_monotonic_seconds"]) <= boundary_end
        ]
        event_contract = (
            len(completion_events) == int(arm["requests"]["whole_run_successes"])
            and len(window_events) == int(arm["requests"]["plateau_successes"])
            and all(
                event.get("manifest_request_id") in request_by_id
                and event.get("prompt_token_ids_sha256")
                == request_by_id[event["manifest_request_id"]]["prompt_token_ids_sha256"]
                and int(event.get("completion_tokens", -1))
                == int(request_by_id[event["manifest_request_id"]]["max_tokens"])
                and int(event.get("sampled_token_logprobs_count", -1)) == int(event.get("completion_tokens", -2))
                and all(
                    isinstance(event.get(field), str) and len(event[field]) == 64
                    for field in (
                        "prompt_token_ids_sha256",
                        "generated_token_ids_sha256",
                        "final_prefix_token_ids_sha256",
                        "sampled_token_logprobs_sha256",
                    )
                )
                for event in completion_events
            )
            and _health_final_prefix_provenance(completion_events) == arm["requests"]["final_prefix_provenance"]
        )
        route_event_contract = True
        for event in window_events:
            summary = event.get("route_summary")
            if arm["settings"]["r3_enabled"]:
                request = request_by_id[event["manifest_request_id"]]
                expected_positions = int(request["max_tokens"])
                assignments = expected_positions * case.num_hidden_layers * case.num_experts_per_tok
                expert_histogram = summary.get("expert_histogram", []) if isinstance(summary, dict) else []
                ep_rank_histogram = summary.get("ep_rank_histogram", []) if isinstance(summary, dict) else []
                route_event_contract = route_event_contract and (
                    isinstance(summary, dict)
                    and request_transport == "chat"
                    and summary.get("enabled") is True
                    and summary.get("shape") == [expected_positions, case.num_hidden_layers, case.num_experts_per_tok]
                    and summary.get("dtype") in {"int32", "int64"}
                    and summary.get("all_expected_positions_layers_topk_aligned") is True
                    and int(summary.get("root_prefix_positions_compared", -1)) == 0
                    and summary.get("root_prefix_aligned") is True
                    and event.get("route_sha256") == summary.get("route_sha256")
                    and isinstance(summary.get("route_sha256"), str)
                    and len(summary["route_sha256"]) == 64
                    and len(expert_histogram) == case.num_experts
                    and len(ep_rank_histogram) == case.data_parallel_size
                    and sum(int(value) for value in expert_histogram) == assignments
                    and int(summary.get("carrier_array_bytes", -1)) == assignments * 8
                    and int(summary.get("carrier_json_bytes", -1)) > 0
                )
            else:
                route_event_contract = route_event_contract and (
                    isinstance(summary, dict)
                    and summary.get("enabled") is False
                    and summary.get("transport") == "absent"
                    and int(summary.get("carrier_payload_bytes", -1)) == 0
                    and event.get("route_sha256") is None
                )
        topology_no_response_routes = arm["matrix"]["role"] not in {"topology-comparison", "audit-control"} or (
            arm["settings"]["r3_enabled"] is False
            and arm["moe_routing"]["expert_histogram"] is None
            and carrier["transport"] == "absent"
            and int(carrier["full_response_bytes"]) > 0
        )
        arm_checks[arm["arm_id"]] = {
            "passed": (
                bool(start)
                and bool(end)
                and arm.get("passed") is True
                and all(arm.get("gates", {}).values())
                and math.isclose(generation_delta, float(arm["plateau"]["generated_tokens"]), abs_tol=0.5)
                and math.isclose(prompt_delta, float(arm["plateau"]["processed_prompt_tokens"]), abs_tol=0.5)
                and math.isclose(
                    expected_rate,
                    float(arm["headline"]["generation_tokens_per_second_per_gpu"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and round(preemption_delta) == int(arm["preemptions"]) == 0
                and arm["resident_capacity"]["zero_preemptions"] is True
                and len(arm["resident_capacity"]["by_data_parallel_rank"]) == gpu_count
                and audit_valid
                and event_contract
                and route_event_contract
                and topology_no_response_routes
                and curve_contract["passed"] is True
                and trajectory_contract["passed"] is True
                and capacity_contract["passed"] is True
            ),
            "generation_delta": generation_delta,
            "prompt_delta": prompt_delta,
            "preemption_delta": preemption_delta,
            "audit_valid": audit_valid,
            "event_contract": event_contract,
            "route_event_contract": route_event_contract,
            "topology_no_response_routes": topology_no_response_routes,
            "coarse_curve": curve_contract,
            "trajectory_65k": trajectory_contract,
            "capacity_stress_131k": capacity_contract,
        }
    checks["arm_recomputation"] = {
        "passed": bool(arm_checks) and all(item["passed"] for item in arm_checks.values()),
        "arms": arm_checks,
    }
    regenerated_workloads: dict[str, Any] = {}
    for key, workload in manifest.get("workloads", {}).items():
        case_name, request_transport = key.split(":", 1)
        regenerated_workloads[key] = _health_workload_manifest(
            deterministic_workload(seed=int(workload["generator_seed"])),
            case=CASES[case_name],
            concurrencies=[int(value) for value in workload["slot_schedules"]],
            request_transport=request_transport,
        )
    checks["frozen_workloads"] = {
        "passed": bool(regenerated_workloads) and manifest.get("workloads") == regenerated_workloads,
        "contracts": sorted(regenerated_workloads),
    }
    regenerated_trajectory = {
        name: _trajectory_workload_manifest(
            deterministic_trajectory_workload(seed=int(workload["generator_seed"])),
            case=CASES[name],
        )
        for name, workload in manifest.get("trajectory_workloads", {}).items()
    }
    regenerated_capacity = {
        name: _capacity_workload_manifest(
            deterministic_capacity_stress_workload(seed=int(workload["generator_seed"])),
            case=CASES[name],
        )
        for name, workload in manifest.get("capacity_workloads", {}).items()
    }
    checks["frozen_sequence_workloads"] = {
        "passed": (
            (
                bool(regenerated_trajectory)
                and bool(regenerated_capacity)
                and manifest.get("trajectory_workloads") == regenerated_trajectory
                and manifest.get("capacity_workloads") == regenerated_capacity
            )
            if plan == "attention-finalist-v1"
            else manifest.get("trajectory_workloads", {}) == {} and manifest.get("capacity_workloads", {}) == {}
        ),
        "trajectory_cases": sorted(regenerated_trajectory),
        "capacity_cases": sorted(regenerated_capacity),
    }
    expected_model_configs = {
        name: dataclasses.asdict(CASES[name]) for name in sorted(manifest.get("model_configs", {}))
    }
    checks["frozen_model_configs"] = {
        "passed": bool(expected_model_configs) and manifest.get("model_configs") == expected_model_configs,
        "cases": sorted(expected_model_configs),
    }
    if plan == "attention-pair-v1":
        planned_attention = [
            phase for phase in manifest.get("phase_plan", []) if phase.get("role") == "attention-comparison"
        ]
        candidate_names = {
            str(phase["case"]) for phase in planned_attention if phase.get("case") != "exact-reference-ep16"
        }
        candidate_name = next(iter(candidate_names)) if len(candidate_names) == 1 else None
        expected_property = {
            "window1024-ep16": "sliding_window",
            "window2048-ep16": "sliding_window",
            "global-every4-ep16": "global_every",
        }.get(candidate_name)
        changed_properties = (
            {
                key
                for key, value in dataclasses.asdict(CASES["exact-reference-ep16"]).items()
                if key != "name" and dataclasses.asdict(CASES[candidate_name])[key] != value
            }
            if candidate_name in ATTENTION_CANDIDATES
            else set()
        )
        normalized_phases = []
        for phase in planned_attention:
            normalized = dict(phase)
            normalized.pop("phase_id", None)
            normalized.pop("case", None)
            normalized_phases.append(normalized)
        order = planned_attention[0].get("order") if len(planned_attention) == 2 else None
        expected_sequence = (
            ["exact-reference-ep16", candidate_name]
            if order == "ab"
            else [candidate_name, "exact-reference-ep16"] if order == "ba" else []
        )
        reference_workload = manifest.get("workloads", {}).get("exact-reference-ep16:completion")
        candidate_workload = manifest.get("workloads", {}).get(f"{candidate_name}:completion")
        checks["attention_pair_protocol"] = {
            "applicable": True,
            "passed": (
                len(planned_attention) == 2
                and candidate_name in ATTENTION_CANDIDATES
                and changed_properties == {expected_property}
                and [phase.get("case") for phase in planned_attention] == expected_sequence
                and len(normalized_phases) == 2
                and normalized_phases[0] == normalized_phases[1]
                and all(phase.get("homogeneous_slices") is True for phase in planned_attention)
                and reference_workload == candidate_workload
            ),
            "candidate": candidate_name,
            "order": order,
            "changed_properties": sorted(changed_properties),
            "expected_property": expected_property,
            "identical_frozen_workloads": reference_workload == candidate_workload,
        }
    else:
        checks["attention_pair_protocol"] = {"applicable": False, "passed": True}
    if plan == "attention-finalist-v1":
        planned_finalist = [
            phase for phase in manifest.get("phase_plan", []) if phase.get("role") == "attention-finalist-validation"
        ]
        cases = [str(phase.get("case")) for phase in planned_finalist]
        reference_case = "exact-reference-131k-ep16"
        candidate_cases = [case for case in cases if case != reference_case]
        candidate_case = candidate_cases[0] if len(candidate_cases) == 1 else None
        expected_property = {
            "window1024-131k-ep16": "sliding_window",
            "window2048-131k-ep16": "sliding_window",
            "global-every4-131k-ep16": "global_every",
        }.get(candidate_case)
        changed_properties = (
            {
                key
                for key, value in dataclasses.asdict(CASES[reference_case]).items()
                if key != "name" and dataclasses.asdict(CASES[candidate_case])[key] != value
            }
            if candidate_case is not None
            else set()
        )
        expected_cases = [reference_case, candidate_case] if candidate_case is not None else [reference_case]
        normalized = []
        for phase in planned_finalist:
            item = dict(phase)
            item.pop("phase_id", None)
            item.pop("case", None)
            item.pop("replicate", None)
            normalized.append(item)
        health_workloads = manifest.get("workloads", {})
        trajectory_workloads = manifest.get("trajectory_workloads", {})
        capacity_workloads = manifest.get("capacity_workloads", {})
        comparison_cases = expected_cases if candidate_case is not None else [reference_case]
        checks["attention_finalist_protocol"] = {
            "applicable": True,
            "passed": (
                cases == expected_cases
                and len(normalized) == len(expected_cases)
                and all(item == normalized[0] for item in normalized)
                and all(CASES[case].max_model_len == CAPACITY_FINAL_TOKENS for case in comparison_cases)
                and all(phase.get("trajectory_65k") is True for phase in planned_finalist)
                and all(phase.get("capacity_stress_131k") is True for phase in planned_finalist)
                and all(phase.get("r3_enabled") is False for phase in planned_finalist)
                and (candidate_case is None or changed_properties == {expected_property})
                and len(
                    {
                        health_workloads.get(f"{case}:completion", {}).get("frozen_inputs_sha256")
                        for case in comparison_cases
                    }
                )
                == 1
                and len({trajectory_workloads.get(case, {}).get("frozen_inputs_sha256") for case in comparison_cases})
                == 1
                and len({capacity_workloads.get(case, {}).get("frozen_inputs_sha256") for case in comparison_cases}) == 1
            ),
            "cases": cases,
            "candidate_case": candidate_case,
            "changed_properties": sorted(changed_properties),
            "expected_property": expected_property,
        }
    else:
        checks["attention_finalist_protocol"] = {"applicable": False, "passed": True}
    recomputed = _matrix_result(
        plan=plan,
        run_id=run_id,
        phase_results=result.get("phases", []),
        placement=result.get("placement", {}),
        all_rank_health=result.get("all_rank_health", {}),
        elapsed_seconds=float(result.get("elapsed_seconds", 0)),
        error=result.get("error"),
    )
    checks["decision_recomputation"] = {
        "passed": (
            result.get("analysis") == recomputed["analysis"]
            and result.get("status") == recomputed["status"]
            and result.get("passed") is recomputed["passed"]
            and result.get("arms") == recomputed["arms"]
            and manifest.get("phase_result_sha256") == _sha256_json(result.get("phases", []))
            and manifest.get("result_aggregate_sha256") == _sha256_json(result.get("arms", []))
        ),
        "benchmark_passed": recomputed["passed"],
        "analysis": recomputed["analysis"],
    }
    phase_plan_matches = [phase.get("phase") for phase in result.get("phases", [])] == manifest.get("phase_plan")
    expected_phases = {
        "instrument-v1": 5,
        "ep8-calibration": 2,
        "topology-v1": 10 + (2 if result.get("analysis", {}).get("ep8_is_targeted_chat_r3_finalist") else 0),
        "attention-pair-v1": 2,
        "attention-finalist-v1": len(manifest.get("phase_plan", [])),
    }[plan]
    phase_job_ids = {
        rank_record.get("job_id")
        for phase in result.get("phases", [])
        for rank_record in phase.get("all_rank_health", {}).get("ranks", [])
    }
    provenance = manifest.get("provenance", {})
    placement = manifest.get("placement", {})
    planned_phases = manifest.get("phase_plan", [])
    phase_results = result.get("phases", [])
    if plan == "topology-v1":
        try:
            stored_sources = manifest["calibration_sources"]
            settings_by_case = {
                case_name: {
                    (
                        int(phase["concurrencies"][0]),
                        int(phase["max_num_batched_tokens"]),
                        int(phase["max_num_seqs"]),
                    )
                    for phase in planned_phases
                    if phase.get("role") == "topology-comparison" and phase.get("case") == case_name
                }
                for case_name in ("reference-ep8", "exact-reference-ep16")
            }
            if any(len(settings) != 1 for settings in settings_by_case.values()):
                raise ValueError("topology phase settings are not unique per case")
            ep8_settings = next(iter(settings_by_case["reference-ep8"]))
            ep16_settings = next(iter(settings_by_case["exact-reference-ep16"]))
            if ep8_settings[2] != ep16_settings[2]:
                raise ValueError("topology cases use different max-num-seqs")
            verified_sources = _verified_topology_calibration_sources(
                filesystem,
                ep8_run_id=str(stored_sources["ep8"]["run_id"]),
                ep16_run_id=str(stored_sources["ep16"]["run_id"]),
                ep8_concurrency=ep8_settings[0],
                ep8_max_num_batched_tokens=ep8_settings[1],
                ep16_concurrency=ep16_settings[0],
                ep16_max_num_batched_tokens=ep16_settings[1],
                max_num_seqs=ep8_settings[2],
                marin_commit=str(provenance["marin_commit"]),
                task_image=str(provenance["task_image"]),
            )
            checks["calibration_sources"] = {
                "applicable": True,
                "passed": stored_sources == verified_sources,
                "sources": verified_sources,
            }
        except Exception as exc:
            checks["calibration_sources"] = {
                "applicable": True,
                "passed": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    elif plan in {"attention-pair-v1", "attention-finalist-v1"}:
        try:
            stored_sources = manifest["calibration_sources"]
            reference_settings = {
                (
                    int(phase["concurrencies"][0]),
                    int(phase["max_num_batched_tokens"]),
                    int(phase["max_num_seqs"]),
                )
                for phase in planned_phases
                if phase.get("role") in {"attention-comparison", "attention-finalist-validation"}
                and phase.get("case") in {"exact-reference-ep16", "exact-reference-131k-ep16"}
            }
            if len(reference_settings) != 1:
                raise ValueError("attention reference settings are not unique")
            concurrency, max_num_batched_tokens, max_num_seqs = next(iter(reference_settings))
            verified_sources = {
                "ep16": _verified_calibration_source(
                    filesystem,
                    plan="instrument-v1",
                    run_id=str(stored_sources["ep16"]["run_id"]),
                    expected_case="exact-reference-ep16",
                    expected_concurrency=concurrency,
                    expected_max_num_batched_tokens=max_num_batched_tokens,
                    expected_max_num_seqs=max_num_seqs,
                    marin_commit=None,
                    task_image=str(provenance["task_image"]),
                )
            }
            checks["calibration_sources"] = {
                "applicable": True,
                "passed": stored_sources == verified_sources,
                "sources": verified_sources,
            }
        except Exception as exc:
            checks["calibration_sources"] = {
                "applicable": True,
                "passed": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
    else:
        checks["calibration_sources"] = {
            "applicable": False,
            "passed": manifest.get("calibration_sources") == {},
        }
    phase_contracts = (
        [
            _matrix_phase_evidence_contract(phase_result, planned_phase, provenance)
            for phase_result, planned_phase in zip(phase_results, planned_phases, strict=True)
        ]
        if len(phase_results) == len(planned_phases)
        else []
    )
    checks["phase_runtime_contracts"] = {
        "passed": len(phase_contracts) == expected_phases and all(item["passed"] for item in phase_contracts),
        "phases": phase_contracts,
    }
    if plan == "instrument-v1":
        chat_pair = _matrix_matched_chat_pair_contract(
            phase_results,
            off_phase_id="ep16-chat-r3off",
            on_phase_id="ep16-chat-r3on",
        )
        chat_pair["applicable"] = True
    elif plan == "topology-v1" and result.get("analysis", {}).get("ep8_is_targeted_chat_r3_finalist"):
        chat_pair = _matrix_matched_chat_pair_contract(
            phase_results,
            off_phase_id="targeted-ep8-chat-r3off",
            on_phase_id="targeted-ep8-chat-r3on",
        )
        chat_pair["applicable"] = True
    else:
        chat_pair = {"applicable": False, "passed": True}
    checks["matched_chat_r3_only_route_return_differs"] = chat_pair
    inventory = [
        gpu
        for rank_record in result.get("all_rank_health", {}).get("ranks", [])
        for gpu in rank_record.get("gpu_inventory", [])
    ]
    checks["same_allocation_and_provenance"] = {
        "passed": (
            phase_plan_matches
            and len(result.get("phases", [])) == expected_phases
            and all(phase.get("passed") is True for phase in result.get("phases", []))
            and phase_job_ids == {provenance.get("iris_job_id")}
            and placement.get("passed") is True
            and len(placement.get("distinct_advertise_hosts", [])) == int(provenance.get("iris_task_count", 0))
            and provenance.get("iris_coscheduling") == UNATTENDED_COSCHEDULING
            and provenance.get("iris_priority") in {priority.value for priority in Priority}
            and provenance.get("iris_retry_policy")
            == {"max_retries_failure": 0, "max_retries_preemption": 0, "max_task_failures": 0}
            and provenance.get("task_image") == _immutable_image(provenance["task_image"])
            and provenance.get("vllm_commit") == VLLM_SHA
            and provenance.get("vllm_commit_url", "").endswith(VLLM_SHA)
            and provenance.get("marin_commit_url", "").endswith(provenance.get("marin_commit", "missing"))
            and len(inventory) == int(provenance.get("iris_task_count", 0)) * LOCAL_DP_SIZE
            and len({gpu["uuid"] for gpu in inventory}) == len(inventory)
            and all("GB200" in gpu["name"] for gpu in inventory)
        ),
        "phase_plan_matches": phase_plan_matches,
        "phase_job_ids": sorted(str(value) for value in phase_job_ids),
        "gpu_count": len(inventory),
    }
    completed_phases = [record for record in event_records if record.get("event") == "matrix_phase_completed"]
    checks["events_and_markdown"] = {
        "passed": (
            len(completed_phases) == expected_phases
            and all(record.get("passed") is True for record in completed_phases)
            and any(record.get("event") == "matrix_worker_completed" for record in event_records)
            and f"# GrugMoE {plan}" in result_md_bytes.decode()
            and "Status: **PASS**" in result_md_bytes.decode()
            and run_id in result_md_bytes.decode()
        ),
        "events": len(event_records),
        "completed_phases": len(completed_phases),
    }
    found = {
        path.removeprefix(f"{root_key}/")
        for path in filesystem.find(root_key)
        if path != root_key and not path.endswith("/")
    }
    expected_files = {
        "manifest.json",
        "events.jsonl",
        "metrics/map.json",
        "result.json",
        "result.md",
        *[entry["path"] for entry in snapshots],
        *[str(arm["kv_cache"]["source"]["path"]) for arm in result.get("arms", [])],
        *[
            str(point["kv_cache"]["source"]["path"])
            for arm in result.get("arms", [])
            for point in arm.get("coarse_curve", [])
        ],
        *[
            str(arm[field]["kv_cache"]["source"]["path"])
            for arm in result.get("arms", [])
            for field in ("trajectory_65k", "capacity_stress_131k")
            if field in arm
        ],
    }
    checks["exact_layout"] = {
        "passed": found == expected_files,
        "missing": sorted(expected_files - found),
        "unexpected": sorted(found - expected_files),
    }
    passed = all(check.get("passed") is True for check in checks.values())
    return {
        "schema_version": 1,
        "kind": "independent-matrix-artifact-readback",
        "plan": plan,
        "run_id": run_id,
        "artifact_prefix": artifact_prefix,
        "read_at": datetime.now(UTC).isoformat(),
        "passed": passed,
        "benchmark_passed": recomputed["passed"],
        "checks": checks,
        "source_object_sha256": {
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            "result.json": hashlib.sha256(result_bytes).hexdigest(),
            "events.jsonl": hashlib.sha256(events_bytes).hexdigest(),
            "metrics/map.json": hashlib.sha256(metrics_map_bytes).hexdigest(),
            "result.md": hashlib.sha256(result_md_bytes).hexdigest(),
        },
    }


def run_matrix_readback_worker(args: argparse.Namespace) -> dict[str, Any]:
    info = get_job_info()
    if info is None or info.num_tasks != 1 or info.task_index != 0:
        raise RuntimeError("matrix readback must run as one independent Iris CPU task")
    filesystem = _s3_filesystem()
    receipt = readback_matrix_artifacts(filesystem, plan=args.plan, run_id=args.run_id)
    receipt.update(
        {
            "iris_job_id": str(info.job_id),
            "iris_task_id": str(info.task_id),
            "task_image": args.task_image,
            "reader_marin_commit": args.marin_commit,
        }
    )
    _put_json_readback(
        filesystem,
        f"{_matrix_control_prefix(args.plan, args.run_id)}independent-readback.json",
        receipt,
    )
    if not receipt["passed"]:
        raise RuntimeError(json.dumps(receipt["checks"]))
    return receipt


def run_health_readback_worker(args: argparse.Namespace) -> dict[str, Any]:
    info = get_job_info()
    if info is None or info.num_tasks != 1 or info.task_index != 0:
        raise RuntimeError("health readback must run as one independent Iris CPU task")
    filesystem = _s3_filesystem()
    receipt = readback_health_artifacts(filesystem, run_id=args.run_id)
    receipt.update(
        {
            "iris_job_id": str(info.job_id),
            "iris_task_id": str(info.task_id),
            "task_image": args.task_image,
            "reader_marin_commit": args.marin_commit,
        }
    )
    _put_json_readback(
        filesystem,
        f"{ARTIFACT_ROOT}/health-control/{args.run_id}/independent-readback.json",
        receipt,
    )
    if not receipt["passed"]:
        raise RuntimeError(json.dumps(receipt["checks"]))
    return receipt


def submit_health_readback(args: argparse.Namespace) -> dict[str, Any]:
    checkout = _clean_pushed_checkout()
    image = _immutable_image(args.task_image)
    command = [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "reader",
        "--run-id",
        args.run_id,
        "--task-image",
        image,
        "--marin-commit",
        checkout["commit"],
    ]
    with controller_client(args.config) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=f"grugmoe-health-readback-{args.run_id}".lower().replace("_", "-"),
            resources=ResourceSpec(cpu=4, memory="16GB", disk="10GB"),
            environment=EnvironmentSpec(sync_packages=["marin-iris", "marin-core"], env_vars={"PYTHONUNBUFFERED": "1"}),
            replicas=1,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            task_image=image,
            priority_band=PRIORITY_BANDS[Priority.INTERACTIVE],
        )
        summary: dict[str, Any] = {
            "status": "submitted",
            "job_id": str(job.job_id),
            "run_id": args.run_id,
            "task_image": image,
            "checkout": checkout,
            "artifact_prefix": f"{HEALTH_ARTIFACT_ROOT}/{args.run_id}/",
            "receipt": f"{ARTIFACT_ROOT}/health-control/{args.run_id}/independent-readback.json",
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.wait:
            status = job.wait(
                timeout=args.wait_timeout,
                poll_interval=10,
                raise_on_failure=False,
                stream_logs=True,
            )
            summary["terminal_job_state"] = int(status.state)
            summary["terminal_job_state_name"] = job_pb2.JobState.Name(status.state)
            summary["terminal_job_succeeded"] = status.state == job_pb2.JOB_STATE_SUCCEEDED
            summary["terminal_error"] = status.error
            summary["status"] = summary["terminal_job_state_name"].removeprefix("JOB_STATE_").lower()
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def _matrix_worker_argv(
    args: argparse.Namespace,
    *,
    run_id: str,
    image: str,
    marin_commit: str,
) -> list[str]:
    command = [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "matrix-worker",
        "--plan",
        args.plan,
        "--run-id",
        run_id,
        "--task-image",
        image,
        "--marin-commit",
        marin_commit,
        "--iris-priority",
        args.priority,
        "--submitted-coscheduling",
        UNATTENDED_COSCHEDULING,
        "--server-timeout",
        str(args.server_timeout),
        "--minimum-seconds",
        str(args.minimum_seconds),
        "--minimum-generated-tokens",
        str(args.minimum_generated_tokens),
        "--ep8-concurrency",
        str(args.ep8_concurrency),
        "--ep8-max-num-batched-tokens",
        str(args.ep8_max_num_batched_tokens),
        "--ep16-concurrency",
        str(args.ep16_concurrency),
        "--ep16-max-num-batched-tokens",
        str(args.ep16_max_num_batched_tokens),
        "--topology-max-num-seqs",
        str(args.topology_max_num_seqs),
    ]
    if args.plan == "topology-v1":
        command.extend(
            [
                "--ep8-calibration-run-id",
                args.ep8_calibration_run_id,
                "--ep16-instrument-run-id",
                args.ep16_instrument_run_id,
            ]
        )
    elif args.plan == "attention-pair-v1":
        command.extend(
            [
                "--attention-candidate",
                args.attention_candidate,
                "--attention-order",
                args.attention_order,
                "--ep16-instrument-run-id",
                args.ep16_instrument_run_id,
            ]
        )
    elif args.plan == "attention-finalist-v1":
        command.extend(
            [
                "--attention-finalist",
                args.attention_finalist,
                "--ep16-instrument-run-id",
                args.ep16_instrument_run_id,
            ]
        )
    return command


def _validate_matrix_args(args: argparse.Namespace) -> None:
    validate_health_thresholds(
        minimum_seconds=args.minimum_seconds,
        minimum_generated_tokens=args.minimum_generated_tokens,
    )
    if args.plan not in MATRIX_PLANS:
        raise ValueError(f"unknown matrix plan: {args.plan}")
    for topology in ("ep8", "ep16"):
        concurrency = int(getattr(args, f"{topology}_concurrency"))
        max_num_batched_tokens = int(getattr(args, f"{topology}_max_num_batched_tokens"))
        if concurrency <= 0 or concurrency % 3:
            raise ValueError(f"{topology} concurrency must be positive and divisible by three")
        if max_num_batched_tokens <= 0:
            raise ValueError(f"{topology} max-num-batched-tokens must be positive")
    if args.topology_max_num_seqs < max(args.ep8_concurrency, args.ep16_concurrency):
        raise ValueError("topology max-num-seqs must cover both selected concurrencies")
    source_run_ids = (args.ep8_calibration_run_id, args.ep16_instrument_run_id)
    attention_fields = (args.attention_candidate, args.attention_order)
    if args.plan == "topology-v1" and not all(source_run_ids):
        raise ValueError("topology requires the EP8 calibration and EP16 instrument run IDs")
    if args.plan == "topology-v1" and (any(attention_fields) or args.attention_finalist is not None):
        raise ValueError("attention settings are only valid for attention plans")
    if args.plan == "attention-pair-v1":
        if args.ep8_calibration_run_id is not None or args.ep16_instrument_run_id is None:
            raise ValueError("attention pairs require only the EP16 instrument run ID")
        if args.attention_candidate not in ATTENTION_CANDIDATES or args.attention_order not in ATTENTION_ORDERS:
            raise ValueError("attention pairs require a frozen candidate and order")
        if args.run_id is not None and args.attention_candidate not in args.run_id:
            raise ValueError("attention pair run ID must include its candidate case")
        if args.attention_finalist is not None:
            raise ValueError("attention finalist is only valid for finalist validation")
    elif args.plan == "attention-finalist-v1":
        if args.ep8_calibration_run_id is not None or args.ep16_instrument_run_id is None:
            raise ValueError("attention finalist validation requires only the EP16 instrument run ID")
        if any(attention_fields) or args.attention_finalist not in ATTENTION_FINALISTS:
            raise ValueError("attention finalist validation requires one frozen finalist")
        if args.run_id is not None and args.attention_finalist not in args.run_id:
            raise ValueError("attention finalist run ID must include its finalist case")
    elif args.plan != "topology-v1" and any(source_run_ids):
        raise ValueError("calibration source run IDs are only valid for topology or attention pairs")
    if args.plan != "attention-pair-v1" and any(attention_fields):
        raise ValueError("attention candidate and order are only valid for attention pairs")
    if args.plan != "attention-finalist-v1" and args.attention_finalist is not None:
        raise ValueError("attention finalist is only valid for finalist validation")


def submit_matrix(args: argparse.Namespace) -> dict[str, Any]:
    _validate_matrix_args(args)
    checkout = _clean_pushed_checkout()
    image = _immutable_image(args.task_image)
    priority = Priority(args.priority)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or (
        f"exp{'3' if args.attention_candidate == 'global-every4-ep16' else '4'}-"
        f"{args.attention_candidate}-{args.attention_order}-{timestamp}"
        if args.plan == "attention-pair-v1"
        else (
            f"exp{'3' if args.attention_finalist == 'global-every4-ep16' else '4'}-"
            f"{args.attention_finalist}-validation-{timestamp}"
            if args.plan == "attention-finalist-v1"
            else timestamp
        )
    )
    replicas = 2 if args.plan == "ep8-calibration" else 4
    coscheduling = CoschedulingConfig(group_by=UNATTENDED_COSCHEDULING)
    worker_argv = _matrix_worker_argv(
        args,
        run_id=run_id,
        image=image,
        marin_commit=checkout["commit"],
    )
    resources = ResourceSpec(
        cpu=64,
        memory="512GB",
        disk="100GB",
        device=gpu_device("GB200", LOCAL_DP_SIZE),
    )
    with controller_client(args.config) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*worker_argv),
            name=f"grugmoe-{args.plan}-{run_id}".lower().replace("_", "-"),
            resources=resources,
            environment=EnvironmentSpec(
                sync_packages=["marin-iris", "marin-core"],
                env_vars={"PYTHONUNBUFFERED": "1"},
            ),
            replicas=replicas,
            coscheduling=coscheduling,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            task_image=image,
            priority_band=PRIORITY_BANDS[priority],
        )
        summary: dict[str, Any] = {
            "status": "submitted",
            "job_id": str(job.job_id),
            "run_id": run_id,
            "plan": args.plan,
            "replicas": replicas,
            "coscheduling": coscheduling.group_by,
            "task_image": image,
            "priority": priority.value,
            "checkout": checkout,
            "artifact_prefix": _matrix_artifact_prefix(args.plan, run_id),
            "receipt": f"{_matrix_control_prefix(args.plan, run_id)}independent-readback.json",
            "retry_policy": {
                "max_retries_failure": 0,
                "max_retries_preemption": 0,
                "max_task_failures": 0,
            },
        }
        if args.plan == "topology-v1":
            summary["calibration_source_run_ids"] = {
                "ep8": args.ep8_calibration_run_id,
                "ep16": args.ep16_instrument_run_id,
            }
        elif args.plan == "attention-pair-v1":
            summary["attention_pair"] = {
                "candidate": args.attention_candidate,
                "order": args.attention_order,
                "ep16_instrument_run_id": args.ep16_instrument_run_id,
            }
        elif args.plan == "attention-finalist-v1":
            summary["attention_finalist"] = {
                "finalist": args.attention_finalist,
                "ep16_instrument_run_id": args.ep16_instrument_run_id,
                "trajectory_65k": True,
                "capacity_stress_131k": True,
            }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.wait:
            status = job.wait(
                timeout=args.wait_timeout,
                poll_interval=10,
                raise_on_failure=False,
                stream_logs=True,
            )
            summary["terminal_job_state"] = int(status.state)
            summary["terminal_job_state_name"] = job_pb2.JobState.Name(status.state)
            summary["terminal_job_succeeded"] = status.state == job_pb2.JOB_STATE_SUCCEEDED
            summary["terminal_error"] = status.error
            summary["status"] = summary["terminal_job_state_name"].removeprefix("JOB_STATE_").lower()
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def submit_matrix_readback(args: argparse.Namespace) -> dict[str, Any]:
    checkout = _clean_pushed_checkout()
    image = _immutable_image(args.task_image)
    command = [
        "python",
        "-m",
        "scripts.iris.grugmoe_inference_preflight",
        "matrix-reader",
        "--plan",
        args.plan,
        "--run-id",
        args.run_id,
        "--task-image",
        image,
        "--marin-commit",
        checkout["commit"],
    ]
    with controller_client(args.config) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=f"grugmoe-{args.plan}-readback-{args.run_id}".lower().replace("_", "-"),
            resources=ResourceSpec(cpu=4, memory="16GB", disk="10GB"),
            environment=EnvironmentSpec(sync_packages=["marin-iris", "marin-core"], env_vars={"PYTHONUNBUFFERED": "1"}),
            replicas=1,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            task_image=image,
            priority_band=PRIORITY_BANDS[Priority.INTERACTIVE],
        )
        summary: dict[str, Any] = {
            "status": "submitted",
            "job_id": str(job.job_id),
            "run_id": args.run_id,
            "plan": args.plan,
            "task_image": image,
            "checkout": checkout,
            "artifact_prefix": _matrix_artifact_prefix(args.plan, args.run_id),
            "receipt": f"{_matrix_control_prefix(args.plan, args.run_id)}independent-readback.json",
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if args.wait:
            status = job.wait(
                timeout=args.wait_timeout,
                poll_interval=10,
                raise_on_failure=False,
                stream_logs=True,
            )
            summary["terminal_job_state"] = int(status.state)
            summary["terminal_job_state_name"] = job_pb2.JobState.Name(status.state)
            summary["terminal_job_succeeded"] = status.state == job_pb2.JOB_STATE_SUCCEEDED
            summary["terminal_error"] = status.error
            summary["status"] = summary["terminal_job_state_name"].removeprefix("JOB_STATE_").lower()
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write frozen config, workload, and manifest locally")
    prepare.add_argument("--case", choices=sorted(CASES), required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run against an existing dev_gpu allocation")
    run.add_argument("--session", required=True)
    run.add_argument("--case", choices=sorted(CASES), required=True)
    run.add_argument("--model-source", choices=("dummy", "fixture"), default="dummy")
    run.add_argument("--run-id")
    run.add_argument("--mode", choices=("smoke", "acceptance"), default="smoke")
    run.add_argument("--state-dir", default=str(STATE_DIR))
    run.add_argument("--local-output")
    run.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
    run.add_argument("--minimum-seconds", type=float, default=ACCEPTANCE_MINIMUM_SECONDS)
    run.add_argument(
        "--minimum-generated-tokens",
        type=int,
        default=ACCEPTANCE_MINIMUM_GENERATED_TOKENS,
    )
    run.add_argument("--no-upload", action="store_true")

    submit = subparsers.add_parser(
        "submit",
        help="submit a zero-retry unattended Iris gang",
    )
    submit.add_argument("--case", choices=sorted(CASES), required=True)
    submit.add_argument("--model-source", choices=("dummy", "fixture"), default="dummy")
    submit.add_argument(
        "--mode",
        choices=("smoke", "kv", "acceptance", "health"),
        default="smoke",
    )
    submit.add_argument("--run-id")
    submit.add_argument("--task-image", required=True)
    submit.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    submit.add_argument(
        "--priority", choices=[priority.value for priority in Priority], default=Priority.INTERACTIVE.value
    )
    submit.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
    submit.add_argument("--minimum-seconds", type=float, default=ACCEPTANCE_MINIMUM_SECONDS)
    submit.add_argument(
        "--minimum-generated-tokens",
        type=int,
        default=ACCEPTANCE_MINIMUM_GENERATED_TOKENS,
    )
    submit.add_argument("--r3", choices=("on", "off"), default="on")
    submit.add_argument("--request-transport", choices=("completion", "chat"), default="completion")
    submit.add_argument("--max-num-batched-tokens", type=int, default=8192)
    submit.add_argument("--max-num-seqs", type=int)
    submit.add_argument("--concurrency", type=int, action="append")
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--wait-timeout", type=float, default=21_600)

    worker = subparsers.add_parser(
        "worker",
        help="internal Iris replica entrypoint",
    )
    worker.add_argument("--case", choices=sorted(CASES), required=True)
    worker.add_argument("--model-source", choices=("dummy", "fixture"), required=True)
    worker.add_argument(
        "--mode",
        choices=("smoke", "kv", "acceptance", "health"),
        required=True,
    )
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--task-image", required=True)
    worker.add_argument("--marin-commit", required=True)
    worker.add_argument("--iris-priority", choices=[priority.value for priority in Priority], required=True)
    worker.add_argument("--submitted-coscheduling")
    worker.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
    worker.add_argument("--minimum-seconds", type=float, default=ACCEPTANCE_MINIMUM_SECONDS)
    worker.add_argument(
        "--minimum-generated-tokens",
        type=int,
        default=ACCEPTANCE_MINIMUM_GENERATED_TOKENS,
    )
    worker.add_argument("--r3", choices=("on", "off"), default="on")
    worker.add_argument("--request-transport", choices=("completion", "chat"), default="completion")
    worker.add_argument("--max-num-batched-tokens", type=int, default=8192)
    worker.add_argument("--max-num-seqs", type=int)
    worker.add_argument("--concurrency", type=int, action="append")

    submit_readback = subparsers.add_parser(
        "submit-readback",
        help="submit an independent zero-retry CPU reader for one health run",
    )
    submit_readback.add_argument("--run-id", required=True)
    submit_readback.add_argument("--task-image", required=True)
    submit_readback.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    submit_readback.add_argument("--wait", action="store_true")
    submit_readback.add_argument("--wait-timeout", type=float, default=3_600)

    reader = subparsers.add_parser("reader", help="internal independent health artifact reader")
    reader.add_argument("--run-id", required=True)
    reader.add_argument("--task-image", required=True)
    reader.add_argument("--marin-commit", required=True)

    def add_matrix_settings(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--plan", choices=MATRIX_PLANS, required=True)
        command_parser.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
        command_parser.add_argument("--minimum-seconds", type=float, default=HEALTH_MINIMUM_SECONDS)
        command_parser.add_argument(
            "--minimum-generated-tokens",
            type=int,
            default=HEALTH_MINIMUM_GENERATED_TOKENS,
        )
        command_parser.add_argument("--ep8-concurrency", type=int, default=144)
        command_parser.add_argument("--ep8-max-num-batched-tokens", type=int, default=8192)
        command_parser.add_argument("--ep16-concurrency", type=int, default=144)
        command_parser.add_argument("--ep16-max-num-batched-tokens", type=int, default=8192)
        command_parser.add_argument("--topology-max-num-seqs", type=int, default=CALIBRATION_MAX_NUM_SEQS)
        command_parser.add_argument("--ep8-calibration-run-id")
        command_parser.add_argument("--ep16-instrument-run-id")
        command_parser.add_argument("--attention-candidate", choices=ATTENTION_CANDIDATES)
        command_parser.add_argument("--attention-order", choices=ATTENTION_ORDERS)
        command_parser.add_argument("--attention-finalist", choices=ATTENTION_FINALISTS)

    submit_matrix_parser = subparsers.add_parser(
        "submit-matrix",
        help="submit a bounded zero-retry instrument, calibration, or topology matrix",
    )
    add_matrix_settings(submit_matrix_parser)
    submit_matrix_parser.add_argument("--run-id")
    submit_matrix_parser.add_argument("--task-image", required=True)
    submit_matrix_parser.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    submit_matrix_parser.add_argument(
        "--priority", choices=[priority.value for priority in Priority], default=Priority.INTERACTIVE.value
    )
    submit_matrix_parser.add_argument("--wait", action="store_true")
    submit_matrix_parser.add_argument("--wait-timeout", type=float, default=21_600)

    matrix_worker = subparsers.add_parser("matrix-worker", help="internal Iris matrix replica entrypoint")
    add_matrix_settings(matrix_worker)
    matrix_worker.add_argument("--run-id", required=True)
    matrix_worker.add_argument("--task-image", required=True)
    matrix_worker.add_argument("--marin-commit", required=True)
    matrix_worker.add_argument("--iris-priority", choices=[priority.value for priority in Priority], required=True)
    matrix_worker.add_argument("--submitted-coscheduling", required=True)

    submit_matrix_reader = subparsers.add_parser(
        "submit-matrix-readback",
        help="submit an independent zero-retry reader for one matrix run",
    )
    submit_matrix_reader.add_argument("--plan", choices=MATRIX_PLANS, required=True)
    submit_matrix_reader.add_argument("--run-id", required=True)
    submit_matrix_reader.add_argument("--task-image", required=True)
    submit_matrix_reader.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    submit_matrix_reader.add_argument("--wait", action="store_true")
    submit_matrix_reader.add_argument("--wait-timeout", type=float, default=3_600)

    matrix_reader = subparsers.add_parser("matrix-reader", help="internal independent matrix artifact reader")
    matrix_reader.add_argument("--plan", choices=MATRIX_PLANS, required=True)
    matrix_reader.add_argument("--run-id", required=True)
    matrix_reader.add_argument("--task-image", required=True)
    matrix_reader.add_argument("--marin-commit", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        write_case(args.output, case=CASES[args.case], run_id=args.run_id, git_sha=_git_sha())
        print(json.dumps({"status": "prepared", "case": args.case, "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "submit":
        summary = submit_unattended(args)
        if not args.wait:
            return 0
        return 0 if summary.get("terminal_job_succeeded") is True else 1
    if args.command == "submit-matrix":
        summary = submit_matrix(args)
        if not args.wait:
            return 0
        return 0 if summary.get("terminal_job_succeeded") is True else 1
    if args.command == "submit-matrix-readback":
        summary = submit_matrix_readback(args)
        if not args.wait:
            return 0
        return 0 if summary.get("terminal_job_succeeded") is True else 1
    if args.command == "submit-readback":
        summary = submit_health_readback(args)
        if not args.wait:
            return 0
        return 0 if summary.get("terminal_job_succeeded") is True else 1
    if args.command == "reader":
        receipt = run_health_readback_worker(args)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "matrix-reader":
        receipt = run_matrix_readback_worker(args)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "matrix-worker":
        result = run_matrix_worker(args)
        print(
            json.dumps(
                {
                    "status": result.get("status", "completed"),
                    "plan": args.plan,
                    "run_id": args.run_id,
                    "rank": get_job_info().task_index,
                    "artifact_prefix": _matrix_artifact_prefix(args.plan, args.run_id),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.command == "worker":
        result = run_unattended_worker(args)
        print(
            json.dumps(
                {
                    "status": result.get("status", "completed"),
                    "case": result["case"] if "case" in result else args.case,
                    "run_id": args.run_id,
                    "rank": get_job_info().task_index,
                    "artifact_prefix": (
                        f"{HEALTH_ARTIFACT_ROOT}/{args.run_id}/"
                        if args.mode == "health"
                        else f"{ARTIFACT_ROOT}/{args.case}/{args.run_id}/"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    result = execute(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "case": result["case"],
                "run_id": result["run_id"],
                "local_output": result["local_output"],
                "artifact_prefix": (result.get("upload") or {}).get("prefix"),
                "upload_passed": (result.get("upload") or {}).get("passed"),
                "error": result.get("error"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
