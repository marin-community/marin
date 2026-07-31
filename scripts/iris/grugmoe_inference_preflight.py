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
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import requests
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device

from experiments.grug.moe.inference_preflight import (
    ARTIFACT_ROOT,
    CASES,
    DUMMY_SEED,
    FROZEN_FIXTURE_PATH,
    SNOWBALL_EXPORT,
    VLLM_SHA,
    ModelCase,
    aggregate_preflight_status,
    decode_routed_experts,
    deterministic_balanced_routing_fixture,
    expert_parallel_rank_histogram,
    frozen_manifest,
    layer_types,
    materialize_prompt,
    metric_delta,
    parse_prometheus,
    predict_kv_bytes,
    routing_histogram,
    write_case,
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
SERVER_TIMEOUT_SECONDS = 3600
REQUEST_TIMEOUT_SECONDS = 3600
ACCEPTANCE_MINIMUM_SECONDS = 600
ACCEPTANCE_MINIMUM_GENERATED_TOKENS = 250_000
ACCEPTANCE_STABLE_MINUTES = 10
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


def vllm_args(
    case: ModelCase,
    *,
    model_dir: str,
    model_source: str,
    leader_ip: str,
    node_index: int,
    smoke: bool,
) -> list[str]:
    if not 0 <= node_index < case.node_count:
        raise ValueError(f"node_index {node_index} is outside case node count {case.node_count}")
    fixture = model_source == "fixture"
    args = [
        "serve",
        model_dir,
        "--trust-remote-code",
        "--dtype",
        "float" if fixture else "bfloat16",
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
        "--enable-return-routed-experts",
        "--enable-prompt-tokens-details",
        "--max-logprobs",
        "64",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        str(min(case.max_model_len, 2048) if smoke else case.max_model_len),
        "--max-num-batched-tokens",
        str(2048 if smoke else 8192),
        "--max-num-seqs",
        str(8 if smoke else 64),
    ]
    if model_source == "dummy":
        args.extend(["--load-format", "dummy", "--skip-tokenizer-init"])
    elif model_source == "fixture":
        args.extend(["--load-format", "safetensors", "--skip-tokenizer-init"])
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


def _completion(base_url: str, model: str, prompt_token_ids: list[int], *, max_tokens: int = 4) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt_token_ids,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "logprobs": 1,
            "return_token_ids": True,
            # Dummy configs intentionally omit a tokenizer. Ask the OpenAI
            # compatibility layer to label logprob entries by token ID.
            "return_tokens_as_token_ids": True,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"completion failed with {response.status_code}: {response.text[:4000]}")
    payload = response.json()
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AssertionError(f"expected one completion choice, got {payload!r}")
    return payload


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


def boundary_requests(workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose one request at each required cache boundary."""
    selected: list[dict[str, Any]] = []
    for boundary in (17, 513):
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
    num_experts: int,
    top_k: int,
    ep_size: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    candidates = boundary_requests(workload)
    before_text, before = _metrics(base_url)
    before_path = _record_metrics(before_text, artifact_dir=artifact_dir, stem="metrics-before")
    boundary_results: list[dict[str, Any]] = []
    histogram = [0] * num_experts
    for request in candidates:
        prompt_token_ids = materialize_prompt(workload, request)
        mutated_prompt_token_ids = materialize_prompt(workload, request, mutated=True)
        cold = _completion(base_url, model, prompt_token_ids)
        after_cold_text, after_cold = _metrics(base_url)
        reused = _completion(base_url, model, prompt_token_ids)
        after_reuse_text, after_reuse = _metrics(base_url)
        mutated = _completion(base_url, model, mutated_prompt_token_ids)
        after_mutated_text, after_mutated = _metrics(base_url)
        _assert_same_reuse(cold, reused)
        reuse_hits = metric_delta(after_cold, after_reuse, "vllm:prefix_cache_hits")
        mutation_hits = metric_delta(after_reuse, after_mutated, "vllm:prefix_cache_hits")
        reuse_cached_tokens = _cached_prompt_tokens(reused)
        mutated_cached_tokens = _cached_prompt_tokens(mutated)
        if reuse_hits <= 0:
            raise AssertionError(f"identical request did not produce prefix cache hits: {reuse_hits}")
        if mutation_hits != 0:
            raise AssertionError(f"mutated request unexpectedly reused prefix cache blocks: {mutation_hits}")
        if reuse_cached_tokens <= 0:
            raise AssertionError(f"identical request reported no cached prompt tokens: {reuse_cached_tokens}")
        if mutated_cached_tokens != 0:
            raise AssertionError(f"mutated request reported cached prompt tokens: {mutated_cached_tokens}")
        routes = decode_routed_experts(_choice(reused)["routed_experts"])
        route_histogram = routing_histogram(routes, num_experts=num_experts)
        histogram = [left + right for left, right in zip(histogram, route_histogram, strict=True)]
        request_id = request["request_id"]
        boundary_results.append(
            {
                "request_id": request_id,
                "prefix_token_count": request["prefix_token_count"],
                "cold": _record_completion(cold, artifact_dir=artifact_dir, stem=f"{request_id}-cold"),
                "reused": _record_completion(reused, artifact_dir=artifact_dir, stem=f"{request_id}-reused"),
                "mutated": _record_completion(mutated, artifact_dir=artifact_dir, stem=f"{request_id}-mutated"),
                "reuse_prefix_hits": reuse_hits,
                "mutated_prefix_hits": mutation_hits,
                "reuse_cached_prompt_tokens": reuse_cached_tokens,
                "mutated_cached_prompt_tokens": mutated_cached_tokens,
                "route_histogram": route_histogram,
                "metrics": {
                    "after_cold": _record_metrics(
                        after_cold_text,
                        artifact_dir=artifact_dir,
                        stem=f"{request_id}-metrics-after-cold",
                    ),
                    "after_reuse": _record_metrics(
                        after_reuse_text,
                        artifact_dir=artifact_dir,
                        stem=f"{request_id}-metrics-after-reuse",
                    ),
                    "after_mutated": _record_metrics(
                        after_mutated_text,
                        artifact_dir=artifact_dir,
                        stem=f"{request_id}-metrics-after-mutated",
                    ),
                },
            }
        )
    after_text, after = _metrics(base_url)
    after_path = _record_metrics(after_text, artifact_dir=artifact_dir, stem="metrics-after")
    ep_rank_histogram = expert_parallel_rank_histogram(histogram, ep_size=ep_size)
    mean_rank_assignments = sum(ep_rank_histogram) / len(ep_rank_histogram)
    imbalance_triggered = any(count == 0 for count in histogram) or any(
        count > 2 * mean_rank_assignments for count in ep_rank_histogram
    )
    return {
        "passed": True,
        "boundaries": boundary_results,
        "route_histogram": histogram,
        "ep_rank_histogram": ep_rank_histogram,
        "routing_balance": {
            "unused_experts": sum(count == 0 for count in histogram),
            "mean_ep_rank_assignments": mean_rank_assignments,
            "max_ep_rank_assignments": max(ep_rank_histogram),
            "balanced_control_triggered": imbalance_triggered,
            "balanced_control": (
                deterministic_balanced_routing_fixture(num_experts=num_experts, top_k=top_k, ep_size=ep_size)
                if imbalance_triggered
                else None
            ),
        },
        "metrics_before": before_path,
        "metrics_after": after_path,
        "metric_deltas": {
            metric: metric_delta(before, after, metric)
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
) -> dict[str, Any]:
    started = time.monotonic()
    generated = 0
    request_count = 0
    latencies: list[float] = []
    completions: list[tuple[float, int, int, str]] = []
    workload_requests = workload["requests"]
    covered_request_ids: set[str] = set()

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
        completions.append((time.monotonic() - started, completion_tokens, prefix_tokens, request_id))

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # The first wave is a literal traversal of all 144 frozen branches.
        coverage_futures = [executor.submit(one, request) for request in workload_requests]
        for future in concurrent.futures.as_completed(coverage_futures):
            record(future)

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
            for future in concurrent.futures.as_completed(futures):
                record(future)

    elapsed = time.monotonic() - started
    complete_minutes = int(elapsed // 60)
    minute_tokens: list[int] = [0] * (complete_minutes + 1)
    cohort_tokens: dict[str, int] = {}
    for completed_at, completion_tokens, prefix_tokens, _ in completions:
        minute_tokens[min(int(completed_at // 60), len(minute_tokens) - 1)] += completion_tokens
        cohort = str(prefix_tokens)
        cohort_tokens[cohort] = cohort_tokens.get(cohort, 0) + completion_tokens
    stable_minute_tokens = minute_tokens[max(0, complete_minutes - ACCEPTANCE_STABLE_MINUTES) : complete_minutes]
    stable_minutes_passed = len(stable_minute_tokens) == ACCEPTANCE_STABLE_MINUTES and all(
        tokens > 0 for tokens in stable_minute_tokens
    )
    stable_mean = sum(stable_minute_tokens) / (60 * len(stable_minute_tokens)) if stable_minute_tokens else 0.0
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
        "full_minute_generated_tokens": minute_tokens[:complete_minutes],
        "last_ten_stable_minute_generated_tokens": stable_minute_tokens,
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
        "--no-project",
        "--with",
        VLLM_FROM_SPEC,
        "--with",
        RUNAI_STREAMER,
        "python",
        "tests/cluster/vllm/grug_exact_reference_check.py",
        "--fixture",
        str(FIXTURE_DIR),
        "--output",
        str(tensor_path),
    ]
    environment = dict(os.environ)
    environment["UV_TORCH_BACKEND"] = "cu130"
    completed = _run(
        command,
        capture_output=True,
        env=environment,
        timeout=SERVER_TIMEOUT_SECONDS,
    )
    tensor_payload = json.loads(tensor_path.read_text())
    if completed.stdout:
        (artifact_dir / "fixture-tensor-parity.stdout").write_text(completed.stdout)

    # This helper has no vLLM import; it scores the live response against the
    # same frozen Levanter observations used by the tensor check.
    from tests.cluster.vllm.grug_exact_reference_check import run_server_parity  # noqa: PLC0415

    server = run_server_parity(base_url, model, FIXTURE_DIR)
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
        "active_requests_by_engine": active_requests_by_engine,
        "semantic_active_bytes": semantic_bytes,
        "semantic_attention_active_bytes": sum(int(group["active_payload_bytes"]) for group in attention_groups),
        "semantic_sconv_active_bytes": sum(int(group["active_payload_bytes"]) for group in sconv_groups),
        "padded_group_active_bytes": padded_group_bytes,
        "padded_attention_active_bytes": sum(int(group["active_padded_bytes"]) for group in attention_groups),
        "padded_sconv_active_bytes": sum(int(group["active_padded_bytes"]) for group in sconv_groups),
        "physical_active_bytes": active_physical_bytes,
        "padding_active_bytes": padded_group_bytes - semantic_bytes,
        "reserved_physical_bytes_per_engine": reserved_by_engine,
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
            future = executor.submit(
                _completion,
                base_url,
                model,
                prompt,
                max_tokens=response_tokens,
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
) -> list[str]:
    return [
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
        "--server-timeout",
        str(args.server_timeout),
        "--minimum-seconds",
        str(args.minimum_seconds),
        "--minimum-generated-tokens",
        str(args.minimum_generated_tokens),
    ]


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
    checkout = _clean_pushed_checkout()
    image = _immutable_image(args.task_image)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    worker_argv = _unattended_worker_argv(
        args,
        case=case,
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
            name=f"grugmoe-{case.name}-{run_id}".lower().replace("_", "-"),
            resources=resources,
            environment=EnvironmentSpec(
                sync_packages=["marin-iris", "marin-core"],
                env_vars={"PYTHONUNBUFFERED": "1"},
            ),
            replicas=case.node_count,
            coscheduling=(CoschedulingConfig(group_by=UNATTENDED_COSCHEDULING) if case.node_count > 1 else None),
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            task_image=image,
            priority_band=PRIORITY_BANDS[Priority.INTERACTIVE],
        )
        summary: dict[str, Any] = {
            "status": "submitted",
            "job_id": str(job.job_id),
            "run_id": run_id,
            "case": case.name,
            "mode": args.mode,
            "replicas": case.node_count,
            "coscheduling": UNATTENDED_COSCHEDULING if case.node_count > 1 else None,
            "task_image": image,
            "checkout": checkout,
            "artifact_prefix": f"{ARTIFACT_ROOT}/{case.name}/{run_id}/",
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
            summary["terminal_error"] = status.error
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


def _start_local_vllm(
    *,
    case: ModelCase,
    model_source: str,
    model_dir: str,
    leader_ip: str,
    node_index: int,
    smoke: bool,
    local_dir: Path,
) -> LocalVllm:
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
        )
    )
    log_path = local_dir / f"vllm-node-{node_index}.log"
    log_stream = log_path.open("w")
    environment = {
        **os.environ,
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


def run_unattended_worker(args: argparse.Namespace) -> dict[str, Any]:
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
        "worker_id": str(info.worker_id),
        "advertise_host": info.advertise_host,
    }
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
                        num_experts=case.num_experts,
                        top_k=case.num_experts_per_tok,
                        ep_size=case.data_parallel_size,
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
            "coscheduling": UNATTENDED_COSCHEDULING if info.num_tasks > 1 else None,
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
    placement = {
        "passed": (
            len(rendezvous) == info.num_tasks
            and {int(endpoint["task_index"]) for endpoint in rendezvous} == set(range(info.num_tasks))
            and len({endpoint["worker_id"] for endpoint in rendezvous}) == info.num_tasks
            and (
                info.num_tasks == 1
                or all(record.get("coscheduling") == UNATTENDED_COSCHEDULING for record in rank_records)
            )
        ),
        "required_coscheduling": UNATTENDED_COSCHEDULING if info.num_tasks > 1 else None,
        "endpoints": rendezvous,
        "distinct_workers": sorted({endpoint["worker_id"] for endpoint in rendezvous}),
    }
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
                "coscheduling": UNATTENDED_COSCHEDULING if info.num_tasks > 1 else None,
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
                    num_experts=case.num_experts,
                    top_k=case.num_experts_per_tok,
                    ep_size=case.data_parallel_size,
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
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
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
                    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                    raise
                result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


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
        choices=("smoke", "kv", "acceptance"),
        default="smoke",
    )
    submit.add_argument("--run-id")
    submit.add_argument("--task-image", required=True)
    submit.add_argument("--config", default=DEFAULT_CLUSTER_CONFIG)
    submit.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
    submit.add_argument("--minimum-seconds", type=float, default=ACCEPTANCE_MINIMUM_SECONDS)
    submit.add_argument(
        "--minimum-generated-tokens",
        type=int,
        default=ACCEPTANCE_MINIMUM_GENERATED_TOKENS,
    )
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
        choices=("smoke", "kv", "acceptance"),
        required=True,
    )
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--task-image", required=True)
    worker.add_argument("--marin-commit", required=True)
    worker.add_argument("--server-timeout", type=float, default=SERVER_TIMEOUT_SECONDS)
    worker.add_argument("--minimum-seconds", type=float, default=ACCEPTANCE_MINIMUM_SECONDS)
    worker.add_argument(
        "--minimum-generated-tokens",
        type=int,
        default=ACCEPTANCE_MINIMUM_GENERATED_TOKENS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        write_case(args.output, case=CASES[args.case], run_id=args.run_id, git_sha=_git_sha())
        print(json.dumps({"status": "prepared", "case": args.case, "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "submit":
        submit_unattended(args)
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
                    "artifact_prefix": f"{ARTIFACT_ROOT}/{args.case}/{args.run_id}/",
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
