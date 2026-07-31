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
import shlex
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from experiments.grug.moe.inference_preflight import (
    ARTIFACT_ROOT,
    CASES,
    DUMMY_SEED,
    SNOWBALL_EXPORT,
    VLLM_SHA,
    ModelCase,
    decode_routed_experts,
    deterministic_balanced_routing_fixture,
    expert_parallel_rank_histogram,
    frozen_manifest,
    materialize_prompt,
    metric_delta,
    parse_prometheus,
    routing_histogram,
    write_case,
)
from scripts.iris.dev_gpu import (
    STATE_DIR,
    DevGpuState,
    PodRef,
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
REQUEST_TIMEOUT_SECONDS = 600
ACCEPTANCE_MINIMUM_SECONDS = 600
ACCEPTANCE_MINIMUM_GENERATED_TOKENS = 250_000
REMOTE_ROOT = "/tmp/grugmoe-inference-preflight"
LOG_TAIL_LINES = 400
GLOO_CONTROL_INTERFACE = "enP6p3s0np0"
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
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=capture_output,
        check=check,
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
    args = [
        "serve",
        model_dir,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
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
            model_dir = self.remote_dir if self.model_source == "dummy" else SNOWBALL_EXPORT
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
            for filename in ("aws-config", "config.json", "workload.json", "manifest.json"):
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
    request_index = 0
    latencies: list[float] = []
    completions: list[tuple[float, int, int]] = []
    workload_requests = workload["requests"]

    def one(request: dict[str, Any]) -> tuple[float, dict[str, Any], int]:
        prompt = materialize_prompt(workload, request)
        max_tokens = min(256, max_model_len - len(prompt))
        if max_tokens <= 0:
            raise AssertionError(f"{request['request_id']} leaves no generation capacity")
        request_started = time.monotonic()
        payload = _completion(base_url, model, prompt, max_tokens=max_tokens)
        return time.monotonic() - request_started, payload, int(request["prefix_token_count"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        while time.monotonic() - started < minimum_seconds or generated < minimum_generated_tokens:
            batch = [
                workload_requests[(request_index + offset) % len(workload_requests)] for offset in range(concurrency)
            ]
            request_index += len(batch)
            futures = [executor.submit(one, request) for request in batch]
            for future in concurrent.futures.as_completed(futures):
                latency, payload, prefix_tokens = future.result()
                completion_tokens = int(payload.get("usage", {}).get("completion_tokens", 0))
                if completion_tokens <= 0:
                    raise AssertionError("load request generated no tokens")
                generated += completion_tokens
                request_count += 1
                latencies.append(latency)
                completions.append((time.monotonic() - started, completion_tokens, prefix_tokens))

    elapsed = time.monotonic() - started
    minute_tokens: list[int] = [0] * (int(elapsed // 60) + 1)
    cohort_tokens: dict[str, int] = {}
    for completed_at, completion_tokens, prefix_tokens in completions:
        minute_tokens[min(int(completed_at // 60), len(minute_tokens) - 1)] += completion_tokens
        cohort = str(prefix_tokens)
        cohort_tokens[cohort] = cohort_tokens.get(cohort, 0) + completion_tokens
    final_five_start = max(0.0, elapsed - 300.0)
    final_five_tokens = sum(tokens for completed_at, tokens, _ in completions if completed_at >= final_five_start)
    final_five_seconds = elapsed - final_five_start
    return {
        "passed": elapsed >= minimum_seconds and generated >= minimum_generated_tokens,
        "elapsed_seconds": elapsed,
        "stable_full_minutes": int(elapsed // 60),
        "generated_tokens": generated,
        "requests": request_count,
        "concurrency": concurrency,
        "throughput_tokens_per_second": {
            "full_mean": generated / elapsed,
            "final_five_minute_mean": final_five_tokens / final_five_seconds,
        },
        "full_minute_generated_tokens": minute_tokens[: int(elapsed // 60)],
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

    arm_throughputs = [arm["throughput_tokens_per_second"]["full_mean"] for arm in arms]
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


def _git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.strip()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        filename: _sha256_path(local_dir / filename) for filename in ("aws-config", "config.json", "workload.json")
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
            result["correctness"] = run_correctness(
                base_url,
                model,
                workload,
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
    run.add_argument("--model-source", choices=("dummy", "snowball"), default="dummy")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        write_case(args.output, case=CASES[args.case], run_id=args.run_id, git_sha=_git_sha())
        print(json.dumps({"status": "prepared", "case": args.case, "output": str(args.output)}, sort_keys=True))
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
