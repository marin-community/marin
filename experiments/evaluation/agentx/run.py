# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exactly one complete AgentX window in one Iris attempt.

Iris owns placement and retries. Each attempt has a new server, cache and output
directory. AgentX alone owns replay, warmup, the measurement clock and validity.
"""

import contextlib
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.resources.state import TaskState
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import InferenceModelConfig, VllmCompilationCacheMode
from marin.inference.iris_vllm import (
    IrisVllmLaunch,
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.recipe import (
    PointManifest,
    benchmark_command,
    load_series,
    series_path,
    workload_environment,
)
from experiments.evaluation.agentx.workload import check_native_cache, restore_workload

SERVER_ENVIRONMENT_KEYS = (
    "MODEL_IMPL_TYPE",
    "TOKENIZERS_PARALLELISM",
    "VLLM_ENGINE_READY_TIMEOUT_S",
    "VLLM_CACHE_ROOT",
    "VLLM_COMPILE_CACHE_SAVE_FORMAT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "NCCL_RAS_ENABLE",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
)


def interrupt(signum: int, _frame: object) -> None:
    raise InterruptedError(f"Attempt interrupted by signal {signum}")


def gang_attempts(hosts: int) -> dict[str, str]:
    """Fence a measurement to one live set of native Iris attempts."""
    client = iris_ctx().client
    info = get_job_info()
    assert client is not None and info is not None
    tasks = [client.task_status(task.task_id) for task in client.list_tasks(info.job_id)]
    if len(tasks) != hosts or any(task.state != TaskState.RUNNING for task in tasks):
        raise RuntimeError("All serving hosts must be running for the complete profiling window")
    return {
        str(task.task_id): next(
            attempt.attempt_uid for attempt in task.attempts if attempt.attempt_number == task.current_attempt_number
        )
        for task in tasks
    }


def checkpoint_config(series: dict, output: Path) -> Path:
    """Verify the export manifest and HF config before the native streaming loader starts."""
    checkpoint = series["checkpoint"]
    filesystem, prefix = filesystem_for(checkpoint["uri"])
    for name, expected in (
        ("export-manifest.json", series["model_revision"]),
        ("config.json", checkpoint["config_sha256"]),
    ):
        with filesystem.open(prefix_join(prefix, name), "rb") as stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Hero export changed: {name}")
        (output / name).write_bytes(content)
    return output / "config.json"


def create_server(manifest: PointManifest, output: Path, launch: IrisVllmLaunch | None = None) -> VllmEnvironment:
    """Use the same pinned native serving recipe for startup checks and measurements."""
    series = load_series(series_path(manifest["profile"]))
    profile = series["profiles"][manifest["profile"]]
    if profile["hosts"] > 1:
        if launch is None or os.environ["IRIS_NODE_NAME"] not in manifest["placement"]["nodes"]:
            raise RuntimeError("Hero requires a multi-host launch on the verified spare nodes")
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK),
        {"VLLM_ENGINE_READY_TIMEOUT_S": str(series["ready_timeout"]), "TOKENIZERS_PARALLELISM": "false"}
        | (launch.subprocess_env if launch is not None else {}),
    )
    port = 38197
    server_args = [] if "checkpoint" in series else ["--revision", series["model_revision"]]
    server_args += [
        "--tokenizer",
        series["tokenizer"],
        "--tokenizer-revision",
        series["tokenizer_revision"],
        "--served-model-name",
        series["model"],
        "--dtype",
        {"bf16": "bfloat16"}[series["precision"]],
        "--tensor-parallel-size",
        str(series["tensor_parallel"]),
        "--data-parallel-size",
        str(profile["gpus"]),
        "--enable-expert-parallel",
        "--max-model-len",
        str(series["max_context"]),
        "--max-num-seqs",
        str(series["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(series["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(series["gpu_memory_utilization"]),
        "--enable-prompt-tokens-details",
        "--reasoning-parser",
        "grug",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    server_args.append("--enable-prefix-caching" if series["enable_prefix_caching"] else "--no-enable-prefix-caching")
    if launch is not None:
        # Iris owns all physical hosts; the shared adapter supplies native global/local DP ranks.
        start = server_args.index("--tensor-parallel-size")
        del server_args[start : start + 4]
        server_args.extend(launch.extra_cli_args)
    if "checkpoint" in series:
        server_args.extend(
            [
                "--seed",
                str(series["server_seed"]),
                "--hf-config-path",
                str(checkpoint_config(series, output).parent),
                "--hf-overrides",
                json.dumps(series["hf_overrides"]),
                "--load-format",
                "runai_streamer",
                "--model-loader-extra-config",
                json.dumps({"distributed": True, "concurrency": 8, "memory_limit": 2 * 1024**3}),
                "--kernel-config",
                json.dumps({"moe_backend": "triton"}),
            ]
        )
    return VllmEnvironment(
        InferenceModelConfig(
            name=series["model"],
            path=series["checkpoint"]["uri"] if "checkpoint" in series else None,
            engine_kwargs={},
        ),
        port=port,
        timeout_seconds=series["ready_timeout"],
        extra_args=server_args,
        launcher=launcher,
        compilation_cache_mode=VllmCompilationCacheMode.MANAGED,
        collect_metrics=series["collect_marin_metrics"],
        collect_nccl=series["collect_nccl"],
        wait_for_ready=False,
    )


def run_point(manifest: PointManifest, output: Path, bundle: Path, launch: IrisVllmLaunch | None = None) -> None:
    path = series_path(manifest["profile"])
    series = load_series(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["series_sha256"]:
        raise RuntimeError("Series changed since submission")
    release = dataclasses.asdict(VLLM_GPU_RELEASE)
    if json.dumps(release, sort_keys=True) != json.dumps(manifest["vllm_release"], sort_keys=True):
        raise RuntimeError("Marin's adopted vLLM release changed since submission")
    profile = series["profiles"][manifest["profile"]]
    devices = profile["hosts"] * profile["gpus"]
    if profile["hosts"] > 1 and os.environ["IRIS_NODE_NAME"] not in manifest["placement"]["nodes"]:
        raise RuntimeError("Iris placed a Hero host outside the verified spare-node list")
    output.mkdir(parents=True, exist_ok=False)
    record = manifest | {
        "architecture": platform.machine(),
        "status": "preparing",
        "server_parallelism": {"tp": series["tensor_parallel"], "dp": devices, "ep": devices, "pp": 1},
        "task_index": launch.task_index if launch is not None else 0,
        "hostname": platform.node(),
        "iris_node_name": os.environ.get("IRIS_NODE_NAME"),
        "gpu_uuids": (
            subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
        ),
        "iris_task_id": os.environ["IRIS_TASK_ID"],
        "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "iris_bundle_id": os.environ["IRIS_BUNDLE_ID"],
        "started_unix": time.time(),
    }
    record_path = output / "attempt.json"
    record_path.write_text(json.dumps(record, indent=2))
    print(
        f"AGENTX_ATTEMPT {record['iris_task_id']}:{record['iris_attempt_uid']} "
        f"profile={manifest['profile']} concurrency={manifest['concurrency']} "
        f"workload={manifest['workload']['key']}",
        flush=True,
    )
    with (output / "hardware.txt").open("w") as log:
        for command in (
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"],
            ["nvidia-smi", "topo", "-m"],
            ["lscpu"],
        ):
            subprocess.run(command, stdout=log, check=True)
        log.write(Path("/proc/meminfo").read_text())
    print("AGENTX_WORKLOAD_RESTORING", flush=True)
    restore_workload(manifest["workload"], bundle)
    check_native_cache(bundle, path, output / "workload-check.log")
    environment = workload_environment(bundle, series)
    record["workload_restored_unix"] = time.time()
    server = create_server(manifest, output, launch)
    url = f"http://{server.host}:{server.port}"
    benchmark = benchmark_command(series, manifest["concurrency"], url, output / "aiperf_artifacts")
    leader = launch is None or launch.is_leader
    record.update({"benchmark_command": benchmark if leader else None, "status": "starting"})
    record_path.write_text(json.dumps(record, indent=2))
    native_log_dir = None
    try:
        with server:
            handle = server.vllm_server
            assert handle is not None
            native_log_dir = Path(handle.log_dir)
            effective_environment = handle.compilation_cache.environment()
            record.update(
                {
                    "server_command": server.command,
                    "server_environment": {
                        key: effective_environment[key]
                        for key in SERVER_ENVIRONMENT_KEYS
                        if key in effective_environment
                    },
                    "compilation_cache": VllmCompilationCacheMode.MANAGED.value,
                }
            )
            record_path.write_text(json.dumps(record, indent=2))
            if not leader:
                assert launch is not None
                wait_for_iris_vllm_shutdown(launch, server.check_alive)
                server.publish_compilation_cache()
                record["status"] = "succeeded"
                return
            server.wait_until_ready()
            if server.model_id != series["model"]:
                raise RuntimeError("vLLM served model differs from the pinned recipe")
            handle = server.vllm_server
            assert handle is not None
            record["collectors"] = {
                "marin_metrics": handle.metrics_collector is not None,
                "nccl": handle.nccl_probe is not None,
                "agentx_measurements": True,
                "agentx_server_metrics": True,
            }
            record.update({"status": "measuring", "ready_unix": time.time()})
            if launch is not None:
                record["gang_attempts"] = gang_attempts(profile["hosts"])
            record_path.write_text(json.dumps(record, indent=2))
            print("AGENTX_SERVER_READY", flush=True)
            with iris_vllm_followers(launch) if launch is not None else contextlib.nullcontext():
                run_logged(
                    benchmark,
                    environment,
                    output / "agentx.log",
                    timeout=series["warmup_grace_period"] + series["duration"] + 3600,
                )
                exported = json.loads((output / "aiperf_artifacts" / "profile_export_aiperf.json").read_text())
                if not exported.get("metadata", {}).get("submission_valid") or exported.get("was_cancelled"):
                    raise RuntimeError("AgentX rejected the completed window; retain it as invalid diagnostics")
                server.check_alive()
                if launch is not None and gang_attempts(profile["hosts"]) != record["gang_attempts"]:
                    raise RuntimeError("A serving host changed attempts during the profiling window")
                record["status"] = "succeeded"
    except BaseException as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        record["finished_unix"] = time.time()
        record_path.write_text(json.dumps(record, indent=2))
        if native_log_dir is not None:
            shutil.copytree(native_log_dir, output / "server-logs")
        print(f"AGENTX_ATTEMPT_STATUS {record['status']}", flush=True)
        if launch is not None and not leader and record["status"] == "succeeded":
            notify_iris_vllm_stopped(launch)


def run(manifest: PointManifest) -> None:
    """Run one native Iris attempt, recording interruption and preparation failures."""
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "point"
    try:
        with tempfile.TemporaryDirectory(prefix="agentx-workload-") as directory:
            series = load_series(series_path(manifest["profile"]))
            profile = series["profiles"][manifest["profile"]]
            if profile["hosts"] == 1:
                run_point(manifest, output, Path(directory) / "bundle")
            else:
                with iris_vllm_launch(
                    pipeline_parallel_size=1,
                    tensor_parallel_size=series["tensor_parallel"],
                    data_parallel_size=profile["hosts"] * profile["gpus"],
                    data_parallel_size_local=profile["gpus"],
                ) as launch:
                    run_point(manifest, output, Path(directory) / "bundle", launch)
    except BaseException as error:
        record_path = output / "attempt.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}", "finished_unix": time.time()}
            )
            record_path.write_text(json.dumps(record, indent=2))
        raise
