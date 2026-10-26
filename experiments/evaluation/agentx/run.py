# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exactly one complete AgentX window in one Iris attempt.

Iris owns placement and retries. Each attempt has a new server, cache and output
directory. AgentX alone owns replay, warmup, the measurement clock and validity.
"""

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

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import InferenceModelConfig, VllmCompilationCacheMode
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType

from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.recipe import SERIES_PATH, benchmark_command, load_series, workload_environment
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


def run_point(manifest: dict, output: Path, bundle: Path) -> None:
    series = load_series()
    if hashlib.sha256(SERIES_PATH.read_bytes()).hexdigest() != manifest["series_sha256"]:
        raise RuntimeError("Series changed since submission")
    release = dataclasses.asdict(VLLM_GPU_RELEASE)
    if json.dumps(release, sort_keys=True) != json.dumps(manifest["vllm_release"], sort_keys=True):
        raise RuntimeError("Marin's adopted vLLM release changed since submission")
    profile = series["profiles"][manifest["profile"]]
    output.mkdir(parents=True, exist_ok=False)
    record = manifest | {
        "architecture": platform.machine(),
        "status": "preparing",
        "server_parallelism": {"tp": series["tensor_parallel"], "dp": profile["gpus"], "ep": profile["gpus"], "pp": 1},
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
    check_native_cache(bundle, SERIES_PATH, output / "workload-check.log")
    environment = workload_environment(bundle, series)
    record["workload_restored_unix"] = time.time()
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK),
        {"VLLM_ENGINE_READY_TIMEOUT_S": str(series["ready_timeout"]), "TOKENIZERS_PARALLELISM": "false"},
    )
    port = 38197
    url = f"http://127.0.0.1:{port}"
    server_args = [
        "--revision",
        series["model_revision"],
        "--tokenizer",
        series["model"],
        "--tokenizer-revision",
        series["model_revision"],
        "--served-model-name",
        series["model"],
        "--dtype",
        {"bf16": "bfloat16"}[series["precision"]],
        "--tensor-parallel-size",
        str(series["tensor_parallel"]),
        "--data-parallel-size",
        str(profile["gpus"]),
        "--enable-expert-parallel",
        "--enable-prefix-caching",
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
    benchmark = benchmark_command(series, manifest["concurrency"], url, output / "aiperf_artifacts")
    record.update({"benchmark_command": benchmark, "status": "starting"})
    record_path.write_text(json.dumps(record, indent=2))
    server = VllmEnvironment(
        InferenceModelConfig(name=series["model"], path=None, engine_kwargs={}),
        port=port,
        timeout_seconds=series["ready_timeout"],
        extra_args=server_args,
        launcher=launcher,
        compilation_cache_mode=VllmCompilationCacheMode.MANAGED,
        collect_metrics=series["collect_marin_metrics"],
        collect_nccl=series["collect_nccl"],
        wait_for_ready=False,
    )
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
            record_path.write_text(json.dumps(record, indent=2))
            print("AGENTX_SERVER_READY", flush=True)
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


def run(manifest: dict) -> None:
    """Run one native Iris attempt, recording interruption and preparation failures."""
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "point"
    try:
        with tempfile.TemporaryDirectory(prefix="agentx-workload-") as directory:
            run_point(manifest, output, Path(directory) / "bundle")
    except BaseException as error:
        record_path = output / "attempt.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}", "finished_unix": time.time()}
            )
            record_path.write_text(json.dumps(record, indent=2))
        raise
