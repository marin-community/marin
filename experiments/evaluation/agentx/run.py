# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run exactly one complete AgentX window in one Iris attempt.

Iris owns placement and retries. Each attempt has a new server, cache and output
directory. AgentX alone owns replay, warmup, the measurement clock and validity.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType

SERIES_PATH = Path(__file__).with_name("series.json")
REQUIREMENTS_PATH = Path(__file__).with_name("requirements-harness.txt")


def logged_process(command: list[str], environment: dict[str, str], log: Path) -> subprocess.Popen:
    """Keep file artifacts and native Iris logs, including interrupted attempts."""
    return subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", '"$@" 2>&1 | tee "$AGENTX_LOG_FILE"', "agentx-log", *command],
        env=environment | {"AGENTX_LOG_FILE": str(log)},
        start_new_session=True,
    )


def stop_group(process: subprocess.Popen) -> None:
    """Reap the command and stop its descendants before releasing the allocation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def wait_ready(process: subprocess.Popen, url: str, model: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as response:
                models = json.load(response)
            if any(item["id"] == model for item in models["data"]):
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError("vLLM readiness deadline exceeded")


def interrupt(signum: int, _frame: object) -> None:
    raise InterruptedError(f"Attempt interrupted by signal {signum}")


def harness_command(series: dict) -> list[str]:
    return [
        "uvx",
        "--python",
        "3.12",
        "--from",
        f"agentx-harness=={series['harness_version']}",
        "--with-requirements",
        str(REQUIREMENTS_PATH),
    ]


def benchmark_command(series: dict, concurrency: int, url: str, output: Path) -> list[str]:
    return [
        *harness_command(series),
        "agentx",
        "profile",
        "--url",
        url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        series["model"],
        "--tokenizer",
        series["model"],
        "--tokenizer-revision",
        series["model_revision"],
        "--public-dataset",
        series["dataset_loader"],
        "--max-context-length",
        str(series["max_context"]),
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(series["duration"]),
        "--failed-request-threshold",
        str(series["max_profiling_error_ratio"]),
        "--random-seed",
        str(series["random_seed"]),
        "--trajectory-start-min-ratio",
        str(series["trajectory_start_min_ratio"]),
        "--trajectory-start-max-ratio",
        str(series["trajectory_start_max_ratio"]),
        "--warmup-requests-per-lane",
        str(series["warmup_requests_per_lane"]),
        "--agentic-warmup-grace-period",
        str(series["warmup_grace_period"]),
        "--trace-idle-gap-cap-seconds",
        str(series["trace_idle_gap_cap_seconds"]),
        "--system-idle-gap-cap-seconds",
        str(series["system_idle_gap_cap_seconds"]),
        "--cache-bust",
        "first_turn_prefix",
        "--use-server-token-count",
        "--server-metrics",
        f"{url}/metrics",
        "--output-artifact-dir",
        str(output),
        "--ui",
        "none",
    ]


def run(manifest: dict, output: Path) -> None:
    series = json.loads(SERIES_PATH.read_text())
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
        "iris_task_id": os.environ["IRIS_TASK_ID"],
        "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "iris_bundle_id": os.environ["IRIS_BUNDLE_ID"],
        "started_unix": time.time(),
    }
    record_path = output / "attempt.json"
    record_path.write_text(json.dumps(record, indent=2))
    print("AGENTX_ATTEMPT " + json.dumps(record, sort_keys=True), flush=True)
    environment = os.environ | {"AIPERF_DATASET_WEKA_PARALLEL_WORKERS": "2", "TOKENIZERS_PARALLELISM": "false"}
    with (output / "hardware.txt").open("w") as log:
        for command in (
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"],
            ["nvidia-smi", "topo", "-m"],
            ["lscpu"],
        ):
            subprocess.run(command, stdout=log, check=True)
        log.write(Path("/proc/meminfo").read_text())
    prepare = [
        *harness_command(series),
        "python",
        "-m",
        "experiments.evaluation.agentx.prepare",
        "--series",
        str(SERIES_PATH),
        "--output",
        str(output / "corpus.json"),
    ]
    record["prepare_command"] = prepare
    with logged_process(prepare, environment, output / "prepare.log") as process:
        try:
            if process.wait(timeout=series["prepare_timeout"]):
                raise RuntimeError("AgentX preparation failed; see prepare.log")
        finally:
            stop_group(process)
    launcher = IsolatedCudaVllm(source=VllmType.MARIN_FORK)
    port = 38197
    url = f"http://127.0.0.1:{port}"
    server = [
        *launcher.command(),
        "serve",
        series["model"],
        "--revision",
        series["model_revision"],
        "--tokenizer",
        series["model"],
        "--tokenizer-revision",
        series["model_revision"],
        "--served-model-name",
        series["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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
    record.update({"server_command": server, "benchmark_command": benchmark, "status": "starting"})
    record_path.write_text(json.dumps(record, indent=2))
    server_environment = environment | launcher.env() | {"VLLM_ENGINE_READY_TIMEOUT_S": str(series["ready_timeout"])}
    with logged_process(server, server_environment, output / "server.log") as server_process:
        try:
            wait_ready(server_process, url, series["model"], series["ready_timeout"])
            record.update({"status": "measuring", "ready_unix": time.time()})
            record_path.write_text(json.dumps(record, indent=2))
            print("AGENTX_SERVER_READY", flush=True)
            with logged_process(
                benchmark,
                environment | {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                output / "agentx.log",
            ) as client:
                try:
                    if client.wait(timeout=series["duration"] + 3600):
                        raise RuntimeError("AgentX failed; see agentx.log")
                finally:
                    stop_group(client)
            exported = json.loads((output / "aiperf_artifacts" / "profile_export_aiperf.json").read_text())
            if not exported.get("metadata", {}).get("submission_valid") or exported.get("was_cancelled"):
                raise RuntimeError("AgentX rejected the completed window; retain it as invalid diagnostics")
            if server_process.poll() is not None:
                raise RuntimeError("Server exited during the measurement")
            record["status"] = "succeeded"
        except BaseException as error:
            record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            stop_group(server_process)
            record["finished_unix"] = time.time()
            record_path.write_text(json.dumps(record, indent=2))
            print(f"AGENTX_ATTEMPT_STATUS {record['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    # Iris creates IRIS_OUTPUT_DIR itself. Each attempt writes a distinct child.
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "point"
    try:
        run(json.loads(args.manifest.read_text()), output)
    except BaseException as error:
        record_path = output / "attempt.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            record.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}", "finished_unix": time.time()}
            )
            record_path.write_text(json.dumps(record, indent=2))
        raise


if __name__ == "__main__":
    main()
