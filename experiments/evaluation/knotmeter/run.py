# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one fresh serving gang, native AgentX replay, checks and output capture."""

import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.resources.state import TaskState
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.inference.backend import OPENAI_API_SUFFIX
from marin.inference.config import InferenceModelConfig, VllmCompilationCacheMode, VllmRouterConfig
from marin.inference.iris_vllm import (
    IrisVllmLaunch,
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType
from marin.testing.inference.goldens import canonical_bytes
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.artifacts import (
    PROFILE_EXPORT_RELATIVE_PATH,
    copy_file,
    pack_server_metrics,
    retain_artifacts,
    write_record,
)
from experiments.evaluation.knotmeter.config import Profile
from experiments.evaluation.knotmeter.processes import run_logged
from experiments.evaluation.knotmeter.recipe import (
    ATTEMPT_RECEIPT_NAME,
    SERVER_PORT,
    PointManifest,
    TrialPlan,
    benchmark_command,
    workload_environment,
)
from experiments.evaluation.knotmeter.runtime import materialize_release
from experiments.evaluation.knotmeter.workload import check_native_cache, restore_workload

logger = logging.getLogger(__name__)


def capture_attempt(plan: TrialPlan, output: Path, record: dict, task_index: int, *, phase: str) -> None:
    """Publish a checksum-addressed snapshot before changing the receipt."""
    snapshot = record | {"capture_phase": phase}
    write_record(str(output / "attempt.json"), snapshot)
    prefix = prefix_join(plan["trial_uri"], f"attempts/{record['iris_attempt_uid']}/hosts/{task_index}")
    # Keep this intermediate archive outside IRIS_OUTPUT_DIR; Iris also captures
    # that directory and must not include a second copy of the same output.
    with tempfile.TemporaryDirectory(prefix="knotmeter-capture-") as directory:
        uri, digest = retain_artifacts(output, Path(directory) / "host.tar.zst", prefix, name="native")
    receipt_uri = prefix_join(prefix, ATTEMPT_RECEIPT_NAME)
    write_record(
        receipt_uri,
        snapshot | {"archive_uri": uri, "archive_sha256": digest, "captured_unix": time.time()},
    )
    print(f"KNOTMETER_ATTEMPT {receipt_uri}", flush=True)


def interrupt(signum: int, _frame: object) -> None:
    raise InterruptedError(f"Attempt interrupted by signal {signum}")


def gang_attempts(hosts: int) -> dict[str, str]:
    """Snapshot the running native attempt on each host for continuity checks."""
    client = iris_ctx().client
    info = get_job_info()
    assert client is not None and info is not None
    tasks = [client.task_status(task.task_id) for task in client.list_tasks(info.job_id)]
    if len(tasks) != hosts or any(task.state != TaskState.RUNNING for task in tasks):
        raise RuntimeError("Every serving host must be running throughout measurement")
    return {
        str(task.task_id): next(
            attempt.attempt_uid for attempt in task.attempts if attempt.attempt_number == task.current_attempt_number
        )
        for task in tasks
    }


def create_server(manifest: PointManifest, output: Path, launch: IrisVllmLaunch | None) -> VllmEnvironment:
    profile = Profile.model_validate(manifest["series"]["profile"])
    model, serving, hardware = profile.model, profile.serving, profile.gpus
    args = [
        "--tokenizer",
        model.tokenizer,
        "--tokenizer-revision",
        model.tokenizer_revision,
        "--served-model-name",
        model.repo,
        "--dtype",
        serving.dtype,
        "--max-model-len",
        str(serving.max_model_len),
        "--max-num-seqs",
        str(serving.max_num_seqs),
        "--max-num-batched-tokens",
        str(serving.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(serving.gpu_memory_utilization),
        "--enable-prefix-caching" if serving.prefix_caching else "--no-enable-prefix-caching",
        "--enable-expert-parallel",
        *serving.extra_args,
    ]
    if profile.quantization:
        args += ["--quantization", profile.quantization]
    if model.chat_template == "marin":
        template = output / "chat-template.jinja"
        template.write_text(MARIN_CHAT_TEMPLATE)
        args += ["--chat-template", str(template)]
    if model.eos_token_id is not None:
        args += ["--override-generation-config", json.dumps({"eos_token_id": model.eos_token_id})]
    if model.path:
        for name, expected in (("export-manifest.json", model.revision), ("config.json", model.config_sha256)):
            local = output / name
            copy_file(prefix_join(model.path, name), str(local))
            if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checkpoint identity differs: {name}")
        args += ["--hf-config-path", str(output)]
    else:
        args += ["--revision", model.revision]
    if launch is None:
        args += [
            "--tensor-parallel-size",
            str(serving.tensor_parallel_size),
            "--data-parallel-size",
            str(hardware.count // serving.tensor_parallel_size),
        ]
    else:
        args += launch.extra_cli_args
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK, release=materialize_release(manifest["vllm_release"], output)),
        {
            "TOKENIZERS_PARALLELISM": "false",
            "UV_NO_CONFIG": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": str(manifest["series"]["ready_timeout"]),
        }
        | serving.environment
        | (launch.subprocess_env if launch else {}),
    )
    return VllmEnvironment(
        InferenceModelConfig(name=model.repo, path=model.path, engine_kwargs={}),
        port=SERVER_PORT,
        timeout_seconds=manifest["series"]["ready_timeout"],
        extra_args=args,
        launcher=launcher,
        compilation_cache_mode=VllmCompilationCacheMode.MANAGED,
        router=VllmRouterConfig() if serving.router and (launch is None or launch.is_leader) else None,
        wait_for_ready=False,
    )


def run_checks(profile: Profile, server_url: str, output: Path, runtime_source: str) -> list[dict]:
    """Retain the shared report and require its outcome as well as the CLI exit."""
    checks = []
    for index, check in enumerate(profile.checks):
        record = {
            "name": check.name,
            "command": check.command,
            "status": "pending",
            "kind": check.kind,
            "checkpoint_revision": check.checkpoint_revision,
            "runtime_source": check.runtime_source,
            "evidence": check.evidence,
        }
        command = check.command
        if check.fixture:
            command = [
                sys.executable,
                "-m",
                "marin.testing.inference.check_endpoint",
                "--fixture",
                check.fixture,
                "--url",
                server_url,
                "--model",
                profile.model.repo,
                "--runtime-source",
                runtime_source,
                "--output",
                str(output / f"check-{index}.json"),
            ]
            record |= {
                "command": command,
                "checkpoint_revision": profile.model.revision,
                "runtime_source": runtime_source,
            }
        if command:
            log = output / f"check-{index}.log"
            started = time.time()
            try:
                run_logged(
                    command,
                    os.environ
                    | {
                        "KNOTMETER_SERVER_URL": server_url,
                        "KNOTMETER_MODEL": profile.model.repo,
                        "KNOTMETER_MODEL_REVISION": profile.model.revision,
                        "KNOTMETER_RUNTIME_SOURCE": runtime_source,
                        "KNOTMETER_PROFILE": str(output / "profile.json"),
                        "KNOTMETER_CHECK_OUTPUT": str(output / f"check-{index}.json"),
                    },
                    log,
                    timeout=check.timeout,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                record |= {"status": "failed", "error": str(error)}
            else:
                record["status"] = "passed"
            if check.fixture:
                report_path = output / f"check-{index}.json"
                if report_path.is_file():
                    report_bytes = report_path.read_bytes()
                    report = json.loads(report_bytes)
                    criteria = {
                        key: value
                        for key, value in report["policy"].items()
                        if key
                        not in {"id", "rationale", "evidence", "hardware", "dtype", "quantization", "kv_cache_dtypes"}
                    }
                    record |= {
                        "report": report_path.name,
                        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                        "reference": report["manifest"],
                        "policy": report["policy"]["id"],
                        "policy_sha256": report["policy_sha256"],
                        "criteria_sha256": hashlib.sha256(canonical_bytes(criteria)).hexdigest(),
                        "provisional": report["policy"]["provisional"],
                    }
                    if report["status"] != "passed":
                        record["status"] = report["status"]
                    if (
                        report["model"]["revision"] != profile.model.revision
                        or report["runtime_source"] != runtime_source
                    ):
                        record |= {"status": "pending", "reason": "The saved check report belongs to another deployment"}
                else:
                    record |= {"status": "pending", "reason": "The shared checker produced no report"}
            record |= {"started_unix": started, "finished_unix": time.time(), "log": log.name}
            if (
                not check.fixture
                and check.kind == "numerical"
                and (check.checkpoint_revision != profile.model.revision or check.runtime_source != runtime_source)
            ):
                record |= {
                    "regression_status": record["status"],
                    "status": "pending",
                    "reason": "The check's declared checkpoint/runtime scope does not match this deployment",
                }
        checks.append(record)
    return checks


def run_point(plan: TrialPlan, output: Path, bundle: Path, launch: IrisVllmLaunch | None) -> None:
    manifest = plan["manifest"]
    series = manifest["series"]
    profile = Profile.model_validate(series["profile"])
    info = get_job_info()
    assert info is not None
    leader = launch is None or launch.is_leader
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "manifest": manifest,
        "status": "preparing",
        "started_unix": time.time(),
        "iris_task_id": os.environ["IRIS_TASK_ID"],
        "iris_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
        "iris_bundle_id": os.environ["IRIS_BUNDLE_ID"],
        "task_index": info.task_index,
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "iris_node_name": os.environ.get("IRIS_NODE_NAME"),
    }
    native_logs = None
    router_logs = None
    failure = None
    try:
        record["gpu_uuids"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).splitlines()
        if len(set(record["gpu_uuids"])) != profile.gpus.per_host:
            raise ValueError("Visible physical GPUs differ from the resolved allocation")
        nodes = manifest["placement"].get("nodes")
        if nodes and record["iris_node_name"] not in nodes:
            raise RuntimeError("Iris placed this host outside the verified node list")
        with (output / "hardware.txt").open("w") as stream:
            for command in (
                ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"],
                ["nvidia-smi", "topo", "-m"],
                ["lscpu"],
            ):
                subprocess.run(command, stdout=stream, check=True)
        (output / "profile.json").write_text(profile.model_dump_json(indent=2))
        restore_workload(manifest["workload"], bundle)
        path = output / "series.json"
        path.write_text(json.dumps(series))
        check_native_cache(bundle, path, output / "workload-check.log")
        record["workload_restored_unix"] = time.time()
        with create_server(manifest, output, launch) as server:
            handle = server.vllm_server
            assert handle is not None
            native_logs = Path(handle.log_dir)
            record |= {"status": "starting", "server_command": server.command}
            write_record(str(output / "attempt.json"), record)
            if not leader:
                assert launch is not None
                wait_for_iris_vllm_shutdown(launch, server.check_alive)
                record |= {"status": "succeeded", "shutdown_started_unix": time.time()}
                return
            server.wait_until_ready()
            router_logs = server.router.log_dir if server.router else None
            if server.model_id != profile.model.repo:
                raise RuntimeError("The serving endpoint returned a different model ID")
            metrics = server.worker_server_url.removesuffix(OPENAI_API_SUFFIX) + "/metrics"
            command = benchmark_command(
                series,
                manifest["concurrency"],
                server.server_url.removesuffix(OPENAI_API_SUFFIX),
                output / "aiperf_artifacts",
                metrics_url=metrics,
            )
            record |= {
                "status": "checking",
                "ready_unix": time.time(),
                "benchmark_command": command,
                "worker_metrics_url": metrics,
                "router_command": server.router.command if server.router else None,
                "gang_attempts": gang_attempts(profile.gpus.count // profile.gpus.per_host),
            }
            write_record(str(output / "attempt.json"), record)
            with iris_vllm_followers(launch) if launch else contextlib.nullcontext():
                record["checks"] = run_checks(
                    profile, server.worker_server_url, output, manifest["vllm_release"]["source_commit"]
                )
                write_record(str(output / "attempt.json"), record)
                if any(check["status"] != "passed" for check in record["checks"]) and not series.get(
                    "measure_failed_checks"
                ):
                    raise RuntimeError("Required correctness evidence did not pass; AgentX measurement was not started")
                # The checks exercise prefix reuse. Clear those entries before native AgentX
                # warmup; the upstream replay and cache-busting controls remain unchanged.
                reset = requests.post(
                    server.worker_server_url.removesuffix(OPENAI_API_SUFFIX) + "/reset_prefix_cache", timeout=60
                )
                reset.raise_for_status()
                record |= {
                    "status": "measuring",
                    "checks_completed_unix": time.time(),
                    "gang_attempts": gang_attempts(profile.gpus.count // profile.gpus.per_host),
                }
                write_record(str(output / "attempt.json"), record)
                run_logged(
                    command,
                    workload_environment(bundle, series),
                    output / "agentx.log",
                    timeout=series["warmup_grace_period"] + series["duration"] + 3600,
                    sample=output / "client-load.jsonl",
                )
                record["measured_unix"] = time.time()
                exported = json.loads((output / PROFILE_EXPORT_RELATIVE_PATH).read_text())
                record["submission_valid"] = exported.get("metadata", {}).get(
                    "submission_valid"
                ) is True and not exported.get("was_cancelled")
                if not record["submission_valid"] and not manifest["diagnostic"]:
                    raise RuntimeError("AgentX rejected the window; retained as invalid diagnostics")
                server.check_alive()
                if gang_attempts(profile.gpus.count // profile.gpus.per_host) != record["gang_attempts"]:
                    raise RuntimeError("The serving gang changed native attempts during profiling")
                server.check_alive()
                record["status"] = "measured"
                pack_server_metrics(output)
                capture_attempt(plan, output, record, info.task_index, phase="before-shutdown")
                record |= {"status": "stopping", "shutdown_started_unix": time.time()}
            if launch and launch.num_tasks > 1:
                record["followers_stopped_unix"] = time.time()
        record["status"] = "succeeded"
    except BaseException as error:
        failure = error
        record |= {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        raise
    finally:
        record["finished_unix"] = time.time()
        if failure is None:
            record["shutdown_completed_unix"] = record["finished_unix"]
        capture_error = None
        for source, name in ((native_logs, "server-logs"), (router_logs, "router-logs")):
            if source:
                try:
                    shutil.copytree(source, output / name)
                except OSError as error:
                    capture_error = capture_error or error
                    record.setdefault("capture_errors", []).append(f"{name}: {type(error).__name__}: {error}")
        if leader:
            try:
                pack_server_metrics(output)
            except Exception as error:
                capture_error = capture_error or error
                record.setdefault("capture_errors", []).append(f"metrics: {type(error).__name__}: {error}")
        try:
            memory_limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            record["host_memory"] = {
                "peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text()),
                "limit_bytes": None if memory_limit == "max" else int(memory_limit),
                "source": "Task cgroup v2, before final artifact upload",
            }
        except (OSError, ValueError) as error:
            record["host_memory"] = {"error": str(error)}
        if capture_error and failure is None:
            record |= {"status": "failed", "error": f"{type(capture_error).__name__}: {capture_error}"}
        try:
            capture_attempt(plan, output, record, info.task_index, phase="final")
        except BaseException:
            if failure is None:
                raise
            logger.exception("Artifact capture failed; preserving the original task failure")
        if capture_error and failure is None:
            raise capture_error
        if launch and not leader and record["status"] == "succeeded":
            notify_iris_vllm_stopped(launch)


def run(plan: TrialPlan) -> None:
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    profile = Profile.model_validate(plan["manifest"]["series"]["profile"])
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "point"
    with tempfile.TemporaryDirectory(prefix="knotmeter-workload-") as directory:
        bundle = Path(directory) / "bundle"
        if profile.gpus.count == profile.gpus.per_host:
            run_point(plan, output, bundle, None)
        else:
            with iris_vllm_launch(profile.geometry) as launch:
                run_point(plan, output, bundle, launch)
