# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded shipped-pin calibration arms; dry-run by default."""

import argparse
import contextlib
import dataclasses
import functools
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import EnvironmentConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT, VLLM_GPU_RELEASE
from marin.inference import iris as serving
from marin.inference import vllm_backend
from marin.inference.config import IrisConfig, ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.inference.vllm_release import current_vllm_gpu_wheel, vllm_gpu_wheel_provenance
from marin.inference.vllm_server import IsolatedTpuVllm, VllmEnvironment
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.serving_parity.settings import HERE, MODEL, PROBE, REVISION, HardwareKind
from experiments.evaluation.serving_parity.workload import run_workload

RUNTIME = 3600


def h100_resources() -> ResourceConfig:
    return ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="200g", target_cluster="cw-rno2a")


def v6e_resources() -> ResourceConfig:
    return ResourceConfig.with_tpu("v6e-8", cpu=32, ram="128g", disk="100g", regions=["us-east5"], zone="us-east5-b")


def v5p_resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(
        "v5p-8", cpu=32, ram="360g", disk="100g", regions=["us-east5"], zone="us-east5-a", preemptible=True
    )


def v4_resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(
        "v4-32", cpu=32, ram="360g", disk="100g", regions=["us-central2"], zone="us-central2-b", preemptible=False
    )


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    name: str
    resources: Callable[[], ResourceConfig]
    launcher: VllmLauncherType
    source: VllmSource
    tensor_parallel_size: int
    extras: tuple[str, ...]
    extra_args: tuple[str, ...] = ()

    @property
    def is_tpu(self) -> bool:
        return self.launcher is VllmLauncherType.TPU


PROFILES = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            HardwareProfile(
                HardwareKind.H100,
                h100_resources,
                VllmLauncherType.CUDA,
                VllmSource.MARIN_FORK,
                1,
                ("cpu",),
                (
                    "--data-parallel-size",
                    "8",
                    "--enable-expert-parallel",
                    "--model-loader-extra-config",
                    '{"distributed":true}',
                ),
            ),
            HardwareProfile(HardwareKind.V6E, v6e_resources, VllmLauncherType.TPU, VllmSource.UPSTREAM, 8, ("tpu",)),
            HardwareProfile(HardwareKind.V5P, v5p_resources, VllmLauncherType.TPU, VllmSource.UPSTREAM, 4, ("tpu",)),
            HardwareProfile(HardwareKind.V4, v4_resources, VllmLauncherType.TPU, VllmSource.UPSTREAM, 16, ("tpu",)),
        )
    }
)


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    hardware: str
    output: str
    weights: str
    diagnostic_v4_tiles: bool = False
    source_commit: str | None = None

    def __post_init__(self) -> None:
        if self.hardware not in PROFILES:
            raise ValueError("Unknown calibration hardware")
        if self.diagnostic_v4_tiles and self.profile.resources is not v4_resources:
            raise ValueError("Experimental tile selection requires a diagnostic v4 arm")

    @property
    def profile(self) -> HardwareProfile:
        return PROFILES[self.hardware]

    def entrypoint(self) -> Entrypoint:
        return Entrypoint.from_callable(run_arm, self)

    def resources(self) -> ResourceConfig:
        return self.profile.resources()

    def service(self) -> serving.IrisServiceConfig:
        profile = self.profile
        return serving.IrisServiceConfig(
            model=ServedModelConfig(
                weights=self.weights,
                api_model=MODEL,
                tokenizer=MODEL,
                tokenizer_revision=REVISION,
                revision=REVISION if self.weights == MODEL else None,
                max_model_len=PROBE.max_model_len,
                tensor_parallel_size=profile.tensor_parallel_size,
            ),
            engine=VllmEngineConfig(
                launcher=profile.launcher,
                source=profile.source,
                startup_timeout_seconds=2400,
                max_num_batched_tokens=PROBE.max_num_batched_tokens,
                max_num_seqs=PROBE.max_num_seqs,
                extra_args=("--trust-remote-code", "--load-format", "runai_streamer", *profile.extra_args),
            ),
            iris=IrisConfig(
                worker_resources=self.resources(),
                worker_environment=EnvironmentConfig(workspace=str(Path.cwd()), extras=profile.extras),
                max_retries_failure=0,
                max_retries_preemption=0,
            ),
            endpoint_name="/serve/" + self.name,
            timeout_hours=RUNTIME / 3600,
        )


@contextlib.contextmanager
def isolated_runtime(arm: Arm):
    """Scope instrumentation to this arm and restore process state on every exit."""
    with tempfile.TemporaryDirectory(prefix=".parity-", dir=Path.cwd()) as directory, contextlib.ExitStack() as stack:
        private = Path(directory)
        (private / "tmp").mkdir()
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "UV_CACHE_DIR": str(private / "uv"),
                    "TMPDIR": str(private / "tmp"),
                    "RAY_DEDUP_LOGS": "0",
                    "HF_HUB_DISABLE_XET": "1",
                },
            )
        )
        report = private / "runtime-report.json"
        if arm.profile.is_tpu:
            config = private / "bootstrap.json"
            config.write_text(
                json.dumps(
                    dict(
                        probe=dataclasses.asdict(PROBE),
                        weights=arm.weights,
                        hardware=arm.hardware,
                        tensor_parallel_size=arm.profile.tensor_parallel_size,
                        diagnostic_v4_tiles=arm.diagnostic_v4_tiles,
                        preflight_path=str(private / "preflight.json"),
                        runtime_report_path=str(report),
                    )
                )
            )
            original_command = IsolatedTpuVllm.command

            def command(launcher):
                return [
                    *original_command(launcher)[:-1],
                    "python",
                    str(HERE / "bootstrap.py"),
                    "--config",
                    str(config),
                    "vllm",
                ]

            stack.enter_context(patch.object(IsolatedTpuVllm, "command", command))
        yield report


def sanitized_logs(text: str) -> str:
    """Do not persist capability URLs in startup diagnostics."""
    return re.sub(r"https?://[^\s\"'<>]+", "<redacted-url>", text)


@contextlib.contextmanager
def capture_backend(root: StoragePath):
    """Archive startup failures before environment teardown discards its handle."""
    original_start = vllm_backend.VllmBackend.start
    original_failure = VllmEnvironment._report_failure_and_close
    holder = {}

    def archive(environment):
        try:
            (root / "server.log").write_text(sanitized_logs(environment.logs()))
        except Exception as error:
            logging.warning("Could not archive server diagnostics (%s)", type(error).__name__)

    def failure(environment, message):
        try:
            archive(environment)
            (root / "startup-failure.json").write_text(json.dumps(dict(category="server_startup_failure")))
        except Exception as error:
            logging.warning("Could not archive startup category (%s)", type(error).__name__)
        finally:
            original_failure(environment, message)

    @contextlib.contextmanager
    def capture(backend, *args, **kwargs):
        try:
            with original_start(backend, *args, **kwargs) as environment:
                holder["environment"] = environment
                try:
                    yield environment
                finally:
                    archive(environment)
        except Exception as error:
            try:
                (root / "backend-failure.json").write_text(json.dumps(dict(category=type(error).__name__)))
            except Exception as upload_error:
                logging.warning("Could not archive backend category (%s)", type(upload_error).__name__)
            raise

    with patch.object(VllmEnvironment, "_report_failure_and_close", failure):
        with patch.object(vllm_backend.VllmBackend, "start", capture):
            yield holder


def runtime_gate(arm: Arm, root: StoragePath, environment, report: Path, started: float) -> dict:
    """Check the actual layer trace against the isolated preflight's derived schedule."""
    runtime = json.loads(report.read_text()) if arm.profile.is_tpu else dict(patches=[])
    observed = {
        int(i): value
        for i, value in re.findall(
            r"SERVING_PARITY_LAYER_STATIC\s+model\.layers\.(\d+)\.self_attn\.attn\s+sliding_window=(None|2048)",
            environment.logs(),
        )
    }
    expected = {
        i: "None" if layer == "full_attention" else "2048"
        for i, layer in enumerate(runtime.get("preflight", {}).get("expected", []))
    }
    gate = dict(
        ready_seconds=time.monotonic() - started,
        observed=observed,
        kv_groups_4=bool(re.search(r"num_kv_cache_groups\s*=\s*4", environment.logs())),
        runtime=runtime,
    )
    (root / "gate.json").write_text(json.dumps(gate))
    if arm.profile.is_tpu and (
        runtime["preflight"]["status"] != "PASS" or observed != expected or not gate["kv_groups_4"]
    ):
        raise RuntimeError("Runtime layer schedule or KV groups differ; refusing eval")
    return runtime


def evaluate_service(arm, root, started, holder, report, check_alive, timeout_hours):
    environment = holder["environment"]
    runtime = runtime_gate(arm, root, environment, report, started)
    check_alive()
    endpoint = environment.server_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    provenance = dict(
        arm=dataclasses.asdict(arm),
        shipped_tpu_pin=TPU_INFERENCE_FORK_REQUIREMENT,
        effective_tpu_pin=TPU_INFERENCE_FORK_REQUIREMENT,
        vllm_tpu_pin=VLLM_FORK_REQUIREMENT,
        runtime_patches=runtime["patches"],
        shipped_stack_eligible=not arm.diagnostic_v4_tiles,
        h100_wheel=(
            dataclasses.asdict(vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, current_vllm_gpu_wheel(VLLM_GPU_RELEASE)))
            if not arm.profile.is_tpu
            else None
        ),
        preemptible=arm.hardware in (HardwareKind.V5P, HardwareKind.V6E),
        calibration_source_commit=arm.source_commit,
        serving_job=str(get_job_info().job_id),
        attempt_uid=get_job_info().attempt_uid,
    )
    remaining = min(RUNTIME, timeout_hours * 3600) - (time.monotonic() - started) - 120
    if remaining <= 0:
        raise TimeoutError("No calibration budget remains after serving startup")
    eval_started = time.monotonic()
    run_workload(endpoint, arm.output, json.loads((HERE / "teacher-item303.json").read_text()), remaining, provenance)
    (root / "phase-timing.json").write_text(
        json.dumps(
            dict(
                worker_start_to_ready_seconds=eval_started - started,
                teacher_and_eval_seconds=time.monotonic() - eval_started,
                queue_seconds=None,
                load_seconds=None,
                compile_seconds=None,
                note=(
                    "Resolve queue/load/compile from controller transitions and server log; "
                    "null means not yet resolved."
                ),
            )
        )
    )
    check_alive()
    (root / "complete.json").write_text(json.dumps(dict(elapsed_seconds=time.monotonic() - started)))


def run_arm(arm: Arm) -> None:
    started = time.monotonic()
    configure_coreweave_s3()
    root = StoragePath(arm.output)
    info = get_job_info()
    if info is None:
        raise RuntimeError("Calibration must execute inside an Iris job")
    if info.task_index == 0:
        if (root / "started.json").exists():
            raise ValueError("Calibration artifacts require a fresh output prefix")
        (root / "started.json").write_text(json.dumps({"job": str(info.job_id), "epoch": time.time()}))
    with isolated_runtime(arm) as report, capture_backend(root) as holder:
        evaluate = functools.partial(evaluate_service, arm, root, started, holder, report)
        (root / f"host-{info.task_index}-config.json").write_text(
            json.dumps(
                dict(
                    arm=dataclasses.asdict(arm),
                    shipped_tpu_pin=TPU_INFERENCE_FORK_REQUIREMENT,
                    effective_tpu_pin=TPU_INFERENCE_FORK_REQUIREMENT,
                    vllm_tpu_pin=VLLM_FORK_REQUIREMENT,
                )
            )
        )
        try:
            with patch.object(serving, "_block_until_timeout", evaluate):
                serving.run_iris_service(arm.service())
        finally:
            if report.exists():
                (root / f"host-{info.task_index}-runtime.json").write_bytes(report.read_bytes())
            (root / f"host-{info.task_index}-wall.json").write_text(json.dumps({"seconds": time.monotonic() - started}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hardware", choices=tuple(PROFILES), required=True)
    parser.add_argument("--weights", required=True, help="Pinned regional checkpoint, or fixed HF model on GPU")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--diagnostic-v4-tiles", action="store_true", help="Experimental kernel; never eligible for shipped-stack PASS"
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit and subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode:
        parser.error("Commit source changes before submitting a reproducible calibration")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    arm = Arm(args.name, args.hardware, args.output, args.weights, args.diagnostic_v4_tiles, source_commit)
    resource = arm.resources()
    arm.service()
    ports = ["http", *RAY_PORTS] if resource.replicas > 1 else ["http"]
    print(
        json.dumps(
            dict(
                arm=dataclasses.asdict(arm),
                replicas=resource.replicas,
                ports=ports,
                coscheduling=str(resolve_coscheduling(resource, resource.replicas)),
                runtime_seconds=RUNTIME,
            ),
            indent=2,
        )
    )
    if not args.submit:
        return
    if not arm.profile.is_tpu and not arm.output.startswith("s3://"):
        raise ValueError("CoreWeave output must use S3; GCS worker credentials are unavailable")
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=arm.name,
                entrypoint=arm.entrypoint(),
                resources=convert_resources(resource),
                constraints=convert_constraints(resource),
                replicas=resource.replicas,
                coscheduling=resolve_coscheduling(resource, resource.replicas),
                ports=ports,
                environment=EnvironmentSpec(
                    extras=arm.profile.extras,
                    env_vars={
                        key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN") if key in os.environ
                    },
                ),
                timeout=Duration.from_seconds(RUNTIME),
                scheduling_timeout=Duration.from_seconds(3600),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            print(job)


if __name__ == "__main__":
    main()
