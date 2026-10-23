# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded calibration arms. Dry-run by default; alternate TPU pins are explicit."""

import argparse
import contextlib
import dataclasses
import json
import os
import re
import tempfile
import time
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import EnvironmentConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from marin.inference import iris as serving
from marin.inference import vllm_backend
from marin.inference.config import IrisConfig, ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.inference.vllm_server import IsolatedTpuVllm
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.evaluation.serving_parity.workload import HERE, MODEL, REVISION, run_workload

QUALIFIED_TPU = (
    "tpu-inference @ git+https://github.com/marin-community/tpu-inference.git@ae387b8bb7a644bbe7e17a9ce11bd701a012aa8d"
)
GLOBALS = {3, 7, 11, 15, 19, 23, 25}
LAYERS = ["full_attention" if i in GLOBALS else "sliding_attention" for i in range(26)]
RUNTIME = 3600


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    hardware: str
    output: str
    weights: str
    workaround: bool

    def resources(self) -> ResourceConfig:
        if self.hardware == "h100":
            return ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="512g", disk="200g", target_cluster="cw-rno2a")
        if self.hardware == "v6e-8":
            return ResourceConfig.with_tpu(
                "v6e-8", cpu=32, ram="128g", disk="100g", regions=["us-east5"], zone="us-east5-b"
            )
        return ResourceConfig.with_tpu(
            "v4-32", cpu=32, ram="360g", disk="100g", regions=["us-central2"], zone="us-central2-b", preemptible=False
        )

    def service(self) -> serving.IrisServiceConfig:
        gpu = self.hardware == "h100"
        args = ("--trust-remote-code", "--load-format", "runai_streamer")
        if gpu:
            args += (
                "--data-parallel-size",
                "8",
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
            )
        elif self.workaround:
            args += ("--hf-overrides", json.dumps({"layer_types": LAYERS}))
        return serving.IrisServiceConfig(
            model=ServedModelConfig(
                weights=self.weights,
                api_model=MODEL,
                tokenizer=MODEL,
                tokenizer_revision=REVISION,
                revision=REVISION if self.weights == MODEL else None,
                max_model_len=8192,
                tensor_parallel_size=1 if gpu else (16 if self.hardware == "v4-32" else 8),
            ),
            engine=VllmEngineConfig(
                launcher=VllmLauncherType.CUDA if gpu else VllmLauncherType.TPU,
                source=VllmSource.MARIN_FORK if gpu else VllmSource.UPSTREAM,
                startup_timeout_seconds=2400,
                max_num_batched_tokens=512,
                max_num_seqs=16,
                extra_args=args,
            ),
            iris=IrisConfig(
                worker_resources=self.resources(),
                worker_environment=EnvironmentConfig(workspace=str(Path.cwd()), extras=("cpu",) if gpu else ("tpu",)),
                max_retries_failure=0,
                max_retries_preemption=0,
            ),
            endpoint_name="/serve/" + self.name,
            timeout_hours=1,
        )


def run_arm(arm: Arm) -> None:
    started = time.monotonic()
    configure_coreweave_s3()
    root = StoragePath(arm.output)
    info = get_job_info()
    if info.task_index == 0:
        if (root / "started.json").exists():
            raise ValueError("Canary artifacts must use a fresh output prefix")
        (root / "started.json").write_text(json.dumps({"job": str(info.job_id), "epoch": time.time()}))
    gpu = arm.hardware == "h100"
    private = Path(tempfile.mkdtemp(prefix=".parity-", dir=Path.cwd()))
    (private / "tmp").mkdir()
    os.environ.update(
        UV_CACHE_DIR=str(private / "uv"), TMPDIR=str(private / "tmp"), RAY_DEDUP_LOGS="0", HF_HUB_DISABLE_XET="1"
    )
    if not gpu:
        os.environ["PARITY_TPU_VARIANT"] = arm.hardware
        original_command = IsolatedTpuVllm.command
        IsolatedTpuVllm.command = lambda self: [
            *original_command(self)[:-1],
            "python",
            str((HERE / "bootstrap.py").resolve()),
            "vllm",
        ]
        if arm.workaround:
            vllm_backend.vllm_launcher = lambda config: IsolatedTpuVllm(
                vllm_ref=VLLM_FORK_REQUIREMENT, tpu_inference_ref=QUALIFIED_TPU, python_version="3.12"
            )
    original_start = vllm_backend.VllmBackend.start
    holder = {}

    @contextlib.contextmanager
    def capture(self, *args, **kwargs):
        with original_start(self, *args, **kwargs) as environment:
            holder["environment"] = environment
            try:
                yield environment
            finally:
                (root / "server.log").write_text(environment.logs())

    vllm_backend.VllmBackend.start = capture

    def evaluate(check_alive, timeout_hours):
        environment = holder["environment"]
        logs = environment.logs()
        observed = {
            int(i): v
            for i, v in re.findall(
                r"RSI236_LAYER_STATIC\s+model\.layers\.(\d+)\.self_attn\.attn\s+sliding_window=(None|2048)", logs
            )
        }
        gate = dict(
            ready_seconds=time.monotonic() - started,
            observed=observed,
            kv_groups_4=bool(re.search(r"num_kv_cache_groups\s*=\s*4", logs)),
        )
        (root / "gate.json").write_text(json.dumps(gate))
        if not gpu and (
            observed != {i: "None" if i in GLOBALS else "2048" for i in range(26)} or not gate["kv_groups_4"]
        ):
            raise RuntimeError("Runtime layer schedule or KV groups differ; refusing eval")
        check_alive()
        endpoint = environment.server_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        item = json.loads((HERE / "teacher-item303.json").read_text())
        run_workload(endpoint, arm.output, item, max(1, RUNTIME - (time.monotonic() - started) - 120))
        check_alive()
        (root / "complete.json").write_text(json.dumps(dict(elapsed_seconds=time.monotonic() - started)))

    serving._block_until_timeout = evaluate
    (root / f"host-{info.task_index}-config.json").write_text(
        json.dumps(
            dict(
                arm=dataclasses.asdict(arm),
                shipped_tpu_pin=TPU_INFERENCE_FORK_REQUIREMENT,
                effective_tpu_pin=QUALIFIED_TPU if arm.workaround else TPU_INFERENCE_FORK_REQUIREMENT,
                vllm_tpu_pin=VLLM_FORK_REQUIREMENT,
            )
        )
    )
    try:
        serving.run_iris_service(arm.service())
    finally:
        (root / f"host-{info.task_index}-wall.json").write_text(json.dumps({"seconds": time.monotonic() - started}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hardware", choices=["h100", "v6e-8", "v4-32"], required=True)
    parser.add_argument("--weights", required=True, help="Pinned local-region checkpoint, or fixed HF model on GPU")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--workaround-pin",
        action="store_true",
        help="Explicit qualified ae387b8 + layer_types; not shipped-stack validation",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    arm = Arm(args.name, args.hardware, args.output, args.weights, args.workaround_pin)
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
    if arm.hardware == "h100" and not arm.output.startswith("s3://"):
        raise ValueError("CoreWeave output must use S3; GCS worker credentials are unavailable")
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=arm.name,
                entrypoint=Entrypoint.from_callable(run_arm, args=(arm,)),
                resources=convert_resources(resource),
                constraints=convert_constraints(resource),
                replicas=resource.replicas,
                coscheduling=resolve_coscheduling(resource, resource.replicas),
                ports=ports,
                environment=EnvironmentSpec(
                    extras=("cpu",) if arm.hardware == "h100" else ("tpu",),
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
