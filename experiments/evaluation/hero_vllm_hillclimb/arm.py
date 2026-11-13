# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the Hero SFT export on an Iris GB200 gang, probe it, record results and exit.

One arm = one Iris job. The leader task starts the vLLM API server, runs the gate probes and the
load sweep from the same host, writes JSON results plus the native vLLM log to ``output_uri`` and
returns; the followers stop with it, which releases the GPUs.

Usage (from a marin checkout):

    uv run python experiments/evaluation/hero_vllm_hillclimb/arm.py \
        --name baseline-3004cd8 --release 3004cd8 --phases gates,long,load
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from rigging.filesystem.factory import open_url
from rigging.log_setup import configure_logging

from experiments.evaluation.hero_vllm_hillclimb.probes import Client, LoadConfig, ProbeConfig, gate_probes, load_sweep, long_probes
from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel
from marin.inference.config import (
    IrisConfig,
    ServedModelConfig,
    ServingGeometry,
    VllmEngineConfig,
    VllmLauncherType,
    VllmSource,
)
from marin.inference.iris import _resolved_model, _vllm_model_spec
from marin.inference.iris_vllm import (
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_backend import VllmBackend

logger = logging.getLogger(__name__)

WEIGHTS = "s3://marin-us-east-02a/marin/exports/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
TOKENIZER = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
SERVED_NAME = "hero-sft-step400407"
CASES_URI = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/inputs/cases-0baf4938.json"
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/runs"
CLUSTER = "marin"
TARGET_CLUSTER = "cw-us-east-08a"
GPUS_PER_HOST = 4


def _release(tag: str, commit: str, date: str, sha_arm: str, sha_x86: str) -> VllmGpuRelease:
    version = f"0.0.0.dev{date}+marin.{commit[:12]}.cu132"
    base = f"https://github.com/marin-community/vllm/releases/download/{tag}/vllm-0.0.0.dev{date}%2Bmarin.{commit[:12]}.cu132-cp38-abi3-manylinux_2_28_"
    return VllmGpuRelease(
        release_tag=tag,
        source_commit=commit,
        version=version,
        torch_backend="cu132",
        torch_version="2.14.1+cu132",
        wheels=(
            VllmGpuWheel("aarch64", ("10.0",), base + "aarch64.whl", sha_arm),
            VllmGpuWheel("x86_64", ("9.0",), base + "x86_64.whl", sha_x86),
        ),
    )


RELEASES = {
    # Rafal's step-400407 server (2026-11-13) runs this wheel.
    "3004cd8": _release(
        "marin-vllm-gpu-20261110-3004cd8f1632",
        "3004cd8f163299116b96d1b324cfff418c387388",
        "20261110",
        "267aeb03bea44339882d277c9b5044cfcce19397c5ac244b48a7981112f9c366",
        "0d034503a2756fe55f4791095f395b45933e39b825607fa0f84c30bd1986a30a",
    ),
    # #11241's current pin.
    "0ac9dcb": _release(
        "marin-vllm-gpu-20261113-0ac9dcb157ee",
        "0ac9dcb157ee86ccb447bc717259159c41ad456a",
        "20261113",
        "e0a2b9e2bb6b16ba6d4e1821770123cc74f04b86128536c2744e91b59c6e4b6a",
        "b8897bde70bf353a08f37450fb89fca5ebf50d768d6756807a43867ef11134f4",
    ),
}

# Rafal's resolved serving flags, minus the dashboard-only chat template and generation config.
BASELINE_ARGS = (
    "--enable-expert-parallel",
    "--gpu-memory-utilization",
    "0.9",
    "--no-enable-prefix-caching",
    "--max-num-seqs",
    "32",
    "--max-num-batched-tokens",
    "2048",
    "--max-model-len",
    "262144",
    "--load-format",
    "runai_streamer",
    "--model-loader-extra-config",
    '{"distributed": true, "concurrency": 4, "memory_limit": 17179869184}',
    "--enable-per-request-metrics",
    "--trust-remote-code",
)


@dataclass(frozen=True)
class ArmConfig:
    name: str
    output_uri: str
    cases_uri: str
    geometry: ServingGeometry
    model: ServedModelConfig
    engine: VllmEngineConfig
    iris: IrisConfig
    phases: tuple[str, ...]
    probes: ProbeConfig
    env: tuple[tuple[str, str], ...] = ()


def _write_json(uri: str, payload: object) -> None:
    with open_url(uri, "w") as handle:
        json.dump(payload, handle)


def _read_json(uri: str) -> dict:
    with open_url(uri, "r") as handle:
        return json.load(handle)


def _copy_native_logs(log_dir: str, output_uri: str) -> None:
    for name in ("stdout.log", "stderr.log"):
        path = Path(log_dir) / name
        if path.exists():
            with path.open("rb") as source, open_url(f"{output_uri}/vllm_{name}", "wb") as target:
                shutil.copyfileobj(source, target)


_PHASES = {"gates": gate_probes, "long": long_probes, "load": load_sweep}


async def _run_probes(base_url: str, config: ArmConfig, cases: dict) -> None:
    """Run each requested phase; a failed phase is recorded and the next still runs on the warm server."""
    failures = []
    async with Client(base_url, SERVED_NAME) as client:
        for phase in config.phases:
            started = time.time()
            try:
                results = await _PHASES[phase](client, cases, config.probes)
            except Exception as error:
                logger.exception("phase %s failed", phase)
                failures.append(phase)
                results = {"error": repr(error)}
            _write_json(f"{config.output_uri}/{phase}.json", results | {"seconds": time.time() - started})
    if failures:
        raise RuntimeError(f"probe phases failed: {failures}")


def run_arm(config: ArmConfig) -> None:
    """Iris entrypoint for every task of an arm's gang."""
    configure_logging()
    geometry = config.geometry
    model, num_chips = _resolved_model(config.model, config.iris)
    spec = _vllm_model_spec(model, num_chips, None)
    backend = VllmBackend(config.engine)
    with iris_vllm_launch(
        pipeline_parallel_size=geometry.pipeline_parallel_size,
        tensor_parallel_size=geometry.tensor_parallel_size,
        data_parallel_size=geometry.global_data_parallel_size,
        data_parallel_size_local=geometry.data_parallel_size,
    ) as launch:
        subprocess_env = {**launch.subprocess_env, **dict(config.env)}
        with backend.start(spec, extra_args=launch.extra_cli_args, subprocess_env=subprocess_env) as environment:
            if launch.is_leader:
                with iris_vllm_followers(launch):
                    started = time.time()
                    environment.wait_until_ready()
                    ready_seconds = time.time() - started
                    server = environment.vllm_server
                    _write_json(
                        f"{config.output_uri}/arm.json",
                        {
                            "name": config.name,
                            "ready_seconds": ready_seconds,
                            "server_url": server.server_url,
                            "engine": dataclasses.asdict(config.engine),
                            "geometry": dataclasses.asdict(geometry),
                            "extra_cli_args": list(launch.extra_cli_args),
                        },
                    )
                    try:
                        asyncio.run(_run_probes(server.server_url, config, _read_json(config.cases_uri)))
                    finally:
                        _copy_native_logs(server.log_dir, config.output_uri)
            else:
                wait_for_iris_vllm_shutdown(launch, environment.check_alive)
        if not launch.is_leader:
            notify_iris_vllm_stopped(launch)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    parser.add_argument("--hosts", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--phases", default="gates,long,load")
    parser.add_argument("--concurrencies", default="64,256,512")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--long-lengths", default="4096,32768,65536")
    parser.add_argument("--vllm-arg", action="append", default=[], help="Extra vLLM flag; repeat per token")
    parser.add_argument("--drop-arg", action="append", default=[], help="Remove a baseline flag and its value")
    parser.add_argument(
        "--overlay-wheel",
        help="FILE:SHA256 - replace the release's aarch64 wheel with the URL stored in FILE (Python overlay)",
    )
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for every vLLM process")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _baseline_args(drop: list[str]) -> tuple[str, ...]:
    args = list(BASELINE_ARGS)
    for flag in drop:
        index = args.index(flag)
        has_value = index + 1 < len(args) and not args[index + 1].startswith("--")
        del args[index : index + (2 if has_value else 1)]
    return tuple(args)


def main() -> None:
    args = _parse_args()
    configure_logging()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    name = f"{args.name}-{stamp}"
    local_dp = GPUS_PER_HOST // args.tensor_parallel_size
    geometry = ServingGeometry(
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=local_dp,
        pipeline_parallel_size=1,
        gpus_per_task=GPUS_PER_HOST,
        data_parallel_hosts=args.hosts,
    )
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_HOST,
        cpu=100,
        ram="880g",
        disk="1t",
        replicas=geometry.task_count,
        preemptible=False,
    )
    workspace = Path.cwd()
    environment = create_environment(workspace=str(workspace))
    iris_config = IrisConfig(
        worker_resources=resources,
        worker_environment=environment,
        serving_geometry=geometry,
        cache_ttl_days=0,
    )
    release = RELEASES[args.release]
    if args.overlay_wheel:
        url_file, sha256 = args.overlay_wheel.rsplit(":", 1)
        url = Path(url_file).read_text().strip()
        release = replace(
            release,
            wheels=tuple(
                replace(wheel, url=url, sha256=sha256) if wheel.architecture == "aarch64" else wheel
                for wheel in release.wheels
            ),
        )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        gpu_release=release,
        startup_timeout_seconds=3600,
        extra_args=(*_baseline_args(args.drop_arg), *args.vllm_arg),
    )
    model = ServedModelConfig(
        weights=WEIGHTS,
        api_model=SERVED_NAME,
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
    )
    config = ArmConfig(
        name=name,
        output_uri=f"{OUTPUT_ROOT}/{name}",
        cases_uri=CASES_URI,
        geometry=geometry,
        model=model,
        engine=engine,
        iris=iris_config,
        phases=tuple(args.phases.split(",")),
        probes=ProbeConfig(
            model=SERVED_NAME,
            load=LoadConfig(
                concurrencies=tuple(int(c) for c in args.concurrencies.split(",")),
                max_tokens=args.max_tokens,
                waves=args.waves,
            ),
            long_lengths=tuple(int(n) for n in args.long_lengths.split(",")),
            data_parallel_size=geometry.global_data_parallel_size,
        ),
        env=tuple(tuple(item.split("=", 1)) for item in args.env),
    )
    submission = replace(resources, target_cluster=TARGET_CLUSTER)
    coscheduling = resolve_coscheduling(submission, submission.replicas)
    print(json.dumps({"name": name, "output": config.output_uri, "args": list(engine.extra_args), "env": args.env}, indent=1))
    print(f"replicas={submission.replicas} coscheduling={coscheduling}")
    if args.dry_run:
        return
    with connect_controller(cluster_name=CLUSTER) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=workspace, credentials=endpoint.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_arm, config),
                name=f"hero-vllm-{name}",
                resources=convert_resources(submission),
                environment=EnvironmentSpec(),
                ports=["http"],
                replicas=submission.replicas,
                coscheduling=coscheduling,
                constraints=convert_constraints(submission) or None,
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                timeout=None,
            )
            print(f"submitted {job}")


if __name__ == "__main__":
    main()
