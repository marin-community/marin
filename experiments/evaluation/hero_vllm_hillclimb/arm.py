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
import socket
import time
from dataclasses import dataclass, replace
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig, create_environment
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
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
from rigging.filesystem.factory import open_url
from rigging.log_setup import configure_logging

from experiments.evaluation.hero_vllm_hillclimb.probes import (
    Client,
    LoadConfig,
    ProbeConfig,
    decode_consistency,
    gate_probes,
    load_sweep,
    long_probes,
    long_repeats,
    long_repeats_profiled,
    peer_load_sweep,
    profile_points,
)

logger = logging.getLogger(__name__)

WEIGHTS = "s3://marin-us-east-02a/marin/exports/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
TOKENIZER = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
SERVED_NAME = "hero-sft-step400407"
CASES_URI = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/inputs/cases-0baf4938.json"
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/runs"
CLUSTER = "marin"
SNOWBALL_WEIGHTS = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm/"


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

# Rafal's resolved compilation config. Without grug_moe_short_conv in splitting_ops the fork's
# ShortConv guard silently downgrades cudagraph_mode to NONE (about 3x slower decode).
BASE_COMPILATION = {
    "cudagraph_mode": "PIECEWISE",
    "splitting_ops": [
        "vllm::unified_attention_with_output",
        "vllm::unified_kv_cache_update",
        "vllm::grug_moe_short_conv",
    ],
    "inductor_compile_config": {
        "enable_auto_functionalized_v2": False,
        "combo_kernels": True,
        "benchmark_combo_kernel": True,
    },
}
# vLLM's capture list at max_num_seqs 32 (max size 64); pinning it keeps prefills of 65-128 tokens off
# the graph path when max_num_seqs grows.
BASELINE_CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

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
    "--compilation-config",
    json.dumps(BASE_COMPILATION),
)
# Snowball (Grug 67B-A2B, no ShortConv layers) shares the GrugMoE code path; it serves as the
# small-model comparison for per-step overhead. Same flags as Hero except context length and the
# ShortConv split point.
SNOWBALL_COMPILATION = BASE_COMPILATION | {
    "splitting_ops": ["vllm::unified_attention_with_output", "vllm::unified_kv_cache_update"]
}
SNOWBALL_ARGS = (
    *BASELINE_ARGS[: BASELINE_ARGS.index("--max-model-len")],
    "--max-model-len",
    "65536",
    *BASELINE_ARGS[BASELINE_ARGS.index("--max-model-len") + 2 : BASELINE_ARGS.index("--compilation-config")],
    "--compilation-config",
    json.dumps(SNOWBALL_COMPILATION),
)


@dataclass(frozen=True)
class ServedTarget:
    """A model export and the hardware it is served on."""

    weights: str
    tokenizer: str
    tokenizer_revision: str | None
    served_name: str
    accelerator: str
    gpus_per_host: int
    cluster: str
    baseline_args: tuple[str, ...]


TARGETS = {
    "hero": ServedTarget(
        WEIGHTS, TOKENIZER, TOKENIZER_REVISION, SERVED_NAME, "GB200", 4, "cw-us-east-08a", BASELINE_ARGS
    ),
    "snowball": ServedTarget(
        SNOWBALL_WEIGHTS,
        "marin-community/marin-tokenizer",
        None,
        "snowball-sft-step630",
        "H100",
        8,
        "cw-us-east-02a",
        SNOWBALL_ARGS,
    ),
}

# Rafal's server runs a single API server process (its log shows one APIServer pid). Only the
# leader starts API servers; headless followers reject the flag.
LEADER_ARGS = ("--api-server-count", "1")

# Leader-only flag overrides per candidate label (API servers start only on the leader).
LEADER_OVERRIDES = {
    "mns128api4": ("--api-server-count", "4"),
    "fg128api4": ("--api-server-count", "4"),
    "fg128stack": ("--api-server-count", "4"),
    "fg192stack": ("--api-server-count", "4"),
}

MOE_CONFIG_DIR = "experiments/evaluation/hero_vllm_hillclimb/moe_configs"

# Environment added to every vLLM process for a candidate label; "{workspace}" is the job's workspace.
LABEL_ENV = {
    # NCCL symmetric-memory windows for the MoE all-gather / reduce-scatter (lower-latency kernels).
    "fg128symm": (("VLLM_USE_NCCL_SYMM_MEM", "1"),),
    # Fused-MoE Triton configs tuned for Hero's EP16 decode shapes on GB200 (moe_tune.py).
    "fg128moe": (("VLLM_TUNED_CONFIG_FOLDER", "{workspace}/" + MOE_CONFIG_DIR),),
    "fg128stack": (
        ("VLLM_TUNED_CONFIG_FOLDER", "{workspace}/" + MOE_CONFIG_DIR),
        ("VLLM_USE_NCCL_SYMM_MEM", "1"),
    ),
    "fg192stack": (
        ("VLLM_TUNED_CONFIG_FOLDER", "{workspace}/" + MOE_CONFIG_DIR),
        ("VLLM_USE_NCCL_SYMM_MEM", "1"),
    ),
}

# Local torch-profiler trace directory on every host; traces are uploaded after the server exits.
PROFILE_DIR = "/tmp/hero-vllm-profile"
# Bounded decode-step traces: no Python stacks or shapes, no blocking key_averages table on stop.
PROFILER_CONFIG = {
    "profiler": "torch",
    "torch_profiler_dir": PROFILE_DIR,
    "torch_profiler_with_stack": False,
    "torch_profiler_dump_cuda_time_total": False,
    "ignore_frontend": True,
    "max_iterations": 20,
}

# Candidate flag sets, selected with --sequence items of the form <release>:<label>. A flag here
# replaces the same baseline flag (and its value) rather than being appended twice.
CANDIDATE_ARGS = {
    "mns48": ("--max-num-seqs", "48"),
    "mns128gmu95": ("--max-num-seqs", "128", "--gpu-memory-utilization", "0.95"),
    "mns64": ("--max-num-seqs", "64"),
    "mns64p": (
        "--max-num-seqs",
        "64",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_capture_sizes": BASELINE_CAPTURE_SIZES}),
    ),
    "mns128": ("--max-num-seqs", "128"),
    "mns256": ("--max-num-seqs", "256"),
    "mns128api4": ("--max-num-seqs", "128"),
    "mns128prof": ("--max-num-seqs", "128", "--profiler-config", json.dumps(PROFILER_CONFIG)),
    # Two-host Hero (DP8/EP8): experts take ~130 GB per GPU, so KV fits only with gmu 0.95.
    "mns32prof95": (
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.95",
        "--profiler-config",
        json.dumps(PROFILER_CONFIG),
    ),
    # Inductor still fuses combo kernels but picks them without timing benchmarks (compile determinism test).
    "bckoff": (
        "--compilation-config",
        json.dumps(
            BASE_COMPILATION
            | {
                "inductor_compile_config": (
                    BASE_COMPILATION["inductor_compile_config"] | {"benchmark_combo_kernel": False}
                )
            }
        ),
    ),
    "mbt8192": ("--max-num-batched-tokens", "8192"),
    # FULL_AND_PIECEWISE: uniform decode batches replay one full CUDA graph (needs the ShortConv
    # guard relaxed in the wheel: --label-overlay fg32prof95=FILE:SHA256); mixed batches stay piecewise.
    "fg32prof95": (
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.95",
        "--profiler-config",
        json.dumps(PROFILER_CONFIG),
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    "fg128": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    "fg128moe": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    "fg128api4": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    # FULL graphs + tuned MoE config + NCCL symmetric memory + four API servers.
    "fg128stack": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    # The full stack at 192 sequences per engine (KV holds about 220 sequences of 1.1K tokens).
    "fg192stack": (
        "--max-num-seqs",
        "192",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    # D1 test: one capture size, so every decode step pads to the same graph regardless of peer load.
    "fg128cap1": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [128]}),
    ),
    "fg32cap1_95": (
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.95",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [32]}),
    ),
    "fg128symm": (
        "--max-num-seqs",
        "128",
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    "fg128prof": (
        "--max-num-seqs",
        "128",
        "--profiler-config",
        json.dumps(PROFILER_CONFIG),
        "--compilation-config",
        json.dumps(BASE_COMPILATION | {"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    ),
    "a2a-nvl1": ("--all2all-backend", "flashinfer_nvlink_one_sided"),
    "a2a-nvl2": ("--all2all-backend", "flashinfer_nvlink_two_sided"),
}


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
    leader_args: tuple[str, ...] = LEADER_ARGS


def _write_json(uri: str, payload: object) -> None:
    with open_url(uri, "w") as handle:
        json.dump(payload, handle, default=lambda value: sorted(value) if isinstance(value, frozenset) else str(value))


def _read_json(uri: str) -> dict:
    with open_url(uri, "r") as handle:
        return json.load(handle)


def _copy_native_logs(log_dir: str, output_uri: str) -> None:
    for name in ("stdout.log", "stderr.log"):
        path = Path(log_dir) / name
        if path.exists():
            with path.open("rb") as source, open_url(f"{output_uri}/vllm_{name}", "wb") as target:
                shutil.copyfileobj(source, target)


_PHASES = {
    "gates": gate_probes,
    "long": long_probes,
    "load": load_sweep,
    "profile": profile_points,
    "decode": decode_consistency,
    "longrep": long_repeats,
    "longprof": long_repeats_profiled,
    "d1": peer_load_sweep,
}


async def _run_probes(base_url: str, config: ArmConfig, cases: dict) -> None:
    """Run each requested phase; a failed phase is recorded and the next still runs on the warm server.

    The load sweep is saved after every concurrency point so a preemption keeps finished points.
    """
    failures = []
    async with Client(base_url, config.probes.model) as client:
        for phase in config.phases:
            started = time.time()
            uri = f"{config.output_uri}/{phase}.json"
            try:
                if phase == "load":
                    results = await load_sweep(
                        client,
                        cases,
                        config.probes,
                        on_progress=lambda partial, uri=uri, started=started: _write_json(
                            uri, partial | {"partial": True, "seconds": time.time() - started}
                        ),
                    )
                else:
                    results = await _PHASES[phase](client, cases, config.probes)
            except Exception as error:
                logger.exception("phase %s failed", phase)
                failures.append(phase)
                results = {"error": repr(error)}
            _write_json(uri, results | {"seconds": time.time() - started})
    if failures:
        raise RuntimeError(f"probe phases failed: {failures}")


def _upload_profiles(output_uri: str) -> None:
    """Copy this host's torch-profiler traces to the run directory and clear them for the next variant."""
    root = Path(PROFILE_DIR)
    if not root.exists():
        return
    host = socket.gethostname()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with (
                path.open("rb") as source,
                open_url(f"{output_uri}/profile/{host}/{path.relative_to(root)}", "wb") as target,
            ):
                shutil.copyfileobj(source, target)
    shutil.rmtree(root)


def _assert_cuda_graphs(log_dir: str) -> None:
    """G0: refuse to measure a server whose CUDA graphs were silently disabled."""
    text = (Path(log_dir) / "stdout.log").read_text(errors="replace")
    text += (Path(log_dir) / "stderr.log").read_text(errors="replace") if (Path(log_dir) / "stderr.log").exists() else ""
    if "CUDAGraphMode.NONE" in text or "overriding cudagraph_mode to NONE" in text:
        raise RuntimeError("G0 failed: vLLM resolved cudagraph_mode=NONE")
    if "Capturing CUDA graphs" not in text:
        raise RuntimeError("G0 failed: no CUDA graph capture in the vLLM log")


def run_arms(configs: tuple[ArmConfig, ...]) -> None:
    """Iris entrypoint: serve and probe each variant in order on the same gang.

    Running variants back to back on one allocation removes rack/placement confounds between
    them. A failed variant is recorded and the sequence continues; the job fails at the end.
    """
    configure_logging()
    failures = []
    for instance, config in enumerate(configs):
        error = run_arm(config, instance)
        if error is not None:
            failures.append((config.name, error))
    if failures:
        raise RuntimeError(f"variants failed: {failures}")


def run_arm(config: ArmConfig, instance: int = 0) -> str | None:
    """Serve and probe one variant on every task of the gang; return the leader's error, if any."""
    error = None
    geometry = config.geometry
    model, num_chips = _resolved_model(config.model, config.iris)
    spec = _vllm_model_spec(model, num_chips, None)
    backend = VllmBackend(config.engine)
    with iris_vllm_launch(
        pipeline_parallel_size=geometry.pipeline_parallel_size,
        tensor_parallel_size=geometry.tensor_parallel_size,
        data_parallel_size=geometry.global_data_parallel_size,
        data_parallel_size_local=geometry.data_parallel_size,
        instance=instance,
    ) as launch:
        subprocess_env = {
            **launch.subprocess_env,
            **{key: value.format(workspace=Path.cwd()) for key, value in config.env},
        }
        extra_args = (*launch.extra_cli_args, *(config.leader_args if launch.is_leader else ()))
        with backend.start(spec, extra_args=extra_args, subprocess_env=subprocess_env) as environment:
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
                            "extra_cli_args": list(extra_args),
                        },
                    )
                    try:
                        _assert_cuda_graphs(server.log_dir)
                        asyncio.run(_run_probes(server.server_url, config, _read_json(config.cases_uri)))
                    except Exception as failure:
                        logger.exception("variant %s failed", config.name)
                        error = repr(failure)
                    finally:
                        _copy_native_logs(server.log_dir, config.output_uri)
            else:
                wait_for_iris_vllm_shutdown(launch, environment.check_alive)
        _upload_profiles(config.output_uri)
        if not launch.is_leader:
            notify_iris_vllm_stopped(launch)
    return error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    parser.add_argument(
        "--sequence",
        help="Comma-separated releases to run back to back on the same gang (e.g. 3004cd8,0ac9dcb,3004cd8); "
        "overrides --release",
    )
    parser.add_argument("--model", choices=sorted(TARGETS), default="hero")
    parser.add_argument("--hosts", type=int, default=4)
    parser.add_argument("--cpu", type=int, default=100, help="CPU request per host")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--phases", default="gates,long,load")
    parser.add_argument("--concurrencies", default="64,256,512")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--client-processes", type=int, default=1, help="Load-generator processes")
    parser.add_argument("--long-lengths", default="4096,32768,65536")
    parser.add_argument("--d1-peer-loads", default="1,2,8,32", help="Requests per peer engine for the d1 phase")
    parser.add_argument("--profile-per-engine", default="1,32,128", help="Sequences per engine for the profile phase")
    parser.add_argument("--vllm-arg", action="append", default=[], help="Extra vLLM flag; repeat per token")
    parser.add_argument("--drop-arg", action="append", default=[], help="Remove a baseline flag and its value")
    parser.add_argument(
        "--overlay-wheel",
        help="FILE:SHA256 - replace the release's aarch64 wheel with the URL stored in FILE (Python overlay)",
    )
    parser.add_argument(
        "--label-overlay",
        action="append",
        default=[],
        help="LABEL=FILE:SHA256 - run variants with this label on an overlay wheel (URL in FILE)",
    )
    parser.add_argument("--tp2-overlay", help="FILE:SHA256 of the #119 overlay wheel URL for tp2 variants")
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for every vLLM process")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Submit to the cw-us-east-08a controller directly instead of federating through the marin hub",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _baseline_args(baseline: tuple[str, ...], drop: list[str]) -> tuple[str, ...]:
    args = list(baseline)
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
    target = TARGETS[args.model]
    local_dp = target.gpus_per_host // args.tensor_parallel_size
    geometry = ServingGeometry(
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=local_dp,
        pipeline_parallel_size=1,
        gpus_per_task=target.gpus_per_host,
        data_parallel_hosts=args.hosts,
    )
    resources = ResourceConfig.with_gpu(
        target.accelerator,
        count=target.gpus_per_host,
        cpu=args.cpu,
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
    releases = args.sequence.split(",") if args.sequence else [args.release]
    assert all(item.split(":")[0] in RELEASES for item in releases), releases
    assert all(item.split(":")[1] in (*CANDIDATE_ARGS, "tp2") for item in releases if ":" in item), releases
    configs = [
        _variant_config(
            args,
            target,
            name if len(releases) == 1 else f"{name}-v{i}-{item.replace(':', '-')}",
            item,
            geometry,
            iris_config,
        )
        for i, item in enumerate(releases)
    ]
    config = configs[0]
    engine = config.engine
    submission = resources if args.direct else replace(resources, target_cluster=target.cluster)
    cluster = target.cluster if args.direct else CLUSTER
    coscheduling = resolve_coscheduling(submission, submission.replicas)
    print(
        json.dumps(
            {"name": name, "outputs": [c.output_uri for c in configs], "args": list(engine.extra_args), "env": args.env},
            indent=1,
        )
    )
    print(f"replicas={submission.replicas} coscheduling={coscheduling}")
    if args.dry_run:
        return
    print(f"controller={cluster} target_cluster={submission.target_cluster}")
    with connect_controller(cluster_name=cluster) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=workspace, credentials=endpoint.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(run_arms, tuple(configs)),
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


def _with_flags(base: tuple[str, ...], flags: tuple[str, ...]) -> tuple[str, ...]:
    """Replace each (flag, value) pair of ``flags`` in ``base``, appending flags not present.

    An empty value marks a bare boolean flag.
    """
    out = list(base)
    for flag, value in zip(flags[::2], flags[1::2], strict=True):
        if value == "":
            if flag not in out:
                out.append(flag)
        elif flag in out:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    return tuple(out)


# TP2 needs #119 (merged onto 0ac9dcb with schema-3 fixes, shipped as an aarch64 overlay wheel) and its
# qualified NCCL recipe: native NCCL all-reduce only, no fused all-reduce passes.
# TP2 overlay wheel: --tp2-overlay FILE:SHA256, where FILE holds the wheel URL (the URL may be presigned,
# so it is read at submission time and never committed).
TP2_ENV = (
    ("VLLM_ALLREDUCE_USE_FLASHINFER", "0"),
    ("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "0"),
    ("VLLM_ALLREDUCE_USE_SYMM_MEM", "0"),
    ("VLLM_USE_NCCL_SYMM_MEM", "0"),
)
TP2_FLAGS = (
    "--disable-custom-all-reduce",
    "",
    "--compilation-config",
    json.dumps(
        BASE_COMPILATION | {"pass_config": {"fuse_allreduce_rms": False, "enable_sp": False, "fuse_gemm_comms": False}}
    ),
)


def _overlay(release: VllmGpuRelease, spec: str) -> VllmGpuRelease:
    url_file, sha256 = spec.rsplit(":", 1)
    url = Path(url_file).read_text().strip()
    return replace(
        release,
        wheels=tuple(
            replace(wheel, url=url, sha256=sha256) if wheel.architecture == "aarch64" else wheel
            for wheel in release.wheels
        ),
    )


def _variant_config(args, target: ServedTarget, name, item, geometry, iris_config) -> ArmConfig:
    tag, _, label = item.partition(":")
    release = RELEASES[tag]
    tensor_parallel_size = args.tensor_parallel_size
    env = (*(tuple(entry.split("=", 1)) for entry in args.env), *LABEL_ENV.get(label, ()))
    flags: tuple[str, ...] = CANDIDATE_ARGS[label] if label in CANDIDATE_ARGS else ()
    if label == "tp2":
        assert tag == "0ac9dcb", "the #119 overlay is built on 0ac9dcb"
        assert args.tp2_overlay, "tp2 needs --tp2-overlay FILE:SHA256 (the #119 overlay wheel)"
        release = _overlay(release, args.tp2_overlay)
        tensor_parallel_size = 2
        geometry = replace(geometry, tensor_parallel_size=2, data_parallel_size=target.gpus_per_host // 2)
        env = (*env, *TP2_ENV)
        flags = TP2_FLAGS
    label_overlays = dict(entry.split("=", 1) for entry in args.label_overlay)
    if label in label_overlays:
        release = _overlay(release, label_overlays[label])
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
        extra_args=_with_flags((*_baseline_args(target.baseline_args, args.drop_arg), *args.vllm_arg), flags),
    )
    model = ServedModelConfig(
        weights=target.weights,
        api_model=target.served_name,
        tokenizer=target.tokenizer,
        tokenizer_revision=target.tokenizer_revision,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
    )
    return ArmConfig(
        name=name,
        output_uri=f"{OUTPUT_ROOT}/{name}",
        cases_uri=CASES_URI,
        geometry=geometry,
        model=model,
        engine=engine,
        iris=iris_config,
        phases=tuple(args.phases.split(",")),
        probes=ProbeConfig(
            model=target.served_name,
            load=LoadConfig(
                concurrencies=tuple(int(c) for c in args.concurrencies.split(",")),
                max_tokens=args.max_tokens,
                waves=args.waves,
                client_processes=args.client_processes,
            ),
            long_lengths=tuple(int(n) for n in args.long_lengths.split(",")),
            data_parallel_size=geometry.global_data_parallel_size,
            profile_per_engine=tuple(int(n) for n in args.profile_per_engine.split(",")),
            d1_peer_loads=tuple(int(n) for n in args.d1_peer_loads.split(",")),
        ),
        env=env,
        leader_args=LEADER_OVERRIDES.get(label, LEADER_ARGS),
    )


if __name__ == "__main__":
    main()
