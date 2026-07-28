# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device, is_job_finished
from marin.inference.config import DEFAULT_CUDA_VLLM_VERSION
from marin.inference.model_preparation import resolve_model_path
from marin.inference.proxy import _reserve_port
from marin.inference.vllm_server import IsolatedCudaVllm, _poll_until_ready
from rigging.timing import Duration, ExponentialBackoff

MODEL = "zai-org/GLM-5.2-FP8"
MODEL_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
CUDA_COMPILER_REQUIREMENT = "cuda-toolkit[cccl,crt,cudart,nvcc,nvvm]==13.0.2"
MODEL_CACHE_TTL_DAYS = 30
GPUS_PER_NODE = 4
VLLM_REPLICAS = 2
TENSOR_PARALLEL_SIZE = GPUS_PER_NODE * VLLM_REPLICAS
GPU_MEMORY_UTILIZATION = 0.9
ENDPOINT_TIMEOUT = 3 * 3600
RUN_TIMEOUT_HOURS = 30 * 24
RAY_PORT = "ray"
HTTP_PORT = "http"


@dataclass(frozen=True)
class ServerConfig:
    max_model_len: int
    max_num_seqs: int
    kv_cache_dtype: str = "auto"
    decode_context_parallel_size: int = 1


@dataclass(frozen=True)
class Glm52LaunchConfig:
    vllm_endpoint: str
    ray_endpoint: str
    server: ServerConfig


def _ray_worker_port_args(*excluded_ports: int) -> list[str]:
    for minimum in (20000, 30000, 40000, 50000):
        maximum = minimum + 9999
        if not any(minimum <= port <= maximum for port in excluded_ports):
            return [f"--min-worker-port={minimum}", f"--max-worker-port={maximum}"]
    raise ValueError(f"Could not select Ray worker ports excluding {excluded_ports}")


def _network_interface(host: str) -> str:
    for name, addresses in psutil.net_if_addrs().items():
        if any(address.family == socket.AF_INET and address.address == host for address in addresses):
            return name
    raise RuntimeError(f"No network interface owns advertised host IP {host}")


def wait_for_endpoint_url(name: str, job=None, timeout: float = ENDPOINT_TIMEOUT) -> str:
    """Return the first resolved URL for an Iris endpoint."""
    ctx = iris_ctx()
    assert ctx is not None
    resolved: list[str] = []

    def endpoint_ready() -> bool:
        result = ctx.resolver.resolve(name)
        if not result.is_empty:
            resolved.append(result.first().url)
            return True
        if job is not None and is_job_finished(job.state):
            raise RuntimeError(f"Job {job} finished before registering endpoint {name!r}")
        return False

    ExponentialBackoff(initial=15, maximum=15, jitter=0).wait_until_or_raise(
        endpoint_ready,
        timeout=Duration.from_seconds(timeout),
        error_message=f"Timed out waiting for endpoint {name!r}",
    )
    return resolved[0]


def _vllm_launch_context() -> tuple[list[str], dict[str, str]]:
    launcher = IsolatedCudaVllm(version=DEFAULT_CUDA_VLLM_VERSION)
    command = launcher.command()
    python_index = command.index("--python")
    command[python_index:python_index] = [
        "--with",
        "ray[cgraph]>=2.55.1",
        "--with",
        CUDA_COMPILER_REQUIREMENT,
    ]
    return command, {**os.environ, **launcher.env()}


def _cuda_overlay(cuda_root: Path) -> str:
    cuda_home = Path(tempfile.mkdtemp(prefix="cuda-home-"))
    for directory in ("bin", "include", "nvvm"):
        (cuda_home / directory).symlink_to(cuda_root / directory, target_is_directory=True)
    lib64 = cuda_home / "lib64"
    lib64.mkdir()
    for library in (cuda_root / "lib").iterdir():
        (lib64 / library.name).symlink_to(library, target_is_directory=library.is_dir())
    cudart = next((cuda_root / "lib").glob("libcudart.so.*"))
    (lib64 / "libcudart.so").symlink_to(cudart)
    return str(cuda_home)


def _cuda_home(vllm_command: list[str], environment: dict[str, str]) -> str:
    script = """\
from pathlib import Path
import sys

nvcc = next(path for entry in sys.path for path in Path(entry).glob("nvidia/**/bin/nvcc"))
print(nvcc.parent.parent)
"""
    command = [*vllm_command[:-1], "python", "-c", script]
    cuda_root = Path(subprocess.check_output(command, env=environment, text=True).strip())
    return _cuda_overlay(cuda_root)


def _check_process_alive(process: subprocess.Popen[bytes]) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(f"vLLM exited with code {return_code} before becoming ready")


def _run_vllm(
    ctx,
    host: str,
    http_port: int,
    ray_address: str,
    vllm_command: list[str],
    environment: dict[str, str],
    weights: str,
    launch: Glm52LaunchConfig,
) -> None:
    process = subprocess.Popen(
        [
            *vllm_command,
            "serve",
            weights,
            "--served-model-name",
            MODEL,
            "--host",
            host,
            "--port",
            str(http_port),
            "--tensor-parallel-size",
            str(TENSOR_PARALLEL_SIZE),
            "--distributed-executor-backend",
            "ray",
            "--enable-expert-parallel",
            "--max-model-len",
            str(launch.server.max_model_len),
            "--max-num-seqs",
            str(launch.server.max_num_seqs),
            "--kv-cache-dtype",
            launch.server.kv_cache_dtype,
            "--decode-context-parallel-size",
            str(launch.server.decode_context_parallel_size),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--trust-remote-code",
        ],
        env={**environment, "RAY_ADDRESS": ray_address},
    )
    try:
        base_url = f"http://{host}:{http_port}"
        _poll_until_ready(
            f"{base_url}/v1",
            timeout_seconds=ENDPOINT_TIMEOUT,
            check_alive=lambda: _check_process_alive(process),
        )
        with ctx.registry.registered(launch.vllm_endpoint, base_url):
            return_code = process.wait()
            raise RuntimeError(f"vLLM exited with code {return_code}")
    finally:
        if process.poll() is None:
            process.terminate()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=30)
        if process.poll() is None:
            process.kill()


def _serve_ray_head(
    ctx,
    host: str,
    vllm_command: list[str],
    ray_command: list[str],
    environment: dict[str, str],
    launch: Glm52LaunchConfig,
) -> None:
    weights = prepare_model_cache()
    ray_port = _reserve_port(host, ctx.get_port(RAY_PORT))
    http_port = _reserve_port(host, ctx.get_port(HTTP_PORT))
    ray_address = f"{host}:{ray_port}"
    subprocess.run(
        [
            *ray_command,
            "start",
            "--head",
            f"--node-ip-address={host}",
            f"--port={ray_port}",
            *_ray_worker_port_args(ray_port, http_port),
            f"--num-gpus={GPUS_PER_NODE}",
            "--disable-usage-stats",
        ],
        check=True,
        env=environment,
    )
    with ctx.registry.registered(launch.ray_endpoint, ray_address):
        try:

            def ray_ready() -> bool:
                status = subprocess.run(
                    [*ray_command, "status", f"--address={ray_address}"],
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                return status.returncode == 0 and f"/{TENSOR_PARALLEL_SIZE}.0 GPU" in status.stdout

            ExponentialBackoff(initial=10, maximum=10, jitter=0).wait_until_or_raise(
                ray_ready,
                timeout=Duration.from_seconds(900),
                error_message=f"Ray cluster did not register all {TENSOR_PARALLEL_SIZE} GB200 GPUs",
            )
            _run_vllm(ctx, host, http_port, ray_address, vllm_command, environment, weights, launch)
        finally:
            subprocess.run([*ray_command, "stop", "--force"], env=environment, check=False)


def _serve_ray_worker(host: str, launch: Glm52LaunchConfig, ray_command: list[str], environment: dict[str, str]) -> None:
    ray_address = wait_for_endpoint_url(launch.ray_endpoint, timeout=ENDPOINT_TIMEOUT)
    subprocess.run(
        [
            *ray_command,
            "start",
            f"--address={ray_address}",
            f"--node-ip-address={host}",
            *_ray_worker_port_args(),
            f"--num-gpus={GPUS_PER_NODE}",
            "--disable-usage-stats",
            "--block",
        ],
        check=True,
        env=environment,
    )


def _serve_glm52(launch: Glm52LaunchConfig) -> None:
    info = get_job_info()
    ctx = iris_ctx()
    if info is None or ctx is None:
        raise RuntimeError("GLM-5.2 serving must run inside an Iris task")

    vllm_command, environment = _vllm_launch_context()
    ray_command = [*vllm_command[:-1], "ray"]
    host = info.advertise_host
    environment["CUDA_HOME"] = _cuda_home(vllm_command, environment)
    environment["VLLM_HOST_IP"] = host
    environment["GLOO_SOCKET_IFNAME"] = _network_interface(host)
    if info.task_index == 0:
        _serve_ray_head(ctx, host, vllm_command, ray_command, environment, launch)
        return
    _serve_ray_worker(host, launch, ray_command, environment)


def submit_glm52(ctx, launch: Glm52LaunchConfig):
    return ctx.client.submit(
        entrypoint=Entrypoint.from_callable(_serve_glm52, launch),
        name="vllm",
        resources=ResourceSpec(
            cpu=120,
            memory="850g",
            disk="1000g",
            device=gpu_device("GB200", GPUS_PER_NODE),
        ),
        environment=EnvironmentSpec(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        ports=[RAY_PORT, HTTP_PORT],
        coscheduling=CoschedulingConfig(group_by="nvlink.domain"),
        replicas=VLLM_REPLICAS,
        timeout=Duration.from_hours(RUN_TIMEOUT_HOURS),
        max_retries_failure=0,
    )


def prepare_model_cache() -> str:
    return resolve_model_path(MODEL, MODEL_CACHE_TTL_DAYS, MODEL_REVISION)
