# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a Hero BF16 export behind the Marin dashboard with native vLLM DP/EP across GB200 hosts.

Each Iris task runs four vLLM data-parallel ranks (TP1, expert parallel across all ranks). Task 0 runs
the API server and the dashboard; the other tasks run headless engines. The geometry, loader and
CUDA-graph settings follow the Hero final-SFT deployment from 2026-11-06
(``/dlwh/hero-final-sft16-graphs-20261106-v4-tools``).

Example::

    PARALLEL_API_KEY=... uv run python -m experiments.evaluation.serve.hero_dashboard \\
      --weights s3://marin-us-east-02a/marin/exports/... --model-id hero-sft-step400407 \\
      --name hero-sft-step400407-dashboard --nodes 16 --max-model-len 65536 \\
      --tokenizer marin-community/marin-tokenizer-tools --tokenizer-revision <commit> \\
      --secret PARALLEL_API_KEY env:PARALLEL_API_KEY
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import click
import requests
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, IrisContext, iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import PROXY_TIMEOUT_METADATA_KEY, EndpointAccess, Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.backend import ModelSpec
from marin.inference.chat_template_protocol import protocol_for_chat_template
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.dashboard_server import ServingInfo, bind_serving_socket, build_dashboard_app, serve_app_background
from marin.inference.iris_vllm import (
    IrisGangCoordinator,
    iris_gang_coordinator,
    iris_vllm_followers,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_backend import VllmBackend
from marin.inference.vllm_server import VllmEnvironment
from marin.inference.web_search import parallel_web_search_from_secret
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rigging.network import interface_for_ipv4
from rigging.secrets import resolve_secret_spec
from rigging.timing import Duration

logger = logging.getLogger(__name__)

GPUS_PER_NODE = 4
# Hero exports carry no generation_config.json; vLLM reads stop tokens from this file.
EOS_TOKEN_ID = 128009
GENERATION_CONFIG_DIR = Path("/cache/hero-dashboard-generation-config")
REQUEST_TIMEOUT_SECONDS = 600
STARTUP_TIMEOUT_SECONDS = 7200
MAX_NUM_SEQS = 32
MAX_NUM_BATCHED_TOKENS = 2048
GPU_MEMORY_UTILIZATION = 0.90
# The worker is pickled by value from __main__, and a protobuf enum wrapper cannot be pickled; keep the int.
DASHBOARD_ACCESS = EndpointAccess.ENDPOINT_ACCESS_LINK
# PIECEWISE graphs around attention, KV-cache updates and ShortConv; MoE stays inside the captured graph.
COMPILATION_CONFIG = {
    "mode": 3,
    "cudagraph_mode": "PIECEWISE",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
    "splitting_ops": [
        "vllm::unified_attention_with_output",
        "vllm::unified_kv_cache_update",
        "vllm::grug_moe_short_conv",
    ],
}
# Each rank streams its own expert shards; the memory limit bounds host RAM per rank.
LOADER_CONFIG = {"distributed": True, "concurrency": 4, "memory_limit": 16 * 1024**3}
SMOKE_PROMPT = "What is 2 + 2? Answer briefly."


@dataclass(frozen=True)
class HeroDashboardConfig:
    """One Hero dashboard deployment: weights, tokenizer, served name, endpoint, gang size and lifetime."""

    weights: str
    model_id: str
    endpoint: str
    nodes: int
    max_model_len: int
    tokenizer: str
    tokenizer_revision: str
    lifetime_hours: float

    @property
    def data_parallel_size(self) -> int:
        return self.nodes * GPUS_PER_NODE


def _vllm_args(config: HeroDashboardConfig, task_index: int, primary_host: str, primary_port: str) -> tuple[str, ...]:
    args = (
        "--data-parallel-size",
        str(config.data_parallel_size),
        "--data-parallel-size-local",
        str(GPUS_PER_NODE),
        "--data-parallel-address",
        primary_host,
        "--data-parallel-rpc-port",
        primary_port,
        "--enable-expert-parallel",
        "--compilation-config",
        json.dumps(COMPILATION_CONFIG),
        "--model-loader-extra-config",
        json.dumps(LOADER_CONFIG),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--generation-config",
        str(GENERATION_CONFIG_DIR),
        "--no-enable-prefix-caching",
        # The dashboard's Request stats panel shows TTFT, queue and decode timings only with this flag.
        "--enable-per-request-metrics",
        "--api-server-count",
        "1" if task_index == 0 else "0",
    )
    if task_index:
        args += ("--headless", "--data-parallel-start-rank", str(task_index * GPUS_PER_NODE))
    return args


def _smoke_test(server_url: str, model_id: str) -> dict:
    response = requests.post(
        f"{server_url}/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": SMOKE_PROMPT}],
            "temperature": 0,
            "max_tokens": 128,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    smoke = response.json()
    choice = smoke["choices"][0]
    if not choice["message"]["content"] or choice["finish_reason"] != "stop":
        raise RuntimeError(f"Serving smoke test failed: {json.dumps(smoke)}")
    return smoke


def serve(config: HeroDashboardConfig) -> None:
    """Run one task of the serving gang; task 0 also serves the dashboard until the lifetime ends."""
    configure_logging()
    configure_coreweave_s3()
    info = get_job_info()
    if info is None or info.num_tasks != config.nodes:
        raise RuntimeError(f"Expected an Iris gang of {config.nodes} tasks, got {info}")
    GENERATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (GENERATION_CONFIG_DIR / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS_TOKEN_ID}))
    identity = IrisGangCoordinator(
        str(info.job_id).strip("/").replace("/", "-") + "-dp-coordinator", info.task_index, config.nodes
    )
    ctx = iris_ctx()
    # Hold the leader's DP RPC port until the gang actor publishes it, then release it for vLLM.
    with bind_serving_socket(info.advertise_host, ctx.get_port("dp_rpc")) as rendezvous_socket:
        rendezvous_socket.listen(1)
        primary = f"{info.advertise_host}:{rendezvous_socket.getsockname()[1]}"
        with iris_gang_coordinator(identity, info.advertise_host, primary) as primary_address:
            rendezvous_socket.close()
            primary_host, primary_port = primary_address.rsplit(":", 1)
            engine = VllmEngineConfig(
                launcher=VllmLauncherType.CUDA,
                source=VllmSource.MARIN_FORK,
                startup_timeout_seconds=STARTUP_TIMEOUT_SECONDS,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                max_num_seqs=MAX_NUM_SEQS,
                extra_args=_vllm_args(config, info.task_index, primary_host, primary_port),
            )
            spec = ModelSpec(
                weights=config.weights,
                api_model=config.model_id,
                num_chips=GPUS_PER_NODE,
                tensor_parallel_size=1,
                dtype="bfloat16",
                max_model_len=config.max_model_len,
                chat_template_content=MARIN_CHAT_TEMPLATE,
                tokenizer=config.tokenizer,
                tokenizer_revision=config.tokenizer_revision,
            )
            network = {
                "VLLM_HOST_IP": info.advertise_host,
                "GLOO_SOCKET_IFNAME": interface_for_ipv4(info.advertise_host),
            }
            with VllmBackend(engine).start(spec, subprocess_env=network) as environment:
                if info.task_index:
                    wait_for_iris_vllm_shutdown(identity, environment.check_alive)
                else:
                    with iris_vllm_followers(identity):
                        environment.wait_until_ready()
                        _serve_dashboard(config, environment, ctx, info.advertise_host)
            if info.task_index:
                notify_iris_vllm_stopped(identity)


def _serve_dashboard(config: HeroDashboardConfig, environment: VllmEnvironment, ctx: IrisContext, host: str) -> None:
    smoke = _smoke_test(environment.server_url, config.model_id)
    logger.info("Serving smoke response: %s", json.dumps(smoke))
    web_search = parallel_web_search_from_secret()
    serving_info = ServingInfo(
        model=config.model_id,
        backend="vllm",
        tensor_parallel_size=1,
        max_model_len=config.max_model_len,
        dtype="bfloat16",
        has_chat_template=True,
        endpoint=config.endpoint,
        chat_template_protocol=protocol_for_chat_template(MARIN_CHAT_TEMPLATE),
    )
    app = build_dashboard_app(
        upstream_base_url=environment.server_url.removesuffix("/v1"),
        model_id=config.model_id,
        info=serving_info,
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        web_search=web_search,
    )
    metadata = {
        "kind": "marin-serve",
        "model": config.model_id,
        "backend": "vllm",
        "tensor_parallel_size": "1",
        "data_parallel_size": str(config.data_parallel_size),
        "pipeline_parallel_size": "1",
        "task_count": str(config.nodes),
        "cuda_graph_mode": COMPILATION_CONFIG["cudagraph_mode"],
        "max_model_len": str(config.max_model_len),
        "streaming": "true",
        PROXY_TIMEOUT_METADATA_KEY: str(REQUEST_TIMEOUT_SECONDS),
    }
    socket = bind_serving_socket(host, ctx.get_port("http"))
    address = f"http://{host}:{socket.getsockname()[1]}"
    with (
        serve_app_background(app, socket),
        ctx.registry.registered(config.endpoint, address, metadata, access=DASHBOARD_ACCESS),
    ):
        logger.info(
            "READY endpoint=%s model=%s DP/EP=%d web_search=%s vllm=%s",
            config.endpoint,
            config.model_id,
            config.data_parallel_size,
            web_search is not None,
            VLLM_GPU_RELEASE.release_tag,
        )
        deadline = time.monotonic() + config.lifetime_hours * 3600
        while time.monotonic() < deadline:
            environment.check_alive()
            time.sleep(30)


def _check_export(weights: str) -> dict:
    root = StoragePath(weights)
    if not (root / "export-manifest.json").exists():
        raise click.ClickException(f"{weights} has no export-manifest.json; the export is incomplete.")
    return json.loads((root / "config.json").read_text())


_PRIORITIES = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


@click.command()
@click.option("--weights", required=True, help="Completed Hero BF16 export (contains export-manifest.json).")
@click.option("--model-id", required=True, help="Served model name.")
@click.option("--name", "job_name", required=True, help="Iris job name; the endpoint is /serve/<name>.")
@click.option("--nodes", type=click.IntRange(min=1), required=True, help="GB200x4 hosts in one NVLink domain.")
@click.option("--max-model-len", type=int, required=True)
@click.option("--tokenizer", required=True, help="Hugging Face tokenizer repository.")
@click.option("--tokenizer-revision", required=True, help="Pinned tokenizer commit.")
@click.option("--lifetime-hours", type=float, default=24.0, show_default=True)
@click.option("--cluster", default="marin", show_default=True)
@click.option("--target-cluster", default="cw-us-east-08a", show_default=True)
@click.option("--priority", type=click.Choice(sorted(_PRIORITIES)), default="interactive", show_default=True)
@click.option(
    "--secret",
    "secret_sources",
    multiple=True,
    type=(str, str),
    help="Deliver a named secret (NAME SOURCE), e.g. --secret PARALLEL_API_KEY env:PARALLEL_API_KEY.",
)
@click.option("--dry-run", is_flag=True)
def main(
    weights: str,
    model_id: str,
    job_name: str,
    nodes: int,
    max_model_len: int,
    tokenizer: str,
    tokenizer_revision: str,
    lifetime_hours: float,
    cluster: str,
    target_cluster: str,
    priority: str,
    secret_sources: tuple[tuple[str, str], ...],
    dry_run: bool,
) -> None:
    """Submit one Hero dashboard serving gang."""
    configure_logging()
    configure_coreweave_s3()
    hf_config = _check_export(weights)
    config = HeroDashboardConfig(
        weights=weights,
        model_id=model_id,
        endpoint=f"/serve/{job_name}",
        nodes=nodes,
        max_model_len=max_model_len,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        lifetime_hours=lifetime_hours,
    )
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_NODE,
        cpu=100,
        ram="880GB",
        disk="1TB",
        target_cluster=target_cluster,
        preemptible=False,
    )
    coscheduling = resolve_coscheduling(resources, nodes)
    secrets = {name: resolve_secret_spec(source).value for name, source in secret_sources}
    click.echo(f"config: {config}")
    click.echo(
        f"export schema: {hf_config.get('grugmoe_artifact_schema_version')}; vLLM: {VLLM_GPU_RELEASE.release_tag}"
    )
    click.echo(f"resources: {nodes} x GB200x{GPUS_PER_NODE}; coscheduling: {coscheduling}; secrets: {sorted(secrets)}")
    if dry_run:
        return
    with connect_controller(cluster_name=cluster) as endpoint_info:
        with IrisClient.remote(
            endpoint_info.url, workspace=find_project_root(), credentials=endpoint_info.credentials
        ) as client:
            if client.list_endpoint_instances(config.endpoint):
                raise click.ClickException(f"Endpoint {config.endpoint} is already registered.")
            job = client.submit(
                entrypoint=Entrypoint.from_callable(serve, config),
                name=job_name,
                resources=convert_resources(resources),
                environment=EnvironmentSpec(
                    sync_packages=["marin-core"],
                    env_vars={"TMPDIR": "/cache", "JAX_PLATFORMS": "cpu"},
                    secrets=secrets,
                ),
                constraints=convert_constraints(resources) or None,
                coscheduling=coscheduling,
                replicas=nodes,
                ports=["http", "dp_rpc"],
                max_retries_failure=0,
                max_retries_preemption=2,
                timeout=Duration.from_seconds(lifetime_hours * 3600 + STARTUP_TIMEOUT_SECONDS),
                priority_band=_PRIORITIES[priority],
            )
            click.echo(f"job: {job.job_id}")
            click.echo(f"proxy: {client.resolve_endpoint(config.endpoint)}/")


if __name__ == "__main__":
    main()
