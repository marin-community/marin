# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-serve`` — one-liner to serve an HF model on an Iris TPU or GPU slice.

Submits a single Iris job that boots a serving backend on a single-host TPU or GPU slice and
registers a browser dashboard + OpenAI-compatible endpoint through the controller proxy. The job
stops itself after ``--timeout-hours`` so a forgotten server frees its slice.

Examples::

    marin-serve Qwen/Qwen3-0.6B --cluster marin --tpu v6e-8
    marin-serve Qwen/Qwen3-0.6B --cluster marin --tpu v6e-8 --backend levanter
    marin-serve gs://my-bucket/ckpt --tpu v5litepod-8 --chat-template delphi_v0.jinja2
    marin-serve Qwen/Qwen3-0.6B --cluster marin --gpu H100x8 --target-cluster cw-rno2a

``--backend`` picks the stack that answers requests: ``vllm`` (default) runs vLLM as a
subprocess, ``levanter`` runs Levanter's inference engine in-process on the slice's chips. Both
expose the same OpenAI API through the same dashboard, so serving one model under each and
pointing one client at both endpoints compares them directly.

``--gpu`` and ``--tpu`` are mutually exclusive (the default is TPU ``v6e-8``). On the vLLM GPU
path CUDA vLLM is provisioned in an isolated ``uv`` tool env — stock PyPI vLLM at ``--vllm-version``,
or Marin's vLLM fork with ``--vllm-source marin-fork`` (needed for Marin-custom architectures like
grug_moe); on the vLLM TPU path, a marin checkout serves vLLM from the workspace lock, and outside a
checkout (or with ``--isolated-vllm``) from an isolated ``uv`` tool env holding Marin's forked TPU
vLLM. The Levanter backend needs no vLLM at all: it serves from the worker venv's JAX.

``--cluster`` selects the controller to submit to; ``--target-cluster`` federates the job to a
named peer. The slice's tensor-parallel size and (for clamped-RoPE models) max sequence length are
inferred automatically; override with ``--tensor-parallel-size`` / ``--max-model-len``.
"""

import contextlib
import importlib.metadata
import logging
import re
import shlex
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import click
import requests
from click.core import ParameterSource
from iris.cli.connect import open_controller_endpoint
from iris.cli.job import parse_gpu_spec
from iris.client import IrisClient, Job
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, region_constraint
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import (
    EndpointAccess,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    gpu_device,
    is_job_finished,
    tpu_device,
)
from iris.rpc import job_pb2
from rigging.config_discovery import find_project_root
from rigging.connect import capability_path, proxy_path
from rigging.timing import Duration

from marin.inference.quick_serve import QuickServeConfig, serve_in_job
from marin.inference.serving_backend import LevanterBackend, ServingBackend, VllmBackend
from marin.inference.tpu_vllm_pins import tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_server import (
    WORKER_PYTHON_VERSION,
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    VllmType,
)

# vLLM and the dashboard need the generic TPU stack plus the TPU-vLLM runtime.
_WORKER_EXTRAS = ("tpu", "vllm")
# The GPU serve worker only runs the dashboard/registry glue plus a `vllm serve`
# subprocess; CUDA vLLM is provisioned in an isolated uv-tool env (not the workspace
# lock), so the worker venv needs no accelerator extra at all — base Marin suffices.
_GPU_WORKER_EXTRAS: tuple[str, ...] = ()
# Levanter serves in-process, so the worker venv needs the accelerator's JAX and nothing else:
# marin-core already depends on marin-levanter[serve] for the FastAPI/uvicorn surface.
_LEVANTER_TPU_EXTRAS = ("tpu",)
_LEVANTER_GPU_EXTRAS = ("gpu",)
# Pinned CUDA vLLM for the GPU path, overridable with --vllm-version. Stock PyPI vLLM
# (>=0.25) targets torch 2.11 / CUDA 13; provisioned per-job via uvx, so a bump is just
# this string with no workspace re-lock.
DEFAULT_CUDA_VLLM_VERSION = "0.25.0"
_ENDPOINT_READY_POLL_SECONDS = 5.0

# Options that only mean something to one backend, by Click parameter name. Passing one to the
# other backend is a mistake worth failing on, but several carry non-None defaults, so only a
# value the user actually typed counts (hence the ParameterSource check in _reject_foreign_flags).
_VLLM_ONLY_OPTIONS = {
    "vllm_version": "--vllm-version",
    "vllm_source": "--vllm-source",
    "vllm_args": "--vllm-arg",
    "isolated_vllm": "--isolated-vllm",
    "max_num_batched_tokens": "--max-num-batched-tokens",
}
_LEVANTER_ONLY_OPTIONS = {
    "max_seqs": "--max-seqs",
    "page_size": "--page-size",
    "hbm_utilization": "--hbm-utilization",
}


def _reject_foreign_flags(backend: str, options: dict[str, str]) -> None:
    """Fail if the user typed an option belonging to a backend they did not select."""
    ctx = click.get_current_context()
    typed = [flag for name, flag in options.items() if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE]
    if typed:
        raise click.ClickException(f"{', '.join(typed)} cannot be used with --backend {backend}.")


@dataclass(frozen=True)
class ServingPlan:
    """How to serve on this slice: which backend, on what device, in which worker environment."""

    backend: ServingBackend
    device: job_pb2.DeviceConfig
    worker_extras: tuple[str, ...]
    tpu_type: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None


def _resolve_serving_plan(
    *,
    backend: str,
    tpu: str,
    gpu: str | None,
    in_checkout: bool,
    isolated_vllm: bool,
    task_image: str | None,
    cuda_vllm_version: str,
    vllm_source: VllmType,
    vllm: VllmBackend,
    levanter: LevanterBackend,
    extras: tuple[str, ...],
) -> ServingPlan:
    """Pick the backend, the slice's device, and the extras the worker venv needs to run it.

    ``vllm`` and ``levanter`` arrive carrying the knobs the user set; what is decided here is which
    of them serves, and where its runtime comes from. Levanter computes in the worker venv, so that
    venv carries the accelerator's JAX and nothing else; vLLM runs as a subprocess, either from the
    workspace lock or from an isolated uv-tool env.
    """
    if vllm_source is VllmType.MARIN_FORK and (gpu is None or backend != "vllm"):
        raise click.ClickException("--vllm-source marin-fork requires --gpu with the vLLM backend.")
    if gpu is not None:
        if not in_checkout:
            raise click.ClickException(
                "marin-serve --gpu serves from a marin checkout (the worker runs marin-core from "
                "it); none was found. Run marin-serve from inside a marin checkout."
            )
        try:
            gpu_type, gpu_count = parse_gpu_spec(gpu)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        device = gpu_device(gpu_type, gpu_count)
        if backend == "levanter":
            return ServingPlan(
                levanter, device, (*_LEVANTER_GPU_EXTRAS, *extras), gpu_type=gpu_type, gpu_count=gpu_count
            )
        # Provision CUDA vLLM in an isolated uv-tool env unless the operator brought a prebuilt
        # --task-image, which is expected to ship its own vLLM on PATH.
        if task_image is None:
            if vllm_source is VllmType.MARIN_FORK:
                launcher = IsolatedCudaVllm(source=VllmType.MARIN_FORK)
            else:
                launcher = IsolatedCudaVllm(source=VllmType.UPSTREAM, version=cuda_vllm_version)
            vllm = replace(vllm, launcher=launcher)
        return ServingPlan(vllm, device, (*_GPU_WORKER_EXTRAS, *extras), gpu_type=gpu_type, gpu_count=gpu_count)

    topology = get_tpu_topology(tpu)
    if topology.vm_count != 1:
        raise click.ClickException(
            f"{tpu!r} is a multi-host slice (vm_count={topology.vm_count}); quick-serve supports "
            f"single-host slices only (e.g. v6e-8, v5litepod-8)."
        )
    device = tpu_device(tpu)
    if backend == "levanter":
        return ServingPlan(levanter, device, (*_LEVANTER_TPU_EXTRAS, *extras), tpu_type=tpu)
    if isolated_vllm or not in_checkout:
        # Provision the forked TPU vLLM from an isolated uvx env when there is no checkout to build
        # it from (or when explicitly requested); otherwise serve the workspace TPU-vLLM from the
        # lock. The worker venv always needs the `tpu` extra for the serving glue's jax/libtpu; the
        # `vllm` extra is only for the in-workspace build.
        vllm = replace(
            vllm, launcher=IsolatedTpuVllm(vllm_ref=vllm_fork_ref(), tpu_inference_ref=tpu_inference_fork_ref())
        )
        return ServingPlan(vllm, device, ("tpu", *extras), tpu_type=tpu)
    return ServingPlan(vllm, device, (*_WORKER_EXTRAS, *extras), tpu_type=tpu)


def _default_job_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")[:24]
    suffix = uuid.uuid4().hex[:6]
    return f"serve-{slug}-{suffix}" if slug else f"serve-{suffix}"


def _marin_core_version() -> str:
    """Installed ``marin-core`` version, used to pin the checkout-free worker install."""
    return importlib.metadata.version("marin-core")


def _checkout_free_setup_script(marin_version: str, extras: tuple[str, ...]) -> str:
    """Build a fresh venv and install ``marin-core`` from PyPI at ``marin_version``.

    The worker installs the launching CLI's exact ``marin-core`` version so cloudpickled
    entrypoints stay compatible. ``--torch-backend cpu`` routes torch/torchvision to the
    CPU PyTorch index (jax and libtpu do TPU compute; torch is only a dependency).
    """
    extras_suffix = f"[{','.join(extras)}]" if extras else ""
    spec = f"marin-core{extras_suffix}=={marin_version}"
    return (
        "set -e\n"
        f'uv venv "$IRIS_VENV" --python {WORKER_PYTHON_VERSION}\n'
        f'uv pip install --python "$IRIS_VENV" --link-mode symlink --torch-backend cpu {shlex.quote(spec)}\n'
    )


def _resolve_chat_template(spec: str | None) -> str | None:
    if spec is None:
        return None
    if spec.startswith(("http://", "https://")):
        response = requests.get(spec, timeout=30)
        response.raise_for_status()
        return response.text
    path = Path(spec)
    if not path.is_file():
        raise click.ClickException(f"--chat-template {spec!r} is not a readable file or a http(s) URL.")
    return path.read_text()


def _wait_for_endpoint(client: IrisClient, job: Job, endpoint_name: str, timeout_seconds: float) -> str:
    """Poll the controller registry until the endpoint registers; return its address."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if is_job_finished(job.state):
            raise click.ClickException(
                f"Job {job} finished before registering an endpoint. Inspect logs with `iris job logs {job}`."
            )
        # The registry probe is the authenticated path to readiness; the controller
        # proxy itself is auth-gated and not pollable with a plain HTTP client.
        endpoints = client._cluster_client.list_endpoints(endpoint_name, exact=True)
        if endpoints:
            return endpoints[0].address
        time.sleep(_ENDPOINT_READY_POLL_SECONDS)
    raise click.ClickException(
        f"Timed out after {timeout_seconds:.0f}s waiting for {endpoint_name!r}. "
        "The job is still booting the model; re-check later via the Iris dashboard."
    )


def _mint_and_print_capability_url(
    client: IrisClient, endpoint: str, dashboard_url: str | None, ttl_hours: float
) -> None:
    """Mint a scoped endpoint token and print the off-cluster capability URL.

    Runs CLI-side under the launching user's identity, so the controller's owner
    check passes. The URL embeds the scoped token in its path (gist-style):
    possession of the URL is the credential, so no auth header is needed. It
    authorizes only this endpoint and expires after ``ttl_hours`` (clamped to the
    controller's maximum).
    """
    resp = client._cluster_client.mint_endpoint_token(endpoint, ttl=Duration.from_hours(ttl_hours))
    hours_left = max(0.0, (resp.expires_at.epoch_ms - int(time.time() * 1000)) / 3_600_000)
    click.echo("  Shared capability URL (token in the path — anyone with the URL can call it):")
    if dashboard_url:
        base_url = f"{dashboard_url.rstrip('/')}{capability_path(endpoint, resp.token)}/v1"
        click.echo(f"    base_url   {base_url}")
        click.echo("    api_key    <any non-empty string>   (the URL already carries the credential)")
        click.echo(f"    expires    in {hours_left:.1f}h")
        click.echo(f"    example    curl {base_url}/models")
    else:
        # No public origin known (bare --controller); front the controller's
        # /proxy/t route for this to be reachable off-cluster.
        click.echo(f"    path       {capability_path(endpoint, resp.token)}/v1  (front the controller /proxy/t route)")
        click.echo(f"    expires    in {hours_left:.1f}h")
    click.echo("")


@click.command(context_settings={"show_default": True})
@click.argument("model")
@click.option(
    "--backend",
    type=click.Choice(["vllm", "levanter"]),
    default="vllm",
    help="Serving stack: vllm (subprocess) or levanter (in-process JAX inference engine).",
)
@click.option("--cluster", default="marin", envvar="IRIS_CLUSTER", help="Named iris cluster to submit to.")
@click.option(
    "--controller", default=None, envvar="IRIS_CONTROLLER", help="Pre-tunneled controller URL (overrides --cluster)."
)
@click.option("--tpu", default="v6e-8", help="Single-host TPU slice type (e.g. v6e-8, v5litepod-8).")
@click.option(
    "--gpu",
    default=None,
    help="GPU slice (e.g. H100x8); mutually exclusive with --tpu. Selects the GPU serving path.",
)
@click.option(
    "--target-cluster",
    default=None,
    help="Federate the job to this peer cluster (e.g. cw-rno2a). --cluster still selects the controller.",
)
@click.option(
    "--isolated-vllm",
    is_flag=True,
    default=False,
    help="TPU path: provision vLLM from the isolated uvx env (Marin's forked TPU vLLM) "
    "even inside a checkout. Auto-selected when marin-serve runs outside a checkout.",
)
@click.option("--name", default=None, help="Iris job name (default: derived from the model).")
@click.option("--endpoint-name", default=None, help="Endpoint name to register (default: /serve/<job-name>).")
@click.option("--chat-template", default=None, help="Jinja chat template: local file path or http(s) URL.")
@click.option(
    "--max-model-len",
    type=int,
    default=None,
    help="Max sequence length (vLLM derives it from the model when unset; levanter serves a 4k window).",
)
@click.option(
    "--max-num-batched-tokens",
    type=int,
    default=512,
    help="vLLM backend: prefill batch size; small values avoid TPU VMEM overflow.",
)
@click.option("--max-seqs", type=int, default=16, help="Levanter backend: concurrent sequence slots.")
@click.option("--page-size", type=int, default=128, help="Levanter backend: tokens per KV-cache page.")
@click.option(
    "--hbm-utilization",
    type=float,
    default=0.8,
    help="Levanter backend: fraction of device HBM the KV cache may claim.",
)
@click.option("--tensor-parallel-size", type=int, default=None, help="TP size (default: auto from heads + chips).")
@click.option("--dtype", default="bfloat16", help="Weight/compute dtype.")
@click.option("--cache-ttl-days", type=int, default=14, help="Mirror HF models to a TTL'd GCS cache (0 disables).")
@click.option(
    "--no-cache", is_flag=True, default=False, help="Skip the GCS model cache; always download from HuggingFace."
)
@click.option("--timeout-hours", type=float, default=24.0, help="Wall-clock lifetime before the server self-stops.")
@click.option(
    "--proxy-timeout",
    type=float,
    default=600.0,
    help="Seconds the controller proxy waits for a single completion before returning 504. "
    "Raise for long reasoning generations; the shorter proxy default cuts those off.",
)
@click.option(
    "--access",
    type=click.Choice(["private", "link"]),
    default="private",
    help="Proxy access. private: cluster identity only. link: mints a scoped capability "
    "URL anyone with the link can call off-cluster (printed once the model is ready).",
)
@click.option("--region", default=None, help="Comma-separated region(s) to pin the slice to.")
@click.option("--cpu", type=float, default=8.0)
@click.option("--memory", default="64g")
@click.option("--disk", default="100g")
@click.option("--max-retries-preemption", type=int, default=10)
@click.option("--vllm-arg", "vllm_args", multiple=True, help="Extra raw flag forwarded to `vllm serve` (repeatable).")
@click.option(
    "--vllm-version",
    default=DEFAULT_CUDA_VLLM_VERSION,
    help="CUDA vLLM version to provision in the isolated uv-tool env on the GPU path "
    "(ignored on the TPU path and when --task-image is set).",
)
@click.option(
    "--vllm-source",
    "vllm_source",
    type=click.Choice(["upstream", "marin-fork"]),
    default="upstream",
    show_default=True,
    help="GPU vLLM source: 'upstream' (stock PyPI vLLM at --vllm-version) or 'marin-fork' "
    "(Marin's vLLM fork; required for Marin-custom architectures like grug_moe).",
)
@click.option(
    "--extra",
    "extras",
    multiple=True,
    help="Extra dependency-group/extra to add to the worker environment (repeatable).",
)
@click.option(
    "--task-image",
    default=None,
    help="Override the task container image. On the GPU path this bypasses the isolated "
    "uv-tool vLLM and serves from the image's own `vllm` on PATH.",
)
@click.option("--wait/--no-wait", default=True, help="Hold the tunnel open until the endpoint is ready, then block.")
@click.option(
    "--wait-timeout",
    type=float,
    default=1800.0,
    help="Seconds the client waits for the endpoint to register; on --backend vllm it also bounds "
    "the in-job server boot.",
)
def main(
    model: str,
    backend: str,
    cluster: str | None,
    controller: str | None,
    tpu: str,
    gpu: str | None,
    target_cluster: str | None,
    isolated_vllm: bool,
    name: str | None,
    endpoint_name: str | None,
    chat_template: str | None,
    max_model_len: int | None,
    max_num_batched_tokens: int,
    max_seqs: int,
    page_size: int,
    hbm_utilization: float,
    tensor_parallel_size: int | None,
    dtype: str,
    cache_ttl_days: int,
    no_cache: bool,
    timeout_hours: float,
    proxy_timeout: float,
    access: str,
    region: str | None,
    cpu: float,
    memory: str,
    disk: str,
    max_retries_preemption: int,
    vllm_args: tuple[str, ...],
    vllm_version: str,
    vllm_source: str,
    extras: tuple[str, ...],
    task_image: str | None,
    wait: bool,
    wait_timeout: float,
) -> None:
    """Serve MODEL (an HF id or gs:// path) on an Iris TPU or GPU slice."""
    logging.basicConfig(level=logging.INFO, format="[marin-serve] %(message)s")

    # None outside a marin checkout: the TPU path then serves checkout-free.
    workspace_dir = find_project_root()

    tpu_from_cli = click.get_current_context().get_parameter_source("tpu") == ParameterSource.COMMANDLINE
    if gpu is not None and tpu_from_cli:
        raise click.ClickException("--gpu and --tpu are mutually exclusive; pass only one.")
    _reject_foreign_flags(backend, _VLLM_ONLY_OPTIONS if backend == "levanter" else _LEVANTER_ONLY_OPTIONS)

    job_name = name or _default_job_name(model)
    if "/" in job_name:
        raise click.ClickException("--name cannot contain '/'.")
    endpoint = endpoint_name or f"/serve/{job_name}"
    if not endpoint.startswith("/"):
        # The in-job registry prefixes a relative name with the job namespace, which
        # the client cannot then resolve; require an absolute name so the printed
        # proxy URL matches what actually registers.
        raise click.ClickException("--endpoint-name must be absolute (start with '/'), e.g. /serve/my-model.")
    if "." in endpoint:
        raise click.ClickException("--endpoint-name cannot contain '.' (it breaks controller proxy routing).")
    if proxy_timeout <= 0:
        raise click.ClickException("--proxy-timeout must be positive.")

    endpoint_access = EndpointAccess.Value(f"ENDPOINT_ACCESS_{access.upper()}")

    vllm_source_enum = VllmType.MARIN_FORK if vllm_source == "marin-fork" else VllmType.UPSTREAM
    plan = _resolve_serving_plan(
        backend=backend,
        tpu=tpu,
        gpu=gpu,
        in_checkout=workspace_dir is not None,
        isolated_vllm=isolated_vllm,
        task_image=task_image,
        cuda_vllm_version=vllm_version,
        vllm_source=vllm_source_enum,
        vllm=VllmBackend(
            max_num_batched_tokens=max_num_batched_tokens,
            # The in-job vLLM boot budget must cover the same window the client waits, so raising
            # --wait-timeout for a slow-booting model actually takes effect.
            startup_timeout_seconds=int(wait_timeout),
            extra_args=tuple(vllm_args),
        ),
        levanter=LevanterBackend(max_seqs=max_seqs, page_size=page_size, hbm_utilization=hbm_utilization),
        extras=extras,
    )

    config = QuickServeConfig(
        model=model,
        endpoint_name=endpoint,
        backend=plan.backend,
        tpu_type=plan.tpu_type,
        gpu_type=plan.gpu_type,
        gpu_count=plan.gpu_count,
        access=endpoint_access,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        chat_template_content=_resolve_chat_template(chat_template),
        cache_ttl_days=0 if no_cache else cache_ttl_days,
        timeout_hours=timeout_hours,
        proxy_timeout_seconds=proxy_timeout,
    )

    # No checkout to bundle → install marin-core from PyPI on the worker (checkout-free
    # TPU path); otherwise sync the bundled workspace with the resolved extras.
    if workspace_dir is None:
        environment = EnvironmentSpec(
            setup_scripts=[_checkout_free_setup_script(_marin_core_version(), plan.worker_extras)]
        )
    else:
        environment = EnvironmentSpec(extras=plan.worker_extras)

    constraints: list[Constraint] = []
    if region:
        regions = [r.strip() for r in region.split(",") if r.strip()]
        if regions:
            constraints.append(region_constraint(regions))
    if target_cluster:
        constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster))

    endpoint_cluster = cluster if controller is None else None
    with open_controller_endpoint(cluster_name=endpoint_cluster, controller_url=controller) as endpoint_info:
        controller_url = endpoint_info.url
        dashboard_url = endpoint_info.config.dashboard_url if endpoint_info.config else None
        click.echo(f"Using controller {controller_url}")
        with IrisClient.remote(controller_url, workspace=workspace_dir, credentials=endpoint_info.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(serve_in_job, config),
                name=job_name,
                resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk, device=plan.device),
                environment=environment,
                ports=["http"],
                constraints=constraints or None,
                max_retries_failure=0,
                max_retries_preemption=max_retries_preemption,
                task_image=task_image,
            )
            proxy_url = client.resolve_endpoint(endpoint)
            click.echo("")
            click.echo(f"  job          {job}")
            click.echo(f"  model        {model}")
            click.echo(f"  backend      {backend}")
            if gpu is not None:
                click.echo(f"  gpu          {config.accelerator_label}")
            else:
                click.echo(f"  tpu          {tpu}")
            if target_cluster:
                click.echo(f"  target       {target_cluster}")
            click.echo(f"  endpoint     {endpoint}")
            if dashboard_url:
                click.echo(f"  share url    {dashboard_url.rstrip('/')}{proxy_path(endpoint)}/")
            else:
                click.echo(f"  proxy path   {proxy_path(endpoint)}/")
            click.echo(f"  timeout      {timeout_hours:g}h")
            click.echo(f"  req timeout  {proxy_timeout:g}s  (per-request proxy budget)")
            if controller is None and cluster:
                click.echo(f"  stop with    iris --cluster {cluster} job stop {job}")
            else:
                click.echo(f"  stop with    iris --controller-url {controller_url} job stop {job}")
            click.echo("")

            if not wait:
                click.echo("Submitted. Open the dashboard from the Iris UI once the model has booted.")
                if endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK:
                    click.echo("Re-run with --wait once the server registers to mint the off-cluster capability URL.")
                return

            click.echo("Waiting for the model to load and register (Ctrl-C to detach; the job keeps running) …")
            _wait_for_endpoint(client, job, endpoint, wait_timeout)
            click.echo("")
            click.echo(f"READY — dashboard: {proxy_url}/")
            click.echo(f"        OpenAI:    {proxy_url}/v1")
            if dashboard_url:
                click.echo(f"        share:     {dashboard_url.rstrip('/')}{proxy_path(endpoint)}/")
            click.echo("")
            if endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK:
                # Mint after the endpoint registers (the controller resolves the row
                # for owner authz), so the token is bound to a live endpoint.
                _mint_and_print_capability_url(client, endpoint, dashboard_url, timeout_hours)
            click.echo("Tunnel held open; press Ctrl-C to detach (the server stays up on Iris).")
            with contextlib.suppress(KeyboardInterrupt):
                while True:
                    time.sleep(3600)
            click.echo("\nDetached. Reconnect from the Iris dashboard or re-run with --no-wait.")


if __name__ == "__main__":
    main()
