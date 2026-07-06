# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``marin-serve`` — one-liner to serve an HF model on an Iris TPU slice.

Submits a single Iris job that boots vLLM on a single-host TPU slice and registers
a browser dashboard + OpenAI-compatible endpoint through the controller proxy. The
job stops itself after ``--timeout-hours`` so a forgotten server frees its slice.

Examples::

    marin-serve Qwen/Qwen3-0.6B --cluster marin --tpu v6e-8
    marin-serve gs://my-bucket/ckpt --tpu v5litepod-8 --chat-template delphi_v0.jinja2

The TPU's tensor-parallel size and (for clamped-RoPE models) max sequence length are
inferred automatically; override with ``--tensor-parallel-size`` / ``--max-model-len``.
"""

import contextlib
import logging
import re
import time
import uuid
from pathlib import Path

import click
import requests
from iris.cli.connect import open_controller_endpoint
from iris.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import EndpointAccess, Entrypoint, EnvironmentSpec, ResourceSpec, is_job_finished, tpu_device
from rigging.connect import capability_path, proxy_path
from rigging.timing import Duration

from marin.inference.quick_serve import QuickServeConfig, serve_in_job

logger = logging.getLogger(__name__)

# vLLM and the dashboard need the generic TPU stack plus the TPU-vLLM runtime.
_WORKER_EXTRAS = ("tpu", "vllm")
_ENDPOINT_READY_POLL_SECONDS = 5.0


def _default_job_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")[:24]
    suffix = uuid.uuid4().hex[:6]
    return f"serve-{slug}-{suffix}" if slug else f"serve-{suffix}"


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
        "The job is still booting vLLM; re-check later via the Iris dashboard."
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
@click.option("--cluster", default="marin", envvar="IRIS_CLUSTER", help="Named iris cluster to submit to.")
@click.option(
    "--controller", default=None, envvar="IRIS_CONTROLLER", help="Pre-tunneled controller URL (overrides --cluster)."
)
@click.option("--tpu", default="v6e-8", help="Single-host TPU slice type (e.g. v6e-8, v5litepod-8).")
@click.option("--name", default=None, help="Iris job name (default: derived from the model).")
@click.option("--endpoint-name", default=None, help="Endpoint name to register (default: /serve/<job-name>).")
@click.option("--chat-template", default=None, help="Jinja chat template: local file path or http(s) URL.")
@click.option("--max-model-len", type=int, default=None, help="vLLM max sequence length (default: derived from model).")
@click.option(
    "--max-num-batched-tokens", type=int, default=512, help="Prefill batch size; small values avoid TPU VMEM overflow."
)
@click.option("--tensor-parallel-size", type=int, default=None, help="TP size (default: auto from heads + chips).")
@click.option("--dtype", default="bfloat16", help="vLLM dtype.")
@click.option("--cache-ttl-days", type=int, default=14, help="Mirror HF models to a TTL'd GCS cache (0 disables).")
@click.option(
    "--no-cache", is_flag=True, default=False, help="Skip the GCS model cache; always download from HuggingFace."
)
@click.option("--timeout-hours", type=float, default=24.0, help="Wall-clock lifetime before the server self-stops.")
@click.option(
    "--access",
    type=click.Choice(["private", "link"]),
    default="private",
    help="Proxy access. private: cluster identity only. link: mints a scoped capability "
    "URL anyone with the link can call off-cluster (printed once vLLM is ready).",
)
@click.option("--region", default=None, help="Comma-separated region(s) to pin the slice to.")
@click.option("--cpu", type=float, default=8.0)
@click.option("--memory", default="64g")
@click.option("--disk", default="100g")
@click.option("--max-retries-preemption", type=int, default=10)
@click.option("--vllm-arg", "vllm_args", multiple=True, help="Extra raw flag forwarded to `vllm serve` (repeatable).")
@click.option("--wait/--no-wait", default=True, help="Hold the tunnel open until the endpoint is ready, then block.")
@click.option(
    "--wait-timeout",
    type=float,
    default=1800.0,
    help="Seconds allowed for vLLM to boot; bounds both the client wait and the in-job startup.",
)
def main(
    model: str,
    cluster: str | None,
    controller: str | None,
    tpu: str,
    name: str | None,
    endpoint_name: str | None,
    chat_template: str | None,
    max_model_len: int | None,
    max_num_batched_tokens: int,
    tensor_parallel_size: int | None,
    dtype: str,
    cache_ttl_days: int,
    no_cache: bool,
    timeout_hours: float,
    access: str,
    region: str | None,
    cpu: float,
    memory: str,
    disk: str,
    max_retries_preemption: int,
    vllm_args: tuple[str, ...],
    wait: bool,
    wait_timeout: float,
) -> None:
    """Serve MODEL (an HF id or gs:// path) on an Iris TPU slice."""
    logging.basicConfig(level=logging.INFO, format="[marin-serve] %(message)s")

    topology = get_tpu_topology(tpu)
    if topology.vm_count != 1:
        raise click.ClickException(
            f"{tpu!r} is a multi-host slice (vm_count={topology.vm_count}); quick-serve supports "
            f"single-host slices only (e.g. v6e-8, v5litepod-8)."
        )

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

    endpoint_access = EndpointAccess.Value(f"ENDPOINT_ACCESS_{access.upper()}")

    config = QuickServeConfig(
        model=model,
        tpu_type=tpu,
        endpoint_name=endpoint,
        access=endpoint_access,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        tensor_parallel_size=tensor_parallel_size,
        chat_template_content=_resolve_chat_template(chat_template),
        cache_ttl_days=0 if no_cache else cache_ttl_days,
        timeout_hours=timeout_hours,
        # The in-job vLLM startup budget must cover the same window the client waits,
        # so raising --wait-timeout for a slow-booting model actually takes effect.
        vllm_startup_timeout_seconds=int(wait_timeout),
        extra_vllm_args=tuple(vllm_args),
    )

    constraints = None
    if region:
        regions = [r.strip() for r in region.split(",") if r.strip()]
        if regions:
            constraints = [region_constraint(regions)]

    endpoint_cluster = cluster if controller is None else None
    with open_controller_endpoint(cluster_name=endpoint_cluster, controller_url=controller) as endpoint_info:
        controller_url = endpoint_info.url
        dashboard_url = endpoint_info.config.dashboard_url if endpoint_info.config else None
        click.echo(f"Using controller {controller_url}")
        with IrisClient.remote(controller_url, workspace=Path.cwd(), credentials=endpoint_info.credentials) as client:
            job = client.submit(
                entrypoint=Entrypoint.from_callable(serve_in_job, config),
                name=job_name,
                resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk, device=tpu_device(tpu)),
                environment=EnvironmentSpec(extras=_WORKER_EXTRAS),
                ports=["http"],
                constraints=constraints,
                max_retries_failure=0,
                max_retries_preemption=max_retries_preemption,
            )
            proxy_url = client.resolve_endpoint(endpoint)
            click.echo("")
            click.echo(f"  job          {job}")
            click.echo(f"  model        {model}")
            click.echo(f"  tpu          {tpu}")
            click.echo(f"  endpoint     {endpoint}")
            if dashboard_url:
                click.echo(f"  share url    {dashboard_url.rstrip('/')}{proxy_path(endpoint)}/")
            else:
                click.echo(f"  proxy path   {proxy_path(endpoint)}/")
            click.echo(f"  timeout      {timeout_hours:g}h")
            if controller is None and cluster:
                click.echo(f"  stop with    iris --cluster {cluster} job stop {job}")
            else:
                click.echo(f"  stop with    iris --controller-url {controller_url} job stop {job}")
            click.echo("")

            if not wait:
                click.echo("Submitted. Open the dashboard from the Iris UI once vLLM has booted.")
                if endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK:
                    click.echo("Re-run with --wait once vLLM registers to mint the off-cluster capability URL.")
                return

            click.echo("Waiting for vLLM to boot and register (Ctrl-C to detach; the job keeps running) …")
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
