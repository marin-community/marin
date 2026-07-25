# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy CLI for always-on Iris service jobs.

The imperative half of :mod:`iac.iris.service`: Pulumi's ``local.Command`` invokes
these verbs with the :class:`~iac.iris.spec.ServiceSpec` JSON in ``$IRIS_SVC_SPEC``.
``up`` submits with RECREATE and probes readiness through the controller proxy;
``down`` terminates (idempotent, for ``pulumi destroy``).

All validation, secret resolution, and bundle checks run before the submit, so a bad
spec never terminates the healthy instance. Logs go to stderr; stdout carries exactly
one JSON document — the outputs Pulumi parses.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import click
import httpx
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf import json_format
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec, is_job_finished
from iris.rpc import job_pb2
from rigging.connect import proxy_path
from rigging.credentials import ClientCredentials
from rigging.secrets import resolve_secret_spec

from iac.iris.spec import SPEC_ENV_VAR, ServiceSpec

logger = logging.getLogger(__name__)

READY_POLL_INTERVAL = 5.0
# Bound on waiting for TerminateJob's cancel to drain to a terminal state on `down`.
TERMINATE_WAIT = 120.0


def load_spec(spec_file: str | None) -> ServiceSpec:
    """Read and validate the spec from ``--spec-file`` or ``$IRIS_SVC_SPEC``."""
    if spec_file is not None:
        text = Path(spec_file).read_text()
    else:
        text = os.environ.get(SPEC_ENV_VAR, "")
        if not text:
            raise click.UsageError(f"pass --spec-file or set {SPEC_ENV_VAR}")
    spec = ServiceSpec.from_json(text)
    spec.validate()
    return spec


def run_build_commands(workspace: Path, commands: tuple[str, ...]) -> None:
    """Run the spec's build commands, in order, from the workspace root.

    Runs before secret resolution and any cluster interaction, so a broken build
    aborts with the running instance untouched. Building inside every ``up`` (never
    conditionally) is what makes a stale or missing build output impossible to
    deploy. Command output goes to stderr to keep the stdout JSON contract.
    """
    for build_command in commands:
        logger.info("running build command: %s", build_command)
        result = subprocess.run(
            build_command, shell=True, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.returncode != 0:
            raise click.ClickException(f"build command failed (exit {result.returncode}): {build_command}")


def check_bundle_includes(workspace: Path, includes: tuple[str, ...]) -> None:
    """Fail when a bundle-include glob matches nothing — a build output is missing.

    Backstops ``run_build_commands``: without this, a build that silently produced
    nothing ships a bundle that lacks the artifact (the glob just matches zero files).
    """
    for pattern in includes:
        if not any(p.is_file() for p in workspace.glob(pattern)):
            raise click.ClickException(
                f"extra bundle include {pattern!r} matches no files under {workspace} — "
                "did the spec's build_commands produce the expected output?"
            )


def resolve_env(spec: ServiceSpec) -> dict[str, str]:
    """The task environment: plain values plus resolved secret references."""
    resolved = dict(spec.env)
    for name in sorted(spec.secret_env):
        resolved[name] = resolve_secret_spec(spec.secret_env[name]).value
    return resolved


def resources_from_spec(spec: ServiceSpec) -> ResourceSpec:
    proto = json_format.ParseDict(spec.resources, job_pb2.ResourceSpecProto())
    return ResourceSpec(
        cpu=proto.cpu_millicores / 1000.0,
        memory=proto.memory_bytes,
        disk=proto.disk_bytes,
        device=proto.device if proto.HasField("device") else None,
    )


def submit_service(client: IrisClient, spec: ServiceSpec, env_vars: dict[str, str]) -> Job:
    """Submit the always-on service job, replacing any running instance (RECREATE).

    The three retry budgets together make preemptions and container deaths
    non-terminal (see ALWAYS_ON_RETRIES in :mod:`iac.iris.spec`)."""
    return client.submit(
        entrypoint=Entrypoint.from_command(*spec.entrypoint),
        name=spec.name,
        user=spec.user,
        resources=resources_from_spec(spec),
        environment=EnvironmentSpec(
            env_vars=env_vars,
            pip_packages=spec.pip_packages,
            sync_packages=spec.sync_packages,
        ),
        ports=[spec.port],
        constraints=[region_constraint(list(spec.regions))],
        max_retries_preemption=spec.max_retries_preemption,
        max_retries_failure=spec.max_retries_failure,
        max_task_failures=spec.max_task_failures,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_RECREATE,
    )


def auth_headers(credentials: ClientCredentials | None) -> dict[str, str]:
    """HTTP headers for the readiness probe, mirroring the RPC interceptor chain."""
    headers: dict[str, str] = {}
    if credentials is None:
        return headers
    if credentials.token_provider is not None and (token := credentials.token_provider.get_token()):
        headers["Authorization"] = f"Bearer {token}"
    if credentials.iap_provider is not None and (token := credentials.iap_provider.get_token()):
        headers["Proxy-Authorization"] = f"Bearer {token}"
    return headers


def probe_ready(url: str, headers: dict[str, str], wait: int) -> bool:
    """Poll ``url`` until it answers 200 or ``wait`` seconds pass.

    A best-effort readiness signal: it proves the server booted and registered its
    endpoint, and catches bad-config boot crashes. It cannot distinguish a capacity
    stall from a slow boot, which is why expiry is a warning and never a failure —
    Iris's retry budgets converge the job once capacity frees.
    """
    deadline = time.monotonic() + wait
    last_status: object = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
            if response.status_code == 200:
                logger.info("service ready at %s", url)
                return True
            status: object = response.status_code
        except httpx.HTTPError as exc:
            status = type(exc).__name__
        if status != last_status:
            logger.info("waiting for %s (last: %s)", url, status)
            last_status = status
        time.sleep(READY_POLL_INTERVAL)
    logger.warning(
        "service did not answer 200 at %s within %ds; the job is submitted and Iris "
        "retries converge it once it can schedule — check the job in the dashboard",
        url,
        wait,
    )
    return False


def _print_outputs(job_id: JobName, url: str, ready: bool) -> None:
    # stdout purity: this is the single JSON document the Pulumi component parses.
    print(json.dumps({"job_id": str(job_id), "url": url, "ready": ready}, sort_keys=True))


def _deploy(spec: ServiceSpec) -> None:
    workspace = Path.cwd()
    run_build_commands(workspace, spec.build_commands)
    check_bundle_includes(workspace, spec.extra_bundle_includes)
    env_vars = resolve_env(spec)
    with connect_controller(cluster_name=spec.cluster) as endpoint:
        with IrisClient.remote(
            endpoint.url,
            workspace=workspace,
            credentials=endpoint.credentials,
            extra_bundle_includes=spec.extra_bundle_includes,
        ) as client:
            job = submit_service(client, spec, env_vars)
            service_url = endpoint.url.rstrip("/") + proxy_path(spec.endpoint)
            logger.info("submitted %s; probing %s", job.job_id, service_url)
            ready = probe_ready(
                service_url.rstrip("/") + spec.health_path,
                auth_headers(endpoint.credentials),
                spec.wait,
            )
            _print_outputs(job.job_id, service_url, ready)


@click.group()
def cli() -> None:
    """Deploy verbs for always-on Iris service jobs (driven by iac.iris.service)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)


@cli.command()
@click.option("--spec-file", default=None, help=f"Spec JSON path (default: ${SPEC_ENV_VAR}).")
def up(spec_file: str | None) -> None:
    """Submit the service, replacing a running instance (RECREATE)."""
    _deploy(load_spec(spec_file))


def terminate_service(client: IrisClient, job_id: JobName, *, wait: float = TERMINATE_WAIT) -> None:
    """Terminate ``job_id`` and wait (bounded) for a terminal state.

    A missing or already-terminal job is success, so ``pulumi destroy`` stays
    idempotent against out-of-band termination.
    """
    try:
        state = client.job_state(job_id)
    except ConnectError as exc:
        if exc.code == Code.NOT_FOUND:
            logger.info("job %s not found; nothing to terminate", job_id)
            return
        raise
    if is_job_finished(state):
        logger.info("job %s already terminal (%s)", job_id, job_pb2.JobState.Name(state))
        return
    client.terminate(job_id)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            if is_job_finished(client.job_state(job_id)):
                logger.info("job %s terminated", job_id)
                return
        except ConnectError as exc:
            if exc.code == Code.NOT_FOUND:
                return
            raise
        time.sleep(2.0)
    logger.warning("job %s not terminal after %ds; termination continues server-side", job_id, wait)


@cli.command()
@click.option("--spec-file", default=None, help=f"Spec JSON path (default: ${SPEC_ENV_VAR}).")
def down(spec_file: str | None) -> None:
    """Terminate the service job (idempotent; runs on resource removal and destroy)."""
    spec = load_spec(spec_file)
    with connect_controller(cluster_name=spec.cluster) as endpoint:
        with IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials) as client:
            terminate_service(client, JobName.root(spec.user, spec.name))


if __name__ == "__main__":
    cli()
