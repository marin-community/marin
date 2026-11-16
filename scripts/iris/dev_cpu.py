#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hold a privileged CPU dev box on a CoreWeave Iris cluster for interactive Linux debugging.

The box is one non-preemptible task under ``CONTAINER_PROFILE_PRIVILEGED``, so bubblewrap sandboxes,
in-pod Docker and grader liveness checks behave as they do on curation workers. ``allocate`` returns
once the pod runs; the holder job ends at ``--max-lifetime`` or on ``release``.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from iris.cli.job import validate_system_reason
from iris.client.client import IrisClient, Job, JobAlreadyExists
from iris.cluster.config import load_config
from iris.cluster.constraints import Constraint, preemptible_constraint
from iris.cluster.redaction import redact_submit_argv
from iris.cluster.types import Entrypoint, JobName, ResourceSpec
from iris.rpc import job_pb2
from rigging.timing import Duration

from scripts.iris.dev_gpu import (
    HOLDER_COMMAND,
    PRIORITY_BANDS,
    CoreweaveTarget,
    PodRef,
    Priority,
    controller_client,
    is_job_active,
    kubectl_base,
    kubectl_connect_cmd,
    release_holder_job,
    require_coreweave_platform,
    run_logged,
    state_path,
    wait_for_running_pod,
    wait_for_running_tasks,
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_cpu_iris"


@dataclass(frozen=True)
class HolderJob:
    """The Iris submission that holds a dev box: a sleeping, privileged, non-preemptible CPU task."""

    name: str
    resources: ResourceSpec
    priority: Priority
    max_lifetime: int

    @property
    def job_name(self) -> str:
        return f"dev-cpu-{self.name}"

    def constraints(self) -> list[Constraint]:
        return [preemptible_constraint(False)]

    def submit(self, client: IrisClient) -> Job:
        return client.submit(
            entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
            name=self.job_name,
            resources=self.resources,
            constraints=self.constraints(),
            timeout=Duration.from_seconds(self.max_lifetime),
            priority_band=PRIORITY_BANDS[self.priority],
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            submit_argv=redact_submit_argv(list(sys.argv)),
        )


@dataclass(frozen=True)
class DevCpuState:
    """Persisted local state for an active dev CPU box."""

    session_name: str
    config_file: str
    job_id: str
    priority: Priority
    target: CoreweaveTarget
    pod: PodRef

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "DevCpuState":
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            priority=Priority(data["priority"]),
            target=CoreweaveTarget(**data["target"]),
            pod=PodRef(**data["pod"]),
        )


def kubectl_exec_cmd(target: CoreweaveTarget, pod: PodRef, command: tuple[str, ...]) -> list[str]:
    return [*kubectl_base(target), "exec", pod.pod_name, "-c", pod.container, "--", *command]


def kubectl_copy_to_cmd(target: CoreweaveTarget, pod: PodRef, local: str, remote: str) -> list[str]:
    return [*kubectl_base(target), "cp", "-c", pod.container, local, f"{pod.pod_name}:{remote}"]


def kubectl_copy_from_cmd(target: CoreweaveTarget, pod: PodRef, remote: str, local: str) -> list[str]:
    return [*kubectl_base(target), "cp", "-c", pod.container, f"{pod.pod_name}:{remote}", local]


def session_summary(state: DevCpuState) -> str:
    return "\n".join(
        [
            f"Session: {state.session_name}",
            f"Job: {state.job_id}",
            f"Config: {state.config_file}",
            f"Priority: {state.priority.value}",
            f"Pod: {state.pod.pod_name} (namespace={state.pod.namespace}, container={state.pod.container})",
        ]
    )


def load_state(path: Path) -> DevCpuState:
    if not path.exists():
        raise click.ClickException(f"No active dev CPU box at {path}")
    return DevCpuState.from_json(path.read_text())


@dataclass
class Context:
    config_file: str | None
    session_name: str
    state_dir: Path = STATE_DIR

    def state(self) -> DevCpuState:
        return load_state(state_path(self.state_dir, self.session_name))


@click.group()
@click.option("--config", help="Path to an Iris cluster config file; required for allocate.")
@click.option("--name", "session_name", required=True, help="Dev box name; also names the holder job.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx, config: str | None, session_name: str, verbose: bool) -> None:
    """Privileged CPU dev box management for CoreWeave Iris clusters."""
    ctx.obj = Context(config_file=str(Path(config).resolve()) if config else None, session_name=session_name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command("allocate")
@click.option("--cpu", type=click.FloatRange(min=0, min_open=True), required=True, help="CPU cores, e.g. 16.")
@click.option("--memory", required=True, help="Memory, e.g. 64GB. Enforced as a hard k8s limit.")
@click.option("--disk", required=True, help="Ephemeral disk, e.g. 200GB. Enforced as a hard k8s limit.")
@click.option("--priority", type=click.Choice([priority.value for priority in Priority]), required=True)
@click.option(
    "--system-reason",
    metavar="REASON",
    help="Required justification for --priority system; must contain hero, finelog, or iris.",
)
@click.option(
    "--max-lifetime",
    type=click.IntRange(min=1),
    required=True,
    help="Seconds before Iris ends the holder job, e.g. 259200 for three days.",
)
@click.option("--timeout", default=900, show_default=True, help="Seconds to wait for the task to run.")
@click.option("--pod-timeout", default=120, show_default=True, help="Seconds to wait for the pod to run.")
@click.pass_obj
def allocate(
    obj: Context,
    cpu: float,
    memory: str,
    disk: str,
    priority: str,
    system_reason: str | None,
    max_lifetime: int,
    timeout: int,
    pod_timeout: int,
) -> None:
    """Submit the holder job, wait for its pod and record the session."""
    if not obj.config_file:
        raise click.ClickException("--config is required")
    validate_system_reason(priority, system_reason)
    state_file = state_path(obj.state_dir, obj.session_name)
    if state_file.exists():
        raise click.ClickException(
            f"Dev CPU box '{obj.session_name}' already exists. Release it or choose a new --name."
        )
    target = require_coreweave_platform(load_config(obj.config_file))
    holder = HolderJob(
        name=obj.session_name,
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        priority=Priority(priority),
        max_lifetime=max_lifetime,
    )
    with controller_client(obj.config_file) as client:
        try:
            job = holder.submit(client)
        except JobAlreadyExists as exc:
            raise click.ClickException(f"Job already exists for dev CPU box '{obj.session_name}': {exc}") from exc
        try:
            (task_id,) = wait_for_running_tasks(job, node_count=1, timeout=timeout)
            pod = wait_for_running_pod(target, task_id, timeout=pod_timeout)
        except BaseException:
            client.cancel_job(JobName.from_wire(str(job.job_id)))
            raise
    state = DevCpuState(
        session_name=obj.session_name,
        config_file=obj.config_file,
        job_id=str(job.job_id),
        priority=holder.priority,
        target=target,
        pod=pod,
    )
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(state.to_json())
    print(session_summary(state))


@cli.command("exec", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.pass_obj
def exec_command(obj: Context, command: tuple[str, ...]) -> None:
    """Run one command in the box and exit with its status."""
    state = obj.state()
    raise SystemExit(run_logged(kubectl_exec_cmd(state.target, state.pod, command)).returncode)


@cli.command("shell")
@click.pass_obj
def shell(obj: Context) -> None:
    """Open an interactive login shell in the box."""
    state = obj.state()
    run_logged(kubectl_connect_cmd(state.target, state.pod), check=True)


@cli.command("cp-to")
@click.argument("local")
@click.argument("remote")
@click.pass_obj
def copy_to(obj: Context, local: str, remote: str) -> None:
    """Copy a local file or directory into the box."""
    state = obj.state()
    run_logged(kubectl_copy_to_cmd(state.target, state.pod, local, remote), check=True)


@cli.command("cp-from")
@click.argument("remote")
@click.argument("local")
@click.pass_obj
def copy_from(obj: Context, remote: str, local: str) -> None:
    """Copy a file or directory out of the box."""
    state = obj.state()
    run_logged(kubectl_copy_from_cmd(state.target, state.pod, remote, local), check=True)


@cli.command("status")
@click.pass_obj
def status(obj: Context) -> None:
    """Show the session and whether its holder job is still active."""
    state = obj.state()
    with controller_client(state.config_file) as client:
        active = is_job_active(client, state.job_id)
    print(session_summary(state))
    print(f"Active: {active}")


@cli.command("release")
@click.option("--force", is_flag=True, help="Delete the local session file even if terminating the holder job fails.")
@click.pass_obj
def release(obj: Context, force: bool) -> None:
    """Terminate the holder job and clear the local session file."""
    state_file = state_path(obj.state_dir, obj.session_name)
    state = load_state(state_file)
    release_holder_job(state.config_file, state.job_id, state_file, force=force)


if __name__ == "__main__":
    cli()
