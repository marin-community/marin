#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hold a privileged CPU dev box on a CoreWeave Iris cluster for interactive Linux debugging.

The box is one non-preemptible task under ``CONTAINER_PROFILE_PRIVILEGED``, so bubblewrap sandboxes,
in-pod Docker and grader liveness checks behave as they do on curation workers. ``allocate`` returns
once the pod runs; the holder job ends at ``--max-lifetime`` or on ``release``. The holder never
retries, so ``--max-lifetime`` is a hard limit and a lost pod ends the box; allocate a new one.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import click
from iris.cli.job import validate_system_reason
from iris.client.client import IrisClient, Job, JobAlreadyExists
from iris.cluster.config import load_config
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, JobName, ResourceSpec
from iris.rpc import job_pb2
from rigging.redaction import redact_env_argv
from rigging.timing import Duration

from scripts.iris.coreweave_dev_pod import (
    HOLDER_COMMAND,
    PRIORITY_BANDS,
    CoreweaveTarget,
    PodRef,
    Priority,
    controller_client,
    is_job_active,
    kubectl_connect_cmd,
    kubectl_copy_from_cmd,
    kubectl_copy_to_cmd,
    kubectl_exec_cmd,
    release_holder_job,
    require_coreweave_platform,
    run_logged,
    state_path,
    wait_for_running_pod,
    wait_for_running_tasks,
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_cpu_iris"
POD_RESOLVE_TIMEOUT = 120


logger = logging.getLogger(__name__)


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

    def submit(self, client: IrisClient) -> Job:
        return client.submit(
            entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
            name=self.job_name,
            resources=self.resources,
            constraints=[preemptible_constraint(False)],
            timeout=Duration.from_seconds(self.max_lifetime),
            priority_band=PRIORITY_BANDS[self.priority],
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            # A retry would land on an empty pod and restart the lifetime clock.
            max_retries_preemption=0,
            submit_argv=redact_env_argv(sys.argv),
        )


@dataclass(frozen=True)
class DevCpuState:
    """Persisted local state for an active dev CPU box."""

    session_name: str
    config_file: str
    job_id: str
    task_id: str
    priority: Priority
    target: CoreweaveTarget

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "DevCpuState":
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            task_id=data["task_id"],
            priority=Priority(data["priority"]),
            target=CoreweaveTarget(**data["target"]),
        )


def save_state(state_dir: Path, state: DevCpuState) -> None:
    path = state_path(state_dir, state.session_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


def resolve_pod(state: DevCpuState, *, timeout: int = POD_RESOLVE_TIMEOUT) -> PodRef:
    """Resolve the holder task's Running pod."""
    return wait_for_running_pod(state.target, state.task_id, timeout=timeout)


def job_is_active(state: DevCpuState) -> bool:
    """Ask the controller whether the box's holder job is still running."""
    with controller_client(state.config_file) as client:
        return is_job_active(client, state.job_id)


def session_summary(state: DevCpuState, pod: PodRef | None) -> str:
    return "\n".join(
        [
            f"Session: {state.session_name}",
            f"Job: {state.job_id}",
            f"Config: {state.config_file}",
            f"Priority: {state.priority.value}",
            f"Pod: {pod.pod_name} (namespace={pod.namespace}, container={pod.container})" if pod else "Pod: none",
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
    state_dir: Path = field(default_factory=lambda: STATE_DIR)

    def state(self) -> DevCpuState:
        return load_state(state_path(self.state_dir, self.session_name))


def load_active_pod(obj: Context) -> tuple[DevCpuState, PodRef]:
    """Resolve the box's pod only while its holder job is still active."""
    state = obj.state()
    if not job_is_active(state):
        raise click.ClickException(f"Dev CPU box '{state.session_name}' has ended; release it")
    if not state.task_id:
        raise click.ClickException(f"Dev CPU box '{state.session_name}' has not started its task yet")
    return state, resolve_pod(state)


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
@click.option(
    "--pod-timeout", default=POD_RESOLVE_TIMEOUT, show_default=True, help="Seconds to wait for the pod to run."
)
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
            raise click.ClickException(
                f"A holder job already exists for dev CPU box '{obj.session_name}' ({holder.job_name})."
                f" If its session file is gone, stop it with: uv run iris job cancel {holder.job_name}"
            ) from exc
        # The job id exists from submission on, so the state file is written
        # immediately: an interrupted allocate always leaves a recoverable box.
        state = DevCpuState(
            session_name=obj.session_name,
            config_file=obj.config_file,
            job_id=str(job.job_id),
            task_id="",
            priority=holder.priority,
            target=target,
        )
        save_state(obj.state_dir, state)
        try:
            (task_id,) = wait_for_running_tasks(job, node_count=1, timeout=timeout)
            pod = wait_for_running_pod(target, task_id, timeout=pod_timeout)
        except BaseException:
            try:
                client.cancel_job(JobName.from_wire(str(job.job_id)))
            except Exception:
                # The holder may still run; keep the state file so `release` can stop it.
                logger.warning(
                    "Failed to cancel holder job %s after a failed allocation; the box may still hold its"
                    " resources. Run `release` to stop it.",
                    job.job_id,
                    exc_info=True,
                )
                raise
            state_file.unlink(missing_ok=True)
            raise
    state = replace(state, task_id=task_id)
    save_state(obj.state_dir, state)
    print(session_summary(state, pod))


@cli.command("exec", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
@click.pass_obj
def exec_command(obj: Context, command: tuple[str, ...]) -> None:
    """Run one command in the box and exit with its status."""
    state, pod = load_active_pod(obj)
    raise SystemExit(run_logged(kubectl_exec_cmd(state.target, pod, command)).returncode)


@cli.command("shell")
@click.pass_obj
def shell(obj: Context) -> None:
    """Open an interactive login shell in the box."""
    state, pod = load_active_pod(obj)
    run_logged(kubectl_connect_cmd(state.target, pod), check=True)


@cli.command("cp-to")
@click.argument("local")
@click.argument("remote")
@click.pass_obj
def copy_to(obj: Context, local: str, remote: str) -> None:
    """Copy a local file or directory into the box."""
    state, pod = load_active_pod(obj)
    run_logged(kubectl_copy_to_cmd(state.target, pod, local, remote), check=True)


@cli.command("cp-from")
@click.argument("remote")
@click.argument("local")
@click.pass_obj
def copy_from(obj: Context, remote: str, local: str) -> None:
    """Copy a file or directory out of the box."""
    state, pod = load_active_pod(obj)
    run_logged(kubectl_copy_from_cmd(state.target, pod, remote, local), check=True)


@cli.command("status")
@click.pass_obj
def status(obj: Context) -> None:
    """Show the session and whether its holder job is still active."""
    state = obj.state()
    active = job_is_active(state)
    pod = resolve_pod(state) if active and state.task_id else None
    print(session_summary(state, pod))
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
