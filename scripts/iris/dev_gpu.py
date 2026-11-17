#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate and use development GPU (CoreWeave) pods on Iris-managed clusters."""

import getpass
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from iris.cli.job import validate_system_reason
from iris.client.client import JobAlreadyExists
from iris.cluster.config import load_config
from iris.cluster.platforms.k8s.coreweave_topology import NVL72_GPUS_PER_NODE, gpu_gang_coscheduling_level
from iris.cluster.redaction import redact_submit_argv
from iris.cluster.types import CoschedulingConfig, Entrypoint, JobName, ResourceSpec, gpu_device

from scripts.iris.coreweave_dev_pod import (
    HOLDER_COMMAND,
    PRIORITY_BANDS,
    CoreweaveTarget,
    PodRef,
    Priority,
    controller_client,
    is_job_active,
    kubectl_connect_cmd,
    release_holder_job,
    require_coreweave_platform,
    run_logged,
    state_path,
    wait_for_running_pod,
    wait_for_running_tasks,
)

logger = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_gpu_iris"
DEFAULT_GPU_VARIANT = "H100"

# GPUs on one whole node of a given variant, used as the default --gpus-per-node so a dev
# session grabs exactly one node. An H100 node is an 8-GPU DGX box; a GB200 node is a
# 4-GPU NVL72 compute tray (a rack/NVLink domain is 18 of them = 72 GPUs). These are the
# only variants the clusters we have access to expose.
GPUS_PER_NODE = {
    "H100": 8,
    "GB200": NVL72_GPUS_PER_NODE,
}

DEFAULT_CPU = 8.0
DEFAULT_MEMORY = "64GB"
DEFAULT_DISK = "100GB"


@dataclass(frozen=True)
class DevGpuState:
    """Persisted local state for an active dev GPU session. One pod per reserved node."""

    session_name: str
    config_file: str
    job_id: str
    gpus_per_node: int
    gpu_variant: str
    priority: Priority
    target: CoreweaveTarget
    pods: list[PodRef]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "DevGpuState":
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            gpus_per_node=data["gpus_per_node"],
            gpu_variant=data["gpu_variant"],
            priority=Priority(data["priority"]),
            target=CoreweaveTarget(**data["target"]),
            pods=[PodRef(**pod) for pod in data["pods"]],
        )


def resolve_coscheduling(gpu_variant: str, gpus_per_node: int, node_count: int) -> CoschedulingConfig | None:
    """Gang-scheduling level for a multi-node dev session, or None for a single node.

    Multi-node sessions must saturate each node's GPUs. Every level Iris derives — the
    NVLink-domain binds for GB200, soft ``leafgroup`` IB colocation for everything else —
    assumes one pod per node, and fractional pods would let two nodes of the session land
    on the same physical machine.
    """
    if node_count == 1:
        return None
    whole_node = GPUS_PER_NODE[gpu_variant]
    if gpus_per_node != whole_node:
        raise click.ClickException(
            f"--nodes {node_count} reserves whole nodes, so --gpus-per-node must be {whole_node} "
            f"for {gpu_variant}; got {gpus_per_node}."
        )
    try:
        level = gpu_gang_coscheduling_level(gpu_variant, gpus_per_node, node_count)
    except ValueError as exc:
        raise click.ClickException(f"--nodes {node_count} for {gpu_variant}: {exc}") from exc
    return CoschedulingConfig(group_by=level)


def session_summary(state: DevGpuState) -> str:
    """Render the session as one human-readable block, one line per reserved node."""
    lines = [
        f"Session: {state.session_name}",
        f"Job: {state.job_id}",
        f"Config: {state.config_file}",
        f"Nodes: {len(state.pods)} x {state.gpus_per_node} {state.gpu_variant}",
        f"Priority: {state.priority.value}",
    ]
    lines += [f"Node {index}: {pod.pod_name} (namespace={pod.namespace})" for index, pod in enumerate(state.pods)]
    return "\n".join(lines)


def load_state(path: Path) -> DevGpuState:
    if not path.exists():
        raise click.ClickException(f"No active dev GPU session at {path}")
    return DevGpuState.from_json(path.read_text())


def save_state(path: Path, state: DevGpuState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


@dataclass
class Context:
    config_file: str | None = None
    session_name: str | None = None
    state_dir: Path = STATE_DIR


@click.group()
@click.option("--config", help="Path to an Iris cluster config file.")
@click.option("--name", "session_name", help="Local dev GPU session name.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx, config: str | None, session_name: str | None, verbose: bool) -> None:
    """Development GPU (CoreWeave) pod management for Iris clusters."""
    ctx.ensure_object(Context)
    ctx.obj.config_file = str(Path(config).resolve()) if config else None
    ctx.obj.session_name = session_name or getpass.getuser()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command("allocate")
@click.option(
    "--gpu-variant",
    type=click.Choice(sorted(GPUS_PER_NODE), case_sensitive=False),
    default=DEFAULT_GPU_VARIANT,
    show_default=True,
    help="GPU variant to reserve. Passed through to Iris, which schedules onto the matching node pool.",
)
@click.option(
    "--gpus-per-node",
    type=click.IntRange(min=1),
    default=None,
    help="GPUs to reserve on each node. Defaults to one whole node of the chosen "
    "--gpu-variant (H100: 8, GB200: 4). Whole-node counts are what the bare-metal pools "
    "validate; smaller values may not schedule, and --nodes > 1 requires the whole-node count.",
)
@click.option(
    "--nodes",
    "node_count",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Whole nodes to reserve in a single session. More than one gang-schedules them "
    "together: a GB200 gang of up to 16 nodes binds hard to a single NVLink domain and a "
    "larger one splits into balanced per-rack slices, while other variants get soft "
    "InfiniBand leafgroup colocation. Every node is a separate pod; connect with --node.",
)
@click.option(
    "--priority",
    type=click.Choice([priority.value for priority in Priority]),
    default=Priority.INTERACTIVE.value,
    show_default=True,
    help="Iris scheduling priority for the holder job.",
)
@click.option(
    "--system-reason",
    type=str,
    default=None,
    metavar="REASON",
    help="Required justification for --priority system; must contain hero, finelog, or iris.",
)
@click.option(
    "--cpu",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_CPU,
    show_default=True,
    help="CPU cores to request for the dev pod (request only, burstable).",
)
@click.option("--task-image", default=None, help="Pin the task container image instead of the cluster default.")
@click.option(
    "--memory",
    default=DEFAULT_MEMORY,
    show_default=True,
    help="Memory for the dev pod (e.g. 64GB). Enforced as a hard k8s limit.",
)
@click.option(
    "--disk",
    default=DEFAULT_DISK,
    show_default=True,
    help="Ephemeral disk for the dev pod (e.g. 100GB). Enforced as a hard k8s limit.",
)
@click.option("--timeout", default=900, show_default=True, help="Seconds to wait for the task to run.")
@click.option("--pod-timeout", default=120, show_default=True, help="Seconds to wait for the pod to run.")
@click.pass_context
def allocate(
    ctx,
    gpu_variant: str,
    gpus_per_node: int | None,
    node_count: int,
    priority: str,
    system_reason: str | None,
    cpu: float,
    task_image: str | None,
    memory: str,
    disk: str,
    timeout: int,
    pod_timeout: int,
) -> None:
    """Allocate one or more dev GPU nodes and hold them until Ctrl-C."""
    if not ctx.obj.config_file:
        raise click.ClickException("--config is required")

    # click.Choice(case_sensitive=False) already normalized gpu_variant to a canonical key
    # of GPUS_PER_NODE, so a bare --gpus-per-node defaults to that variant's whole node.
    if gpus_per_node is None:
        gpus_per_node = GPUS_PER_NODE[gpu_variant]
    coscheduling = resolve_coscheduling(gpu_variant, gpus_per_node, node_count)
    resolved_priority = Priority(priority)
    validate_system_reason(priority, system_reason)

    session_name = ctx.obj.session_name
    state_file = state_path(ctx.obj.state_dir, session_name)
    if state_file.exists():
        raise click.ClickException(
            f"Dev GPU session '{session_name}' already exists. Use release first or choose a new --name."
        )

    target = require_coreweave_platform(load_config(ctx.obj.config_file))

    state: DevGpuState | None = None
    with controller_client(ctx.obj.config_file) as client:
        resources = ResourceSpec(cpu=cpu, memory=memory, disk=disk, device=gpu_device(gpu_variant, gpus_per_node))
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
                name=f"dev-gpu-{session_name}",
                resources=resources,
                priority_band=PRIORITY_BANDS[resolved_priority],
                submit_argv=redact_submit_argv(list(sys.argv)),
                replicas=node_count,
                coscheduling=coscheduling,
                task_image=task_image,
            )
        except JobAlreadyExists as exc:
            raise click.ClickException(f"Job already exists for session '{session_name}': {exc}") from exc

        try:
            task_ids = wait_for_running_tasks(job, node_count=node_count, timeout=timeout)
            state = DevGpuState(
                session_name=session_name,
                config_file=ctx.obj.config_file,
                job_id=str(job.job_id),
                gpus_per_node=gpus_per_node,
                gpu_variant=gpu_variant,
                priority=resolved_priority,
                target=target,
                pods=[wait_for_running_pod(target, task_id, timeout=pod_timeout) for task_id in task_ids],
            )
            save_state(state_file, state)

            print(session_summary(state))
            if coscheduling is not None:
                print(f"Coscheduling: {coscheduling.group_by}")
            print("\nAllocation is active. Press Ctrl-C to release.")

            while True:
                time.sleep(30)
                if not is_job_active(client, str(job.job_id)):
                    raise click.ClickException("The dev GPU holder job terminated unexpectedly.")
        except KeyboardInterrupt:
            print("\nReleasing dev GPU session...")
        finally:
            terminated = False
            try:
                client.cancel_job(JobName.from_wire(str(job.job_id)))
                terminated = True
            except Exception:
                logger.warning(
                    "Failed to terminate holder job %s; the GPU pod may still be running. "
                    "Keeping local session state — run `release` to retry cleanup.",
                    job.job_id,
                    exc_info=True,
                )
            # Only drop the session file once the job is actually gone, so a failed
            # terminate never orphans an expensive pod with no local record of its job id.
            if state is not None and terminated:
                state_file.unlink(missing_ok=True)


@cli.command("connect")
@click.option(
    "--node",
    "node_index",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Which node of the session to shell into, in `status` pod order.",
)
@click.pass_context
def connect(ctx, node_index: int) -> None:
    """Open an interactive shell into one of the reserved pods."""
    state = load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))
    if node_index >= len(state.pods):
        raise click.ClickException(
            f"Dev GPU session '{state.session_name}' has {len(state.pods)} node(s); "
            f"--node {node_index} is out of range."
        )
    # connect fundamentally only needs `kubectl exec`. The controller liveness check is a
    # courtesy: if the controller is reachable and reports the job inactive we fail fast,
    # but if reaching the controller itself fails we proceed, since a healthy pod should
    # still be reachable when the controller is momentarily unreachable.
    try:
        with controller_client(state.config_file) as client:
            job_active = is_job_active(client, state.job_id)
    except Exception:
        logger.warning(
            "Could not verify dev GPU job liveness with the controller; attempting to connect anyway.",
            exc_info=True,
        )
    else:
        if not job_active:
            raise click.ClickException(
                f"Dev GPU session '{state.session_name}' is no longer active. Use release to clean up."
            )
    # These are the pods resolved at allocation time. If Iris rescheduled the task
    # onto a new pod while the job stayed active, this kubectl exec fails; re-allocate.
    run_logged(kubectl_connect_cmd(state.target, state.pods[node_index]), check=True)


@cli.command("status")
@click.pass_context
def status(ctx) -> None:
    """Show the current session state."""
    print(session_summary(load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))))


@cli.command("release")
@click.option("--force", is_flag=True, help="Delete the local session file even if terminating the holder job fails.")
@click.pass_context
def release(ctx, force: bool) -> None:
    """Terminate the holder job and clear the local session file."""
    state_file = state_path(ctx.obj.state_dir, ctx.obj.session_name)
    state = load_state(state_file)
    release_holder_job(state.config_file, state.job_id, state_file, force=force)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
