#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate and use development GPU (CoreWeave H100) pods on Iris-managed clusters."""

import getpass
import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import click
from iris.client import IrisClient, JobAlreadyExists
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value
from iris.cluster.composer import provider_bundle
from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, gpu_device
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

HOLDER_COMMAND = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
    "print('iris dev gpu holder ready', flush=True); "
    "time.sleep(365 * 24 * 60 * 60)"
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_gpu_iris"
DEFAULT_GPU_COUNT = 8
TASK_CONTAINER = "task"
GPU_VARIANT = "H100"

DEFAULT_CPU = 8.0
DEFAULT_MEMORY = "64GB"
DEFAULT_DISK = "100GB"

TERMINAL_JOB_STATES = {
    job_pb2.JOB_STATE_FAILED,
    job_pb2.JOB_STATE_KILLED,
    job_pb2.JOB_STATE_UNSCHEDULABLE,
    job_pb2.JOB_STATE_WORKER_FAILED,
}
INACTIVE_JOB_STATES = TERMINAL_JOB_STATES | {job_pb2.JOB_STATE_SUCCEEDED}


@dataclass(frozen=True)
class PodRef:
    """The k8s pod backing a dev GPU session."""

    namespace: str
    pod_name: str
    container: str = TASK_CONTAINER


@dataclass(frozen=True)
class CoreweaveTarget:
    """Cluster-level kubectl target. Empty kubeconfig_path => kubectl default resolution;
    empty kube_context => the file's current-context."""

    namespace: str
    kubeconfig_path: str
    kube_context: str = ""


@dataclass(frozen=True)
class DevGpuState:
    """Persisted local state for an active dev GPU session."""

    session_name: str
    config_file: str
    job_id: str
    gpu_count: int
    target: CoreweaveTarget
    pod: PodRef

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "DevGpuState":
        data = json.loads(raw)
        return cls(
            session_name=data["session_name"],
            config_file=data["config_file"],
            job_id=data["job_id"],
            gpu_count=data["gpu_count"],
            target=CoreweaveTarget(**data["target"]),
            pod=PodRef(**data["pod"]),
        )


def require_coreweave_platform(config: IrisClusterConfig) -> CoreweaveTarget:
    """Resolve the kubectl target for a CoreWeave cluster, or fail fast.

    Inverts dev_tpu.py's GCP gate: this tool only works against
    CoreWeave/Kubernetes-backed clusters.
    """
    if config.platform.platform_kind() != "coreweave":
        raise click.ClickException(
            "dev_gpu requires a CoreWeave/Kubernetes-backed cluster. "
            "For GCP TPU clusters use scripts/iris/dev_tpu.py."
        )
    # Namespace must come from kubernetes_provider: that is where Iris actually
    # creates and lists task pods (defaulting to "iris" when unset).
    # Kubeconfig comes from platform.coreweave: that is the operator's laptop
    # kubeconfig (kubernetes_provider.kubeconfig is the in-cluster controller's,
    # empty => in-cluster auth, which is wrong for a CLI running off-cluster).
    kp = config.kubernetes_provider
    namespace = (kp.namespace if kp is not None else "") or "iris"
    cw_kubeconfig = config.platform.coreweave.kubeconfig_path
    kubeconfig_path = os.path.expanduser(cw_kubeconfig) if cw_kubeconfig else ""
    return CoreweaveTarget(
        namespace=namespace,
        kubeconfig_path=kubeconfig_path,
        kube_context=config.platform.coreweave.kube_context,
    )


def pod_label_selector(task_id: str) -> str:
    """k8s label selector matching the task pod Iris created for ``task_id``."""
    return f"{_LABEL_TASK_ID}={_sanitize_label_value(task_id)}"


def kubectl_base(target: CoreweaveTarget) -> list[str]:
    cmd = ["kubectl"]
    if target.kubeconfig_path:
        cmd.append(f"--kubeconfig={target.kubeconfig_path}")
    if target.kube_context:
        cmd.append(f"--context={target.kube_context}")
    cmd.append(f"--namespace={target.namespace}")
    return cmd


def kubectl_get_pods_cmd(target: CoreweaveTarget, selector: str) -> list[str]:
    return [*kubectl_base(target), "get", "pods", "-l", selector, "-o", "json"]


def kubectl_connect_cmd(target: CoreweaveTarget, pod: PodRef) -> list[str]:
    return [*kubectl_base(target), "exec", "-it", pod.pod_name, "-c", pod.container, "--", "bash", "-l"]


def parse_running_pod(pods_json: dict) -> str | None:
    """Return the lexicographically-first Running pod name, or None."""
    items = pods_json.get("items", [])
    running = [p for p in items if p.get("status", {}).get("phase") == "Running"]
    if not running:
        return None
    return min(running, key=lambda p: p.get("metadata", {}).get("name", ""))["metadata"]["name"]


def run_logged(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", shlex.join(cmd))
    return subprocess.run(cmd, **kwargs)


def find_workspace_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


@contextmanager
def controller_client(config_file: str) -> Iterable[IrisClient]:
    iris_config = load_config(config_file)
    controller_address = iris_config.controller_address()
    providers = provider_bundle(iris_config)
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace)
        try:
            yield client
        finally:
            client.shutdown()


def state_path(state_dir: Path, session_name: str) -> Path:
    return state_dir / f"{session_name}.json"


def load_state(path: Path) -> DevGpuState:
    if not path.exists():
        raise click.ClickException(f"No active dev GPU session at {path}")
    return DevGpuState.from_json(path.read_text())


def save_state(path: Path, state: DevGpuState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


def is_job_active(client: IrisClient, job_id: str) -> bool:
    return client.job_state(JobName.from_wire(job_id)) not in INACTIVE_JOB_STATES


def wait_for_running_task(job, *, timeout: float) -> str:
    """Block until the holder job's single task is RUNNING; return its task_id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = job.state_only()
        if state in TERMINAL_JOB_STATES:
            error = job.status().error or job_pb2.JobState.Name(state)
            raise click.ClickException(f"Dev GPU allocation failed: {error}")
        tasks = job.tasks()
        if tasks:
            task = tasks[0]
            if task.status().state == job_pb2.TASK_STATE_RUNNING:
                return str(task.task_id)
        time.sleep(5)
    raise click.ClickException(f"Timed out waiting for dev GPU task after {int(timeout)}s")


def wait_for_running_pod(target: CoreweaveTarget, task_id: str, *, timeout: float) -> PodRef:
    """Poll kubectl until the task's backing pod is Running."""
    selector = pod_label_selector(task_id)
    logger.info("Polling for a Running pod matching selector %s", selector)
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        result = subprocess.run(kubectl_get_pods_cmd(target, selector), capture_output=True, text=True)
        if result.returncode == 0:
            pod_name = parse_running_pod(json.loads(result.stdout or "{}"))
            if pod_name:
                return PodRef(namespace=target.namespace, pod_name=pod_name, container=TASK_CONTAINER)
        else:
            last_err = result.stderr.strip()
        time.sleep(3)
    raise click.ClickException(
        f"Timed out resolving a Running pod for selector {selector!r} after {int(timeout)}s. {last_err}"
    )


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
    """Development GPU (CoreWeave H100) pod management for Iris clusters."""
    ctx.ensure_object(Context)
    ctx.obj.config_file = str(Path(config).resolve()) if config else None
    ctx.obj.session_name = session_name or getpass.getuser()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command("allocate")
@click.option(
    "--gpu-count",
    type=click.IntRange(min=1),
    default=DEFAULT_GPU_COUNT,
    show_default=True,
    help="H100 GPUs to reserve. Only 8 (a whole h100-8x node) is validated; "
    "smaller values may not schedule on the bare-metal pool.",
)
@click.option(
    "--cpu",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_CPU,
    show_default=True,
    help="CPU cores to request for the dev pod (request only, burstable).",
)
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
def allocate(ctx, gpu_count: int, cpu: float, memory: str, disk: str, timeout: int, pod_timeout: int) -> None:
    """Allocate a dev GPU H100 pod and hold it until Ctrl-C."""
    if not ctx.obj.config_file:
        raise click.ClickException("--config is required")

    session_name = ctx.obj.session_name
    state_file = state_path(ctx.obj.state_dir, session_name)
    if state_file.exists():
        raise click.ClickException(
            f"Dev GPU session '{session_name}' already exists. Use release first or choose a new --name."
        )

    target = require_coreweave_platform(load_config(ctx.obj.config_file))

    state: DevGpuState | None = None
    with controller_client(ctx.obj.config_file) as client:
        resources = ResourceSpec(cpu=cpu, memory=memory, disk=disk, device=gpu_device(GPU_VARIANT, gpu_count))
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
                name=f"dev-gpu-{session_name}",
                resources=resources,
            )
        except JobAlreadyExists as exc:
            raise click.ClickException(f"Job already exists for session '{session_name}': {exc}") from exc

        try:
            task_id = wait_for_running_task(job, timeout=timeout)
            pod = wait_for_running_pod(target, task_id, timeout=pod_timeout)
            state = DevGpuState(
                session_name=session_name,
                config_file=ctx.obj.config_file,
                job_id=str(job.job_id),
                gpu_count=gpu_count,
                target=target,
                pod=pod,
            )
            save_state(state_file, state)

            print(f"Session: {session_name}")
            print(f"Job: {job.job_id}")
            print(f"GPUs: {gpu_count} x {GPU_VARIANT}")
            print(f"Pod: {pod.pod_name} (namespace={pod.namespace})")
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
                client.terminate(JobName.from_wire(str(job.job_id)))
                terminated = True
            except Exception:
                logger.warning(
                    "Failed to terminate holder job %s; the H100 pod may still be running. "
                    "Keeping local session state — run `release` to retry cleanup.",
                    job.job_id,
                    exc_info=True,
                )
            # Only drop the session file once the job is actually gone, so a failed
            # terminate never orphans an expensive pod with no local record of its job id.
            if state is not None and terminated:
                state_file.unlink(missing_ok=True)


@cli.command("connect")
@click.pass_context
def connect(ctx) -> None:
    """Open an interactive shell into the reserved pod."""
    state = load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))
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
    # state.pod is the pod resolved at allocation time. If Iris rescheduled the task
    # onto a new pod while the job stayed active, this kubectl exec fails; re-allocate.
    run_logged(kubectl_connect_cmd(state.target, state.pod), check=True)


@cli.command("status")
@click.pass_context
def status(ctx) -> None:
    """Show the current session state."""
    state = load_state(state_path(ctx.obj.state_dir, ctx.obj.session_name))
    print(f"Session: {state.session_name}")
    print(f"Job: {state.job_id}")
    print(f"Config: {state.config_file}")
    print(f"GPUs: {state.gpu_count} x {GPU_VARIANT}")
    print(f"Pod: {state.pod.pod_name} (namespace={state.pod.namespace})")


@cli.command("release")
@click.option("--force", is_flag=True, help="Delete the local session file even if terminating the holder job fails.")
@click.pass_context
def release(ctx, force: bool) -> None:
    """Terminate the holder job and clear the local session file."""
    state_file = state_path(ctx.obj.state_dir, ctx.obj.session_name)
    state = load_state(state_file)
    try:
        with controller_client(state.config_file) as client:
            client.terminate(JobName.from_wire(state.job_id))
    except Exception as exc:
        # Keep the state file on failure so the job id isn't lost while the pod may
        # still be running; --force is the escape hatch for already-dead jobs.
        if not force:
            raise click.ClickException(
                f"Failed to terminate holder job {state.job_id}; the H100 pod may still be running. "
                f"Retry `release`, or pass --force to drop local state anyway (then confirm the job is gone "
                f"with `iris job list`). Error: {exc}"
            ) from exc
        logger.warning("Terminate failed but --force given; deleting local session state anyway.", exc_info=True)
    state_file.unlink(missing_ok=True)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
