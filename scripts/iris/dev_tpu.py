#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allocate and use development TPUs on Iris-managed clusters."""

from __future__ import annotations

import atexit
import getpass
import logging
import os
import shlex
import subprocess
import tarfile
import tempfile
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from iris.client import IrisClient, JobAlreadyExists
from iris.cluster.constraints import zone_constraint
from iris.cluster.config import IrisConfig
from iris.cluster.types import Entrypoint, JobName, ResourceSpec, tpu_device
from iris.dev_tpu import DevTpuState, DevTpuWorker, GcpNodeRef, parse_worker_host
from iris.rpc import job_pb2
from marin.cluster import gcp

logger = logging.getLogger(__name__)

DEFAULT_ENV_VARS = [
    "HF_TOKEN",
    "WANDB_API_KEY",
    "MARIN_PREFIX",
    "OPENAI_API_KEY",
    "GCLOUD_PROJECT",
    "GCLOUD_TOKEN_PATH",
    "WANDB_MODE",
    "RUN_ID",
]

HOLDER_COMMAND = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
    "print('iris dev tpu holder ready', flush=True); "
    "time.sleep(365 * 24 * 60 * 60)"
)

STATE_DIR = Path.home() / ".cache" / "marin" / "dev_tpu_iris"


@dataclass
class Context:
    config_file: str | None = None
    session_name: str | None = None
    verbose: bool = False
    state_dir: Path = STATE_DIR


def run_logged(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", shlex.join(cmd))
    return subprocess.run(cmd, **kwargs)


def default_session_name() -> str:
    return getpass.getuser()


def build_env_dict(extra_env: list[str] | None = None, forward_all: bool = False) -> dict[str, str]:
    env_dict: dict[str, str] = {}
    for var in DEFAULT_ENV_VARS:
        if os.environ.get(var) is not None:
            env_dict[var] = os.environ[var]

    if forward_all:
        env_dict = dict(os.environ)

    for config_file in (".levanter.yaml", ".marin.yaml", ".config"):
        path = Path(config_file)
        if not path.exists():
            continue
        logger.info("Injecting environment variables from %s", path)
        config_yaml = yaml.safe_load(path.read_text()) or {}
        for key, value in config_yaml.get("env", {}).items():
            env_dict[str(key)] = str(value)

    if extra_env:
        for env_var in extra_env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                env_dict[key] = value
            elif os.environ.get(env_var) is not None:
                env_dict[env_var] = os.environ[env_var]

    return env_dict


def build_env_string(env_dict: dict[str, str]) -> str:
    return " ".join(f"{shlex.quote(key)}={shlex.quote(value)}" for key, value in env_dict.items())


def list_tracked_files(local_path: Path) -> list[str]:
    result = run_logged(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=local_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(file for file in result.stdout.split("\0") if file.strip())


def create_sync_archive(local_path: Path) -> Path:
    files = list_tracked_files(local_path)
    fd, archive_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    archive = Path(archive_path)
    with tarfile.open(archive, "w:gz") as tar:
        for relative in files:
            source = local_path / relative
            if source.exists():
                tar.add(source, arcname=relative, recursive=True)
    return archive


def state_path(state_dir: Path, session_name: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return DevTpuState.state_file_path(state_dir, session_name)


def load_state(path: Path) -> DevTpuState:
    if not path.exists():
        raise click.ClickException(f"No active dev TPU session at {path}")
    return DevTpuState.from_json(path.read_text())


def save_state(path: Path, state: DevTpuState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json())


def _require_gcp_platform(config) -> str:
    if not config.platform.HasField("gcp"):
        raise click.ClickException("Iris dev TPU currently supports only GCP-backed clusters.")
    project = config.platform.gcp.project_id or gcp.get_project_id()
    if not project:
        raise click.ClickException("Could not determine the GCP project for this cluster.")
    return project


def find_workspace_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


@contextmanager
def controller_client(config_file: str) -> Iterable[IrisClient]:
    iris_config = IrisConfig.load(config_file)
    controller_address = iris_config.controller_address()
    providers = iris_config.provider_bundle()
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.proto.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace)
        try:
            yield client
        finally:
            client.shutdown()


def resolve_node_ref(host: str, project: str) -> GcpNodeRef:
    tpu_match = gcp.find_tpu_by_ip(host, project, zone="-")
    if tpu_match:
        name, zone, worker_id = tpu_match
        return GcpNodeRef(kind="tpu", name=name, zone=zone, project=project, tpu_worker_id=worker_id)

    vm_match = gcp.find_vm_by_ip(host, project)
    if vm_match:
        name, zone = vm_match
        return GcpNodeRef(kind="vm", name=name, zone=zone, project=project)

    raise click.ClickException(f"Could not resolve a GCP TPU or VM for worker host {host}")


def gcloud_ssh_cmd(node: GcpNodeRef, *, command: str | None = None) -> list[str]:
    if node.kind == "tpu":
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            node.name,
            f"--zone={node.zone}",
            f"--project={node.project}",
            f"--worker={node.tpu_worker_id}",
            "--quiet",
        ]
    else:
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            node.name,
            f"--zone={node.zone}",
            f"--project={node.project}",
            "--quiet",
        ]
    if command is not None:
        cmd.append(f"--command={command}")
    return cmd


def gcloud_scp_cmd(node: GcpNodeRef, source: Path, remote_path: str) -> list[str]:
    if node.kind == "tpu":
        return [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "scp",
            str(source),
            f"{node.name}:{remote_path}",
            f"--zone={node.zone}",
            f"--project={node.project}",
            f"--worker={node.tpu_worker_id}",
            "--quiet",
        ]
    return [
        "gcloud",
        "compute",
        "scp",
        str(source),
        f"{node.name}:{remote_path}",
        f"--zone={node.zone}",
        f"--project={node.project}",
        "--quiet",
    ]


def run_remote_command(node: GcpNodeRef, command: str) -> None:
    run_logged(gcloud_ssh_cmd(node, command=command), check=True)


def popen_remote_command(node: GcpNodeRef, command: str) -> subprocess.Popen:
    cmd = gcloud_ssh_cmd(node, command=command)
    logger.info("Running command: %s", shlex.join(cmd))
    return subprocess.Popen(cmd)


def ensure_ssh_access(node: GcpNodeRef) -> None:
    run_remote_command(node, "true")


def sync_to_remote(node: GcpNodeRef, local_path: Path) -> None:
    archive = create_sync_archive(local_path)
    remote_archive = "~/dev-tpu-sync.tar.gz"
    try:
        run_logged(gcloud_scp_cmd(node, archive, remote_archive), check=True)
        run_remote_command(
            node,
            "bash -lc "
            + shlex.quote(
                'mkdir -p "$HOME/marin" '
                '&& find "$HOME/marin" -mindepth 1 -maxdepth 1 ! -name ".venv" -exec rm -rf {} + '
                f'&& tar -xzf {remote_archive} -C "$HOME/marin" && rm -f {remote_archive}'
            ),
        )
    finally:
        archive.unlink(missing_ok=True)


def setup_remote_environment(node: GcpNodeRef) -> None:
    setup_script = """
set -e
cd "$HOME/marin"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync --all-packages --extra=tpu --python=3.11 || true
"""
    run_remote_command(node, "bash -lc " + shlex.quote(setup_script))


def build_remote_command(command: str, env_dict: dict[str, str] | None = None) -> str:
    stripped = command.strip()
    if stripped.startswith("bash -c") or stripped.startswith("bash -lc"):
        raise click.ClickException("Do not include 'bash -c' in the command; the script already wraps the command.")
    env_string = build_env_string(env_dict or {})
    script = f"""
set -e
source "$HOME/.local/bin/env"
cd "$HOME/marin"
{env_string} exec bash -lc {shlex.quote(command)}
"""
    return "bash -lc " + shlex.quote(script)


def wait_for_workers(job, *, timeout: float, project: str) -> list[DevTpuWorker]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = job.status()
        if status.state in (
            job_pb2.JOB_STATE_FAILED,
            job_pb2.JOB_STATE_KILLED,
            job_pb2.JOB_STATE_UNSCHEDULABLE,
            job_pb2.JOB_STATE_WORKER_FAILED,
        ):
            error = status.error or job_pb2.JobState.Name(status.state)
            raise click.ClickException(f"Dev TPU allocation failed: {error}")

        tasks = job.tasks()
        if tasks:
            resolved: list[DevTpuWorker] = []
            all_running = True
            for task in tasks:
                task_status = task.status()
                if task_status.state != job_pb2.TASK_STATE_RUNNING or not task_status.worker_address:
                    all_running = False
                    break
                host = parse_worker_host(task_status.worker_address)
                node = resolve_node_ref(host, project)
                resolved.append(
                    DevTpuWorker(
                        task_id=str(task.task_id),
                        worker_id=task_status.worker_id,
                        worker_address=task_status.worker_address,
                        host=host,
                        node=node,
                    )
                )
            if all_running:
                return resolved
        time.sleep(5)

    raise click.ClickException(f"Timed out waiting for dev TPU workers after {int(timeout)}s")


def is_job_active(client: IrisClient, job_id: str) -> bool:
    status = client.status(JobName.from_wire(job_id))
    return status.state not in {
        job_pb2.JOB_STATE_SUCCEEDED,
        job_pb2.JOB_STATE_FAILED,
        job_pb2.JOB_STATE_KILLED,
        job_pb2.JOB_STATE_WORKER_FAILED,
        job_pb2.JOB_STATE_UNSCHEDULABLE,
    }


def sync_all_workers(workers: list[DevTpuWorker], local_path: Path) -> None:
    for worker in workers:
        logger.info("Syncing %s (%s)", worker.node.name, worker.host)
        sync_to_remote(worker.node, local_path)


def setup_all_workers(workers: list[DevTpuWorker]) -> None:
    for worker in workers:
        logger.info("Setting up environment on %s (%s)", worker.node.name, worker.host)
        setup_remote_environment(worker.node)


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events during watch mode."""

    def __init__(self, callback, debounce_seconds: float = 0.5):
        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None

    def on_any_event(self, event) -> None:
        if event.is_directory:
            return

        event_path = str(event.src_path)
        skip_patterns = (
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".DS_Store",
            ".swp",
            ".tmp",
            "~",
            ".pyc",
            ".pyo",
        )
        if any(pattern in event_path for pattern in skip_patterns):
            return

        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._callback)
        self._timer.start()


class RemoteProcessManager:
    """Run a remote command, synchronizing and restarting on demand."""

    def __init__(
        self,
        workers: list[DevTpuWorker],
        worker: DevTpuWorker,
        command: str,
        sync_path: Path,
        env_dict: dict[str, str],
        no_sync: bool,
    ):
        self._workers = workers
        self._worker = worker
        self._command = command
        self._sync_path = sync_path
        self._env_dict = env_dict
        self._no_sync = no_sync
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            self.kill()
            if not self._no_sync:
                sync_all_workers(self._workers, self._sync_path)
            remote = build_remote_command(self._command, self._env_dict)
            self._process = popen_remote_command(self._worker.node, remote)

    def check_status(self) -> None:
        if self._process is not None and self._process.poll() is not None:
            print(f"Remote process exited with code {self._process.returncode}")

    def kill(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


@click.group()
@click.option("--config", help="Path to an Iris cluster config file.")
@click.option("--tpu-name", help="Local dev TPU session name.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx, config: str | None, tpu_name: str | None, verbose: bool) -> None:
    """Development TPU management for Iris clusters."""
    ctx.ensure_object(Context)
    ctx.obj.config_file = str(Path(config).resolve()) if config else None
    ctx.obj.session_name = tpu_name or default_session_name()
    ctx.obj.verbose = verbose
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


@cli.command("allocate")
@click.option("--tpu-type", required=True, help="TPU type to reserve (for example, v5p-8).")
@click.option("--sync-path", default=".", show_default=True, help="Local path to sync to the remote host(s).")
@click.option("--no-sync", is_flag=True, help="Skip the initial sync after allocation.")
@click.option("--setup-env/--no-setup-env", default=True, help="Install/update the remote uv environment after sync.")
@click.option("--zone", type=str, default=None, help="Restrict the holder job to a specific zone.")
@click.option("--timeout", type=int, default=900, show_default=True, help="Seconds to wait for workers to come up.")
@click.pass_context
def allocate(
    ctx,
    tpu_type: str,
    sync_path: str,
    no_sync: bool,
    setup_env: bool,
    zone: str | None,
    timeout: int,
) -> None:
    """Allocate a dev TPU session and hold it until Ctrl-C."""
    if not ctx.obj.config_file:
        raise click.ClickException("--config is required")

    local_path = Path(sync_path).resolve()
    if not local_path.exists():
        raise click.ClickException(f"Sync path does not exist: {local_path}")

    session_name = ctx.obj.session_name
    state_file = state_path(ctx.obj.state_dir, session_name)
    if state_file.exists():
        raise click.ClickException(
            f"Dev TPU session '{session_name}' already exists. Use release first or choose a new --tpu-name."
        )

    iris_config = IrisConfig.load(ctx.obj.config_file)
    project = _require_gcp_platform(iris_config.proto)
    client_cm = None
    client: IrisClient | None = None
    job = None
    state: DevTpuState | None = None

    try:
        client_cm = controller_client(ctx.obj.config_file)
        client = client_cm.__enter__()
        resources = ResourceSpec(cpu=0.5, memory="1GB", disk="5GB", device=tpu_device(tpu_type))
        constraints = []
        if zone:
            constraints.append(zone_constraint(zone))
        try:
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", "-c", HOLDER_COMMAND),
                name=f"dev-tpu-{session_name}",
                resources=resources,
                constraints=constraints or None,
            )
        except JobAlreadyExists as exc:
            raise click.ClickException(f"Job already exists for session '{session_name}': {exc}") from exc

        workers = wait_for_workers(job, timeout=timeout, project=project)
        for worker in workers:
            ensure_ssh_access(worker.node)

        state = DevTpuState(
            session_name=session_name,
            config_file=ctx.obj.config_file,
            job_id=str(job.job_id),
            tpu_type=tpu_type,
            workers=workers,
        )
        save_state(state_file, state)

        if not no_sync:
            sync_all_workers(workers, local_path)
            if setup_env:
                setup_all_workers(workers)

        print(f"Session: {session_name}")
        print(f"Job: {job.job_id}")
        print(f"TPU type: {tpu_type}")
        for index, worker in enumerate(workers):
            print(f"Worker {index}: {worker.host} ({worker.node.name}, zone={worker.node.zone})")
        print("\nAllocation is active. Press Ctrl-C to release.")

        while True:
            time.sleep(30)
            if not is_job_active(client, str(job.job_id)):
                raise click.ClickException("The dev TPU holder job terminated unexpectedly.")
    except KeyboardInterrupt:
        print("\nReleasing dev TPU session...")
    finally:
        if client is not None and job is not None:
            try:
                client.terminate(JobName.from_wire(str(job.job_id)))
            except Exception:
                logger.warning("Failed to terminate holder job %s", job.job_id, exc_info=True)
        if client_cm is not None:
            client_cm.__exit__(None, None, None)
        if state is not None:
            state_file.unlink(missing_ok=True)


@cli.command("status")
@click.pass_context
def status(ctx) -> None:
    """Show the current session state."""
    session_name = ctx.obj.session_name
    state = load_state(state_path(ctx.obj.state_dir, session_name))
    print(f"Session: {state.session_name}")
    print(f"Job: {state.job_id}")
    print(f"Config: {state.config_file}")
    print(f"TPU type: {state.tpu_type}")
    for index, worker in enumerate(state.workers):
        print(f"Worker {index}: {worker.host} ({worker.node.name}, zone={worker.node.zone})")


def _load_active_state(session_name: str, state_dir: Path) -> DevTpuState:
    return load_state(state_path(state_dir, session_name))


def _verify_active_state(state: DevTpuState) -> None:
    with controller_client(state.config_file) as client:
        if not is_job_active(client, state.job_id):
            raise click.ClickException(
                f"Dev TPU session '{state.session_name}' is no longer active. Use release to clean up the state file."
            )


def _pick_worker(state: DevTpuState, worker_index: int) -> DevTpuWorker:
    if worker_index < 0 or worker_index >= len(state.workers):
        raise click.ClickException(f"--worker must be between 0 and {len(state.workers) - 1}")
    return state.workers[worker_index]


@cli.command("release")
@click.pass_context
def release(ctx) -> None:
    """Terminate the holder job and clear the local session file."""
    session_name = ctx.obj.session_name
    state_file = state_path(ctx.obj.state_dir, session_name)
    state = load_state(state_file)
    try:
        with controller_client(state.config_file) as client:
            client.terminate(JobName.from_wire(state.job_id))
    finally:
        state_file.unlink(missing_ok=True)


@cli.command("connect")
@click.option("--worker", "worker_index", default=0, show_default=True, help="Worker index to connect to.")
@click.pass_context
def connect(ctx, worker_index: int) -> None:
    """Open an interactive SSH session to a reserved worker."""
    state = _load_active_state(ctx.obj.session_name, ctx.obj.state_dir)
    _verify_active_state(state)
    worker = _pick_worker(state, worker_index)
    run_logged(gcloud_ssh_cmd(worker.node), check=True)


@cli.command("setup_env")
@click.option("--worker", "worker_index", default=None, type=int, help="Specific worker index to set up.")
@click.option("--sync-path", default=".", show_default=True, help="Local path to sync to the remote host(s).")
@click.option("--no-sync", is_flag=True, help="Skip syncing local files before installing the environment.")
@click.pass_context
def setup_env(ctx, worker_index: int | None, sync_path: str, no_sync: bool) -> None:
    """Install or refresh the remote uv environment on reserved workers."""
    state = _load_active_state(ctx.obj.session_name, ctx.obj.state_dir)
    _verify_active_state(state)
    workers = state.workers if worker_index is None else [_pick_worker(state, worker_index)]
    local_path = Path(sync_path).resolve()
    if not no_sync:
        if not local_path.exists():
            raise click.ClickException(f"Sync path does not exist: {local_path}")
        sync_all_workers(workers, local_path)

    setup_all_workers(workers)


@cli.command("execute", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--worker", "worker_index", default=0, show_default=True, help="Worker index to use.")
@click.option("--sync-path", default=".", show_default=True, help="Local path to sync to the remote host(s).")
@click.option("--no-sync", is_flag=True, help="Skip syncing local files before running the command.")
@click.option("--env", "-e", multiple=True, help="Environment variables to forward (KEY=VALUE or KEY).")
@click.option("--forward-all-env", is_flag=True, help="Forward all environment variables.")
@click.pass_context
def execute(ctx, command, worker_index: int, sync_path: str, no_sync: bool, env, forward_all_env: bool) -> None:
    """Sync files and execute one command on a reserved worker."""
    state = _load_active_state(ctx.obj.session_name, ctx.obj.state_dir)
    _verify_active_state(state)
    worker = _pick_worker(state, worker_index)

    local_path = Path(sync_path).resolve()
    if not no_sync:
        sync_all_workers(state.workers, local_path)

    env_dict = build_env_dict(extra_env=list(env), forward_all=forward_all_env)
    remote = build_remote_command(shlex.join(command), env_dict)
    process = popen_remote_command(worker.node, remote)
    atexit.register(process.terminate)
    raise SystemExit(process.wait())


@cli.command("watch", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--worker", "worker_index", default=0, show_default=True, help="Worker index to use.")
@click.option("--sync-path", default=".", show_default=True, help="Local path to sync to the remote host(s).")
@click.option("--debounce", default=1.0, show_default=True, help="Debounce time for file changes in seconds.")
@click.option("--no-sync", is_flag=True, help="Skip syncing local files before each run.")
@click.option("--env", "-e", multiple=True, help="Environment variables to forward (KEY=VALUE or KEY).")
@click.option("--forward-all-env", is_flag=True, help="Forward all environment variables.")
@click.pass_context
def watch(
    ctx,
    command,
    worker_index: int,
    sync_path: str,
    debounce: float,
    no_sync: bool,
    env,
    forward_all_env: bool,
) -> None:
    """Watch for file changes and restart the command on a reserved worker."""
    state = _load_active_state(ctx.obj.session_name, ctx.obj.state_dir)
    _verify_active_state(state)
    worker = _pick_worker(state, worker_index)

    local_path = Path(sync_path).resolve()
    env_dict = build_env_dict(extra_env=list(env), forward_all=forward_all_env)
    process_manager = RemoteProcessManager(state.workers, worker, shlex.join(command), local_path, env_dict, no_sync)
    atexit.register(process_manager.kill)
    process_manager()

    observer = Observer()
    observer.schedule(FileChangeHandler(process_manager, debounce_seconds=debounce), str(local_path), recursive=True)
    observer.start()
    print(f"Watching for changes in {local_path}...")
    print("Press Ctrl-C to stop.")

    try:
        while True:
            time.sleep(5)
            process_manager.check_status()
    except KeyboardInterrupt:
        print("\nStopping watch mode...")
    finally:
        observer.stop()
        process_manager.kill()
        observer.join()


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
