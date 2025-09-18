#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Allocate a development TPU on a Ray cluster.

This script allocates a TPU using a Ray job and then "holds it" as long as the script
remains running. It will automatically set up a reasonable home directory and SSH
configuration for the target TPU.

You can connect to the TPU directly using `ssh dev-tpu-<tpu-name>`, or via the `connect`
command.

The `watch` command lets you watch for local file changes and automatically sync
and restart a command on the TPU.


Usage:
  # Allocate a TPU and hold it until Ctrl-C
  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml allocate

  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml connect
  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml execute -- uv run python test_train.py

  # Run a test, checking for changes:
  uv run scripts/ray/dev_tpu.py --cluster us-central2 watch -- uv run pytest tests/post_training/test_rollout_replay.py

"""

import getpass
import glob
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import click
import ray
from draccus import wrap
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# N.B. We have to import from "src" as we are using the `ray.remote` directly from our `uv run`
# script instead of launching a driver job. This confuses cloudpickle for some reason.
from src.marin.cluster import ray as ray_utils
from src.marin.cluster.config import RayClusterConfig, find_config_by_region
from src.marin.utils import _hacky_remove_tpu_lockfile

logger = logging.getLogger(__name__)


class TPUAllocationActor:
    """Actor that holds TPU allocation and manages TPU info."""

    def __init__(self, username: str, tpu_name: str, tpu_type: str = "v4-8"):
        """Initialize the actor and fetch TPU info."""
        import socket

        import requests

        self.username = username
        self.tpu_name = tpu_name
        self.tpu_type = tpu_type
        self.ready = False
        self.hostname = None
        self.ip_address = None
        self.error = None

        try:
            # Fetch TPU metadata from Google metadata server
            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
                headers={"Metadata-Flavor": "Google"},
            )
            self.ip_address = response.text.strip()
            self.hostname = socket.gethostname()
            self.ready = True
            logger.info(f"TPU info fetched - hostname: {self.hostname}, IP: {self.ip_address}")
        except Exception as e:
            logger.error("Failed to fetch TPU metadata", exc_info=True)
            self.ready = True
            self.hostname = "unknown"
            self.ip_address = "unknown"
            self.error = e

    def host_info(self) -> dict[str, str]:
        return {
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "username": self.username,
            "tpu_name": self.tpu_name,
            "tpu_type": self.tpu_type,
            "error": str(self.error) if self.error else "",
        }

    def heartbeat(self) -> str:
        return f"TPU allocation '{self.tpu_name}' active for {self.username}"

    def __del__(self):
        _hacky_remove_tpu_lockfile()


def add_ssh_host_config(hostname: str, ip_address: str, username: str, tpu_name: str) -> None:
    """Add SSH host configuration."""
    config_path = Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(exist_ok=True)

    # Check if Google Compute Engine SSH key exists
    gce_key_path = Path.home() / ".ssh" / "google_compute_engine"
    if not gce_key_path.exists():
        logger.warning(f"Google Compute Engine SSH key not found at {gce_key_path}")
        logger.warning("You may need to run 'gcloud compute ssh' first to set up SSH keys")

    host_alias = f"dev-tpu-{tpu_name}"

    ssh_config_entry = f"""
# BEGIN_DEV_TPU_{tpu_name.upper()}
Host {host_alias}
    HostName {ip_address}
    IdentityFile ~/.ssh/google_compute_engine
    UserKnownHostsFile ~/.ssh/google_compute_known_hosts
    HostKeyAlias compute.{hostname}
    StrictHostKeyChecking no
    IdentitiesOnly yes
    CheckHostIP no
    User {username}
# END_DEV_TPU_{tpu_name.upper()}
"""

    existing_config = ""
    if config_path.exists():
        existing_config = config_path.read_text()

    # Backup the existing config and update it with our new entry
    with open(f"{config_path}.bak", "w") as f:
        f.write(existing_config)

    # Check if this specific TPU config already exists and remove it
    tpu_marker = f"BEGIN_DEV_TPU_{tpu_name.upper()}"
    if tpu_marker in existing_config:
        pattern = f"# BEGIN_DEV_TPU_{tpu_name.upper()}\n(.*?\n)*?# END_DEV_TPU_{tpu_name.upper()}\n"
        existing_config = re.sub(pattern, "", existing_config, flags=re.DOTALL)

    with open(config_path, "w") as f:
        f.write(existing_config + ssh_config_entry)

    logger.info(f"Added SSH configuration for {host_alias}")


def remove_ssh_host_config(tpu_name: str) -> None:
    """Remove SSH host configuration for specific TPU."""
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        return

    existing_config = config_path.read_text()

    tpu_marker = f"BEGIN_DEV_TPU_{tpu_name.upper()}"
    if tpu_marker in existing_config:
        pattern = f"# BEGIN_DEV_TPU_{tpu_name.upper()}\n(.*?\n)*?# END_DEV_TPU_{tpu_name.upper()}\n"
        updated_config = re.sub(pattern, "", existing_config, flags=re.DOTALL)
        config_path.write_text(updated_config)
        logger.info(f"Removed SSH configuration for dev-tpu-{tpu_name}")


def list_tracked_files(local_path: str) -> list[str]:
    files = set()

    # Get all files that git would track (excluding gitignored files)
    # This includes tracked, modified, staged, and untracked files
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=local_path, check=True, capture_output=True, text=True
    )
    all_files = [f for f in result.stdout.split("\0") if f.strip()]
    files.update(all_files)

    return sorted(files)


def sync_to_remote(target_host: str, local_path: os.PathLike = ".") -> None:
    local_path = Path(local_path).resolve()
    sync_files = list_tracked_files(local_path)

    subprocess.run(["ssh", target_host, "mkdir", "-p", "/home/$USER/marin"], check=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        f.write("\n".join(sync_files))
        files_from_path = f.name
        rsync_cmd = [
            "rsync",
            "-az",
            "--delete",
            "--progress",
            "--files-from",
            files_from_path,
            f"{local_path}/",
            f"{target_host}:/home/$USER/marin/",
        ]

        logger.info(f"Syncing {len(sync_files)} files (git-tracked, modified, and important untracked)...")
        subprocess.run(rsync_cmd, check=True)
        logger.info("File sync completed successfully")

def setup_remote_environment(target_host: str) -> None:
    """Set up the remote environment on the TPU."""
    setup_script = """
set -x
cd /home/$USER/marin

echo "Setting up development environment..."
curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

echo "Installing dependencies..."
cd /home/$USER/marin
uv sync --extra=tpu --python=3.11 || true
"""
    logger.info("Setting up remote environment...")
    subprocess.run(["ssh", target_host, "bash", "-c", setup_script], check=True)
    logger.info("Environment setup completed")




@contextmanager
def hold_tpu_allocation(
    username: str, tpu_name: str, config_file: str, sync_path: str = ".", tpu_type: str = "v4-8", duration_minutes: int = 480
) -> Generator[dict[str, str], None, None]:
    """Context manager that holds a TPU allocation until the context exits.

    Uses a Ray actor to manage the TPU allocation lifecycle.
    """
    logger.info(f"START_TPU_{tpu_name.upper()}: Beginning TPU allocation")

    actor = None

    try:
        with ray_utils.ray_dashboard(config_file, ray_init=True):
            logger.info(f"Creating TPU allocation actor for {tpu_name}")
            actor = ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})(TPUAllocationActor).remote(username, tpu_name, tpu_type)

            allocation_info = ray.get(actor.host_info.remote(), timeout=60)

            logger.info("Setting up SSH configuration")
            add_ssh_host_config(allocation_info["hostname"], allocation_info["ip_address"], username, tpu_name)

            logger.info("Syncing environment")
            sync_to_remote(f"dev-tpu-{tpu_name}", sync_path)
            setup_remote_environment(f"dev-tpu-{tpu_name}")

            print("TPU allocated successfully!")
            print(f"Hostname: {allocation_info['hostname']}")
            print(f"IP Address: {allocation_info['ip_address']}")
            print(f"TPU name: {tpu_name}")
            print(f"SSH alias: dev-tpu-{tpu_name}")
            yield allocation_info
    finally:
        remove_ssh_host_config(tpu_name)
        if actor:
            ray.kill(actor)


class Context:
    def __init__(self):
        self.verbose: bool = False
        self.config_file: Optional[str] = None
        self.config_obj: Optional[RayClusterConfig] = None
        self.tpu_name: Optional[str] = None


@click.group()
@click.option("--config", help="Path to cluster config file")
@click.option("--cluster", help="Cluster name to connect to")
@click.option("--tpu-name", help="TPU name identifier")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, cluster, tpu_name, verbose):
    """Development TPU management for Ray clusters."""
    ctx.ensure_object(Context)
    if cluster:
        config = find_config_by_region(cluster)

    ctx.obj.config_file = config
    ctx.obj.tpu_name = tpu_name or getpass.getuser()
    ctx.obj.verbose = verbose

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG if verbose else logging.INFO
    )

    if config:
        ctx.obj.config_obj = RayClusterConfig.from_yaml(config)


@cli.command("allocate")
@click.option("--tpu-type", default="v4-8", help="TPU type")
@click.option("--sync-path", default=".", help="Local path to sync")
@click.option("--username", help="Username to use for ssh", default=getpass.getuser())
@click.option("--duration", default=480, help="Allocation duration in minutes")
@click.pass_context
def allocate(ctx, tpu_type, sync_path, username, duration):
    """Allocate a development TPU. Holds until Ctrl-C."""
    if not ctx.obj.config_file:
        print("Error: --config required", file=sys.stderr)
        sys.exit(1)

    if not username:
        username = getpass.getuser()

    tpu_name = ctx.obj.tpu_name

    if tpu_type not in ["v4-8", "v5p-8"]:
        print(f"Warning: TPU type {tpu_type} may not be supported", file=sys.stderr)

    print(f"Allocating development TPU '{tpu_name}' for {username}...")
    print(f"TPU type: {tpu_type}")
    print(f"Duration: {duration} minutes")

    with hold_tpu_allocation(username, tpu_name, ctx.obj.config_file, sync_path, tpu_type, duration):
        print("\nTPU allocation is active. Press Ctrl-C to release...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived Ctrl-C, releasing TPU allocation...")

    print("TPU allocation released.")


@cli.command("connect")
@click.option("--username", help="Username to use for SSH", default=getpass.getuser())
@click.pass_context
def connect(ctx, username):
    """Connect to development TPU via SSH."""
    if not username:
        username = getpass.getuser()

    tpu_name = ctx.obj.tpu_name
    host_alias = f"dev-tpu-{tpu_name}"

    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print(f"You need to run 'allocate --tpu-name {tpu_name}' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {host_alias}...")
    subprocess.run(["ssh", host_alias])


@cli.command("execute", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--username", help="Username to use for ssh", default=getpass.getuser())
@click.option("--sync-path", default=".", help="Local path to sync")
@click.pass_context
def execute(ctx, command, username, sync_path):
    """Execute a command on the development TPU.

    This command will:
    1. Sync any local changes to the TPU
    2. Execute the provided command in the /home/$USER/marin directory
    3. Return the exit code from the remote command

    Examples:
        uv run scripts/ray/dev_tpu.py execute -- uv run python train.py
        uv run scripts/ray/dev_tpu.py execute -- uv run pytest tests/ -v
    """
    if not username:
        username = getpass.getuser()

    tpu_name = ctx.obj.tpu_name
    host_alias = f"dev-tpu-{tpu_name}"

    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print(f"You need to run 'allocate --tpu-name {tpu_name}' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    # Sync files
    print(f"Syncing local changes to {host_alias}...")
    sync_to_remote(host_alias, sync_path)

    command_str = " ".join(command)
    full_cmd = f"ssh -t {host_alias} 'source $HOME/.local/bin/env && cd marin && {command_str}'"
    result = subprocess.run(full_cmd, check=False, shell=True)

    sys.exit(result.returncode)


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events during watch mode."""

    def __init__(self, callback, debounce_seconds=0.5):
        super().__init__()
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.last_trigger = 0
        self._timer = None

    def on_any_event(self, event):
        if event.is_directory:
            return

        # Skip temp files, build artifacts, etc.
        skip_patterns = [
            '.git/', '__pycache__/', '.pytest_cache/', '.mypy_cache/',
            '.DS_Store', '.swp', '.tmp', '~', '.pyc', '.pyo'
        ]

        event_path = str(event.src_path)
        if any(pattern in event_path for pattern in skip_patterns):
            return

        # Debounce rapid file changes
        current_time = time.time()
        if self._timer:
            self._timer.cancel()

        self._timer = threading.Timer(self.debounce_seconds, self._trigger_callback)
        self._timer.start()

    def _trigger_callback(self):
        self.callback()


def kill_remote_process(host_alias: str, process_pattern: str) -> None:
    """Kill remote process matching pattern."""
    try:
        # Find and kill processes matching the pattern
        kill_cmd = f"ssh {host_alias} 'pkill -f \"{process_pattern}\"'"
        subprocess.run(kill_cmd, shell=True, check=False, capture_output=True)
        logger.info(f"Killed remote processes matching: {process_pattern}")
    except Exception as e:
        logger.warning(f"Failed to kill remote process: {e}")


class RemoteProcessManager:
    """Run a remote process, synchronizing and restarting on demand."""
    def __init__(self, host_alias: str, command_str: str, sync_path: str):
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._command_str = command_str
        self._host_alias = host_alias
        self._sync_path = sync_path

    def __call__(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                print("Killing remote...")
                self._process.send_signal(signal.SIGINT)
                start_time = time.time()
                while self._process.poll() is None and time.time() - start_time < 5:
                    time.sleep(0.1)
                if self._process.poll() is None:
                    print("Remote process did not exit, killing...")
                    self._process.terminate()

            print(f"Syncing changes to {self._host_alias}...")
            sync_to_remote(self._host_alias, self._sync_path)

            full_cmd = f"ssh -t {self._host_alias} 'source $HOME/.local/bin/env && cd marin && {self._command_str}'"
            print(f"Running: {self._command_str}")
            self._process = subprocess.Popen(full_cmd, shell=True)

    def check_status(self):
        if self._process and self._process.poll() is not None:
            print(f"Remote process exited with code {self._process.returncode}")
            # restart if we exited with an error, otherwise wait for a file change.
            if self._process.returncode != 0:
                print("Restarting remote process...")
                self()
      

@cli.command("watch", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--username", help="Username to connect as", default=getpass.getuser())
@click.option("--sync-path", default=".", help="Local path to sync")
@click.option("--debounce", default=1, help="Debounce time for file changes in seconds")
@click.pass_context
def watch(ctx, command, username, sync_path, debounce):
    """Watch for file changes and restart command on TPU.

    This command will:
    1. Sync files and start the initial command
    2. Watch for local file changes
    3. On changes: kill remote process, sync, restart command

    Examples:
        uv run scripts/ray/dev_tpu.py watch -- uv run python train.py
        uv run scripts/ray/dev_tpu.py watch --watch-path src --watch-path tests -- uv run pytest
    """
    tpu_name = ctx.obj.tpu_name
    host_alias = f"dev-tpu-{tpu_name}"

    # Check if SSH config exists
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print(f"You need to run 'allocate --tpu-name {tpu_name}' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    command_str = " ".join(command)
    remote_process: subprocess.Popen = None

    process_mgr = RemoteProcessManager(host_alias, command_str, sync_path)
    process_mgr()
    observer = Observer()
    event_handler = FileChangeHandler(process_mgr, debounce_seconds=debounce)
    observer.schedule(event_handler, sync_path, recursive=True)
    observer.start()
    print(f"Watching for changes in {sync_path}...")
    print("Press Ctrl-C to stop")

    try:
        while True:
            time.sleep(5)
            process_mgr.check_status()
    except KeyboardInterrupt:
        print("\nStopping watch mode...")
        observer.stop()
        if remote_process:
            kill_remote_process(host_alias, command_str.split()[0])

    observer.join()
    print("Watch mode stopped.")


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()


if __name__ == "__main__":
    main()
