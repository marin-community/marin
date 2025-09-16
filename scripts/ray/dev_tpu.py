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
Development TPU management for Ray clusters.

This script provides a simple context manager approach for TPU allocation.
The TPU is held only while the script is running and is automatically
released when you press Ctrl-C or the script exits.

Usage:
  # Allocate a TPU and hold it until Ctrl-C
  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml allocate

  # Connect to TPU via SSH (requires prior allocation)
  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml connect

  # Execute a command on the TPU (syncs changes first)
  uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml execute -- uv run python train.py

Commands:
  allocate  Allocate TPU, set up SSH config, sync environment, and wait for Ctrl-C
  connect   Connect to TPU via SSH using the configured alias
  execute   Sync local changes and execute a command on the TPU
"""

import re
import getpass
import logging
import tempfile
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import click

# Disable Ray auto-initialization to prevent conflicts
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
from draccus import wrap
import ray

from src.marin.cluster import ray as ray_utils
from src.marin.cluster.config import RayClusterConfig, find_config_by_region
from src.marin.utils import _hacky_remove_tpu_lockfile

logger = logging.getLogger(__name__)


class TPUAllocationActor:
    """Actor that holds TPU allocation and manages TPU info."""

    def __init__(self, username: str, tpu_type: str = "v4-8"):
        """Initialize the actor and fetch TPU info."""
        import requests
        import socket

        self.username = username
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
            "tpu_type": self.tpu_type,
            "error": str(self.error) if self.error else "",
        }

    def heartbeat(self) -> str:
        """Keep-alive method to check actor status."""
        return f"TPU allocation active for {self.username}"

    def __del__(self):
        """Cleanup when actor is destroyed."""
        _hacky_remove_tpu_lockfile()
        logger.info(f"TPU allocation actor cleanup for {self.username}")


def add_ssh_host_config(hostname: str, ip_address: str, username: str) -> None:
    """Add SSH host configuration."""
    config_path = Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(exist_ok=True)

    # Check if Google Compute Engine SSH key exists
    gce_key_path = Path.home() / ".ssh" / "google_compute_engine"
    if not gce_key_path.exists():
        logger.warning(f"Google Compute Engine SSH key not found at {gce_key_path}")
        logger.warning("You may need to run 'gcloud compute ssh' first to set up SSH keys")

    host_alias = f"dev-tpu-{username}"

    ssh_config_entry = f"""
# BEGIN_DEV_TPU
Host {host_alias}
    HostName {ip_address}
    IdentityFile ~/.ssh/google_compute_engine
    UserKnownHostsFile ~/.ssh/google_compute_known_hosts
    HostKeyAlias compute.{hash(hostname) % 1000000000000000}
    StrictHostKeyChecking no
    IdentitiesOnly yes
    CheckHostIP no
    User {username}
# END_DEV_TPU
"""

    # Read existing config
    existing_config = ""
    if config_path.exists():
        existing_config = config_path.read_text()

    # Backup existing config
    with open(f"{config_path}.bak", "w") as f:
        f.write(existing_config)

    # Check if host already exists and remove it
    if "BEGIN_DEV_TPU\n" in existing_config:
        existing_config = re.sub(r"# BEGIN_DEV_TPU\n(.*?\n)*?# END_DEV_TPU\n", "", existing_config, flags=re.DOTALL)

    # Append new config
    with open(config_path, "w") as f:
        f.write(existing_config + ssh_config_entry)

    logger.info(f"Added SSH configuration for {host_alias}")


def remove_ssh_host_config(username: str) -> None:
    """Remove SSH host configuration."""
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists():
        return

    existing_config = config_path.read_text()

    # Remove the dev TPU configuration section
    if "BEGIN_DEV_TPU" in existing_config:
        updated_config = re.sub(r"# BEGIN_DEV_TPU\n(.*?\n)*?# END_DEV_TPU\n", "", existing_config, flags=re.DOTALL)
        config_path.write_text(updated_config)
        logger.info(f"Removed SSH configuration for dev-tpu-{username}")


def get_git_files(local_path: str, include_modified: bool = True) -> list[str]:
    """Get list of files to sync based on git tracking.

    Args:
        local_path: Local directory path
        include_modified: If True, include modified/untracked files

    Returns:
        List of file paths relative to local_path
    """
    files = set()

    # Get all tracked files
    result = subprocess.run(["git", "ls-files", "-z"], cwd=local_path, check=True, capture_output=True, text=True)
    tracked = [f for f in result.stdout.split("\0") if f.strip()]
    files.update(tracked)

    if include_modified:
        # Get modified files (staged and unstaged)
        result = subprocess.run(
            ["git", "diff", "--name-only", "-z"], cwd=local_path, check=True, capture_output=True, text=True
        )
        modified = [f for f in result.stdout.split("\0") if f.strip()]
        files.update(modified)

        # Get staged files
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "-z"], cwd=local_path, check=True, capture_output=True, text=True
        )
        staged = [f for f in result.stdout.split("\0") if f.strip()]
        files.update(staged)

    return sorted(files)


def sync_files_to_remote(target_host: str, local_path: str = ".") -> None:
    """Sync files to remote host using git-aware file selection.

    Args:
        target_host: SSH host alias or address
        local_path: Local directory to sync from
    """
    local_path_obj = Path(local_path).resolve()
    if not local_path_obj.exists():
        raise RuntimeError(f"Local path does not exist: {local_path}")

    # Create remote directory
    subprocess.run(["ssh", target_host, "mkdir", "-p", "/home/$USER/marin"], check=True, capture_output=True)

    # Get files to sync
    files = get_git_files(local_path, include_modified=True)

    if not files:
        logger.info("No files to sync")
        return

    # Write files to temporary file for rsync
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(files))
        files_from_path = f.name

    try:
        # Sync files using rsync
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

        logger.info(f"Syncing {len(files)} tracked files...")

        subprocess.run(rsync_cmd, check=True)
        logger.info("Sync completed successfully")
    finally:
        # Clean up temporary file
        os.unlink(files_from_path)


def sync_environment(target_host: str, local_path: str = ".") -> None:
    """Sync local environment to remote TPU."""
    logger.info(f"Syncing environment to {target_host}")

    # Sync all tracked files
    sync_files_to_remote(target_host, local_path)

    # Install uv and sync dependencies
    setup_script = """
set -x
cd /home/$USER/marin

echo "Setting up development environment..."
curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

echo "Installing dependencies..."
cd /home/$USER/marin
uv sync --extra=tpu
"""

    logger.info("Setting up remote environment...")
    subprocess.run(["ssh", target_host, "bash", "-c", setup_script], check=False)

    logger.info("Environment sync completed successfully")


@contextmanager
def hold_tpu_allocation(
    username: str, config_file: str, sync_path: str = ".", tpu_type: str = "v4-8", duration_minutes: int = 480
) -> Generator[dict[str, str], None, None]:
    """Context manager that holds a TPU allocation until the context exits.

    Uses a Ray actor to manage the TPU allocation lifecycle.
    """
    logger.info("START_DEV_TPU: Beginning TPU allocation")

    actor = None

    try:
        with ray_utils.ray_dashboard(config_file, ray_init=True):
            # Create TPU allocation actor
            logger.info("Creating TPU allocation actor")
            actor = ray.remote(resources={"TPU": 4, "TPU-v4-8-head": 1})(TPUAllocationActor).remote(username, tpu_type)

            while True:
                allocation_info = ray.get(actor.host_info.remote(), timeout=60)

                logger.info("Setting up SSH configuration")
                add_ssh_host_config(allocation_info["hostname"], allocation_info["ip_address"], username)
                ssh_configured = True

                logger.info("Syncing environment")
                sync_environment(f"dev-tpu-{username}", sync_path)

                print("TPU allocated successfully!")
                print(f"Hostname: {allocation_info['hostname']}")
                print(f"IP Address: {allocation_info['ip_address']}")
                print(f"SSH alias: dev-tpu-{username}")
                print(f"\nTo connect: ssh dev-tpu-{username}")

                logger.info("START_DEV_TPU: TPU allocation complete, yielding control")

                # Yield control to caller
                yield allocation_info

    finally:
        try:
            ray.kill(actor)
        except Exception as e:
            logger.warning(f"Failed to kill actor: {e}")
        if ssh_configured:
            remove_ssh_host_config(username)


class Context:
    def __init__(self):
        self.verbose: bool = False
        self.config_file: Optional[str] = None
        self.config_obj: Optional[RayClusterConfig] = None


@click.group()
@click.option("--config", help="Path to cluster config file")
@click.option("--cluster", help="Cluster name to connect to")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, config, cluster, verbose):
    """Development TPU management for Ray clusters."""
    ctx.ensure_object(Context)
    if cluster:
        config = find_config_by_region(cluster)

    ctx.obj.config_file = config
    ctx.obj.verbose = verbose

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG if verbose else logging.INFO
    )

    if config:
        ctx.obj.config_obj = RayClusterConfig.from_yaml(config)


@cli.command("allocate")
@click.option("--tpu-type", default="v4-8", help="TPU type (default: v4-8)")
@click.option("--sync-path", default=".", help="Local path to sync (default: current directory)")
@click.option("--username", help="Username (default: current user)")
@click.option("--duration", default=480, help="Allocation duration in minutes (default: 480)")
@click.pass_context
def allocate(ctx, tpu_type, sync_path, username, duration):
    """Allocate a development TPU. Holds until Ctrl-C."""
    if not ctx.obj.config_file:
        print("Error: --config required", file=sys.stderr)
        sys.exit(1)

    if not username:
        username = getpass.getuser()

    if tpu_type not in ["v4-8", "v5p-8"]:
        print(f"Warning: TPU type {tpu_type} may not be supported", file=sys.stderr)

    print(f"Allocating development TPU for {username}...")
    print(f"TPU type: {tpu_type}")
    print(f"Duration: {duration} minutes")

    with hold_tpu_allocation(username, ctx.obj.config_file, sync_path, tpu_type, duration):
        print("\nTPU allocation is active. Press Ctrl-C to release...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived Ctrl-C, releasing TPU allocation...")

    print("TPU allocation released.")


@cli.command("connect")
@click.option("--username", help="Username (default: current user)")
@click.pass_context
def connect(ctx, username):
    """Connect to development TPU via SSH."""
    if not username:
        username = getpass.getuser()

    host_alias = f"dev-tpu-{username}"

    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print("You need to run 'allocate' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {host_alias}...")
    subprocess.run(["ssh", host_alias])


@cli.command("execute", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--username", help="Username (default: current user)")
@click.pass_context
def execute(ctx, command, username, sync_path="."):
    """Execute a command on the development TPU.

    This command will:
    1. Sync any local changes to the TPU (unless --no-sync is specified)
    2. Execute the provided command in the /home/$USER/marin directory
    3. Return the exit code from the remote command

    Examples:
        uv run scripts/ray/dev_tpu.py execute -- uv run python train.py
        uv run scripts/ray/dev_tpu.py execute -- uv run pytest tests/ -v
    """
    if not username:
        username = getpass.getuser()

    host_alias = f"dev-tpu-{username}"

    # Check if SSH config exists
    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print("You need to run 'allocate' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    # Sync files unless --no-sync is specified
    print(f"Syncing local changes to {host_alias}...")
    sync_files_to_remote(host_alias, sync_path)

    # Join command tuple into a single string
    command_str = " ".join(command)
    full_cmd = f"ssh -t {host_alias} 'source $HOME/.local/bin/env && cd marin && {command_str}'"
    print(full_cmd)
    result = subprocess.run(full_cmd, check=False, shell=True)

    sys.exit(result.returncode)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()


if __name__ == "__main__":
    main()
