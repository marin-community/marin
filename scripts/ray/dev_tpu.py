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

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import atexit
import click
import getpass
import logging
import os
import ray
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import yaml

import ray.cloudpickle as cloudpickle

import marin.utils
from fray.cluster import ray as ray_utils
from marin.cluster.config import RayClusterConfig, find_config_by_region
from marin.utils import _hacky_remove_tpu_lockfile

from fray.cluster.ray.auth import maybe_fetch_local_ray_token

# Register `marin.utils` by value, so it can work over `ray.remote` without `marin` being installed on the worker.
# See also #1786 / #1789.
cloudpickle.register_pickle_by_value(marin.utils)

logger = logging.getLogger(__name__)


_RAY_DISCONNECT_SUBSTRINGS = (
    "grpc client is shut down",
    "attempted to reconnect to a session that has already been cleaned up",
    "failed during this or a previous request",
)


class RayDisconnectError(RuntimeError):
    """Raised when Ray disconnects and the allocation should be recreated."""


@dataclass(frozen=True)
class TPUAllocation:
    host_info: dict[str, str]
    actor: object


def run_logged(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess command with logging.

    Args:
        cmd: Command to run (list or string)
        **kwargs: Arguments passed to subprocess.run

    Returns:
        CompletedProcess result from subprocess.run
    """
    if isinstance(cmd, list):
        cmd_str = shlex.join(cmd)
    else:
        cmd_str = cmd

    logger.info(f"Running command: {cmd_str}")
    return subprocess.run(cmd, **kwargs)


# Default environment variables to forward to remote
DEFAULT_ENV_VARS = [
    "HF_TOKEN",
    "WANDB_API_KEY",
    "MARIN_PREFIX",
    "OPENAI_API_KEY",
    "GCLOUD_PROJECT",
    "GCLOUD_TOKEN_PATH",
    "RAY_ADDRESS",
    "RAY_API_SERVER_ADDRESS",
    "RAY_DASHBOARD_ADDRESS",
    "RAY_AUTH_MODE",
    "RAY_AUTH_TOKEN_PATH",
    "WANDB_MODE",
    "RUN_ID",
]


def build_env_dict(extra_env: list[str] | None = None, forward_all: bool = False) -> dict[str, str]:
    """Build environment variable dictionary for forwarding."""
    # Start with all environment variables if requested, otherwise just defaults
    env_dict = {}
    for var in DEFAULT_ENV_VARS:
        if os.environ.get(var) is not None:
            env_dict[var] = os.environ[var]

    if forward_all:
        env_dict = dict(os.environ)

    CONFIG_FILES = [".levanter.yaml", ".marin.yaml", ".config"]
    for config_file in CONFIG_FILES:
        if os.path.exists(config_file):
            logger.info(f"Injecting environment variables from {config_file}")
            try:
                config_yaml = yaml.safe_load(open(config_file).read())
            except Exception as e:
                raise RuntimeError("Failed to load config from environment") from e

            for key, value in config_yaml.get("env", {}).items():
                env_dict[key] = str(value)

    # Add extra environment variables from command line (these override existing env vars)
    if extra_env:
        for env_var in extra_env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                env_dict[key] = value
            else:
                # Just the key, get value from current environment
                if os.environ.get(env_var) is not None:
                    env_dict[env_var] = os.environ[env_var]

    return env_dict


def build_env_string(env_dict: dict[str, str]) -> str:
    """Build properly escaped environment variable string for shell."""
    if not env_dict:
        return ""

    env_parts = []
    for key, value in env_dict.items():
        # Escape both key and value for shell safety
        escaped_key = shlex.quote(key)
        escaped_value = shlex.quote(value)
        env_parts.append(f"{escaped_key}={escaped_value}")

    return " ".join(env_parts)


def build_ssh_command(
    host_alias: str,
    command: str,
    env_dict: dict[str, str] | None = None,
    working_dir: str = "marin",
) -> list[str]:
    """Build SSH command with proper cleanup and environment forwarding.

    Always wraps the user command in bash -c for consistent shell behavior.
    User should NOT include 'bash -c' in their command - this will raise an error.
    """
    env_string = build_env_string(env_dict or {})

    # Error if user tries to use bash -c themselves
    if command.strip().startswith("bash -c"):
        raise ValueError(
            "Do not include 'bash -c' in your command. "
            "The script automatically wraps commands in bash -c. "
            f"Use: execute -- {command.strip()[7:].strip()} "
            f"instead of: execute -- {command}"
        )

    # Always wrap user command in bash -c for consistent shell behavior
    remote_cmd_parts = [
        "exec bash -c",
        shlex.quote(
            f"""
            set -e
            trap 'jobs -p | xargs -r kill' EXIT HUP INT TERM
            source $HOME/.local/bin/env
            cd {shlex.quote(working_dir)}
            {env_string} exec bash -c {shlex.quote(command)}
        """
        ),
    ]

    remote_cmd = " ".join(remote_cmd_parts)

    return ["ssh", "-t", host_alias, remote_cmd]


def kill_ssh_session(host_alias: str) -> None:
    """Kill SSH session using control socket."""
    try:
        subprocess.run(["ssh", "-O", "exit", host_alias], capture_output=True, check=False, timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout killing SSH control connection to {host_alias}")
    except Exception as e:
        logger.debug(f"Failed to kill SSH control connection to {host_alias}: {e}")


def _iter_exception_chain(exc: BaseException) -> Generator[BaseException, None, None]:
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current
        stack.append(current.__cause__)
        stack.append(current.__context__)


def _is_ray_disconnect_error(exc: BaseException) -> bool:
    for err in _iter_exception_chain(exc):
        message = str(err).lower()
        if any(substr in message for substr in _RAY_DISCONNECT_SUBSTRINGS):
            return True
    return False


def _monitor_allocation(actor: object, heartbeat_interval: float, heartbeat_timeout: float) -> None:
    while True:
        time.sleep(heartbeat_interval)
        try:
            ray.get(actor.heartbeat.remote(), timeout=heartbeat_timeout)
        except Exception as e:
            if _is_ray_disconnect_error(e):
                raise RayDisconnectError("Ray disconnect detected.") from e
            raise


class TPUAllocationActor:
    """Actor that holds TPU allocation and manages TPU info."""

    def __init__(self, username: str, tpu_name: str, tpu_type: str = "v4-8"):
        """Initialize the actor."""
        self.username = username
        self.tpu_name = tpu_name
        self.tpu_type = tpu_type

    def host_info(self) -> dict:
        """Fetch and return TPU metadata from Google metadata server."""
        import socket

        import requests

        # Fetch TPU metadata from Google metadata server
        headers = {"Metadata-Flavor": "Google"}
        metadata_base = "http://metadata.google.internal/computeMetadata/v1"

        # Get external IP
        response = requests.get(
            f"{metadata_base}/instance/network-interfaces/0/access-configs/0/external-ip",
            headers=headers,
        )
        ip_address = response.text.strip()

        # Get hostname
        hostname = socket.gethostname()

        # Get actual TPU instance ID from metadata (this is the gcloud TPU name)
        response = requests.get(
            f"{metadata_base}/instance/attributes/instance-id",
            headers=headers,
        )
        gcloud_tpu_name = response.text.strip()

        # Get zone (e.g., us-central2-b)
        response = requests.get(
            f"{metadata_base}/instance/zone",
            headers=headers,
        )
        # Zone comes back as projects/PROJECT_ID/zones/ZONE, extract just the zone
        zone_path = response.text.strip()
        gcloud_zone = zone_path.split("/")[-1]

        # Extract region from zone (e.g., us-central2-b -> us-central2)
        gcloud_region = "-".join(gcloud_zone.split("-")[:-1])

        logger.info(
            f"TPU info fetched - hostname: {hostname}, IP: {ip_address}, "
            f"gcloud TPU name: {gcloud_tpu_name}, zone: {gcloud_zone}, region: {gcloud_region}"
        )

        return {
            "hostname": hostname,
            "ip_address": ip_address,
            "username": self.username,
            "tpu_name": self.tpu_name,
            "gcloud_tpu_name": gcloud_tpu_name,
            "gcloud_zone": gcloud_zone,
            "gcloud_region": gcloud_region,
            "tpu_type": self.tpu_type,
            "error": "",
        }

    def heartbeat(self) -> str:
        return f"TPU allocation '{self.tpu_name}' active for {self.username}"

    def __del__(self):
        # delete the work directories in the background, use the ls to make sure we don't
        # accidentally run this on our local machine
        subprocess.Popen(
            [
                "bash",
                "-c",
                "ls /dev/accel* && (rm -rf $HOME/marin/; rm -rf $HOME/.cache/; sudo rm -f /tmp/libtpu_lockfile)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _hacky_remove_tpu_lockfile()


def add_ssh_host_config(
    hostname: str,
    ip_address: str,
    username: str,
    tpu_name: str,
    gcloud_tpu_name: str,
    gcloud_zone: str,
) -> None:
    """Add SSH host configuration."""
    config_path = Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(exist_ok=True)

    # try using gcloud compute tpu-vm ssh to forward keys
    logger.info("Tryng to setup SSH keys via gcloud. You may register your key manually if this fails.")
    gcloud_ssh_cmd = f"gcloud compute tpus tpu-vm ssh {gcloud_tpu_name} --zone={gcloud_zone} -- hostname"
    try:
        run_logged(
            shlex.split(gcloud_ssh_cmd),
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("SSH keys set up successfully via gcloud")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error when running gcloud compute tpu-vm ssh: {e.stdout}, error: {e.stderr}")
        logger.warning("gcloud compute tpu-vm ssh failed to set up SSH keys")

    # Check if Google Compute Engine SSH key exists
    gce_key_path = Path.home() / ".ssh" / "google_compute_engine"
    if not gce_key_path.exists():
        logger.warning(f"Google Compute Engine SSH key not found at {gce_key_path}")
        logger.warning("SSH may fail if your key isn't available on the VM.")
        logger.warning("You can add it at https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys")

    host_alias = f"dev-tpu-{tpu_name}"

    ssh_config_entry = f"""
# BEGIN_DEV_TPU_{tpu_name.upper()}
Host {host_alias}
    HostName {ip_address}
    HostKeyAlias compute.{hostname}
    StrictHostKeyChecking no
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
        updated_config = updated_config.strip() + "\n"
        config_path.write_text(updated_config)
        logger.info(f"Removed SSH configuration for dev-tpu-{tpu_name}")


def list_tracked_files(local_path: Path) -> list[str]:
    """List all files that git would track (excluding gitignored files).

    This includes tracked, modified, staged, and untracked files.
    git ls-files already handles recursive directory traversal.
    """
    result = run_logged(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=local_path,
        check=True,
        capture_output=True,
        text=True,
    )
    # Split on null byte and filter empty strings
    all_files = [f for f in result.stdout.split("\0") if f.strip()]
    return sorted(all_files)


def sync_to_remote(target_host: str, local_path: os.PathLike | str = ".") -> None:
    local_path = Path(local_path).resolve()
    sync_files = list_tracked_files(local_path)

    # Here we use the relative path "marin" because the ssh/rsync command use a relative path to
    # the remote user's home directory.
    run_logged(["ssh", target_host, "mkdir", "-p", "marin"], check=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
        f.write("\n".join(sync_files))
        f.flush()  # Ensure file is written before rsync reads it
        files_from_path = f.name
        rsync_cmd = [
            "rsync",
            "-az",
            "--delete",
            "--progress",
            "--exclude=.git",
            "--exclude=.venv",
            "--files-from",
            files_from_path,
            "-r",  # Explicitly activate recursion -- disabled by --files-from
            f"{local_path}/",
            f"{target_host}:marin/",
        ]

        logger.info(f"Syncing {len(sync_files)} files (git-tracked, modified, and important untracked)...")
        run_logged(rsync_cmd, check=True)
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
    uv sync --all-packages --extra=tpu --python=3.11 || true
"""
    logger.info("Setting up remote environment...")
    run_logged(["ssh", target_host, "bash", "-c", setup_script], check=True)
    logger.info("Environment setup completed")


@contextmanager
def hold_tpu_allocation(
    username: str,
    tpu_name: str,
    config_file: str,
    sync_path: str = ".",
    tpu_type: str = "v4-8",
) -> Generator[TPUAllocation, None, None]:
    """Context manager that holds a TPU allocation until the context exits.

    Uses a Ray actor to manage the TPU allocation lifecycle.
    """
    logger.info(f"{tpu_name}: Beginning TPU allocation")

    from fray.cluster.ray import DashboardConfig

    with ray_utils.ray_dashboard(DashboardConfig.from_cluster(config_file, ray_init=True)):
        actor = None
        try:
            logger.info(f"Creating TPU allocation actor for {tpu_name}")
            actor = ray.remote(resources={"TPU": 4, f"TPU-{tpu_type}-head": 1})(TPUAllocationActor).remote(
                username, tpu_name, tpu_type
            )

            logger.info("Waiting up to 10 minutes for TPU to be ready...")
            host_info = ray.get(actor.host_info.remote(), timeout=600)
            logger.info("TPU allocated successfully!")
            print(f"Hostname: {host_info['hostname']}")
            print(f"IP Address: {host_info['ip_address']}")
            print(f"User TPU name: {tpu_name}")
            print(f"GCloud TPU name: {host_info['gcloud_tpu_name']}")
            print(f"GCloud Zone: {host_info['gcloud_zone']}")
            print(f"GCloud Region: {host_info['gcloud_region']}")

            logger.info("Setting up SSH configuration")
            add_ssh_host_config(
                host_info["hostname"],
                host_info["ip_address"],
                username,
                tpu_name,
                host_info["gcloud_tpu_name"],
                host_info["gcloud_zone"],
            )

            logger.info("Syncing environment")
            try:
                sync_to_remote(f"dev-tpu-{tpu_name}", sync_path)
                setup_remote_environment(f"dev-tpu-{tpu_name}")
            except Exception as e:
                logger.warning(f"Environment setup failed, keeping TPU allocation alive. {e}")

            print(f"SSH alias: dev-tpu-{tpu_name}")
            yield TPUAllocation(host_info=host_info, actor=actor)
        except Exception as e:
            if _is_ray_disconnect_error(e):
                logger.error("Ray disconnect detected during allocation; releasing allocation.", exc_info=True)
                raise RayDisconnectError("Ray disconnect detected during allocation.") from e
            logger.error(f"Error during TPU allocation or setup: {e}", exc_info=True)
            raise
        finally:
            if actor is not None:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as e:
                    logger.debug(f"Failed to kill TPU allocation actor: {e}")
            remove_ssh_host_config(tpu_name)
            if ray.is_initialized():
                try:
                    ray.shutdown()
                except Exception as e:
                    logger.debug(f"Failed to shutdown Ray: {e}")


class Context:
    def __init__(self):
        self.verbose: bool = False
        self.config_file: str | None = None
        self.config_obj: RayClusterConfig | None = None
        self.tpu_name: str | None = None
        self.config_data: dict | None = None


def _infer_tpu_type_from_config(config_data: dict | None) -> str | None:
    if not config_data:
        return None

    try:
        return config_data["available_node_types"]["tpu_worker"]["node_config"]["acceleratorType"]
    except KeyError:
        return None


def _default_tpu_name_from_config(config_data: dict | None, username: str) -> str | None:
    if not config_data:
        return None
    cluster_name = config_data.get("cluster_name")
    if not cluster_name:
        return None
    return f"dev-{cluster_name}-{username}"


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
    ctx.obj.tpu_name = tpu_name
    ctx.obj.verbose = verbose

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG if verbose else logging.INFO
    )

    if config:
        ctx.obj.config_obj = RayClusterConfig.from_yaml(config)
        with open(config, "r", encoding="utf-8") as f:
            ctx.obj.config_data = yaml.safe_load(f)
        gcp_project = (ctx.obj.config_data or {}).get("provider", {}).get("project_id")
        token_path = maybe_fetch_local_ray_token(gcp_project=gcp_project)
        os.environ["RAY_AUTH_TOKEN_PATH"] = token_path
        os.environ["RAY_AUTH_MODE"] = "token"
    if ctx.obj.tpu_name is None:
        username = getpass.getuser()
        ctx.obj.tpu_name = _default_tpu_name_from_config(ctx.obj.config_data, username) or username


@cli.command("allocate")
@click.option("--tpu-type", help="TPU type")
@click.option("--sync-path", default=".", help="Local path to sync")
@click.option("--username", help="Username to use for ssh", default=getpass.getuser())
@click.option(
    "--heartbeat-interval",
    default=30.0,
    type=float,
    show_default=True,
    help="Seconds between Ray heartbeat checks.",
)
@click.option(
    "--heartbeat-timeout",
    default=15.0,
    type=float,
    show_default=True,
    help="Seconds to wait for Ray heartbeat responses.",
)
@click.option(
    "--retry/--no-retry",
    "--retry-on-ray-disconnect/--no-retry-on-ray-disconnect",
    "retry_on_ray_disconnect",
    default=True,
    show_default=True,
    help="Recreate the allocation if the Ray client disconnects.",
)
@click.option(
    "--retry-backoff",
    default=30.0,
    type=float,
    show_default=True,
    help="Seconds to wait before recreating after a Ray disconnect.",
)
@click.pass_context
def allocate(
    ctx, tpu_type, sync_path, username, heartbeat_interval, heartbeat_timeout, retry_on_ray_disconnect, retry_backoff
):
    """Allocate a development TPU. Holds until Ctrl-C."""
    if not ctx.obj.config_file:
        print("Error: --config required", file=sys.stderr)
        sys.exit(1)

    if not username:
        username = getpass.getuser()

    tpu_name = ctx.obj.tpu_name

    if not tpu_type:
        inferred_tpu_type = _infer_tpu_type_from_config(ctx.obj.config_data)
        if inferred_tpu_type:
            tpu_type = inferred_tpu_type
        else:
            raise click.ClickException("Could not infer TPU type from config; please specify --tpu-type.")

    if tpu_type not in ["v4-8", "v5p-8"]:
        print(f"Warning: TPU type {tpu_type} may not be supported", file=sys.stderr)

    print(f"Allocating development TPU '{tpu_name}' for {username}...")
    print(f"TPU type: {tpu_type}")

    while True:
        try:
            with hold_tpu_allocation(
                username,
                tpu_name,
                ctx.obj.config_file,
                sync_path,
                tpu_type,
            ) as allocation:
                print("\nTPU allocation is active. Press Ctrl-C to release...")
                _monitor_allocation(allocation.actor, heartbeat_interval, heartbeat_timeout)
        except KeyboardInterrupt:
            print("\nReceived Ctrl-C, releasing TPU allocation...")
            break
        except RayDisconnectError:
            logger.error("Ray disconnect detected; releasing allocation.")
            if not retry_on_ray_disconnect:
                break
            if retry_backoff > 0:
                print(f"Recreating TPU allocation in {retry_backoff} seconds...")
                time.sleep(retry_backoff)
            continue

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
    env = build_env_dict()
    env_string = build_env_string(env)

    config_path = Path.home() / ".ssh" / "config"
    if not config_path.exists() or f"Host {host_alias}" not in config_path.read_text():
        print(f"Error: SSH configuration for {host_alias} not found", file=sys.stderr)
        print(f"You need to run 'allocate --tpu-name {tpu_name}' first to set up the TPU", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {host_alias}...")
    cmd = shlex.quote(f"source $HOME/.local/bin/env && cd marin && {env_string} exec bash")
    run_logged(["ssh", host_alias, "-t", "bash", "-c", cmd])


@cli.command("setup_env")
@click.pass_context
def setup_env(ctx):
    """Set up the remote environment on the development TPU."""
    tpu_name = ctx.obj.tpu_name
    host_alias = f"dev-tpu-{tpu_name}"
    setup_remote_environment(host_alias)


@cli.command("execute", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--username", help="Username to use for ssh", default=getpass.getuser())
@click.option("--sync-path", default=".", help="Local path to sync")
@click.option("--env", "-e", multiple=True, help="Environment variables to forward (KEY=VALUE or KEY)")
@click.option("--forward-all-env", is_flag=True, help="Forward all environment variables")
@click.pass_context
def execute(ctx, command, username, sync_path, env, forward_all_env):
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

    # Build environment variables
    env_dict = build_env_dict(extra_env=list(env), forward_all=forward_all_env)

    command_str = shlex.join(command)
    ssh_cmd = build_ssh_command(host_alias, command_str, env_dict)

    print(f"Running: {ssh_cmd}")
    ssh_session = subprocess.Popen(ssh_cmd)
    atexit.register(lambda: kill_ssh_session(host_alias))
    result = ssh_session.wait()
    sys.exit(result)


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
        ]

        event_path = str(event.src_path)
        if any(pattern in event_path for pattern in skip_patterns):
            return

        # Debounce rapid file changes
        if self._timer:
            self._timer.cancel()

        self._timer = threading.Timer(self.debounce_seconds, self._trigger_callback)
        self._timer.start()

    def _trigger_callback(self):
        self.callback()


class RemoteProcessManager:
    """Run a remote process, synchronizing and restarting on demand."""

    def __init__(
        self,
        host_alias: str,
        command_str: str,
        sync_path: str,
        env_dict: dict[str, str] | None = None,
    ):
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._command_str = command_str
        self._host_alias = host_alias
        self._sync_path = sync_path
        self._env_dict = env_dict or {}

    def __call__(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                self.kill()

            print(f"Syncing changes to {self._host_alias}...")
            sync_to_remote(self._host_alias, self._sync_path)

            ssh_cmd = build_ssh_command(self._host_alias, self._command_str, self._env_dict)
            print(f"Running: {self._command_str}")
            self._process = subprocess.Popen(ssh_cmd, stdin=subprocess.DEVNULL)

    def check_status(self):
        if self._process and self._process.poll() is not None:
            print(f"Remote process exited with code {self._process.returncode}")

    def kill(self):
        if self._process and self._process.poll() is None:
            print("Killing remote...")
            # First try to kill via SSH control socket
            kill_ssh_session(self._host_alias)

            # Give it a moment to clean up
            start_time = time.time()
            while self._process.poll() is None and time.time() - start_time < 3:
                time.sleep(0.1)

            # If still running, send SIGINT to the subprocess
            if self._process.poll() is None:
                self._process.send_signal(signal.SIGINT)
                start_time = time.time()
                while self._process.poll() is None and time.time() - start_time < 5:
                    time.sleep(0.1)

            # Final resort: terminate
            if self._process.poll() is None:
                print("Remote process did not exit, terminating...")
                self._process.terminate()

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.kill()
        except Exception:
            pass


@cli.command("watch", context_settings={"ignore_unknown_options": True})
@click.argument("command", nargs=-1, required=True)
@click.option("--username", help="Username to connect as", default=getpass.getuser())
@click.option("--sync-path", default=".", help="Local path to sync")
@click.option("--debounce", default=1, help="Debounce time for file changes in seconds")
@click.option("--env", "-e", multiple=True, help="Environment variables to forward (KEY=VALUE or KEY)")
@click.option("--forward-all-env", is_flag=True, help="Forward all environment variables")
@click.pass_context
def watch(ctx, command, username, sync_path, debounce, env, forward_all_env):
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

    # Build environment variables
    env_dict = build_env_dict(extra_env=list(env), forward_all=forward_all_env)

    command_str = shlex.join(command)

    process_mgr = RemoteProcessManager(host_alias, command_str, sync_path, env_dict)
    atexit.register(process_mgr.kill)  # ensure we clean up on exit
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
        process_mgr.kill()

    observer.join()
    print("Watch mode stopped.")


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cli()


if __name__ == "__main__":
    main()
