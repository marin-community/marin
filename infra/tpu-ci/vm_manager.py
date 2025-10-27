#!/usr/bin/env python3
#
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
# /// script
# dependencies = [
#   "click",
#   "google-cloud-tpu",
# ]
# ///

"""
TPU VM Manager for GitHub Actions CI

Maintains a pool of preemptible TPU VMs with GitHub Actions runners.
Continuously monitors and ensures the desired number of VMs are running:
- Creates new VMs if count is below desired
- Deletes preempted/failed VMs
- Each VM auto-registers as a GitHub Actions runner via startup script
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import click
import config
from google.cloud import tpu_v2


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def get_startup_script() -> str:
    """
    Generate startup script for TPU VMs.

    Installs Docker and GitHub Actions runner (ephemeral mode).
    Fetches GitHub token from Secret Manager at runtime.
    """
    return f"""#!/bin/bash
set -e

echo "=== TPU VM Setup Starting ==="

# Read configuration from VM metadata
PROJECT_ID=$(curl -sSf -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/project/project-id)
ZONE=$(curl -sSf -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
INSTANCE_NAME=$(curl -sSf -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/name)

echo "Instance: $INSTANCE_NAME in $ZONE"

# Kill unattended-upgrades to avoid apt lock conflicts
echo "Stopping unattended-upgrades..."
systemctl stop unattended-upgrades.service || true
systemctl disable unattended-upgrades.service || true
killall -9 unattended-upgrade || true
killall -9 apt apt-get || true

# Wait briefly for locks to be released
sleep 2

# Install Docker if not already present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y docker.io jq curl
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed, skipping..."
    # Ensure jq and curl are installed
    apt-get update
    apt-get install -y jq curl
fi

# Ensure Docker is running
systemctl enable docker
systemctl start docker

# Pull TPU CI image to speed up first test run
echo "Pre-pulling TPU CI Docker image..."
docker pull {config.DOCKER_IMAGE_FULL} || true

# Install GitHub Actions runner
echo "Installing GitHub Actions runner..."
RUNNER_VERSION="2.311.0"
RUNNER_USER="github-runner"

# Create runner user if it doesn't exist
if ! id -u $RUNNER_USER > /dev/null 2>&1; then
    useradd -m -s /bin/bash $RUNNER_USER
fi
usermod -aG docker $RUNNER_USER

# Download and extract runner if not already present
cd /home/$RUNNER_USER
if [ ! -f config.sh ]; then
    curl -o actions-runner-linux-x64-$RUNNER_VERSION.tar.gz -L \\
      https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
    tar xzf actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
    rm actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
fi
chown -R $RUNNER_USER:$RUNNER_USER /home/$RUNNER_USER

# Get GitHub registration token from Secret Manager
echo "Fetching GitHub token from Secret Manager..."
GITHUB_TOKEN=$(gcloud secrets versions access latest \\
  --secret="tpu-ci-github-token" \\
  --project="$PROJECT_ID")

# Get runner registration token
REGISTRATION_TOKEN=$(curl -s -X POST \\
  -H "Accept: application/vnd.github+json" \\
  -H "Authorization: Bearer $GITHUB_TOKEN" \\
  https://api.github.com/repos/{config.GITHUB_ORG}/{config.GITHUB_REPO}/actions/runners/registration-token \\
  | jq -r .token)

# Configure runner (ephemeral mode)
echo "Configuring GitHub Actions runner..."
cd /home/$RUNNER_USER

# Remove existing runner configuration if present
if [ -f .runner ]; then
    echo "Removing existing runner configuration..."
    ./svc.sh stop || true
    ./svc.sh uninstall || true
    sudo -u $RUNNER_USER ./config.sh remove --token $REGISTRATION_TOKEN || true
fi

sudo -u $RUNNER_USER ./config.sh \\
  --url https://github.com/{config.GITHUB_ORG}/{config.GITHUB_REPO} \\
  --token $REGISTRATION_TOKEN \\
  --name "tpu-$INSTANCE_NAME" \\
  --labels {','.join(config.RUNNER_LABELS)} \\
  --work _work \\
  --unattended

# Install and start runner as systemd service
echo "Installing runner service..."
cd /home/$RUNNER_USER
# Uninstall existing service if present to avoid "already exists" error
./svc.sh uninstall || true
./svc.sh install $RUNNER_USER
./svc.sh start

echo "=== TPU VM Setup Complete ==="
"""


def create_tpu_vm(index: int) -> None:
    """Create a single preemptible TPU VM with GitHub Actions runner."""
    vm_name = f"{config.TPU_VM_PREFIX}-{index}"

    logging.info(f"Creating TPU VM: {vm_name}")

    # Write startup script to temp file
    startup_script_path = Path("/tmp/tpu-startup-script.sh")
    startup_script_path.write_text(get_startup_script())

    # Create TPU VM with labels for automatic cleanup
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "create",
        vm_name,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--accelerator-type",
        config.TPU_ACCELERATOR_TYPE,
        "--version",
        config.TPU_VERSION,
        "--preemptible",
        "--metadata-from-file",
        f"startup-script={startup_script_path}",
        "--scopes",
        "https://www.googleapis.com/auth/cloud-platform",
        "--labels",
        "tpu-ci-component=runner,tpu-ci-managed=true",
    ]

    result = run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        # VM might already exist
        if "already exists" in result.stderr:
            logging.info(f"TPU VM {vm_name} already exists")
        else:
            logging.error(f"Failed to create TPU VM {vm_name}: {result.stderr}")
            raise RuntimeError(f"Failed to create TPU VM {vm_name}")
    else:
        logging.info(f"✓ Created TPU VM: {vm_name}")


def delete_tpu_vm(vm_name: str) -> None:
    """Delete a single TPU VM."""
    logging.info(f"Deleting TPU VM: {vm_name}")

    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        vm_name,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--quiet",
    ]

    result = run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0 and "not found" not in result.stderr:
        logging.error(f"Failed to delete TPU VM {vm_name}: {result.stderr}")
        raise RuntimeError(f"Failed to delete TPU VM {vm_name}")

    logging.info(f"✓ Deleted TPU VM: {vm_name}")


def list_tpu_vms() -> list[dict]:
    """List all TPU VMs with our labels, returning name and state."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--format",
        "json",
        "--filter",
        "labels.tpu-ci-managed=true",
    ]

    result = run(cmd, capture_output=True, text=True, check=True)

    if not result.stdout.strip():
        return []

    vms = json.loads(result.stdout)
    return [{"name": vm["name"], "state": vm.get("state", "UNKNOWN")} for vm in vms]


def get_next_available_index(existing_vms: list[dict]) -> int:
    """Find the lowest available index for a new VM."""
    existing_indices = set()
    for vm in existing_vms:
        parts = vm["name"].rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            existing_indices.add(int(parts[1]))

    # Find first missing index from 0 to TPU_VM_COUNT-1
    for i in range(config.TPU_VM_COUNT):
        if i not in existing_indices:
            return i

    # All slots filled, return next number
    return config.TPU_VM_COUNT


def ensure_tpu_vms(tpu_client: tpu_v2.TpuClient):
    """
    Ensure desired number of TPU VMs are running.
    - Delete any preempted/failed VMs
    - Create new VMs if count is below desired
    """
    vms = list_tpu_vms()

    logging.info(f"Found {len(vms)} TPU VMs")

    # Delete preempted/failed VMs
    bad_states = ["PREEMPTED", "TERMINATED", "FAILED"]
    for vm in vms:
        if vm["state"] in bad_states:
            logging.warning(f"TPU {vm['name']} is in state {vm['state']}, deleting...")
            try:
                delete_tpu_vm(vm["name"])
                vms.remove(vm)
            except Exception as e:
                logging.error(f"Failed to delete {vm['name']}: {e}")

    # Count healthy VMs (READY or CREATING)
    healthy_states = ["READY", "CREATING"]
    healthy_count = sum(1 for vm in vms if vm["state"] in healthy_states)

    logging.info(f"Healthy TPU VMs: {healthy_count}/{config.TPU_VM_COUNT}")

    # Create missing VMs
    needed = config.TPU_VM_COUNT - healthy_count
    if needed > 0:
        logging.info(f"Creating {needed} new TPU VMs...")
        for _ in range(needed):
            index = get_next_available_index(vms)
            try:
                create_tpu_vm(index)
                vms.append({"name": f"{config.TPU_VM_PREFIX}-{index}", "state": "CREATING"})
            except Exception as e:
                logging.error(f"Failed to create TPU VM: {e}")


def monitor_loop(tpu_client: tpu_v2.TpuClient):
    """Main monitoring loop - runs indefinitely."""
    while True:
        try:
            ensure_tpu_vms(tpu_client)
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)

        time.sleep(60)


@click.group()
def cli():
    """TPU VM Manager - Manage preemptible TPU VMs for GitHub Actions CI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@cli.command()
def monitor():
    """Run the monitoring daemon (continuously ensures desired TPU count)."""
    logging.info("Starting TPU VM Manager")
    logging.info(f"Target: {config.TPU_VM_COUNT} TPU VMs in {config.ZONE}")

    tpu_client = tpu_v2.TpuClient()

    try:
        monitor_loop(tpu_client)
    except KeyboardInterrupt:
        logging.info("Shutdown")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
def list_vms():
    """List all TPU VMs."""
    vms = list_tpu_vms()
    if not vms:
        click.echo("No TPU VMs found")
        return

    click.echo(f"Found {len(vms)} TPU VMs:")
    for vm in vms:
        click.echo(f"  {vm['name']}: {vm['state']}")


@cli.command()
@click.argument("index", type=int)
def create(index: int):
    """Create a TPU VM with the given index."""
    create_tpu_vm(index)


@cli.command()
@click.argument("name")
def delete(name: str):
    """Delete a TPU VM by name."""
    delete_tpu_vm(name)


@cli.command()
def ensure():
    """One-time check to ensure desired number of TPU VMs are running."""
    tpu_client = tpu_v2.TpuClient()
    ensure_tpu_vms(tpu_client)


@cli.command()
@click.argument("index", type=int)
@click.option("--lines", "-n", default=100, help="Number of log lines to show (default: 100)")
@click.option("--follow", "-f", is_flag=True, help="Follow log output in real-time")
def check_logs(index: int, lines: int, follow: bool):
    """Show GitHub Actions runner logs and diagnostics for a TPU VM."""
    vm_name = f"{config.TPU_VM_PREFIX}-{index}"

    logging.info(f"Fetching logs from TPU VM: {vm_name}")

    if follow:
        # For follow mode, just tail the service logs
        journalctl_cmd = f"sudo journalctl -u actions.runner.* -n {lines} -f"
        ssh_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            vm_name,
            "--zone",
            config.ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--command",
            journalctl_cmd,
        ]
        result = run(ssh_cmd, check=False)
        if result.returncode != 0:
            logging.error(f"Failed to fetch logs from {vm_name}")
            raise RuntimeError(f"Failed to fetch logs from {vm_name}")
    else:
        # For static mode, get comprehensive diagnostics
        diag_cmd = f"""
set -e
echo "=== Runner Service Status ==="
sudo systemctl status actions.runner.* --no-pager || true

echo ""
echo "=== Recent Service Logs (last {lines} lines) ==="
sudo journalctl -u actions.runner.* -n {lines} --no-pager || true

echo ""
echo "=== Runner Directory ==="
ls -la /home/github-runner/ || true

echo ""
echo "=== Recent Runner Logs ==="
if [ -d /home/github-runner/_diag ]; then
    echo "Diagnostic logs:"
    sudo find /home/github-runner/_diag -name "*.log" -type f -exec echo "{{}} ---" \\; -exec tail -n 50 {{}} \\; 2>/dev/null || true
fi

echo ""
echo "=== Runner Process ==="
ps aux | grep -E "(Runner.Listener|Runner.Worker)" | grep -v grep || echo "No runner processes found"

echo ""
echo "=== Docker Status ==="
sudo docker ps -a || true
"""  # noqa: E501
        ssh_cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            vm_name,
            "--zone",
            config.ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--command",
            diag_cmd,
        ]
        result = run(ssh_cmd, check=False)
        if result.returncode != 0:
            logging.error(f"Failed to fetch diagnostics from {vm_name}")
            raise RuntimeError(f"Failed to fetch diagnostics from {vm_name}")


@cli.command()
@click.argument("index", type=int)
@click.option("--test-path", default="tests/tpu/", help="Path to tests to run (default: tests/)")
@click.option("--pytest-args", default="-vs -o log_cli_level=INFO", help="Additional pytest arguments (default: -v)")
def debug_tpu(index: int, test_path: str, pytest_args: str):
    """Rsync marin directory to VM and run pytest in Docker container."""
    vm_name = f"{config.TPU_VM_PREFIX}-{index}"

    logging.info(f"Running TPU tests on: {vm_name}")

    # Get project root directory (parent of infra/tpu-ci)
    project_root = Path(__file__).parent.parent.parent
    remote_dir = "/tmp/marin-test"

    # Sync project directory to VM using git-tracked files only
    # This automatically excludes resource forks, build artifacts, etc.
    # COPYFILE_DISABLE=1 prevents macOS from including extended attributes/resource forks
    logging.info(f"Syncing project directory to {vm_name}...")

    tar_cmd = f"""cd {project_root} && \
        COPYFILE_DISABLE=1 git ls-files | COPYFILE_DISABLE=1 tar czf - --no-xattrs -T - | \
        gcloud compute tpus tpu-vm ssh {vm_name} \
        --zone {config.ZONE} \
        --project {config.GCP_PROJECT_ID} \
        --command 'rm -rf {remote_dir} && mkdir -p {remote_dir} && tar xzf - -C {remote_dir}'"""

    result = run(tar_cmd, shell=True, check=False)
    if result.returncode != 0:
        logging.error("Failed to sync directory")
        raise RuntimeError(f"Failed to sync project to {vm_name}")

    logging.info("✓ Project synced")

    # Clean up TPU resources before running tests
    logging.info(f"Cleaning up TPU resources on {vm_name}...")
    cleanup_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        vm_name,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--command",
        "sudo rm -f /tmp/libtpu_lockfile && sudo lsof -t /dev/vfio/* 2>/dev/null | xargs -r sudo kill -9 || true",
    ]
    result = run(cleanup_cmd, check=False)
    if result.returncode != 0:
        logging.warning("TPU cleanup had non-zero exit (may be normal if no processes to kill)")

    # Run pytest in Docker container as github-runner user
    logging.info(f"Running pytest on {vm_name}...")
    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        vm_name,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--command",
        f"""timeout 60 sudo -u github-runner bash -c 'docker run --rm --privileged \
            --device /dev/vfio:/dev/vfio \
            --shm-size=10g \
            -e JAX_PLATFORMS=tpu \
            -e PJRT_DEVICE=TPU \
            -e TPU_CI=true \
            -e START_RAY_TPU_CLUSTER=true \
            -e PYTHONPATH=/workspace \
            -e UV_PROJECT_ENVIRONMENT=/opt/marin/.venv \
            -v {remote_dir}:/workspace:ro \
            -w /workspace \
            {config.DOCKER_IMAGE_FULL} \
            uv run pytest {test_path} {pytest_args}'""",
    ]

    result = run(ssh_cmd, check=False)

    if result.returncode != 0:
        logging.error(f"Tests failed with exit code {result.returncode}")


@cli.command()
@click.argument("index", type=int)
def debug_setup(index: int):
    """Re-run startup script on an existing TPU VM for debugging."""
    vm_name = f"{config.TPU_VM_PREFIX}-{index}"

    logging.info(f"Re-running startup script on TPU VM: {vm_name}")

    # Generate and save startup script locally
    startup_script = get_startup_script()
    local_script_path = Path("/tmp/tpu-debug-startup.sh")
    local_script_path.write_text(startup_script)
    remote_script_path = "/tmp/tpu-debug-startup.sh"

    # Copy script to VM
    logging.info(f"Uploading startup script to {vm_name}...")
    scp_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "scp",
        str(local_script_path),
        f"{vm_name}:{remote_script_path}",
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
    ]

    result = run(scp_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logging.error(f"Failed to upload script: {result.stderr}")
        raise RuntimeError(f"Failed to upload startup script to {vm_name}")

    logging.info("✓ Script uploaded")

    # Execute script remotely
    logging.info(f"Executing startup script on {vm_name}...")
    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        vm_name,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--command",
        f"sudo bash {remote_script_path}",
    ]

    # Run with output streaming to console
    result = run(ssh_cmd, check=False)

    if result.returncode != 0:
        logging.error(f"Startup script execution failed with exit code {result.returncode}")
        raise RuntimeError(f"Failed to execute startup script on {vm_name}")

    logging.info("✓ Startup script execution complete")


if __name__ == "__main__":
    cli()
