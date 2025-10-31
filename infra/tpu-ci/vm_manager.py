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
#   "fastapi",
#   "uvicorn",
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
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import config
from fastapi import FastAPI
from google.cloud import tpu_v2
import uvicorn


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def run_sh(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run command from string with logging."""
    return run(shlex.split(cmd), **kwargs)


def vm_name(zone: str, index: int) -> str:
    """Generate VM name from zone and index."""
    return f"{config.TPU_VM_PREFIX}-{zone}-{index}"


def get_startup_script() -> str:
    """
    Generate startup script for TPU VMs.

    Installs Docker and GitHub Actions runner (ephemeral mode).
    Fetches GitHub token from Secret Manager at runtime.
    """
    return f"""#!/bin/bash
set -e

echo "=== TPU VM Setup Starting ==="

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

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y docker.io jq curl
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed, skipping..."
    apt-get update
    apt-get install -y jq curl
fi

systemctl enable docker
systemctl start docker

echo "Configuring Docker authentication..."
gcloud auth configure-docker {config.DOCKER_REGISTRY} --quiet

echo "Pre-pulling TPU CI Docker image..."
docker pull {config.DOCKER_IMAGE_FULL} || true

echo "Installing GitHub Actions runner..."
RUNNER_VERSION="2.311.0"
RUNNER_USER="github-runner"

if ! id -u $RUNNER_USER > /dev/null 2>&1; then
    useradd -m -s /bin/bash $RUNNER_USER
fi
usermod -aG docker $RUNNER_USER

cd /home/$RUNNER_USER
if [ ! -f config.sh ]; then
    curl -o actions-runner-linux-x64-$RUNNER_VERSION.tar.gz -L \\
      https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
    tar xzf actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
    rm actions-runner-linux-x64-$RUNNER_VERSION.tar.gz
fi
chown -R $RUNNER_USER:$RUNNER_USER /home/$RUNNER_USER

echo "Fetching GitHub token from Secret Manager..."
GITHUB_TOKEN=$(gcloud secrets versions access latest \\
  --secret="tpu-ci-github-token" \\
  --project="$PROJECT_ID")

REGISTRATION_TOKEN=$(curl -s -X POST \\
  -H "Accept: application/vnd.github+json" \\
  -H "Authorization: Bearer $GITHUB_TOKEN" \\
  https://api.github.com/repos/{config.GITHUB_ORG}/{config.GITHUB_REPO}/actions/runners/registration-token \\
  | jq -r .token)

echo "Configuring GitHub Actions runner..."
cd /home/$RUNNER_USER

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
  --labels {",".join(config.RUNNER_LABELS)} \\
  --work _work \\
  --unattended

echo "Installing runner service..."
cd /home/$RUNNER_USER
# Avoid "already exists" error
./svc.sh uninstall || true
./svc.sh install $RUNNER_USER
./svc.sh start

echo "=== TPU VM Setup Complete ==="
"""


def create_tpu_vm(index: int, zone: str) -> None:
    """Create a single preemptible TPU VM with GitHub Actions runner."""
    name = vm_name(zone, index)

    logging.info(f"Creating TPU VM: {name} in {zone}")

    startup_script_path = Path("/tmp/tpu-startup-script.sh")
    startup_script_path.write_text(get_startup_script())

    result = run_sh(
        f"gcloud compute tpus tpu-vm create {name} "
        f"--zone {zone} --project {config.GCP_PROJECT_ID} "
        f"--accelerator-type {config.TPU_ACCELERATOR_TYPE} --version {config.TPU_VERSION} "
        f"--preemptible --metadata-from-file startup-script={startup_script_path} "
        f"--scopes https://www.googleapis.com/auth/cloud-platform "
        f"--labels tpu-ci-component=runner,tpu-ci-managed=true",
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        if "already exists" in result.stderr:
            logging.info(f"TPU VM {name} already exists")
        else:
            logging.error(f"Failed to create TPU VM {name}: {result.stderr}")
            raise RuntimeError(f"Failed to create TPU VM {name}")
    else:
        logging.info(f"✓ Created TPU VM: {name}")


def delete_tpu_vm(vm_name: str, zone: str) -> None:
    """Delete a single TPU VM."""
    logging.info(f"Deleting TPU VM: {vm_name} in {zone}")

    result = run_sh(
        f"gcloud compute tpus tpu-vm delete {vm_name} --zone {zone} --project {config.GCP_PROJECT_ID} --quiet",
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 and "not found" not in result.stderr:
        logging.error(f"Failed to delete TPU VM {vm_name}: {result.stderr}")
        raise RuntimeError(f"Failed to delete TPU VM {vm_name}")

    logging.info(f"✓ Deleted TPU VM: {vm_name}")


def list_tpu_vms(zone: str) -> list[dict]:
    """List all TPU VMs with our labels in the specified zone, returning name and state."""
    result = run_sh(
        f"gcloud compute tpus tpu-vm list --zone {zone} --project {config.GCP_PROJECT_ID} "
        f"--format json --filter labels.tpu-ci-managed=true",
        capture_output=True,
        text=True,
        check=True,
    )

    if not result.stdout.strip():
        return []

    vms = json.loads(result.stdout)
    return [{"name": vm["name"], "state": vm.get("state", "UNKNOWN"), "zone": zone} for vm in vms]


def get_next_available_index(existing_vms: list[dict], count: int) -> int:
    """Find the lowest available index for a new VM."""
    existing_indices = set()
    for vm in existing_vms:
        parts = vm["name"].rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            existing_indices.add(int(parts[1]))

    for i in range(count):
        if i not in existing_indices:
            return i

    return count


def ensure_tpu_vms(tpu_client: tpu_v2.TpuClient, zone: str, count: int):
    """
    Ensure desired number of TPU VMs are running in the specified zone.
    - Delete any preempted/failed VMs
    - Create new VMs if count is below desired
    """
    vms = list_tpu_vms(zone)

    logging.info(f"[{zone}] Found {len(vms)} TPU VMs")

    bad_states = ["PREEMPTED", "TERMINATED", "FAILED"]
    for vm in vms[:]:
        if vm["state"] in bad_states:
            logging.warning(f"[{zone}] TPU {vm['name']} is in state {vm['state']}, deleting...")
            try:
                delete_tpu_vm(vm["name"], zone)
                vms.remove(vm)
            except Exception as e:
                logging.error(f"[{zone}] Failed to delete {vm['name']}: {e}")

    healthy_states = ["READY", "CREATING"]
    healthy_count = sum(1 for vm in vms if vm["state"] in healthy_states)

    logging.info(f"[{zone}] Healthy TPU VMs: {healthy_count}/{count}")

    needed = count - healthy_count
    if needed > 0:
        logging.info(f"[{zone}] Creating {needed} new TPU VMs...")
        for _ in range(needed):
            index = get_next_available_index(vms, count)
            try:
                create_tpu_vm(index, zone)
                vms.append({"name": vm_name(zone, index), "state": "CREATING", "zone": zone})
            except Exception as e:
                logging.error(f"[{zone}] Failed to create TPU VM: {e}")


def monitor_loop(tpu_client: tpu_v2.TpuClient):
    """Main monitoring loop - runs indefinitely."""
    while True:
        try:
            for zone, count in config.TPU_ZONES_CONFIG.items():
                ensure_tpu_vms(tpu_client, zone, count)
        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)

        time.sleep(60)


app = FastAPI(title="TPU CI Dashboard")


@app.get("/")
def status():
    """Get cluster status as JSON."""
    all_vms = []
    for zone in config.TPU_ZONES_CONFIG.keys():
        try:
            vms = list_tpu_vms(zone)
            all_vms.extend(vms)
        except Exception as e:
            logging.error(f"Failed to list VMs in {zone}: {e}")

    total_desired = sum(config.TPU_ZONES_CONFIG.values())
    healthy_states = ["READY", "CREATING"]
    healthy_count = sum(1 for vm in all_vms if vm["state"] in healthy_states)

    zone_status = {}
    for zone, desired_count in config.TPU_ZONES_CONFIG.items():
        zone_vms = [vm for vm in all_vms if vm["zone"] == zone]
        zone_healthy = sum(1 for vm in zone_vms if vm["state"] in healthy_states)
        zone_status[zone] = {
            "desired": desired_count,
            "healthy": zone_healthy,
            "vms": zone_vms,
        }

    return {
        "total_desired": total_desired,
        "total_healthy": healthy_count,
        "zones": zone_status,
        "timestamp": datetime.now().isoformat(),
    }


@click.group()
def cli():
    """TPU VM Manager - Manage preemptible TPU VMs for GitHub Actions CI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def dashboard(host: str, port: int):
    """Run the dashboard server (JSON status endpoint)."""
    logging.info(f"Starting TPU CI Dashboard on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
def monitor():
    """Run the monitoring daemon (continuously ensures desired TPU count)."""
    logging.info("Starting TPU VM Manager")
    total_vms = sum(config.TPU_ZONES_CONFIG.values())
    logging.info(f"Target: {total_vms} TPU VMs across {len(config.TPU_ZONES_CONFIG)} zones")
    for zone, count in config.TPU_ZONES_CONFIG.items():
        logging.info(f"  {zone}: {count} VMs")

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
    """List all TPU VMs across all zones."""
    all_vms = []
    for zone in config.TPU_ZONES_CONFIG.keys():
        vms = list_tpu_vms(zone)
        all_vms.extend(vms)

    if not all_vms:
        click.echo("No TPU VMs found")
        return

    click.echo(f"Found {len(all_vms)} TPU VMs:")
    for vm in all_vms:
        click.echo(f"  {vm['name']} ({vm['zone']}): {vm['state']}")


@cli.command()
@click.argument("index", type=int)
@click.argument("zone", type=str)
def create(index: int, zone: str):
    """Create a TPU VM with the given index in the specified zone."""
    create_tpu_vm(index, zone)


@cli.command()
@click.argument("name")
@click.argument("zone", type=str)
def delete(name: str, zone: str):
    """Delete a TPU VM by name in the specified zone."""
    delete_tpu_vm(name, zone)


@cli.command()
def ensure():
    """One-time check to ensure desired number of TPU VMs are running across all zones."""
    tpu_client = tpu_v2.TpuClient()
    for zone, count in config.TPU_ZONES_CONFIG.items():
        ensure_tpu_vms(tpu_client, zone, count)


@cli.command()
@click.argument("index", type=int)
@click.argument("zone", type=str)
@click.option("--lines", "-n", default=100, help="Number of log lines to show (default: 100)")
@click.option("--follow", "-f", is_flag=True, help="Follow log output in real-time")
def check_logs(index: int, zone: str, lines: int, follow: bool):
    """Show GitHub Actions runner logs and diagnostics for a TPU VM."""
    name = vm_name(zone, index)

    logging.info(f"Fetching logs from TPU VM: {name} in {zone}")

    if follow:
        journalctl_cmd = f"sudo journalctl -u actions.runner.* -n {lines} -f"
        result = run_sh(
            f"gcloud compute tpus tpu-vm ssh {name} --zone {zone} "
            f"--project {config.GCP_PROJECT_ID} --command '{journalctl_cmd}'",
            check=False,
        )
        if result.returncode != 0:
            logging.error(f"Failed to fetch logs from {name}")
            raise RuntimeError(f"Failed to fetch logs from {name}")
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
        result = run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                name,
                "--zone",
                zone,
                "--project",
                config.GCP_PROJECT_ID,
                "--command",
                diag_cmd,
            ],
            check=False,
        )
        if result.returncode != 0:
            logging.error(f"Failed to fetch diagnostics from {name}")
            raise RuntimeError(f"Failed to fetch diagnostics from {name}")


@cli.command()
@click.argument("index", type=int)
@click.argument("zone", type=str)
@click.option("--test-path", default="tests/tpu/")
@click.option("--pytest-args", default="-vs -o log_cli_level=INFO")
def debug_tpu(index: int, zone: str, test_path: str, pytest_args: str):
    """Rsync marin directory to VM and run pytest in Docker container."""
    name = vm_name(zone, index)

    logging.info(f"Running TPU tests on: {name} in {zone}")

    project_root = Path(__file__).parent.parent.parent
    remote_dir = "/tmp/marin-test"

    logging.info(f"Syncing project directory to {name}...")

    tar_cmd = f"""cd {project_root} && \
        COPYFILE_DISABLE=1 git ls-files | COPYFILE_DISABLE=1 tar czf - --no-xattrs -T - | \
        gcloud compute tpus tpu-vm ssh {name} \
        --zone {zone} \
        --project {config.GCP_PROJECT_ID} \
        --command 'sudo rm -rf {remote_dir} && mkdir -p {remote_dir} && tar xzf - -C {remote_dir}'"""

    result = run(tar_cmd, shell=True, check=False)
    if result.returncode != 0:
        logging.error("Failed to sync directory")
        raise RuntimeError(f"Failed to sync project to {name}")

    logging.info("✓ Project synced")

    logging.info(f"Cleaning up TPU resources on {name}...")
    cleanup_cmd = (
        "sudo rm -f /tmp/libtpu_lockfile && sudo lsof -t /dev/vfio/* 2>/dev/null | xargs -r sudo kill -9 || true"
    )
    result = run_sh(
        f"gcloud compute tpus tpu-vm ssh {name} --zone {zone} "
        f"--project {config.GCP_PROJECT_ID} --command '{cleanup_cmd}'",
        check=False,
    )
    if result.returncode != 0:
        logging.warning("TPU cleanup had non-zero exit (may be normal if no processes to kill)")

    # Build environment variable flags for Docker
    env_flags = []
    for var in ["HF_TOKEN", "WANDB_API_KEY"]:
        value = os.getenv(var)
        if value:
            env_flags.append(f"-e {var}={shlex.quote(value)}")
            logging.info(f"Forwarding {var} to Docker container")

    env_flags_str = " ".join(env_flags)

    logging.info(f"Running pytest on {name}...")
    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        name,
        "--zone",
        zone,
        "--project",
        config.GCP_PROJECT_ID,
        "--command",
        f"""sudo -u github-runner bash -c 'docker run --rm \
            --device /dev/vfio:/dev/vfio \
            --shm-size=100g \
            --stop-timeout=1 \
            --cap-add=SYS_RESOURCE \
            --ulimit memlock=68719476736:68719476736 \
            -e JAX_COORDINATOR_ADDRESS=127.0.0.1 \
            -e JAX_PLATFORMS=tpu \
            -e PJRT_DEVICE=TPU \
            -e TPU_CI=true \
            -e START_RAY_TPU_CLUSTER=true \
            -e PYTHONPATH=/workspace \
            -e UV_PROJECT_ENVIRONMENT=/opt/marin/.venv \
            {env_flags_str} \
            -v {remote_dir}:/workspace:rw \
            --tmpfs /workspace/logs:rw \
            --tmpfs /workspace/.pytest_cache:rw \
            -w /workspace \
            {config.DOCKER_IMAGE_FULL} \
            timeout --kill-after=5 --signal=TERM 120 uv run --frozen pytest {test_path} {pytest_args}' || true""",
    ]

    result = run(ssh_cmd, check=False)

    if result.returncode != 0:
        logging.error(f"Tests failed with exit code {result.returncode}")


@cli.command()
@click.argument("index", type=int)
@click.argument("zone", type=str)
def debug_setup(index: int, zone: str):
    """Re-run startup script on an existing TPU VM for debugging."""
    name = vm_name(zone, index)

    logging.info(f"Re-running startup script on TPU VM: {name} in {zone}")

    startup_script = get_startup_script()
    local_script_path = Path("/tmp/tpu-debug-startup.sh")
    local_script_path.write_text(startup_script)
    remote_script_path = "/tmp/tpu-debug-startup.sh"

    logging.info(f"Uploading startup script to {name}...")
    result = run_sh(
        f"gcloud compute tpus tpu-vm scp {local_script_path} {name}:{remote_script_path} "
        f"--zone {zone} --project {config.GCP_PROJECT_ID}",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logging.error(f"Failed to upload script: {result.stderr}")
        raise RuntimeError(f"Failed to upload startup script to {name}")

    logging.info("✓ Script uploaded")

    logging.info(f"Executing startup script on {name}...")
    result = run_sh(
        f"gcloud compute tpus tpu-vm ssh {name} --zone {zone} "
        f"--project {config.GCP_PROJECT_ID} --command 'sudo bash {remote_script_path}'",
        check=False,
    )

    if result.returncode != 0:
        logging.error(f"Startup script execution failed with exit code {result.returncode}")
        raise RuntimeError(f"Failed to execute startup script on {name}")

    logging.info("✓ Startup script execution complete")


if __name__ == "__main__":
    cli()
