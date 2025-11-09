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
# ]
# ///

"""
TPU CI Infrastructure Management

Manages preemptible TPU VMs with GitHub Actions runners.
"""

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import click
import config


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def run_sh(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run command from string with logging."""
    return run(shlex.split(cmd), **kwargs)


def get_github_token() -> str:
    """Get GitHub token from environment variable."""
    token = os.getenv("MARIN_CI_TOKEN")

    if not token:
        logging.error("MARIN_CI_TOKEN environment variable not set")
        logging.error("Create a fine-grained PAT with repo:actions scope")
        sys.exit(1)

    return token


def store_github_token_in_secret_manager(token: str):
    """Store GitHub token in Secret Manager."""
    logging.info("Storing GitHub token in Secret Manager...")

    result = run_sh(
        f"gcloud secrets describe tpu-ci-github-token --project {config.GCP_PROJECT_ID}",
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        logging.info("✓ Secret already exists")
        return

    run_sh(
        f"gcloud secrets create tpu-ci-github-token --project {config.GCP_PROJECT_ID} "
        f"--replication-policy automatic --data-file=-",
        input=token.encode(),
        check=True,
    )

    logging.info("✓ GitHub token stored")


def check_ghcr_authentication() -> bool:
    """Check if user is already authenticated to ghcr.io."""
    docker_config_path = Path.home() / ".docker" / "config.json"

    if not docker_config_path.exists():
        return False

    try:
        with docker_config_path.open() as f:
            config_data = json.load(f)

        # Check for ghcr.io in auths
        auths = config_data.get("auths", {})
        if "ghcr.io" in auths:
            return True

        # Check for ghcr.io in credHelpers (credential store)
        cred_helpers = config_data.get("credHelpers", {})
        if "ghcr.io" in cred_helpers:
            return True

        # Check for global credential helper
        if config_data.get("credsStore"):
            # If there's a global credential store, assume it works for ghcr.io
            return True

    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Could not read Docker config: {e}")
        return False

    return False


def build_and_push_docker_image():
    """Build and push TPU CI Docker image to GitHub Container Registry."""
    logging.info("Building and pushing Docker image to ghcr.io...")

    # Build the full image URLs (both latest and date-tagged)
    base_url = f"ghcr.io/{config.GITHUB_REPOSITORY}/{config.DOCKER_IMAGE_NAME}"
    latest_image = f"{base_url}:{config.DOCKER_IMAGE_TAG}"

    date_tag = datetime.utcnow().strftime("%Y-%m-%d-%H%M")
    date_image = f"{base_url}:{date_tag}"

    logging.info("Target images:")
    logging.info(f"  - {latest_image}")
    logging.info(f"  - {date_image}")

    # Check if user is authenticated to ghcr.io
    if not check_ghcr_authentication():
        logging.error("Not authenticated to ghcr.io")
        logging.error("")
        logging.error("To authenticate, run:")
        logging.error("")
        logging.error("  echo $GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin")
        logging.error("")
        logging.error("Where:")
        logging.error("  - GITHUB_PAT is a Personal Access Token with 'write:packages' scope")
        logging.error("  - Create a token at: https://github.com/settings/tokens/new")
        logging.error("  - YOUR_GITHUB_USERNAME is your GitHub username")
        logging.error("")
        sys.exit(1)

    logging.info("✓ Authenticated to ghcr.io")

    run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--push",
            "-t",
            latest_image,
            "-t",
            date_image,
            "-f",
            config.DOCKERFILE_TPU_CI_PATH,
            ".",
        ],
        check=True,
    )

    logging.info("✓ Docker images built and pushed to ghcr.io")
    logging.info(f"  - {latest_image}")
    logging.info(f"  - {date_image}")


def delete_controller_vm():
    """Delete controller VM if it exists."""
    logging.info("Checking for existing controller VM...")

    result = run_sh(
        f"gcloud compute instances describe {config.CONTROLLER_NAME} "
        f"--zone {config.CONTROLLER_ZONE} --project {config.GCP_PROJECT_ID}",
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logging.info("✓ No existing controller VM")
        return

    logging.info("Deleting existing controller VM...")
    run_sh(
        f"gcloud compute instances delete {config.CONTROLLER_NAME} "
        f"--zone {config.CONTROLLER_ZONE} --project {config.GCP_PROJECT_ID} --quiet",
        check=True,
    )
    logging.info("✓ Controller VM deleted")


def wait_for_controller_service():
    logging.info("Waiting for controller service to start (timeout: 10 minutes)...")
    start_time = time.time()
    timeout_seconds = 600
    retry_interval = 10

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logging.error("Controller service failed to start within 10 minutes")
            sys.exit(1)

        result = run_sh(
            f"gcloud compute ssh {config.CONTROLLER_NAME} --zone {config.CONTROLLER_ZONE} "
            f"--project {config.GCP_PROJECT_ID} --command 'curl -sf http://localhost:8000/'",
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logging.info("Controller service is active and responding")
            return

        remaining = timeout_seconds - elapsed
        logging.info(f"Service not ready yet, retrying in {retry_interval}s (time remaining: {int(remaining)}s)...")
        time.sleep(retry_interval)


def create_controller_vm():
    """Create controller VM for running TPU VM manager."""
    logging.info("Creating controller VM...")

    result = run_sh("git branch --show-current", capture_output=True, text=True, check=False)
    branch = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else config.GITHUB_BRANCH
    logging.info(f"Controller will use branch: {branch}")

    startup_script = f"""#!/bin/bash
set -e

apt-get update
apt-get install -y git curl

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

cd /opt
git clone --branch {branch} {config.GITHUB_URL}
cd marin/infra/tpu-ci

cat > /etc/systemd/system/tpu-monitor.service <<'EOF'
[Unit]
Description=TPU VM Manager
After=network.target

[Service]
Type=simple
Environment="PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
WorkingDirectory=/opt/marin/infra/tpu-ci
ExecStart=/root/.local/bin/uv run vm_manager.py monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tpu-monitor
systemctl start tpu-monitor
"""

    startup_script_path = Path("/tmp/controller-startup-script.sh")
    startup_script_path.write_text(startup_script)

    run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            config.CONTROLLER_NAME,
            "--zone",
            config.CONTROLLER_ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--machine-type",
            config.CONTROLLER_MACHINE_TYPE,
            "--boot-disk-size",
            f"{config.CONTROLLER_DISK_SIZE_GB}GB",
            "--scopes",
            "https://www.googleapis.com/auth/cloud-platform",
            "--labels",
            "tpu-ci-component=controller,tpu-ci-managed=true",
            "--metadata-from-file",
            f"startup-script={startup_script_path}",
        ],
        check=True,
    )

    logging.info("✓ Controller VM created")
    wait_for_controller_service()


def delete_all_tpu_vms():
    """Delete all TPU VMs with tpu-ci labels across all zones in parallel."""
    logging.info("Discovering TPU VMs across all zones...")

    zones = list(config.TPU_ZONES_CONFIG.keys())

    def list_vms_in_zone(zone: str) -> list[tuple[str, str]]:
        """List all TPU VMs in a zone. Returns list of (zone, vm_name) tuples."""
        result = run_sh(
            f"gcloud compute tpus tpu-vm list --zone {zone} --project {config.GCP_PROJECT_ID} "
            f"--filter labels.tpu-ci-managed=true --format value(name)",
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0 or not result.stdout.strip():
            return []

        vm_names = [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]
        return [(zone, vm_name) for vm_name in vm_names]

    def delete_vm(zone: str, vm_name: str):
        """Delete a single TPU VM."""
        logging.info(f"Deleting {vm_name} in {zone}")
        run_sh(
            f"gcloud compute tpus tpu-vm delete {vm_name} --zone {zone} --project {config.GCP_PROJECT_ID} --quiet",
            check=False,
        )

    # Fetch all VM names in parallel
    all_vms = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(list_vms_in_zone, zone) for zone in zones]
        for future in as_completed(futures):
            try:
                all_vms.extend(future.result())
            except Exception as e:
                logging.error(f"Error listing TPU VMs: {e}")

    if not all_vms:
        logging.info("✓ No TPU VMs to delete")
        return

    logging.info(f"Found {len(all_vms)} TPU VMs, deleting in parallel...")

    # Delete all VMs in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(delete_vm, zone, vm_name) for zone, vm_name in all_vms]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error deleting TPU VM: {e}")

    logging.info(f"✓ Deleted {len(all_vms)} TPU VMs across all zones")


@click.group()
def cli():
    """TPU CI Infrastructure Management - Manage preemptible TPU VMs for GitHub Actions CI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


@cli.command("setup-controller")
def setup_controller():
    """Create controller VM and supporting infrastructure."""
    logging.info("=" * 60)
    logging.info("TPU CI Controller Setup")
    logging.info("=" * 60)

    github_token = get_github_token()

    store_github_token_in_secret_manager(github_token)
    build_and_push_docker_image()
    delete_controller_vm()
    create_controller_vm()

    logging.info("=" * 60)
    logging.info("Controller setup complete!")
    logging.info("=" * 60)
    logging.info("")
    total_vms = sum(config.TPU_ZONES_CONFIG.values())
    logging.info(
        f"Controller VM created and will manage {total_vms} TPU VMs across {len(config.TPU_ZONES_CONFIG)} zones"
    )
    for zone, count in config.TPU_ZONES_CONFIG.items():
        logging.info(f"  {zone}: {count} VMs")
    logging.info("TPU VMs will be automatically created and maintained by the vm_manager daemon")
    logging.info("Each VM will auto-register as a GitHub Actions runner")
    logging.info("")


@cli.command()
def teardown():
    """Destroy all TPU CI infrastructure."""
    delete_controller_vm()
    delete_all_tpu_vms()


@cli.command("build-image")
def build_image():
    """Build and push the TPU CI Docker image to ghcr.io."""
    logging.info("Building TPU CI Docker image...")
    build_and_push_docker_image()
    logging.info("✓ Image build complete")


@cli.command("controller-logs")
@click.option("--lines", "-n", default=100, help="Number of log lines to show for TPU monitor")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def controller_logs(lines: int, follow: bool):
    """Show logs from the controller VM's startup script and TPU monitor service."""
    logging.info("Fetching controller logs...")

    # Show both startup and monitor logs
    cmd_str = (
        f"echo '=== Startup Script Logs ===' && "
        f"sudo journalctl -u google-startup-scripts.service --no-pager | tail -50 && "
        f"echo '' && echo '=== TPU Monitor Service Logs ===' && "
        f"sudo journalctl -u tpu-monitor -n {lines}" + (" -f" if follow else "")
    )

    run(
        [
            "gcloud",
            "compute",
            "ssh",
            config.CONTROLLER_NAME,
            "--zone",
            config.CONTROLLER_ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--command",
            cmd_str,
        ],
        check=True,
    )


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        logging.info("Cancelled")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)
