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

import logging
import os
import subprocess
import sys

import click
import config


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


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

    # Check if secret exists
    result = run(
        [
            "gcloud",
            "secrets",
            "describe",
            "tpu-ci-github-token",
            "--project",
            config.GCP_PROJECT_ID,
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        logging.info("✓ Secret already exists")
        return

    # Create secret
    run(
        [
            "gcloud",
            "secrets",
            "create",
            "tpu-ci-github-token",
            "--project",
            config.GCP_PROJECT_ID,
            "--replication-policy",
            "automatic",
            "--data-file=-",
        ],
        input=token.encode(),
        check=True,
    )

    logging.info("✓ GitHub token stored")


def ensure_artifact_registry():
    """Ensure Artifact Registry repository exists."""
    logging.info("Checking Artifact Registry...")

    result = run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            config.ARTIFACT_REGISTRY_REPO_NAME,
            "--location",
            config.REGION,
            "--project",
            config.GCP_PROJECT_ID,
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        logging.info("✓ Artifact Registry exists")
        return

    logging.info("Creating Artifact Registry...")
    run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "create",
            config.ARTIFACT_REGISTRY_REPO_NAME,
            "--repository-format=docker",
            "--location",
            config.REGION,
            "--project",
            config.GCP_PROJECT_ID,
        ],
        check=True,
    )

    logging.info("✓ Artifact Registry created")


def build_and_push_docker_image():
    """Build and push TPU CI Docker image."""
    logging.info("Building TPU CI Docker image...")

    run(["gcloud", "auth", "configure-docker", config.DOCKER_REGISTRY, "-q"], check=True)

    run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--push",
            "-t",
            config.DOCKER_IMAGE_FULL,
            "-f",
            config.DOCKERFILE_TPU_CI_PATH,
            ".",
        ],
        check=True,
    )

    logging.info("✓ Docker image built and pushed")


def delete_controller_vm():
    """Delete controller VM if it exists."""
    logging.info("Checking for existing controller VM...")

    result = run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            config.CONTROLLER_NAME,
            "--zone",
            config.ZONE,
            "--project",
            config.GCP_PROJECT_ID,
        ],
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logging.info("✓ No existing controller VM")
        return

    logging.info("Deleting existing controller VM...")
    run(
        [
            "gcloud",
            "compute",
            "instances",
            "delete",
            config.CONTROLLER_NAME,
            "--zone",
            config.ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--quiet",
        ],
        check=True,
    )
    logging.info("✓ Controller VM deleted")


def create_controller_vm():
    """Create controller VM for running TPU VM manager."""
    logging.info("Creating controller VM...")

    # Get current branch to clone on the controller
    result = run(["git", "branch", "--show-current"], capture_output=True, text=True, check=False)
    branch = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else config.GITHUB_BRANCH
    logging.info(f"Controller will use branch: {branch}")

    startup_script = f"""#!/bin/bash
set -e

apt-get update
apt-get install -y git curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Clone repo to get monitor code
cd /opt
git clone --branch {branch} https://github.com/{config.GITHUB_ORG}/{config.GITHUB_REPO}.git
cd marin/infra/tpu-ci

# Create systemd service
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

    run(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            config.CONTROLLER_NAME,
            "--zone",
            config.ZONE,
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
            "--metadata",
            f"startup-script={startup_script}",
        ],
        check=True,
    )

    logging.info("✓ Controller VM created")


def delete_all_tpu_vms():
    """Delete all TPU VMs with tpu-ci labels."""
    logging.info("Deleting TPU VMs...")

    # List all TPU VMs with our label
    result = run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--zone",
            config.ZONE,
            "--project",
            config.GCP_PROJECT_ID,
            "--filter",
            "labels.tpu-ci-managed=true",
            "--format",
            "value(name)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0 or not result.stdout.strip():
        logging.info("✓ No TPU VMs to delete")
        return

    vm_names = result.stdout.strip().split("\n")
    logging.info(f"Found {len(vm_names)} TPU VMs to delete")

    for vm_name in vm_names:
        if vm_name:
            logging.info(f"Deleting TPU VM: {vm_name}")
            run(
                [
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
                ],
                check=False,
            )

    logging.info("✓ All TPU VMs deleted")


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
    ensure_artifact_registry()
    build_and_push_docker_image()
    delete_controller_vm()
    create_controller_vm()

    logging.info("=" * 60)
    logging.info("Controller setup complete!")
    logging.info("=" * 60)
    logging.info("")
    logging.info(f"Controller VM created and will manage {config.TPU_VM_COUNT} TPU VMs")
    logging.info("TPU VMs will be automatically created and maintained by the vm_manager daemon")
    logging.info("Each VM will auto-register as a GitHub Actions runner")
    logging.info("")


@cli.command()
def teardown():
    """Destroy all TPU CI infrastructure."""
    delete_all_tpu_vms()
    delete_controller_vm()


@cli.command("build-image")
def build_image():
    """Build and push the TPU CI Docker image."""
    logging.info("Building TPU CI Docker image...")
    ensure_artifact_registry()
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

    cmd = [
        "gcloud",
        "compute",
        "ssh",
        config.CONTROLLER_NAME,
        "--zone",
        config.ZONE,
        "--project",
        config.GCP_PROJECT_ID,
        "--command",
        cmd_str,
    ]

    run(cmd, check=True)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        logging.info("Cancelled")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)
