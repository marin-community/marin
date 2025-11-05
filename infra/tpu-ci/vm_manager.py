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
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import click
import config
import uvicorn
from fastapi import FastAPI
from google.cloud import tpu_v2

# Word lists for generating fun VM names
VERBS = [
    "running",
    "flying",
    "swimming",
    "jumping",
    "dancing",
    "singing",
    "climbing",
    "racing",
    "soaring",
    "dashing",
    "spinning",
    "bouncing",
]

NOUNS = [
    "tiger",
    "eagle",
    "dolphin",
    "mountain",
    "river",
    "forest",
    "cloud",
    "thunder",
    "phoenix",
    "dragon",
    "falcon",
    "canyon",
]


def generate_random_id() -> str:
    """Generate a short random ID (4 alphanumeric characters)."""
    import random
    import string

    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=4))


def generate_fun_name(zone: str) -> str:
    """
    Generate a fun VM name in wandb style.

    Format: {verb}-{noun}-{zone}-{random_id}
    Example: running-tiger-us-west4-a-7x3k
    """
    import random

    verb = random.choice(VERBS)
    noun = random.choice(NOUNS)
    random_id = generate_random_id()
    return f"{verb}-{noun}-{zone}-{random_id}"


def extract_zone_from_name(vm_name: str) -> str:
    """
    Extract zone from VM name.

    Expected format: {prefix}-{verb}-{noun}-{zone}-{random_id}
    For zone like 'us-west4-a', this returns the zone portion.
    """
    # Remove the prefix (tpu-ci-)
    parts = vm_name.split("-")
    # Format is: [prefix, verb, noun, region, subregion, zoneletter, randomid]
    # e.g., tpu-ci-running-tiger-us-west4-a-7x3k
    # parts: ['tpu', 'ci', 'running', 'tiger', 'us', 'west4', 'a', '7x3k']
    # Zone is the last 3 parts before the random ID
    if len(parts) >= 6:
        # Join the zone parts (e.g., 'us', 'west4', 'a')
        return f"{parts[-4]}-{parts[-3]}-{parts[-2]}"
    raise ValueError(f"Invalid VM name format: {vm_name}")


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command with logging."""
    logging.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def run_sh(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    """Run command from string with logging."""
    return run(shlex.split(cmd), **kwargs)


def vm_name(zone: str) -> str:
    """Generate VM name from zone using fun naming."""
    return f"{config.TPU_VM_PREFIX}-{generate_fun_name(zone)}"


def get_startup_script() -> str:
    """
    Generate startup script for TPU VMs.

    Installs Docker and GitHub Actions runner (ephemeral mode).
    Fetches GitHub token from Secret Manager at runtime.
    """
    return f"""#!/bin/bash
set -ex

echo "=== TPU VM Setup Starting ==="

# Completely disable unattended-upgrades to avoid apt lock conflicts
echo "Disabling unattended-upgrades..."
systemctl stop unattended-upgrades.service || true
systemctl disable unattended-upgrades.service || true
systemctl mask unattended-upgrades.service || true
pkill -9 -f unattended-upgrade || true
killall -9 apt apt-get || true

# Remove auto-upgrade configs
echo 'APT::Periodic::Update-Package-Lists "0";' > /etc/apt/apt.conf.d/20auto-upgrades
echo 'APT::Periodic::Unattended-Upgrade "0";' >> /etc/apt/apt.conf.d/20auto-upgrades

# Wait for locks to be released
for i in {{1..30}}; do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        echo "dpkg lock released"
        break
    fi
    echo "Waiting for dpkg lock... ($i/30)"
    sleep 2
done

if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y docker.io jq curl
else
    echo "Docker already installed, skipping..."
    apt-get update
    apt-get install -y jq curl
fi

systemctl enable docker
systemctl start docker

echo "Pre-pulling TPU CI Docker image..."
docker pull {config.DOCKER_IMAGE} || true

echo "Pre-populating uv cache..."
echo "Using Docker image: {config.DOCKER_IMAGE}"
# Create persistent uv cache directory
mkdir -p /var/cache/uv
# Fix permissions for Docker container user (UID 1000)
chown -R 1000:1000 /var/cache/uv
# Clone repo temporarily to get pyproject.toml and uv.lock for initial sync
TEMP_REPO=$(mktemp -d)
git clone --depth 1 https://github.com/{config.GITHUB_REPOSITORY}.git "$TEMP_REPO" || true
if [ -d "$TEMP_REPO" ]; then
    # Fix permissions for Docker container user (UID 1000)
    chown -R 1000:1000 "$TEMP_REPO"
    # Run uv sync to populate the cache
    docker run --rm \\
        -v /var/cache/uv:/opt/uv-cache:rw \\
        -v $TEMP_REPO:/workspace:rw \\
        -e UV_CACHE_DIR=/opt/uv-cache \\
        -e UV_LINK_MODE=copy \\
        -w /workspace \\
        {config.DOCKER_IMAGE}  uv sync --frozen --all-packages --extra tpu --extra gcp --group test
    rm -rf "$TEMP_REPO"
    echo "uv cache pre-populated"
else
    echo "Failed to clone repo, skipping cache pre-population"
fi

echo "Configuring TPU device permissions..."
# Create udev rule to allow non-root access to PCI device reset
cat > /etc/udev/rules.d/99-tpu-reset.rules <<'UDEV_EOF'
# Allow users to reset VFIO PCI devices (TPUs)
SUBSYSTEM=="pci", DRIVER=="vfio-pci", RUN+="/bin/chmod 0666 /sys%p/reset"
UDEV_EOF

# Reload udev rules and apply to existing devices
udevadm control --reload-rules
udevadm trigger --subsystem-match=pci

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

PROJECT_ID=$(curl -sSf -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/project/project-id)

echo "Fetching GitHub token from Secret Manager..."
GITHUB_TOKEN=$(gcloud secrets versions access latest \\
  --secret="tpu-ci-github-token" \\
  --project="$PROJECT_ID")

REGISTRATION_TOKEN=$(curl -s -X POST \\
  -H "Accept: application/vnd.github+json" \\
  -H "Authorization: Bearer $GITHUB_TOKEN" \\
  https://api.github.com/repos/marin-community/marin/actions/runners/registration-token \\
  | jq -r .token)

echo "Configuring GitHub Actions runner..."
cd /home/$RUNNER_USER

# Get instance name from metadata
INSTANCE_NAME=$(curl -sSf -H "Metadata-Flavor: Google" \\
  http://metadata.google.internal/computeMetadata/v1/instance/name)

# Check if runner is configured by looking for config files OR .runner file
if [ -f .runner ] || [ -f .credentials ] || [ -f .path ]; then
    echo "Runner already configured, removing..."

    # Stop service first if it exists
    if [ -f ./svc.sh ]; then
        ./svc.sh stop || true
        ./svc.sh uninstall || true
    fi

    # Try to cleanly remove configuration if config.sh exists
    if [ -f ./config.sh ]; then
        sudo -u $RUNNER_USER ./config.sh remove --token $REGISTRATION_TOKEN || true
    fi

    # Force cleanup of ALL state files and workspace to ensure clean slate
    rm -f .runner .credentials .path
    rm -rf _work

    echo "Existing runner configuration removed"
fi

# Now configure (works for both new and re-configured runners)
# Use --replace flag to replace any existing runner with the same name on GitHub
sudo -u $RUNNER_USER ./config.sh \\
  --url https://github.com/marin-community/marin \\
  --token $REGISTRATION_TOKEN \\
  --name "tpu-$INSTANCE_NAME" \\
  --labels {",".join(config.RUNNER_LABELS)} \\
  --work _work \\
  --unattended \\
  --replace

echo "Installing runner service..."
cd /home/$RUNNER_USER
./svc.sh install $RUNNER_USER
./svc.sh start

echo "=== TPU VM Setup Complete ==="
"""


def create_tpu_vm(zone: str) -> str:
    """Create a single preemptible TPU VM with GitHub Actions runner. Returns the VM name."""
    name = vm_name(zone)

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

    return name


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


def check_runner_service(vm_name: str, zone: str) -> bool:
    """
    Check if GitHub Actions runner service is active on a VM.

    Returns True if service is running, False otherwise.
    """
    try:
        result = run_sh(
            f"gcloud compute tpus tpu-vm ssh {vm_name} --zone {zone} "
            f"--project {config.GCP_PROJECT_ID} "
            f"--command 'systemctl is-active actions.runner.*'",
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

        is_active = result.returncode == 0 and result.stdout.strip() == "active"
        if not is_active:
            logging.warning(f"[{zone}] Runner service not active on {vm_name}: {result.stdout.strip()}")
        return is_active
    except subprocess.TimeoutExpired:
        logging.warning(f"[{zone}] Runner service check timed out on {vm_name}, assuming dead")
        return False


def _try_delete_vm(vm: dict, zone: str) -> bool:
    """
    Try to delete a VM. Returns True if deletion was successful.
    """
    logging.warning(f"[{zone}] TPU {vm['name']} is in state {vm['state']}, deleting...")
    try:
        delete_tpu_vm(vm["name"], zone)
        return True
    except Exception as e:
        logging.error(f"[{zone}] Failed to delete {vm['name']}: {e}")
        return False


def ensure_tpu_vms(tpu_client: tpu_v2.TpuClient, zone: str, count: int):
    """
    Ensure desired number of TPU VMs are running in the specified zone.
    - Delete any preempted/failed VMs
    - Create new VMs if count is below desired
    """
    vms = list_tpu_vms(zone)

    logging.info(f"[{zone}] Found {len(vms)} TPU VMs")

    bad_states = ["PREEMPTED", "TERMINATED", "FAILED"]
    healthy_states = ["READY", "CREATING"]
    healthy_count = len([vm for vm in vms if vm["state"] in healthy_states])

    bad_vms = [vm for vm in vms if vm["state"] in bad_states]

    # Verify runner service health for READY VMs
    # SSH doesn't work on the controller, skip for now.
    # ready_vms = [vm for vm in vms if vm["state"] == "READY"]
    # unhealthy_runners = []
    # for vm in ready_vms:
    #     if not check_runner_service(vm["name"], zone):
    #         unhealthy_runners.append(vm)
    #         healthy_count -= 1  # Reduce healthy count
    # if unhealthy_runners:
    #     logging.info(f"[{zone}] Found {len(unhealthy_runners)} VMs with unhealthy runner services")
    # bad_vms.extend(unhealthy_runners)

    logging.info(f"[{zone}] Healthy TPU VMs: {healthy_count}/{count}")

    if bad_vms:
        logging.info(f"[{zone}] Deleting {len(bad_vms)} bad/unhealthy TPU VMs...")
        for vm in bad_vms:
            _try_delete_vm(vm, zone)

    needed = count - healthy_count
    if needed > 0:
        logging.info(f"[{zone}] Creating {needed} new TPU VMs...")
        for _ in range(needed):
            try:
                name = create_tpu_vm(zone)
                vms.append({"name": name, "state": "CREATING", "zone": zone})
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

        time.sleep(600)  # Check every 10 minutes to allow time for TPU startup


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


def _run_dashboard(host: str, port: int):
    """Run the FastAPI dashboard server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


@cli.command()
@click.option("--dashboard-host", default="127.0.0.1", help="Dashboard host")
@click.option("--dashboard-port", default=8000, help="Dashboard port")
def monitor(dashboard_host: str, dashboard_port: int):
    """Run the monitoring daemon (continuously ensures desired TPU count)."""
    logging.info("Starting TPU VM Manager")
    total_vms = sum(config.TPU_ZONES_CONFIG.values())
    logging.info(f"Target: {total_vms} TPU VMs across {len(config.TPU_ZONES_CONFIG)} zones")
    for zone, count in config.TPU_ZONES_CONFIG.items():
        logging.info(f"  {zone}: {count} VMs")

    # Start dashboard in background thread
    logging.info(f"Starting dashboard on {dashboard_host}:{dashboard_port}")
    dashboard_thread = threading.Thread(
        target=_run_dashboard,
        args=(dashboard_host, dashboard_port),
        daemon=True,
    )
    dashboard_thread.start()

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
    """List all TPU VMs across all zones with runner status."""
    all_vms = []
    for zone in config.TPU_ZONES_CONFIG.keys():
        vms = list_tpu_vms(zone)
        all_vms.extend(vms)

    if not all_vms:
        click.echo("No TPU VMs found")
        return

    click.echo(f"Found {len(all_vms)} TPU VMs:")
    for vm in all_vms:
        vm_state = vm["state"]
        if vm_state == "READY":
            is_healthy = check_runner_service(vm["name"], vm["zone"])
            runner_status = " [runner: active]" if is_healthy else " [runner: inactive]"
            vm["runner"] = "active" if is_healthy else "inactive"
        else:
            vm["runner"] = "N/A"

    for vm in all_vms:
        vm_state = vm["state"]
        runner_status = f" [runner: {vm['runner']}]"
        click.echo(f" - {vm['name']} ({vm_state}){runner_status}")


@cli.command()
@click.argument("zone", type=str)
def create(zone: str):
    """Create a TPU VM with a random fun name in the specified zone."""
    name = create_tpu_vm(zone)
    click.echo(f"Created TPU VM: {name}")


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


def get_diagnostic_script(lines: int) -> str:
    """
    Generate comprehensive diagnostic script for TPU VM debugging.

    Organized into two phases:
    - Setup Phase: Startup script execution, Docker setup, network connectivity
    - Runtime Phase: GitHub Actions runner status, logs, and processes
    """
    return f"""
set -e

echo "=========================================="
echo "===       SETUP PHASE DIAGNOSTICS     ==="
echo "=========================================="

echo ""
echo "=== Startup Script Logs (last {lines} lines) ==="
sudo journalctl -u google-startup-scripts.service -n {lines} --no-pager || echo "No startup script logs found"

echo ""
echo "=== Startup Script Completion Status ==="
grep "startup-script exit status" /var/log/syslog | tail -5 || echo "No completion marker found in syslog"

echo ""
echo "=== Docker Authentication ==="
if [ -f ~/.docker/config.json ]; then
    echo "Docker config exists"
    cat ~/.docker/config.json | jq -r '.auths | keys[]' 2>/dev/null || echo "Unable to parse Docker auth config"
else
    echo "No Docker config found"
fi

echo ""
echo "=== Metadata Server Access ==="
curl -sSf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id \
    && echo " - OK" || echo " - FAILED"

echo ""
echo "=========================================="
echo "===      RUNTIME PHASE DIAGNOSTICS    ==="
echo "=========================================="

echo ""
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
    sudo find /home/github-runner/_diag -name "*.log" -type f \
        -exec echo "{{}} ---" \\; -exec tail -n 50 {{}} \\; 2>/dev/null || true
fi

echo ""
echo "=== Runner Process ==="
ps aux | grep -E "(Runner.Listener|Runner.Worker)" | grep -v grep || echo "No runner processes found"

echo ""
echo "=== Docker Status ==="
sudo docker ps -a || true
"""


@cli.command()
@click.argument("name", type=str)
@click.option("--lines", "-n", default=100, help="Number of log lines to show (default: 100)")
@click.option("--follow", "-f", is_flag=True, help="Follow log output in real-time")
def check_logs(name: str, lines: int, follow: bool):
    """Show GitHub Actions runner logs and diagnostics for a TPU VM."""
    zone = extract_zone_from_name(name)

    logging.info(f"Fetching logs from TPU VM: {name} in {zone}")

    if follow:
        journalctl_cmd = f"sudo journalctl -u actions.runner.* -n {lines} -f"
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
                journalctl_cmd,
            ],
            check=False,
        )
        if result.returncode != 0:
            logging.error(f"Failed to fetch logs from {name}")
            raise RuntimeError(f"Failed to fetch logs from {name}")
    else:
        # For static mode, get comprehensive diagnostics
        diag_cmd = get_diagnostic_script(lines)
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
@click.argument("name", type=str)
@click.option("--test-path", default="tests/tpu/")
@click.option("--pytest-args", default="-v --tb=short -s --log-cli-level=INFO")
@click.option("--timeout", default=900, help="Timeout in seconds for pytest execution")
@click.option("--env-vars", multiple=True, help="Environment variables to pass to Docker")
def debug_tpu(name: str, test_path: str, pytest_args: str, timeout: int, env_vars: tuple[str, ...]):
    """Rsync marin directory to VM and run pytest in Docker container."""
    zone = extract_zone_from_name(name)

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

    logging.info(f"Running pytest on {name}...")

    # Build env var flags for Docker
    env_var_flags = ""
    for env_var in env_vars:
        env_var_flags += f"  -e {env_var} \\\n"

    # Use the same script structure as GitHub Actions workflow
    test_script = f"""
sudo rm -f /tmp/libtpu_lockfile || true
sudo lsof -t /dev/vfio/* 2>/dev/null | xargs -r sudo kill -9 || true

sudo docker run --rm \\
  --device /dev/vfio:/dev/vfio \\
  --shm-size=100g \\
  --stop-timeout=1 \\
  --cap-add=SYS_RESOURCE \\
  --ulimit memlock=68719476736:68719476736 \\
  -e JAX_PLATFORMS=tpu \\
  -e PJRT_DEVICE=TPU \\
  -e TPU_CI=true \\
  -e JAX_COORDINATOR_ADDRESS=127.0.0.1 \\
  -e START_RAY_TPU_CLUSTER=true \\
  -e PYTHONPATH=/workspace \\
  -e UV_PROJECT_ENVIRONMENT=/opt/marin/.venv \\
{env_var_flags}  -v {remote_dir}:/workspace:rw \\
  --tmpfs /workspace/logs:rw \\
  --tmpfs /workspace/.pytest_cache:rw \\
  -w /workspace \\
  ghcr.io/{config.GITHUB_REPOSITORY}/{config.DOCKER_IMAGE_NAME}:{config.DOCKER_IMAGE_TAG} \\
  timeout --kill-after=5 --signal=TERM {timeout} uv run pytest {test_path} {pytest_args}
"""

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
        test_script,
    ]

    result = run(ssh_cmd, check=False)

    if result.returncode != 0:
        logging.error(f"Tests failed with exit code {result.returncode}")


@cli.command()
@click.argument("name", type=str)
def debug_setup(name: str):
    """Re-run startup script on an existing TPU VM for debugging."""
    zone = extract_zone_from_name(name)

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
