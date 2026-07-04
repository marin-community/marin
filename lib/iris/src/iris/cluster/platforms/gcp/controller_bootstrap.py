# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for controller VMs.

Centralizes the controller bootstrap script template and generation logic.
Controller bootstrap installs Docker, pulls the image, writes the cluster
config, and starts the controller container. Reuses the shared registry
helpers from the worker bootstrap module.
"""

from collections.abc import Callable

import yaml

from iris.cluster.config import IrisClusterConfig, assert_no_inlined_secrets, config_to_dict
from iris.cluster.platforms.gcp.worker_bootstrap import render_template

CONTROLLER_CONTAINER_NAME = "iris-controller"

CONTROLLER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-controller] ================================================"
echo "[iris-controller] Starting controller bootstrap at $(date -Iseconds)"
echo "[iris-controller] ================================================"

# Write config file if provided
{{ config_setup }}

# Install host telemetry. sysstat records memory/CPU/IO to /var/log/sysstat/
# every 10 minutes so a wedged VM can be diagnosed after reboot. The Ops Agent
# streams the same data to Cloud Monitoring while the VM is alive; install is
# best-effort since it depends on the VM service account having metricWriter.
echo "[iris-controller] [telemetry] Installing sysstat + Ops Agent..."
export DEBIAN_FRONTEND=noninteractive
if ! dpkg -s sysstat >/dev/null 2>&1; then
    sudo apt-get update -qq || true
    sudo apt-get install -y -qq sysstat || true
fi
if [ -f /etc/default/sysstat ]; then
    sudo sed -i 's/^ENABLED="false"/ENABLED="true"/' /etc/default/sysstat || true
    sudo systemctl enable --now sysstat || true
fi
if ! systemctl is-active --quiet google-cloud-ops-agent; then
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh \
        && sudo bash add-google-cloud-ops-agent-repo.sh --also-install \
        || echo "[iris-controller] [telemetry] Ops Agent install failed (non-fatal)"
    rm -f add-google-cloud-ops-agent-repo.sh
fi

# Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[iris-controller] [1/5] Docker not found, installing..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "[iris-controller] [1/5] Docker installation complete"
else
    echo "[iris-controller] [1/5] Docker already installed: $(docker --version)"
fi

echo "[iris-controller] [2/5] Ensuring Docker daemon is running..."
sudo systemctl start docker || true
if sudo docker info > /dev/null 2>&1; then
    echo "[iris-controller] [2/5] Docker daemon is running"
else
    echo "[iris-controller] [2/5] ERROR: Docker daemon failed to start"
    exit 1
fi

# gcloud ships as a snap on the base image; snapd mounts snaps asynchronously
# during boot. Wait for seeding to finish so `gcloud` is on PATH for Artifact
# Registry auth below.
if command -v snap &> /dev/null; then
    timeout 300 snap wait system seed.loaded || echo "[iris-controller] Warning: snap seed wait timed out"
fi
export PATH="$PATH:/snap/bin"

# Tune network stack for high-connection workloads (#3066).
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

echo "[iris-controller] [3/5] Pulling image: {{ docker_image }}"
echo "[iris-controller]       This may take several minutes for large images..."

# Resolve the Artifact Registry host (empty for non-AR images). Auth is only
# configured when pulling from AR; root's docker config is used by `sudo docker`.
AR_HOST=""
if echo "{{ docker_image }}" | grep -q -- "-docker.pkg.dev/"; then
    AR_HOST=$(echo "{{ docker_image }}" | cut -d/ -f1)
fi

# Retry AR auth + pull -- gcloud ships as a snap and can be slow to become
# usable at first boot, so a single configure-docker + pull may hit an
# unauthenticated denial. Re-running both on each attempt absorbs the race.
IRIS_PULL_OK=0
for attempt in $(seq 1 20); do
    if [ -n "$AR_HOST" ]; then
        echo "[iris-controller] [3/5] Configuring docker auth for $AR_HOST (attempt $attempt/20)"
        if command -v gcloud &> /dev/null; then
            sudo gcloud auth configure-docker "$AR_HOST" -q || true
        else
            echo "[iris-controller] [3/5] gcloud not yet on PATH; waiting for snap to settle"
        fi
    fi
    if sudo docker pull {{ docker_image }}; then
        IRIS_PULL_OK=1
        break
    fi
    echo "[iris-controller] [3/5] docker pull failed (attempt $attempt/20); retrying in 15s"
    sleep 15
done

if [ "$IRIS_PULL_OK" -eq 1 ]; then
    echo "[iris-controller] [4/5] Image pull complete"
else
    echo "[iris-controller] [4/5] ERROR: Image pull failed after 20 attempts"
    exit 1
fi

# Stop existing controller if running.
# Use `docker kill` (SIGKILL) instead of `docker stop` (SIGTERM) because the
# controller's SIGTERM handler runs autoscaler.shutdown() → terminate_all(),
# which deletes every worker VM. On a controller restart the CLI has already
# taken a checkpoint via RPC, so the graceful shutdown path is unnecessary.
echo "[iris-controller] [5/5] Starting controller container..."
if sudo docker ps -a --format '{{.Names}}' | grep -q "^{{ container_name }}$"; then
    echo "[iris-controller]       Killing existing container..."
    sudo docker kill {{ container_name }} 2>/dev/null || true
    sudo docker rm {{ container_name }} 2>/dev/null || true
fi

# Create cache directory
sudo mkdir -p /var/cache/iris

# Start controller container with restart policy.
# Raise the open-file soft limit so the controller can handle many concurrent
# worker connections (endpoint RPCs, heartbeats, gcloud subprocesses, etc.).
sudo docker run -d --name {{ container_name }} \\
    --network=host \\
    --restart=unless-stopped \\
    --ulimit nofile=65536:524288 \\
    --ulimit core=0:0 \\
    -v /var/cache/iris:/var/cache/iris \\
    {{ config_volume }} \\
    {{ docker_image }} \\
    .venv/bin/python -m iris.cluster.controller.main serve \\
        --host 0.0.0.0 --port {{ port }} {{ config_flag }} {{ fresh_flag }}

echo "[iris-controller] [5/5] Controller container started"

# Wait for health
echo "[iris-controller] Waiting for controller to become healthy..."
RESTART_COUNT=0
MAX_ATTEMPTS=150
for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "[iris-controller] Health check attempt $i/$MAX_ATTEMPTS at $(date -Iseconds)..."
    if curl -sf http://localhost:{{ port }}/health > /dev/null 2>&1; then
        echo "[iris-controller] ================================================"
        echo "[iris-controller] Controller is healthy! Bootstrap complete."
        echo "[iris-controller] ================================================"
        exit 0
    fi
    # Check container status and detect restart loop
    STATUS=$(sudo docker inspect --format='{{.State.Status}}' {{ container_name }} 2>/dev/null || echo 'unknown')
    echo "[iris-controller] Container status: $STATUS"

    # Detect restart loop - if container keeps restarting, fail early
    if [ "$STATUS" = "restarting" ]; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge 3 ]; then
            echo "[iris-controller] ================================================"
            echo "[iris-controller] ERROR: Container in restart loop (restarting $RESTART_COUNT times)"
            echo "[iris-controller] ================================================"
            echo "[iris-controller] Full container logs:"
            sudo docker logs {{ container_name }} 2>&1
            echo "[iris-controller] ================================================"
            echo "[iris-controller] Container inspect:"
            sudo docker inspect {{ container_name }} 2>&1
            exit 1
        fi
    else
        RESTART_COUNT=0
    fi
    sleep 2
done

echo "[iris-controller] ================================================"
echo "[iris-controller] ERROR: Controller failed to become healthy after 300 seconds"
echo "[iris-controller] ================================================"
echo "[iris-controller] Full container logs:"
sudo docker logs {{ container_name }} 2>&1
echo "[iris-controller] ================================================"
echo "[iris-controller] Container inspect:"
sudo docker inspect {{ container_name }} 2>&1
exit 1
"""

CONFIG_SETUP_TEMPLATE = """
sudo mkdir -p /etc/iris
cat > /tmp/iris_config.yaml << 'IRIS_CONFIG_EOF'
{{ config_yaml }}
IRIS_CONFIG_EOF
sudo mv /tmp/iris_config.yaml /etc/iris/config.yaml
echo "{{ log_prefix }} Config written to /etc/iris/config.yaml"
"""


def _build_config_setup(config_yaml: str, log_prefix: str) -> str:
    """Generate config setup script fragment with given log prefix."""
    return render_template(CONFIG_SETUP_TEMPLATE, config_yaml=config_yaml, log_prefix=log_prefix)


def build_controller_bootstrap_script(
    docker_image: str,
    port: int,
    config_yaml: str = "",
    fresh: bool = False,
) -> str:
    """Build bootstrap script for controller VM.

    Args:
        docker_image: Docker image to run
        port: Controller port
        config_yaml: Optional YAML config to write to /etc/iris/config.yaml
        fresh: When True, pass ``--fresh`` to the controller serve command so
            it starts with an empty local database and skips checkpoint restore.
    """
    if config_yaml:
        config_setup = _build_config_setup(config_yaml, log_prefix="[iris-controller]")
        config_volume = "-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro"
        config_flag = "--config /etc/iris/config.yaml"
    else:
        config_setup = "# No config file provided"
        config_volume = ""
        config_flag = ""

    return render_template(
        CONTROLLER_BOOTSTRAP_SCRIPT,
        docker_image=docker_image,
        container_name=CONTROLLER_CONTAINER_NAME,
        port=port,
        config_setup=config_setup,
        config_volume=config_volume,
        config_flag=config_flag,
        fresh_flag="--fresh" if fresh else "",
    )


def build_controller_bootstrap_script_from_config(
    config: IrisClusterConfig,
    resolve_image: Callable[[str, str | None], str],
    fresh: bool = False,
) -> str:
    """Build controller bootstrap script from the full cluster config.

    Args:
        config: Full cluster configuration.
        resolve_image: Resolves a container image tag for the target registry.
        fresh: When True, pass ``--fresh`` to the controller serve command so
            it starts with an empty local database and skips checkpoint restore.
    """
    # #6873 render guard: a raw inlined secret must never reach GCE startup
    # metadata. References render verbatim (resolved at the controller runtime);
    # a literal fails the deploy here.
    assert_no_inlined_secrets(config)
    config_yaml = yaml.dump(config_to_dict(config), default_flow_style=False)
    ctrl = config.controller
    gcp_port = ctrl.gcp.port if ctrl.gcp is not None else 0
    manual_port = ctrl.manual.port if ctrl.manual is not None else 0
    port = gcp_port or manual_port or 10000
    image = ctrl.image

    zone: str | None = None
    if ctrl.gcp is not None and ctrl.gcp.zone:
        zone = ctrl.gcp.zone

    image = resolve_image(image, zone)

    return build_controller_bootstrap_script(image, port, config_yaml, fresh=fresh)
