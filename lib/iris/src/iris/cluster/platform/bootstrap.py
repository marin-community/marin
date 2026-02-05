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

"""Bootstrap scripts and health checks for controller and worker VMs."""

from __future__ import annotations

import logging
import shlex
import time
from dataclasses import dataclass

from iris.cluster.platform.ssh import SshConnection
from iris.rpc import config_pb2
from iris.time_utils import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)

# Controller defaults
CONTROLLER_CONTAINER_NAME = "iris-controller"
DEFAULT_CONTROLLER_PORT = 10000
DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50

# Health check settings
HEALTH_CHECK_TIMEOUT_SECONDS = 120
RESTART_LOOP_THRESHOLD = 3  # Fail after N consecutive "restarting" statuses
EARLY_TIMEOUT_SECONDS = 60  # Exit early after N seconds if in restart loop
HEALTH_CHECK_BACKOFF_INITIAL = 2.0
HEALTH_CHECK_BACKOFF_MAX = 10.0


# ============================================================================
# Health Checking
# ============================================================================


@dataclass
class HealthCheckResult:
    """Result of a health check with diagnostic info."""

    healthy: bool
    curl_output: str = ""
    curl_error: str = ""
    container_status: str = ""
    container_logs: str = ""

    def __bool__(self) -> bool:
        return self.healthy

    def summary(self) -> str:
        if self.healthy:
            return "healthy"
        parts = []
        if self.container_status:
            parts.append(f"container={self.container_status}")
        if self.curl_error:
            parts.append(f"curl_error={self.curl_error[:50]}")
        return ", ".join(parts) if parts else "unknown failure"


def check_health(
    conn: SshConnection,
    port: int = 10001,
    container_name: str = "iris-worker",
) -> HealthCheckResult:
    """Check if worker/controller is healthy via health endpoint."""
    result = HealthCheckResult(healthy=False)

    try:
        logger.info("Running health check: curl -sf http://localhost:%d/health", port)
        curl_result = conn.run(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
        if curl_result.returncode == 0:
            result.healthy = True
            result.curl_output = curl_result.stdout.strip()
            logger.info("Health check succeeded")
            return result
        result.curl_error = curl_result.stderr.strip() or f"exit code {curl_result.returncode}"
        logger.info("Health check failed: %s", result.curl_error)
    except Exception as exc:
        result.curl_error = str(exc)
        logger.info("Health check exception: %s", exc)

    # Gather diagnostics on failure
    try:
        status_result = conn.run(
            f"sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'not_found'",
            timeout=Duration.from_seconds(10),
        )
        result.container_status = status_result.stdout.strip()
    except Exception as exc:
        result.container_status = f"error: {exc}"

    if result.container_status in ("restarting", "exited", "dead", "not_found"):
        try:
            cmd = f"sudo docker logs {container_name} --tail 20 2>&1"
            logs_result = conn.run(cmd, timeout=Duration.from_seconds(15))
            if logs_result.returncode == 0 and logs_result.stdout.strip():
                result.container_logs = logs_result.stdout.strip()
        except Exception as exc:
            result.container_logs = f"error fetching logs: {exc}"

    return result


def wait_healthy_via_ssh(
    conn: SshConnection,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    container_name: str = CONTROLLER_CONTAINER_NAME,
) -> bool:
    """Poll health endpoint via SSH until healthy or timeout."""
    logger.info("Starting SSH-based health check (port=%d, timeout=%ds)", port, int(timeout))
    start_time = time.monotonic()
    backoff = ExponentialBackoff(
        initial=HEALTH_CHECK_BACKOFF_INITIAL,
        maximum=HEALTH_CHECK_BACKOFF_MAX,
    )

    attempt = 0
    last_result = None
    consecutive_restarts = 0

    while True:
        attempt += 1
        last_result = check_health(conn, port, container_name)
        elapsed = time.monotonic() - start_time

        if last_result.healthy:
            logger.info("SSH health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
            return True

        logger.info("SSH health check attempt %d failed (%.1fs): %s", attempt, elapsed, last_result.summary())

        if last_result.container_status == "restarting":
            consecutive_restarts += 1
            if consecutive_restarts >= RESTART_LOOP_THRESHOLD and elapsed >= EARLY_TIMEOUT_SECONDS:
                logger.error(
                    "Container is in restart loop (restarting %d times in %.1fs). Failing early.",
                    consecutive_restarts,
                    elapsed,
                )
                break
        else:
            consecutive_restarts = 0

        if elapsed >= timeout:
            break

        interval = backoff.next_interval()
        remaining = timeout - elapsed
        time.sleep(min(interval, remaining))

    logger.error("=" * 60)
    logger.error("SSH health check FAILED after %d attempts (%.1fs)", attempt, elapsed)
    logger.error("=" * 60)
    logger.error("Final status: %s", last_result.summary())
    if last_result.container_status:
        logger.error("Container status: %s", last_result.container_status)
    if last_result.curl_error:
        logger.error("Health endpoint error: %s", last_result.curl_error)

    try:
        logs_result = conn.run(f"sudo docker logs {container_name} 2>&1", timeout=Duration.from_seconds(30))
        if logs_result.returncode == 0 and logs_result.stdout.strip():
            logger.error("Container logs (full output):\n%s", logs_result.stdout.strip())
        elif last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")

        inspect_result = conn.run(
            f"sudo docker inspect {container_name} 2>&1",
            timeout=Duration.from_seconds(15),
        )
        if inspect_result.returncode == 0:
            logger.error("Container inspect output:\n%s", inspect_result.stdout.strip())

    except Exception as exc:
        logger.error("Failed to fetch diagnostics: %s", exc)
        if last_result.container_logs:
            logger.error("Container logs:\n%s", last_result.container_logs)
        else:
            logger.error("No container logs available (container may not exist)")

    logger.error("=" * 60)
    return False


# ============================================================================
# Controller bootstrap
# ============================================================================


CONTROLLER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-controller] ================================================"
echo "[iris-controller] Starting controller bootstrap at $(date -Iseconds)"
echo "[iris-controller] ================================================"

# Write config file if provided
{config_setup}

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

# Configure docker for GCP Artifact Registry
echo "[iris-controller] [3/5] Configuring Docker for GCP Artifact Registry..."
sudo gcloud auth configure-docker \
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true
echo "[iris-controller] [3/5] Docker registry configuration complete"

echo "[iris-controller] [4/5] Pulling image: {docker_image}"
echo "[iris-controller]       This may take several minutes for large images..."
if sudo docker pull {docker_image}; then
    echo "[iris-controller] [4/5] Image pull complete"
else
    echo "[iris-controller] [4/5] ERROR: Image pull failed"
    exit 1
fi

# Stop existing controller if running
echo "[iris-controller] [5/5] Starting controller container..."
if sudo docker ps -a --format '{{{{.Names}}}}' | grep -q "^{container_name}$"; then
    echo "[iris-controller]       Stopping existing container..."
    sudo docker stop {container_name} 2>/dev/null || true
    sudo docker rm {container_name} 2>/dev/null || true
fi

# Create cache directory
sudo mkdir -p /var/cache/iris

# Start controller container with restart policy
sudo docker run -d --name {container_name} \
    --network=host \
    --restart=unless-stopped \
    -v /var/cache/iris:/var/cache/iris \
    {config_volume} \
    {docker_image} \
    .venv/bin/python -m iris.cluster.controller.main serve \
        --host 0.0.0.0 --port {port} {config_flag}

echo "[iris-controller] [5/5] Controller container started"

# Wait for health
echo "[iris-controller] Waiting for controller to become healthy..."
RESTART_COUNT=0
for i in $(seq 1 30); do
    echo "[iris-controller] Health check attempt $i/30 at $(date -Iseconds)..."
    if curl -sf http://localhost:{port}/health > /dev/null 2>&1; then
        echo "[iris-controller] ================================================"
        echo "[iris-controller] Controller is healthy! Bootstrap complete."
        echo "[iris-controller] ================================================"
        exit 0
    fi
    # Check container status and detect restart loop
    STATUS=$(sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'unknown')
    echo "[iris-controller] Container status: $STATUS"

    # Detect restart loop - if container keeps restarting, fail early
    if [ "$STATUS" = "restarting" ]; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge 3 ]; then
            echo "[iris-controller] ================================================"
            echo "[iris-controller] ERROR: Container in restart loop (restarting $RESTART_COUNT times)"
            echo "[iris-controller] ================================================"
            echo "[iris-controller] Full container logs:"
            sudo docker logs {container_name} 2>&1
            echo "[iris-controller] ================================================"
            echo "[iris-controller] Container inspect:"
            sudo docker inspect {container_name} 2>&1
            exit 1
        fi
    else
        RESTART_COUNT=0
    fi
    sleep 2
done

echo "[iris-controller] ================================================"
echo "[iris-controller] ERROR: Controller failed to become healthy after 60 seconds"
echo "[iris-controller] ================================================"
echo "[iris-controller] Full container logs:"
sudo docker logs {container_name} 2>&1
echo "[iris-controller] ================================================"
echo "[iris-controller] Container inspect:"
sudo docker inspect {container_name} 2>&1
exit 1
"""

CONFIG_SETUP_TEMPLATE = """
sudo mkdir -p /etc/iris
cat > /tmp/iris_config.yaml << 'IRIS_CONFIG_EOF'
{config_yaml}
IRIS_CONFIG_EOF
sudo mv /tmp/iris_config.yaml /etc/iris/config.yaml
echo "[iris-controller] Config written to /etc/iris/config.yaml"
"""

CONTROLLER_STOP_SCRIPT = """
set -e
echo "[iris-controller] Stopping controller container"
sudo docker stop {container_name} 2>/dev/null || true
sudo docker rm {container_name} 2>/dev/null || true
echo "[iris-controller] Controller stopped"
"""


def build_controller_bootstrap_script(
    docker_image: str,
    port: int,
    config_yaml: str = "",
) -> str:
    """Build bootstrap script for controller VM."""
    if config_yaml:
        config_setup = CONFIG_SETUP_TEMPLATE.format(config_yaml=config_yaml)
        config_volume = "-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro"
        config_flag = "--config /etc/iris/config.yaml"
    else:
        config_setup = "# No config file provided"
        config_volume = ""
        config_flag = ""

    return CONTROLLER_BOOTSTRAP_SCRIPT.format(
        docker_image=docker_image,
        container_name=CONTROLLER_CONTAINER_NAME,
        port=port,
        config_setup=config_setup,
        config_volume=config_volume,
        config_flag=config_flag,
    )


# ============================================================================
# Worker bootstrap
# ============================================================================

# Bootstrap script template for worker VMs.
BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-init] Starting Iris worker bootstrap"

# Fetch TPU metadata from GCE and export as environment variables.
# GCP TPU VMs expose TPU info via instance metadata, not environment variables.
echo "[iris-init] Probing TPU metadata..."
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

# Derive TPU slice name from instance name by stripping worker suffix.
# Handles both -wN (original) and -w-N (GCP multi-host) formats.
# Example: "iris-v5litepod_16-abc123-w0" -> "iris-v5litepod_16-abc123"
# Example: "t1v-n-598bede5-w-0" -> "t1v-n-598bede5"
INSTANCE_NAME=$(curl -sf -H "$METADATA_HEADER" \
    "http://metadata.google.internal/computeMetadata/v1/instance/name" 2>/dev/null || echo "")
if [ -n "$INSTANCE_NAME" ]; then
    export TPU_NAME=$(echo "$INSTANCE_NAME" | sed 's/-w-*[0-9]*$//')
fi

# Fetch accelerator-type as TPU_TYPE (e.g., v5litepod-16)
export TPU_TYPE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/accelerator-type" 2>/dev/null || echo "")

# Fetch agent-worker-number as TPU_WORKER_ID
export TPU_WORKER_ID=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/agent-worker-number" 2>/dev/null || echo "")

# Fetch worker hostnames for multi-host slices (JSON array of network endpoints)
TPU_HOSTNAMES_RAW=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/worker-network-endpoints" 2>/dev/null || echo "")
if [ -n "$TPU_HOSTNAMES_RAW" ]; then
    # Format is "unknown:unknown:ip1,unknown:unknown:ip2,..." — extract IPs (fields with dots)
    export TPU_WORKER_HOSTNAMES=$(echo "$TPU_HOSTNAMES_RAW" | \
        tr ',' '\\n' | while read -r entry; do echo "$entry" | tr ':' '\\n' | grep '[.]'; done | \
        paste -sd ',' - 2>/dev/null || echo "")
fi

# Fetch chips per host bounds from tpu-env metadata (YAML-like key: value format)
TPU_ENV_RAW=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/tpu-env" 2>/dev/null || echo "")
if [ -n "$TPU_ENV_RAW" ]; then
    TOPO_RAW=$(echo "$TPU_ENV_RAW" | grep "^CHIPS_PER_HOST_BOUNDS:" | sed "s/.*: *'\\(.*\\)'/\\1/" | tr -d "'")
    if [ -n "$TOPO_RAW" ]; then
        export TPU_CHIPS_PER_HOST_BOUNDS="$TOPO_RAW"
    fi
fi

if [ -n "$TPU_NAME" ]; then
    echo "[iris-init] TPU detected: name=$TPU_NAME type=$TPU_TYPE worker_id=$TPU_WORKER_ID"
else
    echo "[iris-init] No TPU metadata found"
fi

# Ensure controller address is set
if [ -z "$CONTROLLER_ADDRESS" ]; then
    echo "[iris-init] ERROR: CONTROLLER_ADDRESS not set"
    exit 1
fi

echo "[iris-init] Controller address: $CONTROLLER_ADDRESS"

echo "[iris-init] Installing Docker if needed..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
fi

sudo systemctl start docker || true
if ! sudo docker info > /dev/null 2>&1; then
    echo "[iris-init] ERROR: Docker daemon failed to start"
    exit 1
fi

# Configure docker for GCP Artifact Registry
sudo gcloud auth configure-docker \
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true

# Pull worker image
sudo docker pull {docker_image}

# Stop existing worker if running
if sudo docker ps -a --format '{{{{.Names}}}}' | grep -q "^iris-worker$"; then
    sudo docker stop iris-worker 2>/dev/null || true
    sudo docker rm iris-worker 2>/dev/null || true
fi

# Create cache dir
sudo mkdir -p {cache_dir}

# Start worker container without restart policy first (fail fast during bootstrap)
sudo docker run -d --name iris-worker \
    --network=host \
    -v {cache_dir}:{cache_dir} \
    {env_flags} \
    {docker_image} \
    .venv/bin/python -m iris.cluster.worker.main serve \
        --host 0.0.0.0 --port {worker_port}

echo "[iris-init] Worker container started"

# Wait for health
for i in $(seq 1 60); do
    if curl -sf http://localhost:{worker_port}/health > /dev/null 2>&1; then
        echo "[iris-init] Worker healthy"
        sudo docker update --restart=unless-stopped iris-worker
        exit 0
    fi
    sleep 2
done

echo "[iris-init] ERROR: Worker failed to become healthy after 120s"
echo "[iris-init] Container status:"
sudo docker ps -a -f name=iris-worker --format "table {{{{.Status}}}}\t{{{{.State}}}}"
echo "[iris-init] Container logs:"
sudo docker logs iris-worker --tail 100
exit 1
"""


def _build_env_flags(config: config_pb2.BootstrapConfig, vm_address: str) -> str:
    flags = []
    for key, value in config.env_vars.items():
        flags.append(f"-e {shlex.quote(key)}={shlex.quote(value)}")
    flags.append('-e IRIS_CONTROLLER_ADDRESS="$CONTROLLER_ADDRESS"')
    if vm_address:
        flags.append(f"-e IRIS_VM_ADDRESS={shlex.quote(vm_address)}")

    tpu_env_vars = [
        "TPU_NAME",
        "TPU_TYPE",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
    ]
    for var in tpu_env_vars:
        flags.append(f'-e {var}="${{{var}:-}}"')

    return " ".join(flags)


def build_worker_bootstrap_script(
    config: config_pb2.BootstrapConfig,
    vm_address: str,
    discovery_preamble: str = "",
) -> str:
    """Build the bootstrap script for worker VMs."""
    env_flags = _build_env_flags(config, vm_address)
    main_script = BOOTSTRAP_SCRIPT.format(
        cache_dir=config.cache_dir or "/var/cache/iris",
        docker_image=config.docker_image,
        worker_port=config.worker_port or 10001,
        env_flags=env_flags,
    )
    if discovery_preamble:
        return discovery_preamble + main_script
    return main_script
