# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for worker and controller VMs.

Centralizes all bootstrap script templates and generation logic. Worker
bootstrap handles TPU metadata discovery, Docker setup, and container
startup. Controller bootstrap is simpler — Docker setup plus container start.
"""

from __future__ import annotations

import logging
import shlex
import time

import yaml

from iris.cluster.platform.base import PlatformError, SliceHandle
from iris.cluster.types import get_tpu_topology
from iris.rpc import config_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)

# ============================================================================
# Worker Bootstrap
# ============================================================================


class WorkerBootstrap:
    """Bootstraps worker VMs with embedded cluster config.

    Workers receive the full cluster config.yaml and use it to discover
    the controller themselves via platform.discover_controller().
    The autoscaler creates a WorkerBootstrap and calls bootstrap_slice()
    after each scale-up.
    """

    def __init__(self, cluster_config: config_pb2.IrisClusterConfig):
        self._cluster_config = cluster_config

    def bootstrap_slice(self, handle: SliceHandle) -> dict[str, str]:
        """Bootstrap all VMs in a newly created slice.

        Args:
            handle: SliceHandle for the newly created slice.

        Returns:
            Mapping of vm_id to bootstrap log text captured during bootstrap.

        Raises:
            PlatformError: If any VM has no internal address after becoming reachable,
                or if wait_for_connection times out, or if not all expected VMs are present.
        """
        # Wait for all expected VMs to be present before bootstrapping.
        # TPU slices can return partial VM lists during provisioning (e.g., 3 of 4 VMs).
        self._wait_for_all_vms(handle)

        logs: dict[str, str] = {}
        for vm in handle.list_vms():
            if not vm.wait_for_connection(timeout=Duration.from_seconds(300)):
                raise PlatformError(
                    f"VM {vm.vm_id} in slice {handle.slice_id} failed to become reachable "
                    f"within timeout. The VM may be stuck in a boot loop or networking may be misconfigured."
                )
            if not vm.internal_address:
                raise PlatformError(
                    f"VM {vm.vm_id} in slice {handle.slice_id} has no internal address. "
                    f"The slice may still be provisioning network endpoints."
                )
            script = self._build_script(vm.internal_address)
            vm.bootstrap(script)
            logs[vm.vm_id] = vm.bootstrap_log
        return logs

    def _wait_for_all_vms(self, handle: SliceHandle) -> None:
        """Wait for all expected VMs to be present in the slice.

        During TPU provisioning, list_vms() may return partial results as network
        endpoints are gradually created. This polls until the expected VM count
        (from TPU topology) is reached or times out.

        Raises:
            PlatformError: If the expected VM count is not reached within timeout.
        """
        # Derive expected VM count from accelerator_variant label if present.
        # For slices created by the autoscaler, this label is always set.
        # For manually created slices or other platforms (manual, local), skip the check.
        accelerator_variant = handle.labels.get("iris-accelerator-variant", "")
        if not accelerator_variant:
            logger.debug(
                "Slice %s has no iris-accelerator-variant label; skipping VM count check",
                handle.slice_id,
            )
            return

        try:
            topology = get_tpu_topology(accelerator_variant)
            expected_vm_count = topology.vm_count
        except ValueError:
            logger.warning(
                "Unknown accelerator variant %s for slice %s; skipping VM count check",
                accelerator_variant,
                handle.slice_id,
            )
            return

        # Poll for up to 60 seconds
        timeout = 60.0
        poll_interval = 2.0
        start = time.time()

        while time.time() - start < timeout:
            vms = handle.list_vms()
            if len(vms) >= expected_vm_count:
                logger.info(
                    "Slice %s has all %d expected VMs ready for bootstrap",
                    handle.slice_id,
                    expected_vm_count,
                )
                return
            logger.debug(
                "Slice %s has %d/%d VMs ready; waiting for remaining VMs...",
                handle.slice_id,
                len(vms),
                expected_vm_count,
            )
            time.sleep(poll_interval)

        # Final check after timeout
        vms = handle.list_vms()
        if len(vms) < expected_vm_count:
            raise PlatformError(
                f"Slice {handle.slice_id} has only {len(vms)}/{expected_vm_count} VMs "
                f"ready after {timeout}s. The slice may be stuck in provisioning."
            )

    def _build_script(self, vm_address: str) -> str:
        """Build the full bootstrap script for a single VM."""
        return build_worker_bootstrap_script(self._cluster_config, vm_address)


# Bootstrap script template for worker VMs.
#
# Workers receive the full cluster config.yaml and use platform.discover_controller()
# to find the controller address themselves. The config is embedded in the script
# and written to /etc/iris/config.yaml.
WORKER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-init] Starting Iris worker bootstrap"

# Write config file
{config_setup}

# Fetch TPU metadata from GCE and export as environment variables.
# GCP TPU VMs expose TPU info via instance metadata, not environment variables.
echo "[iris-init] Probing TPU metadata..."
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

# Derive TPU slice name from instance name by stripping worker suffix.
# Handles both -wN (original) and -w-N (GCP multi-host) formats.
# Example: "iris-v5litepod_16-abc123-w0" -> "iris-v5litepod_16-abc123"
# Example: "t1v-n-598bede5-w-0" -> "t1v-n-598bede5"
INSTANCE_NAME=$(curl -sf -H "$METADATA_HEADER" \\
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
    export TPU_WORKER_HOSTNAMES=$(echo "$TPU_HOSTNAMES_RAW" | \\
        tr ',' '\\n' | while read -r entry; do echo "$entry" | tr ':' '\\n' | grep '[.]'; done | \\
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
    echo "[iris-init] No TPU metadata found (this may be a non-TPU VM)"
fi

echo "[iris-init] Phase: prerequisites"

# Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[iris-init] Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "[iris-init] Docker installed"
else
    echo "[iris-init] Docker already installed"
fi

# Ensure docker daemon is running
sudo systemctl start docker || true

# Create cache directory
sudo mkdir -p {cache_dir}

echo "[iris-init] Phase: docker_pull"
echo "[iris-init] Configuring docker authentication"
# Configure docker for common GCP Artifact Registry regions
sudo gcloud auth configure-docker \\
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true

echo "[iris-init] Pulling image: {docker_image}"
sudo docker pull {docker_image}

# Pull the pre-built task image (base image for job containers).
# Derive registry path from the worker image by replacing the image name.
TASK_IMAGE_REGISTRY=$(echo "{docker_image}" | sed 's|/iris-worker:|/iris-task:|')
echo "[iris-init] Pulling task image: $TASK_IMAGE_REGISTRY"
if sudo docker pull "$TASK_IMAGE_REGISTRY"; then
    sudo docker tag "$TASK_IMAGE_REGISTRY" iris-task:latest
    echo "[iris-init] Task image tagged as iris-task:latest"
else
    echo "[iris-init] WARNING: Failed to pull task image, jobs may fail"
fi

echo "[iris-init] Phase: worker_start"

# Force-remove existing worker (handles restart policy race)
sudo docker rm -f iris-worker 2>/dev/null || true

# Clean up ALL iris-managed task containers by label
echo "[iris-init] Cleaning up iris task containers"
IRIS_CONTAINERS=$(sudo docker ps -aq --filter "label=iris.managed=true" 2>/dev/null || true)
if [ -n "$IRIS_CONTAINERS" ]; then
    sudo docker rm -f $IRIS_CONTAINERS 2>/dev/null || true
fi

# Start worker container without restart policy first (fail fast during bootstrap)
sudo docker run -d --name iris-worker \\
    --network=host \\
    -v {cache_dir}:{cache_dir} \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    {config_volume} \\
    {env_flags} \\
    {docker_image} \\
    .venv/bin/python -m iris.cluster.worker.main serve \\
        --host 0.0.0.0 --port {worker_port} \\
        --cache-dir {cache_dir} \\
        --config /etc/iris/config.yaml

echo "[iris-init] Worker container started"
echo "[iris-init] Phase: registration"
echo "[iris-init] Waiting for worker to register with controller..."

# Wait for worker to be healthy (poll health endpoint)
for i in $(seq 1 60); do
    # Check if container is still running
    if ! sudo docker ps -q -f name=iris-worker | grep -q .; then
        echo "[iris-init] ERROR: Worker container exited unexpectedly"
        echo "[iris-init] Container status:"
        sudo docker ps -a -f name=iris-worker --format "table {{{{.Status}}}}\\t{{{{.State}}}}"
        echo "[iris-init] Container logs:"
        sudo docker logs iris-worker --tail 100
        exit 1
    fi

    if curl -sf http://localhost:{worker_port}/health > /dev/null 2>&1; then
        echo "[iris-init] Worker is healthy"
        # Now add restart policy for production
        sudo docker update --restart=unless-stopped iris-worker
        echo "[iris-init] Bootstrap complete"
        exit 0
    fi
    sleep 2
done

echo "[iris-init] ERROR: Worker failed to become healthy after 120s"
echo "[iris-init] Container status:"
sudo docker ps -a -f name=iris-worker --format "table {{{{.Status}}}}\\t{{{{.State}}}}"
echo "[iris-init] Container logs:"
sudo docker logs iris-worker --tail 100
exit 1
"""


def build_worker_env_flags(config: config_pb2.BootstrapConfig, vm_address: str) -> str:
    """Generate docker -e flags with proper escaping.

    TPU environment variables (TPU_NAME, TPU_WORKER_ID, etc.) are passed through
    from the host if they exist. These are set by GCP on TPU VMs and are required
    for the worker to register with tpu-name and tpu-worker-id attributes needed
    for coscheduled job scheduling.
    """
    flags = []
    for k, v in config.env_vars.items():
        flags.append(f"-e {shlex.quote(k)}={shlex.quote(v)}")
    # Inject VM address so worker can include it in registration for autoscaler tracking
    if vm_address:
        flags.append(f"-e IRIS_VM_ADDRESS={shlex.quote(vm_address)}")

    # Pass through TPU environment variables from host if they exist.
    # These are fetched from GCE metadata by the bootstrap script preamble.
    tpu_env_vars = [
        "TPU_NAME",
        "TPU_TYPE",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
    ]
    for var in tpu_env_vars:
        # Use shell syntax to pass through only if set on host
        flags.append(f'-e {var}="${{{var}:-}}"')

    return " ".join(flags)


def build_worker_bootstrap_script(
    cluster_config: config_pb2.IrisClusterConfig,
    vm_address: str,
) -> str:
    """Build the bootstrap script for a worker VM.

    The worker receives the full cluster config.yaml and discovers the controller
    itself via platform.discover_controller().

    Args:
        cluster_config: Full cluster configuration
        vm_address: VM IP address for autoscaler tracking
    """
    # Local import to avoid circular dependency (config.py imports WorkerBootstrap)
    from iris.cluster.config import config_to_dict

    bootstrap_config = cluster_config.defaults.bootstrap

    # Serialize cluster config to YAML and embed it
    config_yaml = yaml.dump(config_to_dict(cluster_config), default_flow_style=False)
    config_setup = _build_config_setup(config_yaml, log_prefix="[iris-init]")

    env_flags = build_worker_env_flags(bootstrap_config, vm_address)

    return WORKER_BOOTSTRAP_SCRIPT.format(
        config_setup=config_setup,
        cache_dir=bootstrap_config.cache_dir or "/var/cache/iris",
        docker_image=bootstrap_config.docker_image,
        worker_port=bootstrap_config.worker_port or 10001,
        config_volume="-v /etc/iris/config.yaml:/etc/iris/config.yaml:ro",
        env_flags=env_flags,
    )


# ============================================================================
# Controller Bootstrap
# ============================================================================

CONTROLLER_CONTAINER_NAME = "iris-controller"

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
sudo gcloud auth configure-docker \\
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
sudo docker run -d --name {container_name} \\
    --network=host \\
    --restart=unless-stopped \\
    -v /var/cache/iris:/var/cache/iris \\
    {config_volume} \\
    {docker_image} \\
    .venv/bin/python -m iris.cluster.controller.main serve \\
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
echo "{log_prefix} Config written to /etc/iris/config.yaml"
"""


def _build_config_setup(config_yaml: str, log_prefix: str) -> str:
    """Generate config setup script fragment with given log prefix."""
    return CONFIG_SETUP_TEMPLATE.format(config_yaml=config_yaml, log_prefix=log_prefix)


def build_controller_bootstrap_script(
    docker_image: str,
    port: int,
    config_yaml: str = "",
) -> str:
    """Build bootstrap script for controller VM.

    Args:
        docker_image: Docker image to run
        port: Controller port
        config_yaml: Optional YAML config to write to /etc/iris/config.yaml
    """
    if config_yaml:
        config_setup = _build_config_setup(config_yaml, log_prefix="[iris-controller]")
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


def build_controller_bootstrap_script_from_config(
    config: config_pb2.IrisClusterConfig,
) -> str:
    """Build controller bootstrap script from the full cluster config.

    Serializes the config to YAML and embeds it in the bootstrap script.
    """
    # Local import to avoid circular dependency (config.py imports WorkerBootstrap)
    from iris.cluster.config import config_to_dict

    config_yaml = yaml.dump(config_to_dict(config), default_flow_style=False)
    port = config.controller.gcp.port or config.controller.manual.port or 10000
    return build_controller_bootstrap_script(config.controller.image, port, config_yaml)
