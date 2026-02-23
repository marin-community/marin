# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for worker and controller VMs.

Centralizes all bootstrap script templates and generation logic. Worker
bootstrap handles Docker setup and container startup. TPU metadata discovery
is performed by the worker environment probe at runtime.
"""

from __future__ import annotations

import logging
import re
import shlex

import yaml

from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


def parse_artifact_registry_tag(image_tag: str) -> tuple[str, str, str, str] | None:
    """Parse ``REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:VERSION``.

    Returns:
        (region, project, image_name, version) or None if not an AR tag.
    """
    if "-docker.pkg.dev/" not in image_tag:
        return None
    parts = image_tag.split("/")
    if len(parts) < 4:
        return None
    registry = parts[0]
    if not registry.endswith("-docker.pkg.dev"):
        return None
    region = registry.replace("-docker.pkg.dev", "")
    project = parts[1]
    image_and_version = parts[3]
    if ":" in image_and_version:
        image_name, version = image_and_version.split(":", 1)
    else:
        image_name = image_and_version
        version = "latest"
    return region, project, image_name, version


def rewrite_artifact_registry_region(image_tag: str, target_region: str) -> str:
    """Rewrite an Artifact Registry image tag to use a different region.

    Non-AR images pass through unchanged. If the image is already in the
    target region, returns the original tag.
    """
    parsed = parse_artifact_registry_tag(image_tag)
    if parsed is None:
        return image_tag
    current_region = parsed[0]
    if current_region == target_region:
        return image_tag
    parts = image_tag.split("/")
    parts[0] = f"{target_region}-docker.pkg.dev"
    return "/".join(parts)


def collect_all_regions(config: config_pb2.IrisClusterConfig) -> set[str]:
    """Extract all unique GCP regions from an Iris cluster config.

    Includes regions from all scale group zones and the controller zone.
    """
    regions: set[str] = set()

    for sg in config.scale_groups.values():
        template = sg.slice_template
        if template.HasField("gcp") and template.gcp.zone:
            regions.add(template.gcp.zone.rsplit("-", 1)[0])

    ctrl = config.controller
    if ctrl.HasField("gcp") and ctrl.gcp.zone:
        regions.add(ctrl.gcp.zone.rsplit("-", 1)[0])

    return regions


def render_template(template: str, **variables: str | int) -> str:
    """Render a template string with {{ variable }} placeholders.

    Uses ``{{ variable }}`` syntax (double braces with exactly one space) to
    avoid conflicts with shell ``${var}`` and Docker ``{{.Field}}`` syntax.

    Args:
        template: Template string with ``{{ variable }}`` placeholders.
        **variables: Variable values to substitute.

    Returns:
        Rendered template string.

    Raises:
        ValueError: If a required variable is missing from the template or if
            variables are passed that do not appear in the template.
    """
    used_vars: set[str] = set()

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name not in variables:
            raise ValueError(f"Template variable '{var_name}' not provided")
        used_vars.add(var_name)
        value = variables[var_name]
        return str(value)

    # Match {{ variable_name }} — exactly one space inside each brace pair.
    result = re.sub(r"\{\{ (\w+) \}\}", replace_var, template)

    unused = set(variables) - used_vars
    if unused:
        raise ValueError(f"Unused template variables: {', '.join(sorted(unused))}")

    return result


# ============================================================================
# Worker Bootstrap Script
# ============================================================================


# Bootstrap script template for worker VMs.
WORKER_BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-init] Starting Iris worker bootstrap"

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
sudo mkdir -p {{ cache_dir }}

echo "[iris-init] Phase: docker_pull"
echo "[iris-init] Pulling image: {{ docker_image }}"

# Configure Artifact Registry auth on demand.
if echo "{{ docker_image }}" | grep -q -- "-docker.pkg.dev/"; then
    AR_HOST=$(echo "{{ docker_image }}" | cut -d/ -f1)
    echo "[iris-init] Configuring docker auth for $AR_HOST"
    if command -v gcloud &> /dev/null; then
        gcloud auth configure-docker "$AR_HOST" -q || true
    else
        echo "[iris-init] Warning: gcloud not found; AR pull may fail without prior auth"
    fi
fi

sudo docker pull {{ docker_image }}

echo "[iris-init] Phase: config_setup"
sudo mkdir -p /etc/iris
cat > /tmp/iris_config.json << 'IRIS_CONFIG_EOF'
{{ config_json }}
IRIS_CONFIG_EOF
sudo mv /tmp/iris_config.json /etc/iris/config.json

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
    -v {{ cache_dir }}:{{ cache_dir }} \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    -v /etc/iris/config.json:/etc/iris/config.json:ro \\
    {{ env_flags }} \\
    {{ docker_image }} \\
    .venv/bin/python -m iris.cluster.worker.main serve \\
        --host 0.0.0.0 --port {{ worker_port }} \\
        --cache-dir {{ cache_dir }} \\
        --controller-address {{ controller_address }} \\
        --config /etc/iris/config.json

echo "[iris-init] Worker container started"
echo "[iris-init] Phase: registration"
echo "[iris-init] Waiting for worker to register with controller..."

# Wait for worker to be healthy (poll health endpoint)
for i in $(seq 1 60); do
    # Check if container is still running
    if ! sudo docker ps -q -f name=iris-worker | grep -q .; then
        echo "[iris-init] ERROR: Worker container exited unexpectedly"
        echo "[iris-init] Container status:"
        sudo docker ps -a -f name=iris-worker --format "table {{.Status}}\\t{{.State}}"
        echo "[iris-init] Container logs:"
        sudo docker logs iris-worker --tail 100
        exit 1
    fi

    if curl -sf http://localhost:{{ worker_port }}/health > /dev/null 2>&1; then
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
sudo docker ps -a -f name=iris-worker --format "table {{.Status}}\\t{{.State}}"
echo "[iris-init] Container logs:"
sudo docker logs iris-worker --tail 100
exit 1
"""


def build_worker_env_flags(
    config: config_pb2.BootstrapConfig,
    vm_address: str,
) -> str:
    """Generate docker -e flags with proper escaping.

    TPU metadata is probed by the worker process via env_probe.py, so bootstrap
    only forwards explicit bootstrap env vars plus IRIS_VM_ADDRESS.
    """
    env_vars = dict(config.env_vars)

    flags = []
    for k, v in env_vars.items():
        flags.append(f"-e {shlex.quote(k)}={shlex.quote(v)}")
    # Inject VM address so worker can include it in registration for autoscaler tracking
    if vm_address:
        flags.append(f"-e IRIS_VM_ADDRESS={shlex.quote(vm_address)}")

    return " ".join(flags)


def build_worker_bootstrap_script(
    bootstrap_config: config_pb2.BootstrapConfig,
    vm_address: str,
) -> str:
    """Build the bootstrap script for a worker VM.

    Args:
        bootstrap_config: Worker bootstrap settings
        vm_address: VM IP address for autoscaler tracking
    """
    env_flags = build_worker_env_flags(bootstrap_config, vm_address)
    if not bootstrap_config.controller_address:
        raise ValueError("bootstrap_config.controller_address is required for worker bootstrap")
    if not bootstrap_config.docker_image:
        raise ValueError("bootstrap_config.docker_image is required for worker bootstrap")
    if bootstrap_config.worker_port <= 0:
        raise ValueError("bootstrap_config.worker_port must be > 0 for worker bootstrap")
    if not bootstrap_config.cache_dir:
        raise ValueError("bootstrap_config.cache_dir is required for worker bootstrap")

    return render_template(
        WORKER_BOOTSTRAP_SCRIPT,
        cache_dir=bootstrap_config.cache_dir,
        docker_image=bootstrap_config.docker_image,
        worker_port=bootstrap_config.worker_port,
        controller_address=bootstrap_config.controller_address,
        env_flags=env_flags,
        config_json=bootstrap_config.config_json,
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
{{ config_setup }}

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

echo "[iris-controller] [3/5] Pulling image: {{ docker_image }}"
echo "[iris-controller]       This may take several minutes for large images..."

# Configure Artifact Registry auth on demand.
if echo "{{ docker_image }}" | grep -q -- "-docker.pkg.dev/"; then
    AR_HOST=$(echo "{{ docker_image }}" | cut -d/ -f1)
    echo "[iris-controller] [3/5] Configuring docker auth for $AR_HOST"
    if command -v gcloud &> /dev/null; then
        gcloud auth configure-docker "$AR_HOST" -q || true
    else
        echo "[iris-controller] [3/5] Warning: gcloud not found; AR pull may fail without prior auth"
    fi
fi

if sudo docker pull {{ docker_image }}; then
    echo "[iris-controller] [4/5] Image pull complete"
else
    echo "[iris-controller] [4/5] ERROR: Image pull failed"
    exit 1
fi

# Stop existing controller if running
echo "[iris-controller] [5/5] Starting controller container..."
if sudo docker ps -a --format '{{.Names}}' | grep -q "^{{ container_name }}$"; then
    echo "[iris-controller]       Stopping existing container..."
    sudo docker stop {{ container_name }} 2>/dev/null || true
    sudo docker rm {{ container_name }} 2>/dev/null || true
fi

# Create cache directory
sudo mkdir -p /var/cache/iris

# Start controller container with restart policy
sudo docker run -d --name {{ container_name }} \\
    --network=host \\
    --restart=unless-stopped \\
    -v /var/cache/iris:/var/cache/iris \\
    {{ config_volume }} \\
    {{ docker_image }} \\
    .venv/bin/python -m iris.cluster.controller.main serve \\
        --host 0.0.0.0 --port {{ port }} {{ config_flag }}

echo "[iris-controller] [5/5] Controller container started"

# Wait for health
echo "[iris-controller] Waiting for controller to become healthy..."
RESTART_COUNT=0
for i in $(seq 1 30); do
    echo "[iris-controller] Health check attempt $i/30 at $(date -Iseconds)..."
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
echo "[iris-controller] ERROR: Controller failed to become healthy after 60 seconds"
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

    return render_template(
        CONTROLLER_BOOTSTRAP_SCRIPT,
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
    # Local import to avoid circular dependency (config.py imports from bootstrap)
    from iris.cluster.config import config_to_dict

    config_yaml = yaml.dump(config_to_dict(config), default_flow_style=False)
    port = config.controller.gcp.port or config.controller.manual.port or 10000
    image = config.controller.image
    ctrl = config.controller
    if ctrl.HasField("gcp") and ctrl.gcp.zone:
        controller_region = ctrl.gcp.zone.rsplit("-", 1)[0]
        image = rewrite_artifact_registry_region(image, controller_region)

    return build_controller_bootstrap_script(image, port, config_yaml)
