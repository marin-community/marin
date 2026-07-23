# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for worker VMs.

Centralizes the worker bootstrap script template and generation logic. Worker
bootstrap handles Docker setup and container startup. TPU metadata discovery
is performed by the worker environment probe at runtime. The shared registry
helpers (registry_host, rewrite_image_to_mirror, render_template) live here so
controller bootstrap can reuse them.
"""

import json
import re
from collections.abc import Mapping

from iris.cluster.config import WorkerConfig
from iris.cluster.runtime.docker import EPHEMERAL_PORT_RANGE, RESERVED_HOST_PORTS

# gVisor release the worker installs so the GVISOR container profile can run task
# containers under `docker --runtime=runsc`. Pin explicitly; bump by checking
# https://github.com/google/gvisor/releases (tag "release-YYYYMMDD.P" publishes
# to the bare "YYYYMMDD.P" path under releases/release/).
RUNSC_VERSION = "20260714.0"


# Canonical ``registry_mirrors`` key for Docker Hub, the implicit default registry.
DOCKER_HUB = "docker.io"

# Host names that all denote Docker Hub.
_DOCKER_HUB_HOSTS = frozenset({DOCKER_HUB, "index.docker.io", "registry-1.docker.io"})


def registry_host(image_tag: str) -> str | None:
    """Return the registry host named by a container image reference, or None if it names none.

    Applies Docker's default-registry rule: the first ``/``-segment is the
    registry host only when it looks like one — it contains a ``.`` or ``:``
    (domain or port) or equals ``localhost``. A reference with no such segment
    (``ubuntu:24.04``, ``library/python``, ``bitnami/redis``) names no registry and
    defaults to Docker Hub.
    """
    if "/" not in image_tag:
        return None
    first_segment = image_tag.split("/", 1)[0]
    if "." in first_segment or ":" in first_segment or first_segment == "localhost":
        return first_segment
    return None


def docker_hub_repo_path(image_tag: str) -> str | None:
    """Return the Docker Hub repository path for *image_tag*, or None if it names another registry.

    Official single-name images gain the implicit ``library/`` namespace, matching
    how Docker Hub stores them:

        ubuntu:24.04                → library/ubuntu:24.04
        bitnami/redis:latest        → bitnami/redis:latest
        docker.io/library/python:3  → library/python:3
        gcr.io/proj/img:v1          → None (another registry)
        us-docker.pkg.dev/p/r/i:v1  → None (Artifact Registry)
    """
    host = registry_host(image_tag)
    if host is None:
        # No registry host: a Docker Hub reference. A single-name image (no '/')
        # gains the implicit 'library/' namespace.
        return image_tag if "/" in image_tag else f"library/{image_tag}"
    if host in _DOCKER_HUB_HOSTS:
        # Explicit Docker Hub host: strip it, keeping (and namespacing) the repo path.
        remainder = image_tag.split("/", 1)[1]
        return remainder if "/" in remainder else f"library/{remainder}"
    return None


def upstream_registry(image_tag: str) -> str:
    """Return the canonical registry key an image reference pulls from.

    Docker Hub aliases (no host, ``docker.io``, ``index.docker.io``,
    ``registry-1.docker.io``) all collapse to ``docker.io``; any other host is
    returned as-is. The result indexes ``registry_mirrors``.
    """
    host = registry_host(image_tag)
    # Exact set membership, never a substring/prefix check: a crafted tag cannot
    # smuggle a trusted host name into an untrusted position.
    if host is None or host in _DOCKER_HUB_HOSTS:
        return DOCKER_HUB
    return host


def rewrite_image_to_mirror(
    image_tag: str,
    zone: str,
    registry_mirrors: Mapping[str, Mapping[str, str]],
) -> str:
    """Rewrite an image reference to the pull-through mirror for its registry and *zone*.

    *registry_mirrors* maps upstream registry → zone prefix (the zone's leading
    dash-separated segment) → mirror repo prefix; see
    ``GcpPlatformConfig.registry_mirrors``. With::

        {"docker.io": {"us": "us-docker.pkg.dev/hai-gcp-models/docker-mirror"}}

    ``ubuntu:24.04`` in ``us-central1-a`` becomes
    ``us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04``.
    Registries and zone prefixes absent from the map pass through unchanged.
    """
    upstream = upstream_registry(image_tag)
    mirror = registry_mirrors.get(upstream, {}).get(zone.split("-", 1)[0])
    if mirror is None:
        return image_tag
    if upstream == DOCKER_HUB:
        path = docker_hub_repo_path(image_tag)
        assert path is not None  # upstream_registry classified this as a Docker Hub reference
    else:
        path = image_tag.split("/", 1)[1]
    return f"{mirror}/{path}"


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
WORKER_BOOTSTRAP_SCRIPT = """#!/bin/bash
set -e

echo "[iris-init] Starting Iris worker bootstrap"

echo "[iris-init] Phase: tpu_ready_gate"

# Gate the whole bootstrap -- and thus controller registration -- on the TPU
# node reaching READY, before any Docker install/pull work. A host can boot and
# run this script while sibling hosts in the same slice are still provisioning;
# registering early would let the controller schedule a task before the slice is
# up. The node's aggregate state flips to READY only once every host is healthy,
# so it is the cleanest in-VM signal that the whole slice is Active.
#
# The gate applies only to TPU slices (accelerator-type metadata present); CPU
# and standalone GCE VMs have no such attribute and skip it. It is fail-open: if
# gcloud is unavailable, the describe never succeeds, or the node never reaches
# READY within the window, bootstrap proceeds anyway and the autoscaler's slice
# health probe owns give-up -- exactly as it does for the /health wait below.
IRIS_META="http://metadata.google.internal/computeMetadata/v1/instance"
IRIS_ACCEL_TYPE=$(curl -sf -H "Metadata-Flavor: Google" "$IRIS_META/attributes/accelerator-type" || true)
if [ -n "$IRIS_ACCEL_TYPE" ]; then
    export PATH="$PATH:/snap/bin:/opt/google-cloud-sdk/bin"
    IRIS_TPU_NODE=$(curl -sf -H "Metadata-Flavor: Google" "$IRIS_META/attributes/instance-id" || true)
    IRIS_TPU_ZONE=$(curl -sf -H "Metadata-Flavor: Google" "$IRIS_META/zone" | sed 's#.*/##')
    if command -v gcloud &> /dev/null && [ -n "$IRIS_TPU_NODE" ]; then
        echo "[iris-init] Gating bootstrap on TPU node $IRIS_TPU_NODE (zone=$IRIS_TPU_ZONE) reaching READY"
        # Reserved/queued multi-host slices can take a long time to fully
        # provision; wait up to an hour before failing open. SECONDS is a bash
        # builtin reset to 0 here, so the deadline accounts for gcloud latency.
        SECONDS=0
        IRIS_TPU_GATE_TIMEOUT=3600
        IRIS_TPU_GATE_INTERVAL=15
        IRIS_TPU_GATE_ATTEMPT=0
        while [ "$SECONDS" -lt "$IRIS_TPU_GATE_TIMEOUT" ]; do
            IRIS_TPU_GATE_ATTEMPT=$((IRIS_TPU_GATE_ATTEMPT + 1))
            IRIS_TPU_STATE=$(gcloud compute tpus tpu-vm describe "$IRIS_TPU_NODE" \
                --zone="$IRIS_TPU_ZONE" --format='value(state)' 2>/dev/null || true)
            if [ "$IRIS_TPU_STATE" = "READY" ]; then
                echo "[iris-init] TPU node READY after ${SECONDS}s (${IRIS_TPU_GATE_ATTEMPT} check(s)); proceeding"
                break
            fi
            echo "[iris-init] TPU node state=${IRIS_TPU_STATE:-<describe-failed>}; waited ${SECONDS}s"
            sleep "$IRIS_TPU_GATE_INTERVAL"
        done
        if [ "$IRIS_TPU_STATE" != "READY" ]; then
            echo "[iris-init] WARNING: TPU node not READY after ${IRIS_TPU_GATE_TIMEOUT}s; proceeding (fail-open)"
        fi
    else
        echo "[iris-init] gcloud or node name unavailable; skipping TPU ready gate"
    fi
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

# Install gVisor (runsc) and register it as a docker runtime so the GVISOR
# container profile can launch task containers under `docker --runtime=runsc`.
# The host dockerd (root) builds the sandbox — see lib/iris/docs/container-profiles.md.
# Best-effort: a failed install leaves the worker usable for every other profile.
if ! command -v runsc &> /dev/null; then
    echo "[iris-init] Installing gVisor (runsc {{ runsc_version }})..."
    RUNSC_ARCH="$(uname -m)"
    RUNSC_BASE="https://storage.googleapis.com/gvisor/releases/release/{{ runsc_version }}/${RUNSC_ARCH}"
    if sudo curl -fsSL "${RUNSC_BASE}/runsc" -o /usr/local/bin/runsc \
        && sudo curl -fsSL "${RUNSC_BASE}/runsc.sha512" -o /tmp/runsc.sha512 \
        && (cd /usr/local/bin && sudo sha512sum -c /tmp/runsc.sha512); then
        sudo chmod 0755 /usr/local/bin/runsc
        echo "[iris-init] runsc installed: $(runsc --version | head -1)"
    else
        echo "[iris-init] Warning: runsc install failed; GVISOR profile unavailable on this worker"
        sudo rm -f /usr/local/bin/runsc
    fi
fi

# Register the runsc runtime in daemon.json (merging, not clobbering) and reload
# dockerd only when it is not already present, so re-runs cause no restart churn.
if command -v runsc &> /dev/null && ! grep -q '"runsc"' /etc/docker/daemon.json 2>/dev/null; then
    echo "[iris-init] Registering runsc docker runtime..."
    sudo python3 - <<'RUNSC_DAEMON_EOF'
import json, os
path = "/etc/docker/daemon.json"
cfg = {}
if os.path.exists(path):
    try:
        with open(path) as f:
            cfg = json.load(f) or {}
    except (ValueError, OSError):
        cfg = {}
cfg.setdefault("runtimes", {})["runsc"] = {"path": "/usr/local/bin/runsc"}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
RUNSC_DAEMON_EOF
    sudo systemctl restart docker
    echo "[iris-init] runsc runtime registered"
fi

# gcloud ships as a snap on tpu-ubuntu2204-base; snapd mounts snaps
# asynchronously during boot. Wait for seeding to finish here so `gcloud`
# is on PATH for Artifact Registry auth below. Placed right after the
# Docker daemon start so seeding overlaps with it and usually returns
# immediately.
if command -v snap &> /dev/null; then
    timeout 300 snap wait system seed.loaded || echo "[iris-init] Warning: snap seed wait timed out"
fi
export PATH="$PATH:/snap/bin"

# Tune network stack for high-connection workloads (#3066).
# Expands ephemeral port range, allows reuse of TIME_WAIT sockets,
# and raises listen backlog for actor servers handling 1000s of workers.
# The ephemeral floor stays above every port the cluster statically allocates:
# the fixed service ports (TPU/JAX 8081/8431/8470-8482, iris 10000/10001) and
# the task named-port range (default 12000-13999). A statically allocated port
# inside the ephemeral range gets squatted by outbound connections — the TPU
# trainer crash-looping on "[::]:8431 ... Address already in use" when the
# floor was 1024, and task binds dying with EADDRINUSE in the window between
# port allocation and container setup finishing when the task range lived at
# 30000-40000 (#7392). Host-network task containers share this netns, so this
# covers them. ip_local_reserved_ports pins the same ports as defense-in-depth
# for clusters that override the task range back into the ephemeral span; it
# only exempts ports from automatic assignment, explicit binds are unaffected.
sudo sysctl -w net.ipv4.ip_local_port_range="{{ port_range }}"
sudo sysctl -w net.ipv4.ip_local_reserved_ports="{{ reserved_ports }}"
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
sudo sysctl -w net.core.somaxconn=4096

# Create cache directory
sudo mkdir -p {{ cache_dir }}

echo "[iris-init] Phase: docker_pull"
echo "[iris-init] Pulling image: {{ docker_image }}"

# Resolve the Artifact Registry host (empty for non-AR images). Auth is only
# configured when pulling from AR; root's docker config is used by `sudo docker`.
AR_HOST=""
if echo "{{ docker_image }}" | grep -q -- "-docker.pkg.dev/"; then
    AR_HOST=$(echo "{{ docker_image }}" | cut -d/ -f1)
fi

# Retry AR auth + pull. gcloud ships as a snap on tpu-ubuntu2204-base and can be
# slow to become usable at first boot even after `snap wait system seed.loaded`:
# /snap/bin/gcloud may not be linked yet, or docker-credential-gcloud may fail
# mid-pull ("the required argument <snap> was not provided"). Either way docker
# falls back to an unauthenticated request and Artifact Registry denies it.
# Re-running configure-docker + pull on each attempt absorbs the race. This MUST
# retry: the pull runs before the self-healing --restart=unless-stopped worker
# container is created, so a single transient failure here strands the worker
# permanently -- its /health never comes up and the slice health probe
# eventually reaps the whole slice, healthy siblings included.
IRIS_PULL_OK=0
for attempt in $(seq 1 20); do
    if [ -n "$AR_HOST" ]; then
        echo "[iris-init] Configuring docker auth for $AR_HOST (attempt $attempt/20)"
        if command -v gcloud &> /dev/null; then
            sudo gcloud auth configure-docker "$AR_HOST" -q || true
        else
            echo "[iris-init] gcloud not yet on PATH; waiting for snap to settle"
        fi
    fi
    if sudo docker pull {{ docker_image }}; then
        IRIS_PULL_OK=1
        break
    fi
    echo "[iris-init] docker pull failed (attempt $attempt/20); retrying in 15s"
    sleep 15
done

if [ "$IRIS_PULL_OK" -ne 1 ]; then
    echo "[iris-init] ERROR: docker pull failed after 20 attempts; giving up"
    exit 1
fi

echo "[iris-init] Phase: config_setup"
sudo mkdir -p /etc/iris
cat > /tmp/iris_worker_config.json << 'IRIS_WORKER_CONFIG_EOF'
{{ worker_config_json }}
IRIS_WORKER_CONFIG_EOF
sudo mv /tmp/iris_worker_config.json /etc/iris/worker_config.json

echo "[iris-init] Phase: worker_start"

# Force-remove existing worker (handles restart policy race).
# Task containers are NOT removed here — the worker process handles
# adoption-or-cleanup in start() so it can adopt running containers
# from a previous worker during rolling restarts.
sudo docker rm -f iris-worker 2>/dev/null || true

# Start worker container with restart policy from the start so transient
# failures (image pull races, network hiccups, etc.) self-heal. Give-up is
# owned by the autoscaler's slice health probe, not by docker.
sudo docker run -d --name iris-worker \\
    --restart=unless-stopped \\
    --network=host \\
    --ulimit core=0:0 \\
    -v {{ cache_dir }}:{{ cache_dir }} \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    -v /etc/iris/worker_config.json:/etc/iris/worker_config.json:ro \\
    {{ docker_image }} \\
    .venv/bin/python -m iris.cluster.worker.main serve \\
        --worker-config /etc/iris/worker_config.json

echo "[iris-init] Worker container started"
echo "[iris-init] Phase: registration"
echo "[iris-init] Waiting for worker to register with controller..."

# Poll the health endpoint to report bootstrap status. Docker handles
# restarts; the autoscaler health probe handles give-up if /health never
# comes up.
for i in $(seq 1 60); do
    if curl -sf http://localhost:{{ worker_port }}/health > /dev/null 2>&1; then
        echo "[iris-init] Worker is healthy"
        echo "[iris-init] Bootstrap complete"
        exit 0
    fi
    sleep 2
done

echo "[iris-init] WARNING: Worker not healthy after 120s. Docker will keep restarting the"
echo "[iris-init] container (--restart=unless-stopped); the autoscaler health probe will reap"
echo "[iris-init] this slice if /health stays down for ~100s of probes."
echo "[iris-init] Container status:"
sudo docker ps -a -f name=iris-worker --format "table {{.Status}}\\t{{.State}}" 2>&1 | sed 's/^/[iris-init] /'
echo "[iris-init] Container logs:"
sudo docker logs iris-worker --tail 100 2>&1 | sed 's/^/[iris-init] /'
exit 1
"""


def build_worker_bootstrap_script(
    worker_config: WorkerConfig,
) -> str:
    """Build the bootstrap script for a worker VM.

    Serializes the WorkerConfig as JSON and embeds it in the bootstrap script.
    The worker reads the JSON at startup via --worker-config.
    """
    if not worker_config.controller_address:
        raise ValueError("worker_config.controller_address is required for worker bootstrap")
    if not worker_config.docker_image:
        raise ValueError("worker_config.docker_image is required for worker bootstrap")
    if worker_config.port <= 0:
        raise ValueError("worker_config.port must be > 0 for worker bootstrap")
    if not worker_config.cache_dir:
        raise ValueError("worker_config.cache_dir is required for worker bootstrap")

    worker_config_json = json.dumps(
        worker_config.model_dump(mode="json", exclude_none=True),
        indent=2,
    )

    # Reserve the task named-port range from kernel ephemeral assignment (see
    # the sysctl comment in the template). The range's end bound is exclusive
    # (PortAllocator scans range(start, end)), so the last reserved port is
    # end - 1. Parsing validates the configured range and fails the deploy on
    # a malformed value.
    task_port_start, task_port_end = map(int, worker_config.port_range.split("-"))

    return render_template(
        WORKER_BOOTSTRAP_SCRIPT,
        cache_dir=worker_config.cache_dir,
        docker_image=worker_config.docker_image,
        worker_port=worker_config.port,
        worker_config_json=worker_config_json,
        port_range=EPHEMERAL_PORT_RANGE,
        reserved_ports=f"{RESERVED_HOST_PORTS},{task_port_start}-{task_port_end - 1}",
        runsc_version=RUNSC_VERSION,
    )
