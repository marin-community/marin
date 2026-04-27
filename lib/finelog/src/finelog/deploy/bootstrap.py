# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for finelog GCE VMs.

The bootstrap script is idempotent: it can be re-run on an existing VM (over
SSH for `finelog restart`) and will replace any running `finelog` container
with a fresh one pulled from the requested image.

The finelog Docker image is assumed to be public on GHCR; no Artifact Registry
or `docker login` wiring is performed here.
"""

from __future__ import annotations

import re

# Container/host conventions baked into the bootstrap.
CONTAINER_NAME = "finelog"
CACHE_DIR = "/var/cache/finelog"


def render_template(template: str, **variables: str | int) -> str:
    """Render a `{{ variable }}` template (single space inside the braces).

    Tiny stand-alone copy of the iris helper — kept local so finelog has no
    dependency on iris.
    """
    used: set[str] = set()

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            raise ValueError(f"Template variable '{name}' not provided")
        used.add(name)
        return str(variables[name])

    result = re.sub(r"\{\{ (\w+) \}\}", replace, template)
    unused = set(variables) - used
    if unused:
        raise ValueError(f"Unused template variables: {', '.join(sorted(unused))}")
    return result


# Bootstrap script run on the VM. Used both as a startup-script (on `create`)
# and over SSH (on `restart`). Idempotent.
BOOTSTRAP_SCRIPT = """#!/bin/bash
set -e

echo "[finelog-init] Starting finelog bootstrap at $(date -Iseconds)"

# Install Docker if missing.
if ! command -v docker &> /dev/null; then
    echo "[finelog-init] Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
fi
sudo systemctl start docker || true

# Cache directory on the boot disk. Finelog offloads parquet segments to GCS
# via FINELOG_REMOTE_DIR, so the boot disk only needs working space.
sudo mkdir -p {{ cache_dir }}

echo "[finelog-init] Pulling image: {{ docker_image }}"
sudo docker pull {{ docker_image }}

# Force-remove any existing finelog container so re-runs replace it.
sudo docker rm -f {{ container_name }} 2>/dev/null || true

sudo docker run -d --name {{ container_name }} \\
    --network=host \\
    --restart=unless-stopped \\
    --ulimit core=0:0 \\
    -e FINELOG_REMOTE_DIR={{ remote_log_dir }} \\
    -v {{ cache_dir }}:{{ cache_dir }} \\
    {{ docker_image }}

echo "[finelog-init] Container started; waiting for /health on port {{ port }}..."

for i in $(seq 1 60); do
    if ! sudo docker ps -q -f name={{ container_name }} | grep -q .; then
        echo "[finelog-init] ERROR: finelog container exited unexpectedly"
        sudo docker ps -a -f name={{ container_name }}
        sudo docker logs {{ container_name }} --tail 200 || true
        exit 1
    fi
    if curl -sf http://localhost:{{ port }}/health > /dev/null 2>&1; then
        echo "[finelog-init] finelog is healthy"
        echo "[finelog-init] Bootstrap complete"
        exit 0
    fi
    sleep 2
done

echo "[finelog-init] ERROR: finelog failed to become healthy after 120s"
sudo docker ps -a -f name={{ container_name }}
sudo docker logs {{ container_name }} --tail 200 || true
exit 1
"""


def render_bootstrap(image: str, port: int, remote_log_dir: str) -> str:
    """Render the finelog bootstrap script."""
    if not image:
        raise ValueError("image is required")
    if port <= 0:
        raise ValueError("port must be > 0")
    return render_template(
        BOOTSTRAP_SCRIPT,
        docker_image=image,
        port=port,
        remote_log_dir=remote_log_dir,
        cache_dir=CACHE_DIR,
        container_name=CONTAINER_NAME,
    )
