# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap script generation for finelog GCE VMs.

The bootstrap script is idempotent: it can be re-run on an existing VM (over
SSH for `finelog restart`) and will replace any running `finelog` container
with a fresh one pulled from the requested image.

The finelog Docker image is assumed to be public on GHCR; no Artifact Registry
or `docker login` wiring is performed here.
"""

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

# Cache directory on the boot disk. Finelog copies parquet segments to GCS
# via FINELOG_REMOTE_DIR, so the boot disk only needs working space.
sudo mkdir -p {{ cache_dir }}

echo "[finelog-init] Pulling image: {{ docker_image }}"
sudo docker pull {{ docker_image }}

# Gracefully stop any existing container so the server can flush RAM
# buffers to L0 on disk, then remove it. This MUST precede the chown below:
# a running server constantly creates and renames transient files
# (seg_L*.parquet.tmp, _finelog_catalog.sqlite-journal). One vanishing between
# readdir and the chown syscall makes `chown -R` exit non-zero, and `set -e`
# would then abort the whole bootstrap before the new image is ever started.
sudo docker stop --timeout 5 {{ container_name }} 2>/dev/null || true
sudo docker rm -f {{ container_name }} 2>/dev/null || true

# Own the cache dir as UID/GID 1000 to match the in-container `finelog` user
# (the Dockerfile's chown is shadowed by this bind mount). Safe to recurse now
# that the writer is stopped.
sudo chown -R 1000:1000 {{ cache_dir }}

host_cpus=$(nproc)
host_mem_mib=$(awk '/^MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo)
host_reserved_mem_mib=1700
container_mem_mib=$(( host_mem_mib - host_reserved_mem_mib ))
if [ "$container_mem_mib" -lt 256 ]; then container_mem_mib=256; fi
echo "[finelog-init] host=${host_cpus}cpu/${host_mem_mib}MiB" \\
    "container_mem=${container_mem_mib}MiB" \\
    "host_reserved_mem=${host_reserved_mem_mib}MiB"

# No --cpus cap: with Docker's --cpus the cgroup gets a CFS bandwidth quota
# (e.g. quota=350000us per 100000us period for --cpus=3.5). When the
# container's combined threads exhaust the quota mid-period the kernel
# parks every thread in the cgroup until the next 100ms window, which
# shows up as multi-hundred-ms spikes on otherwise-fast operations like
# parquet encode. The VM is dedicated to finelog, so let it use all
# host CPUs without bandwidth throttling.
sudo docker run -d --name {{ container_name }} \\
    --network=host \\
    --restart=unless-stopped \\
    --ulimit core=0:0 \\
    --ulimit nofile=1048576:1048576 \\
    --cap-add SYS_PTRACE \\
    --memory="${container_mem_mib}m" \\
    -e FINELOG_PORT={{ port }} \\
    -e FINELOG_REMOTE_DIR={{ remote_log_dir }} \\
    {{ auth_env }}-v {{ cache_dir }}:{{ cache_dir }} \\
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


def render_bootstrap(image: str, port: int, remote_log_dir: str, auth_policy: str = "") -> str:
    """Render the finelog bootstrap script.

    ``auth_policy`` is the ``FINELOG_AUTH_POLICY`` JSON (see
    ``deploy.config.auth_policy_json``); empty leaves the server on its private
    allow-localhost default. It is passed single-quoted, so it must not contain a
    single quote (the JSON never does — hex secrets, identifiers, CIDRs).
    """
    if not image:
        raise ValueError("image is required")
    if port <= 0:
        raise ValueError("port must be > 0")
    if "'" in auth_policy:
        raise ValueError("auth_policy must not contain a single quote")
    auth_env = f"-e FINELOG_AUTH_POLICY='{auth_policy}' " if auth_policy else ""
    return render_template(
        BOOTSTRAP_SCRIPT,
        docker_image=image,
        port=port,
        remote_log_dir=remote_log_dir,
        auth_env=auth_env,
        cache_dir=CACHE_DIR,
        container_name=CONTAINER_NAME,
    )
