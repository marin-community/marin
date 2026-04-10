# Task image for the swe_rebench_trace experiment.
#
# Extends the standard iris-task image with the tooling needed to pull,
# extract, and run untrusted Docker images under a gVisor sandbox without
# a nested Docker daemon:
#
#   - runsc:    gVisor OCI runtime, invoked directly by the map function
#   - skopeo:   pulls images from Docker registries into local OCI layout
#   - umoci:    unpacks an OCI image to a runnable rootfs + config.json
#
# The allowlisted egress proxy is implemented in pure Python in proxy.py
# and runs in-process inside the Zephyr worker — no system proxy package
# required.
#
# Build context must be the marin repo root:
#   docker buildx build \
#     --target task-swetrace \
#     -f experiments/swe_rebench_trace/Dockerfile \
#     -t iris-task-swetrace:dev .
#
# This image is intended to be passed to ZephyrContext via
#   ResourceConfig(image="<registry>/iris-task-swetrace:<tag>")
# which routes through Iris's per-task task_image override.

# Pin runsc to a known-good point release. gVisor serves point releases at
# `release/${yyyymmdd}.${rc}/${ARCH}`.
ARG RUNSC_VERSION=20240916.0

# Use the existing iris-task base so we inherit uv, python 3.12, rust, node,
# and the standard Marin tooling. The base lives at lib/iris/Dockerfile.
FROM ghcr.io/marin-community/iris-task:latest AS task-swetrace

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="Iris task image for SWE-rebench tracing under gVisor"

ARG RUNSC_VERSION

# Install runsc directly from the Google Storage release bucket. We do not
# install the gVisor docker integration (no /etc/docker/daemon.json edits)
# because the experiment invokes runsc as a standalone OCI runtime, not via
# a docker daemon.
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64) runsc_arch="x86_64" ;; \
      aarch64) runsc_arch="aarch64" ;; \
      *) echo "unsupported arch: $arch"; exit 1 ;; \
    esac; \
    base="https://storage.googleapis.com/gvisor/releases/release/${RUNSC_VERSION}/${runsc_arch}"; \
    curl -fsSL "${base}/runsc" -o /usr/local/bin/runsc; \
    curl -fsSL "${base}/runsc.sha512" -o /tmp/runsc.sha512; \
    (cd /usr/local/bin && sha512sum -c /tmp/runsc.sha512); \
    chmod 0755 /usr/local/bin/runsc; \
    rm /tmp/runsc.sha512; \
    runsc --version

# skopeo + umoci from Debian.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        skopeo \
        umoci \
    && rm -rf /var/lib/apt/lists/*
