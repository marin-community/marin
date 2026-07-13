# Unified Iris Dockerfile — controller, worker, and task images.
#
# Build context must be the marin repo root for all targets:
#   docker buildx build --target controller -f lib/iris/Dockerfile -t iris-controller:latest .
#   docker buildx build --target worker     -f lib/iris/Dockerfile -t iris-worker:latest .
#   docker buildx build --target task       -f lib/iris/Dockerfile -t iris-task:latest .
#
# The controller and worker share all layers up to and including Python
# dependencies. The worker adds only the Docker CLI on top.

# ── Stage: dashboard ─────────────────────────────────────────────────
# Build the Vue SPA. Output is consumed by the deps stage at
# /app/lib/iris/dashboard/dist; the controller/worker ASGI apps serve it
# (see lib/iris/src/iris/cluster/dashboard_common.py).
FROM node:22-slim AS dashboard

WORKDIR /build/dashboard

# Copy manifests first so dependency install is cached across source edits.
COPY lib/iris/dashboard/package.json lib/iris/dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY lib/iris/dashboard/ ./
RUN npm run build


# ── Stage: base ──────────────────────────────────────────────────────
# System dependencies shared by controller and worker.
FROM python:3.12-slim AS base

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"

ARG KUBECTL_VERSION=v1.32.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    apt-transport-https \
    openssh-client \
    git \
    lldb \
    && rm -rf /var/lib/apt/lists/*

# Google Cloud SDK (controller needs it for TPU management, worker for discovery)
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y --no-install-recommends \
    google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# kubectl — TARGETARCH is auto-set by buildx (amd64 on x86 nodes, arm64 on Grace).
ARG TARGETARCH
RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
        -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl \
    && kubectl version --client

# uv
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/


# ── Stage: deps ──────────────────────────────────────────────────────
# Python dependencies — shared between controller and worker.
FROM base AS deps

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Build against the committed root `uv.lock` (`--frozen`) so the image ships the
# same pinned, tested dependency set as local dev and the rest of CI — no fresh
# PyPI resolve that can float a transitive (e.g. protobuf) to an untested major.
# marin-iris's closure is two workspace members built from source — marin-rigging
# and marin-finelog (the pure-Python finelog client) — plus external/registry
# deps (the native marin-finelog-server ships as a published wheel). It never
# pulls the Rust-built members (marin-core, levanter, …), so `--package
# marin-iris` never asks the toolchain-free slim base to build them.
# uv must still discover every workspace member to map the lock, so copy each
# member's pyproject.toml (metadata only) — but only iris+rigging+finelog install.
# The first sync uses `--no-install-workspace` to install just the external deps
# (cached across source edits); the second builds iris+rigging+finelog from source.
# Metadata alone suffices even for marin-haliax's `dynamic = ["version"]`
# (sourced from src/haliax/__about__.py): under --frozen, uv takes the version
# from the lock and never invokes a non-installed member's build backend.
COPY uv.lock pyproject.toml ./
COPY lib/iris/pyproject.toml ./lib/iris/pyproject.toml
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml
COPY lib/fray/pyproject.toml ./lib/fray/pyproject.toml
COPY lib/haliax/pyproject.toml ./lib/haliax/pyproject.toml
COPY lib/levanter/pyproject.toml ./lib/levanter/pyproject.toml
COPY lib/marin/pyproject.toml ./lib/marin/pyproject.toml
COPY lib/zephyr/pyproject.toml ./lib/zephyr/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-iris --no-install-workspace

# Now copy full source for rigging, finelog, and iris, then install. rigging's
# wheel force-includes the repo-root config/ cluster YAMLs (-> rigging/clusters)
# and finelog's force-includes lib/finelog/config/, so copy both: the editable
# builds read them.
COPY lib/rigging/src/ ./lib/rigging/src/
COPY config/ ./config/
COPY lib/finelog/src/ ./lib/finelog/src/
COPY lib/finelog/config/ ./lib/finelog/config/
COPY lib/iris/src/ ./lib/iris/src/
COPY lib/iris/config/ ./lib/iris/config/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-iris

# Profiling tools — installed after uv sync so they aren't pruned by the exact sync.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install py-spy memray

# Dashboard assets — built by the `dashboard` stage above.
COPY --from=dashboard /build/dashboard/dist ./lib/iris/dashboard/dist

RUN mkdir -p /var/cache/iris

ARG IRIS_GIT_HASH=unknown
ENV IRIS_GIT_HASH=${IRIS_GIT_HASH}
ARG IRIS_PROVENANCE={}
ENV IRIS_PROVENANCE=${IRIS_PROVENANCE}


# ── Stage: docker-cli ────────────────────────────────────────────────
# Built from base (not deps) so it stays cached across source changes.
FROM base AS docker-cli

# Docker CLI and buildx (buildx required for BuildKit).
# TODO(#3341): TPU host VMs ship Docker daemon 24.0 (API 1.43). Pinning the
# CLI to the same major keeps client and daemon compatible. Revisit once GCP
# updates the base TPU VM image.
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends 'docker-ce-cli=5:24.0*' 'docker-buildx-plugin=0.11*' && \
    rm -rf /var/lib/apt/lists/*


# ── Stage: controller ────────────────────────────────────────────────
FROM deps AS controller

# Controller-only deps: kubernetes (marin-iris's `controller` extra).
# --inexact keeps the py-spy/memray installed above from being pruned.
RUN uv sync --frozen --package marin-iris --extra controller --inexact

LABEL org.opencontainers.image.description="Iris controller image"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

CMD [".venv/bin/python", "-m", "iris.cluster.controller.main", "serve", "--host", "0.0.0.0", "--port", "10000"]


# ── Stage: worker ────────────────────────────────────────────────────
# Only adds Docker CLI + buildx on top of the shared deps layer.
FROM deps AS worker

LABEL org.opencontainers.image.description="Iris worker image"

# Copy pre-built Docker CLI + buildx from the docker-cli stage (cached
# independently of Python source changes).
COPY --from=docker-cli /usr/bin/docker /usr/bin/docker
COPY --from=docker-cli /usr/libexec/docker/cli-plugins/ /usr/libexec/docker/cli-plugins/

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:10001/health || exit 1

CMD [".venv/bin/python", "-m", "iris.cluster.worker.main", "serve", "--host", "0.0.0.0", "--port", "10001"]


# ── Stage: task ──────────────────────────────────────────────────────
# Base task image for Iris jobs. No application source — bundle is mounted at
# runtime via bind mount to /app. Build context must be the marin repo root.
FROM python:3.12-slim AS task

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="Iris task base image"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
    openssh-client \
    git \
    build-essential \
    ffmpeg \
    lldb \
    libibverbs1 \
    ibverbs-providers \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

ENV UV_CACHE_DIR=/uv/cache
ENV CARGO_TARGET_DIR=/root/.cargo/target
# Only disable bytecode during build to keep image layers clean.
# At runtime, bytecode compilation is re-enabled so .pyc files accumulate
# in the persistent uv cache, speeding up repeated imports across pods.
ARG PYTHONDONTWRITEBYTECODE=1

ARG IRIS_GIT_HASH=unknown
ENV IRIS_GIT_HASH=${IRIS_GIT_HASH}
ARG IRIS_PROVENANCE={}
ENV IRIS_PROVENANCE=${IRIS_PROVENANCE}

WORKDIR /app
