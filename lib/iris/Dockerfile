# Unified Iris Dockerfile — controller, worker, and task images.
#
# Build context must be the marin repo root for all targets:
#   docker buildx build --target controller -f lib/iris/Dockerfile -t iris-controller:latest .
#   docker buildx build --target worker     -f lib/iris/Dockerfile -t iris-worker:latest .
#   docker buildx build --target task       -f lib/iris/Dockerfile -t iris-task:latest .
#
# The controller and worker share all layers up to and including Python
# dependencies. The worker adds only the Docker CLI on top.

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

# kubectl
RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
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

# Rigging and finelog are workspace-local deps not on PyPI with compatible APIs.
# Copy only metadata first so source changes don't bust the dep cache.
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml
COPY lib/iris/pyproject.toml ./lib/iris/
RUN --mount=type=cache,target=/root/.cache/uv \
    printf '[tool.uv.workspace]\nmembers = ["lib/iris", "lib/rigging", "lib/finelog"]\n\n[tool.uv.sources]\nmarin-rigging = { workspace = true }\nmarin-finelog = { workspace = true }\n' > pyproject.toml \
    && uv sync --package marin-iris --no-install-project

# Now copy full source for rigging, finelog, and iris, then install.
COPY lib/rigging/src/ ./lib/rigging/src/
COPY lib/finelog/src/ ./lib/finelog/src/
COPY lib/finelog/config/ ./lib/finelog/config/
COPY lib/iris/src/ ./lib/iris/src/
COPY lib/iris/examples/ ./lib/iris/examples/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package marin-iris

# Profiling tools — installed after uv sync so they aren't pruned by the exact sync.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install py-spy memray

# Dashboard assets (pre-built by `iris build dashboard` before docker build)
COPY lib/iris/dashboard/dist ./lib/iris/dashboard/dist

RUN mkdir -p /var/cache/iris

ARG IRIS_GIT_HASH=unknown
ENV IRIS_GIT_HASH=${IRIS_GIT_HASH}


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

# Controller-only deps: duckdb, pyarrow, zstandard.
RUN uv sync --extra controller --inexact

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

WORKDIR /app
