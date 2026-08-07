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
FROM --platform=$BUILDPLATFORM node:22-slim AS dashboard

WORKDIR /build/dashboard

# Copy manifests first so dependency install is cached across source edits.
COPY lib/iris/dashboard/package.json lib/iris/dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY lib/iris/dashboard/ ./
RUN npm run build


# ── Stage: iris-native-wheel ────────────────────────────────────────
# Build the PyO3 companion once per target architecture without carrying a Rust
# toolchain into the controller or worker images.
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:0.10.3 AS iris-native-uv
FROM --platform=$BUILDPLATFORM rust:1-bookworm AS iris-native-wheel

COPY --from=iris-native-uv /uv /uvx /bin/

WORKDIR /build/iris-native

ARG BUILDARCH
ARG TARGETARCH
# Direct Docker builds use release. Iris CLI builds default to fast, which
# keeps optimized parallel codegen but skips LTO.
ARG CARGO_PROFILE=release

# Keep rustc native to the builder. The extension uses PyO3's abi3-py312 ABI,
# so a target-architecture Python interpreter is not needed for cross builds.
RUN case "${TARGETARCH}" in \
        amd64) rust_target=x86_64-unknown-linux-gnu; cross_cc=x86_64-linux-gnu-gcc; cross_package=gcc-x86-64-linux-gnu; cross_libc=libc6-dev-amd64-cross ;; \
        arm64) rust_target=aarch64-unknown-linux-gnu; cross_cc=aarch64-linux-gnu-gcc; cross_package=gcc-aarch64-linux-gnu; cross_libc=libc6-dev-arm64-cross ;; \
        *) echo "Unsupported Iris native target architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    rustup target add "${rust_target}" && \
    if [ "${BUILDARCH}" = "${TARGETARCH}" ]; then \
        ln -sf "$(command -v gcc)" "/usr/local/bin/${cross_cc}"; \
    else \
        apt-get update && \
        apt-get install -y --no-install-recommends "${cross_package}" "${cross_libc}" && \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc

# Compile third-party dependencies from manifests before copying Iris source.
# The target cache is shared by the dependency and wheel builds, while its ID
# keeps target triples and optimization profiles isolated.
COPY lib/iris/rust/Cargo.toml lib/iris/rust/Cargo.lock ./
COPY lib/iris/rust/pyext/Cargo.toml ./pyext/Cargo.toml
RUN mkdir -p src pyext/src && \
    printf 'pub fn placeholder() {}\n' > src/lib.rs && \
    printf 'pub fn placeholder() {}\n' > pyext/src/lib.rs
RUN --mount=type=cache,id=iris-native-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=iris-native-target-linux-gnu-${TARGETARCH}-${CARGO_PROFILE},target=/build/iris-native/target,sharing=locked \
    case "${TARGETARCH}" in \
        amd64) rust_target=x86_64-unknown-linux-gnu ;; \
        arm64) rust_target=aarch64-unknown-linux-gnu ;; \
    esac && \
    cargo build --locked --workspace --profile "${CARGO_PROFILE}" --target "${rust_target}"

COPY lib/iris/rust/pyproject.toml lib/iris/rust/iris_native.pyi ./
COPY lib/iris/rust/src/ ./src/
COPY lib/iris/rust/pyext/src/ ./pyext/src/
RUN --mount=type=cache,id=iris-native-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,id=iris-native-target-linux-gnu-${TARGETARCH}-${CARGO_PROFILE},target=/build/iris-native/target,sharing=locked \
    case "${TARGETARCH}" in \
        amd64) rust_target=x86_64-unknown-linux-gnu ;; \
        arm64) rust_target=aarch64-unknown-linux-gnu ;; \
    esac && \
    find src pyext/src -type f -exec touch {} + && \
    uvx --python 3.12 --from 'maturin>=1.5,<2.0' maturin build \
        --target "${rust_target}" \
        --profile "${CARGO_PROFILE}" \
        --out /wheels


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
COPY lib/iris/rust/pyproject.toml lib/iris/rust/Cargo.toml lib/iris/rust/Cargo.lock \
    lib/iris/rust/iris_native.pyi ./lib/iris/rust/
COPY lib/iris/rust/pyext/Cargo.toml ./lib/iris/rust/pyext/Cargo.toml
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml
COPY lib/fray/pyproject.toml ./lib/fray/pyproject.toml
COPY lib/haliax/pyproject.toml ./lib/haliax/pyproject.toml
COPY lib/levanter/pyproject.toml ./lib/levanter/pyproject.toml
COPY lib/marin/pyproject.toml ./lib/marin/pyproject.toml
COPY lib/zephyr/pyproject.toml ./lib/zephyr/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-iris --no-install-workspace \
        --no-install-package marin-iris-native

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
    uv sync --frozen --package marin-iris --no-install-package marin-iris-native
COPY --from=iris-native-wheel /wheels/marin_iris_native-*.whl /tmp/
RUN uv pip install --no-deps /tmp/marin_iris_native-*.whl \
    && rm /tmp/marin_iris_native-*.whl

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
RUN uv sync --frozen --package marin-iris --extra controller --inexact \
    --no-install-package marin-iris-native

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

# Cache locations (UV_CACHE_DIR, CARGO_TARGET_DIR, HF_HUB_CACHE) are injected per
# task by iris.cluster.runtime.env, which owns the matching mounts, so that they
# cover every task image rather than just this one.

# Only disable bytecode during build to keep image layers clean.
# At runtime, bytecode compilation is re-enabled so .pyc files accumulate
# in the persistent uv cache, speeding up repeated imports across pods.
ARG PYTHONDONTWRITEBYTECODE=1

ARG IRIS_GIT_HASH=unknown
ENV IRIS_GIT_HASH=${IRIS_GIT_HASH}
ARG IRIS_PROVENANCE={}
ENV IRIS_PROVENANCE=${IRIS_PROVENANCE}

# Nsight Systems CLI, extracted from NVIDIA's deb (the target binary is
# self-contained; apt-installing it pulls the whole Qt/GUI chain). Arch-aware:
# `sbsa` on Grace/aarch64, `x86_64` elsewhere. Kept in sync with the version the
# reports are opened against. Baked into the one task image so a GPU job can
# profile with `nsys` already on PATH and iris needs no notion of a GPU image.
ARG NSYS_VERSION=2026.1.3
ARG NSYS_BUILD=2026.1.3.425-1
RUN set -eux; \
    case "$(uname -m)" in \
      aarch64) nsys_arch=sbsa;   pkg_arch=arm64 ;; \
      x86_64)  nsys_arch=x86_64; pkg_arch=amd64 ;; \
      *) echo "no nsight-systems build for $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/nsys.deb \
      "https://developer.download.nvidia.com/compute/cuda/repos/debian12/${nsys_arch}/nsight-systems-${NSYS_VERSION}_${NSYS_BUILD}_${pkg_arch}.deb"; \
    dpkg-deb -x /tmp/nsys.deb /opt/nsight; \
    rm -f /tmp/nsys.deb; \
    ln -s "$(ls /opt/nsight/opt/nvidia/nsight-systems/${NSYS_VERSION}/target-linux-*/nsys | head -1)" /usr/local/bin/nsys; \
    nsys --version

WORKDIR /app
