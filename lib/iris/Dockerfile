# Unified Iris Dockerfile — controller, worker, and task images.
#
# Build context must be the marin repo root for all targets:
#   docker buildx build --target controller -f lib/iris/Dockerfile -t iris-controller:latest .
#   docker buildx build --target worker     -f lib/iris/Dockerfile -t iris-worker:latest .
#   docker buildx build --target task       -f lib/iris/Dockerfile -t iris-task:latest .
#   docker buildx build --target task-h100-evidence -f lib/iris/Dockerfile -t iris-task-h100-evidence:dev .
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


# ── Stage: h100-evidence-runtime ─────────────────────────────────────
# Build the frozen Python runtime separately so the final task image receives
# only an installed environment. The tile_lifetime sources arrive later in the
# reviewed source capsule; Shuttle is a runtime dependency and is installed as
# a non-editable wheel rather than copied as a repository tree.
FROM task AS h100-evidence-runtime

WORKDIR /build/runtime
ENV UV_PROJECT_ENVIRONMENT=/opt/h100-evidence-runtime

COPY uv.lock pyproject.toml ./
COPY lib/iris/pyproject.toml ./lib/iris/pyproject.toml
COPY lib/fray/pyproject.toml ./lib/fray/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml
COPY lib/dupekit/pyproject.toml ./lib/dupekit/pyproject.toml
COPY lib/ducky/pyproject.toml ./lib/ducky/pyproject.toml
COPY lib/shuttle/pyproject.toml lib/shuttle/README.md ./lib/shuttle/
COPY lib/tile_lifetime/pyproject.toml ./lib/tile_lifetime/pyproject.toml
COPY lib/haliax/pyproject.toml ./lib/haliax/pyproject.toml
COPY lib/levanter/pyproject.toml ./lib/levanter/pyproject.toml
COPY lib/marin/pyproject.toml ./lib/marin/pyproject.toml
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/zephyr/pyproject.toml ./lib/zephyr/pyproject.toml
COPY infra/pulumi/pyproject.toml ./infra/pulumi/pyproject.toml
COPY lib/shuttle/src/ ./lib/shuttle/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-tile-lifetime --extra cuda13 \
      --no-default-groups --no-editable --no-install-project


# ── Stage: task-h100-evidence
# H100 evidence image. This stays separate from the default task
# image because Nsight Compute and the CUDA compiler add more than 1 GiB. It is
# amd64-only: the reviewed evidence target is one Hopper H100, not Grace/GB200.
FROM task AS task-h100-evidence

LABEL org.opencontainers.image.description="Iris H100 evidence task image"

COPY --from=h100-evidence-runtime /opt/h100-evidence-runtime /opt/h100-evidence-runtime
ENV PATH="/opt/h100-evidence-runtime/bin:${PATH}"

COPY lib/iris/images/h100-evidence-debian12-amd64.sha256 /tmp/h100-evidence-debian12-amd64.sha256

RUN set -eux; \
    test "$(dpkg --print-architecture)" = amd64; \
    while read -r expected_sha256 expected_size package_url extra; do \
      test -z "${extra}"; \
      case "${expected_size}" in \
        ""|*[!0-9]*) echo "invalid NVIDIA package size: ${expected_size}" >&2; exit 1 ;; \
      esac; \
      test "${expected_size}" -gt 0; \
      case "${package_url}" in \
        https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/*) ;; \
        *) echo "invalid NVIDIA package URL: ${package_url}" >&2; exit 1 ;; \
      esac; \
      filename="${package_url##*/}"; \
      case "${filename}" in \
        ""|*[!A-Za-z0-9._+-]*) echo "invalid NVIDIA package filename: ${filename}" >&2; exit 1 ;; \
      esac; \
      curl -fsSL --proto '=https' --tlsv1.2 \
        -o "/tmp/${filename}" \
        "${package_url}"; \
      actual_size="$(wc -c < "/tmp/${filename}")"; \
      test "${actual_size}" -eq "${expected_size}"; \
      echo "${expected_sha256}  /tmp/${filename}" | sha256sum --check --strict; \
      dpkg-deb --extract "/tmp/${filename}" /; \
      rm -f "/tmp/${filename}"; \
    done < /tmp/h100-evidence-debian12-amd64.sha256; \
    rm -f /tmp/h100-evidence-debian12-amd64.sha256; \
    ln -sfn /usr/local/cuda-13.2 /usr/local/cuda; \
    ln -s libnvtx3interop.so /usr/local/cuda-13.2/lib64/libnvToolsExt.so; \
    ln -s /opt/nvidia/nsight-compute/2026.1.1/ncu /usr/local/bin/ncu; \
    test -x /usr/local/cuda-13.2/bin/nvcc; \
    test -x /usr/local/cuda-13.2/bin/ptxas; \
    test -x /usr/local/cuda-13.2/bin/cuobjdump; \
    test -x /usr/local/cuda-13.2/bin/nvdisasm; \
    test -x /usr/local/bin/ncu; \
    test -x /usr/local/bin/nsys; \
    /usr/local/cuda-13.2/bin/nvcc --version; \
    /usr/local/cuda-13.2/bin/ptxas --version; \
    /usr/local/cuda-13.2/bin/cuobjdump --version; \
    /usr/local/cuda-13.2/bin/nvdisasm --version; \
    /usr/local/bin/ncu --version; \
    /usr/local/bin/nsys --version

ENV CUDA_HOME=/usr/local/cuda-13.2
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV NVDISASM_PATH="${CUDA_HOME}/bin"

# Bind the pinned CLI's own option contract to the capture policy used by the
# evidence runner. This invokes help only; it does not load a driver or query a
# device. The validator is build-only and is not copied into the final image.
RUN --mount=type=bind,source=lib/iris/images/h100_evidence_nsys_help.py,target=/tmp/h100-evidence-nsys-help.py,ro \
    set -eux; \
    /usr/local/bin/nsys profile --help > /tmp/h100-evidence-nsys-profile-help.txt; \
    /opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-nsys-help.py \
      /tmp/h100-evidence-nsys-profile-help.txt; \
    rm -f /tmp/h100-evidence-nsys-profile-help.txt

# Exercise the runner's exact ctypes NVTX range under the pinned profiler.
# The module and harness are build-only bind mounts. This records no CUDA work
# and neither loads a driver nor queries a device.
RUN --mount=type=bind,source=lib/tile_lifetime/src/tile_lifetime/nvtx_range.py,target=/tmp/h100-evidence-nvtx-range.py,ro \
    --mount=type=bind,source=lib/iris/images/h100_evidence_nvtx_smoke.py,target=/tmp/h100-evidence-nvtx-smoke.py,ro \
    set -eux; \
    env NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
      /usr/local/bin/nsys profile --force-overwrite=true --trace=nvtx \
      --capture-range=nvtx --capture-range-end=stop \
      --nvtx-capture=h100-evidence-nvtx-smoke \
      --output /tmp/h100-evidence-nvtx \
      /opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-nvtx-smoke.py emit \
        --module /tmp/h100-evidence-nvtx-range.py \
        --library "${CUDA_HOME}/lib64/libnvToolsExt.so" \
        --output /tmp/h100-evidence-nvtx-result.json \
        --name h100-evidence-nvtx-smoke; \
    test -s /tmp/h100-evidence-nvtx.nsys-rep; \
    /usr/local/bin/nsys export --force-overwrite=true --type=sqlite \
      --output /tmp/h100-evidence-nvtx.sqlite /tmp/h100-evidence-nvtx.nsys-rep; \
    /opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-nvtx-smoke.py validate \
      --result /tmp/h100-evidence-nvtx-result.json \
      --sqlite /tmp/h100-evidence-nvtx.sqlite \
      --name h100-evidence-nvtx-smoke; \
    rm -f /tmp/h100-evidence-nvtx-result.json /tmp/h100-evidence-nvtx.nsys-rep \
      /tmp/h100-evidence-nvtx.sqlite

# Exercise the runner's exact compiler/disassembler dependency without loading
# a driver or querying a device. cuobjdump delegates SASS decoding to nvdisasm.
RUN --mount=type=bind,source=lib/iris/images/h100_evidence_sass_smoke.py,target=/tmp/h100-evidence-sass-smoke.py,ro \
    set -eux; \
    printf '%s\n' 'extern "C" __global__ void h100_evidence_smoke(int *output) { if (threadIdx.x == 0) *output = 1; }' \
      > /tmp/h100-evidence-smoke.cu; \
    "${CUDA_HOME}/bin/nvcc" -arch=sm_90a --cubin \
      /tmp/h100-evidence-smoke.cu -o /tmp/h100-evidence-smoke.cubin; \
    env PATH=/usr/bin:/bin NVDISASM_PATH="${CUDA_HOME}/bin" \
      "${CUDA_HOME}/bin/cuobjdump" --dump-sass /tmp/h100-evidence-smoke.cubin \
      > /tmp/h100-evidence-cuobjdump.sass; \
    "${CUDA_HOME}/bin/nvdisasm" /tmp/h100-evidence-smoke.cubin > /tmp/h100-evidence-nvdisasm.sass; \
    /opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-sass-smoke.py \
      --format cuobjdump --expected-kernel h100_evidence_smoke /tmp/h100-evidence-cuobjdump.sass; \
    /opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-sass-smoke.py \
      --format nvdisasm --expected-kernel h100_evidence_smoke /tmp/h100-evidence-nvdisasm.sass; \
    rm -f /tmp/h100-evidence-smoke.cu /tmp/h100-evidence-smoke.cubin \
      /tmp/h100-evidence-cuobjdump.sass /tmp/h100-evidence-nvdisasm.sass

# The runner loads these libraries from the selected NVCC toolkit before it
# records any evidence. Resolve and load every required symbol without calling
# a CUDA device API.
RUN /opt/h100-evidence-runtime/bin/python - <<'PY'
from ctypes import CDLL
from pathlib import Path

probes = {
    "cudart": "cudaFree",
    "nvToolsExt": "nvtxRangePushA",
}
for library, symbol in probes.items():
    handle = CDLL(str(Path("/usr/local/cuda-13.2/lib64") / f"lib{library}.so"))
    getattr(handle, symbol)
PY

# Import and version validation is forced onto the CPU backend. It proves the
# installed runtime is internally coherent without enumerating or initializing
# a GPU during the image build.
RUN JAX_PLATFORMS=cpu /opt/h100-evidence-runtime/bin/python - <<'PY'
from importlib.metadata import version

import jax
import jaxlib
import numpy
import scipy
from shuttle.ir import DType

expected = {
    "jax": "0.10.1",
    "jax-cuda13-pjrt": "0.10.1",
    "jax-cuda13-plugin": "0.10.1",
    "jaxlib": "0.10.1",
    "marin-shuttle": "0.1.0",
}
actual = {distribution: version(distribution) for distribution in expected}
if actual != expected:
    raise SystemExit(f"H100 evidence runtime version mismatch: {actual!r}")

if jax.__version__ != expected["jax"] or jaxlib.__version__ != expected["jaxlib"]:
    raise SystemExit("imported JAX versions disagree with installed distribution metadata")
if not numpy.__version__ or not scipy.__version__ or DType.BF16.value != "bf16":
    raise SystemExit("H100 evidence runtime import smoke failed")
PY
