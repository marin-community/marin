# finelog server image — structured-log store + push/fetch RPC server.
#
# Build context must be the marin repo root:
#   docker build -f lib/finelog/deploy/Dockerfile -t finelog:dev .
#
# The server is unauthenticated by design — restrict at the network layer
# (k8s NetworkPolicy, GCP firewall). Do NOT expose it to the public internet.

# ── Stage: dashboard ─────────────────────────────────────────────────
# Build the Vue SPA. Output is consumed by the runtime stage at
# /app/dashboard/dist; the finelog ASGI app serves it on the same port as
# the RPCs (see lib/finelog/src/finelog/server/asgi.py).
FROM node:22-slim AS dashboard

WORKDIR /build/dashboard

# Copy manifests first so dependency install is cached across source edits.
COPY lib/finelog/dashboard/package.json lib/finelog/dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY lib/finelog/dashboard/ ./
RUN npm run build


# ── Stage: base ──────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="finelog log server image"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (matches the version pinned by lib/iris/Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/


# ── Stage: deps ──────────────────────────────────────────────────────
# Install Python dependencies. Mirrors the iris Dockerfile pattern: copy
# pyproject metadata first so source edits don't bust the dep cache.
FROM base AS deps

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# rigging is a workspace-local dep not on PyPI.
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    printf '[tool.uv.workspace]\nmembers = ["lib/finelog", "lib/rigging"]\n\n[tool.uv.sources]\nmarin-rigging = { workspace = true }\n' > pyproject.toml \
    && uv sync --package marin-finelog --no-install-project

# Full source, then install the project itself. `lib/finelog/config/` is
# force-included by hatchling's wheel build (see lib/finelog/pyproject.toml);
# copy it before the project install so non-editable wheel builds succeed.
COPY lib/rigging/src/ ./lib/rigging/src/
COPY lib/finelog/src/ ./lib/finelog/src/
COPY lib/finelog/config/ ./lib/finelog/config/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package marin-finelog

# gcsfs is a runtime requirement when --remote-log-dir uses gs:// URIs, but
# is declared in the dev dependency group. Install it explicitly so the
# production image can offload to GCS.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "gcsfs>=2024.0.0"


# ── Stage: runtime ───────────────────────────────────────────────────
FROM deps AS runtime

# Pre-built dashboard SPA. Served by the asgi app at "/" and "/static/*".
COPY --from=dashboard /build/dashboard/dist ./dashboard/dist

# Local segment cache. Declared as a volume so k8s/Docker mount a PV here.
# Only the cache dir needs to be writable by the finelog user; /app and
# /app/.venv stay root-owned but are world-readable, so the recursive chown
# (which walks tens of thousands of venv files and re-layers every inode)
# is unnecessary.
RUN groupadd --system --gid 1000 finelog \
    && useradd --system --uid 1000 --gid finelog --home-dir /app --no-create-home finelog \
    && mkdir -p /var/cache/finelog \
    && chown finelog:finelog /var/cache/finelog

VOLUME ["/var/cache/finelog"]

USER finelog

EXPOSE 10001

# All server flags read from FINELOG_* envvars (see finelog/server/main.py).
# Defaults: FINELOG_PORT=10001, FINELOG_LOG_DIR=/var/cache/finelog,
# FINELOG_REMOTE_DIR="" (no offload). Override per deployment via env.
CMD ["python", "-m", "finelog.server.main"]
