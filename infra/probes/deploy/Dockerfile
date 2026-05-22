# probes daemon image — synthetic infra canary.
#
# Build context must be the marin repo root:
#   docker build -f infra/probes/deploy/Dockerfile -t probes:dev .
#
# The daemon makes outbound RPCs to Iris and Finelog. No inbound network
# surface; it exposes nothing.

# ── Stage: base ──────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="probes daemon image"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (matches the version pinned by lib/finelog/Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/


# ── Stage: deps ──────────────────────────────────────────────────────
# Install Python dependencies. Mirrors the finelog Dockerfile pattern: copy
# pyproject metadata first so source edits don't bust the dep cache.
FROM base AS deps

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workspace metadata for every member marin-probes transitively depends on.
COPY lib/rigging/pyproject.toml ./lib/rigging/pyproject.toml
COPY lib/iris/pyproject.toml ./lib/iris/pyproject.toml
COPY lib/finelog/pyproject.toml ./lib/finelog/pyproject.toml
COPY infra/probes/pyproject.toml ./infra/probes/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    printf '[tool.uv.workspace]\nmembers = ["infra/probes", "lib/iris", "lib/finelog", "lib/rigging"]\n\n[tool.uv.sources]\nmarin-probes = { workspace = true }\nmarin-iris = { workspace = true }\nmarin-finelog = { workspace = true }\nmarin-rigging = { workspace = true }\n' > pyproject.toml \
    && uv sync --package marin-probes --no-install-project

# Full source, then install the project.
COPY lib/rigging/src/ ./lib/rigging/src/
COPY lib/iris/src/ ./lib/iris/src/
COPY lib/finelog/src/ ./lib/finelog/src/
COPY infra/probes/src/ ./infra/probes/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package marin-probes


# ── Stage: runtime ───────────────────────────────────────────────────
FROM deps AS runtime

# SQLite database lives on a PD-SSD mounted at /var/lib/probes by the VM.
RUN groupadd --system --gid 1000 probes \
    && useradd --system --uid 1000 --gid probes --home-dir /app --no-create-home probes \
    && mkdir -p /var/lib/probes \
    && chown probes:probes /var/lib/probes

VOLUME ["/var/lib/probes"]

USER probes

# All flags read from MARIN_PROBES_* env vars set on the VM.
CMD ["python", "-m", "probes"]
