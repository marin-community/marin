# finelog server image — structured-log store + push/fetch RPC server.
#
# Build context must be the marin repo root:
#   docker build -f lib/finelog/deploy/Dockerfile -t finelog:dev .
#
# The server is unauthenticated by design — restrict at the network layer
# (k8s NetworkPolicy, GCP firewall). Do NOT expose it to the public internet.

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

# Full source, then install the project itself.
COPY lib/rigging/src/ ./lib/rigging/src/
COPY lib/finelog/src/ ./lib/finelog/src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package marin-finelog

# gcsfs is a runtime requirement when --remote-log-dir uses gs:// URIs, but
# is declared in the dev dependency group. Install it explicitly so the
# production image can offload to GCS.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "gcsfs>=2024.0.0"


# ── Stage: runtime ───────────────────────────────────────────────────
FROM deps AS runtime

# Local segment cache. Declared as a volume so k8s/Docker mount a PV here.
RUN mkdir -p /var/cache/finelog \
    && groupadd --system --gid 1000 finelog \
    && useradd --system --uid 1000 --gid finelog --home-dir /app finelog \
    && chown -R finelog:finelog /var/cache/finelog /app

VOLUME ["/var/cache/finelog"]

USER finelog

EXPOSE 10001

# FINELOG_REMOTE_DIR drives the same image in dev (unset → no offload) and
# prod (gs://bucket/path → archive segments to GCS).
ENV FINELOG_REMOTE_DIR=""

CMD ["sh", "-c", "exec python -m finelog.server.main --port 10001 --log-dir /var/cache/finelog --remote-log-dir \"${FINELOG_REMOTE_DIR}\""]
