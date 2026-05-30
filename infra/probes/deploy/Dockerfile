# probes daemon image — synthetic infra canary.
#
# Build context is THIS directory's parent (infra/probes/):
#   docker build -f deploy/Dockerfile -t probes:dev infra/probes
#
# marin-iris, marin-finelog, marin-rigging come from the per-package rolling
# GitHub releases (see pyproject.toml [tool.uv] find-links). No marin source
# is required in the build context.

FROM python:3.11-slim AS base

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="probes daemon image"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /uvx /bin/


# ── deps stage ───────────────────────────────────────────────────────
# Install dependencies first so source edits don't bust the dep cache.
FROM base AS deps

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Project source, then install the project itself.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ── runtime stage ────────────────────────────────────────────────────
FROM deps AS runtime

# SQLite database lives on a PD-SSD mounted at /var/lib/probes by the VM.
RUN groupadd --system --gid 1000 infra-probes \
    && useradd --system --uid 1000 --gid infra-probes --home-dir /app --no-create-home infra-probes \
    && mkdir -p /var/lib/probes \
    && chown infra-probes:infra-probes /var/lib/probes

VOLUME ["/var/lib/probes"]

USER infra-probes

CMD ["python", "-m", "marin_infra_probes"]
