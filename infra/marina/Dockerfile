# Build context is the repository root; application packages are released separately.
FROM python:3.12-slim AS build
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
# The workspace lives at /app: rigging resolves the repo-root config/ directory by walking up
# from the working directory to the workspace pyproject, so the process runs from here.
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
# Install against the committed workspace lock so the image runs the same pinned set as CI.
# uv needs every member's pyproject.toml to map the lock; only marina's closure is installed.
# External dependencies first (cached across source edits), then the workspace members.
COPY uv.lock pyproject.toml ./
COPY lib/iris/pyproject.toml lib/iris/
COPY lib/rigging/pyproject.toml lib/rigging/
COPY lib/finestore/pyproject.toml lib/finestore/
COPY lib/finelog/pyproject.toml lib/finelog/
COPY lib/shellbox/pyproject.toml lib/shellbox/
COPY lib/taskcompendium/pyproject.toml lib/taskcompendium/
COPY lib/evalrecords/pyproject.toml lib/evalrecords/
COPY lib/verifyit/pyproject.toml lib/verifyit/
COPY infra/marina/pyproject.toml infra/marina/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-marina --no-install-workspace
# rigging's wheel force-includes config/ and finelog's its own config/; shellbox and verifyit
# package their README.
COPY config config
COPY lib/shellbox/README.md lib/shellbox/
COPY lib/verifyit/README.md lib/verifyit/
COPY lib/rigging/src lib/rigging/src
COPY lib/finestore/src lib/finestore/src
COPY lib/finelog/src lib/finelog/src
COPY lib/finelog/config lib/finelog/config
COPY lib/iris/hatch_build.py lib/iris/
COPY lib/iris/src lib/iris/src
COPY lib/iris/config lib/iris/config
COPY lib/shellbox/src lib/shellbox/src
COPY lib/taskcompendium/src lib/taskcompendium/src
COPY lib/evalrecords/src lib/evalrecords/src
COPY lib/verifyit/src lib/verifyit/src
COPY infra/marina/src infra/marina/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-marina
COPY infra/marina/runtime.toml infra/marina/runtime.toml
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -m marina.model_cache

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models
WORKDIR /app
COPY --from=build /bin/uv /bin/uv
COPY --from=build /app /app
CMD ["marina", "serve"]
