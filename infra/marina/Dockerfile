# Build context is the repository root; application packages are released separately.
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /bin/uv
# The workspace lives at /app: rigging resolves the repo-root config/ directory by walking up
# from the working directory to the workspace pyproject, so the process runs from here.
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
# Install against the committed workspace lock so the image runs the same pinned set as CI.
# uv needs every member's pyproject.toml to map the lock; only marina's closure is installed.
# External dependencies first (cached across source edits), then the workspace members.
COPY uv.lock pyproject.toml ./
COPY lib/fray/pyproject.toml lib/fray/
COPY lib/haliax/pyproject.toml lib/haliax/
COPY lib/levanter/pyproject.toml lib/levanter/
COPY lib/marin/pyproject.toml lib/marin/
COPY lib/zephyr/pyproject.toml lib/zephyr/
COPY lib/iris/pyproject.toml lib/iris/
COPY lib/rigging/pyproject.toml lib/rigging/
COPY lib/finestore/pyproject.toml lib/finestore/
COPY lib/finelog/pyproject.toml lib/finelog/
COPY infra/marina/pyproject.toml infra/marina/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-marina --no-install-workspace
# rigging's wheel force-includes config/ and finelog's its own config/; haliax reads its
# version from src; levanter, haliax, and zephyr package their README; marin-core its LICENSE.
COPY LICENSE ./
COPY config config
COPY lib/levanter/README.md lib/levanter/
COPY lib/haliax/README.md lib/haliax/
COPY lib/zephyr/README.md lib/zephyr/
COPY lib/rigging/src lib/rigging/src
COPY lib/finestore/src lib/finestore/src
COPY lib/finelog/src lib/finelog/src
COPY lib/finelog/config lib/finelog/config
COPY lib/iris/hatch_build.py lib/iris/
COPY lib/iris/src lib/iris/src
COPY lib/iris/config lib/iris/config
COPY lib/fray/src lib/fray/src
COPY lib/haliax/src lib/haliax/src
COPY lib/levanter/src lib/levanter/src
COPY lib/marin/src lib/marin/src
COPY lib/zephyr/src lib/zephyr/src
COPY infra/marina/src infra/marina/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --package marin-marina
COPY infra/marina/runtime.toml infra/marina/runtime.toml
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -m marina.model_cache
CMD ["marina", "serve"]
