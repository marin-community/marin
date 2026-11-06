# A disposable application testbed, independent of production deployment images.
# Supply native binaries and the Iris wheel through named BuildKit contexts.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN --mount=type=bind,source=.,target=/src,readonly \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /src --frozen --package marin-auth --package marin-marina \
        --package marin-iris --extra controller \
        --no-editable --no-dev --reinstall-package marin-rigging
COPY --from=proof-wheels / /wheels/
RUN uv pip install --python /app/.venv/bin/python --no-deps --reinstall /wheels/*.whl

FROM python:3.12-slim
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
RUN apt-get update && apt-get install -y --no-install-recommends git nodejs postgresql-17 postgresql-17-pgvector \
    && useradd --uid 1000 --create-home proof \
    && rm -rf /var/lib/apt/lists/*
COPY --from=proof-tools /bao /openfga /loom /tapestry /finelog-server /usr/local/bin/
COPY --from=loom-source /crates/loom/tests/fixtures/fake-acp-agent.mjs /opt/loom-source/crates/loom/tests/fixtures/fake-acp-agent.mjs
COPY infra/auth/*.py /app/infra/auth/
COPY infra/grafana/src /app/grafana/
ENV PYTHONPATH=/app/grafana
COPY infra/auth/policy /app/infra/auth/policy
USER 1000:1000
CMD ["python", "-m", "infra.auth.prove", "--bao", "/usr/local/bin/bao", "--openfga", "/usr/local/bin/openfga", "--loom", "/usr/local/bin/loom", "--loom-source", "/opt/loom-source", "--postgres-bin", "/usr/lib/postgresql/17/bin", "--finelog", "/usr/local/bin/finelog-server"]
