# Build from the repository root against its committed dependency lock.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN --mount=type=bind,source=.,target=/src,readonly \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --project /src --frozen --package marin-auth --no-editable --no-dev \
        --reinstall-package marin-rigging

FROM python:3.12-slim
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY infra/auth/service.py infra/auth/loom.py infra/auth/openfga.py /app/infra/auth/
USER 1000:1000
CMD ["python", "-m", "infra.auth.service", "--config", "/etc/marin-auth/policy.yaml", "--host", "0.0.0.0", "--port", "8090"]
