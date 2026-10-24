FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv
FROM python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY infra/package_gateway/pyproject.toml infra/package_gateway/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY vendor/harbor/src/harbor/security ./harbor/security
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["python", "-m", "harbor.security.gateway"]
