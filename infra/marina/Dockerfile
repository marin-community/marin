# Build context is the repository root (the image needs lib/rigging); the allowlist in
# Dockerfile.dockerignore keeps everything else out of the context.
FROM node:22-slim AS web
WORKDIR /build
COPY infra/marina/web web
COPY infra/marina/apps apps
# Every app with a web/ directory is built in place so its dist/ lands beside app.toml.
RUN set -e; for dir in apps/*/web; do (cd "$dir" && npm ci && npm run build); done \
    && rm -rf apps/*/web/node_modules

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1
COPY lib/rigging /src/lib/rigging
COPY infra/marina /src/infra/marina
RUN pip install --no-cache-dir /src/lib/rigging /src/infra/marina && rm -rf /src
WORKDIR /app
COPY --from=web /build/apps /app/apps
ENV MARINA_APPS_DIR=/app/apps
CMD ["marina", "serve"]
