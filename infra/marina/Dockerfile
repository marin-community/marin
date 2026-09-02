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
COPY lib/finestore /src/lib/finestore
COPY infra/marina /src/infra/marina
RUN pip install --no-cache-dir /src/lib/rigging /src/lib/finestore /src/infra/marina && rm -rf /src
WORKDIR /app
COPY --from=web /build/apps /app/apps
# EvalDash reaches past the kernel's own dependencies for three import-light things: the record
# contract and statistics engine the eval runners share, and the generated Connect clients its live
# job and log panels use. They are copied as modules rather than installed, because installing
# marin-core or marin-iris would pull JAX and the whole training stack into this image.
COPY lib/marin/src/marin/__init__.py /app/vendor/marin/__init__.py
COPY lib/marin/src/marin/evaluation/__init__.py /app/vendor/marin/evaluation/__init__.py
COPY lib/marin/src/marin/evaluation/archive.py /app/vendor/marin/evaluation/archive.py
COPY lib/marin/src/marin/evaluation/records.py /app/vendor/marin/evaluation/records.py
COPY lib/marin/src/marin/evaluation/eval_stats.py /app/vendor/marin/evaluation/eval_stats.py
COPY lib/marin/src/marin/evaluation/eval_measurements.py /app/vendor/marin/evaluation/eval_measurements.py
COPY lib/iris/src/iris/__init__.py /app/vendor/iris/__init__.py
COPY lib/iris/src/iris/rpc/ /app/vendor/iris/rpc/
COPY lib/finelog/src/finelog/__init__.py /app/vendor/finelog/__init__.py
COPY lib/finelog/src/finelog/rpc/ /app/vendor/finelog/rpc/
ENV MARINA_APPS_DIR=/app/apps
# Each app package is importable by its own name, the way the kernel imports it, so a
# job can run one of an app's modules: `python -m echo.sync.main` is Echo's sync job.
ENV PYTHONPATH=/app/apps:/app/vendor
# Echo's two ONNX models are baked in rather than fetched on demand: Cloud Run keeps no
# disk between instances, so every cold start would otherwise download them before it
# could answer the first search. FASTEMBED_CACHE_PATH keeps them out of /tmp, which is
# memory on Cloud Run.
ENV FASTEMBED_CACHE_PATH=/app/models
RUN python -c "from echo import models; loaded = models.SearchModels(); loaded.embedding(); loaded.reranker()"
CMD ["marina", "serve"]
