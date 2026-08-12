# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Eval-results dashboard: a Vue SPA bundled by rsbuild, served by a Starlette app.
#
# IMPORTANT: the build context is the REPO ROOT, not this directory — the runtime stage
# copies import-light Marin record contracts and the generated Iris/finelog RPC packages. All COPY sources are
# therefore repo-root-relative. The repo-root .dockerignore applies (it excludes .git,
# node_modules, .venv, caches), so the context stays reasonable despite being the whole repo.
# The Pulumi entry point passes build_context=<repo root> and dockerfile=infra/evaldash/Dockerfile.

# --- Stage 1: build the dashboard ------------------------------------------------------
FROM node:22-slim AS dashboard

WORKDIR /build
# Install against the lockfile first so the dep layer caches across source-only changes.
COPY infra/evaldash/dashboard/package.json infra/evaldash/dashboard/package-lock.json ./
RUN npm ci
# node_modules is dockerignored, so this brings only sources and config.
COPY infra/evaldash/dashboard/ ./
RUN npm run build

# --- Stage 2: the server ---------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# marin-rigging (installed from the repo checkout) supplies fsspec/gcsfs/s3fs/starlette and
# owns the S3-compatible endpoint setup (rigging.filesystem.s3_compat) used for the CW records
# prefix; the rest is the server + CloudSQL stack. rigging's wheel force-includes the repo-root
# config/ cluster YAMLs at ../../config, so the build keeps the checkout's relative layout.
COPY lib/rigging /build/lib/rigging
COPY lib/finestore /build/lib/finestore
COPY lib/evalstore /build/lib/evalstore
COPY config /build/config
RUN pip install --no-cache-dir \
    /build/lib/rigging \
    /build/lib/finestore \
    /build/lib/evalstore \
    uvicorn \
    sqlalchemy \
    pg8000 \
    cloud-sql-python-connector \
    google-cloud-secret-manager \
    google-cloud-compute \
    connect-python \
    protobuf \
    pyarrow \
    pydantic \
    anthropic \
    && rm -rf /build

# The record contract, the statistics engine, and its record adapter are the only bespoke Marin
# modules in the image; the per-sample contract lives in evalstore.archive, installed above, and the
# private Postgres index lives with the server under infra/evaldash/src. The image does not install
# marin-core.
# tests/evaluation/test_evaldash_image.py walks the server's imports and fails if one is missing here.
COPY lib/marin/src/marin/__init__.py /app/marin/__init__.py
COPY lib/marin/src/marin/evaluation/__init__.py /app/marin/evaluation/__init__.py
COPY lib/marin/src/marin/evaluation/records.py /app/marin/evaluation/records.py
COPY lib/marin/src/marin/evaluation/eval_stats.py /app/marin/evaluation/eval_stats.py
COPY lib/marin/src/marin/evaluation/eval_measurements.py /app/marin/evaluation/eval_measurements.py
# Copy the generated Connect clients and protobufs as package directories; evaldash does not need
# the full Iris or finelog runtime packages.
COPY lib/iris/src/iris/__init__.py /app/iris/__init__.py
COPY lib/iris/src/iris/rpc/ /app/iris/rpc/
COPY lib/finelog/src/finelog/__init__.py /app/finelog/__init__.py
COPY lib/finelog/src/finelog/rpc/ /app/finelog/rpc/

# Preserve the repository package path so the image and tests import the server identically.
COPY infra/evaldash/src/ /app/infra/evaldash/src/
COPY --from=dashboard /build/dist /app/infra/evaldash/dashboard/dist

# Cloud Run injects PORT and routes to it; the server reads it (default 8080).
ENV PORT=8080
EXPOSE 8080

# Connect to Postgres, start the object-store ingestor, and serve the app on $PORT.
CMD ["python", "-m", "infra.evaldash.src.server"]
