# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Build context is the repository root; infra/ops/__main__.py pins this Dockerfile.
FROM node:22-slim AS dashboard
WORKDIR /build
COPY infra/ops/dashboard/package.json infra/ops/dashboard/package-lock.json ./
RUN npm ci
COPY infra/ops/dashboard/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY infra/ops/pyproject.toml /build/ops/pyproject.toml
COPY infra/ops/src /build/ops/src
RUN pip install --no-cache-dir /build/ops && rm -rf /build
COPY infra/ops/migrations /app/migrations
COPY infra/ops/entrypoint.sh /app/entrypoint.sh
COPY --from=dashboard /build/dist /app/dashboard
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
