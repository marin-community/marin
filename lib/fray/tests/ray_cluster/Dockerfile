# Ray node image for the fray Docker test lane: the stock Ray image plus uv and a
# compiler, with writable mount points for the marin venv, the uv cache and the
# scratch directory shared by every node. The venv itself is synced at head start
# from the bind-mounted checkout (see head.sh), so the image never embeds the repo.
ARG RAY_VERSION=2.58.0
FROM rayproject/ray:${RAY_VERSION}-py312
ARG RAY_VERSION

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/
RUN mkdir -p /opt/venv /opt/uv-cache /shared && chown -R ray:users /opt/venv /opt/uv-cache /shared
USER ray

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON=/home/ray/anaconda3/bin/python \
    UV_LINK_MODE=copy \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_VERSION=${RAY_VERSION}
