FROM python:3.10-bookworm

# Force root user. Some base images (notably ocaml/opam, dotnet/sdk) set a
# non-root USER, which makes `mkdir /logs /testbed` fail at build time and
# `apt-get` fail at runtime. Setting USER root unconditionally is safe — if
# the image was already root, this is a no-op.
USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Common build essentials. Bake in git/curl/jq/build-essential so every
# language family shares the same apt layer (kept identical across
# Dockerfiles to maximize snapshot reuse). `--no-install-recommends`
# keeps the layer small and stable across base images. The trailing
# `|| true` makes apt failures non-fatal — some images (e.g. `swift`,
# `dart:stable`) ship with a frozen apt cache that breaks `apt-get update`.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget jq ca-certificates \
    build-essential \
    libffi-dev libssl-dev \
    locales locales-all tzdata \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* || true

RUN mkdir -p /logs /testbed /output && chmod 777 /output
WORKDIR /testbed
RUN mkdir -p /output && chmod 777 /output
RUN (pip install --no-cache-dir pytest-json-report || pip3 install --no-cache-dir pytest-json-report)
