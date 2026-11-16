# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab
ENV PYTHONNOUSERSITE=1
WORKDIR /
COPY requirements.lock /opt/multiswe/requirements.lock
RUN uv pip sync --system --no-cache --require-hashes /opt/multiswe/requirements.lock
