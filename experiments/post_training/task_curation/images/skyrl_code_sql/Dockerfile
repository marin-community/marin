# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN pip install --no-cache-dir numpy==2.3.5 pandas==2.2.2

ADD --checksum=sha256:0426a2b81ec67fe519027ac6245a64ee3edda15a5929d51dc75b2bf42e7b0c93 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/544d5d6f14116a06bde0209352585903133bd618/skyrl-gym/skyrl_gym/envs/lcb/livecodebench.py \
    /opt/skyrl_gym/skyrl_gym/envs/lcb/livecodebench.py

ADD --checksum=sha256:04b5b15dc7b9bb297d93a28ce7ae2ff03072df177a004137e8a7fda0d39c70ac \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/544d5d6f14116a06bde0209352585903133bd618/skyrl-gym/skyrl_gym/envs/text_to_sql/scoring.py \
    /opt/skyrl_gym/skyrl_gym/envs/text_to_sql/scoring.py

ENV PYTHONPATH=/opt/skyrl_gym

WORKDIR /app
