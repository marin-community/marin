# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN pip install --no-cache-dir langdetect==1.0.9

ADD --checksum=sha256:3194b7a44ada0a4cd185ab3c85d883f85da8641970fef3dfc4b66d13780079fc \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/544d5d6f14116a06bde0209352585903133bd618/skyrl-gym/skyrl_gym/envs/ifeval/utils.py \
    /opt/skyrl_gym/skyrl_gym/envs/ifeval/utils.py

RUN touch /opt/skyrl_gym/skyrl_gym/__init__.py \
    /opt/skyrl_gym/skyrl_gym/envs/__init__.py \
    /opt/skyrl_gym/skyrl_gym/envs/ifeval/__init__.py

ENV PYTHONPATH=/opt/skyrl_gym
