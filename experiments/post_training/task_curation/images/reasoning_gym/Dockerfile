# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Preserve the pinned Python 3.11 packages used to regenerate source records.
FROM ghcr.io/marin-community/iris-task@sha256:aa369fcb572bbd33e18ff31d9e7f618b3a76dd6ca7a834a42c49eb45d42fa727

ADD --checksum=sha256:af348f9e5d92c45767d0e7bd4a55eec642acc303efd588ed344d59648af262c2 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/aime/utils.py \
    /opt/skyrl_gym/skyrl_gym/envs/aime/utils.py

ADD --checksum=sha256:1d48e25122f452147e795eb6e8e21eea217654756ece9e918fe0a8ece2ab108e \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/answer_extraction.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/answer_extraction.py

ENV PYTHONPATH=/opt/skyrl_gym

WORKDIR /app
