# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN apt-get update && apt-get install -y --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.3.5 pandas==2.2.2 scipy==1.17.0 rdkit==2025.9.3 \
    xmltodict==0.14.2 PyYAML==6.0.3 openapi-schema-validator==0.6.3 jsonschema==4.25.1 \
    nltk==3.9.2 langdetect==1.0.9 absl-py==2.3.1 immutabledict==4.3.0 \
    requests==2.33.1 flask==3.1.2 psutil==7.2.2 ipython==9.10.0 traitlets==5.14.3

ADD --chmod=0644 --checksum=sha256:af348f9e5d92c45767d0e7bd4a55eec642acc303efd588ed344d59648af262c2 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/aime/utils.py \
    /opt/skyrl_gym/skyrl_gym/envs/aime/utils.py

ADD --chmod=0644 --checksum=sha256:c171a926c790faa1046cf5684fe4e98e458904a4963f19a21387ff9857bc7346 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/lcb/livecodebench.py \
    /opt/skyrl_gym/skyrl_gym/envs/lcb/livecodebench.py

ADD --chmod=0644 --checksum=sha256:1d48e25122f452147e795eb6e8e21eea217654756ece9e918fe0a8ece2ab108e \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/answer_extraction.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/answer_extraction.py

ADD --chmod=0644 --checksum=sha256:2fc03229fb6c76454b7bed3b7ff0ac417facead6169cec3d2fc5aa9f571c0f53 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/calendar.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/calendar.py

ADD --chmod=0644 --checksum=sha256:da243b691ba49cdea20bb313e6bd0d96af7d65dddc701b2fe65b6da4b4dd0fa3 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/code_gen.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/code_gen.py

ADD --chmod=0644 --checksum=sha256:0629283c54f1fa8af410cf7315c85b9f68d2abb971113ae49199f30daa2ab4a5 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/format_verification.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/format_verification.py

ADD --chmod=0644 --checksum=sha256:82e55bcba81139627049cea2a832f0a9a5a9e9ee4c18ed224e3f8f56719fbcdf \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/instruction_following.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/instruction_following.py

ADD --chmod=0644 --checksum=sha256:9c1fc806447b83b3113609e99ad73bb02acb42cb34e66a7a0aa0b957d6ab3b0f \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/mcqa.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/mcqa.py

ADD --chmod=0644 --checksum=sha256:34e3a26415c662c42a88023b62bec9a1ec9e5cc257a70c23b7edf55a19525928 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/nvarc.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/nvarc.py

ADD --chmod=0644 --checksum=sha256:8c41da86140963eae6eaa46ef12e3fcabbd5b14e79978a81096c1d4a0c98dc36 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/rdkit_chemistry.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/rdkit_chemistry.py

ADD --chmod=0644 --checksum=sha256:ad70cd3250bcc17ff67b695c8a5d37c77d36e2ee843b84ab614791175f9628f5 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/sandbox.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/sandbox.py

ADD --chmod=0644 --checksum=sha256:a92f84e3f5f3be11905295ef85bf8c060c92aae3da9109297fe59de762566a04 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/structured_outputs.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/structured_outputs.py

ADD --chmod=0644 --checksum=sha256:eb0f3d62b9ab32d51e860a52c68496f9d335e0917b11c6d30fae18c8af68bb54 \
    https://raw.githubusercontent.com/marin-community/MarinSkyRL/d8b6e8c163def3660e9d3072c1c174226a1709fa/skyrl-gym/skyrl_gym/envs/nemotron_ultra/tool_call.py \
    /opt/skyrl_gym/skyrl_gym/envs/nemotron_ultra/tool_call.py

ADD --chmod=0644 --checksum=sha256:c79197dc02e33b63b58b30783e3cfe0076e8edf6ceb409835b5d0762756b621d \
    https://raw.githubusercontent.com/NVIDIA/NeMo-Skills/bcf059af55c20a89f797724598f9908d126153e6/nemo_skills/code_execution/local_sandbox/local_sandbox_server.py \
    /opt/nemo_skills/local_sandbox_server.py

ADD --chmod=0644 --checksum=sha256:63910badb26dd45657244b9a775a38dab9d6a73e15c32cdb90503ba89a1bd9b8 \
    https://codeload.github.com/abukharin-nv/verifiable-instructions/tar.gz/f46a5ac87b1400a4f8973039844b6be9b56e3faf \
    /tmp/verifiable-instructions.tar.gz

RUN mkdir -p /opt/instructions \
    && tar -xzf /tmp/verifiable-instructions.tar.gz --strip-components=1 -C /opt/instructions \
    && pip install --no-deps /opt/instructions \
    && rm /tmp/verifiable-instructions.tar.gz

RUN chmod -R a+rX /opt/skyrl_gym /opt/nemo_skills

ENV PYTHONPATH=/opt/skyrl_gym NLTK_DATA=/opt/nltk_data
RUN python -m nltk.downloader -d /opt/nltk_data punkt_tab

WORKDIR /app
