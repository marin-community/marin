# SWE-rebench V2.1 repair — python group base (conda).
# Covers v1-python, python_base_310, python_base_37 rows (11,925 tasks).
# Upstream anchors: the V2-era python bases were conda images (rows' test_cmds activate
# /opt/conda envs `testbed || base`); the repair authors the conda base explicitly.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Upstream common layer (union of the upstream ubuntu-family recipes).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip zip jq gnupg ca-certificates \
    build-essential pkg-config cmake \
    libffi-dev libssl-dev zlib1g-dev \
    locales locales-all tzdata \
    && rm -rf /var/lib/apt/lists/*

# Declared interpreter provisioning (repair, disclosed): Miniconda's base env provides the
# controller python; asserted below. Miniconda pinned installer for reproducible builds.
ARG MINICONDA_VERSION=24.7.1-0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) installer="Miniconda3-py311_${MINICONDA_VERSION}-Linux-x86_64.sh" ;; \
      arm64) installer="Miniconda3-py311_${MINICONDA_VERSION}-Linux-aarch64.sh" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://repo.anaconda.com/miniconda/${installer}" -o /tmp/miniconda.sh; \
    bash /tmp/miniconda.sh -b -p /opt/conda; \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Per-version toolchains: pre-create the declared python envs (the setup clones the row's
# declared env into `testbed`, offline from the local envs) WITH the test tooling the upstream
# python base recipe preinstalls. Versions are PINNED to what the upstream images actually ship,
# read from digest-pinned upstream images:
#   - v1-python rows run a conda testbed on PYTHON 3.9 (spectree@be06b21a: python 3.9.21,
#     pytest 8.3.4 + the full recipe set; clique@fbc9e2ab: python 3.9.20, pytest 8.3.3 + the
#     same set) — so the unlabeled default recipe env is py39, NOT py310.
#   - python_base_310 rows run the SYSTEM python 3.10 with pytest 9.0.2 + coverage
#     (django-migration-linter@6e86ff38: python 3.10.19, pytest 9.0.2, coverage 7.13.4, no
#     mock/responses) — reproduced here as a py310 conda env with the same pins.
#   - python_base_37 rows run python 3.7.17 with pytest 7.4.4 (websockets@5ffbd660: python
#     3.7.17, pytest 7.4.4 only) — py37 keeps 7.4.4 plus the last py37-compatible plugin set
#     (disclosed reconstruction; the read image predates the recipe's plugin additions).
# Without pytest the grading test_cmd dies with exit 127 (pytest not found) — confirmed by the
# v17/v18 evidence (six bare-`pytest` test_cmds). The grading test_cmd runs WITHOUT conda
# activation, so `testbed/bin` is placed FIRST on PATH (the upstream images expose their env
# the same way) and the build pre-creates `testbed` as a py39 clone (the setup re-clones it
# from the row's declared env before the row's install commands run).
RUN conda create -n py39 -y python=3.9 \
    && conda create -n py310 -y python=3.10 \
    && conda create -n py37 -y python=3.7 \
    && conda run -n py39 python -m pip install --no-cache-dir \
        pytest==8.3.4 pytest-asyncio==0.25.0 pytest-bdd==8.1.0 pytest-benchmark==5.1.0 \
        pytest-cov==6.0.0 pytest-mock==3.14.0 pytest-randomly==3.16.0 pytest-xdist==3.6.1 \
        mock==5.1.0 responses==0.25.3 requests-mock==1.12.1 freezegun==1.5.1 \
        hypothesis==6.123.1 trustme==1.2.0 coverage==7.6.9 \
    && conda run -n py310 python -m pip install --no-cache-dir \
        pytest==9.0.2 coverage==7.13.4 \
    && conda run -n py37 python -m pip install --no-cache-dir \
        pytest==7.4.4 pytest-asyncio==0.20.3 pytest-bdd==6.1.1 pytest-benchmark==3.4.1 \
        pytest-cov==4.0.0 pytest-mock==3.6.1 pytest-randomly==3.12.0 pytest-xdist==2.5.0 \
        mock==4.0.3 responses==0.17.0 requests-mock==1.9.0 freezegun==1.2.2 \
        hypothesis==6.56.4 trustme==0.9.0 coverage==7.2.7 \
    && conda create -n testbed --clone py39 -y \
    && conda clean -afy

ENV PATH="/opt/conda/envs/testbed/bin:/opt/conda/bin:${PATH}"

# Build-time guards: the controller bootstrap precondition and the declared toolchains.
RUN test "$(python -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "controller python >= 3.9 required" >&2; exit 1; }
RUN conda env list | grep -q py39 && conda env list | grep -q py310 && conda env list | grep -q py37 \
    && conda run -n testbed python -m pytest --version \
    || { echo "declared conda envs or testbed tooling missing" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
