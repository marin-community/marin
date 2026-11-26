# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The condensed OpenSWE python base images and their digest-pin bindings.

Upstream builds ten ``openswe-python-*`` base images locally
(``scripts/prepare_baseimg.py``): ``continuumio/miniconda3:25.3.1-1`` plus a
conda ``testbed`` env per Python version, with ``conda activate testbed``
appended to ``~/.bashrc``. The upstream images are never published, so the
profile rebuilds the same recipe in-region (Kaniko, internal GHCR package) and
records one digest-pinned reference per version into the bindings manifest the
converter requires. The recipe stays byte-equivalent to the upstream template;
the fidelity gate proves outcome equivalence per task.
"""

import json

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.openswe.sources import UPSTREAM_REVISION

MINICONDA3_PIN = "continuumio/miniconda3:25.3.1-1"
APT_MIRROR = "deb.debian.org"
PYPI_INDEX = "https://pypi.org/simple/"
BINDINGS_FILENAME = "condensed-base-bindings.json"

# The upstream prepare_baseimg.py versions, in upstream order.
PYTHON_VERSIONS = ("2.7", *(f"3.{minor}" for minor in range(5, 15)))

DOCKERFILE_TEMPLATE = f"""# The condensed OpenSWE base: the upstream prepare_baseimg.py recipe verbatim,
# bound to the digest-pinned miniconda3 base (the fidelity gate proves outcome
# equivalence per task against the locally built upstream image).
FROM {{miniconda_pin}}
RUN sed -i 's|deb.debian.org|{APT_MIRROR}|g' /etc/apt/sources.list.d/debian.sources && \\
    apt update && \\
    rm -rf /var/lib/apt/lists/*
RUN conda create -n testbed python={{python_version}} -y; \\
    echo "conda activate testbed" >> ~/.bashrc; \\
    conda activate testbed; \\
    pip config set global.index-url {PYPI_INDEX};
"""


def base_dockerfile(python_version: str) -> str:
    """The condensed base recipe for one Python version."""
    return DOCKERFILE_TEMPLATE.format(miniconda_pin=MINICONDA3_PIN, python_version=python_version)


def load_condensed_bases(bindings_input: object) -> dict[str, str]:
    """The digest-pinned condensed base references from the staged bindings input.

    The bindings must record the exact upstream pin the converter serves; a
    missing or unpinned version is the converter's typed rejection, never a
    mutable reference.
    """
    if bindings_input is None:
        raise ValueError("The condensed base bindings are not staged for this conversion")
    path = StoragePath(str(bindings_input)) / BINDINGS_FILENAME
    manifest = json.loads(path.read_text())
    if manifest["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("The condensed base bindings do not cover the pinned OpenSWE revision")
    images = manifest["images"]
    for version, reference in images.items():
        if "@sha256:" not in reference:
            raise ValueError(f"The condensed base binding for Python {version} is not a digest-pinned reference")
    return dict(images)
