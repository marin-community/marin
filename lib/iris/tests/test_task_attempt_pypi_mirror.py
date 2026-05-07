# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AR PyPI mirror helpers inlined in ``task_attempt``."""

import pytest
from iris.cluster.worker.task_attempt import (
    _AR_PROJECT,
    _PYPI_MIRROR_REPO,
    _PYTORCH_CPU_MIRROR_REPO,
    _build_pypi_mirror_env,
)


def test_build_pypi_mirror_env_url_shape():
    env = _build_pypi_mirror_env("us")
    assert env.default_index == f"https://us-python.pkg.dev/{_AR_PROJECT}/{_PYPI_MIRROR_REPO}/simple/"
    assert env.pytorch_cpu_index == f"https://us-python.pkg.dev/{_AR_PROJECT}/{_PYTORCH_CPU_MIRROR_REPO}/simple/"


def test_build_pypi_mirror_env_unsupported_region_raises():
    with pytest.raises(ValueError, match="_SUPPORTED_MULTI_REGIONS"):
        _build_pypi_mirror_env("asia")


def test_build_pypi_mirror_env_as_env_wiring():
    env = _build_pypi_mirror_env("europe")
    assert env.as_env() == {
        "UV_DEFAULT_INDEX": f"https://europe-python.pkg.dev/{_AR_PROJECT}/{_PYPI_MIRROR_REPO}/simple/",
        "UV_INDEX_PYTORCH_CPU": f"https://europe-python.pkg.dev/{_AR_PROJECT}/{_PYTORCH_CPU_MIRROR_REPO}/simple/",
        "UV_KEYRING_PROVIDER": "subprocess",
    }
