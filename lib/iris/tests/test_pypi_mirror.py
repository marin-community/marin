# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``iris.cluster.providers.gcp.pypi_mirror``."""

import pytest
from iris.cluster.providers.gcp.pypi_mirror import (
    AR_PROJECT,
    PYPI_MIRROR_REPO,
    PYTORCH_CPU_MIRROR_REPO,
    PypiMirrorEnv,
    build_pypi_mirror_env,
)


def test_build_pypi_mirror_env_us():
    env = build_pypi_mirror_env("us")
    assert env.default_index == f"https://us-python.pkg.dev/{AR_PROJECT}/{PYPI_MIRROR_REPO}/simple/"
    assert env.pytorch_cpu_index == f"https://us-python.pkg.dev/{AR_PROJECT}/{PYTORCH_CPU_MIRROR_REPO}/simple/"
    assert env.keyring_provider == "subprocess"


def test_build_pypi_mirror_env_europe():
    env = build_pypi_mirror_env("europe")
    assert env.default_index == f"https://europe-python.pkg.dev/{AR_PROJECT}/{PYPI_MIRROR_REPO}/simple/"
    assert env.pytorch_cpu_index == f"https://europe-python.pkg.dev/{AR_PROJECT}/{PYTORCH_CPU_MIRROR_REPO}/simple/"


def test_build_pypi_mirror_env_unsupported_region_raises():
    with pytest.raises(ValueError, match="SUPPORTED_MULTI_REGIONS"):
        build_pypi_mirror_env("asia")


def test_pypi_mirror_env_as_env_returns_three_keys():
    env = PypiMirrorEnv(
        default_index="https://example/a/simple/",
        pytorch_cpu_index="https://example/b/simple/",
    )
    materialized = env.as_env()
    assert materialized == {
        "UV_DEFAULT_INDEX": "https://example/a/simple/",
        "UV_INDEX_PYTORCH_CPU": "https://example/b/simple/",
        "UV_KEYRING_PROVIDER": "subprocess",
    }


def test_build_pypi_mirror_env_custom_project_flows_through():
    env = build_pypi_mirror_env("us", project="custom-project")
    assert env.default_index == f"https://us-python.pkg.dev/custom-project/{PYPI_MIRROR_REPO}/simple/"
    assert env.pytorch_cpu_index == f"https://us-python.pkg.dev/custom-project/{PYTORCH_CPU_MIRROR_REPO}/simple/"


def test_url_trailing_slash():
    """URLs must end with `/simple/` (uv's UV_INDEX_<NAME> requires trailing slash on some versions)."""
    env = build_pypi_mirror_env("us")
    assert env.default_index.endswith("/simple/")
    assert env.pytorch_cpu_index.endswith("/simple/")
