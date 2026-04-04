# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped fixtures for kind (Kubernetes in Docker) integration tests.

Creates a disposable kind cluster once per test session and tears it down
on exit. Tests that depend on `kind_cluster` are skipped entirely when
the `kind` binary is not on PATH.
"""

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

CLUSTER_NAME = "iris-test"


@dataclass(frozen=True)
class KindCluster:
    kubeconfig_path: Path
    cluster_name: str


def _kind_installed() -> bool:
    return shutil.which("kind") is not None


@pytest.fixture(scope="session")
def kind_cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[KindCluster]:
    """Create a kind cluster for the session, tearing it down on exit.

    Skips the test session if `kind` is not installed. Fails (not skips)
    if kind is installed but cluster creation fails — that indicates a
    real environment problem.
    """
    if not _kind_installed():
        pytest.skip("kind not installed")

    kubeconfig_path = tmp_path_factory.mktemp("kind") / "kubeconfig"

    subprocess.run(
        ["kind", "create", "cluster", "--name", CLUSTER_NAME, "--wait", "60s"],
        check=True,
    )
    try:
        subprocess.run(
            ["kind", "export", "kubeconfig", "--name", CLUSTER_NAME, "--kubeconfig", str(kubeconfig_path)],
            check=True,
        )
        yield KindCluster(kubeconfig_path=kubeconfig_path, cluster_name=CLUSTER_NAME)
    finally:
        subprocess.run(
            ["kind", "delete", "cluster", "--name", CLUSTER_NAME],
            check=True,
        )
