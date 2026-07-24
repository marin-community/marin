# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from click.testing import CliRunner
from iris.cli import cluster as cluster_cli
from iris.cluster.config import (
    AuthConfig,
    ControllerVmConfig,
    CoreweaveControllerConfig,
    CoreweavePlatformConfig,
    IrisClusterConfig,
    PlatformConfig,
    StorageConfig,
)
from iris.cluster.platforms.k8s.controller import K8sControllerProvider
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource


def test_controller_restart_missing_secret_fails_before_remote_rollout_access(monkeypatch):
    monkeypatch.delenv("IRIS_SIGNING_KEY", raising=False)
    monkeypatch.delenv("MISSING_OPERATOR_SIGNING_KEY", raising=False)
    config = IrisClusterConfig(
        name="test",
        platform=PlatformConfig(coreweave=CoreweavePlatformConfig(region="LGA1", namespace="iris")),
        controller=ControllerVmConfig(coreweave=CoreweaveControllerConfig()),
        storage=StorageConfig(remote_state_dir="gs://test-bucket/iris/test/state"),
        auth=AuthConfig(
            signing_key=["env:IRIS_SIGNING_KEY", "env:MISSING_OPERATOR_SIGNING_KEY"],
        ),
    )
    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sControllerProvider(
        config=config.platform.coreweave,
        label_prefix="iris",
        kubectl=k8s,
    )
    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _: SimpleNamespace(controller=provider))

    def fail_remote_read(_):
        raise AssertionError("rollout state was read before controller preflight")

    monkeypatch.setattr(cluster_cli, "read_rollout_record", fail_remote_read)

    result = CliRunner().invoke(cluster_cli.controller_restart, [], obj={"config": config})

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert k8s.list_json(K8sResource.SECRETS) == []
    assert k8s.list_json(K8sResource.DEPLOYMENTS) == []
