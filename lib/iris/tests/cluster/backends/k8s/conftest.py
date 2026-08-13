# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Kubernetes backend tests."""

import pytest
from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.testing.k8s import make_kueue_provider, pod_config


@pytest.fixture
def k8s():
    return InMemoryK8sService(namespace="iris")


@pytest.fixture
def provider(k8s):
    result = K8sTaskProvider(
        kubectl=k8s,
        pods=pod_config(),
        cluster_scan_interval=0.0,
    )
    yield result
    result.close()


@pytest.fixture
def kueue_provider(k8s):
    result = make_kueue_provider(k8s)
    yield result
    result.close()
