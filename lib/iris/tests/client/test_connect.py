# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the click-free cluster connection helpers in ``iris.client.connect``."""

import pytest

from iris.client.connect import IRIS_CLUSTER_CONFIG_DIRS, connect_to_cluster, resolve_cluster_name
from iris.rpc import config_pb2


def test_resolve_cluster_name_prefers_cli_name():
    config = config_pb2.IrisClusterConfig(name="from-config")
    assert resolve_cluster_name(config, None, "from-cli") == "from-cli"


def test_resolve_cluster_name_falls_back_to_config_name():
    config = config_pb2.IrisClusterConfig(name="from-config")
    assert resolve_cluster_name(config, None, None) == "from-config"


def test_resolve_cluster_name_defaults_when_nothing_known():
    assert resolve_cluster_name(None, None, None) == "default"


def test_iris_cluster_config_dirs_include_intree_config():
    assert any(d.endswith("lib/iris/config") for d in IRIS_CLUSTER_CONFIG_DIRS)


def test_connect_to_cluster_unknown_cluster_raises():
    # Resolution happens before any network I/O, so an unknown cluster name
    # fails fast with FileNotFoundError rather than a connection error.
    with pytest.raises(FileNotFoundError):
        with connect_to_cluster("definitely-not-a-real-cluster-xyz"):
            pass
