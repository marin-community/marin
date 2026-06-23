# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for marin_cluster.config: provisioning parse + current-cluster pointer."""

import pytest
from marin_cluster import config as cluster_config
from marin_cluster.config import _parse_provisioning
from rigging.cluster_manifest import AuthProvider, ClusterAuth, ClusterManifest


def test_full_provisioning_parsed():
    prov = _parse_provisioning(
        {
            "gcp": {"project": "hai-gcp-models", "default_zone": "us-central1-a"},
            "iam": {
                "service_accounts": {"controller": "iris-controller", "worker": "iris-worker"},
                "principals": {"ci": "serviceAccount:ci@x", "operators": ["a@x", "b@x"]},
            },
            "iap_gclb": {"domain": "iris-marin.oa.dev", "resources": {"prefix": "iris-marin"}},
        }
    )
    assert prov is not None
    assert prov.gcp is not None and prov.gcp.project == "hai-gcp-models"
    assert prov.gcp.network == "default"  # defaulted
    assert prov.iam is not None and prov.iam.operators == ("a@x", "b@x")
    assert prov.iap_gclb is not None and prov.iap_gclb.resource_prefix == "iris-marin"
    assert prov.iap_gclb.controller_port == 10000  # defaulted


def test_resource_prefix_defaults_to_domain_label():
    prov = _parse_provisioning({"iap_gclb": {"domain": "iris-marin.oa.dev"}})
    assert prov is not None and prov.iap_gclb is not None
    assert prov.iap_gclb.resource_prefix == "iris-marin"


def test_absent_provisioning_is_none():
    assert _parse_provisioning(None) is None
    assert _parse_provisioning({}) is None


def test_gcp_property_raises_for_non_gcp_cluster():
    cfg = cluster_config.ClusterConfig(
        manifest=ClusterManifest(name="coreweave", dashboard_url=None, auth=ClusterAuth(AuthProvider.NONE)),
        provisioning=None,
    )
    with pytest.raises(ValueError, match="not a GCP cluster"):
        _ = cfg.gcp


def test_iap_gclb_without_domain_rejected():
    with pytest.raises(ValueError, match="requires a 'domain'"):
        _parse_provisioning({"iap_gclb": {"resources": {"prefix": "x"}}})


def test_current_cluster_pointer_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_config, "_CURRENT_CLUSTER_POINTER", tmp_path / "cluster")
    assert cluster_config.current_cluster() is None
    cluster_config.set_current_cluster("marin")
    assert cluster_config.current_cluster() == "marin"
