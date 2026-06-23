# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the admin layer's config-driving: Stack.from_config, iam input
derivation, and provider gating (GCP verbs reject non-GCP clusters)."""

import click
import pytest
from click.testing import CliRunner
from marin_cluster.admin.cli import _iam_inputs, admin
from marin_cluster.admin.iap_gclb import Stack
from marin_cluster.cli import main
from marin_cluster.config import (
    ClusterConfig,
    GcpProvisioning,
    IamProvisioning,
    IapGclbProvisioning,
    Provisioning,
)
from rigging.cluster_manifest import AuthProvider, ClusterAuth, ClusterManifest


def _marin_config() -> ClusterConfig:
    manifest = ClusterManifest(name="marin", dashboard_url="https://iris.oa.dev", auth=ClusterAuth(AuthProvider.IAP))
    provisioning = Provisioning(
        gcp=GcpProvisioning(project="hai-gcp-models", default_zone="us-central1-a"),
        iam=IamProvisioning(
            controller_service_account="iris-controller",
            worker_service_account="iris-worker",
            ci_principal="serviceAccount:iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com",
            operators=("russell.power@gmail.com",),
        ),
        iap_gclb=IapGclbProvisioning(
            domain="iris-marin.oa.dev",
            resource_prefix="iris-marin",
            controller_port=10000,
            discovery_label="iris-marin-controller",
        ),
    )
    return ClusterConfig(manifest=manifest, provisioning=provisioning)


def _coreweave_config() -> ClusterConfig:
    manifest = ClusterManifest(name="coreweave", dashboard_url=None, auth=ClusterAuth(AuthProvider.NONE))
    return ClusterConfig(manifest=manifest, provisioning=None)


def test_stack_from_config_maps_provisioning_to_resource_names():
    stack = Stack.from_config(_marin_config())
    assert stack.cluster == "marin"
    assert stack.project == "hai-gcp-models"
    assert stack.zone == "us-central1-a"
    assert stack.domain == "iris-marin.oa.dev"
    assert stack.prefix == "iris-marin"
    assert stack.controller_port == 10000
    # discovery_label drives both VM discovery and the firewall network tag.
    assert stack.controller_label == "iris-marin-controller"
    # Resource names all derive from the prefix.
    assert stack.backend == "iris-marin-be"
    assert stack.neg == "iris-marin-neg"
    assert stack.cert == "iris-marin-cert"
    assert stack.deny_firewall == "iris-marin-deny-public-10000"


def test_controller_label_falls_back_to_prefix_without_discovery_label():
    cfg = _marin_config()
    iap = cfg.provisioning.iap_gclb
    cfg = ClusterConfig(
        manifest=cfg.manifest,
        provisioning=Provisioning(
            gcp=cfg.provisioning.gcp,
            iam=cfg.provisioning.iam,
            iap_gclb=IapGclbProvisioning(domain=iap.domain, resource_prefix="iris-marin"),
        ),
    )
    stack = Stack.from_config(cfg)
    assert stack.controller_label == "iris-marin-controller"


def test_stack_from_config_rejects_non_gcp_cluster():
    with pytest.raises(ValueError, match="not a GCP cluster"):
        Stack.from_config(_coreweave_config())


def test_iam_inputs_derive_from_config():
    project, controller_sa, worker_sa, operators, ci_principal = _iam_inputs(_marin_config())
    assert project == "hai-gcp-models"
    assert controller_sa == "iris-controller"
    assert worker_sa == "iris-worker"
    assert operators == ("russell.power@gmail.com",)
    assert ci_principal == "serviceAccount:iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com"


def test_iam_inputs_reject_non_gcp_cluster():
    with pytest.raises(click.ClickException, match="not a GCP cluster"):
        _iam_inputs(_coreweave_config())


def test_iap_status_on_coreweave_fails_with_clear_message(monkeypatch):
    # The GCP gate must trip (with a clean error) before any gcloud probe runs.
    monkeypatch.setattr(ClusterConfig, "load", classmethod(lambda cls, cluster=None: _coreweave_config()))
    result = CliRunner().invoke(admin, ["iap", "status"], obj={"cluster": "coreweave"})
    assert result.exit_code != 0
    assert "not a GCP cluster" in result.output


def test_admin_help_lists_groups():
    result = CliRunner().invoke(admin, ["--help"])
    assert result.exit_code == 0, result.output
    for group in ("iap", "iam", "user", "tunnel"):
        assert group in result.output


def test_admin_iap_help_lists_stages():
    result = CliRunner().invoke(admin, ["iap", "--help"])
    assert result.exit_code == 0, result.output
    for verb in ("deploy", "address", "cert", "firewall", "backend", "frontend", "grant", "status", "teardown"):
        assert verb in result.output


def test_admin_group_mounted_on_main():
    result = CliRunner().invoke(main, ["admin", "--help"])
    assert result.exit_code == 0, result.output
    assert "iap" in result.output and "iam" in result.output
