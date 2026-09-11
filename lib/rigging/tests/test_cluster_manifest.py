# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the narrow cluster-manifest slice rigging models (identity + auth)."""

import pytest
from rigging.cluster_manifest import AuthProvider, load_manifest, parse_manifest


def test_iap_provider_parsed_with_audiences():
    doc = {
        "identity": {"name": "marin", "dashboard_url": "https://iris.oa.dev"},
        "auth": {
            "iap": {
                "url": "https://iris-marin.oa.dev",
                "desktop_oauth_client_id": "abc.apps.googleusercontent.com",
                "programmatic_audiences": ["abc.apps.googleusercontent.com"],
                "signed_header_audience": "/projects/1/global/backendServices/2",
            },
            "admin_users": ["alice@x.com", "bob@x.com"],
        },
    }
    m = parse_manifest(doc, name="fallback")

    assert m.name == "marin"
    assert m.dashboard_url == "https://iris.oa.dev"
    assert m.auth.provider is AuthProvider.IAP
    assert m.auth.iap is not None
    assert m.auth.iap.url == "https://iris-marin.oa.dev"
    assert m.auth.iap.programmatic_audiences == ("abc.apps.googleusercontent.com",)
    assert m.auth.admin_users == ("alice@x.com", "bob@x.com")


def test_provider_dispatch_gcp_none():
    assert parse_manifest({"auth": {"gcp": True}}, name="c").auth.provider is AuthProvider.GCP
    assert parse_manifest({}, name="c").auth.provider is AuthProvider.NONE


def test_name_falls_back_to_stem_when_identity_absent():
    assert parse_manifest({"data": {"scheme": "gs"}}, name="from-stem").name == "from-stem"


def test_iap_without_url_is_rejected():
    with pytest.raises(ValueError, match=r"auth\.iap requires a 'url'"):
        parse_manifest({"auth": {"iap": {"desktop_oauth_client_id": "x"}}}, name="c")


def test_document_preserves_unmodeled_sections():
    doc = {"identity": {"name": "marin"}, "provisioning": {"gcp": {"project": "p"}}}
    m = parse_manifest(doc, name="marin")
    # rigging does not model provisioning, but exposes it for callers above.
    assert m.document["provisioning"]["gcp"]["project"] == "p"


def test_load_manifest_from_dir(tmp_path):
    (tmp_path / "demo.yaml").write_text("identity:\n  name: demo\nauth:\n  iap:\n    url: https://demo.example\n")
    m = load_manifest("demo", dirs=(str(tmp_path),))
    assert m.name == "demo"
    assert m.auth.provider is AuthProvider.IAP
    assert m.auth.iap is not None and m.auth.iap.url == "https://demo.example"
