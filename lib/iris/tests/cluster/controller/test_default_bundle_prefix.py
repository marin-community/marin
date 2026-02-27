# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for default bundle prefix resolution via marin_temp_bucket."""

import os

from iris.cluster.controller.main import default_bundle_prefix


def test_default_bundle_prefix_uses_temp_bucket(monkeypatch):
    """When no bundle_prefix is configured, fallback resolves via marin_temp_bucket."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central2")
    prefix = default_bundle_prefix()
    assert prefix == "gs://marin-tmp-us-central2/ttl=7d/iris/bundles"


def test_default_bundle_prefix_local_fallback(monkeypatch):
    """Without GCS env, fallback produces a local file:// path."""
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    # region_from_metadata will fail in test env, so we get local fallback
    prefix = default_bundle_prefix()
    assert "iris/bundles" in prefix
