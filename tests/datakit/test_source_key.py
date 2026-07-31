# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.source_key import datakit_source_key


def test_datakit_source_key_removes_marin_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    assert (
        datakit_source_key("s3://marin-us-east-02a/marin/datakit/normalize/foo/outputs/main")
        == "datakit/normalize/foo/outputs/main"
    )


def test_datakit_source_key_preserves_paths_outside_marin_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    assert datakit_source_key("/tmp/datakit/source") == "/tmp/datakit/source"


def test_datakit_source_key_recognizes_other_marin_regions(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1/marin")

    assert (
        datakit_source_key("s3://marin-us-east-02a/marin/datakit/normalize/foo/outputs/main")
        == "datakit/normalize/foo/outputs/main"
    )


def test_datakit_source_key_rejects_marin_prefix_itself(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    with pytest.raises(ValueError, match="must be below a Marin data prefix"):
        datakit_source_key("s3://marin-us-east-02a/marin")


def test_datakit_source_key_rejects_unknown_object_store(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")

    with pytest.raises(ValueError, match="not under a configured Marin data prefix"):
        datakit_source_key("s3://unmanaged-bucket/data/source")
