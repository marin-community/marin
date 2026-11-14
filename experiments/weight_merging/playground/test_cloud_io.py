# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from rigging.filesystem.cluster_config import StoreType

from experiments.weight_merging.playground import cloud_io


@pytest.mark.parametrize("endpoint", [None, "https://s3.amazonaws.com"])
def test_reject_aws_fallback(monkeypatch, endpoint):
    spec = SimpleNamespace(store=StoreType.COREWEAVE, signing_region="RNO2A")
    monkeypatch.setattr(cloud_io, "data_buckets", lambda: {"marin-rno2a": spec})
    monkeypatch.setattr(cloud_io, "filesystem_for", lambda _: (SimpleNamespace(endpoint_url=endpoint), ""))
    with pytest.raises(ValueError, match="Expected a CoreWeave endpoint"):
        cloud_io.require_coreweave("s3://marin-rno2a/checkpoint")


def test_unknown_bucket_never_invokes_transfer(monkeypatch):
    monkeypatch.setattr(cloud_io, "data_buckets", lambda: {})
    monkeypatch.setattr(cloud_io.sys, "argv", ["cloud_io", "cp", "/tmp/weights", "s3://unknown/checkpoint"])
    invoked = []
    monkeypatch.setattr(cloud_io, "fsutil_main", lambda: invoked.append(True))
    with pytest.raises(ValueError, match="explicit CoreWeave signing region"):
        cloud_io.main()
    assert not invoked
