# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""scan_storage_options must mirror the ambient fsspec S3 config for polars/object_store.

The failure this guards against: shuffle chunks written through fsspec (virtual-hosted, per
``FSSPEC_S3``) being read back through ``pl.scan_parquet``'s object_store client at path-style,
which CoreWeave's gateways reject with a bare 400 on the first HEAD.
"""

import fsspec
import pytest

from zephyr.polars_io import scan_storage_options


@pytest.fixture
def coreweave_fsspec_conf(monkeypatch):
    """The FSSPEC_S3 shape rigging exports inside a CoreWeave cluster."""
    monkeypatch.setitem(
        fsspec.config.conf,
        "s3",
        {
            "endpoint_url": "http://cwlota.com",
            "client_kwargs": {"region_name": "auto"},
            "config_kwargs": {"s3": {"addressing_style": "virtual"}},
            "key": "test-key-id",
            "secret": "test-secret",
        },
    )


def test_mirrors_coreweave_config(coreweave_fsspec_conf):
    options = scan_storage_options("s3://bucket/tmp/ttl=1d/chunk.parquet")
    assert options == {
        # In virtual-hosted mode object_store wants the bucket already in the endpoint host;
        # setting only the flag would strip the bucket from the path but not add it to the host.
        "aws_endpoint_url": "http://bucket.cwlota.com",
        "aws_allow_http": "true",
        "aws_region": "auto",
        "aws_virtual_hosted_style_request": "true",
        "aws_access_key_id": "test-key-id",
        "aws_secret_access_key": "test-secret",
    }


def test_virtual_endpoint_is_per_bucket(coreweave_fsspec_conf):
    a = scan_storage_options("s3://marin-us-east-02a/tmp/chunk.parquet")
    b = scan_storage_options("s3://other-bucket/tmp/chunk.parquet")
    assert a is not None and b is not None
    assert a["aws_endpoint_url"] == "http://marin-us-east-02a.cwlota.com"
    assert b["aws_endpoint_url"] == "http://other-bucket.cwlota.com"


def test_non_s3_paths_get_no_options(coreweave_fsspec_conf):
    assert scan_storage_options("/local/run-0000.spill") is None
    assert scan_storage_options("gs://bucket/chunk.parquet") is None
    assert scan_storage_options("memory://chunk.parquet") is None


def test_plain_aws_defers_to_polars_defaults(monkeypatch):
    monkeypatch.setitem(fsspec.config.conf, "s3", {})
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    assert scan_storage_options("s3://bucket/chunk.parquet") is None


def test_path_style_endpoint_is_not_forced_virtual(monkeypatch):
    monkeypatch.setitem(fsspec.config.conf, "s3", {"endpoint_url": "http://minio.local:9000"})
    options = scan_storage_options("s3://bucket/chunk.parquet")
    assert options is not None
    assert "aws_virtual_hosted_style_request" not in options
    assert options["aws_endpoint_url"] == "http://minio.local:9000"
