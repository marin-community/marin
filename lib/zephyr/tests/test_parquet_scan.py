# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from zephyr.parquet_scan import storage_options_for_path


def test_storage_options_qualify_coreweave_bucket_endpoint(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://cwobject.com")

    assert storage_options_for_path("s3://marin-us-east-02a/tmp/ttl=30d/stack-v2") == {
        "aws_endpoint_url": "https://marin-us-east-02a.cwobject.com",
        "aws_virtual_hosted_style_request": "true",
    }


def test_storage_options_leave_local_paths_to_polars_defaults(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://cwobject.com")

    assert storage_options_for_path("/tmp/stack-v2") is None
