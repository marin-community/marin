# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for zephyr/polars_io.py.

Zephyr writes its intermediate Parquet through rigging's fsspec layer and reads
it back through Polars' object_store client. The two disagree about CoreWeave
object storage, which rejects path-style requests; these tests pin the endpoint
translation that keeps the read side addressed like the write side.
"""

import polars as pl
import pytest
from zephyr.polars_io import scan_parquet

_CW_ENDPOINT = "https://cwlota.com"
_S3_PATH = "s3://marin-us-east-02a/execution/stage0/run.parquet"


@pytest.fixture
def scan_calls(monkeypatch):
    """Record the arguments Polars' object_store scanner receives."""
    calls: list[tuple[str, dict | None]] = []

    def fake_scan(path, *, storage_options=None):
        calls.append((path, storage_options))
        return pl.LazyFrame()

    monkeypatch.setattr(pl, "scan_parquet", fake_scan)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    return calls


@pytest.mark.parametrize(
    ("path", "endpoint"),
    [
        pytest.param("/local/run.parquet", None, id="local-path"),
        pytest.param("gs://bucket/run.parquet", None, id="gcs-path"),
        pytest.param(_S3_PATH, None, id="s3-without-endpoint"),
        pytest.param(_S3_PATH, "https://s3.us-east-1.amazonaws.com", id="s3-path-style-endpoint"),
    ],
)
def test_scan_parquet_passes_through_when_addressing_is_not_special(scan_calls, monkeypatch, path, endpoint):
    if endpoint is not None:
        monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)

    scan_parquet(path)

    assert scan_calls == [(path, None)]


def test_scan_parquet_moves_bucket_into_virtual_host_endpoint(scan_calls, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", _CW_ENDPOINT)

    scan_parquet(_S3_PATH)

    assert scan_calls == [
        (
            _S3_PATH,
            {
                "aws_endpoint_url": "https://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]


def test_scan_parquet_keeps_endpoint_that_already_names_the_bucket(scan_calls, monkeypatch):
    """A pre-resolved virtual-host endpoint must not gain a second bucket label."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://marin-us-east-02a.cwlota.com")

    scan_parquet(_S3_PATH)

    assert scan_calls == [
        (
            _S3_PATH,
            {
                "aws_endpoint_url": "https://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]


def test_scan_parquet_prefers_the_s3_specific_endpoint_variable(scan_calls, monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://s3.us-east-1.amazonaws.com")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", _CW_ENDPOINT)

    scan_parquet(_S3_PATH)

    assert scan_calls == [
        (
            _S3_PATH,
            {
                "aws_endpoint_url": "https://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]
