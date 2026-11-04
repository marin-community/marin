# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars scans combine the store's file-delivered credentials with endpoint qualification."""

import polars as pl
import pytest
from zephyr import parquet_scan


@pytest.fixture
def scan_calls(monkeypatch):
    calls: list[tuple[str, dict[str, str]]] = []

    def capture(path: str, *, schema: pl.Schema | None = None, storage_options: dict[str, str]) -> pl.LazyFrame:
        calls.append((path, storage_options))
        return pl.LazyFrame()

    monkeypatch.setattr(pl, "scan_parquet", capture)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    return calls


def test_scan_parquet_passes_declared_store_credentials(monkeypatch, scan_calls):
    monkeypatch.setattr(
        parquet_scan,
        "polars_s3_storage_options",
        lambda path: {"aws_access_key_id": "cw-key", "aws_secret_access_key": "cw-secret"},
    )
    parquet_scan.scan_parquet("s3://marin-us-east-02a/spill/run-0.parquet")
    assert scan_calls == [
        (
            "s3://marin-us-east-02a/spill/run-0.parquet",
            {"aws_access_key_id": "cw-key", "aws_secret_access_key": "cw-secret"},
        )
    ]


def test_scan_parquet_merges_credentials_with_virtual_host_endpoint(monkeypatch, scan_calls):
    monkeypatch.setattr(
        parquet_scan,
        "polars_s3_storage_options",
        lambda path: {"aws_access_key_id": "cw-key", "aws_secret_access_key": "cw-secret"},
    )
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://cwlota.com")
    parquet_scan.scan_parquet("s3://marin-us-east-02a/spill/run-0.parquet")
    assert scan_calls == [
        (
            "s3://marin-us-east-02a/spill/run-0.parquet",
            {
                "aws_access_key_id": "cw-key",
                "aws_secret_access_key": "cw-secret",
                "aws_endpoint_url": "http://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]
