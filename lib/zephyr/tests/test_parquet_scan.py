# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polars scans authenticate with the store credentials the resolver finds."""

import polars as pl
import pytest
import rigging.filesystem.cluster_config as cluster_config
from zephyr import parquet_scan

CLUSTER_YAML = """
data:
  scheme: s3
  region_buckets:
    us-east-02a: { bucket: marin-us-east-02a, store: coreweave, signing_region: US-EAST-02A }
  stores:
    coreweave:
      endpoint: https://cwobject.example
      endpoint_env: CW_S3_ENDPOINT
      key_id_env: CW_KEY_ID
      key_secret_env: CW_KEY_SECRET
"""


@pytest.fixture
def pod_secrets(tmp_path, monkeypatch):
    """The Iris task-pod shape: a mounted bundle of named secret files, no storage env."""
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    (cluster_dir / "test.yaml").write_text(CLUSTER_YAML)
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (str(cluster_dir),))
    cluster_config.reset_data_config_cache()
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "CW_KEY_ID").write_text("file-key")
    (secrets / "CW_KEY_SECRET").write_text("file-secret")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(secrets))
    for name in (
        "CW_KEY_ID",
        "CW_KEY_SECRET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    yield
    cluster_config.reset_data_config_cache()


@pytest.fixture
def scan_calls(monkeypatch):
    calls: list[tuple[str, dict[str, str]]] = []

    def capture(path: str, *, schema: pl.Schema | None = None, storage_options: dict[str, str]) -> pl.LazyFrame:
        calls.append((path, storage_options))
        return pl.LazyFrame()

    monkeypatch.setattr(pl, "scan_parquet", capture)
    return calls


def test_scan_parquet_authenticates_with_file_delivered_pod_keys(pod_secrets, scan_calls):
    """The credential regression: the scan receives the mounted pair, resolved for real."""
    parquet_scan.scan_parquet("s3://marin-us-east-02a/spill/run-0.parquet")
    assert scan_calls == [
        (
            "s3://marin-us-east-02a/spill/run-0.parquet",
            {"aws_access_key_id": "file-key", "aws_secret_access_key": "file-secret"},
        )
    ]


def test_scan_parquet_merges_pod_keys_with_virtual_host_endpoint(pod_secrets, scan_calls, monkeypatch):
    """Endpoint qualification and credential resolution compose: the pod's failing
    request shape needed the bucket-qualified host and the file-delivered keys together.
    """
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://cwlota.com")
    parquet_scan.scan_parquet("s3://marin-us-east-02a/spill/run-0.parquet")
    assert scan_calls == [
        (
            "s3://marin-us-east-02a/spill/run-0.parquet",
            {
                "aws_access_key_id": "file-key",
                "aws_secret_access_key": "file-secret",
                "aws_endpoint_url": "http://marin-us-east-02a.cwlota.com",
                "aws_virtual_hosted_style_request": "true",
            },
        )
    ]
