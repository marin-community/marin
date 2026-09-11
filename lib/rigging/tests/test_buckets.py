# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bucket-routed filesystems: which backend a URL resolves to, and the
endpoint and credentials it is built with."""

import pytest
import rigging.filesystem.cluster_config as cluster_config
import s3fs
from fsspec.implementations.local import LocalFileSystem
from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.cluster_config import StoreType, load_cluster_config, use_data_config

CLUSTER_YAML = """
data:
  scheme: s3
  region_buckets:
    na:          { bucket: marin-na, store: r2 }
    us-east-02a: { bucket: marin-cw, store: coreweave, signing_region: US-EAST-02A }
  stores:
    coreweave:
      endpoint: https://cwobject.example
      endpoint_env: CW_S3_ENDPOINT
      key_id_env: CW_KEY_ID
      key_secret_env: CW_KEY_SECRET
    r2:
      endpoint: https://r2.example
      endpoint_env: R2_S3_ENDPOINT
      key_id_env: R2_KEY_ID
      key_secret_env: R2_KEY_SECRET
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    """A two-backend cluster config, with each backend's credentials in the environment."""
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    (cluster_dir / "test.yaml").write_text(CLUSTER_YAML)
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (str(cluster_dir),))
    cluster_config.reset_data_config_cache()
    for name, value in [
        ("CW_KEY_ID", "cw-key"),
        ("CW_KEY_SECRET", "cw-secret"),
        ("R2_KEY_ID", "r2-key"),
        ("R2_KEY_SECRET", "r2-secret"),
    ]:
        monkeypatch.setenv(name, value)
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "CW_S3_ENDPOINT", "R2_S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    with use_data_config(load_cluster_config("test")):
        yield
    cluster_config.reset_data_config_cache()


def test_each_bucket_routes_to_its_own_backend(config, tmp_path):
    """The point of bucket routing: two s3:// backends live at once, with the endpoint,
    signing region, and keys each was declared with — and no environment mutation.

    CoreWeave routes on the bucket's own region; R2 rejects the AWS region scheme
    entirely. A bucket no config declares must not inherit either backend's endpoint,
    and local paths still fall through to the guarded factory.
    """
    cw_fs, cw_path = filesystem_for("s3://marin-cw/data/x.json")
    r2_fs, r2_path = filesystem_for("s3://marin-na/data/x.json")
    unknown_fs, unknown_path = filesystem_for("s3://not-a-marin-bucket/x")
    local_fs, local_path = filesystem_for(str(tmp_path / "x"))

    assert (cw_path, r2_path, unknown_path) == ("marin-cw/data/x.json", "marin-na/data/x.json", "not-a-marin-bucket/x")
    assert (cw_fs.key, cw_fs.endpoint_url, cw_fs.client_kwargs["region_name"]) == (
        "cw-key",
        "https://cwobject.example",
        "US-EAST-02A",
    )
    assert (r2_fs.key, r2_fs.endpoint_url, r2_fs.client_kwargs["region_name"]) == (
        "r2-key",
        "https://r2.example",
        "auto",
    )
    assert isinstance(unknown_fs, s3fs.S3FileSystem)
    assert unknown_fs.endpoint_url not in ("https://cwobject.example", "https://r2.example")
    assert isinstance(local_fs, LocalFileSystem)
    assert local_path == str(tmp_path / "x")


def test_credentials_come_from_the_variables_the_backend_declares(config, monkeypatch):
    """Namespaced keys first, then the generic AWS_* pair for a single-backend process;
    with neither, the error names the variables to set. The endpoint variable is how a
    pod reaches its node-local endpoint instead of the public one.
    """
    monkeypatch.setenv("CW_S3_ENDPOINT", "https://lota.internal")
    assert filesystem_for("s3://marin-cw/x")[0].endpoint_url == "https://lota.internal"

    monkeypatch.delenv("CW_KEY_ID")
    monkeypatch.delenv("CW_KEY_SECRET")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "generic-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "generic-secret")
    assert filesystem_for("s3://marin-cw/x")[0].key == "generic-key"

    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
    s3fs.S3FileSystem.clear_instance_cache()
    with pytest.raises(MissingCredentials, match="CW_KEY_ID"):
        filesystem_for("s3://marin-cw/x")


def test_gcs_is_not_an_s3_backend(config):
    """GCS uses application default credentials, so asking for its S3 settings is an error."""
    with pytest.raises(ValueError, match="stores"):
        cluster_config.store_config(StoreType.GCS)
