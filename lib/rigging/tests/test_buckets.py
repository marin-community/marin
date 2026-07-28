# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bucket-routed filesystems: which backend a URL resolves to, and the
config-declared endpoint and credentials it is built with."""

import pytest
import rigging.filesystem.cluster_config as cluster_config
import s3fs
from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.cluster_config import BucketSpec, StoreType, load_cluster_config, use_data_config
from rigging.filesystem.s3_compat import credentials_hint, s3_credentials, s3_endpoint

CLUSTER_YAML = """
data:
  scheme: s3
  region_buckets:
    us-east5:    { bucket: marin-us-east5, store: gcs }
    na:          { bucket: marin-na,       store: r2 }
    us-east-02a: { bucket: marin-cw,       store: coreweave, signing_region: US-EAST-02A }
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
    """Load a test cluster config as the only one on the search path.

    Routing decisions come from parsed YAML rather than a hand-built object, so the
    ``stores`` block these tests depend on is exercised end to end.
    """
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    (cluster_dir / "test.yaml").write_text(CLUSTER_YAML)
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (str(cluster_dir),))
    cluster_config.reset_data_config_cache()
    with use_data_config(load_cluster_config("test")):
        yield
    cluster_config.reset_data_config_cache()


@pytest.fixture
def keys(monkeypatch):
    """Distinct credentials per backend, as a machine talking to both would have."""
    monkeypatch.setenv("CW_KEY_ID", "cw-key")
    monkeypatch.setenv("CW_KEY_SECRET", "cw-secret")
    monkeypatch.setenv("R2_KEY_ID", "r2-key")
    monkeypatch.setenv("R2_KEY_SECRET", "r2-secret")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("CW_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("R2_S3_ENDPOINT", raising=False)


def test_declared_buckets_carry_their_backend_and_signing_region(config):
    assert cluster_config.data_buckets()["marin-cw"] == BucketSpec(
        "marin-cw", StoreType.COREWEAVE, signing_region="US-EAST-02A"
    )
    assert "some-unregistered-bucket" not in cluster_config.data_buckets()


def test_two_s3_backends_get_separate_filesystems(config, keys):
    """The point of bucket routing: one process, two s3:// backends, no env mutation."""
    cw_fs, cw_path = filesystem_for("s3://marin-cw/data/x.json")
    r2_fs, r2_path = filesystem_for("s3://marin-na/data/x.json")

    assert cw_path == "marin-cw/data/x.json"
    assert r2_path == "marin-na/data/x.json"
    assert cw_fs is not r2_fs
    assert cw_fs.key == "cw-key"
    assert r2_fs.key == "r2-key"


def test_coreweave_signs_with_the_bucket_region_and_r2_with_auto(config, keys):
    """CoreWeave routes on the bucket's region; R2 rejects the AWS region scheme entirely."""
    cw_fs, _ = filesystem_for("s3://marin-cw/x")
    r2_fs, _ = filesystem_for("s3://marin-na/x")

    assert cw_fs.client_kwargs["region_name"] == "US-EAST-02A"
    assert cw_fs.endpoint_url == "https://cwobject.example"
    assert r2_fs.client_kwargs["region_name"] == "auto"
    assert r2_fs.endpoint_url == "https://r2.example"


def test_endpoint_env_overrides_the_configured_endpoint(config, keys, monkeypatch):
    """How a pod reaches its node-local endpoint instead of the public one."""
    monkeypatch.setenv("CW_S3_ENDPOINT", "https://lota.internal")
    fs, _ = filesystem_for("s3://marin-cw/x")
    assert fs.endpoint_url == "https://lota.internal"


def test_missing_credentials_names_the_variables_to_set(config, monkeypatch):
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    with pytest.raises(MissingCredentials, match="CW_KEY_ID"):
        filesystem_for("s3://marin-cw/x")


def test_generic_aws_credentials_serve_a_single_backend(config, monkeypatch):
    """The AWS_* fallback: enough for a process talking to one backend."""
    monkeypatch.delenv("CW_KEY_ID", raising=False)
    monkeypatch.delenv("CW_KEY_SECRET", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "generic-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "generic-secret")

    assert s3_credentials(StoreType.COREWEAVE) == ("generic-key", "generic-secret")


def test_gcs_and_unregistered_buckets_use_the_guarded_factory(config, keys):
    """Anything not routed to a declared S3 backend keeps its existing behavior."""
    gcs_fs, gcs_path = filesystem_for("gs://marin-us-east5/x")
    unknown_fs, _ = filesystem_for("s3://not-a-marin-bucket/x")

    assert gcs_path == "marin-us-east5/x"
    assert not isinstance(gcs_fs, s3fs.S3FileSystem)  # wrapped in the cross-region guard
    assert isinstance(unknown_fs, s3fs.S3FileSystem)  # ambient env, not a routed backend


def test_gcs_has_no_s3_connection_settings(config):
    """GCS authenticates with application default credentials, so asking is an error."""
    with pytest.raises(ValueError, match="stores"):
        s3_endpoint(StoreType.GCS)


def test_credentials_hint_names_the_configured_variables(config):
    """The hint has to name the config's variables, not a hardcoded pair."""
    hint = credentials_hint(StoreType.R2)
    assert "R2_KEY_ID" in hint and "R2_KEY_SECRET" in hint
