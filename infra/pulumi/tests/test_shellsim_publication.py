# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use a CLI boundary fake; guest execution belongs to Shellsim's own tests."""

import argparse
import json
import subprocess
from dataclasses import replace

import fsspec
import pytest
import s3fs
from fsspec.implementations.memory import MemoryFileSystem
from rigging.filesystem import cluster_config
from rigging.filesystem.storage_path import StoragePath

from infra.shellsim.ports import ReleaseConfig, StageConfig, publish_ports


@pytest.fixture
def public_store(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "r2.yaml").write_text(
        """data:
  scheme: s3
  region_buckets:
    na: { bucket: public-releases, store: r2 }
  stores:
    r2:
      endpoint: https://r2.example
      endpoint_env: R2_S3_ENDPOINT
      key_id_env: R2_KEY_ID
      key_secret_env: R2_KEY_SECRET
"""
    )
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (str(config),))
    monkeypatch.setitem(fsspec.config.conf, "s3", {"endpoint_url": "https://cw.example"})
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "cw-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cw-secret")
    monkeypatch.setenv("R2_KEY_ID", "r2-id")
    monkeypatch.setenv("R2_KEY_SECRET", "r2-secret")
    monkeypatch.delenv("R2_S3_ENDPOINT", raising=False)
    cluster_config.reset_data_config_cache()
    storage = MemoryFileSystem(skip_instance_cache=True)

    def authenticated(fs):
        assert (fs.endpoint_url, fs.key, fs.secret) == ("https://r2.example", "r2-id", "r2-secret")
        return storage

    monkeypatch.setattr(s3fs.S3FileSystem, "exists", lambda fs, key: authenticated(fs).exists(key))
    monkeypatch.setattr(s3fs.S3FileSystem, "open", lambda fs, key, mode: authenticated(fs).open(key, mode))
    yield storage
    cluster_config.reset_data_config_cache()
    s3fs.S3FileSystem.clear_instance_cache()


def test_r2_publication_preserves_stable_version_under_coreweave_configuration(tmp_path, monkeypatch, public_store):
    publication = StoragePath("s3://public-releases/shellsim/ports")

    def key(relative):
        return str(publication / relative).removeprefix("s3://")

    def read(relative):
        return public_store.cat_file(key(relative))

    public_store.pipe_file(key("release-version"), b"1")
    public_store.pipe_file(key("catalog.json"), b"previous catalog")
    cache = tmp_path / "cache"
    (cache / "assets/build").mkdir(parents=True)
    (cache / "assets/build/package.whl").write_bytes(b"checked release")
    checked = tmp_path / "checked"
    checked.mkdir()
    (checked / "graph.json").write_text("{}")
    (checked / "results.json").write_text("{}")
    release = ReleaseConfig(
        source_revision="a" * 40,
        version="2",
        work_directory=str(tmp_path / "work"),
        worker_image="worker:checked",
        cache_prefix=str(cache),
        publication_prefix=str(publication),
        asset_url_prefix="https://public.example/shellsim/ports",
    )
    (tmp_path / "work/source" / release.source_revision / ".git").mkdir(parents=True)
    fail_stable = True

    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=release.source_revision)
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=StoragePath, required=True)
        parser.add_argument("--asset-output", type=StoragePath)
        parser.add_argument("--url-prefix", required=True)
        parser.add_argument("--copy-assets", action="store_true")
        args, _ = parser.parse_known_args(command[4:])
        if args.output == publication and fail_stable:
            raise subprocess.CalledProcessError(1, command)
        if args.copy_assets:
            target = str(args.asset_output / "assets/build/package.whl").removeprefix("s3://")
            public_store.pipe_file(target, (cache / "assets/build/package.whl").read_bytes())
        target = str(args.output / "catalog.json").removeprefix("s3://")
        public_store.pipe_file(target, json.dumps({"url": args.url_prefix + "/assets/build/package.whl"}).encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    config = StageConfig(release, str(tmp_path / "stages/published"), str(checked))
    with pytest.raises(subprocess.CalledProcessError):
        publish_ports(config)
    assert read("release-version") == b"1"
    assert read("catalog.json") == b"previous catalog"
    assert read("assets/build/package.whl") == b"checked release"
    snapshot = read("releases/2/catalog.json")
    fail_stable = False
    publish_ports(config)
    assert read("catalog.json") == snapshot
    assert read("release-version") == b"2"
    publish_ports(replace(config, release=replace(release, version="1")))
    assert read("catalog.json") == snapshot
    assert read("release-version") == b"2"
    assert public_store.exists(key("releases/1/catalog.json"))
