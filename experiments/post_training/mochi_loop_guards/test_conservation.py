# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the conserved initializer's regional/HF verification."""

import hashlib

import pytest

from experiments.post_training.mochi_loop_guards.conservation import verified_files


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    """A three-file checkpoint: two weight shards (LFS) and one small config."""
    manifest = {
        "identity": "step-400407/2af3f828aa-cpu-a1",
        "files": [
            {"path": "shard-0.safetensors", "size": 8, "sha256": sha("shard-0")},
            {"path": "shard-1.safetensors", "size": 8, "sha256": sha("shard-1")},
            {"path": "config.json", "size": 5, "sha256": sha("conf!")},
        ],
    }
    hf = {
        "private": False,
        "sha": "cfc315a0",
        "siblings": [
            {"path": "shard-0.safetensors", "size": 8, "lfs": {"sha256": sha("shard-0")}},
            {"path": "shard-1.safetensors", "size": 8, "lfs": {"sha256": sha("shard-1")}},
            {"path": "config.json", "size": 5, "lfs": None},
            {"path": "README.md", "size": 12, "lfs": None},
        ],
    }
    listing = {
        "shard-0.safetensors": {"size": 8, "ETag": "etag-0"},
        "shard-1.safetensors": {"size": 8, "ETag": "etag-1"},
        "config.json": {"size": 5, "ETag": "etag-2"},
    }
    contents = {"config.json": "conf!"}
    return manifest, hf, listing, contents


def read_source(contents):
    return lambda name: contents[name].encode()


def test_verified_files_matches_sizes_hashes_and_flags_hf_only_files():
    manifest, hf, listing, contents = fixture()
    results = verified_files(hf, manifest, listing, read_source(contents))

    shard0, _shard1, config, readme = results
    assert shard0["hash_basis"].endswith("HF LFS SHA256")
    assert config["hash_basis"] == "regional content read SHA256"
    assert config["regional_ETag"] == "etag-2"
    assert readme == {"path": "README.md", "HF_repository_metadata_only": True}


def test_verified_files_rejects_a_size_mismatch():
    manifest, hf, listing, contents = fixture()
    listing["shard-1.safetensors"]["size"] = 9
    with pytest.raises(AssertionError, match="shard-1"):
        verified_files(hf, manifest, listing, read_source(contents))


def test_verified_files_rejects_a_changed_small_file():
    manifest, hf, listing, contents = fixture()
    contents["config.json"] = "conf?"
    with pytest.raises(AssertionError, match=r"config\.json"):
        verified_files(hf, manifest, listing, read_source(contents))


def test_verified_files_rejects_a_changed_lfs_hash():
    manifest, hf, listing, contents = fixture()
    hf["siblings"][1]["lfs"]["sha256"] = sha("tampered")
    with pytest.raises(AssertionError, match="shard-1"):
        verified_files(hf, manifest, listing, read_source(contents))
