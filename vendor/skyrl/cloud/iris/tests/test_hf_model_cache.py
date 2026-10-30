import json
import hashlib
import io
import os
from pathlib import Path

import fsspec
import huggingface_hub.constants
import numpy as np
import pytest
import rigging.timing
from safetensors.numpy import save, save_file

from cloud.iris import hf_model_cache
from cloud.iris.hf_model_cache import (
    HuggingFaceSnapshot,
    HuggingFaceSnapshotFile,
    ensure_hugging_face_model_cache,
    publish_hugging_face_snapshot,
    stage_artifact_model,
    stage_artifact_model_metadata,
    stage_model_metadata,
)
from marinskyrl.model_manifest import ModelManifest, snapshot_model_manifest


def _snapshot_file(path: str, payload: bytes) -> HuggingFaceSnapshotFile:
    return HuggingFaceSnapshotFile(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _write_memory_file(filesystem, root: str, path: str, payload: bytes) -> HuggingFaceSnapshotFile:
    with filesystem.open(f"{root}/{path}", "wb") as destination:
        destination.write(payload)
    return _snapshot_file(path, payload)


def _model_snapshot(filesystem, root: str, *, tokenizer_class: str | None = None):
    config = b"{}"
    tokenizer = b"{}"
    tokenizer_config = json.dumps({"tokenizer_class": tokenizer_class}).encode() if tokenizer_class else b"{}"
    weights = save({"weight": np.arange(4, dtype=np.float32)})
    files = (
        _write_memory_file(filesystem, root, "config.json", config),
        _write_memory_file(filesystem, root, "tokenizer.json", tokenizer),
        _write_memory_file(filesystem, root, "tokenizer_config.json", tokenizer_config),
        _write_memory_file(filesystem, root, "model.safetensors", weights),
    )
    return files, weights


def _snapshot(filesystem, root: str, files: tuple[HuggingFaceSnapshotFile, ...]) -> HuggingFaceSnapshot:
    return HuggingFaceSnapshot(filesystem=filesystem, root=root, files=files)


class _BoundedReader(io.BufferedIOBase):
    def __init__(self, source, read_sizes: list[int]) -> None:
        self.source = source
        self.read_sizes = read_sizes

    def read(self, size: int = -1) -> bytes:
        assert size >= 0, "streamed model parameters must not use an unbounded read"
        self.read_sizes.append(size)
        return self.source.read(size)

    def __enter__(self):
        self.source.__enter__()
        return self

    def __exit__(self, *args):
        return self.source.__exit__(*args)


class _InterruptedReader(_BoundedReader):
    def __init__(self, source, interrupted: list[bool]) -> None:
        super().__init__(source, [])
        self.interrupted = interrupted

    def read(self, size: int = -1) -> bytes:
        if size == 8 * 2**20 and not self.interrupted:
            self.interrupted.append(True)
            raise OSError("injected Hugging Face connection reset")
        return super().read(size)


def test_hugging_face_snapshot_streams_weights_and_normalizes_metadata(monkeypatch) -> None:
    source = fsspec.filesystem("memory")
    source_root = "stream-source/model"
    destination = "memory://stream-destination/model"
    files, weights = _model_snapshot(source, source_root, tokenizer_class="TokenizersBackend")
    original_open = source.open
    weight_read_sizes: list[int] = []

    def bounded_open(path: str, mode: str = "rb", **kwargs):
        opened = original_open(path, mode, **kwargs)
        if path.startswith(source_root) and path.endswith(".safetensors") and "r" in mode:
            return _BoundedReader(opened, weight_read_sizes)
        return opened

    monkeypatch.setattr(source, "open", bounded_open)

    manifest = publish_hugging_face_snapshot(
        _snapshot(source, source_root, files),
        destination,
        model_id="org/model",
        revision="a" * 40,
    )

    destination_fs, destination_root = fsspec.core.url_to_fs(destination)
    with destination_fs.open(f"{destination_root}/model.safetensors", "rb") as mirrored:
        assert mirrored.read() == weights
    with destination_fs.open(f"{destination_root}/tokenizer_config.json") as mirrored:
        assert json.load(mirrored)["tokenizer_class"] == "PreTrainedTokenizerFast"
    with destination_fs.open(f"{destination_root}/model.safetensors.index.json") as mirrored:
        assert json.load(mirrored)["weight_map"] == {"weight": "model.safetensors"}
    assert manifest.identity.startswith("sha256:")
    assert weight_read_sizes


def test_interrupted_snapshot_reuses_verified_weight_object(monkeypatch) -> None:
    source = fsspec.filesystem("memory")
    source_root = "resume-source/model"
    destination = "memory://resume-destination/model"
    files, _weights = _model_snapshot(source, source_root)
    publish_hugging_face_snapshot(
        _snapshot(source, source_root, files),
        destination,
        model_id="org/model",
        revision="b" * 40,
    )
    destination_fs, destination_root = fsspec.core.url_to_fs(destination)
    destination_fs.rm(f"{destination_root}/.marinskyrl-model-manifest.json")
    original_open = source.open
    opened_for_read: list[str] = []

    def recording_open(path: str, mode: str = "rb", **kwargs):
        if path.startswith(source_root) and "r" in mode:
            opened_for_read.append(path)
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(source, "open", recording_open)

    publish_hugging_face_snapshot(
        _snapshot(source, source_root, files),
        destination,
        model_id="org/model",
        revision="b" * 40,
    )

    assert not any(path.endswith("model.safetensors") for path in opened_for_read)


def test_interrupted_snapshot_replaces_corrupt_weight_object() -> None:
    source = fsspec.filesystem("memory")
    source_root = "repair-source/model"
    destination = "memory://repair-destination/model"
    files, weights = _model_snapshot(source, source_root)
    destination_fs, destination_root = fsspec.core.url_to_fs(destination)
    destination_fs.makedirs(destination_root, exist_ok=True)
    with destination_fs.open(f"{destination_root}/model.safetensors", "wb") as corrupt:
        corrupt.write(b"corrupt")

    publish_hugging_face_snapshot(
        _snapshot(source, source_root, files),
        destination,
        model_id="org/model",
        revision="c" * 40,
    )

    with destination_fs.open(f"{destination_root}/model.safetensors", "rb") as mirrored:
        assert mirrored.read() == weights


def test_hugging_face_snapshot_retries_an_interrupted_weight_stream(monkeypatch) -> None:
    source = fsspec.filesystem("memory")
    source_root = "retry-source/model"
    destination = "memory://retry-destination/model"
    files, weights = _model_snapshot(source, source_root)
    original_open = source.open
    interrupted: list[bool] = []
    weight_opens = 0

    def interrupted_open(path: str, mode: str = "rb", **kwargs):
        nonlocal weight_opens
        opened = original_open(path, mode, **kwargs)
        if path.startswith(source_root) and path.endswith(".safetensors") and "r" in mode:
            weight_opens += 1
            return _InterruptedReader(opened, interrupted)
        return opened

    monkeypatch.setattr(source, "open", interrupted_open)
    monkeypatch.setattr(rigging.timing.time, "sleep", lambda _delay: None)

    publish_hugging_face_snapshot(
        _snapshot(source, source_root, files),
        destination,
        model_id="org/model",
        revision="e" * 40,
    )

    destination_fs, destination_root = fsspec.core.url_to_fs(destination)
    with destination_fs.open(f"{destination_root}/model.safetensors", "rb") as mirrored:
        assert mirrored.read() == weights
    assert interrupted == [True]
    assert weight_opens == 2


def test_snapshot_rejects_weight_index_that_disagrees_with_shards() -> None:
    source = fsspec.filesystem("memory")
    source_root = "bad-index-source/model"
    destination = "memory://bad-index-destination/model"
    files, _weights = _model_snapshot(source, source_root)
    bad_index = json.dumps({"metadata": {}, "weight_map": {"other": "model.safetensors"}}).encode()
    files = (*files, _write_memory_file(source, source_root, "model.safetensors.index.json", bad_index))

    with pytest.raises(ValueError, match="does not match its shards"):
        publish_hugging_face_snapshot(
            _snapshot(source, source_root, files),
            destination,
            model_id="org/model",
            revision="d" * 40,
        )


def test_hub_download_temporarily_enables_network_access(tmp_path: Path, monkeypatch) -> None:
    def snapshot_download(*_args, **_kwargs):
        assert not huggingface_hub.constants.is_offline_mode()
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hf_model_cache, "snapshot_download", snapshot_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    resolved = hf_model_cache.download_hugging_face_snapshot(
        "laion/draft",
        revision="4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        destination=tmp_path,
    )

    assert resolved == tmp_path
    assert huggingface_hub.constants.is_offline_mode()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_repeated_draft_staging_uses_the_completed_region_cache(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    local_model = tmp_path / "node" / "draft"
    source = fsspec.filesystem("memory")
    source_root = f"ensure-source/{tmp_path.name}"
    files, _weights = _model_snapshot(source, source_root)
    snapshots = []

    def open_snapshot(model_id: str, revision: str):
        snapshots.append((model_id, revision))
        return HuggingFaceSnapshot(source, source_root, files)

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)
    model_id = "laion/draft"
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

    cache_uri, manifest = ensure_hugging_face_model_cache(
        model_id, revision, ttl_days=14, source_prefix="s3://region/experiments/run"
    )
    stage_model_metadata(cache_uri, manifest, str(local_model))
    repeated_uri, repeated_manifest = ensure_hugging_face_model_cache(
        model_id, revision, ttl_days=14, source_prefix="s3://region/experiments/next-run"
    )
    (local_model / "stale.bin").write_bytes(b"stale weights")
    stage_model_metadata(cache_uri, manifest, str(local_model))

    assert snapshots == [(model_id, revision)]
    assert repeated_uri == cache_uri
    assert repeated_manifest == manifest
    assert manifest.revision == revision
    assert manifest.identity.startswith("sha256:")
    assert {entry.path for entry in manifest.files} == {
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert (local_model / "config.json").read_text() == "{}"
    assert (local_model / "model.safetensors.index.json").is_file()
    assert not (local_model / "model.safetensors").exists()
    assert not (local_model / "stale.bin").exists()


def test_artifact_metadata_staging_leaves_weight_shards_remote(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
    destination = tmp_path / "node" / "model"

    metadata_bytes = stage_artifact_model_metadata(source.as_uri(), "artifact@v1:abc123", str(destination))

    assert metadata_bytes == sum(path.stat().st_size for path in (source / "config.json", source / "tokenizer.json"))
    assert (destination / "config.json").read_text() == "{}"
    assert (destination / "tokenizer.json").read_text() == "{}"
    assert not (destination / "model.safetensors").exists()


def test_artifact_model_staging_materializes_weight_shards(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
    destination = tmp_path / "node" / "model"

    materialized_bytes = stage_artifact_model(source.as_uri(), "artifact@v1:abc123", str(destination))

    assert materialized_bytes == sum(path.stat().st_size for path in source.iterdir())
    assert (destination / "config.json").read_text() == "{}"
    assert (destination / "tokenizer.json").read_text() == "{}"
    assert (destination / "model.safetensors").is_file()


def test_corrupt_completed_cache_is_repaired_under_the_distributed_lock(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "region-cache"
    cache.mkdir()
    (cache / ".marinskyrl-model-manifest.json").write_text("not json")
    (cache / "stale.safetensors").write_bytes(b"stale")
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    revision = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

    source = fsspec.filesystem("memory")
    source_root = f"repair-ensure-source/{tmp_path.name}"
    files, _weights = _model_snapshot(source, source_root)
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda _model_id, _revision: HuggingFaceSnapshot(source, source_root, files),
    )

    cache_uri, manifest = ensure_hugging_face_model_cache(
        "laion/draft", revision, ttl_days=14, source_prefix="s3://region/run"
    )

    assert cache_uri == str(cache)
    assert manifest == hf_model_cache.load_model_manifest(str(cache))
    assert not (cache / "stale.safetensors").exists()


def test_draft_manifest_can_share_the_policy_tokenizer(tmp_path: Path, monkeypatch) -> None:
    cache = tmp_path / "draft-cache"
    source = fsspec.filesystem("memory")
    source_root = f"draft-ensure-source/{tmp_path.name}"
    config = b"{}"
    weights = save({"weight": np.arange(4, dtype=np.float32)})
    files = (
        _write_memory_file(source, source_root, "config.json", config),
        _write_memory_file(source, source_root, "model.safetensors", weights),
    )

    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_args, **_kwargs: str(cache))
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda _model_id, revision: HuggingFaceSnapshot(source, source_root, files),
    )

    cache_uri, manifest = ensure_hugging_face_model_cache(
        "laion/draft",
        "4bdb47c08e5b5190bea3c7a93c3e14470230e469",
        ttl_days=14,
        source_prefix="s3://region/run",
        tokenizer_mode="policy",
    )

    assert cache_uri == str(cache)
    assert manifest.tokenizer_mode == "policy"
    assert not any(entry.path.startswith("tokenizer") for entry in manifest.files)
    assert (
        ModelManifest.from_mapping(json.loads((cache / ".marinskyrl-model-manifest.json").read_text()), str(cache))
        == manifest
    )


def test_model_manifest_rejects_metadata_paths_outside_the_model_root(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    manifest = snapshot_model_manifest(tmp_path, model_id="laion/draft", revision="pinned")
    value = manifest.model_dump(mode="json")
    value["files"][0]["path"] = "../config.json"

    with pytest.raises(ValueError, match="relative and contained"):
        ModelManifest.from_mapping(value, "memory://models/draft")


def test_policy_manifest_without_tokenizer_mode_remains_compatible(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file({"weight": np.arange(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    manifest = snapshot_model_manifest(tmp_path, model_id="snowball/policy", revision="pinned")
    value = manifest.model_dump(mode="json")
    value.pop("tokenizer_mode")

    assert ModelManifest.from_mapping(value, "memory://models/policy") == manifest
