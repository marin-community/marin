# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from marin.inference.config import VllmCompilationCacheMode
from marin.inference.vllm_cache import VllmCompilationCache, VllmCompileIdentity


@pytest.fixture
def local_marin_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefix = tmp_path / "marin"
    monkeypatch.setenv("MARIN_PREFIX", str(prefix))
    return prefix


def _prepare_cache(
    *,
    launcher_identity: str = "workspace",
    model: str = "test/model",
    extra_cli_args: tuple[str, ...] = (),
    environment: dict[str, str] | None = None,
    mode: VllmCompilationCacheMode = VllmCompilationCacheMode.MANAGED,
) -> VllmCompilationCache:
    return VllmCompilationCache.prepare(
        launcher_identity=launcher_identity,
        compile_identity=VllmCompileIdentity(
            model_name_or_path=model,
            extra_cli_args=extra_cli_args,
        ),
        environment=environment or {},
        mode=mode,
    )


def _local_path(storage_url: str) -> Path:
    return Path(storage_url.removeprefix("file://"))


def test_managed_cache_owns_every_compiler_path(local_marin_prefix: Path) -> None:
    existing = {
        "UNRELATED": "kept",
        "JAX_COMPILATION_CACHE_DIR": "/caller/xla",
        "VLLM_CACHE_ROOT": "/caller/vllm",
        "TRITON_CACHE_DIR": "/caller/triton",
    }
    cache = _prepare_cache(environment=existing)
    try:
        expected_remote = local_marin_prefix / "tmp" / "vllm-compilation-cache" / "v3" / cache.namespace
        assert cache.remote_prefix == f"file://{expected_remote}"
        environment = cache.environment()
        assert environment["UNRELATED"] == "kept"
        assert environment["JAX_COMPILATION_CACHE_DIR"] == str(cache.root / "xla")
        assert environment["VLLM_XLA_CACHE_PATH"] == str(cache.root / "xla")
        assert environment["JAX_ENABLE_COMPILATION_CACHE"] == "1"
        assert environment["VLLM_CACHE_ROOT"] == str(cache.root / "vllm")
        assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(cache.root / "torchinductor")
        assert environment["TRITON_CACHE_DIR"] == str(cache.root / "triton")
        assert environment["CUDA_CACHE_PATH"] == str(cache.root / "cuda")
        assert environment["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "-1"
        assert environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "-1"
        assert environment["VLLM_COMPILE_CACHE_SAVE_FORMAT"] == "binary"
    finally:
        root = cache.root
        cache.close()
    assert not root.exists()


def test_disabled_cache_preserves_caller_environment(local_marin_prefix: Path) -> None:
    existing = {
        "JAX_COMPILATION_CACHE_DIR": "/caller/xla",
        "VLLM_CACHE_ROOT": "/caller/vllm",
    }

    cache = _prepare_cache(environment=existing, mode=VllmCompilationCacheMode.DISABLED)
    assert cache.environment() == existing
    cache.publish_after_ready()
    cache.close()

    assert not (local_marin_prefix / "tmp" / "vllm-compilation-cache").exists()


def test_cache_round_trip_preserves_contents_and_skips_unchanged_publish(local_marin_prefix: Path) -> None:
    first = _prepare_cache()
    xla_entry = Path(first.environment()["JAX_COMPILATION_CACHE_DIR"]) / "sampling" / "entry.bin"
    triton_entry = Path(first.environment()["TRITON_CACHE_DIR"]) / "kernel.so"
    xla_entry.parent.mkdir(parents=True)
    triton_entry.parent.mkdir(parents=True)
    xla_entry.write_bytes(b"xla executable")
    triton_entry.write_bytes(b"triton executable")
    triton_entry.chmod(0o755)
    first.publish_after_ready()
    remote_prefix = _local_path(first.remote_prefix)
    first.close()

    generations = remote_prefix / "generations"
    assert len(list(generations.iterdir())) == 1

    second = _prepare_cache()
    restored_xla = Path(second.environment()["JAX_COMPILATION_CACHE_DIR"]) / "sampling" / "entry.bin"
    restored_triton = Path(second.environment()["TRITON_CACHE_DIR"]) / "kernel.so"
    assert restored_xla.read_bytes() == b"xla executable"
    assert restored_triton.read_bytes() == b"triton executable"
    assert os.stat(restored_triton).st_mode & 0o777 == 0o755

    second.publish_after_ready()
    second.close()
    assert len(list(generations.iterdir())) == 1


def test_corrupt_generation_is_a_cache_miss(local_marin_prefix: Path) -> None:
    first = _prepare_cache()
    entry = first.root / "xla" / "entry.bin"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"compiled")
    first.publish_after_ready()
    remote_prefix = _local_path(first.remote_prefix)
    first.close()

    latest = json.loads((remote_prefix / "latest.json").read_text())
    manifest_path = remote_prefix / "generations" / latest["generation_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    second = _prepare_cache()
    try:
        assert not (second.root / "xla" / "entry.bin").exists()
    finally:
        second.close()


def test_unsafe_archive_is_a_cache_miss(
    local_marin_prefix: Path,
    tmp_path: Path,
) -> None:
    first = _prepare_cache()
    remote_prefix = _local_path(first.remote_prefix)
    namespace = first.namespace
    first.close()

    archive_path = tmp_path / "unsafe.tar.gz"
    payload = b"outside"
    with archive_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                info = tarfile.TarInfo("../escaped")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    generation_id = archive_digest
    generation = remote_prefix / "generations" / generation_id
    generation.mkdir(parents=True)
    remote_archive = generation / "cache.tar.gz"
    remote_archive.write_bytes(archive_path.read_bytes())
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "namespace": namespace,
                "namespace_identity": {
                    "schema_version": 3,
                    "launcher": "workspace",
                    "compile": {
                        "model_name_or_path": "test/model",
                        "extra_cli_args": [],
                    },
                },
                "generation_id": generation_id,
                "archive_sha256": archive_digest,
                "archive_size_bytes": remote_archive.stat().st_size,
                "source_size_bytes": len(payload),
                "entry_count": 1,
            }
        )
    )
    remote_prefix.mkdir(parents=True, exist_ok=True)
    (remote_prefix / "latest.json").write_text(json.dumps({"generation_id": generation_id}))

    second = _prepare_cache()
    try:
        assert not (tmp_path / "escaped").exists()
        assert not any(second.root.rglob("*"))
    finally:
        second.close()


def test_namespace_tracks_launcher_and_compile_identity(local_marin_prefix: Path) -> None:
    base = _prepare_cache()
    same = _prepare_cache()
    other_launcher = _prepare_cache(launcher_identity="cuda-vllm==0.25.1")
    other_args = _prepare_cache(extra_cli_args=("--dtype", "float16"))
    try:
        assert base.namespace == same.namespace
        assert base.namespace != other_launcher.namespace
        assert base.namespace != other_args.namespace
    finally:
        base.close()
        same.close()
        other_launcher.close()
        other_args.close()


def test_invalid_cache_entry_skips_publication(
    local_marin_prefix: Path,
) -> None:
    cache = _prepare_cache()
    outside = local_marin_prefix / "outside"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"not cache content")
    (cache.root / "unsafe-link").symlink_to(outside)
    remote_prefix = _local_path(cache.remote_prefix)

    cache.publish_after_ready()
    cache.close()

    assert not (remote_prefix / "latest.json").exists()
