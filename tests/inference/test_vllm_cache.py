# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from marin.inference.config import VllmCompilationCacheMode
from marin.inference.vllm_cache import (
    ARCHIVE_FILENAME,
    GENERATIONS_DIR,
    LATEST_FILENAME,
    MANIFEST_FILENAME,
    VllmCompilationCache,
    VllmCompileIdentity,
)

_CACHE_DIR_ENVIRONMENT_VARIABLES = (
    "JAX_COMPILATION_CACHE_DIR",
    "VLLM_XLA_CACHE_PATH",
    "VLLM_CACHE_ROOT",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
)


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


def _local_cache_dir(cache: VllmCompilationCache) -> Path:
    return Path(cache.environment()["JAX_COMPILATION_CACHE_DIR"]).parent


def _only_remote_cache_dir(local_marin_prefix: Path) -> Path:
    cache_dirs = list((local_marin_prefix / "tmp" / "vllm-compilation-cache").glob("v*/*"))
    assert len(cache_dirs) == 1
    return cache_dirs[0]


def _write_compiler_cache_entries(cache: VllmCompilationCache) -> dict[str, bytes]:
    expected: dict[str, bytes] = {}
    environment = cache.environment()
    for variable in _CACHE_DIR_ENVIRONMENT_VARIABLES:
        payload = f"compiled by {variable}".encode()
        entry = Path(environment[variable]) / f"{variable.lower()}.bin"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(payload)
        expected[variable] = payload
    Path(environment["TRITON_CACHE_DIR"], "triton_cache_dir.bin").chmod(0o755)
    return expected


def _assert_compiler_cache_entries(cache: VllmCompilationCache, expected: dict[str, bytes]) -> None:
    environment = cache.environment()
    for variable, payload in expected.items():
        entry = Path(environment[variable]) / f"{variable.lower()}.bin"
        assert entry.read_bytes() == payload


def test_disabled_cache_preserves_caller_environment(local_marin_prefix: Path) -> None:
    existing = {
        "JAX_COMPILATION_CACHE_DIR": "/caller/xla",
        "VLLM_CACHE_ROOT": "/caller/vllm",
    }

    cache = _prepare_cache(environment=existing, mode=VllmCompilationCacheMode.DISABLED)
    assert cache.environment() == existing
    cache.publish()
    cache.close()

    assert not (local_marin_prefix / "tmp" / "vllm-compilation-cache").exists()


def test_cache_round_trip_restores_every_compiler_cache(local_marin_prefix: Path) -> None:
    first = _prepare_cache(
        environment={
            "JAX_COMPILATION_CACHE_DIR": "/caller/xla",
            "VLLM_CACHE_ROOT": "/caller/vllm",
            "TRITON_CACHE_DIR": "/caller/triton",
        }
    )
    expected = _write_compiler_cache_entries(first)
    first_local_dir = _local_cache_dir(first)
    first.publish()
    remote_cache_dir = _only_remote_cache_dir(local_marin_prefix)
    first.close()
    assert not first_local_dir.exists()

    generations = remote_cache_dir / GENERATIONS_DIR
    assert len(list(generations.iterdir())) == 1

    second = _prepare_cache()
    _assert_compiler_cache_entries(second, expected)
    restored_triton = Path(second.environment()["TRITON_CACHE_DIR"], "triton_cache_dir.bin")
    assert os.stat(restored_triton).st_mode & 0o777 == 0o755

    second.publish()
    second.close()
    assert len(list(generations.iterdir())) == 1


def test_corrupt_generation_is_a_cache_miss(local_marin_prefix: Path) -> None:
    first = _prepare_cache()
    entry = Path(first.environment()["JAX_COMPILATION_CACHE_DIR"]) / "entry.bin"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"compiled")
    first.publish()
    remote_cache_dir = _only_remote_cache_dir(local_marin_prefix)
    first.close()

    latest = json.loads((remote_cache_dir / LATEST_FILENAME).read_text())
    manifest_path = remote_cache_dir / GENERATIONS_DIR / latest["generation_id"] / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    second = _prepare_cache()
    try:
        assert not (Path(second.environment()["JAX_COMPILATION_CACHE_DIR"]) / "entry.bin").exists()
    finally:
        second.close()


def test_unsafe_archive_is_a_cache_miss(
    local_marin_prefix: Path,
    tmp_path: Path,
) -> None:
    first = _prepare_cache()
    seed_entry = Path(first.environment()["JAX_COMPILATION_CACHE_DIR"]) / "entry.bin"
    seed_entry.parent.mkdir(parents=True)
    seed_entry.write_bytes(b"seed")
    first.publish()
    remote_cache_dir = _only_remote_cache_dir(local_marin_prefix)
    first.close()

    archive_path = tmp_path / ARCHIVE_FILENAME
    payload = b"outside"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("../escaped", payload)

    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    generation_id = archive_digest
    old_latest = json.loads((remote_cache_dir / LATEST_FILENAME).read_text())
    old_manifest_path = remote_cache_dir / GENERATIONS_DIR / old_latest["generation_id"] / MANIFEST_FILENAME
    manifest = json.loads(old_manifest_path.read_text())
    manifest.update(
        {
            "generation_id": generation_id,
            "archive_sha256": archive_digest,
            "archive_size_bytes": archive_path.stat().st_size,
            "source_size_bytes": len(payload),
            "entry_count": 1,
        }
    )
    generation = remote_cache_dir / GENERATIONS_DIR / generation_id
    generation.mkdir(parents=True)
    remote_archive = generation / ARCHIVE_FILENAME
    remote_archive.write_bytes(archive_path.read_bytes())
    (generation / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    (remote_cache_dir / LATEST_FILENAME).write_text(json.dumps({"generation_id": generation_id}))

    second = _prepare_cache()
    try:
        assert not (tmp_path / "escaped").exists()
        assert not any(_local_cache_dir(second).rglob("*"))
    finally:
        second.close()


def test_cache_isolated_by_launcher_and_compile_identity(local_marin_prefix: Path) -> None:
    first = _prepare_cache()
    marker = Path(first.environment()["VLLM_CACHE_ROOT"]) / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("compiled")
    first.publish()
    first.close()

    same = _prepare_cache()
    assert (Path(same.environment()["VLLM_CACHE_ROOT"]) / "marker").read_text() == "compiled"
    same.close()

    other_launcher = _prepare_cache(launcher_identity="cuda-vllm==0.25.1")
    assert not (Path(other_launcher.environment()["VLLM_CACHE_ROOT"]) / "marker").exists()
    other_launcher.close()

    other_args = _prepare_cache(extra_cli_args=("--dtype", "float16"))
    assert not (Path(other_args.environment()["VLLM_CACHE_ROOT"]) / "marker").exists()
    other_args.close()


def test_chat_template_cache_identity_uses_content(
    local_marin_prefix: Path,
    tmp_path: Path,
) -> None:
    first_template = tmp_path / "first-template.jinja"
    second_template = tmp_path / "second-template.jinja"
    changed_template = tmp_path / "changed-template.jinja"
    first_template.write_text("{{ messages }}")
    second_template.write_text("{{ messages }}")
    changed_template.write_text("{{ messages[-1] }}")

    def prepare(template: Path) -> VllmCompilationCache:
        return VllmCompilationCache.prepare(
            launcher_identity="workspace",
            compile_identity=VllmCompileIdentity.from_vllm_args(
                model_name_or_path="test/model",
                extra_cli_args=("--chat-template", str(template)),
            ),
            environment={},
        )

    first = prepare(first_template)
    marker = Path(first.environment()["VLLM_CACHE_ROOT"]) / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("compiled")
    first.publish()
    first.close()

    same_content = prepare(second_template)
    assert (Path(same_content.environment()["VLLM_CACHE_ROOT"]) / "marker").read_text() == "compiled"
    same_content.close()

    changed_content = prepare(changed_template)
    assert not (Path(changed_content.environment()["VLLM_CACHE_ROOT"]) / "marker").exists()
    changed_content.close()


def test_invalid_cache_entry_skips_publication(
    local_marin_prefix: Path,
) -> None:
    cache = _prepare_cache()
    outside = local_marin_prefix / "outside"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"not cache content")
    (_local_cache_dir(cache) / "unsafe-link").symlink_to(outside)

    cache.publish()
    cache.close()

    assert not list((local_marin_prefix / "tmp" / "vllm-compilation-cache").rglob(LATEST_FILENAME))
