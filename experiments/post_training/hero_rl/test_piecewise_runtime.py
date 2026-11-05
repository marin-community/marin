# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for the disposable Hero wheel overlay."""

import hashlib
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments.post_training.hero_rl import piecewise_runtime as runtime


@pytest.fixture
def wheel_environment(tmp_path, monkeypatch):
    # Staging runs before imports in a fresh task. Other suites may import vLLM
    # in this process; restore that task boundary for the temporary wheel.
    for name in tuple(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name)
    workspace = tmp_path / "workspace"
    payload = workspace / runtime.PAYLOAD_DIRECTORY
    payload.mkdir(parents=True)
    cache = tmp_path / "shared-wheel-cache" / "vllm"
    cache.mkdir(parents=True)
    site = tmp_path / "environment" / "lib" / "site-packages"
    site.mkdir(parents=True)
    overlay = []
    for relative in ("config/compilation.py", "model_executor/models/config.py", "model_executor/models/grugmoe.py"):
        target = cache / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# Original wheel: {relative}\n")
        changed = payload / relative.replace("/", "-")
        changed.write_text(f"# Experimental Python source: {relative}\n")
        overlay.append(
            {
                "path": "vllm/" + relative,
                "payload_path": changed.name,
                "base_sha256": runtime.digest(target),
                "patched_sha256": runtime.digest(changed),
            }
        )
    (cache / "_C.abi3.so").write_bytes(b"Immutable compiled CUDA extension")
    (cache / "untouched.py").write_text("# Unmodified Python module\n")
    archive_path = tmp_path / "vllm-1.0-py3-none-any.whl"
    with ZipFile(archive_path, "w") as archive:
        for path in cache.rglob("*"):
            if path.is_file():
                archive.write(path, "vllm/" + str(path.relative_to(cache)))
    wheel = {"version": "1.0", "url": archive_path.as_uri(), "hash": "sha256:" + runtime.digest(archive_path)}
    direct = {"url": wheel["url"], "archive_info": {}}
    for name, version in {"vllm": wheel["version"], "torch": "2.13.0+cu132"}.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if name == "vllm":
            (info / "direct_url.json").write_text(json.dumps(direct))
    manifest = {
        "wheel": wheel,
        "overlay": overlay,
        "expected_core": {"torch": "2.13.0+cu132"},
        "managed_inputs": {},
        "source_revision": "pinned-vllm-source",
        "patch_sha256": "b" * 64,
        "qualification": "EXPERIMENTAL",
    }
    content = json.dumps(manifest).encode()
    (payload / "manifest.json").write_bytes(content)
    monkeypatch.setattr(runtime, "MANIFEST_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.syspath_prepend(str(site))
    return workspace, cache, site, manifest


def link_package(cache: Path, entry: Path, layout: str) -> None:
    if layout == "directory":
        entry.symlink_to(cache, target_is_directory=True)
        return
    entry.mkdir()
    for source in cache.rglob("*"):
        target = entry / source.relative_to(cache)
        if source.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.symlink_to(source)


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("layout", ["directory", "files"])
def test_overlay_preserves_shared_wheel_and_native_extensions(wheel_environment, tmp_path, layout):
    workspace, cache, site, manifest = wheel_environment
    entry = site / "vllm"
    link_package(cache, entry, layout)
    before = tree_bytes(cache)
    receipt_path = tmp_path / "task-output" / "hero-piecewise-runtime.json"

    runtime.stage_runtime(workspace, site.parent.parent, receipt_path)

    assert tree_bytes(cache) == before
    assert not entry.is_symlink()
    assert not any(path.is_symlink() for path in entry.rglob("*"))
    assert (entry / "_C.abi3.so").read_bytes() == before["_C.abi3.so"]
    assert (entry / "untouched.py").read_bytes() == before["untouched.py"]
    for item in manifest["overlay"]:
        assert (entry / Path(item["path"]).relative_to("vllm")).read_bytes() == (
            workspace / runtime.PAYLOAD_DIRECTORY / item["payload_path"]
        ).read_bytes()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["extensions_before"] == receipt["extensions_after"]
    assert receipt["wheel_direct_url"]["archive_info"] == {}
    assert receipt["wheel_archive_sha256"] == manifest["wheel"]["hash"].removeprefix("sha256:")
    assert receipt["wheel_extensions"] == receipt["extensions_before"]
    assert receipt["manifest_sha256"] == runtime.MANIFEST_SHA256
    assert receipt["core_dependencies"] == {"torch": "2.13.0+cu132"}


@pytest.mark.parametrize("corruption", ["payload", "installed-source", "wheel-receipt"])
def test_rejected_overlay_keeps_original_package_and_cache(wheel_environment, tmp_path, corruption):
    workspace, cache, site, manifest = wheel_environment
    entry = site / "vllm"
    link_package(cache, entry, "directory")
    if corruption == "payload":
        (workspace / runtime.PAYLOAD_DIRECTORY / manifest["overlay"][-1]["payload_path"]).write_text("Changed payload")
    elif corruption == "installed-source":
        (cache / "model_executor/models/grugmoe.py").write_text("Wrong wheel source")
    else:
        direct = site / "vllm-1.0.dist-info/direct_url.json"
        record = json.loads(direct.read_text())
        record["archive_info"]["hashes"] = {"sha256": "c" * 64}
        direct.write_text(json.dumps(record))
    before = tree_bytes(cache)
    receipt = tmp_path / "receipt.json"

    with pytest.raises(ValueError):
        runtime.stage_runtime(workspace, site.parent.parent, receipt)

    assert entry.is_symlink() and entry.resolve() == cache
    assert tree_bytes(cache) == before
    assert not receipt.exists()
    assert not entry.with_name("vllm.before-hero-piecewise").exists()


@pytest.mark.parametrize(
    "corruption", ["wheel-bytes", "installed-native-library", "installed-python-library", "extra-native-library"]
)
def test_missing_receipt_hash_still_requires_wheel_and_binary_identity(wheel_environment, tmp_path, corruption):
    workspace, cache, site, _ = wheel_environment
    entry = site / "vllm"
    link_package(cache, entry, "directory")
    if corruption == "wheel-bytes":
        (tmp_path / "vllm-1.0-py3-none-any.whl").write_bytes(b"Different wheel at the same URL")
    elif corruption == "installed-native-library":
        (cache / "_C.abi3.so").write_bytes(b"Different installed native library")
    elif corruption == "installed-python-library":
        (cache / "untouched.py").write_text("Different installed Python module")
    else:
        (cache / "untracked.so").write_bytes(b"Extra installed native library")
    before = tree_bytes(cache)
    receipt = tmp_path / "receipt.json"

    with pytest.raises(ValueError, match="wheel"):
        runtime.stage_runtime(workspace, site.parent.parent, receipt)

    assert entry.is_symlink() and entry.resolve() == cache
    assert tree_bytes(cache) == before
    assert not receipt.exists()


@pytest.mark.parametrize(
    "archive_info",
    [{"hash": "sha256=" + "0" * 64}, {"hash": "sha512=" + "0" * 128}, {"hashes": {"sha512": "0" * 128}}],
)
def test_conflicting_receipt_hashes_preserve_original_package(wheel_environment, tmp_path, archive_info):
    workspace, cache, site, _ = wheel_environment
    entry = site / "vllm"
    link_package(cache, entry, "directory")
    direct_path = site / "vllm-1.0.dist-info/direct_url.json"
    direct = json.loads(direct_path.read_text())
    direct["archive_info"] = archive_info
    direct_path.write_text(json.dumps(direct))
    before = tree_bytes(cache)
    receipt = tmp_path / "receipt.json"

    with pytest.raises(ValueError):
        runtime.stage_runtime(workspace, site.parent.parent, receipt)

    assert entry.is_symlink() and entry.resolve() == cache
    assert tree_bytes(cache) == before
    assert not receipt.exists()
    assert not entry.with_name("vllm.before-hero-piecewise").exists()


def test_failed_package_publish_restores_original_entry(wheel_environment, monkeypatch, tmp_path):
    workspace, cache, site, _ = wheel_environment
    entry = site / "vllm"
    link_package(cache, entry, "directory")
    before = tree_bytes(cache)
    replace = os.replace

    def fail_publish(source, destination):
        if Path(source).parent.name.startswith(".hero-piecewise-") and Path(destination) == entry:
            raise OSError("Simulated package publication failure")
        return replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publication failure"):
        runtime.stage_runtime(workspace, site.parent.parent, tmp_path / "receipt.json")

    assert entry.is_symlink() and entry.resolve() == cache
    assert tree_bytes(cache) == before
    assert not entry.with_name("vllm.before-hero-piecewise").exists()
