# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged external pins must match the fork descriptors they are generated from."""

import importlib.util
import json
from pathlib import Path

import pytest
from marin.external_dependencies import VLLM_GPU_RELEASE
from rigging.config_discovery import find_project_root


def _workspace_root() -> Path:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to compare against")
    return root


def _update_external():
    """Load config/update-external.py, which is a standalone script rather than a package module."""
    path = _workspace_root() / "config" / "update-external.py"
    spec = importlib.util.spec_from_file_location("update_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _promoted_manifest() -> dict:
    return {
        "release": {
            "status": "released",
            "tag": "marin-vllm-gpu-20260101-abcdef012345",
            "repository": "marin-community/vllm",
        },
        "validation": {"status": "passed"},
        "source": {"fork_commit": "a" * 40},
        "distribution": {"version": "0.0.0.dev20260101+marin.abcdef012345"},
        "abi": {"cuda_variant": "cu130"},
        "platforms": [
            {
                "architecture": "x86_64",
                "sm_targets": ["9.0"],
                "wheel": {
                    "filename": "vllm-0.0.0.dev20260101+marin.abcdef012345-cp38-abi3-manylinux_2_28_x86_64.whl",
                    "sha256": "b" * 64,
                },
            },
        ],
    }


def _tpu_candidate() -> tuple[dict, dict]:
    tag = "marin-vllm-tpu-candidate-test"
    prefix = f"https://github.com/marin-community/vllm/releases/download/{tag}/"
    source = {
        "vllm": {"repository": "marin-community/vllm", "commit": "a" * 40},
        "tpu-inference": {"repository": "marin-community/tpu-inference", "commit": "b" * 40},
    }
    workflow = {
        "commit": "c" * 40,
        "run_url": "https://github.com/marin-community/vllm/actions/runs/123",
    }
    index = {
        "filename": "marin-vllm-tpu-index-dddddddddddddddd.html",
        "url": prefix + "marin-vllm-tpu-index-dddddddddddddddd.html",
        "sha256": "d" * 64,
    }
    wheels = {
        "vllm": {"filename": "vllm-test.whl", "url": prefix + "vllm-test.whl", "sha256": "e" * 64},
        "tpu-inference": {
            "filename": "tpu_inference-test.whl",
            "url": prefix + "tpu_inference-test.whl",
            "sha256": "f" * 64,
        },
    }
    packages = [
        {
            "distribution": distribution,
            "version": version,
            "wheel": wheel,
        }
        for distribution, version, wheel in (
            ("vllm", "0.20.1rc1.dev0+marin.aaaaaaaaaaaa.tpu", wheels["vllm"]),
            ("tpu-inference", "0.26.0+marin.bbbbbbbbbbbb", wheels["tpu-inference"]),
        )
    ]
    manifest = {
        "release": {"status": "candidate", "tag": tag, "repository": "marin-community/vllm"},
        "workflow": workflow,
        "source": source,
        "index": index,
        "compatibility": {
            "python_version": "3.12",
            "exclude_newer": "2026-08-12T00:00:00Z",
        },
        "packages": packages,
    }
    validation = {
        "candidate_tag": tag,
        "hardware": "v6e-8",
        "run_url": "https://github.com/marin-community/vllm/actions/runs/456",
    }
    return manifest, validation


def test_render_tpu_wheels_round_trips_a_qualified_pair(tmp_path):
    update_external = _update_external()
    manifest, validation = _tpu_candidate()
    path = tmp_path / "tpu_wheels.toml"

    path.write_text(update_external.render_tpu_wheels_toml(manifest, validation))
    release = update_external.load_vllm_tpu_release(path)
    packages = {package["distribution"]: package for package in manifest["packages"]}

    assert release.release_tag == manifest["release"]["tag"]
    assert release.vllm.sha256 == packages["vllm"]["wheel"]["sha256"]
    assert release.tpu_inference.sha256 == packages["tpu-inference"]["wheel"]["sha256"]


def test_render_tpu_wheels_rejects_a_result_for_another_candidate():
    update_external = _update_external()
    manifest, validation = _tpu_candidate()
    validation["candidate_tag"] = "marin-vllm-tpu-candidate-other"

    with pytest.raises(ValueError, match="changed candidate_tag"):
        update_external.render_tpu_wheels_toml(manifest, validation)


def test_render_tpu_wheels_accepts_the_promoted_copy_of_qualified_bytes(tmp_path):
    update_external = _update_external()
    manifest, validation = _tpu_candidate()
    candidate_tag = manifest["release"]["tag"]
    release_tag = "marin-vllm-tpu-20260101-aaaaaaaaaaaa-bbbbbbbbbbbb"
    prefix = f"https://github.com/marin-community/vllm/releases/download/{release_tag}/"
    manifest["release"] = {
        "status": "released",
        "tag": release_tag,
        "repository": "marin-community/vllm",
        "candidate_tag": candidate_tag,
    }
    for package in manifest["packages"]:
        package["wheel"]["url"] = prefix + package["wheel"]["filename"]
    manifest["validation"] = validation
    path = tmp_path / "tpu_wheels.toml"

    path.write_text(update_external.render_tpu_wheels_toml(manifest, validation))
    release = update_external.load_vllm_tpu_release(path)

    assert release.release_tag == release_tag
    assert release.vllm.url.startswith(prefix)
    assert release.tpu_inference.url.startswith(prefix)


def test_tpu_candidate_selection_rejects_another_source(tmp_path):
    update_external = _update_external()
    source_config = tmp_path / "tpu.toml"
    source_config.write_text(
        f"""[vllm]
repository = "https://github.com/marin-community/vllm.git"
commit = "{'a' * 40}"

[tpu-inference]
repository = "https://github.com/marin-community/tpu-inference.git"
commit = "{'b' * 40}"
"""
    )

    manifest, _ = _tpu_candidate()
    manifest["source"]["vllm"]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="selected vllm source"):
        update_external.validate_tpu_candidate_selection(manifest, source_config)


def test_gpu_release_pin_matches_its_descriptor():
    update_external = _update_external()
    descriptor = update_external.load_vllm_gpu_release(update_external.VLLM_GPU_RELEASE_CONFIG)

    assert VLLM_GPU_RELEASE.release_tag == descriptor.release_tag
    assert VLLM_GPU_RELEASE.source_commit == descriptor.source_commit
    assert VLLM_GPU_RELEASE.version == descriptor.version
    assert VLLM_GPU_RELEASE.torch_backend == descriptor.torch_backend
    generated = {(w.architecture, w.sm_targets, w.url, w.sha256) for w in VLLM_GPU_RELEASE.wheels}
    pinned = {(w.architecture, w.sm_targets, w.url, w.sha256) for w in descriptor.wheels}
    assert generated == pinned


def test_render_gpu_release_toml_reencodes_the_wheel_url_and_round_trips(tmp_path):
    update_external = _update_external()
    rendered = update_external.render_gpu_release_toml(_promoted_manifest())

    # The manifest carries the raw '+' filename; the pin must percent-encode it so the
    # loader's quote(version, safe='') URL check passes.
    assert "%2Bmarin.abcdef012345-" in rendered
    assert "+marin.abcdef012345-" not in rendered

    path = tmp_path / "gpu.toml"
    path.write_text(rendered)
    release = update_external.load_vllm_gpu_release(path)
    assert release.release_tag == "marin-vllm-gpu-20260101-abcdef012345"
    assert release.source_commit == "a" * 40
    assert release.torch_backend == "cu130"
    assert [wheel.architecture for wheel in release.wheels] == ["x86_64"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m["release"].__setitem__("status", "candidate"),
        lambda m: m["validation"].__setitem__("status", "pending"),
        lambda m: m["release"].__setitem__("repository", "someone-else/vllm"),
    ],
    ids=["unpromoted", "unvalidated", "foreign-repository"],
)
def test_render_gpu_release_toml_refuses_an_unpromoted_manifest(mutation):
    update_external = _update_external()
    manifest = _promoted_manifest()
    mutation(manifest)
    with pytest.raises(ValueError):
        update_external.render_gpu_release_toml(manifest)


def test_promote_gpu_release_keeps_the_pin_when_the_rendered_wheel_fails_validation(tmp_path, monkeypatch):
    # A manifest can clear the render-time status/repository gate yet still carry a wheel
    # invariant (here a malformed SHA-256) that only the loader rejects. The existing pin
    # must survive that failure rather than be overwritten with an invalid descriptor.
    update_external = _update_external()
    pin = tmp_path / "gpu.toml"
    original = 'release_tag = "keep-me"\n'
    pin.write_text(original)
    monkeypatch.setattr(update_external, "VLLM_GPU_RELEASE_CONFIG", pin)

    manifest = _promoted_manifest()
    manifest["platforms"][0]["wheel"]["sha256"] = "not-a-sha"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        update_external.promote_gpu_release(manifest_path)
    assert pin.read_text() == original
    assert not list(tmp_path.glob("gpu.*.toml.tmp"))
