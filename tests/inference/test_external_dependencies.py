# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged external pins must match the fork descriptors they are generated from."""

import importlib.util
import tomllib
from pathlib import Path

import pytest
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT, VLLM_GPU_RELEASE
from rigging.config_discovery import find_project_root


def _workspace_root() -> Path:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to compare against")
    return root


def _descriptor_requirement(name: str) -> str:
    config = tomllib.loads((_workspace_root() / "config" / "external" / "vllm" / "tpu-forks.toml").read_text())
    entry = config[name]
    return f"{name} @ git+{entry['repository']}@{entry['commit']}"


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


def test_tpu_vllm_requirements_match_fork_descriptor():
    assert VLLM_FORK_REQUIREMENT == _descriptor_requirement("vllm")
    assert TPU_INFERENCE_FORK_REQUIREMENT == _descriptor_requirement("tpu-inference")


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

    path = tmp_path / "gpu-release.toml"
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
