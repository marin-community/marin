# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the external TPU-vLLM descriptors."""

import runpy
import tomllib

import pytest
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from rigging.config_discovery import find_project_root


def _descriptor_requirement(name: str) -> str:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to compare against")
    config = tomllib.loads((root / "config" / "external" / "vllm" / "tpu-forks.toml").read_text())
    entry = config[name]
    return f"{name} @ git+{entry['repository']}@{entry['commit']}"


def test_tpu_vllm_requirements_match_fork_descriptor():
    assert VLLM_FORK_REQUIREMENT == _descriptor_requirement("vllm")
    assert TPU_INFERENCE_FORK_REQUIREMENT == _descriptor_requirement("tpu-inference")


@pytest.mark.parametrize("field", ["source_commit", "url", "sha256"])
def test_tpu_release_descriptor_rejects_drift(tmp_path, field: str):
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to validate")
    descriptor = (root / "config" / "external" / "vllm" / "tpu-release.toml").read_text()
    vllm = tomllib.loads(descriptor)["vllm"]
    replacement = {
        "source_commit": "0" * 40,
        "url": "https://example.com/vllm.whl",
        "sha256": "bad",
    }[field]
    mutated_descriptor = descriptor.replace(f'{field} = "{vllm[field]}"', f'{field} = "{replacement}"', 1)
    assert mutated_descriptor != descriptor
    mutated_path = tmp_path / "tpu-release.toml"
    mutated_path.write_text(mutated_descriptor)
    load_release = runpy.run_path(str(root / "config" / "update-external.py"))["load_vllm_tpu_release"]

    with pytest.raises(ValueError):
        load_release(mutated_path)
