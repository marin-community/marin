# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged TPU-vLLM requirements must match the fork descriptor they are generated from."""

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
