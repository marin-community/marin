# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packaged TPU-vLLM requirements must match the root workspace lock."""

import tomllib

import pytest
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from rigging.config_discovery import find_project_root


def _locked_requirement(distribution: str) -> str:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no Marin workspace checkout; nothing to compare against")
    lock = tomllib.loads((root / "uv.lock").read_text())
    (package,) = (package for package in lock["package"] if package["name"] == distribution)
    repository_with_query, _, commit = package["source"]["git"].partition("#")
    repository, _, _ = repository_with_query.partition("?")
    return f"{distribution} @ git+{repository}@{commit}"


def test_tpu_vllm_requirements_match_root_lock():
    assert VLLM_FORK_REQUIREMENT == _locked_requirement("vllm")
    assert TPU_INFERENCE_FORK_REQUIREMENT == _locked_requirement("tpu-inference")
