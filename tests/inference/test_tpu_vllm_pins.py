# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The vendored TPU-vLLM fork pins must match the root pyproject's uv sources.

``marin.inference.tpu_vllm_pins`` vendors the fork revisions the checkout-free TPU serve
path uses (where the root pyproject is unavailable). This test fails if they drift from
``[tool.uv.sources]`` — so a fork refresh has to update both, keeping the isolated serve
path on the same revisions as the in-checkout workspace lock.
"""

import tomllib

import pytest
from marin.inference import tpu_vllm_pins
from rigging.config_discovery import find_project_root


def _root_uv_sources() -> dict:
    root = find_project_root(__file__)
    if root is None:
        pytest.skip("no marin workspace checkout (published-package install); nothing to compare against")
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    return pyproject["tool"]["uv"]["sources"]


def test_tpu_vllm_pins_match_pyproject():
    sources = _root_uv_sources()
    assert sources["vllm"] == {"git": tpu_vllm_pins.VLLM_FORK_URL, "rev": tpu_vllm_pins.VLLM_FORK_REV}
    assert sources["tpu-inference"] == {
        "git": tpu_vllm_pins.TPU_INFERENCE_FORK_URL,
        "rev": tpu_vllm_pins.TPU_INFERENCE_FORK_REV,
    }
