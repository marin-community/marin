"""Test whether vLLM can generate simple completions on a large model."""

import os

import pytest
import ray

try:
    from tests.vllm.utils import run_vllm_inference
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)

from tests.conftest import large_model_engine_kwargs


@ray.remote(resources={"TPU-v6e-8-head": 1})
def _test_llm_func(model_path):
    return run_vllm_inference(model_path, **large_model_engine_kwargs)


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_local_llm_inference(gcsfuse_mount_llama_70b_model_path):
    ray.get(_test_llm_func.remote(gcsfuse_mount_llama_70b_model_path))
