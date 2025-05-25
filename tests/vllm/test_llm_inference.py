"""Test whether vLLM can generate simple completions"""

import os

import pytest
import ray

try:
    from tests.vllm.utils import run_vllm_inference
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)


@ray.remote(resources={"TPU-v6e-8-head": 1})
def _test_llm_func(model_config):
    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")

    run_vllm_inference(model_path, **model_config.engine_kwargs)

    model_config.destroy()


@pytest.mark.skipif(os.getenv("TPU_CI") != "true", reason="Skip this test if not running with a TPU in CI.")
def test_local_llm_inference(ray_cluster, model_config):
    ray.get(_test_llm_func.remote(model_config))
