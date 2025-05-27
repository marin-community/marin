"""Test whether vLLM can generate simple completions"""

import os

import pytest
import ray

try:
    from tests.vllm.utils import run_vllm_inference
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)

from tests.conftest import SINGLE_GPU_CONFIG, TPU_V6E_8_WITH_HEAD_CONFIG


@TPU_V6E_8_WITH_HEAD_CONFIG.as_decorator()
def _test_llm_func_single_tpu(model_config):
    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")

    run_vllm_inference(model_path, **model_config.engine_kwargs)

    model_config.destroy()


@SINGLE_GPU_CONFIG.as_decorator()
def _test_llm_func_single_gpu(model_config):
    model_path = model_config.ensure_downloaded("/tmp/test-llama-eval")
    run_vllm_inference(model_path, **model_config.engine_kwargs)
    model_config.destroy()


@pytest.mark.skipif(
    os.getenv("TPU_CI") != "true" and os.getenv("GPU_CI") != "true",
    reason="Skip this test if not running with a TPU or GPU in CI.",
)
def test_local_llm_inference(ray_cluster, model_config):
    if os.getenv("TPU_CI") == "true":
        test_fn = _test_llm_func_single_tpu
    elif os.getenv("GPU_CI") == "true":
        test_fn = _test_llm_func_single_gpu

    ray.get(test_fn.remote(model_config))


@pytest.mark.skipif(
    os.getenv("TPU_CI") != "true",
    reason="Skip this test if not running with a TPU in CI.",
)
def test_local_llm_inference_tpu(ray_cluster, model_config):
    ray.get(_test_llm_func_single_tpu.remote(model_config))


@pytest.mark.skipif(
    os.getenv("GPU_CI") != "true",
    reason="Skip this test if not running with a GPU in CI.",
)
def test_local_llm_inference_gpu(ray_cluster, model_config):
    ray.get(_test_llm_func_single_gpu.remote(model_config))
