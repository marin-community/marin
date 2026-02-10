# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether vLLM can generate simple completions"""


import pytest

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import resolve_model_name_or_path

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pytest.skip("vLLM is not installed", allow_module_level=True)


def run_vllm_inference(model_path, **model_init_kwargs):
    llm = LLM(model=model_path, **model_init_kwargs)

    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.7,
    )

    generated_texts = llm.generate(
        "Hello, how are you?",
        sampling_params=sampling_params,
    )

    return generated_texts


@pytest.mark.tpu_ci
def test_local_llm_inference():
    config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs={"enforce_eager": True, "max_model_len": 1024},
        generation_params={"max_tokens": 16},
    )
    model_name_or_path, config = resolve_model_name_or_path(config)
    run_vllm_inference(model_name_or_path, **config.engine_kwargs)
