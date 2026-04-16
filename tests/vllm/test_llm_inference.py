# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether vLLM can generate simple completions"""


import pytest

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
    run_vllm_inference(
        "gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        enforce_eager=True,
        max_model_len=1024,
    )
