# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test whether vLLM can generate simple completions"""


import shutil

import pytest

from marin.evaluation.evaluators.evaluator import ModelConfig

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
    model_path = config.ensure_downloaded("/tmp/test-llama-eval")
    run_vllm_inference(model_path, **config.engine_kwargs)
    shutil.rmtree("/tmp/test-llama-eval")


@pytest.mark.tpu_ci
def test_large_model_inference():
    gcsfuse_mount_llama_70b_model_path = "/opt/gcsfuse_mount/models/meta-llama--Llama-3-3-70B-Instruct"
    return run_vllm_inference(gcsfuse_mount_llama_70b_model_path, max_model_len=1024, tensor_parallel_size=8)
