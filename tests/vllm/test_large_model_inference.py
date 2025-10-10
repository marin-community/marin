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


@pytest.mark.gcp
@pytest.mark.skipif(
    os.getenv("TPU_CI") != "true" or os.getenv("SLOW_TEST") != "true",
    reason="Skip this test if not running with a TPU in CI or if we don't want to run slow tests.",
)
def test_local_llm_inference(gcsfuse_mount_llama_70b_model_path):
    ray.get(_test_llm_func.remote(gcsfuse_mount_llama_70b_model_path))
