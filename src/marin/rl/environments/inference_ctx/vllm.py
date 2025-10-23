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

import logging
import numpy as np
import os
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        sampling_params: SamplingParams,
    ):

        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = sampling_params

    def response_tokens_from_choice(self, choice: CompletionOutput) -> np.ndarray:
        """Extract token IDs with BPE round-trip."""
        return np.array(choice.token_ids, dtype=np.int32)

    def logprobs_from_choice(self, choice: CompletionOutput) -> np.ndarray:
        """Extract logprobs array."""
        logprobs = []
        for logprob_dict in choice.logprobs:
            for _, logprob in logprob_dict.items():
                if (
                    logprob.rank == 1
                ):  # Only support taking the top logprob since multiple tokens can have the same logprob
                    logprobs.append(logprob.logprob)

        return np.array(logprobs, dtype=np.float32)

    def batch_completions(
        self,
        prompts: list[str],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[RequestOutput]:
        """Batch completions from the inference server."""
        # TODO(chris): allow the override of sampling params for each lessonconfig
        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            max_tokens=max_tokens or self.sampling_params.max_tokens,
            # NOTE(chris): This is a bandaid patch because envs don't usually pass in the max token, so we need
            # to do this or else VLLM will fail.
            stop=stop or self.sampling_params.stop,
            logprobs=1,
        )

        prompts_with_templates = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        outputs = self.llm.generate(prompts_with_templates, sampling_params)
        return outputs
