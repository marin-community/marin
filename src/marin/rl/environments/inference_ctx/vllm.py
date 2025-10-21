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
from vllm.outputs import RequestOutput, CompletionOutput
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""

    def __init__(self, model_name: str, max_model_len: int, tensor_parallel_size: int):
        from vllm import LLM

        self.llm = LLM(model=model_name, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = self.llm.get_tokenizer()

    def response_tokens_from_choice(self, choice: CompletionOutput) -> np.ndarray:
        """Extract token IDs with BPE round-trip."""
        return np.array(choice.token_ids, dtype=np.int32)

    def logprobs_from_choice(self, choice: CompletionOutput) -> np.ndarray:
        """Extract logprobs array."""
        logprobs = []
        for logprob_dict in choice.logprobs:
            for _, logprob in logprob_dict.items():
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
        from vllm import SamplingParams

        prompts_with_templates = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            seed=None,
            logprobs=1,
        )
        outputs = self.llm.generate(prompts_with_templates, sampling_params)
        return outputs
