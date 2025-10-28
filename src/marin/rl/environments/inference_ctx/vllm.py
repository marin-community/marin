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
import time
import numpy as np
import os
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput, CompletionOutput
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)

# Disable multiprocessing to have direct access to the model weights
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Init vLLM model with random weights to speed up bootstrap time, because
# model weights are synced from trainer later on
os.environ["JAX_RANDOM_WEIGHTS"] = "True"


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""
    
    def get_metrics(self) -> dict:
        """Get inference metrics."""
        metrics = {
            "total_tokens_generated": self.total_tokens_generated,
            "total_inference_time_sec": self.total_inference_time,
            "total_requests": self.total_requests,
        }
        if self.total_inference_time > 0:
            metrics["avg_tokens_per_second"] = self.total_tokens_generated / self.total_inference_time
        return metrics

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
        
        # Metrics for throughput tracking
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.total_requests = 0

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
        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            # NOTE(chris): We allow the override to take precedence over the default sampling params.
            max_tokens=max_tokens or self.sampling_params.max_tokens,
            stop=stop or self.sampling_params.stop,
            logprobs=1,
        )

        prompts_with_templates = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        start_time = time.time()
        outputs = self.llm.generate(prompts_with_templates, sampling_params)
        inference_time = time.time() - start_time
        
        # Track tokens generated
        tokens_generated = sum(
            len(output.outputs[0].token_ids) 
            for output in outputs
        ) if outputs else 0
        
        self.total_tokens_generated += tokens_generated
        self.total_inference_time += inference_time
        self.total_requests += 1
        
        if inference_time > 0:
            throughput = tokens_generated / inference_time
            logger.info(
                f"vLLM batch inference: {len(prompts)} prompts, "
                f"{tokens_generated} tokens in {inference_time:.2f}s, "
                f"throughput: {throughput:.1f} tokens/sec"
            )
        
        return outputs
