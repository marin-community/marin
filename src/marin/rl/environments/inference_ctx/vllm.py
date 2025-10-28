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
import os
import time
import numpy as np
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)

# Disable multiprocessing to have direct access to the model weights
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Init vLLM model with random weights to speed up bootstrap time, because
# model weights are synced from trainer later on
os.environ["JAX_RANDOM_WEIGHTS"] = "True"


@dataclass
class vLLMInferenceContextConfig:
    """Configuration for vLLM engine and sampling parameters."""

    model_name: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    sampling_params: SamplingParams


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""

    def __init__(
        self,
        inference_config: vLLMInferenceContextConfig,
    ):
        self.llm = LLM(
            model=inference_config.model_name,
            max_model_len=inference_config.max_model_len,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = inference_config.model_name
        self.sampling_params = inference_config.sampling_params

    def tokenize_prompt(self, prompt: str, choice: Choice) -> np.ndarray:
        """Tokenize the prompt with the choice's prompt token IDs.

        NOTE(chris): This is a hack to get the prompt token IDs the same since
        vLLM is injecting an extra BOS token at the start of the prompt.
        This is a known issue documented here:
        https://github.com/vllm-project/vllm/issues/27486
        """
        return np.array(choice.prompt_token_ids, dtype=np.int32)

    def _convert_vllm_to_openai(self, request_output: RequestOutput) -> ChatCompletion:
        """Convert vLLM RequestOutput to OpenAI ChatCompletion format."""
        choices = []
        for output_idx, output in enumerate(request_output.outputs):
            # Convert logprobs
            logprobs_content = []
            for token_id, logprob_dict in zip(output.token_ids, output.logprobs, strict=False):
                # Get the token string
                token = self.tokenizer.convert_ids_to_tokens(token_id)

                # Get the logprob for the selected token
                selected_logprob = None
                top_logprobs = []

                for tid, logprob_obj in logprob_dict.items():
                    if logprob_obj.rank == 1:
                        selected_logprob = logprob_obj.logprob

                    top_logprobs.append(
                        TopLogprob(
                            token=self.tokenizer.convert_ids_to_tokens(tid),
                            logprob=logprob_obj.logprob,
                            bytes=None,
                        )
                    )

                logprobs_content.append(
                    ChatCompletionTokenLogprob(
                        token=token, logprob=selected_logprob, bytes=None, top_logprobs=top_logprobs
                    )
                )

            choice = Choice(
                finish_reason=output.finish_reason or "stop",
                index=output_idx,
                logprobs=ChoiceLogprobs(content=logprobs_content),
                message=ChatCompletionMessage(
                    content=output.text,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )

            # Attach the prompt token IDs as a custom attribute to the choice since
            # we need this since vLLM is injecting an extra BOS token at the start
            # of the prompt.
            choice.prompt_token_ids = request_output.prompt_token_ids
            choices.append(choice)

        # Create usage information
        usage = CompletionUsage(
            completion_tokens=sum(len(output.token_ids) for output in request_output.outputs),
            prompt_tokens=len(request_output.prompt_token_ids) if request_output.prompt_token_ids else 0,
            total_tokens=(len(request_output.prompt_token_ids) if request_output.prompt_token_ids else 0)
            + sum(len(output.token_ids) for output in request_output.outputs),
        )

        return ChatCompletion(
            id=request_output.request_id,
            choices=choices,
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            usage=usage,
        )

    def batch_completions(
        self,
        prompts: list[str],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[ChatCompletion]:
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

        outputs = self.llm.generate(prompts_with_templates, sampling_params)

        # Convert vLLM outputs to OpenAI ChatCompletion format
        return [self._convert_vllm_to_openai(output) for output in outputs]
