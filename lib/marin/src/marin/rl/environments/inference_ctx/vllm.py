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

import gc
import logging
import os
import time
import re
import jax
import jax.numpy as jnp
import numpy as np
from enum import StrEnum
from dataclasses import dataclass
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from levanter.models.lm_model import LmHeadModel
from transformers import AutoTokenizer
from marin.rl.weight_utils import levanter_state_dict_to_nnx_state_on_cpu
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.environments.inference_ctx.inflight.worker import SyncVLLMWrapper
from marin.rl.environments.inference_ctx.vllm_utils import MODEL_MAPPINGS, MODEL_TRANSPOSE_KEYS
from marin.rl.environments.inference_ctx.render import Llama3Renderer, Qwen3Renderer, Renderer, Message

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import RequestOutputKind
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to use vLLM inference context.")
    LLM = None
    SamplingParams = None
    TokensPrompt = None
    RequestOutput = None
    RequestOutputKind = None

# Disable multiprocessing to have direct access to the model weights
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Init vLLM model with random weights to speed up bootstrap time, because
# model weights are synced from trainer later on
os.environ["JAX_RANDOM_WEIGHTS"] = "True"
# Skip jax precompile to speed up bootstrap time
os.environ["SKIP_JAX_PRECOMPILE"] = "1"


class InferenceMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


@dataclass
class vLLMInferenceContextConfig:
    """Configuration for vLLM engine and sampling parameters."""

    model_name: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    sampling_params: SamplingParams
    mode: InferenceMode = InferenceMode.SYNC


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""

    def __init__(
        self,
        inference_config: vLLMInferenceContextConfig,
    ):
        self.llm = self._get_llm_engine(inference_config)
        # Mesh for the weight transfer client should be on the CPU and then
        # in sync_weights function, we will reshard to the TPU
        # self.mesh = jax.make_mesh(
        #     (1, 1, 1),
        #     (ResourceAxis.DATA, ResourceAxis.REPLICA, ResourceAxis.MODEL),
        #     devices=jax.local_devices(backend="cpu")[:1],
        # )
        self.mesh = None
        self.axis_mapping = {}
        self.tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name)
        self.model_name = inference_config.model_name
        self.sampling_params = inference_config.sampling_params

        # Initialize the appropriate renderer based on model type
        self.renderer = self._get_renderer(inference_config.model_name, self.tokenizer)

        if inference_config.mode == InferenceMode.ASYNC:
            self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

    @staticmethod
    def _get_renderer(model_name: str, tokenizer) -> Renderer:
        """Get the appropriate renderer based on model name."""
        model_name_lower = model_name.lower()
        if "qwen" in model_name_lower:
            return Qwen3Renderer(tokenizer)
        elif "llama" in model_name_lower:
            return Llama3Renderer(tokenizer)
        else:
            raise ValueError(f"Unsupported model type for {model_name}. Only Qwen3 and Llama3.1 models are supported.")

    def _render_messages_to_tokens(self, messages: list[Message]) -> list[int]:
        """Render a list of messages to token IDs using the appropriate renderer.

        Uses the renderer's build_generation_prompt method to generate the complete
        prompt with proper formatting, BOS tokens, and assistant header.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            List of token IDs ready to be passed to vLLM
        """
        return self.renderer.build_generation_prompt(messages)

    @staticmethod
    def _patch_tpu_inference_registry():
        """Register Qwen2ForCausalLM in tpu_inference if not present."""
        try:
            from tpu_inference.models.common import model_loader

            if "Qwen2ForCausalLM" not in model_loader._MODEL_REGISTRY:
                logger.info("Patching tpu_inference to support Qwen2ForCausalLM")
                from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM

                model_loader.register_model("Qwen2ForCausalLM", Qwen2ForCausalLM)
        except ImportError:
            logger.exception("Failed to patch tpu_inference registry")
            raise

    @staticmethod
    def _get_llm_engine(inference_config: vLLMInferenceContextConfig):
        vLLMInferenceContext._patch_tpu_inference_registry()

        if inference_config.mode == InferenceMode.SYNC:
            if LLM is None:
                raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
            llm_engine_cls = LLM
        elif inference_config.mode == InferenceMode.ASYNC:
            llm_engine_cls = SyncVLLMWrapper
        else:
            raise ValueError(f"Invalid inference mode: {inference_config.mode}")

        return llm_engine_cls(
            model=inference_config.model_name,
            max_model_len=inference_config.max_model_len,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
        )

    def _convert_vllm_state_dict_to_trainer_keys(
        self, state_dict_trainer: dict, state_dict_vllm: dict, mapping: dict
    ) -> dict:
        state_dict_vllm_with_trainer_keys = {}
        for src_path, _ in state_dict_trainer.items():
            src_key = ".".join(str(p) for p in src_path)

            # Try to find a matching pattern
            matched = False
            for src_pattern, (dst_pattern, _) in mapping.items():

                if not re.match(src_pattern, src_key):
                    continue

                match_layer_number = re.match(r".*layers\.(\d+).*", src_key)
                if match_layer_number:
                    layer_number = int(match_layer_number.group(1))
                    dst_path = []
                    for part in dst_pattern.split("."):
                        if part == "*":
                            dst_path.append(layer_number)
                        else:
                            dst_path.append(part)
                    dst_path = tuple(dst_path)
                    if dst_path in state_dict_vllm:
                        state_dict_vllm_with_trainer_keys[src_path] = state_dict_vllm[dst_path]
                        matched = True
                        break
                else:
                    dst_path = tuple(dst_pattern.split("."))
                    if dst_path in state_dict_vllm:
                        state_dict_vllm_with_trainer_keys[src_path] = state_dict_vllm[dst_path]
                        matched = True
                        break

            if not matched:
                print(f"Warning: No mapping found for {src_key}")

        return state_dict_vllm_with_trainer_keys

    def _check_weight_differences(self, state_dict: dict, state_dict_other: dict):
        for key in state_dict:
            if key in state_dict_other:
                assert (
                    state_dict[key].shape == state_dict_other[key].shape
                ), f"Shape mismatch for key {key}: {state_dict[key].shape} != {state_dict_other[key].shape}"
                weight = jax.device_get(state_dict[key]).astype(jnp.bfloat16)
                weight_other = jax.device_get(state_dict_other[key]).astype(jnp.bfloat16)
                print(
                    f"Weight {key}, max diff: {jnp.max(jnp.abs(weight - weight_other))}, \
                    mean diff: {jnp.mean(jnp.abs(weight - weight_other))}"
                )

    def tokenize_prompt(self, prompt: str, choice: Choice | None = None, system_prompt: str | None = None) -> np.ndarray:
        """Tokenize the prompt with the choice's prompt token IDs.

        NOTE(chris): This is a hack to get the prompt token IDs the same since
        vLLM is injecting an extra BOS token at the start of the prompt.
        This is a known issue documented here:
        https://github.com/vllm-project/vllm/issues/27486
        """
        return np.array(choice.prompt_token_ids, dtype=np.int32)

    def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
        """Extract response token IDs directly from the choice.

        Uses the response_token_ids attached during vLLM-to-OpenAI conversion,
        avoiding the lossy convert_ids_to_tokens/convert_tokens_to_ids round-trip
        that fails for padding token IDs not in the tokenizer vocabulary.
        """
        return np.array(choice.response_token_ids, dtype=np.int32)

    def _convert_vllm_to_openai(self, request_output: RequestOutput) -> ChatCompletion:
        """Convert vLLM RequestOutput to OpenAI ChatCompletion format."""
        choices = []
        for output_idx, output in enumerate(request_output.outputs):
            # Convert logprobs
            logprobs_content = []
            for token_id, logprob_dict in zip(output.token_ids, output.logprobs, strict=False):
                # Get the token string (may be None for padding token IDs)
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                if token is None:
                    token = f"<id_{token_id}>"

                # Get the logprob for the selected token
                selected_logprob = None
                top_logprobs = []

                for tid, logprob_obj in logprob_dict.items():
                    if logprob_obj.rank == 1:
                        selected_logprob = logprob_obj.logprob

                    token_str = self.tokenizer.convert_ids_to_tokens(tid)
                    if token_str is None:
                        token_str = f"<id_{tid}>"
                    top_logprobs.append(
                        TopLogprob(
                            token=token_str,
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
                finish_reason=output.finish_reason,
                index=output_idx,
                logprobs=ChoiceLogprobs(content=logprobs_content),
                message=ChatCompletionMessage(
                    content=output.text,
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                ),
            )

            # Attach token IDs as custom attributes to the choice.
            # prompt_token_ids: needed since vLLM injects an extra BOS token at the start.
            # response_token_ids: avoids lossy convert_ids_to_tokens/convert_tokens_to_ids round-trip.
            choice.prompt_token_ids = request_output.prompt_token_ids
            choice.response_token_ids = list(output.token_ids)
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

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> LmHeadModel | None:
        # Reset prefix cache before syncing weights to free up memory
        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()

        # TODO(chris): levanter to vllm state dict
        nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)
        self.llm.llm_engine.model_executor.driver_worker.sync_weights(
            nnx_state,
            mappings=MODEL_MAPPINGS[self.model_name],
            transpose_keys=MODEL_TRANSPOSE_KEYS[self.model_name],
            reshard_fn=None,
        )

        self.llm.llm_engine.reset_prefix_cache()  # Reset prefix cache because of new weights
        return model

    def start_server(self, model: LmHeadModel) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def batch_completions(
        self,
        prompts: list[str] | list[list[dict]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[ChatCompletion]:
        """Batch completions from the inference server.

        Args:
            prompts: Either a list of strings or a list of message lists (with few-shot examples)
            temperature: Sampling temperature
            n: Number of completions per prompt
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            system_prompt: Optional system prompt (only used if prompts are strings)
        """
        if SamplingParams is None:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            # NOTE(chris): We allow the override to take precedence over the default sampling params.
            max_tokens=max_tokens or self.sampling_params.max_tokens,
            top_k=top_k or self.sampling_params.top_k,
            stop=stop or self.sampling_params.stop,
            logprobs=1,
            include_stop_str_in_output=self.sampling_params.include_stop_str_in_output,
            output_kind=self.sampling_params.output_kind,
        )

        # Convert prompts to message lists if they aren't already
        message_lists: list[list[Message]] = []
        if prompts and isinstance(prompts[0], list):
            # Prompts are already message lists with few-shot examples
            message_lists = prompts  # type: ignore
        elif system_prompt:
            # Plain string prompts with system prompt
            assert all(isinstance(p, str) for p in prompts), "prompts must be strings when system_prompt is provided"
            message_lists = [
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}] for prompt in prompts  # type: ignore
            ]
        else:
            # Plain string prompts without system prompt
            assert all(isinstance(p, str) for p in prompts), "prompts must be strings when no system_prompt is provided"
            message_lists = [[{"role": "user", "content": prompt}] for prompt in prompts]  # type: ignore

        # Render messages to token IDs using the appropriate renderer
        prompt_token_ids = []
        for messages in message_lists:
            tokens = self._render_messages_to_tokens(messages)
            prompt_token_ids.append(tokens)

        # Pass token IDs directly to vLLM
        # See: https://docs.vllm.ai/en/v0.4.3/dev/offline_inference/llm_inputs.html
        # vLLM accepts a list of TokensPrompt objects, which can be created by passing dicts with prompt_token_ids
        if TokensPrompt is None:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        prompts_for_vllm = [TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_token_ids]
        outputs = self.llm.generate(prompts_for_vllm, sampling_params)

        # Convert vLLM outputs to OpenAI ChatCompletion format
        return [self._convert_vllm_to_openai(output) for output in outputs]
