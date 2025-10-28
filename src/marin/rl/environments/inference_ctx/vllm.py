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
import re
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob, TopLogprob
from haliax.partitioning import ResourceAxis
from levanter.models.lm_model import LmHeadModel
from marin.rl.weight_utils import levanter_to_nnx_state
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


def levanter_llama_to_vllm_mapping():
    return {
        "lm_head": ("model.lm_head", (None, "model")),
        "model.embed_tokens": (
            "model.embed.embedding",
            ("model", None),
        ),
        "model.layers.*.input_layernorm": (
            "model.layers.*.input_layernorm.scale",
            (None,),
        ),
        "model.layers.*.mlp.down_proj": (
            "model.layers.*.mlp.down_proj.kernel",
            ("model", None),
        ),
        "model.layers.*.mlp.gate_proj": (
            "model.layers.*.mlp.gate_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.mlp.up_proj": (
            "model.layers.*.mlp.up_proj.kernel",
            (None, "model"),
        ),
        "model.layers.*.post_attention_layernorm": (
            "model.layers.*.post_attention_layernorm.scale",
            (None,),
        ),
        "model.layers.*.self_attn.k_proj": (
            "model.layers.*.self_attn.k_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.o_proj": (
            "model.layers.*.self_attn.o_proj.kernel",
            ("model", None, None),
        ),
        "model.layers.*.self_attn.q_proj": (
            "model.layers.*.self_attn.q_proj.kernel",
            (None, "model", None),
        ),
        "model.layers.*.self_attn.v_proj": (
            "model.layers.*.self_attn.v_proj.kernel",
            (None, "model", None),
        ),
        "model.norm": ("model.norm.scale", (None,)),
    }


def levanter_qwen_to_vllm_mapping():
    mapping = levanter_llama_to_vllm_mapping()
    mapping.update(
        {
            "model.layers.*.self_attn.q_norm": ("model.layers.*.self_attn.q_norm.scale", (None,)),
            "model.layers.*.self_attn.k_norm": ("model.layers.*.self_attn.k_norm.scale", (None,)),
        }
    )
    return mapping


llama_transpose_keys = {
    "lm_head": (1, 0),
    "gate_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "q_proj": (2, 0, 1),
    "k_proj": (2, 0, 1),
    "v_proj": (2, 0, 1),
    "o_proj": (1, 2, 0),
}

MODEL_MAPPINGS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "meta-llama/Llama-3.2-1B-Instruct": levanter_llama_to_vllm_mapping(),
    "meta-llama/Llama-3.2-3B-Instruct": levanter_llama_to_vllm_mapping(),
    "Qwen/Qwen3-0.6B": levanter_qwen_to_vllm_mapping(),
    "Qwen/Qwen3-1.7B": levanter_qwen_to_vllm_mapping(),
    "meta-llama/Llama-3.1-8B-Instruct": levanter_llama_to_vllm_mapping(),
}

MODEL_TRANSPOSE_KEYS: dict[str, tuple[int, ...]] = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_transpose_keys,
    "meta-llama/Llama-3.2-3B-Instruct": llama_transpose_keys,
    "Qwen/Qwen3-0.6B": llama_transpose_keys,
    "Qwen/Qwen3-1.7B": llama_transpose_keys,
    "meta-llama/Llama-3.1-8B-Instruct": llama_transpose_keys,
}


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
        # Mesh for the weight transfer client should be on the CPU and then
        # in sync_weights function, we will reshard to the TPU
        self.mesh = jax.make_mesh(
            (1, 1, 1),
            (ResourceAxis.DATA, ResourceAxis.REPLICA, ResourceAxis.MODEL),
            devices=jax.local_devices(backend="cpu")[:1],
        )
        self.axis_mapping = {}
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = inference_config.model_name
        self.sampling_params = inference_config.sampling_params

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

    def reload_model(self, model: LmHeadModel) -> None:
        nnx_state = levanter_to_nnx_state(model)
        self.llm.llm_engine.model_executor.driver_worker.sync_weights(
            nnx_state,
            mappings=MODEL_MAPPINGS[self.model_name],
            transpose_keys=MODEL_TRANSPOSE_KEYS[self.model_name],
            reshard_fn=None,
        )

        self.llm.llm_engine.reset_prefix_cache()  # Reset prefix cache because of new weights

    def start_server(self, model: LmHeadModel) -> None:
        pass

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
