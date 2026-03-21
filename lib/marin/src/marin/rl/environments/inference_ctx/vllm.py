# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import json
import gc
import logging
import os
import re
import shutil
import tempfile
import time
from urllib.parse import urlparse

import jax
import jax.numpy as jnp
import numpy as np
from enum import StrEnum
from dataclasses import dataclass
from typing import Any
from iris.marin_fs import url_to_fs
from levanter.compat.fsspec_safetensor import read_safetensors_fsspec
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

SAFE_TENSORS_MODEL = "model.safetensors"
SAFE_TENSORS_INDEX_NAME = "model.safetensors.index.json"
BOOTSTRAP_METADATA_FILENAMES: tuple[str, ...] = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "generation_config.json",
    "chat_template.jinja",
)


def _patch_tpu_inference_llama_nnx_compat(llama3_module, nnx_module) -> None:
    """Make TPU-inference Llama pipeline layers construct as ``nnx.List``."""
    original_make_layers = llama3_module.make_layers
    if getattr(original_make_layers, "_marin_nnx_list_patch", False):
        return

    def _patched_make_layers(*args, **kwargs):
        start_layer, end_layer, layers = original_make_layers(*args, **kwargs)
        if not isinstance(layers, nnx_module.List):
            layers = nnx_module.List(layers)
        return start_layer, end_layer, layers

    _patched_make_layers._marin_nnx_list_patch = True  # type: ignore[attr-defined]
    _patched_make_layers.__wrapped__ = original_make_layers  # type: ignore[attr-defined]
    llama3_module.make_layers = _patched_make_layers


class InferenceMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


@dataclass
class VllmSamplingConfig:
    """Serializable sampling configuration for vLLM-backed inference.

    This config intentionally belongs to Marin rather than vLLM so it can be
    pickled on controller hosts that do not have a matching local vLLM install.
    TPU workers convert it into a real ``vllm.SamplingParams`` object at the
    point of use.
    """

    temperature: float = 1.0
    n: int = 1
    max_tokens: int = 1024
    top_k: int = -1
    stop: list[str] | None = None
    include_stop_str_in_output: bool = False
    logprobs: int | None = None
    output_kind: str | None = None


def coerce_vllm_sampling_config(value: Any) -> VllmSamplingConfig:
    """Normalize legacy or foreign sampling-param objects into Marin config."""

    if isinstance(value, VllmSamplingConfig):
        return value

    return VllmSamplingConfig(
        temperature=float(getattr(value, "temperature", 1.0)),
        n=int(getattr(value, "n", 1)),
        max_tokens=int(getattr(value, "max_tokens", 1024)),
        top_k=getattr(value, "top_k", -1),
        stop=getattr(value, "stop", None),
        include_stop_str_in_output=bool(getattr(value, "include_stop_str_in_output", False)),
        logprobs=getattr(value, "logprobs", None),
        output_kind=getattr(value, "output_kind", None),
    )


@dataclass
class vLLMInferenceContextConfig:
    """Configuration for vLLM engine and sampling parameters."""

    model_name: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    sampling_params: VllmSamplingConfig
    mode: InferenceMode = InferenceMode.SYNC
    load_format: str = "auto"
    enforce_eager: bool = True
    enable_fast_bootstrap: bool = False
    bootstrap_checkpoint_path: str | None = None


class vLLMInferenceContext(BaseInferenceContext):
    """Inference context for vLLM."""

    def __init__(
        self,
        inference_config: vLLMInferenceContextConfig,
    ):
        self.llm = self._build_llm_with_optional_fast_bootstrap(inference_config)
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
        self.sampling_params = coerce_vllm_sampling_config(inference_config.sampling_params)

        # Initialize the appropriate renderer based on model type
        self.renderer = self._get_renderer(inference_config.model_name, self.tokenizer)

        if inference_config.mode == InferenceMode.ASYNC:
            self.sampling_params.output_kind = "final_only"

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
        """Register Marin TPU-inference compatibility patches before vLLM starts."""
        try:
            from flax import nnx
            from tpu_inference.models.common import model_loader
            from tpu_inference.models.jax import llama3

            _patch_tpu_inference_llama_nnx_compat(llama3, nnx)

            if "Qwen2ForCausalLM" not in model_loader._MODEL_REGISTRY:
                logger.info("Patching tpu_inference to support Qwen2ForCausalLM")
                from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM

                model_loader.register_model("Qwen2ForCausalLM", Qwen2ForCausalLM)
        except ImportError:
            logger.exception("Failed to patch tpu_inference registry")
            raise

    @staticmethod
    def _get_llm_engine(inference_config: vLLMInferenceContextConfig, model_source: str | None = None):
        vLLMInferenceContext._patch_tpu_inference_registry()

        if inference_config.mode == InferenceMode.SYNC:
            if LLM is None:
                raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
            llm_engine_cls = LLM
        elif inference_config.mode == InferenceMode.ASYNC:
            llm_engine_cls = SyncVLLMWrapper
        else:
            raise ValueError(f"Invalid inference mode: {inference_config.mode}")

        model_source = model_source or inference_config.model_name
        return llm_engine_cls(
            model=model_source,
            max_model_len=inference_config.max_model_len,
            tensor_parallel_size=inference_config.tensor_parallel_size,
            gpu_memory_utilization=inference_config.gpu_memory_utilization,
            load_format=inference_config.load_format,
            enforce_eager=inference_config.enforce_eager,
        )

    @classmethod
    def _build_llm_with_optional_fast_bootstrap(cls, inference_config: vLLMInferenceContextConfig):
        if not inference_config.enable_fast_bootstrap:
            return cls._get_llm_engine(inference_config)

        bootstrap_checkpoint_path = inference_config.bootstrap_checkpoint_path
        if not bootstrap_checkpoint_path:
            raise ValueError(
                "Fast bootstrap requested but bootstrap_checkpoint_path is unset. "
                "Alternating RL cannot fall back to the base model without changing policy weights."
            )

        if not _is_object_store_path(bootstrap_checkpoint_path):
            raise ValueError(
                "Fast bootstrap requested with a non-object-store checkpoint path: " f"{bootstrap_checkpoint_path!r}"
            )

        bootstrap_local_dir = None
        try:
            bootstrap_local_dir = _stage_bootstrap_metadata(bootstrap_checkpoint_path)
            bootstrap_config = dataclasses.replace(inference_config, load_format="dummy")
            llm = cls._get_llm_engine(bootstrap_config, model_source=bootstrap_local_dir)
            _bootstrap_weights_into_engine(llm, inference_config.model_name, bootstrap_checkpoint_path)
            logger.info(
                "Fast bootstrap completed for model %s from %s", inference_config.model_name, bootstrap_checkpoint_path
            )
            return llm
        except Exception as exc:
            logger.exception(
                "Fast bootstrap failed for model %s from %s.",
                inference_config.model_name,
                bootstrap_checkpoint_path,
            )
            raise RuntimeError(
                "Fast bootstrap failed while loading policy weights. "
                "Alternating RL cannot safely fall back to the base model for a later policy version."
            ) from exc
        finally:
            if bootstrap_local_dir is not None:
                shutil.rmtree(bootstrap_local_dir, ignore_errors=True)

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
        t0 = time.time()
        logger.info("reload_model: starting prefix cache reset")
        # Reset prefix cache before syncing weights to free up memory
        self.llm.llm_engine.reset_prefix_cache()
        gc.collect()

        # TODO(chris): levanter to vllm state dict
        logger.info("reload_model: converting state dict")
        nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)
        t1 = time.time()
        logger.info("reload_model: calling sync_weights (%d params, %.1fs so far)", len(nnx_state), t1 - t0)
        self.llm.llm_engine.model_executor.driver_worker.sync_weights(
            nnx_state,
            mappings=MODEL_MAPPINGS[self.model_name],
            transpose_keys=MODEL_TRANSPOSE_KEYS[self.model_name],
            reshard_fn=None,
        )
        t2 = time.time()
        logger.info("reload_model: sync_weights done in %.1fs, resetting prefix cache", t2 - t1)

        self.llm.llm_engine.reset_prefix_cache()  # Reset prefix cache because of new weights
        logger.info("reload_model: complete in %.1fs", time.time() - t0)
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
        output_kind = None
        if self.sampling_params.output_kind is not None:
            if RequestOutputKind is None:
                raise ImportError("vLLM RequestOutputKind is unavailable")
            output_kind = RequestOutputKind(self.sampling_params.output_kind)
        sampling_params = SamplingParams(
            temperature=temperature,
            n=n,
            # NOTE(chris): We allow the override to take precedence over the default sampling params.
            max_tokens=max_tokens or self.sampling_params.max_tokens,
            top_k=top_k or self.sampling_params.top_k,
            stop=stop or self.sampling_params.stop,
            logprobs=1,
            include_stop_str_in_output=self.sampling_params.include_stop_str_in_output,
            output_kind=output_kind,
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
        logger.info("generate: starting, %d prompts, max_tokens=%d", len(prompts_for_vllm), sampling_params.max_tokens)
        t0 = time.time()
        outputs = self.llm.generate(prompts_for_vllm, sampling_params)
        logger.info("generate: done in %.1fs", time.time() - t0)

        # Convert vLLM outputs to OpenAI ChatCompletion format
        return [self._convert_vllm_to_openai(output) for output in outputs]


def _is_object_store_path(path: str) -> bool:
    return urlparse(path).scheme in {"gs", "s3"}


def _discover_safetensor_shards(fs, remote_path: str) -> list[str]:
    index_path = os.path.join(remote_path, SAFE_TENSORS_INDEX_NAME)
    if fs.exists(index_path):
        with fs.open(index_path, "r") as f:
            index_payload = json.load(f)
        shard_files = list(dict.fromkeys(index_payload["weight_map"].values()))
        if shard_files:
            return shard_files

    single_file_path = os.path.join(remote_path, SAFE_TENSORS_MODEL)
    if fs.exists(single_file_path):
        return [SAFE_TENSORS_MODEL]

    raise FileNotFoundError(f"No safetensors checkpoint found under {remote_path}")


def _load_safetensors_from_remote(model_path: str) -> dict[str, np.ndarray]:
    fs, remote_path = url_to_fs(model_path)
    shard_files = _discover_safetensor_shards(fs, remote_path)

    async def _load_shard(shard_path: str) -> dict[str, np.ndarray]:
        return await read_safetensors_fsspec(shard_path, fs=fs, sharding_fn=None)

    cpu_device = jax.devices("cpu")[0]
    loop = asyncio.new_event_loop()
    state_dict: dict[str, np.ndarray] = {}

    try:
        with jax.default_device(cpu_device):
            for shard_file in shard_files:
                shard_path = os.path.join(remote_path, shard_file)
                shard_state = loop.run_until_complete(_load_shard(shard_path))
                state_dict.update(shard_state)
    finally:
        loop.close()

    return state_dict


def _stage_bootstrap_metadata(model_path: str) -> str:
    fs, remote_path = url_to_fs(model_path)
    local_dir = tempfile.mkdtemp(prefix="marin-vllm-bootstrap-")
    copied_any = False

    try:
        for filename in BOOTSTRAP_METADATA_FILENAMES:
            remote_file = os.path.join(remote_path, filename)
            if not fs.exists(remote_file):
                continue

            local_file = os.path.join(local_dir, filename)
            with fs.open(remote_file, "rb") as src:
                payload = src.read()
            with open(local_file, "wb") as dst:
                dst.write(payload)
            copied_any = True

        if not copied_any:
            raise FileNotFoundError(f"No bootstrap metadata files found under {model_path!r}")
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            raise FileNotFoundError(f"Missing config.json under {model_path!r}; cannot initialize vLLM dummy model")

        return local_dir
    except Exception:
        shutil.rmtree(local_dir, ignore_errors=True)
        raise


def _serialize_state_dict_for_rpc(state_dict: dict[str, np.ndarray]) -> dict:
    serialized = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = (value.tobytes(), str(value.dtype), value.shape)
        else:
            serialized[key] = value
    return serialized


def _bootstrap_weights_into_engine(llm: object, model_name: str, checkpoint_path: str) -> None:
    if model_name not in MODEL_MAPPINGS or model_name not in MODEL_TRANSPOSE_KEYS:
        raise ValueError(f"No vLLM weight mapping found for model {model_name!r}")

    state_dict = _load_safetensors_from_remote(checkpoint_path)
    nnx_state = levanter_state_dict_to_nnx_state_on_cpu(state_dict)

    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is not None:
        sync_weights = getattr(llm_engine.model_executor.driver_worker, "sync_weights", None)
        if not callable(sync_weights):
            raise RuntimeError("driver_worker.sync_weights is unavailable; cannot perform fast bootstrap")
        sync_weights(
            nnx_state,
            mappings=MODEL_MAPPINGS[model_name],
            transpose_keys=MODEL_TRANSPOSE_KEYS[model_name],
            reshard_fn=None,
        )
        llm_engine.reset_prefix_cache()
        return

    update_weights = getattr(llm, "update_weights", None)
    if callable(update_weights):
        serialized_state_dict = _serialize_state_dict_for_rpc(state_dict)
        update_weights(serialized_state_dict, model_name)
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if callable(reset_prefix_cache):
            reset_prefix_cache()
        return

    raise RuntimeError("Unsupported vLLM engine type for fast bootstrap weight injection")
