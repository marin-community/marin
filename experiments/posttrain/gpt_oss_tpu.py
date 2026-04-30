# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared TPU bootstrap config for GPT-OSS posttrain local-vLLM experiments."""

from dataclasses import dataclass
from enum import StrEnum

from experiments.models import gpt_oss_120b_vllm, gpt_oss_20b_heretic_vllm, gpt_oss_20b_vllm, mixtral_8x7b_instruct
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import output_path_of

GPT_OSS_TPU_ADDITIONAL_CONFIG = {
    "skip_quantization": True,
    "tpu_bootstrap": {
        "model_bootstrap": "abstract_load",
    },
}
GPT_OSS_TPU_MAX_MODEL_LEN = 4096
GPT_OSS_TPU_DEFAULT_MAX_TOKENS = 2048
GPT_OSS_TPU_DEFAULT_MODEL_IMPL_TYPE = "vllm"
GPT_OSS_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP = False
REJECTED_MODEL_MAX_MODEL_LEN = 8192


@dataclass(frozen=True)
class GptOssTpuDefaults:
    cpu: int
    disk: str
    ram: str


GPT_OSS_20B_TPU_DEFAULTS = GptOssTpuDefaults(cpu=16, disk="60g", ram="128g")
GPT_OSS_120B_TPU_DEFAULTS = GptOssTpuDefaults(cpu=32, disk="80g", ram="400g")


class RejectedModelPreset(StrEnum):
    MIXTRAL = "mixtral"
    HERETIC_20B = "heretic-20b"


def _gpt_oss_tpu_additional_config(*, prefer_jax_for_bootstrap: bool) -> dict[str, object]:
    tpu_bootstrap = dict(GPT_OSS_TPU_ADDITIONAL_CONFIG["tpu_bootstrap"])
    tpu_bootstrap["prefer_jax_for_bootstrap"] = prefer_jax_for_bootstrap
    return {
        **GPT_OSS_TPU_ADDITIONAL_CONFIG,
        "tpu_bootstrap": tpu_bootstrap,
    }


def gpt_oss_tpu_vllm_config(
    *,
    model: str,
    tpu_type: str = "v5p-8",
    tensor_parallel_size: int = 4,
    max_model_len: int = GPT_OSS_TPU_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    cpu: int,
    disk: str,
    ram: str,
    # GPT-OSS must use model_impl_type="vllm". The flax_nnx backend is not the
    # supported path for these posttrain experiments.
    model_impl_type: str = GPT_OSS_TPU_DEFAULT_MODEL_IMPL_TYPE,
    prefer_jax_for_bootstrap: bool = GPT_OSS_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP,
) -> VLLMConfig:
    return VLLMConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tpu_type=tpu_type,
        cpu=cpu,
        disk=disk,
        ram=ram,
        additional_config=_gpt_oss_tpu_additional_config(
            prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
        ),
        model_impl_type=model_impl_type,
    )


def gpt_oss_20b_tpu_vllm_config(
    *,
    tpu_type: str = "v5p-8",
    tensor_parallel_size: int = 4,
    max_model_len: int = GPT_OSS_TPU_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    cpu: int = GPT_OSS_20B_TPU_DEFAULTS.cpu,
    disk: str = GPT_OSS_20B_TPU_DEFAULTS.disk,
    ram: str = GPT_OSS_20B_TPU_DEFAULTS.ram,
    model_impl_type: str = GPT_OSS_TPU_DEFAULT_MODEL_IMPL_TYPE,
    prefer_jax_for_bootstrap: bool = GPT_OSS_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP,
) -> VLLMConfig:
    return gpt_oss_tpu_vllm_config(
        model=output_path_of(gpt_oss_20b_vllm),
        tpu_type=tpu_type,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        cpu=cpu,
        disk=disk,
        ram=ram,
        model_impl_type=model_impl_type,
        prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
    )


def gpt_oss_120b_tpu_vllm_config(
    *,
    tpu_type: str = "v5p-8",
    tensor_parallel_size: int = 4,
    max_model_len: int = GPT_OSS_TPU_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    cpu: int = GPT_OSS_120B_TPU_DEFAULTS.cpu,
    disk: str = GPT_OSS_120B_TPU_DEFAULTS.disk,
    ram: str = GPT_OSS_120B_TPU_DEFAULTS.ram,
    model_impl_type: str = GPT_OSS_TPU_DEFAULT_MODEL_IMPL_TYPE,
    prefer_jax_for_bootstrap: bool = GPT_OSS_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP,
) -> VLLMConfig:
    return gpt_oss_tpu_vllm_config(
        model=output_path_of(gpt_oss_120b_vllm),
        tpu_type=tpu_type,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        cpu=cpu,
        disk=disk,
        ram=ram,
        model_impl_type=model_impl_type,
        prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
    )


def rejected_model_label(preset: RejectedModelPreset) -> str:
    match preset:
        case RejectedModelPreset.MIXTRAL:
            return "Mixtral-8x7B-Instruct"
        case RejectedModelPreset.HERETIC_20B:
            return "Heretic GPT-OSS 20B"


def rejected_model_tag(preset: RejectedModelPreset) -> str:
    return f"{preset.value}-rejected"


def rejected_model_name_suffix(preset: RejectedModelPreset) -> str:
    return preset.value.replace("-", "_")


def rejected_model_vllm_config(
    preset: RejectedModelPreset,
    *,
    max_model_len: int = REJECTED_MODEL_MAX_MODEL_LEN,
) -> VLLMConfig:
    match preset:
        case RejectedModelPreset.MIXTRAL:
            return VLLMConfig(
                model=output_path_of(mixtral_8x7b_instruct),
                tensor_parallel_size=4,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.9,
                tpu_type="v5p-8",
                disk="10g",
                ram="256g",
            )
        case RejectedModelPreset.HERETIC_20B:
            return gpt_oss_tpu_vllm_config(
                model=output_path_of(gpt_oss_20b_heretic_vllm),
                cpu=GPT_OSS_20B_TPU_DEFAULTS.cpu,
                disk=GPT_OSS_20B_TPU_DEFAULTS.disk,
                ram=GPT_OSS_20B_TPU_DEFAULTS.ram,
                max_model_len=max_model_len,
            )
