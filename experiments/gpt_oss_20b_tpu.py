# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared TPU bootstrap config for BF16 `gpt-oss-20b` local-vLLM experiments."""

from experiments.gpt_oss_120b_tpu import GPT_OSS_TPU_ADDITIONAL_CONFIG, GPT_OSS_TPU_INFERENCE_PACKAGE
from experiments.models import gpt_oss_20b_vllm
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import output_path_of

GPT_OSS_20B_TPU_MAX_MODEL_LEN = 4096
GPT_OSS_20B_TPU_DEFAULT_MAX_TOKENS = 2048
GPT_OSS_20B_TPU_DEFAULT_MODEL_IMPL_TYPE = "flax_nnx"
GPT_OSS_20B_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP = True


def _gpt_oss_20b_tpu_additional_config(*, prefer_jax_for_bootstrap: bool) -> dict[str, object]:
    tpu_bootstrap = dict(GPT_OSS_TPU_ADDITIONAL_CONFIG["tpu_bootstrap"])
    tpu_bootstrap["prefer_jax_for_bootstrap"] = prefer_jax_for_bootstrap
    return {
        **GPT_OSS_TPU_ADDITIONAL_CONFIG,
        "tpu_bootstrap": tpu_bootstrap,
    }


def gpt_oss_20b_tpu_vllm_config(
    *,
    tpu_type: str = "v5p-8",
    tensor_parallel_size: int = 4,
    max_model_len: int = GPT_OSS_20B_TPU_MAX_MODEL_LEN,
    cpu: int = 16,
    disk: str = "60g",
    ram: str = "128g",
    model_impl_type: str = GPT_OSS_20B_TPU_DEFAULT_MODEL_IMPL_TYPE,
    prefer_jax_for_bootstrap: bool = GPT_OSS_20B_TPU_DEFAULT_PREFER_JAX_FOR_BOOTSTRAP,
) -> VLLMConfig:
    return VLLMConfig(
        model=output_path_of(gpt_oss_20b_vllm),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tpu_type=tpu_type,
        cpu=cpu,
        disk=disk,
        ram=ram,
        additional_config=_gpt_oss_20b_tpu_additional_config(
            prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
        ),
        model_impl_type=model_impl_type,
        extra_pip_packages=(GPT_OSS_TPU_INFERENCE_PACKAGE,),
    )
