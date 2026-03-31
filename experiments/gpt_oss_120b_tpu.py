# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared TPU bootstrap config for BF16 `gpt-oss-120b` local-vLLM experiments."""

from experiments.models import gpt_oss_120b_vllm
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import output_path_of

GPT_OSS_TPU_INFERENCE_PACKAGE = (
    "tpu_inference @ " "git+https://github.com/marin-community/tpu-inference.git@ahmed/gpt-oss-tpu-bringup-v5"
)

GPT_OSS_TPU_ADDITIONAL_CONFIG = {
    "skip_quantization": True,
    "tpu_bootstrap": {
        "model_bootstrap": "abstract_load",
        "prefer_jax_for_bootstrap": True,
    },
}
GPT_OSS_TPU_MAX_MODEL_LEN = 4096
GPT_OSS_TPU_DEFAULT_MAX_TOKENS = 2048


def _gpt_oss_120b_tpu_additional_config(*, prefer_jax_for_bootstrap: bool) -> dict[str, object]:
    tpu_bootstrap = dict(GPT_OSS_TPU_ADDITIONAL_CONFIG["tpu_bootstrap"])
    tpu_bootstrap["prefer_jax_for_bootstrap"] = prefer_jax_for_bootstrap
    return {
        **GPT_OSS_TPU_ADDITIONAL_CONFIG,
        "tpu_bootstrap": tpu_bootstrap,
    }


def gpt_oss_120b_tpu_vllm_config(
    *,
    tpu_type: str = "v5p-8",
    tensor_parallel_size: int = 4,
    max_model_len: int = GPT_OSS_TPU_MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.9,
    cpu: int = 32,
    disk: str = "80g",
    ram: str = "256g",
    # GPT-OSS must use model_impl_type="vllm". The flax_nnx backend produces
    # incoherent token soup (see .agents/logbooks/gpt-oss-tpu.md GTPU-001 to GTPU-004).
    model_impl_type: str = "vllm",
    prefer_jax_for_bootstrap: bool = False,
) -> VLLMConfig:
    return VLLMConfig(
        model=output_path_of(gpt_oss_120b_vllm),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tpu_type=tpu_type,
        cpu=cpu,
        disk=disk,
        ram=ram,
        additional_config=_gpt_oss_120b_tpu_additional_config(
            prefer_jax_for_bootstrap=prefer_jax_for_bootstrap,
        ),
        model_impl_type=model_impl_type,
        extra_pip_packages=(GPT_OSS_TPU_INFERENCE_PACKAGE,),
    )
