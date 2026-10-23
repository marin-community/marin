"""Provider topology and kernel settings for Megatron-Bridge models, in one place."""

import importlib.util

import torch


def configure_bridge_provider(
    provider,
    *,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int,
    bf16: bool,
    attention_backend: str,
) -> None:
    """Apply the worker's topology and kernel choices to a bridge provider in place."""
    provider.tensor_model_parallel_size = tensor_model_parallel_size
    provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
    provider.context_parallel_size = context_parallel_size
    provider.expert_model_parallel_size = expert_model_parallel_size
    provider.expert_tensor_parallel_size = expert_tensor_parallel_size
    provider.sequence_parallel = tensor_model_parallel_size > 1
    provider.attention_backend = attention_backend
    provider.variable_seq_lengths = True
    provider.masked_softmax_fusion = True
    provider.moe_token_dispatcher_type = "alltoall"
    # Megatron-Bridge enables wgrad fusion whenever Transformer Engine is present, but the
    # non-TE output layer still needs APEX's fused_weight_gradient_mlp_cuda extension.
    provider.gradient_accumulation_fusion = importlib.util.find_spec("fused_weight_gradient_mlp_cuda") is not None
