"""Project unsharded GPT output in its consumer layout without duplicating vocabulary logits."""

import torch
from loguru import logger
from torch import nn

from marinskyrl.runtime_options import PolicyLossType


def _batch_major_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return (inputs[0].transpose(0, 1).contiguous(), *inputs[1:])


def _sequence_major_output(
    _module: nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: tuple[torch.Tensor, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    logits, bias = output
    return logits.transpose(0, 1), bias


def use_batch_major_output_projection(projection: nn.Module) -> None:
    """Make GPT's final batch-major conversion a view for TP1, non-sequence-parallel output layers.

    Moving the smaller hidden-state tensor before projection avoids a second full-vocabulary
    allocation in GPTModel._postprocess. Hooks retain the module and its checkpoint parameter names.
    """
    projection.register_forward_pre_hook(_batch_major_input)
    projection.register_forward_hook(_sequence_major_output)


def _extract_weight(projection: nn.Module, inputs: tuple[torch.Tensor, ...], kwargs: dict) -> torch.Tensor:
    if len(inputs) >= 2:
        return projection.weight if inputs[1] is None else inputs[1]
    weight = kwargs.get("weight")
    return projection.weight if weight is None else weight


def _fp32_projection_input(
    projection: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    kwargs: dict,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    weight = _extract_weight(projection, inputs, kwargs).float()
    if len(inputs) >= 2:
        return (inputs[0].float(), weight, *inputs[2:]), kwargs
    return (inputs[0].float(),), {**kwargs, "weight": weight}


def use_fp32_output_projection(projection: nn.Module) -> None:
    """Run the existing Megatron output projection with FP32 operands and gradients."""
    # A cast weight is not a parameter with Megatron's main_grad buffer. Let
    # autograd return its gradient through the cast to the original parameter.
    projection.gradient_accumulation_fusion = False
    raw_dtypes: dict[str, torch.dtype] = {}
    receipt_handles: list[torch.utils.hooks.RemovableHandle] = []

    def _capture_raw_dtypes(module: nn.Module, inputs: tuple[torch.Tensor, ...], kwargs: dict) -> None:
        raw_dtypes["input"] = inputs[0].dtype
        raw_dtypes["weight"] = _extract_weight(module, inputs, kwargs).dtype

    def _log_receipt_once(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict,
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        while receipt_handles:
            receipt_handles.pop().remove()
        logits = output[0] if isinstance(output, tuple) else output
        gemm_weight = _extract_weight(module, inputs, kwargs)
        logger.info(
            "fp32_lm_head runtime receipt: raw_input={} raw_weight={} gemm_input={} gemm_weight={} "
            "output={} allow_tf32={} fp32_matmul_precision={} gradient_accumulation_fusion={}",
            raw_dtypes.get("input"),
            raw_dtypes.get("weight"),
            inputs[0].dtype,
            gemm_weight.dtype,
            logits.dtype,
            torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision(),
            getattr(module, "gradient_accumulation_fusion", None),
        )

    receipt_handles.append(projection.register_forward_pre_hook(_capture_raw_dtypes, with_kwargs=True))
    projection.register_forward_pre_hook(_fp32_projection_input, with_kwargs=True)
    receipt_handles.append(projection.register_forward_hook(_log_receipt_once, with_kwargs=True))


def install_fp32_output_projection(model_chunks) -> None:
    """Enable the FP32 head on supported TP1 GPT output layers."""
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.utils import get_attr_wrapped_model

    for chunk in model_chunks:
        if not get_attr_wrapped_model(chunk, "post_process"):
            continue
        projection = get_attr_wrapped_model(chunk, "output_layer")
        if (
            not isinstance(projection, ColumnParallelLinear)
            or projection.output_size != projection.output_size_per_partition
            or projection.sequence_parallel
            or projection.config.defer_embedding_wgrad_compute
            or projection.bias is not None
        ):
            raise ValueError("fp32_lm_head requires a TP1 ColumnParallelLinear output without deferred gradients")
        use_fp32_output_projection(projection)


def install_dpo_batch_major_output_projection(model_chunks, provider, policy_loss_type) -> None:
    """Install the batch-major projection on every post-process chunk when DPO can use it.

    No-op unless the loss type is DPO and the topology is TP1 without context or
    sequence parallelism.
    """
    from megatron.core.utils import get_attr_wrapped_model

    if (
        policy_loss_type == PolicyLossType.DPO
        and provider.tensor_model_parallel_size == 1
        and provider.context_parallel_size == 1
        and not provider.sequence_parallel
    ):
        for chunk in model_chunks:
            if get_attr_wrapped_model(chunk, "post_process"):
                use_batch_major_output_projection(get_attr_wrapped_model(chunk, "output_layer"))
