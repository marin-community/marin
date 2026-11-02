"""Project unsharded GPT output in its consumer layout without duplicating vocabulary logits."""

import torch
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
