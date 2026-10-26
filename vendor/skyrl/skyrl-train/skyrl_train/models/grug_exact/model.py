"""The exact Grug pipeline-stage model: it runs the exact layers and carries each token's input-norm statistic between
layers and stages."""

from __future__ import annotations

from collections.abc import Callable

import torch
from megatron.core import tensor_parallel
from megatron.core.utils import make_viewless_tensor

from skyrl_train.models.grug_exact.context import ExactForward, ResidualSum
from skyrl_train.models.grug_exact.layers import ExactEmbeddingGatedNorm, ExactFinalGatedNorm, ExactTransformerLayer
from skyrl_train.models.grug_exact.spec import exact_layer_spec
from skyrl_train.models.grug_exact.support import FULL_RECOMPUTE
from skyrl_train.models.grug_megatron import GrugGPTModel


class _AppendStageStatistic(torch.autograd.Function):
    """``[S, B, H]`` bf16 hidden states with each token's fp32 statistic appended as two bf16 words, ``[S, B, H + 2]``;
    only the hidden states take gradient."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        ctx.width = hidden.shape[-1]
        words = statistic.float().contiguous().view(torch.int16)
        return torch.cat((hidden.contiguous().view(torch.int16), words), dim=-1).view(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad[..., : ctx.width], None


class _SplitStageStatistic(torch.autograd.Function):
    """The inverse of ``_AppendStageStatistic``: the ``[S, B, H]`` hidden states and ``[S, B, 1]`` fp32 statistic."""

    @staticmethod
    def forward(ctx, packed: torch.Tensor, width: int):
        ctx.packed_width = packed.shape[-1]
        words = packed.view(torch.int16)
        hidden = words[..., :width].contiguous().view(torch.bfloat16)
        statistic = words[..., width:].contiguous().view(torch.float32)
        ctx.mark_non_differentiable(statistic)
        return hidden, statistic

    @staticmethod
    def backward(ctx, grad_hidden: torch.Tensor, grad_statistic: torch.Tensor | None):
        grad = grad_hidden.new_zeros(*grad_hidden.shape[:-1], ctx.packed_width)
        grad[..., : grad_hidden.shape[-1]] = grad_hidden
        return grad, None


def _layer_unit(layer: ExactTransformerLayer, rotary_pos_emb, exact: ExactForward) -> Callable:
    """The checkpointed function of one layer: its tensor inputs and outputs, with the residual sum's parts flat."""

    def run(hidden_states: torch.Tensor, statistic: torch.Tensor):
        output, statistic, parts = layer(hidden_states, statistic, rotary_pos_emb, exact, keep=True)
        return output, statistic, parts.residual, parts.routed, parts.shared

    return run


class GrugExactGPTModel(GrugGPTModel):
    """GrugGPTModel whose forward runs the exact layers; a stage that is not the last appends each token's input-norm
    statistic to the hidden states it hands on, and the next stage splits it off."""

    embed_norm_class = ExactEmbeddingGatedNorm
    final_norm_class = ExactFinalGatedNorm
    layer_spec = staticmethod(exact_layer_spec)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params=None,
        *,
        exact: ExactForward,
    ) -> torch.Tensor:
        if packed_seq_params is not None:
            raise ValueError("the exact Grug trainer runs without sample packing")
        # On the first stage the exact embedding norm returns the hidden states and layer 0's statistic.
        embedded, rotary_pos_emb, *_ = self._preprocess(input_ids=input_ids, position_ids=position_ids)
        if self.pre_process:
            hidden_states, statistic = embedded
        else:
            hidden_states, statistic = _SplitStageStatistic.apply(self.decoder.input_tensor, self.config.hidden_size)
            statistic = statistic.reshape(-1)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        hidden_states, statistic, parts = self._decode(hidden_states, statistic, rotary_pos_emb, exact)
        if not self.post_process:
            return _AppendStageStatistic.apply(hidden_states, statistic.view(*hidden_states.shape[:-1], 1))
        hidden_states = self.decoder.final_layernorm(hidden_states, parts)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            mtp_in_postprocess=self.mtp_process,
            attention_mask=attention_mask,
        )

    def _decode(
        self, hidden_states: torch.Tensor, statistic: torch.Tensor, rotary_pos_emb, exact: ExactForward
    ) -> tuple[torch.Tensor, torch.Tensor, ResidualSum]:
        """Run this stage's layers, each as its own checkpoint unit under full activation recompute."""
        config = self.config
        checkpointing = config.recompute_granularity == FULL_RECOMPUTE and self.training
        for layer in self.decoder.layers:
            if checkpointing:
                hidden_states, statistic, *parts = tensor_parallel.checkpoint(
                    _layer_unit(layer, rotary_pos_emb, exact),
                    config.distribute_saved_activations,
                    hidden_states,
                    statistic,
                )
                parts = ResidualSum(*parts)
            else:
                hidden_states, statistic, parts = layer(hidden_states, statistic, rotary_pos_emb, exact, keep=False)
        return hidden_states, statistic, parts
