# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl plugin: train the MoE router through its auxiliary loss (pod-side, image environment).

Axolotl reads no ``output_router_logits`` setting, and the pinned Qwen configs default it
to false, so without this plugin the router trains through the LM loss alone. Setting
``model.config.output_router_logits`` is not enough either: Qwen3.5-MoE loads as the
``Qwen3_5MoeForConditionalGeneration`` shell, whose forward (and Cut Cross Entropy's
replacement of it, patched on the class before load) adds the aux loss only when the
``output_router_logits`` keyword is passed. This plugin passes that keyword on every forward,
which the causal-LM (Qwen3-MoE) and the shell (Qwen3.5-MoE) forwards both honor, and fails
the first forward if no aux loss comes back.

transformers' ``load_balancing_loss_func`` one-hot encodes every layer's top-k choices
(``[layers x tokens, top_k, experts]`` int64, about 21 GB for Qwen3.5 at 32k tokens), which
ran the Qwen3.5 smoke out of memory. The plugin swaps in :func:`load_balancing_loss`, the
same loss accumulated layer by layer from expert counts.
"""

import functools
import logging
import sys

import torch
from axolotl.integrations.base import BasePlugin

# Under "axolotl." so Axolotl's logging config prints it.
LOG = logging.getLogger("axolotl.integrations.router_aux_loss")
LOSS_FUNCTION_NAME = "load_balancing_loss_func"
# Modules whose forwards call the loss by name: transformers' MoE models and CCE's patches of them.
PATCHED_MODULE_PREFIXES = ("transformers.models.", "cut_cross_entropy.transformers.")


def load_balancing_loss(
    gate_logits: tuple[torch.Tensor, ...] | None,
    num_experts: int | None = None,
    top_k: int = 2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    """transformers' Switch-style balancing loss, without the one-hot tensor.

    ``num_experts * sum_e f_e * P_e``: ``f_e`` is the share of tokens whose top-k choices include
    expert ``e`` (summed over the k slots, as transformers does) and ``P_e`` the mean router
    probability, both over every layer's unmasked tokens. Gradients flow through ``P_e`` only.
    """
    if not isinstance(gate_logits, tuple):
        return 0
    device = gate_logits[0].device
    num_experts = gate_logits[0].shape[-1] if num_experts is None else num_experts
    counts = torch.zeros(num_experts, device=device)
    probability_sum = torch.zeros(num_experts, device=device)
    weight = None if attention_mask is None else attention_mask.reshape(-1).to(device, torch.float32)
    tokens = 0.0
    for layer_logits in gate_logits:
        probabilities = torch.softmax(layer_logits.to(device).float(), dim=-1)
        selected = torch.topk(probabilities, top_k, dim=-1).indices
        if weight is None:
            counts = counts + torch.bincount(selected.reshape(-1), minlength=num_experts)
            probability_sum = probability_sum + probabilities.sum(dim=0)
            tokens += probabilities.shape[0]
        else:
            slot_weight = weight[:, None].expand_as(selected).reshape(-1)
            counts = counts + torch.bincount(selected.reshape(-1), weights=slot_weight, minlength=num_experts)
            probability_sum = probability_sum + (probabilities * weight[:, None]).sum(dim=0)
            tokens += weight.sum()
    return num_experts * torch.sum((counts / tokens) * (probability_sum / tokens))


def _replace_loss_function() -> list[str]:
    patched = []
    for name, module in list(sys.modules.items()):
        if name.startswith(PATCHED_MODULE_PREFIXES) and hasattr(module, LOSS_FUNCTION_NAME):
            setattr(module, LOSS_FUNCTION_NAME, load_balancing_loss)
            patched.append(name)
    return patched


def _pass_router_logits(model) -> None:
    original = model.forward
    checked = False

    # A plain function over the bound original, wrapped so its signature is the original's:
    # the Trainer keeps only the dataset columns that the forward's signature names.
    @functools.wraps(original)
    def forward(*args, **kwargs):
        nonlocal checked
        kwargs["output_router_logits"] = True
        outputs = original(*args, **kwargs)
        if not checked:
            if outputs.aux_loss is None:
                raise RuntimeError(f"{type(model).__name__} returned no aux_loss with output_router_logits=True")
            LOG.info("router aux loss on the first forward: %.6f", outputs.aux_loss.item())
            checked = True
        return outputs

    model.forward = forward


class RouterAuxLossPlugin(BasePlugin):
    """Add the router load-balancing loss, at the config's ``router_aux_loss_coef``, to training."""

    def post_model_load(self, cfg, model):
        LOG.info("balancing loss replaced in %s", _replace_loss_function())
        _pass_router_logits(model)
