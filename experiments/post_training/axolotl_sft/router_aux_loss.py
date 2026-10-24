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
"""

import functools
import logging

from axolotl.integrations.base import BasePlugin

LOG = logging.getLogger(__name__)


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
            LOG.info("router aux loss on the first forward: %.6f", float(outputs.aux_loss))
            checked = True
        return outputs

    model.forward = forward


class RouterAuxLossPlugin(BasePlugin):
    """Add the router load-balancing loss, at the config's ``router_aux_loss_coef``, to training."""

    def post_model_load(self, cfg, model):
        _pass_router_logits(model)
