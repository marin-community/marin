# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl plugin: fp32 master weights under bf16 compute (pod-side, image environment).

With ``bf16: true`` Axolotl loads the model in bf16 and the optimizer updates bf16
parameters in place. At Stage 0's learning rate (5e-6) most updates fall below
bf16's resolution for typical weight magnitudes and are lost. Levanter, which trains
Snowball, keeps fp32 master parameters and computes in bf16; this plugin does the
same here: the model loads in fp32, and FSDP2's bf16 mixed-precision policy (from
``bf16: true``) casts parameters to bf16 for compute only. A callback asserts the
sharded parameters and the Adam state are fp32 at the first step.
"""

import logging

import torch
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

LOG = logging.getLogger(__name__)


class _MasterDtypeCheck(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if state.global_step != 1:
            return
        parameter_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
        state_dtypes = {
            value.dtype
            for per_param in optimizer.state.values()
            for value in per_param.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dim() > 0
        }
        LOG.info("fp32 master check: parameter dtypes %s, optimizer state dtypes %s", parameter_dtypes, state_dtypes)
        if parameter_dtypes != {torch.float32} or state_dtypes - {torch.float32}:
            raise RuntimeError(f"expected fp32 masters, got parameters {parameter_dtypes} and state {state_dtypes}")


class Fp32MasterWeightsPlugin(BasePlugin):
    """Load the model in fp32 so the optimizer updates fp32 masters."""

    def pre_model_load(self, cfg):
        cfg.torch_dtype = torch.float32

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_MasterDtypeCheck()]
