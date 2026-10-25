# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt saved TPU expert activations to native and local-token MoE."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from levanter.grug._moe import ep_ragged_all_to_all as ep

from experiments.grug.belay_tpu.saved_relu2 import make_saved_relu2_mlp

saved_relu2_mlp = make_saved_relu2_mlp()
_source_selector = ep._select_expert_mlp


@dataclass(frozen=True)
class SavedRelu2ExpertMlp:
    interpret: bool = False

    def forward(self, x, up, down, physical_groups, active_groups):
        active_rows = active_groups.sum()
        mlp = make_saved_relu2_mlp(interpret=True) if self.interpret else saved_relu2_mlp
        out, pullback = jax.vjp(lambda x, u, d: mlp(x, u, d, physical_groups, active_rows), x, up, down)
        return out, (out, pullback, active_rows)

    def backward(self, residuals, cotangent):
        out, pullback, active_rows = residuals
        active = jnp.arange(out.shape[0])[:, None] < active_rows
        cotangent = jnp.where(active, cotangent, 0)
        dx, dup, ddown = pullback(cotangent)
        output_dot = jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32), axis=-1)
        return dx, dup, ddown, output_dot

    backward_parts = ep._whole_backward_parts


def select_saved_relu2(activation_fn, *, ungated):
    if ungated and activation_fn in ep._RELU2_FNS:
        return SavedRelu2ExpertMlp()
    return _source_selector(activation_fn, ungated=ungated)
