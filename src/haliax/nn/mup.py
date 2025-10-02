# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
import math
from typing import Optional

from jax.random import PRNGKey

import haliax as hax

from .._src.state_dict import (
    Mod,
    default_eqx_module_from_state_dict,
    default_eqx_module_to_state_dict,
    StateDict,
)
from ..axis import AxisSpec
from ..jax_utils import named_call
from ..quantization import DotGeneralOp

from .linear import Linear


class MupMixin:
    @staticmethod
    def mup_init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @property
    def mup_lr_scale(self):
        return 1

    @property
    def mup_active_scale(self):
        return 1


class MupLinear(Linear, MupMixin):
    @classmethod
    def init(
        cls,
        In: AxisSpec,
        Out: AxisSpec,
        *,
        key: PRNGKey,
        use_bias: bool = True,
        out_first: bool = True,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ) -> "MupLinear":
        """
        Args:
            In: AxisSpec: The input axis spec
            Out: AxisSpec: The output axis spec
            key: PRNGKeyArray: The PRNG key to use for initialization
            use_bias: bool: Whether to use a bias term
            out_first: bool: Whether to put output axes first in the weight matrix. out_first is how PyTorch does it.
            dot_general: Callable: The dot_general function to use. Defaults to jax.lax.dot_general.
            init_scale: float: The scale to use for initialization. We scale init by 1/sqrt(Input.size)*init_scale
        """
        joint_spec = hax.concat_axis_specs(Out, In) if out_first else hax.concat_axis_specs(In, Out)

        weight = hax.random.truncated_normal(key, joint_spec, -3, 3) * (init_scale * cls.mup_init_scale(In, Out))
        bias = hax.zeros(Out) if use_bias else None

        if dot_general is None:
            dot_general = DotGeneralOp.default()

        return cls(weight, bias, In, Out, dot_general=dot_general)

    @named_call
    def __call__(self, inputs, *, key: Optional[PRNGKey] = None):
        """
        Args:
            inputs (NamedArray): Input array
            key: Not used, but there for compat with other modules
        """
        del key
        q = inputs.dot(
            self.weight * self.mup_active_scale,
            axis=self.In,
            dot_general=self.dot_general,
        )
        q = hax.auto_sharded(q)

        if self.bias is not None:
            q = q + self.bias
            q = hax.auto_sharded(q)

        return q

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        scaled = dataclasses.replace(self, weight=self.weight * self.mup_active_scale)
        return default_eqx_module_to_state_dict(scaled, prefix)

    def from_state_dict(self: Mod, state_dict: StateDict, prefix: Optional[str] = None) -> Mod:
        unscaled = default_eqx_module_from_state_dict(self, state_dict, prefix)
        return dataclasses.replace(unscaled, weight=unscaled.weight / self.mup_active_scale)


class InputLinear(MupLinear):
    pass


class OutputLinear(MupLinear):
    @property
    def mup_active_scale(self):
        return 1 / hax.axis_size(self.In)


class HiddenLinear(MupLinear):
    @property
    def mup_lr_scale(self):
        return 1 / hax.axis_size(self.In)

    @property
    def mup_init_scale(self):
        return 1 / math.sqrt(hax.axis_size(self.In))
