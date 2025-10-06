# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
from dataclasses import dataclass
from typing import Optional

import draccus

from haliax import AxisSpec
from haliax.nn import LayerNorm, RmsNorm
from haliax.nn.normalization import LayerNormBase


@dataclass(frozen=True)
class LayerNormConfigBase(draccus.ChoiceRegistry, abc.ABC):
    """Base class for layer normalization configurations."""

    eps: float = 1e-6
    use_weight: bool = True
    use_bias: bool = False

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "rms"

    @abc.abstractmethod
    def build(self, axis: AxisSpec) -> LayerNormBase:
        """Build the normalization layer."""
        raise NotImplementedError


@LayerNormConfigBase.register_subclass("rms")
@dataclass(frozen=True)
class RmsNormConfig(LayerNormConfigBase):
    """Configuration for RMS normalization."""

    def build(self, axis: AxisSpec) -> RmsNorm:
        return RmsNorm.init(axis, eps=self.eps, use_weight=self.use_weight, use_bias=self.use_bias)


@LayerNormConfigBase.register_subclass("layer")
@dataclass(frozen=True)
class LayerNormConfig(LayerNormConfigBase):
    """Configuration for standard layer normalization."""

    def build(self, axis: AxisSpec) -> LayerNorm:
        return LayerNorm.init(axis, eps=self.eps, use_weight=self.use_weight, use_bias=self.use_bias)
