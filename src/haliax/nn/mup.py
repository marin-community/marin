# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod
import math

import haliax as hax
import equinox as eqx

from ..axis import AxisSpec


class AbstractReparam(ABC):
    @staticmethod
    @abstractmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        raise NotImplementedError

    @abstractmethod
    def lr_scale(self):
        raise NotImplementedError

    @abstractmethod
    def active_scale(self):
        raise NotImplementedError


class AbstractLinearMup(AbstractReparam):
    In: AxisSpec
    Out: AxisSpec


class DefaultLinearMup(AbstractLinearMup):
    In: AxisSpec
    Out: AxisSpec

    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1 / hax.axis_size(In)

    @abstractmethod
    def active_scale(self):
        return 1

    @abstractmethod
    def lr_scale(self):
        return 1


class StandardLinear(AbstractLinearMup):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @abstractmethod
    def active_scale(self):
        return 1

    @abstractmethod
    def lr_scale(self):
        return 1


class HiddenLinearMup(AbstractLinearMup):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1 / math.sqrt(hax.axis_size(In))

    @abstractmethod
    def active_scale(self):
        return 1

    @abstractmethod
    def lr_scale(self):
        return 1 / hax.axis_size(self.In)


class OutputLinearMup(AbstractLinearMup):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @abstractmethod
    def active_scale(self):
        return 1 / hax.axis_size(self.In)

    @abstractmethod
    def lr_scale(self):
        return 1


class AbstractEmbeddingReparam(AbstractReparam):
    Embed: AxisSpec
    Vocab: AxisSpec


class ReparamEnabled(ABC):
    mup_config: eqx.AbstractVar[AbstractReparam]
