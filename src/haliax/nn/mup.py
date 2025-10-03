# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import haliax as hax
import equinox as eqx

from ..axis import AxisSpec


class AbstractReparam(ABC):
    @staticmethod
    @abstractmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        raise NotImplementedError

    @property
    @abstractmethod
    def lr_scale(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def active_scale(self):
        raise NotImplementedError


@dataclass
class AbstractLinearReparam(AbstractReparam):
    In: AxisSpec
    Out: AxisSpec


class LinearStandardParam(AbstractLinearReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1 / math.sqrt(hax.axis_size(In))

    @property
    def active_scale(self):
        return 1

    @property
    def lr_scale(self):
        return 1


class InputLinearMup(AbstractLinearReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @property
    def active_scale(self):
        return 1

    @property
    def lr_scale(self):
        return 1


class HiddenLinearMup(AbstractLinearReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1 / math.sqrt(hax.axis_size(In))

    @property
    def active_scale(self):
        return 1

    @property
    def lr_scale(self):
        return 1 / hax.axis_size(self.In)


class OutputLinearMup(AbstractLinearReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @property
    def active_scale(self):
        return 1 / hax.axis_size(self.In)

    @property
    def lr_scale(self):
        return 1


@dataclass
class AbstractEmbeddingReparam(AbstractReparam):
    Embed: AxisSpec
    Vocab: AxisSpec

    @property
    @abstractmethod
    def unembed_active_scale(self):
        raise NotImplementedError


class EmbeddingStandardParam(AbstractEmbeddingReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1 / hax.axis_size(Out)

    @property
    def active_scale(self):
        return 1

    @property
    def lr_scale(self):
        return 1

    @property
    def unembed_active_scale(self):
        return 1 / hax.axis_size(self.Embed)


class EmbeddingMup(AbstractEmbeddingReparam):
    @staticmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        return 1

    @property
    def active_scale(self):
        return 1

    @property
    def lr_scale(self):
        return 1

    @property
    def unembed_active_scale(self):
        return 1 / hax.axis_size(self.Embed)


class ReparamEnabled(ABC):
    reparam: eqx.AbstractVar[AbstractReparam]
