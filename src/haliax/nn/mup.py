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
    """Abstract base class for abc-parameterization rules.

    Defines the interface for active scaling of parameters (a),
    computing initialization scales (b), and learning rate scaling (c)

    See: https://arxiv.org/abs/2011.14522
    """

    @staticmethod
    @abstractmethod
    def init_scale(In: AxisSpec, Out: AxisSpec):
        """Return the scaling factor for initializing weights
        given input and output axes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def lr_scale(self):
        """Return the learning-rate scaling factor."""
        raise NotImplementedError

    @property
    @abstractmethod
    def active_scale(self):
        """Return the scaling applied to activations."""
        raise NotImplementedError


@dataclass
class AbstractLinearReparam(AbstractReparam):
    """Base class for linear-layer reparameterizations.

    Stores input and output axis specifications, and inherits
    the reparameterization interface.
    """

    In: AxisSpec
    Out: AxisSpec


class LinearStandardParam(AbstractLinearReparam):
    """Standard (non-muP) parameterization for linear layers.

    Uses the usual fan-in scaling for initialization and
    leaves learning rate and activation scaling unchanged.
    """

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
    """muP-style parameterization for input linear layers.

    Uses no scaling on initialization or learning rate.
    See: https://arxiv.org/abs/2011.14522 (Maximal Update Parametrization)
    """

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
    """muP-style parameterization for hidden linear layers.

    Applies fan-in scaling at initialization and scales
    learning rate inversely with layer width.
    """

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
    """muP-style parameterization for output linear layers.

    Uses unit initialization and applies inverse-width
    scaling to the output activations.
    """

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
    """Base class for embedding-layer reparameterizations.

    Defines the interface for both embedding and unembedding
    scaling rules.
    """

    Embed: AxisSpec
    Vocab: AxisSpec

    @property
    @abstractmethod
    def unembed_active_scale(self):
        """Scaling factor applied when unembedding embeddings."""
        raise NotImplementedError


class EmbeddingStandardParam(AbstractEmbeddingReparam):
    """Standard embedding parameterization."""

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
    """muP-style parameterization for embeddings.

    Keeps initialization and learning-rate scaling neutral,
    but applies inverse-width scaling to unembedding outputs for tied weights.
    See: https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization
    """

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
    """Mixin for modules that support reparameterization.

    Stores an abstract `reparam` attribute that specifies
    how initialization and scaling are handled.
    """

    reparam: eqx.AbstractVar[AbstractReparam]
