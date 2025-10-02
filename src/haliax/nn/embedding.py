# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
import warnings

import equinox as eqx
from jaxtyping import PRNGKeyArray

import haliax as hax

from ..axis import Axis, AxisSpec, concat_axes
from ..core import NamedArray
from ..jax_utils import named_call
from ..tree_util import resize_axis


class Embedding(eqx.Module):
    weight: NamedArray

    # axes
    Vocab: Axis = eqx.field(static=True)
    Embed: AxisSpec = eqx.field(static=True)

    @staticmethod
    def init(Vocab: Axis, Embed: AxisSpec, *, init_scale: float = 1, key, initializer_range: float | None = None):
        """
        Initialize an Embedding module.

        An embedding module is a simple lookup table that maps integer indices to vectors or tensors.
        Weights are initialized with a truncated normal distribution with a standard deviation of
          `init_scale / output_size`.

        Args:
            Vocab: Size of the vocabulary
            Embed: Shape of the embedding vectors. May be a single axis or a full AxisSpec
            init_scale: Scale of the initialization
            key: PRNG key
            initializer_range: Deprecated. Use init_scale instead.
        """
        if initializer_range is not None:
            warnings.warn("initializer_range is deprecated. Use init_std instead.", DeprecationWarning)
            init_scale = initializer_range

        all_axes = concat_axes(Vocab, Embed)
        output_size = hax.axis_size(Embed)
        weight = hax.random.truncated_normal(key, all_axes, -3, 3) * (init_scale / output_size)
        return Embedding(weight=weight, Vocab=Vocab, Embed=Embed)

    def __call__(self, input_ids: NamedArray, *, key: PRNGKeyArray | None = None):
        """Alias for `embed`. key is ignored."""
        return self.embed(input_ids)

    @named_call
    def embed(self, input_ids: NamedArray):
        """
        Args:
            input_ids: token IDs with shape > {Vocab}
        """
        input_embeds = self.weight.take(self.Vocab, input_ids)
        return input_embeds

    def unembed(self, input_embeds: NamedArray):
        """
        Unembed the input embeddings back to the vocabulary space.

        Equivalent to `input_embeds.dot(self.weight, axis=self.Embed)`.
        """
        return input_embeds.dot(self.weight, axis=self.Embed)

    def resize_embeddings(self, new_size: int, key: PRNGKeyArray | None = None):
        """
        Resize the embedding layer to a new size.
        Args:
            new_size: New size of the vocabulary
            key: PRNG key for initialization of any new weights

        Returns:
            Embedding: Resized embedding layer

        """
        new_weights = resize_axis(self.weight, self.Vocab, new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), weight=new_weights)  # type: ignore
