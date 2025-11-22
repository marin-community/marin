# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Optional, Type

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import cross_entropy_loss

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig


class AudioTextExample(eqx.Module):
    audio: hax.NamedArray
    tokens: hax.NamedArray
    loss_weight: hax.NamedArray
    attn_mask: AttentionMask | hax.NamedArray = AttentionMask.causal()

    @staticmethod
    def init(
        audio: hax.NamedArray,
        tokens: hax.NamedArray,
        *,
        attn_mask: Optional[hax.NamedArray | AttentionMask] = None,
        loss_weight: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
    ) -> "AudioTextExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        Pos = tokens.axes[0]

        # don't predict the last token.
        if loss_weight is None:
            loss_weight = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)
        else:
            loss_weight = loss_weight.astype(jnp.result_type(loss_weight.dtype, jnp.float32))

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_weight = loss_weight * ignore_mask.astype(loss_weight.dtype)

        return AudioTextExample(audio=audio, tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)


class ASRConfig(LmConfig):
    def build_asr(self, Vocab: Axis, *, key: PRNGKeyArray) -> "ASRMixin":
        return self.asr_model_type.init(Vocab, self, key=key)  # type: ignore

    @property
    @abc.abstractmethod
    def asr_model_type(cls) -> Type["ASRMixin"]:
        pass


class ASRMixin(abc.ABC):
    """
    Superclass for models performing ASR
    """

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Pos(self) -> Axis:
        pass

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "ASRMixin":
        """
        Resizes the vocabulary of the ASR Output space. Key may be provided to use random initialization, otherwise,
        there should be some deterministic initialization of any new parameters.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        pass

    def compute_loss(
        self,
        example: AudioTextExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jnp.ndarray | NamedArray:
        """
        Computes the cross-entropy loss for predicted ASR tokens. If reduction is not None, the loss is reduced
        across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
        reduced, and the result is a named array with axes (*batch axes, sequence_length).
        """
        logits = self(example.audio, example.tokens, example.attn_mask, key=key)
        logits = logits.astype(jnp.float32)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        loss = cross_entropy_loss(
            logits, self.Vocab, target_y, reduction, reduction_axis=reduction_axis, weight=example.loss_weight
        )

        return loss

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
