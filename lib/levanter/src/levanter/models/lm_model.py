# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, Type, TypeVar, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray, NamedOrNumeric

from levanter.layers.attention import AttentionMask
from levanter.models.loss import maybe_fused_next_token_loss


LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")

if TYPE_CHECKING:
    from levanter.data.text.examples import GrugLmExample


class LmExample(eqx.Module):
    tokens: hax.NamedArray
    loss_weight: hax.NamedArray
    attn_mask: AttentionMask | NamedArray = AttentionMask.causal()

    @staticmethod
    def causal(
        tokens: hax.NamedArray,
        *,
        loss_weight: Optional[hax.NamedArray] = None,
        ignore_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        segment_ids: Optional[hax.NamedArray] = None,
        sliding_window: int | None = None,
        block_cross_document_attention: bool = True,
    ) -> "LmExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        Pos = tokens.axes[0]

        causal_loss_mask = LmExample.causal_loss_mask(Pos)

        if loss_weight is not None:
            dtype = jnp.result_type(loss_weight.dtype, jnp.float32)
            loss_weight = loss_weight.astype(dtype) * causal_loss_mask.astype(dtype)
        else:
            dtype = jnp.float32
            loss_weight = causal_loss_mask.astype(dtype)

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            ignore_mask = ignore_mask.astype(loss_weight.dtype)
            loss_weight = loss_weight * ignore_mask

        loss_weight = loss_weight.astype(dtype)

        attn_mask = AttentionMask.causal(sliding_window=sliding_window)

        if block_cross_document_attention:
            if eos_id is not None and segment_ids is None:
                # the next token after an eos token is in a new segment
                eos_mask = hax.roll(tokens, 1, Pos) == eos_id
                # first token is always in segment 0
                eos_mask = eos_mask.at[Pos, 0].set(False).astype(jnp.int32)
                segment_ids = hax.cumsum(eos_mask, axis=Pos)
                attn_mask = attn_mask.with_segment_ids(segment_ids)
            elif segment_ids is not None:
                attn_mask = attn_mask.with_segment_ids(segment_ids)

        return LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)

    @staticmethod
    def from_prompt_and_completion(
        Pos,
        tokens: hax.NamedArray,
        prompt_length: NamedOrNumeric,
        *,
        ignore_id: Optional[int] = None,
        all_causal: bool = True,
        sliding_window: int | None = None,
    ) -> "LmExample":
        if all_causal:
            attn_mask = AttentionMask.causal(sliding_window=sliding_window)
        else:
            # causal just for the completion part. We don't have a special structured mask for this, so we just
            raise NotImplementedError("Not implemented yet")

        # mask out the prompt tokens
        loss_weight = LmExample.causal_loss_mask(Pos, prompt_length=prompt_length).astype(jnp.float32)

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_weight = loss_weight * ignore_mask.astype(loss_weight.dtype)

        return LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)

    @staticmethod
    def causal_loss_mask(Pos: Axis, prompt_length: Optional[int] = None) -> NamedArray:
        loss_weight = hax.logical_not(hax.nn.one_hot(-1, Pos, dtype=jnp.bool_))

        if prompt_length is not None:
            # don't predict the prompt tokens
            prompt_mask = hax.arange(Pos) >= prompt_length - 1
            loss_weight = hax.logical_and(loss_weight, prompt_mask)

        return loss_weight


# TODO: for some reason, mypy doesn't like the discover_packages_path argument?
@dataclass(frozen=True)
class LmConfig(draccus.PluginRegistry, abc.ABC, Generic[LmT], discover_packages_path="levanter.models"):  # type: ignore
    max_seq_len: int

    @property
    @abc.abstractmethod
    def model_type(cls) -> Type[LmT]:
        pass

    @property
    def KeyPos(self) -> Axis:
        return self.max_Pos.alias("key_position")

    @property
    def max_Pos(self) -> Axis:
        return Axis("position", self.max_seq_len)

    @property
    @abc.abstractmethod
    def Embed(self) -> Axis:
        pass

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        return None

    def total_trainable_params(self, vocab_size: int) -> Optional[float]:
        return None

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "LmT":
        return self.model_type.init(Vocab, self, key=key)  # type: ignore


class LmHeadModel(eqx.Module, Generic[LmConfigT]):
    """
    Superclass for models with a language modeling head.
    """

    @property
    @abc.abstractmethod
    def config(self) -> LmConfigT:
        pass

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @property
    def Pos(self) -> Axis:
        return self.config.max_Pos

    @property
    def KeyPos(self) -> Axis:
        return self.config.KeyPos

    @property
    def max_length(self) -> int:
        """Maximum sequence length the model supports for inputs."""
        return self.config.max_seq_len

    @property
    def Embed(self) -> Axis:
        return self.config.Embed

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: LmConfigT, *, key: PRNGKeyArray) -> "LmHeadModel[LmConfigT]":
        pass

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        """
        Compute the logits for the next token in a sequence.
        Args:
            input_ids: token IDs with shape [..., Pos]
            attn_mask: attention mask with shape [..., Pos, KeyPos]
            key: PRNGKeyArray for random number generation

        Returns:
            NamedArray: logits with shape [..., Pos, Vocab]

        """
        try:
            x = self.activations(input_ids, attn_mask, key=key, pos_ids=pos_ids)
        except TypeError:
            # For backward compatibility with models that don't yet support pos_ids
            x = self.activations(input_ids, attn_mask, key=key)

        lm_logits = hax.dot(x, self.get_lm_head(), axis=self.Embed)

        return lm_logits

    def logits_from_token_ids_array(
        self,
        input_ids: jax.Array,
        *,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        batch_axis: Axis | str | None = "batch",
        key=None,
    ) -> jax.Array:
        """
        Compute logits from array token IDs and return a plain JAX array.

        This adapter is intended for migration paths that avoid named tensors at
        call sites while keeping existing model internals unchanged.
        """
        resolved_batch_axis: Axis | None = None
        if input_ids.ndim == 1:
            Pos = self.Pos.resize(input_ids.shape[0])
            named_input_ids = hax.named(input_ids, Pos)
        elif input_ids.ndim == 2:
            if batch_axis is None:
                Batch = Axis("batch", input_ids.shape[0])
            elif isinstance(batch_axis, str):
                Batch = Axis(batch_axis, input_ids.shape[0])
            else:
                Batch = batch_axis
                if Batch.size != input_ids.shape[0]:
                    raise ValueError(
                        f"Batch axis size ({Batch.size}) must match input batch size ({input_ids.shape[0]})."
                    )
            resolved_batch_axis = Batch

            Pos = self.Pos.resize(input_ids.shape[1])
            named_input_ids = hax.named(input_ids, (Batch, Pos))
        else:
            raise ValueError(f"input_ids must have rank 1 or 2, got rank={input_ids.ndim}")

        if attn_mask is not None:
            from levanter.data.text.examples import named_attention_mask_from_grug
            from levanter.grug.attention import AttentionMask as GrugAttentionMask

            if isinstance(attn_mask, GrugAttentionMask):
                attn_mask = named_attention_mask_from_grug(attn_mask, Pos, batch_axis=resolved_batch_axis)

        activations = self.activations(named_input_ids, attn_mask=attn_mask, key=key)
        if isinstance(activations, tuple):
            activations, _ = activations
        logits = hax.dot(activations, self.get_lm_head(), axis=self.Embed)
        return logits.array

    @abc.abstractmethod
    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray | tuple[NamedArray, NamedArray | float]:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKeyArray for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        pass

    @abc.abstractmethod
    def get_lm_head(self) -> hax.NamedArray:
        """
        The language modeling head of the model. Should have shape {Embed, Vocab}.
        """
        raise NotImplementedError("get_lm_head not implemented")

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "LmHeadModel[LmConfigT]":
        """
        Resizes the vocabulary of the model. Key may be provided to use random initialization, otherwise, there
        should be some deterministic initialization of any new parameters.
        """
        pass

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
        axis_mapping: hax.partitioning.ResourceMapping | None = None,
    ) -> jnp.ndarray | NamedArray:
        """
        Compute next-token cross-entropy for a language modeling example.

        If `reduction` is not None, the loss is reduced across `reduction_axis` (`None` means all axes).
        If `reduction` is None, the loss is returned unreduced as a `NamedArray` with axes
        (*batch axes, sequence_length).
        """
        activations = self.activations(example.tokens, example.attn_mask, key=key)

        aux_loss = 0
        if isinstance(activations, tuple):
            activations, aux_loss = activations

        loss = maybe_fused_next_token_loss(
            self.Pos,
            self.Embed,
            self.Vocab,
            activations,
            self.get_lm_head(),
            example.tokens,
            loss_weight=example.loss_weight,
            reduction=reduction,
            reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            logit_soft_cap=logit_soft_cap,
            axis_mapping=axis_mapping,
        )

        return loss + aux_loss

    def compute_next_token_loss_array(
        self,
        example: "LmExample | GrugLmExample",
        *,
        batch_axis: Axis | str | None = "batch",
        key=None,
        reduction: Optional[hax.ReductionFunction] = cast(Optional[hax.ReductionFunction], hax.mean),
        reduction_axis: Optional[hax.AxisSelection] = None,
        logsumexp_weight: Optional[float] = None,
        loss_dtype: Optional[jnp.dtype] = jnp.float32,
        logit_soft_cap: Optional[float] = None,
    ) -> jax.Array:
        """
        Compute next-token cross-entropy and always return a plain JAX array.

        This bridges array-native batches (for example `GrugLmExample`) to the legacy
        named-tensor model interface during migration.
        """
        named_example: LmExample
        if isinstance(example, LmExample):
            named_example = example
        else:
            from levanter.data.text.examples import GrugLmExample, named_lm_example_from_grug

            if not isinstance(example, GrugLmExample):
                raise TypeError(f"Unsupported example type: {type(example)}")
            Pos = self.Pos.resize(example.tokens.shape[-1])
            named_example = named_lm_example_from_grug(example, Pos=Pos, batch_axis=batch_axis)

        loss = self.compute_next_token_loss(
            named_example,
            key=key,
            reduction=reduction,
            reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight,
            loss_dtype=loss_dtype,
            logit_soft_cap=logit_soft_cap,
        )

        if isinstance(loss, hax.NamedArray):
            return loss.array
        return jnp.asarray(loss)
