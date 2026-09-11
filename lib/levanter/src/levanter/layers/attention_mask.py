# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured attention-mask primitives."""

import warnings
from numbers import Integral
from typing import Optional, cast, overload

import equinox as eqx
import haliax
import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or


def _materialize_segment_mask(
    segment_ids: NamedArray | tuple[NamedArray, NamedArray],
    QPos,
    KPos,
    q_slice,
    k_slice,
) -> NamedArray:
    """
    Make a segment mask for attention. This is a mask that prevents attention between different segments.
    """
    if isinstance(segment_ids, tuple):
        if len(segment_ids) != 2:
            raise ValueError("segment_ids must be a tuple of two NamedArrays")
        q_segment_ids, kv_segment_ids = segment_ids
        kv_segment_ids = kv_segment_ids.rename({QPos.name: KPos.name})[KPos.name, k_slice]
        q_segment_ids = q_segment_ids.rename({QPos.name: QPos})[QPos.name, q_slice]
    else:
        kv_segment_ids = segment_ids.rename({QPos.name: KPos.name})[KPos.name, k_slice]
        q_segment_ids = segment_ids[QPos.name, q_slice]

    return cast(NamedArray, q_segment_ids.broadcast_axis(kv_segment_ids.axes) == kv_segment_ids)


def _materialize_sliding_window_mask(
    window: int, QPos: Axis, KPos: Axis, q_slice: haliax.dslice, k_slice: haliax.dslice
) -> NamedArray:
    """Materialize a causal sliding window mask."""
    sub_q = QPos.resize(q_slice.size)
    sub_k = KPos.resize(k_slice.size)
    q_pos = hax.arange(sub_q) + q_slice.start
    k_pos = hax.arange(sub_k) + k_slice.start
    diff = q_pos.broadcast_axis(sub_k) - k_pos.broadcast_axis(sub_q)
    return (diff >= 0) & (diff < window)


def _materialize_bidirectional_window_mask(
    radius: int, QPos: Axis, KPos: Axis, q_slice: haliax.dslice, k_slice: haliax.dslice
) -> NamedArray:
    """Materialize a symmetric (bidirectional) sliding window mask: query i attends to key j iff |i - j| <= radius."""
    sub_q = QPos.resize(q_slice.size)
    sub_k = KPos.resize(k_slice.size)
    q_pos = hax.arange(sub_q) + q_slice.start
    k_pos = hax.arange(sub_k) + k_slice.start
    diff = q_pos.broadcast_axis(sub_k) - k_pos.broadcast_axis(sub_q)
    return (diff <= radius) & (diff >= -radius)


class AttentionMask(eqx.Module):
    """

    !!! warning
        This class is still experimental. I'm not super happy with it yet.

    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    The abstraction is based on two concepts:

    1) Materialization: An AttentionMask can be materialized for a particular slice of the query and key position axes.
       Most naively, you can just get the whole mask as a NamedArray. However, in some cases, you might want to
       only get a particular chunk (e.g. for flash attention).
    2) Combination: AttentionMasks are represented as an implicit conjunction of multiple masks, each with different
        kinds of structure. You can combine masks with `&` and `|`. Due to the way jit works, we don't use inheritance
        or similar to represent different kinds of masks. Instead, we use a single class with different fields.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same set of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    (Perhaps it's ok to use inheritance here? I'm not sure. Splash attention landed on inheritance, so maybe
    that's a good sign.)

    """

    # If ``is_causal`` is True we apply a lower-triangular causal mask. If ``causal_offset`` is not ``None``
    # we apply a shifted causal mask such that a query at position *i* can attend to key *j* whenever
    # ``j <= i + causal_offset``. A ``None`` offset means a static offset of 0 (i.e., standard causal masking).
    is_causal: bool = eqx.field(default=False, static=True)
    causal_offset: None | NamedArray = None
    explicit_mask: Optional[NamedArray] = None
    segment_ids: tuple[NamedArray, NamedArray] | None = None
    sliding_window: Optional[int] = eqx.field(default=None, static=True)
    # Symmetric window radius for bidirectional (encoder) attention: query i attends to key j iff
    # |i - j| <= bidirectional_window. Unlike ``sliding_window`` (which is causal), this looks both ways.
    bidirectional_window: Optional[int] = eqx.field(default=None, static=True)
    # CF https://github.com/jax-ml/jax/blob/47858c4ac2fd4757a3b6fc5bb2981b71a71f00c2/jax/experimental/pallas/ops/tpu/flash_attention.py#L34
    # TODO: add prefixlm
    # cf https://github.com/google-research/t5x/blob/51a99bff8696c373cc03918707ada1e98cbca407/t5x/examples/decoder_only/layers.py#L978

    def materialize(
        self,
        QPos: Axis,
        KPos: Axis,
        q_slice: Optional[haliax.dslice] = None,
        k_slice: Optional[haliax.dslice] = None,
    ) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        if q_slice is None:
            q_slice = haliax.dslice(0, QPos.size)
        if k_slice is None:
            k_slice = haliax.dslice(0, KPos.size)

        if self.is_causal:
            # None means static 0 offset
            offset = 0 if self.causal_offset is None else self.causal_offset
            shifted_k_start = k_slice.start - offset
            if isinstance(shifted_k_start, NamedArray):
                # need to vmap
                causal = hax.vmap(causal_mask, shifted_k_start.axes)(
                    QPos.resize(q_slice.size),
                    KPos.resize(k_slice.size),
                    q_slice.start,
                    shifted_k_start,  # type: ignore
                )
            else:
                causal = causal_mask(
                    QPos.resize(q_slice.size),
                    KPos.resize(k_slice.size),
                    q_slice.start,
                    shifted_k_start,
                )
        else:
            causal = None

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        mask = combine_masks_and(causal, explicit)

        if self.sliding_window is not None:
            sw_mask = _materialize_sliding_window_mask(
                self.sliding_window, QPos, KPos, q_slice=q_slice, k_slice=k_slice
            )
            mask = combine_masks_and(mask, sw_mask)

        if self.bidirectional_window is not None:
            bw_mask = _materialize_bidirectional_window_mask(
                self.bidirectional_window, QPos, KPos, q_slice=q_slice, k_slice=k_slice
            )
            mask = combine_masks_and(mask, bw_mask)

        if self.segment_ids is not None:
            segment_mask = _materialize_segment_mask(self.segment_ids, QPos, KPos, q_slice, k_slice)
            mask = combine_masks_and(mask, segment_mask)

        return mask

    # Static constructors --------------------------------------------------

    @staticmethod
    def causal(
        *,
        sliding_window: Optional[int] = None,
        offset: int | NamedArray | None = None,
        segment_ids: tuple[NamedArray, NamedArray] | None = None,
    ) -> "AttentionMask":
        """Create a causal AttentionMask.

        Args:
            sliding_window: If provided, restrict each query position to attend only to keys within
                ``sliding_window`` previous positions.
            offset:
                For ``offset == 0`` this is identical to the old ``AttentionMask.causal()``
                behaviour; larger offsets loosen the restriction so that each query can
                see ``offset`` additional future tokens.
        """
        if isinstance(offset, int | Integral):
            causal_offset = hax.named(offset, ())
        else:
            causal_offset = offset

        return AttentionMask(
            is_causal=True, causal_offset=causal_offset, sliding_window=sliding_window, segment_ids=segment_ids
        )

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, causal_offset=None, explicit_mask=mask)

    @staticmethod
    def bidirectional_sliding_window(radius: int) -> "AttentionMask":
        """Symmetric (non-causal) sliding window: query i attends to key j iff ``|i - j| <= radius``."""
        return AttentionMask(is_causal=False, bidirectional_window=radius)

    def __post_init__(self):
        # Normalize legacy single-array segment_ids to a tuple for consistency
        if self.segment_ids is not None and not isinstance(self.segment_ids, tuple):
            warnings.warn("Storing segment_ids as a single NamedArray is deprecated. Use a tuple instead.")
            object.__setattr__(self, "segment_ids", (self.segment_ids, self.segment_ids))

    def with_segment_ids(self, segment_ids: NamedArray, kv_segment_ids: NamedArray | None = None) -> "AttentionMask":
        """Attach segment ids to the mask.

        Always stores segment ids internally as a tuple ``(q_segment_ids, kv_segment_ids)``.
        If only a single array is provided, it is used for both queries and keys/values.
        """
        # Always store as a tuple; duplicate if only one provided.
        seg_field: tuple[NamedArray, NamedArray]
        if kv_segment_ids is None:
            seg_field = (segment_ids, segment_ids)
        else:
            seg_field = (segment_ids, kv_segment_ids)

        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=seg_field,
            sliding_window=self.sliding_window,
            bidirectional_window=self.bidirectional_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        """Return a copy of this mask with ``sliding_window`` applied."""
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
            bidirectional_window=self.bidirectional_window,
        )

    def __and__(self, other) -> "AttentionMask":
        # Conjunction: causal if either component is causal.
        if self.is_causal and other.is_causal:
            # If both are causal, offsets must agree if both specified; otherwise take the specified one.
            if self.causal_offset is not None and other.causal_offset is not None:
                causal_offset = eqx.error_if(
                    self.causal_offset,
                    self.causal_offset != other.causal_offset,
                    "Mismatched causal offsets cannot be combined with &",
                )
            else:
                causal_offset = self.causal_offset if self.causal_offset is not None else other.causal_offset
            is_causal = True
        elif self.is_causal:
            causal_offset = self.causal_offset
            is_causal = True
        elif other.is_causal:
            causal_offset = other.causal_offset
            is_causal = True
        else:
            causal_offset = None
            is_causal = False
        explicit_mask = combine_masks_and(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)
        if self.sliding_window is None:
            sliding_window = other.sliding_window
        elif other.sliding_window is None:
            sliding_window = self.sliding_window
        else:
            sliding_window = min(self.sliding_window, other.sliding_window)

        if self.bidirectional_window is None:
            bidirectional_window = other.bidirectional_window
        elif other.bidirectional_window is None:
            bidirectional_window = self.bidirectional_window
        else:
            bidirectional_window = min(self.bidirectional_window, other.bidirectional_window)

        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
            bidirectional_window=bidirectional_window,
        )

    def __or__(self, other) -> "AttentionMask":
        # Union: causal only if both are causal with the same offset; otherwise non-causal
        if (
            self.is_causal
            and other.is_causal
            and (
                (self.causal_offset is None and other.causal_offset is None)
                or (self.causal_offset is not None and self.causal_offset == other.causal_offset)
            )
        ):
            is_causal = True
            causal_offset = self.causal_offset
        else:
            is_causal = False
            causal_offset = None
        explicit_mask = combine_masks_or(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)
        if self.sliding_window is None or other.sliding_window is None:
            sliding_window = None
        else:
            sliding_window = max(self.sliding_window, other.sliding_window)
        if self.bidirectional_window is None or other.bidirectional_window is None:
            bidirectional_window = None
        else:
            bidirectional_window = max(self.bidirectional_window, other.bidirectional_window)
        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
            bidirectional_window=bidirectional_window,
        )

    def _check_for_same_segment_ids(self, other):
        # Normalize possibly non-tuple representations to tuples for comparison.
        def _as_tuple(si):
            if si is None:
                return None
            if isinstance(si, tuple):
                return si
            else:
                return (si, si)

        self_si = _as_tuple(self.segment_ids)
        other_si = _as_tuple(other.segment_ids)

        if self_si is not None and other_si is not None:
            # only one segment mask is allowed
            # b/c we might do this in jit, we use eqx.error_if
            # in theory we can do this one by just assigning unique ids to each unique pair...
            # (but i don't really anticipate needing this)
            segment_ids = eqx.error_if(
                hax.logical_or(self_si[0] != other_si[0], self_si[1] != other_si[1]),
                "Only one segment mask is allowed",
            )
        elif self_si is not None:
            segment_ids = self_si
        else:
            segment_ids = other_si
        return segment_ids


@overload
def materialize_mask(
    mask: NamedArray | AttentionMask,
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> NamedArray: ...


@overload
def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]: ...


def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize(QPos, KPos, q_slice=q_slice, k_slice=k_slice)
        return mask
    elif isinstance(mask, NamedArray):
        if q_slice is not None or k_slice is not None:
            if q_slice is None:
                q_slice = haliax.dslice(0, QPos.size)
            if k_slice is None:
                k_slice = haliax.dslice(0, KPos.size)
            mask = mask[QPos, q_slice, KPos, k_slice]

        return mask
    else:
        assert mask is None
        return None
