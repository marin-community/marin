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
    prefix_length: Optional[int] = eqx.field(default=None, static=True)
    prefix_lengths: Optional[NamedArray] = None
    prefix_mask: Optional[NamedArray] = None

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

        if self.sliding_window is not None:
            sw_mask = _materialize_sliding_window_mask(
                self.sliding_window, QPos, KPos, q_slice=q_slice, k_slice=k_slice
            )
            causal = combine_masks_and(causal, sw_mask)

        prefix = None
        if self.prefix_length is not None:
            sub_k = KPos.resize(k_slice.size)
            prefix = hax.arange(sub_k) + k_slice.start < self.prefix_length
            prefix = prefix.broadcast_axis(QPos.resize(q_slice.size))

        if self.prefix_lengths is not None:
            sub_k = KPos.resize(k_slice.size)
            kv_positions = hax.arange(sub_k) + k_slice.start
            dynamic_prefix = kv_positions < self.prefix_lengths.broadcast_axis(sub_k)
            dynamic_prefix = dynamic_prefix.broadcast_axis(QPos.resize(q_slice.size))
            prefix = combine_masks_or(prefix, dynamic_prefix)

        if self.prefix_mask is not None:
            prefix_mask = self.prefix_mask.rename({QPos.name: KPos.name})[KPos.name, k_slice]
            prefix = combine_masks_or(prefix, prefix_mask.broadcast_axis(QPos.resize(q_slice.size)))

        causal = combine_masks_or(causal, prefix)

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        mask = combine_masks_and(causal, explicit)

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
    def prefix_lm(
        *,
        prefix_length: int | None = None,
        prefix_lengths: NamedArray | None = None,
        prefix_mask: NamedArray | None = None,
        sliding_window: Optional[int] = None,
        segment_ids: tuple[NamedArray, NamedArray] | None = None,
    ) -> "AttentionMask":
        """Create a prefix-LM mask.

        Keys marked as prefix are visible to every query. Non-prefix keys use
        causal masking, optionally constrained by ``sliding_window``.
        """
        if prefix_length is None and prefix_lengths is None and prefix_mask is None:
            raise ValueError("prefix_lm requires prefix_length, prefix_lengths, or prefix_mask.")
        if prefix_length is not None and prefix_length < 0:
            raise ValueError(f"prefix_length must be non-negative, got {prefix_length}.")

        return AttentionMask(
            is_causal=True,
            prefix_length=prefix_length,
            prefix_lengths=prefix_lengths,
            prefix_mask=prefix_mask,
            sliding_window=sliding_window,
            segment_ids=segment_ids,
        )

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, causal_offset=None, explicit_mask=mask)

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
            prefix_length=self.prefix_length,
            prefix_lengths=self.prefix_lengths,
            prefix_mask=self.prefix_mask,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        """Return a copy of this mask with ``sliding_window`` applied."""
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
            prefix_length=self.prefix_length,
            prefix_lengths=self.prefix_lengths,
            prefix_mask=self.prefix_mask,
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
        prefix_mask = combine_masks_and(self.prefix_mask, other.prefix_mask)
        prefix_length = _combine_prefix_lengths_and(self.prefix_length, other.prefix_length)
        prefix_lengths = _combine_dynamic_prefix_lengths_and(self.prefix_lengths, other.prefix_lengths)
        if self.sliding_window is None:
            sliding_window = other.sliding_window
        elif other.sliding_window is None:
            sliding_window = self.sliding_window
        else:
            sliding_window = min(self.sliding_window, other.sliding_window)

        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
            prefix_length=prefix_length,
            prefix_lengths=prefix_lengths,
            prefix_mask=prefix_mask,
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
        prefix_mask = combine_masks_or(self.prefix_mask, other.prefix_mask)
        prefix_length = _combine_prefix_lengths_or(self.prefix_length, other.prefix_length)
        prefix_lengths = _combine_dynamic_prefix_lengths_or(self.prefix_lengths, other.prefix_lengths)
        if self.sliding_window is None or other.sliding_window is None:
            sliding_window = None
        else:
            sliding_window = max(self.sliding_window, other.sliding_window)
        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
            prefix_length=prefix_length,
            prefix_lengths=prefix_lengths,
            prefix_mask=prefix_mask,
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


def _combine_prefix_lengths_and(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _combine_prefix_lengths_or(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _combine_dynamic_prefix_lengths_and(left: NamedArray | None, right: NamedArray | None) -> NamedArray | None:
    if left is None:
        return right
    if right is None:
        return left
    return hax.minimum(left, right)


def _combine_dynamic_prefix_lengths_or(left: NamedArray | None, right: NamedArray | None) -> NamedArray | None:
    if left is None:
        return right
    if right is None:
        return left
    return hax.maximum(left, right)


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
