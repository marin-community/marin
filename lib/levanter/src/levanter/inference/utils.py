# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

import haliax as hax
import haliax.haxtyping as ht
import numpy as np
from haliax import NamedArray


INVALID = 2_000_000


def is_invalid(x, invalid=INVALID):
    return (x < 0) | (x == invalid)


def is_valid(x, invalid=INVALID):
    """
    Returns a boolean array indicating whether each token in the input is valid.
    A token is considered valid if it is not negative and not equal to INVALID.
    """
    return (x >= 0) & (x != invalid)


def masked_set(dest: NamedArray, axis, start, src, num_to_copy) -> NamedArray:
    """
    jit-safe masked memcpy-like operation.
    Copy into dest[selector, axis, start:start+num_to_copy] the values from src[axis, :num_to_copy].

    Probably faster to not use an arange (which lowers to a scatter) and use blocks? Probably not a bottleneck

    num_to_copy may be dynamic
    """

    src_arange = hax.arange(src.resolve_axis(axis))
    dest_axis_size = dest.axis_size(axis)
    # mask out the tail
    dest_arange = hax.where(src_arange >= num_to_copy, dest_axis_size, src_arange + start)
    src_arange = hax.where(src_arange >= num_to_copy, src_arange.size, src_arange)

    return dest.at[{axis: dest_arange}].set(src[axis, src_arange], mode="drop")


def is_stop_signal(tail_tokens: ht.i32[NamedArray, "position"], stop_sequences: ht.i32[NamedArray, "seq position"], invalid=INVALID) -> ht.bool_[NamedArray, ""]:  # type: ignore
    """
    Check if tail_tokens ends with any of the stop sequences.
    Stop sequences are **left padded** to the tokens.
    """
    if stop_sequences.size == 0:
        return hax.zeros((), dtype=bool)

    # first, which stop sequences are even valid?
    valid_stop_sequences = hax.any(is_valid(stop_sequences, invalid), axis="position")

    # next, count up the number of valid tokens in each stop sequence
    total_stop_tokens = hax.sum(is_valid(stop_sequences, invalid), axis="position")

    count_match = (
        hax.sum((tail_tokens == stop_sequences) & (tail_tokens != invalid), axis="position") == total_stop_tokens
    )

    return hax.any(valid_stop_sequences & count_match)


def purge_raw(array: jnp.array, mask, max_nnz=None, invalid=INVALID) -> jnp.ndarray:
    """
    Set elements of the array to `invalid` where the `mask` is True and slides the rest to the front.

    Args:
        array: The input array to be purged.
        mask: A boolean array indicating which elements to purge (True means purge).
        max_nnz: The maximum size of the array. If None, it is inferred from the array shape.
        invalid: The value to pad with (default is INVALID).

    Returns:
        the purged array.
    """
    if array.ndim != 1:
        raise ValueError("purge function only supports 1D arrays")

    if max_nnz is None:
        max_nnz = array.size

    indices = jnp.nonzero(~mask, size=max_nnz, fill_value=INVALID)[0]
    new_values = array.at[indices].get(mode="fill", fill_value=invalid)

    return new_values


def purge(array: NamedArray, mask: NamedArray, invalid=INVALID) -> NamedArray:
    """
    Set elements of the array to `invalid` where the `mask` is True and slides the rest to the front.

    Args:
        array: The input array to be purged.
        mask: A boolean array indicating which elements to purge (True means purge).
        invalid: The value to pad with (default is INVALID).

    Returns:
        The purged array.
    """
    if array.ndim != 1:
        raise ValueError("purge function only supports 1D arrays")

    new_values = purge_raw(array.array, mask.array, invalid=invalid)
    return hax.named(new_values, array.axes)


def pad_to_standard_length(tokens: np.ndarray, allowed_lengths: list[int], pad_token_id: int) -> np.ndarray:
    """Pad the token array to the nearest allowed length using the pad_token_id."""
    current_length = tokens.shape[0]
    target_length = min((length for length in allowed_lengths if length >= current_length), default=None)

    if target_length is None:
        raise ValueError(f"Current length {current_length} exceeds all allowed lengths {allowed_lengths}")

    padding_length = target_length - current_length
    if padding_length > 0:
        padding = np.full((padding_length,), pad_token_id, dtype=tokens.dtype)
        tokens = np.concatenate([tokens, padding], axis=0)

    return tokens


def get_unique_in_order(array, **kwargs):
    """
    Finds unique elements in a JAX array, preserving the order of first appearance.

    Args:
      array: The input 1D JAX array.
      **kwargs: Additional arguments for jnp.unique (e.g., size, fill_value).

    Returns:
      A tuple of:
        - unique_ids_in_order: The unique values in appearance order.
        - dense_ids_in_order: An array where original values are replaced by their
                              new 0-indexed position in appearance order.
    """
    # jnp.unique sorts the unique values and provides the indices
    # of their first appearance corresponding to that sorted order.
    unique_sorted, first_indices, dense_sorted, counts = jnp.unique(
        array, return_index=True, return_inverse=True, return_counts=True, **kwargs
    )

    first_indices = jnp.where(counts == 0, jnp.iinfo(first_indices.dtype).max, first_indices)

    # To restore the original appearance order, we sort the `first_indices`.
    # The result of argsort gives us the permutation needed.
    perm = jnp.argsort(first_indices)
    unique_ids_in_order = unique_sorted[perm]

    # We also need to remap the dense_ids to match the new order.
    # We can create a remapping array using the inverse of the permutation.
    remap_indices = jnp.argsort(perm)
    dense_ids_in_order = remap_indices[dense_sorted]

    return unique_ids_in_order, dense_ids_in_order
