# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import haliax as hax
from levanter.inference.utils import is_stop_signal, INVALID


def test_is_stop_signal_exact_match():
    # tail_tokens matches stop_sequence exactly
    tail_tokens = hax.named(jnp.array([5, 6, 7], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(jnp.array([[5, 6, 7]], dtype=jnp.int32), axis=("seq", "position"))
    assert is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_no_match():
    # tail_tokens does not match stop_sequence
    tail_tokens = hax.named(jnp.array([5, 6, 7], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(jnp.array([[1, 2, 3]], dtype=jnp.int32), axis=("seq", "position"))
    assert not is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_partial_match_with_padding():
    # stop_sequence is shorter and left padded with INVALID
    tail_tokens = hax.named(jnp.array([5, 6, 7], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(jnp.array([[INVALID, 6, 7]], dtype=jnp.int32), axis=("seq", "position"))
    assert is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_empty_stop_sequences():
    # stop_sequences is empty
    tail_tokens = hax.named(jnp.array([1, 2, 3], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(jnp.zeros((0, 3), dtype=jnp.int32), axis=("seq", "position"))
    assert not is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_invalid_tokens_in_stop_sequences():
    # stop_sequence contains only INVALID tokens (should not match)
    tail_tokens = hax.named(jnp.array([1, 2, 3], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(jnp.array([[INVALID, INVALID, INVALID]], dtype=jnp.int32), axis=("seq", "position"))
    assert not is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_multiple_stop_sequences_one_matches():
    # Multiple stop_sequences, only one matches
    tail_tokens = hax.named(jnp.array([8, 9, 10], dtype=jnp.int32), axis=("position",))
    stop_sequences = hax.named(
        jnp.array([[1, 2, 3], [8, 9, 10], [4, 5, 6]], dtype=jnp.int32), axis=("seq", "position")
    )
    assert is_stop_signal(tail_tokens, stop_sequences)


def test_is_stop_signal_problem():
    tail_tokens = hax.named(
        jnp.array(
            [2000000, 28070, 5573, 323, 8254, 1667, 4227, 11, 1057, 2291, 69, 19568, 323, 40317, 743, 704],
            dtype=jnp.int32,
        ),
        axis=("position",),
    )
    stop_sequences = hax.named(
        jnp.array(
            [
                [
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    2000000,
                    13,
                ]
            ],
            dtype=jnp.int32,
        ),
        axis=("stop_seq", "position"),
    )

    assert not is_stop_signal(tail_tokens, stop_sequences)
