import json
from pathlib import Path
import sys
import tempfile
import typing
from haliax import Axis, NamedArray
from haliax.nn import hax
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import numpy as np
import pytest

from levanter.data.text import (
    TextLmDatasetFormat,
    UrlSingleDatasetLMConfig,
    build_lm_dataset_cache,
    preprocessor_for_format,
)
from levanter.data.ul2r import (
    RX_TASK_KIND,
    TokenizedDict,
    compute_denoising_length,
    noise_span_to_unique_sentinel,
    num_noise_spans_tokens_and_spans,
    to_ul2r_tokens,
    ul2r_loss_mask,
    random_segmentation,
    random_spans_noise_mask,
    to_ul2r_rx_tokens,
    to_ul2r_s_tokens,
    SENTINEL_TOKEN_IDS,
    create_ul2r_example,
    RXDenoisingConfig,
    SDenoisingConfig,
    Ul2rDataset,
    R_TASK_TOKEN_ID,
    X_TASK_TOKEN_ID,
    S_TASK_TOKEN_ID,
)
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache


def test_random_segmentation():
    padded_length = 100

    test_cases = [
        (6, 2),
        (14, 2),
        (5, 2),
        (30, 1),
    ]

    for num_items, num_segments in test_cases:
        key = jax.random.PRNGKey(35)

        segment_lengths = random_segmentation(num_items, num_segments, key, padded_length)

        relevant_lengths = segment_lengths[:num_segments]

        assert (
            jnp.sum(relevant_lengths) == num_items
        ), f"Sum of segments {jnp.sum(relevant_lengths)} != num_items {num_items}"

        assert jnp.all(
            relevant_lengths > 0
        ), f"Found zero-length segments for num_items={num_items}, num_segments={num_segments}"

        # Check determinism - same key should give same result
        segment_lengths2 = random_segmentation(num_items, num_segments, key, padded_length)
        np.testing.assert_array_equal(segment_lengths, segment_lengths2)

        # Different key should give different result (usually)
        key2 = jax.random.PRNGKey(43)
        segment_lengths3 = random_segmentation(num_items, num_segments, key2, padded_length)
        # For edge cases like single segment, result might be same
        if num_segments > 1 and num_segments < num_items:
            assert not np.array_equal(segment_lengths[:num_segments], segment_lengths3[:num_segments])


def test_random_spans_noise_mask():
    padded_length = 256
    # NOTE At short lengths, it can be deterministic because the num_segments =
    # 1 for noise/non-noise.
    test_cases = [
        (20, 0.3, 3.0, False),
        (100, 0.15, 3.0, False),
        # (100, 0.15, 3.0, True),
        (200, 0.5, 10.0, False),
        (10, 0.3, 3.0, True),
    ]

    for length, noise_density, mean_span_length, random_roll in test_cases:
        key = jax.random.PRNGKey(37)

        mask = random_spans_noise_mask(length, noise_density, key, mean_span_length, random_roll, padded_length)
        print(mask)

        assert mask.shape == (padded_length,), f"Expected shape ({padded_length},), got {mask.shape}"
        assert mask.dtype == jnp.bool_, f"Expected bool dtype, got {mask.dtype}"

        assert jnp.all(mask[length:] == False), "Mask should be False after length"

        # Check noise density approximately matches (for non-zero cases)
        actual_density = jnp.sum(mask[:length]) / length
        # Allow for some variance due to rounding
        tolerance = 0.2
        assert (
            abs(actual_density - noise_density) < tolerance
        ), f"Noise density {actual_density} too far from target {noise_density}"

        # Test that the function is deterministic with same key
        mask2 = random_spans_noise_mask(length, noise_density, key, mean_span_length, random_roll, padded_length)
        np.testing.assert_array_equal(mask, mask2, "Function should be deterministic with same key")

        # Test that different keys produce different results
        keyb = jax.random.PRNGKey(38)
        mask3 = random_spans_noise_mask(
            length,
            noise_density,
            keyb,
            mean_span_length,
            random_roll,
            padded_length,
        )

        # DEBUG START ONE

        num_noise_tokens, num_noise_spans, num_nonnoise_tokens = num_noise_spans_tokens_and_spans(
            length, noise_density, mean_span_length
        )

        keyu, keyv, keyw = jax.random.split(key, 3)

        noise_span_lengths4 = random_segmentation(num_noise_tokens, num_noise_spans, keyu, padded_length)
        nonnoise_span_lengths4 = random_segmentation(num_nonnoise_tokens, num_noise_spans, keyv, padded_length)

        print(f"noise_span_lengths4 {noise_span_lengths4}")
        print(f"nonnoise_span_lengths4 {nonnoise_span_lengths4}")

        # Interleave using reshape
        interleaved_span_lengths = jnp.reshape(
            jnp.stack([nonnoise_span_lengths4, noise_span_lengths4], axis=1),
            [padded_length * 2],
        )[:padded_length]

        # Create span_start_indicator using bincount
        span_starts = jnp.cumsum(interleaved_span_lengths)
        span_start_indicator = jnp.bincount(span_starts, length=padded_length)

        print(f"span_starts4 {span_starts}")
        print(f"span_start_indicator4 {span_start_indicator}")

        span_num = jnp.cumsum(span_start_indicator)
        # Only odd spans less than 2*num_noise_spans are noise
        is_noise = ((span_num % 2) == 1) & (span_num < 2 * num_noise_spans)
        is_noise = is_noise.astype(jnp.bool_)

        print(f"span_num4 {span_num}")
        print(f"is_noise4 {is_noise}")

        # Zero everything at length and after
        indices = jnp.arange(padded_length)
        is_noise = jnp.where(indices < length, is_noise, False)
        is_noise = typing.cast(jnp.ndarray, is_noise)

        print(f"length4={length}")
        print("is_noise4")
        print(is_noise)

        offset = jax.random.randint(keyw, (), 0, length, dtype=jnp.int32)
        print(f"keyw={keyw} offset={offset}")
        # Roll the mask
        rolled = jnp.roll(is_noise, offset)
        # We want to roll within [0, length) so we need to overwrite values that
        # came from the end
        rolled = jnp.where(
            indices < offset,
            jnp.roll(is_noise, offset - length),
            rolled,
        )
        rolled = typing.cast(jnp.ndarray, rolled)
        mask_debug4 = jnp.where(indices < length, rolled, False)

        print("mask4 rolled")
        print(mask_debug4)

        # DEBUG START

        num_noise_tokens, num_noise_spans, num_nonnoise_tokens = num_noise_spans_tokens_and_spans(
            length, noise_density, mean_span_length
        )

        key1, key2, key3 = jax.random.split(keyb, 3)

        noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans, key1, padded_length)
        nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans, key2, padded_length)

        print(f"noise_span_lengths {noise_span_lengths}")
        print(f"nonnoise_span_lengths {nonnoise_span_lengths}")

        keyx, keyy, keyz = jax.random.split(key, 3)

        noise_span_lengths2 = random_segmentation(num_noise_tokens, num_noise_spans, keyx, padded_length)
        nonnoise_span_lengths2 = random_segmentation(num_nonnoise_tokens, num_noise_spans, keyy, padded_length)

        print(f"noise_span_lengths2 {noise_span_lengths2}")
        print(f"nonnoise_span_lengths2 {nonnoise_span_lengths2}")

        # Interleave using reshape
        interleaved_span_lengths = jnp.reshape(
            jnp.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [padded_length * 2],
        )[:padded_length]

        # Create span_start_indicator using bincount
        span_starts = jnp.cumsum(interleaved_span_lengths)
        span_start_indicator = jnp.bincount(span_starts, length=padded_length)

        print(f"span_starts {span_starts}")
        print(f"span_start_indicator {span_start_indicator}")

        span_num = jnp.cumsum(span_start_indicator)
        # Only odd spans less than 2*num_noise_spans are noise
        is_noise = ((span_num % 2) == 1) & (span_num < 2 * num_noise_spans)
        is_noise = is_noise.astype(jnp.bool_)

        print(f"span_num {span_num}")
        print(f"is_noise {is_noise}")

        # Zero everything at length and after
        indices = jnp.arange(padded_length)
        is_noise = jnp.where(indices < length, is_noise, False)
        is_noise = typing.cast(jnp.ndarray, is_noise)

        print(f"length={length}")
        print("is_noise")
        print(is_noise)

        offset = jax.random.randint(key3, (), 0, length, dtype=jnp.int32)
        print(f"key3={key3} offset={offset}")
        # Roll the mask
        rolled = jnp.roll(is_noise, offset)
        # We want to roll within [0, length) so we need to overwrite values that
        # came from the end
        rolled = jnp.where(
            indices < offset,
            jnp.roll(is_noise, offset - length),
            rolled,
        )
        rolled = typing.cast(jnp.ndarray, rolled)
        mask_debug = jnp.where(indices < length, rolled, False)

        print("mask rolled")
        print(mask_debug)
        print("mask")
        print(mask)
        print("mask3")
        print(mask3)

        # DEBUG END

        # NOTE This test could fail spuriously because of randomness. Inspect
        # the results.
        assert not jnp.array_equal(mask, mask3), "Different keys should produce different masks"


def test_noise_span_to_unique_sentinel():
    """Test that noise_span_to_unique_sentinel works correctly with static shapes."""
    padded_length = 256
    pad_token_id = 0
    sentinel_tokens = jnp.array([100, 101, 102, 103, 104])

    # Test case 1: First token is a noise span (single span)
    tokens = jnp.arange(10, 20)  # [10, 11, 12, ..., 19]
    tokens = jnp.pad(tokens, (0, padded_length - 10), constant_values=pad_token_id)
    noise_mask = jnp.array(
        [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,  # Span 1: [10, 11, 12]
        ]
    )
    noise_mask = jnp.pad(noise_mask, (0, padded_length - 10), constant_values=False)

    result = noise_span_to_unique_sentinel(tokens, 10, noise_mask, sentinel_tokens, force_initial_sentinel=False)

    expected = jnp.array([100, 13, 14, 15, 16, 17, 18, 19])
    np.testing.assert_array_equal(result[:8], expected)
    assert jnp.all(result[8:] == pad_token_id)

    # Test case 2: First token is NOT a noise span
    # force_initial_sentinel=True (for use w/ targets, which must always start
    # with a sentinel even if the noise mask is 010)
    tokens = jnp.arange(10, 25)  # [10, 11, 12, ..., 24]
    tokens = jnp.pad(tokens, (0, padded_length - 15), constant_values=pad_token_id)
    noise_mask = jnp.array(
        [
            False,  # Does not start with noise span
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
        ]
    )
    noise_mask = jnp.pad(noise_mask, (0, padded_length - 15), constant_values=False)

    result = noise_span_to_unique_sentinel(tokens, 15, noise_mask, sentinel_tokens, force_initial_sentinel=True)

    # Should still start with sentinel
    expected = jnp.array([100, 10, 101, 13, 14, 15, 102, 17, 18, 19, 103, 23, 24])
    np.testing.assert_array_equal(result[:13], expected)
    assert jnp.all(result[13:] == pad_token_id)

    # Test case 3: Empty noise mask (no noise)
    tokens = jnp.arange(10, 20)
    tokens = jnp.pad(tokens, (0, padded_length - 10), constant_values=pad_token_id)
    noise_mask = jnp.zeros(padded_length, dtype=jnp.bool_)

    result = noise_span_to_unique_sentinel(tokens, 10, noise_mask, sentinel_tokens, force_initial_sentinel=False)

    # Should be unchanged except for padding
    np.testing.assert_array_equal(result[:10], jnp.arange(10, 20))
    assert jnp.all(result[10:] == pad_token_id)


def test_to_ul2r_rx_tokens():
    max_length = 256
    pad_token_id = 0
    sentinel_tokens = jnp.array([100, 101, 102, 103, 104])

    # Test case: Simple sequence with known noise pattern
    tokens = jnp.arange(10, 20)
    tokens = jnp.pad(tokens, (0, max_length - tokens.shape[0]), constant_values=pad_token_id)
    length = 10

    key = jax.random.PRNGKey(38)

    # DEBUG

    padded_length = tokens.shape[0]
    noise_mask = random_spans_noise_mask(
        length,
        0.3,
        key,
        3.0,
        True,
        padded_length,
    )

    print(noise_mask)
    print(noise_mask[:length])

    num_noise_tokens, num_noise_spans, num_nonnoise_tokens = num_noise_spans_tokens_and_spans(length, 0.3, 3.0)

    key1, key2, key3 = jax.random.split(key, 3)

    print(num_noise_tokens)
    print(num_nonnoise_tokens)
    print(num_noise_spans)

    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans, key1, padded_length)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans, key2, padded_length)

    print(noise_span_lengths)
    print(nonnoise_span_lengths)

    # Interleave using reshape
    interleaved_span_lengths = jnp.reshape(
        jnp.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [2 * padded_length],
    )[:padded_length]

    # Create span_start_indicator using bincount
    span_starts = jnp.cumsum(interleaved_span_lengths)
    span_start_indicator = jnp.bincount(span_starts, length=padded_length)

    span_num = jnp.cumsum(span_start_indicator)
    # Only odd spans less than 2*num_noise_spans are noise
    is_noise = ((span_num % 2) == 1) & (span_num < 2 * num_noise_spans)
    is_noise = is_noise.astype(jnp.bool_)

    # Zero everything at length and after
    indices = jnp.arange(padded_length)
    is_noise = jnp.where(indices < length, is_noise, False)
    is_noise = typing.cast(jnp.ndarray, is_noise)

    inputs = noise_span_to_unique_sentinel(
        tokens,
        length,
        noise_mask,
        sentinel_tokens,
        force_initial_sentinel=False,
    )
    targets = noise_span_to_unique_sentinel(
        tokens,
        length,
        ~noise_mask,
        sentinel_tokens,
        force_initial_sentinel=True,
    )

    print(inputs)
    print(targets)

    # END DEBUG

    input_length, result = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=0.3,
        mean_noise_span_length=3.0,
        random_roll=False,
        pad_token_id=pad_token_id,
        sentinel_token_ids=sentinel_tokens,
        max_length=max_length,
    )

    print(tokens)
    print(result)

    assert result.shape == (max_length,)

    assert input_length.shape == ()
    assert input_length > 0
    assert input_length <= length

    inputs_part = result[:input_length]
    contains_sentinels = jnp.any(jnp.isin(inputs_part, sentinel_tokens))
    assert contains_sentinels, "Inputs should contain sentinel tokens"

    # Find where all padding starts (after both inputs and targets)
    # Look for continuous padding at the end
    is_pad = result == pad_token_id
    # Find the last non-padding position
    non_pad_positions = jnp.where(~is_pad, jnp.arange(max_length), -1)
    last_non_pad = jnp.max(non_pad_positions)

    if last_non_pad < max_length - 1:
        assert jnp.all(result[last_non_pad + 1 :] == pad_token_id), "Should have continuous padding at the end"

    # Test with random_roll=True
    # input_length_roll, result_roll = to_ul2r_rx_tokens(
    #     key,
    #     tokens,
    #     length,
    #     mask_prob=0.3,
    #     mean_noise_span_length=3.0,
    #     random_roll=True,
    #     sentinel_token_ids=sentinel_tokens,
    #     max_length=max_length,
    # )

    # assert result_roll.shape == (max_length,)
    # assert input_length_roll.shape == ()

    # is_pad_roll = result_roll == pad_token_id
    # non_pad_positions_roll = jnp.where(~is_pad_roll, jnp.arange(max_length), -1)
    # last_non_pad_roll = jnp.max(non_pad_positions_roll)
    # if last_non_pad_roll < max_length - 1:
    #     assert jnp.all(
    #         result_roll[last_non_pad_roll + 1 :] == pad_token_id
    #     ), "Should have continuous padding at the end with roll"


def test_to_ul2r_rx_tokens_roll():
    max_length = 256
    pad_token_id = 0
    min_sentinel_id = 100
    num_sentinels = 5
    sentinel_ids = jnp.arange(min_sentinel_id, min_sentinel_id + num_sentinels)

    tokens = jnp.arange(10, 20)
    tokens = jnp.pad(tokens, (0, max_length - tokens.shape[0]), constant_values=pad_token_id)
    length = 10

    for i in range(5):
        key = jax.random.PRNGKey(i)

        input_length, result = to_ul2r_rx_tokens(
            key,
            tokens,
            length,
            mask_prob=0.3,
            mean_noise_span_length=3.0,
            random_roll=True,
            pad_token_id=pad_token_id,
            sentinel_token_ids=sentinel_ids,
            max_length=max_length,
        )

        print(tokens)
        print(result)

        assert result.shape == (max_length,)

        assert input_length.shape == ()
        assert input_length > 0
        assert input_length <= length

        inputs = result[:input_length]
        targets = result[input_length:]

        # The same sentinels should exist in inputs/targets.
        # We previously had a bug where a noise mask of 010 (created from
        # random_roll) would produce a target noise mask of 101, which would create
        # 2 sentinels in the target vs. only 1 in the input.

        print("inputs", inputs)
        print("targets", targets)
        np.testing.assert_array_equal(jnp.isin(sentinel_ids, inputs), jnp.isin(sentinel_ids, targets))
        assert jnp.any(jnp.isin(sentinel_ids, inputs))
        assert jnp.any(jnp.isin(sentinel_ids, targets))


def test_compute_denoising_length_rx_random_roll():
    """
    Test that compute_denoising_length with random_roll=True reserves enough
    space and sets pad_token_id when it doesn't create an extra span.

    When random_roll=True, we reserve space for an extra span. However, rolling doesn't
    always create an additional span, so we should see both cases:
    - no pad_token_id (rolling created an extra span)
    - 1 pad_token_id (rolling created an extra span)
    """
    max_length = 16
    pad_token_id = 999  # Use non-zero pad token to verify it's actually being used
    sentinel_ids = jnp.arange(100, 120)

    length = 12
    tokens = jnp.arange(1, length + 1)
    tokens = jnp.pad(tokens, (0, max_length - length), constant_values=pad_token_id)

    mask_prob = 0.3
    mean_noise_span_length = 3.0
    random_roll = True
    task_params = jnp.array([RX_TASK_KIND, R_TASK_TOKEN_ID, mask_prob, mean_noise_span_length, int(random_roll)])

    predicted_length = compute_denoising_length(task_params, length)

    padding_counts = []
    for i in range(16):
        key = jax.random.PRNGKey(i)
        _input_length, result = to_ul2r_rx_tokens(
            key, tokens, length, mask_prob, mean_noise_span_length, True,
            pad_token_id, sentinel_ids, max_length
        )

        # Subtract 1 because `result` doesn't include the task token.
        # print(result[:predicted_length - 1])
        num_padding = jnp.sum(result[:predicted_length - 1] == pad_token_id)
        padding_counts.append(int(num_padding))

        actual_length = jnp.sum(result != pad_token_id)
        assert actual_length <= predicted_length

    assert any(p == 0 for p in padding_counts)
    assert any(p == 2 for p in padding_counts)
    assert all(p == 0 or p == 2 for p in padding_counts)


def test_ul2r_loss_mask():
    # Test case 1: Simple single segment
    input_masks = jnp.array([1, 1, 0, 0])  # First 2 are inputs
    segment_ids = jnp.array([0, 0, 0, 0])  # All in same segment
    tokens = jnp.array([10, 11, 12, 13])  # No padding
    pad_token_id = 0

    mask = ul2r_loss_mask(input_masks, segment_ids, tokens, pad_token_id)

    # Expected: loss on outputs, shifted by 1
    # Original:     1 1 0 0 (input_masks)
    # Inverted:     0 0 1 1
    # Shifted:      0 1 1 0 (roll -1)
    # No boundaries or padding to worry about
    expected = jnp.array([False, True, True, False])

    np.testing.assert_array_equal(mask, expected)

    # Test case 2: Multiple segments with padding
    input_masks = jnp.array([1, 1, 0, 0, 1, 0, 0, 0])
    tokens = jnp.array([10, 11, 12, 13, 14, 15, 0, 0])  # Last 2 are padding tokens
    segment_ids = jnp.array([0, 0, 0, 0, 1, 1, -1, -1])  # Two segments plus padding

    mask = ul2r_loss_mask(input_masks, segment_ids, tokens, pad_token_id)

    # Expected: no loss across segment boundaries or on padding
    # Original:     1 1 0 0 | 1 0 | 0 0 (padding)
    # Inverted:     0 0 1 1 | 0 1 | 1 1
    # Shifted:      0 1 1 0 | 1 1 | 1 0
    # Segment cont: T T T F | T F | F F (boundaries and padding)
    # Valid mask:   T T T T | T T | F F (padding tokens are 0)
    # Final:        0 1 1 0 | 1 0 | 0 0
    expected = jnp.array([False, True, True, False, True, False, False, False])

    np.testing.assert_array_equal(mask, expected)


def test_to_ul2r_rx_tokens_truncates_both_sections_and_contains_sentinels():
    """
    Ensure we truncate from both inputs and outputs and retain sentinels in each section.

    We create a long padded input (padded_length > max_length) and compare the
    pre-truncation lengths to the output of to_ul2r_rx_tokens with a small
    max_length to force truncation.
    """
    padded_length = 512
    max_length = 400
    pad_token_id = 0
    sentinel_tokens = jnp.arange(100, 140)  # plenty of sentinels

    length = 450  # true sequence length (greater than max_length)
    tokens = jnp.arange(1, length + 1)
    tokens = jnp.pad(tokens, (0, padded_length - length), constant_values=pad_token_id)

    key = jax.random.PRNGKey(123)
    mask_prob = 0.3
    mean_noise_span_length = 3.0
    random_roll = False

    # First, run with a large max_length to observe untruncated section lengths
    input_len_full, result_full = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=mask_prob,
        mean_noise_span_length=mean_noise_span_length,
        random_roll=random_roll,
        pad_token_id=pad_token_id,
        sentinel_token_ids=sentinel_tokens,
        max_length=padded_length,
    )
    # Compute outputs length (non-padding after inputs)
    outputs_full = result_full[input_len_full:]
    outputs_nonpad_full = jnp.sum(outputs_full != pad_token_id)

    # Now run with a smaller max_length to force truncation
    input_len_small, result_small = to_ul2r_rx_tokens(
        key,
        tokens,
        length,
        mask_prob=mask_prob,
        mean_noise_span_length=mean_noise_span_length,
        random_roll=random_roll,
        pad_token_id=pad_token_id,
        sentinel_token_ids=sentinel_tokens,
        max_length=max_length,
    )

    assert result_small.shape == (padded_length,)
    assert input_len_small > 0
    assert input_len_small < input_len_full  # inputs truncated

    # Outputs section length and checks
    outputs_slice = result_small[input_len_small:]
    outputs_nonpad = jnp.sum(outputs_slice != pad_token_id)
    assert outputs_nonpad > 0
    assert outputs_nonpad < outputs_nonpad_full  # outputs truncated

    # Sentinel presence in both sections
    inputs_slice = result_small[:input_len_small]
    has_sentinel_inputs = jnp.any(jnp.isin(inputs_slice, sentinel_tokens))
    has_sentinel_outputs = jnp.any(jnp.isin(outputs_slice[:outputs_nonpad], sentinel_tokens))
    assert has_sentinel_inputs
    assert has_sentinel_outputs


def test_to_ul2r_s_tokens():
    # Test case 1: Basic functionality with simple sequence
    tokens = jnp.arange(10, 20)  # [10, 11, 12, ..., 19]
    padded_tokens = jnp.pad(tokens, (0, 256 - 10), constant_values=0)
    length = 10
    key = jax.random.PRNGKey(42)

    pivot, result = to_ul2r_s_tokens(key, padded_tokens, length, SENTINEL_TOKEN_IDS[0])

    assert result.shape == padded_tokens.shape
    assert pivot.shape == ()
    assert 1 <= pivot < length - 1, f"Pivot {pivot} should be between 1 and {length-2}"
    assert result[pivot] == SENTINEL_TOKEN_IDS[0]

    # Check prefix is unchanged (before pivot)
    np.testing.assert_array_equal(result[:pivot], padded_tokens[:pivot])

    # Check continuation (after pivot) is shifted from original
    # The continuation should be tokens[pivot:] starting at position pivot+1
    expected_continuation = padded_tokens[pivot:]
    # Shift by 1 to account for sentinel at pivot
    np.testing.assert_array_equal(result[pivot + 1 :], expected_continuation[:-1])

    # Test case 2: Determinism - same key should give same result
    pivot2, result2 = to_ul2r_s_tokens(key, padded_tokens, length, SENTINEL_TOKEN_IDS[0])
    np.testing.assert_array_equal(result, result2)
    assert pivot == pivot2

    # Test case 3: Different keys should give different pivots (usually)
    key2 = jax.random.PRNGKey(43)
    pivot3, result3 = to_ul2r_s_tokens(key2, padded_tokens, length, SENTINEL_TOKEN_IDS[0])
    assert pivot != pivot3, "Different keys should produce different pivots"


def test_create_ul2r_example():
    QPos = Axis("QPos", 128)
    KPos = Axis("KPos", 128)
    pad_token_id = 0
    max_segments_per_example = 8

    task_configs = [
        RXDenoisingConfig(R_TASK_TOKEN_ID, 0.15, 3.0, False),
        RXDenoisingConfig(X_TASK_TOKEN_ID, 0.5, 3.0, False),
        SDenoisingConfig(S_TASK_TOKEN_ID),
    ]
    task_params = jnp.array([cfg.to_task_params() for cfg in task_configs])
    task_indices = jnp.array([0, 1, 2])

    in_len_r = 10
    in_len_x = 8
    in_len_s = 5
    in_len = in_len_r + in_len_x + in_len_s

    out_len_r = compute_denoising_length(task_params[0], in_len_r)
    out_len_x = compute_denoising_length(task_params[1], in_len_x)
    out_len_s = compute_denoising_length(task_params[2], in_len_s)

    tokens = jnp.concatenate(
        [
            jnp.arange(10, 10 + in_len_r),  # 0-10 segment 0 [R]
            # When we pack examples together, we compute the amount of padding
            # we need to reserve using `compute_denoising_length`. Here we just
            # manually add padding.
            jnp.zeros(10, dtype=jnp.int32),  # 10-20 padding
            jnp.arange(20, 20 + in_len_x),  # 20-28 segment 1 [X]
            jnp.zeros(10, dtype=jnp.int32),  # 28-38 padding
            jnp.arange(30, 30 + in_len_s),  # 38-43 segment 2 [S]
            jnp.zeros(128 - in_len - 20, dtype=jnp.int32),  # 43-128 padding
        ]
    )
    tokens = hax.named(tokens, QPos)

    print(tokens.array)

    # The segment_ids need to
    segment_ids = jnp.concatenate(
        [
            jnp.full(in_len_r, 0),
            jnp.full(10, -1),
            jnp.full(in_len_x, 1),
            jnp.full(10, -1),
            jnp.full(in_len_s, 2),
            jnp.full(128 - in_len - 20, -1),
        ]
    )
    segment_ids = hax.named(segment_ids, QPos)

    key = jax.random.PRNGKey(37)

    # DEBUG START

    # TODO Use NamedArrays more idiomatically
    # `unique_seg_ids = [3, 4, ..., -1, ...]`
    # Valid segment IDs come first, padded with -1.
    # Sorted; assumes `segment_ids` is also sorted in ascending order.
    # We use the same ordering for `out_starts` etc.
    max_seg_id = jnp.max(segment_ids.array)
    seg_ids = jnp.where(segment_ids.array == -1, max_seg_id, segment_ids.array)
    unique_seg_ids = jnp.unique(seg_ids, size=max_segments_per_example, fill_value=-1)

    def prepare_segment(id: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Returns `(in_start, in_length, out_length)` for segment `id`.
        The result is undefined for ids not in `segment_ids` or the id -1.
        """

        mask = segment_ids.array == id
        n = mask.shape[0]
        idx = jnp.arange(n)

        in_start = jnp.min(jnp.where(mask, idx, n))
        in_end = jnp.max(jnp.where(mask, idx + 1, 0))
        in_length = in_end - in_start

        # Eventually we don't want to have to instantiate all of `offsets`
        # and `task_indices` (because they could take up a lot of
        # space). We can generate `task_params` / `task_idx` for each
        # batch independently (but in a way that matches how we computed
        # lengths for packing).
        task_idx = task_indices[id]
        out_length = compute_denoising_length(task_params[task_idx], in_length)

        return in_start, in_length, out_length

    in_starts, in_lengths, out_lengths = jax.vmap(prepare_segment)(unique_seg_ids)

    # `out_starts[i]` is the offset of the beginning of the i-th output segment.
    # Segment lengths increase when we turn them into denoising examples.
    out_starts = jax.lax.cumsum(out_lengths) - out_lengths
    out_starts = jnp.where(unique_seg_ids == -1, -1, out_starts)
    out_starts = typing.cast(jnp.ndarray, out_starts)

    print(out_starts)

    def process_segment(key, id: int) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """
        Applies UL2R denoising to a single segment.
        Returns `(input_mask, denoising_tokens, start, length)`.
        `input_mask` is true when the token is an input given to the LLM
        (i.e. one where we don't compute loss).
        `denoising_tokens` is `0... to_ul2r_tokens(...) 0...` where the
        prefix is `out_starts[idx]` zeros.
        We'll or all the segments together at the end; the nonzero parts
        shouldn't overlap.
        """

        task_idx = task_indices[id]

        idx = jnp.nonzero(unique_seg_ids == id, size=1)[0]
        in_start = typing.cast(int, jnp.squeeze(in_starts[idx]))
        in_length = typing.cast(int, jnp.squeeze(in_lengths[idx]))
        out_length = typing.cast(int, jnp.squeeze(out_lengths[idx]))
        out_start = typing.cast(int, jnp.squeeze(out_starts[idx]))

        segment = jnp.roll(tokens.array, -in_start)
        print(key, task_params[task_idx], segment, in_length, QPos.size)
        inputs_len, denoising_tokens = to_ul2r_tokens(key, task_params[task_idx], segment, in_length, pad_token_id, SENTINEL_TOKEN_IDS, QPos.size)

        n_tokens = tokens.array.shape[0]
        input_mask = jnp.arange(n_tokens) < inputs_len
        input_mask = jnp.roll(input_mask, out_start)
        denoising_tokens = jnp.roll(denoising_tokens, out_start)
        return (input_mask, denoising_tokens, out_start, out_length)

    process_key, _key = jax.random.split(key)
    print(process_key)
    print(process_segment(process_key, 0))

    def loop(
        acc: tuple[PRNGKeyArray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[PRNGKeyArray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        (key, i, input_mask, denoising_tokens, out_seg_ids) = acc
        process_key, key = jax.random.split(key)
        id = unique_seg_ids[i]
        (seg_input_mask, seg_denoising_tokens, start, length) = process_segment(process_key, id)
        input_mask = input_mask | seg_input_mask
        denoising_tokens = denoising_tokens | seg_denoising_tokens
        indices = jnp.arange(out_seg_ids.shape[0])
        out_seg_ids = jnp.where((indices >= start) & (indices < start + length), id, out_seg_ids)
        out_seg_ids = typing.cast(jnp.ndarray, out_seg_ids)
        return (key, i + 1, input_mask, denoising_tokens, out_seg_ids)

    key2, i2, input_mask2, denoising_tokens2, out_seg_ids2 = loop(
        (
            key,
            0,
            jnp.zeros_like(tokens.array, dtype=jnp.bool_),
            jnp.zeros_like(tokens.array),
            jnp.full(tokens.array.shape, -1),
        )
    )
    key3, i3, input_mask3, denoising_tokens3, out_seg_ids3 = loop(
        (key2, i2, input_mask2, denoising_tokens2, out_seg_ids2)
    )

    jnp.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    input_attn_mask = jnp.outer(input_mask3, input_mask3)
    print("input_attn_mask")
    print(input_attn_mask.astype(jnp.int32))
    print("out_seg_ids3")
    print(out_seg_ids3)
    segment_mask = out_seg_ids3 == out_seg_ids3[:, jnp.newaxis]
    print("segment_mask")
    print(segment_mask.astype(jnp.int32))
    prefix_mask = segment_mask & input_attn_mask
    print("prefix_mask")
    print(prefix_mask.astype(jnp.int32))

    # DEBUG END

    example = create_ul2r_example(
        key,
        QPos,
        KPos,
        pad_token_id,
        SENTINEL_TOKEN_IDS,
        max_segments_per_example,
        task_params,
        task_indices,
        tokens,
        segment_ids,
    )

    # Basic smoke checks
    assert example.tokens.array.shape == (QPos.size,)
    assert example.loss_mask.array.shape == (QPos.size,)
    assert example.attn_mask.is_causal

    # Should contain sentinel tokens after denoising
    contains_sentinels = jnp.any(jnp.isin(example.tokens.array, SENTINEL_TOKEN_IDS))
    assert contains_sentinels

    # Should have some loss positions
    assert jnp.any(example.loss_mask.array)

    # No loss on padding tokens
    is_padding = example.tokens.array == pad_token_id
    assert not jnp.any(example.loss_mask.array & is_padding)

    print(example.tokens.array)
    print(example.loss_mask.array.astype(jnp.int32))
    print(example.attn_mask.segment_ids[0].array)
    print(example.attn_mask.materialize(QPos, KPos).array.astype(jnp.int32))

    # The R-denoising example should contain exactly 2 of the first sentinel
    # (one to identify the masked span, one to identify the output span)
    assert jnp.sum(example.tokens.array[0:out_len_r] == SENTINEL_TOKEN_IDS[0]) == 2


@pytest.fixture
def dummy_text_data():
    texts = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is a subset of artificial intelligence."},
        {"text": "Python is a popular programming language."},
        {"text": "Data science combines statistics and computer science."},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "text.jsonl"
        with path.open("w") as f:
            for text in texts:
                f.write(json.dumps(text) + "\n")
        yield str(path)


@pytest.fixture(scope="module")
def hf_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        "stanford-crfm/marin-tokenizer",
        revision="49a09e626c220e9daae74124ea41be1bf5cd331d",
    )


@pytest.mark.ray
def test_ul2r_dataset_build(dummy_text_data, hf_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = hf_tokenizer

        config = UrlSingleDatasetLMConfig(train_urls=[dummy_text_data], format=TextLmDatasetFormat())

        processor = preprocessor_for_format(config.format, tokenizer)

        # Test the processor
        source = config.get_shard_source("train")
        assert source is not None
        assert source.shard_names is not None
        processed = []
        for doc in source.open_shard(source.shard_names[0]):
            processed += processor([doc])

        assert len(processed) == 4

        # Build cache
        cache = build_lm_dataset_cache(tmpdir, source, config.format, tokenizer)
        cache.await_finished()
        cache_sync = cache.as_sync_dataset()
        assert len(cache_sync) == 4

        sample = next(iter(cache))
        assert sample["input_ids"].shape[0] > 0  # pyright: ignore
        cache = typing.cast(TreeCache[TokenizedDict], cache)

        # Test Ul2rDataset
        QPos = hax.Axis("QPos", 128)
        KPos = hax.Axis("KPos", 128)
        task_configs = {
            "r": RXDenoisingConfig(R_TASK_TOKEN_ID, 0.15, 3.0, False),
            "x": RXDenoisingConfig(X_TASK_TOKEN_ID, 0.5, 3.0, False),
            "s": SDenoisingConfig(S_TASK_TOKEN_ID),
        }

        dataset = Ul2rDataset(
            cache=cache,
            QPos=QPos,
            KPos=KPos,
            task_configs=task_configs,
            task_probs={"r": 0.33, "x": 0.33, "s": 0.34},
            key=jax.random.PRNGKey(123),
            pad_token_id=tokenizer.pad_token_id or 0,
            sentinel_token_ids=SENTINEL_TOKEN_IDS,
            max_segments_per_example=4,
        )
        dataset_sync = dataset.as_sync_dataset()

        assert len(dataset_sync) > 0

        ex = dataset_sync[0]

        # Structure checks
        assert isinstance(ex, LmExample)
        assert ex.tokens.axes == (QPos,)
        assert ex.loss_mask.axes == (QPos,)
        assert isinstance(ex.attn_mask, AttentionMask)
        assert ex.attn_mask.is_causal

        # Content checks
        pad_id = tokenizer.pad_token_id or 0
        non_padding = ex.tokens.array != pad_id
        num_loss = jnp.sum(ex.loss_mask.array)

        assert jnp.sum(non_padding) > 0
        assert num_loss > 0
        assert num_loss < jnp.sum(non_padding)  # Loss < content (some tokens are inputs)
        assert not jnp.any(ex.loss_mask.array & ~non_padding)  # No loss on padding
        assert jnp.any(jnp.isin(ex.tokens.array, SENTINEL_TOKEN_IDS))  # Has sentinels from denoising

        # Collect all original input tokens from the cache
        original_tokens = set()
        for item in cache_sync:
            original_tokens.update(int(t) for t in item["input_ids"] if t != pad_id)

        # Check that all output tokens (except pad, sentinels, and task tokens) were present in the input
        # This helps verify we're not creating gibberish by overlapping spans
        ul2r_special_tokens = set(SENTINEL_TOKEN_IDS.tolist()) | {
            R_TASK_TOKEN_ID,
            X_TASK_TOKEN_ID,
            S_TASK_TOKEN_ID,
            pad_id,
        }
        allowed_tokens = original_tokens | ul2r_special_tokens
        output_tokens = set(int(t) for t in ex.tokens.array)
        unexpected_tokens = output_tokens - allowed_tokens
        assert len(unexpected_tokens) == 0, f"Found unexpected tokens not in input: {unexpected_tokens}"

        # Attention mask checks
        input_mask = typing.cast(NamedArray, ex.attn_mask.input_mask)
        assert input_mask.array.shape == (QPos.size,)
        # Materialize full attention mask (causal + prefix)
        materialized = ex.attn_mask.materialize(QPos, KPos)
        assert materialized is not None
        # Diagonal should be True for all non-padding (tokens attend to themselves)
        diag = jnp.diag(materialized.array)
        assert jnp.all(diag[non_padding])
        # Some off-diagonal should be True (bidirectional attention on input positions)
        off_diag_sum = jnp.sum(materialized.array) - jnp.sum(diag)
        assert off_diag_sum > 0, "Expected some bidirectional attention for input positions"

        # Check consistency across multiple examples
        for ex_i in [dataset_sync[i] for i in range(min(3, len(dataset_sync)))]:
            assert ex_i.tokens.axes == (QPos,) and ex_i.loss_mask.axes == (QPos,)
            non_pad_i = jnp.sum(ex_i.tokens.array != pad_id)
            loss_i = jnp.sum(ex_i.loss_mask.array)
            assert 0 < loss_i < non_pad_i
