from dataclasses import dataclass
from haliax import Axis
from jaxtyping import PRNGKeyArray
from typing import Literal, Optional, Dict
import dataclasses
import draccus
import equinox as eqx
import functools
import jax
import jax.numpy as jnp
import numpy as np
import typing
import haliax as hax

from levanter.data.dataset import MappedAsyncDataset
from levanter.data.packing import GreedyPrepackedDataset
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.store.cache import TreeCache


# From https://huggingface.co/stanford-crfm/marin-tokenizer/blob/main/tokenizer.json
R_TASK_TOKEN_ID = 128011
X_TASK_TOKEN_ID = 128012
S_TASK_TOKEN_ID = 128013


# Use the last 100 reserved token IDs as sentinel tokens (equivalent to T5's
# extra_ids); <|reserved_special_token_148|> to 247
# https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/models/gin/objectives/span.gin#L46-L47
NUM_TOKEN_IDS = 128256
NUM_SENTINELS = 100
SENTINEL_TOKEN_IDS = jnp.arange(NUM_TOKEN_IDS - NUM_SENTINELS, NUM_TOKEN_IDS, dtype=jnp.int32)

RX_TASK_KIND = 0
S_TASK_KIND = 1

MIN_S_LENGTH = 3


@dataclass(frozen=True)
class DenoisingConfig(draccus.ChoiceRegistry):
    task_token_id: int

    def with_task_token_id(self, task_token_id: int) -> "DenoisingConfig":
        return dataclasses.replace(self, task_token_id=task_token_id)

    @staticmethod
    def ul2_configs(
        r_task_token_id: int = R_TASK_TOKEN_ID,
        x_task_token_id: int = X_TASK_TOKEN_ID,
        s_task_token_id: int = S_TASK_TOKEN_ID,
    ) -> typing.Dict[str, "DenoisingConfig"]:
        # Table 1 https://arxiv.org/pdf/2205.05131
        return {
            "r1": RXDenoisingConfig(r_task_token_id, 0.15, 3.0, False),
            "r2": RXDenoisingConfig(r_task_token_id, 0.15, 8.0, False),
            "x1": RXDenoisingConfig(x_task_token_id, 0.5, 3.0, False),
            "x2": RXDenoisingConfig(x_task_token_id, 0.5, 8.0, False),
            "x3": RXDenoisingConfig(x_task_token_id, 0.15, 64.0, False),
            "x4": RXDenoisingConfig(x_task_token_id, 0.5, 64.0, False),
            "s": SDenoisingConfig(s_task_token_id),
        }

    @staticmethod
    def ul2r_configs(
        r_task_token_id: int = R_TASK_TOKEN_ID,
        x_task_token_id: int = X_TASK_TOKEN_ID,
        s_task_token_id: int = S_TASK_TOKEN_ID,
    ) -> Dict[str, "DenoisingConfig"]:
        # Section 3.3 Loss Objectives https://arxiv.org/pdf/2210.11399v2
        return {
            "r": RXDenoisingConfig(r_task_token_id, 0.15, 3.0, False),
            "x1": RXDenoisingConfig(x_task_token_id, 0.15, 32.0, False),
            "x2": RXDenoisingConfig(x_task_token_id, 0.5, 3.0, False),
            "s": SDenoisingConfig(s_task_token_id),
        }

    def to_task_params(self) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def task_kind_from_task_params(task_params: jnp.ndarray) -> jnp.ndarray:
        return task_params[0]

    @staticmethod
    def task_token_id_from_task_params(task_params: jnp.ndarray) -> jnp.ndarray:
        return task_params[1]


@DenoisingConfig.register_subclass("rx")
@dataclass(frozen=True)
class RXDenoisingConfig(DenoisingConfig):
    mask_prob: float  # r in the paper
    mean_span_length: float  # mu in the paper
    random_roll: bool

    def to_task_params(self) -> jnp.ndarray:
        args = [
            RX_TASK_KIND,
            self.task_token_id,
            self.mask_prob,
            self.mean_span_length,
            int(self.random_roll),
        ]
        return jnp.array(args)

    @staticmethod
    def mask_prob_from_task_params(task_params: jnp.ndarray) -> jnp.ndarray:
        return task_params[2]

    @staticmethod
    def mean_span_length_from_task_params(task_params: jnp.ndarray) -> jnp.ndarray:
        return task_params[3]

    @staticmethod
    def random_roll_from_task_params(task_params: jnp.ndarray) -> jnp.ndarray:
        return task_params[4].astype(jnp.bool_)


@DenoisingConfig.register_subclass("s")
@dataclass(frozen=True)
class SDenoisingConfig(DenoisingConfig):
    def to_task_params(self) -> jnp.ndarray:
        # should match size of RXDenoisingConfig
        return jnp.array([S_TASK_KIND, self.task_token_id, 0, 0, 0])


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_segmentation_1(num_items: int, padded_length: int) -> jnp.ndarray:
    out = jnp.zeros((padded_length,), dtype=jnp.int32)
    return out.at[0].set(num_items)


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_segmentation_2(num_items: int, num_segments: int, key: PRNGKeyArray, padded_length: int) -> jnp.ndarray:
    """
    Generates a random partition of `num_items` items into `num_segments`
    segments described by their lengths. The length of the returned tensor is
    `padded_length`.

    Based on `_random_segmentation` in `random_spans_noise_mask` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2992-L3010


    Precondition:
        1 <= num_segments < num_items

    Returns:
        A tensor of segment lengths with shape `(padded_length,)` where the
        first `num_segments` values sum to `num_items` and all other values are
        zero.
    """

    indices = jnp.arange(padded_length)

    perm = jax.random.permutation(key, indices)

    # Threshold is the (num_segments - 1)th smallest value in the first
    # num_items - 1 elements
    perm_masked = jnp.where(indices < num_items - 1, perm, padded_length)
    sorted_perm = jnp.sort(perm_masked)
    threshold = sorted_perm[num_segments - 1]

    # 1 where perm < threshold; so num_segments-1 random 1s out of the first
    # num_items-1 elements, everything else is 0
    first_in_segment = jnp.where(perm_masked < threshold, 1, 0)

    # Roll right by 1; guarantees first element is 0
    # Now there are num_segments-1 random 1s out of the first num_items elements
    # (perm_masked[padded_length-1] = padded_length because
    # padded_length-1 >= num_items-1 and
    # padded_length >= max perm_masked >= threshold)
    first_in_segment = jnp.roll(first_in_segment, 1)

    # Set position num_items to 1 (marks end of all segments)
    first_in_segment = first_in_segment.at[num_items].set(1)

    # The 1s mark the ends of the num_segments random segments, so segment_sum
    # on the cumsum gives us the segment lengths
    segment_id = jnp.cumsum(first_in_segment)
    segment_length = jax.ops.segment_sum(jnp.ones_like(segment_id), segment_id, padded_length)

    return segment_length


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_segmentation(num_items: int, num_segments: int, key: PRNGKeyArray, padded_length: int) -> jnp.ndarray:
    return jax.lax.cond(
        num_segments > 1,
        lambda: random_segmentation_2(num_items, num_segments, key, padded_length),
        lambda: random_segmentation_1(num_items, padded_length),
    )


@jax.jit
def num_noise_spans_tokens_and_spans(
    length: int, noise_density: float, mean_noise_span_length: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    adjusted_length = jnp.maximum(length, 2)

    num_noise_tokens = jnp.round(adjusted_length * noise_density).astype(jnp.int32)
    num_noise_tokens = jnp.clip(num_noise_tokens, 1, adjusted_length - 1)
    num_noise_spans = jnp.maximum(jnp.round(num_noise_tokens / mean_noise_span_length).astype(jnp.int32), 1)
    num_nonnoise_tokens = adjusted_length - num_noise_tokens

    return num_noise_tokens, num_noise_spans, num_nonnoise_tokens


@functools.partial(jax.jit, static_argnames=["padded_length"])
def random_spans_noise_mask(
    length: int,
    noise_density: float,
    key: PRNGKeyArray,
    mean_noise_span_length: float,
    random_roll: bool,
    padded_length: int,
) -> jnp.ndarray:
    """
    Generates a random 1D Boolean mask tensor where `noise_density` gives the
    fraction of tokens that are 1s, occurring in runs of length
    `mean_noise_span_length`.

    TODO Implement random_roll! This implementation has a bug because it turns
    input noise masks like 010 into target noise masks like 101, which will
    create one masked span in the input but two in the target.

    You must use `random_roll` to make each mask equally likely; otherwise the
    distribution is not uniform and the mask will have a prefix of 0s.

    Only the first `length` tokens describe the mask; the contents of the
    remaining `padded_length - length` tokens are 0s.

    Based on `random_spans_noise_mask` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2930-L3039

    Returns:
        Boolean mask tensor of shape (padded_length,).
    """
    num_noise_tokens, num_noise_spans, num_nonnoise_tokens = num_noise_spans_tokens_and_spans(
        length, noise_density, mean_noise_span_length
    )

    key1, key2, key3 = jax.random.split(key, 3)

    noise_span_lengths = random_segmentation(num_noise_tokens, num_noise_spans, key1, padded_length)
    nonnoise_span_lengths = random_segmentation(num_nonnoise_tokens, num_noise_spans, key2, padded_length)

    # Interleave using reshape
    interleaved_span_lengths = jnp.reshape(
        jnp.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [padded_length * 2],
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

    def apply_roll(m):
        offset = jax.random.randint(key3, (), 0, length, dtype=jnp.int32)
        # Roll the mask
        rolled = jnp.roll(m, offset)
        # We want to roll within [0, length) so we need to overwrite values that
        # came from the end
        rolled = jnp.where(
            indices < offset,
            jnp.roll(m, offset - length),
            rolled,
        )
        rolled = typing.cast(jnp.ndarray, rolled)
        rolled = jnp.where(indices < length, rolled, False)
        return rolled

    mask = jax.lax.cond(random_roll, apply_roll, lambda m: m, is_noise)
    return mask


@functools.partial(jax.jit, static_argnames=["force_initial_sentinel"])
def noise_span_to_unique_sentinel(
    tokens: jnp.ndarray,
    length: int,
    noise_mask: jnp.ndarray,
    sentinel_tokens: jnp.ndarray,
    force_initial_sentinel: bool,
) -> jnp.ndarray:
    """
    Replace each run of consecutive noise tokens with a different sentinel.
    `length` must be the un-padded length of `tokens`.

    For example:

        tokens = "The longest river in the world is the Amazon"
        noise_mask = [0, 1, 0, ...]
        noise_span_to_unique_sentinel(...) =
            "The <sentinel_0> river in the world is the Amazon <sentinel_0> Amazon"

    Based on `noise_span_to_unique_sentinel` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L3141

    Returns:
        A tensor with the same shape and dtype as `tokens`.
    """

    # Identify first noise tokens in each span
    prev_token_is_noise = jnp.roll(noise_mask, 1)
    prev_token_is_noise = prev_token_is_noise.at[0].set(False)
    first_noise_tokens = noise_mask & ~prev_token_is_noise
    subsequent_noise_tokens = noise_mask & prev_token_is_noise

    # max_segments = jnp.max(segments) + 1  # Number of unique noise spans

    # If max_segments > len(sentinel_tokens) we will reuse sentinel tokens which
    # isn't good. Ideally we'd log a warning but we can't do that inside of
    # jax.jit.
    # TODO Warn in the non-JIT wrapper?
    # checkify.checkify(
    #     lambda: checkify.check(
    #         max_segments <= len(sentinel_tokens),
    #         f"Too many noise spans: {max_segments} > {len(sentinel_tokens)}",
    #     ),
    #     errors=checkify.index_checks,
    # )()

    def loop_body(read_pos, state):
        out, write_pos, sentinel_count = state
        # Advance the read position from left to right in `tokens`.
        # When we see the first noise token for a span, write the sentinel for
        # the span.
        # When we see a subsequent noise token, advance the read position
        # without writing anything.
        # Otherwise just copy from `tokens`.
        sentinel_id = sentinel_tokens[sentinel_count % len(sentinel_tokens)]
        is_first_noise_token = first_noise_tokens[read_pos]
        is_subsequent_noise_token = subsequent_noise_tokens[read_pos]

        should_write_sentinel = (read_pos == -1) | is_first_noise_token
        should_write = (read_pos == -1) | ~is_subsequent_noise_token
        token_to_write = jax.lax.select(should_write_sentinel, sentinel_id, tokens[read_pos])
        sentinel_count = jax.lax.select(should_write_sentinel, sentinel_count + 1, sentinel_count)

        out, write_pos = jax.lax.cond(
            ~should_write,
            lambda: (out, write_pos),
            lambda: (out.at[write_pos].set(token_to_write), write_pos + 1),
        )

        return out, write_pos, sentinel_count

    result = jnp.zeros_like(tokens)
    i0 = jax.lax.select(force_initial_sentinel & ~first_noise_tokens[0], -1, 0)
    result, _, _ = jax.lax.fori_loop(i0, length, loop_body, (result, 0, 0))

    return result


@jax.jit
def to_ul2r_rx_tokens(
    key: PRNGKeyArray,
    tokens: jnp.ndarray,
    length: int,
    mask_prob: float,
    mean_noise_span_length: float,
    random_roll: bool,
    pad_token_id: int,
    sentinel_token_ids: jnp.ndarray,
    max_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a UL2R R/X-denoising to the first `length` elements of `tokens`.

    Based on `span_corruption` in T5:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L1931

    Does not set task tokens.

    Returns:
        - The length of `inputs` (before `targets`).
          For use when generating the loss mask / PrefixLM attention mask.
        - A tensor with the same shape as `tokens` containing
          `inputs targets 0...` where `inputs targets` is truncated to fit `max_length`.
          There is no leading padding.
    """

    padded_length = tokens.shape[0]
    noise_mask = random_spans_noise_mask(
        length,
        mask_prob,
        key,
        mean_noise_span_length,
        random_roll,
        padded_length,
    )

    # When random_roll is True we can get noise masks that end with non-noise.
    # 010 as noise_mask for inputs -> 101 as noise_mask for targets
    # foo <m1> buzz (inputs) -> <m1> bar <m2> (ouputs -- wrong)
    # want: <m1> bar
    # So for targets we must not read past the last noised part of inputs.
    target_in_len = jnp.where(noise_mask, jnp.arange(noise_mask.shape[0]), 0).max() + 1

    inputs = noise_span_to_unique_sentinel(
        tokens,
        length,
        noise_mask,
        sentinel_token_ids,
        force_initial_sentinel=False,
    )
    targets = noise_span_to_unique_sentinel(
        tokens,
        target_in_len,
        ~noise_mask,
        sentinel_token_ids,
        force_initial_sentinel=True,
    )

    indices = jnp.arange(padded_length)
    input_len = jnp.where(inputs == 0, indices, padded_length).min()
    target_len = jnp.where(targets == 0, indices, padded_length).min()

    # If `inputs + targets` exceed available `max_length`, truncate both proportionally
    combined_len = input_len + target_len
    overflow = jnp.maximum(combined_len - max_length, 0)

    # Distribute overflow proportionally between `inputs` and `targets`
    drop_inputs = jnp.where(
        combined_len > 0,
        (overflow * input_len) // jnp.maximum(combined_len, 1),
        0,
    )
    drop_targets = overflow - drop_inputs

    trunc_input_len = jnp.maximum(input_len - drop_inputs, 0)
    trunc_target_len = jnp.maximum(target_len - drop_targets, 0)

    # Truncate `targets` to the new length; `inputs` are gated by `new_input_len` below
    targets = jnp.where(indices < trunc_target_len, targets, pad_token_id)
    targets = typing.cast(jnp.ndarray, targets)

    targets = jnp.roll(targets, trunc_input_len)
    result = jnp.where(indices < trunc_input_len, inputs, targets)
    result = typing.cast(jnp.ndarray, result)
    return trunc_input_len, result


@jax.jit
def to_ul2r_s_tokens(
    key: PRNGKeyArray,
    tokens: jnp.ndarray,
    length: int,
    sentinel_token_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a S-denoising example from the first `length` elements of `tokens`.

    Unlike the T5 version, all start positions for the continuation are equally
    likely given that at least one token is included in the prefix.

    In Tay et al 2022 they mention reusing random_spans_noise_mask and setting
    noise_density to 0.25 and `mean_noise_span_length` to `length / 4`.
    for S-denoising. https://arxiv.org/abs/2210.11399
    But given their code I think this will deterministically just make the
    last quarter of the input noise?!
    https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2929-L3039
    Here we instead choose a random pivot, including at least one token from
    the beginning of the input.

    Does not set task tokens.

    Returns:
        - The length of `inputs` (before `targets`).
          For use when generating the loss mask / PrefixLM attention mask.
        - A tensor with the same shape as `tokens` containing
          `inputs prefix s[0] continuation 0...`.
    """

    # TODO support parameters?
    # TODO no longe rincluding <|mask_0|>; doing opposite of below

    # I'm not sure whether S-denoising examples look like
    #   [S] <prefix> <sentinel_0> <continuation>
    # or
    #   [S] <prefix> <continuation>
    # We currently do the former.
    # The latter would be identical to what we had as CausalLmConfig.
    # The code in UL2 mentions noise_span_to_unique_sentinel for R/X-denoising
    # but not S-denoising (see Section 9.2) and a figure in UL2R says
    # "For PaLM and U-PaLM default, we pass the input as-is to the model.
    # For the rest, we prepend one of [S2S], [NLU], or [NLG] to the
    # beginning of the input, and in the case of [NLU] and [NLG], we add
    # the infill token at the end of the input, as typical for these modes."
    # (implying there is no infill token for S-denoising) but I'm not sure.
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/e081111ccd51df425aa7124e8e311ed6d403d767/t5/data/preprocessors.py#L2033

    pivot = jax.random.randint(key, (), 1, length - 1)
    n_tokens = tokens.shape[0]
    tokens = jnp.where(jnp.arange(n_tokens) < length, tokens, 0)
    return pivot, tokens
    #targets = jnp.roll(tokens, 1)
    #indices = jnp.arange(n_tokens)
    #result = jnp.where(indices < pivot, tokens, targets)
    #result = result.at[pivot].set(sentinel_token_id)
    #return pivot, result


@jax.jit
def to_ul2r_tokens(
    key: PRNGKeyArray,
    task_params: jnp.ndarray,
    tokens: jnp.ndarray,
    length: int,
    pad_token_id: int,
    sentinel_token_ids: jnp.ndarray,
    # TODO maybe we don't actually need the truncation logic in
    # to_ul2r_rx_tokens given that we truncate while packing
    # See slice_strategy.
    # However that slices using offsets[stop] - offsets[start], which can be
    # less than the manually-specified lengths.
    max_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Applies UL2R denoising to the first `length` `tokens`.

    Returns `(inputs_len, denoising_tokens)` where
    `denoising_tokens = task_token inputs targets 0...`.

    Loss should not be computed on inputs; inputs may have PrefixLM attention.
    If the example given is too short, we return `(inputs_len = length, tokens)`
    with `tokens` unmodified. The caller should ignore this example and not
    compute loss on `tokens[:length]` at all (which is what should happen
    anyways for input tokens).
    """

    task_kind = DenoisingConfig.task_kind_from_task_params(task_params)
    task_token_id = DenoisingConfig.task_token_id_from_task_params(task_params)

    def rx_tokens():
        noise_density = RXDenoisingConfig.mask_prob_from_task_params(task_params)
        mean_noise_span_length = RXDenoisingConfig.mean_span_length_from_task_params(task_params)
        random_roll = RXDenoisingConfig.random_roll_from_task_params(task_params)
        inputs_len, out = to_ul2r_rx_tokens(
            key,
            tokens[:-1],
            length,
            noise_density,
            mean_noise_span_length,
            random_roll,
            pad_token_id,
            sentinel_token_ids,
            max_length - 1,
        )
        # Add 1 to inputs_len for the task token.
        return inputs_len + 1, jnp.concatenate([jnp.array([task_token_id], dtype=jnp.int32), out])

    def s_tokens():
        # TODO Do we ensure that the prefix/continuation isn't just <{begin,end}_of_text>?
        inputs_len, out = to_ul2r_s_tokens(key, tokens[:-1], length, sentinel_token_ids[0])
        return inputs_len + 1, jnp.concatenate([jnp.array([task_token_id], dtype=jnp.int32), out])

    def too_short():
        return length, tokens

    # Should probably remove short examples from the dataset beforehand?
    return jax.lax.cond(
        length < MIN_S_LENGTH,
        too_short,
        lambda: jax.lax.cond(task_kind == RX_TASK_KIND, rx_tokens, s_tokens),
    )


@jax.jit
def ul2r_loss_mask(
    input_mask: jnp.ndarray,
    segment_ids: jnp.ndarray,
    tokens: jnp.ndarray,
    pad_token_id: int,
) -> jnp.ndarray:
    """
    Creates a loss mask for UL2R training.

    Loss is computed only on output tokens (where input_mask is 0),
    excluding the last token in each segment and padding tokens.

        - input_mask: Binary mask indicating input positions
          positions (0)
        - segment_ids: Segment IDs for packed sequences (-1 for padding)
        - tokens: Token IDs
        - pad_token_id: Padding token ID

    Returns:
        Loss mask array of same shape as inputs
    """
    #   aaaabbb  segment_ids
    #   1100110  input_mask
    # → 0110010
    #   || ^ no loss on last output (would predict the next segment)
    #   |^ loss on last input / first output
    #   ^ no loss on inputs / task token

    loss_mask = jnp.logical_not(input_mask)
    loss_mask = jnp.roll(loss_mask, -1)

    # Don't compute loss across segment boundaries
    segment_continues = jnp.concatenate([segment_ids[1:] == segment_ids[:-1], jnp.array([True])])
    loss_mask = loss_mask & segment_continues

    # Don't compute loss on padding
    valid_mask = tokens != pad_token_id
    loss_mask = loss_mask & valid_mask

    return loss_mask


@jax.jit
def compute_denoising_length(
    task_params: jnp.ndarray,
    length: jnp.ndarray,
) -> jnp.ndarray:
    def _rx_length() -> jnp.ndarray:
        noise_density = RXDenoisingConfig.mask_prob_from_task_params(task_params)
        mean_noise_span_length = RXDenoisingConfig.mean_span_length_from_task_params(task_params)
        random_roll = RXDenoisingConfig.random_roll_from_task_params(task_params)
        _num_noise_tokens, num_noise_spans, _num_nonnoise_tokens = num_noise_spans_tokens_and_spans(
            length, noise_density, mean_noise_span_length
        )
        # When random_roll is True, we might create an additional noise span by
        # rolling a noise span so that it is cut by the beginning/end. Reserve
        # space for it.
        num_noise_spans = jax.lax.select(random_roll, num_noise_spans + 1, num_noise_spans)
        # [task_token] one <sentinel_0> three <sentinel_0> two
        return 1 + 2 * num_noise_spans + length

    def _s_length() -> jnp.ndarray:
        # [task_token] one <sentinel_0> two three
        return jax.lax.select(length < MIN_S_LENGTH, length, 2 + length)

    task_kind = DenoisingConfig.task_kind_from_task_params(task_params)
    return jax.lax.cond(task_kind == RX_TASK_KIND, _rx_length, _s_length)


@functools.partial(jax.jit, static_argnames=("max_segments_per_example", "QPos", "KPos"))
def create_ul2r_example(
    key: PRNGKeyArray,
    QPos: Axis,
    KPos: Axis,
    pad_token_id: int,
    sentinel_token_ids: jnp.ndarray,
    max_segments_per_example: int,
    task_params: jnp.ndarray,
    task_indices: jnp.ndarray,
    tokens: hax.NamedArray,
    segment_ids: hax.NamedArray,
) -> LmExample:
    # jax.debug.print("create_ul2r_example start")

    # TODO Use NamedArrays more idiomatically
    # `unique_seg_ids = [3, 4, ..., -1, ...]`
    # Valid segment IDs come first, padded with -1.
    # Sorted; assumes `segment_ids` is also sorted in ascending order.
    # We use the same ordering for `out_starts` etc.
    max_seg_id = jnp.max(segment_ids.array)
    seg_ids = jnp.where(segment_ids.array == -1, max_seg_id, segment_ids.array)
    unique_seg_ids = jnp.unique(seg_ids, size=max_segments_per_example, fill_value=-1)

    def _prepare_segment(id: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

    in_starts, in_lengths, out_lengths = jax.vmap(_prepare_segment)(unique_seg_ids)

    # `out_starts[i]` is the offset of the beginning of the i-th output segment.
    # Segment lengths increase when we turn them into denoising examples.
    out_starts = jax.lax.cumsum(out_lengths) - out_lengths
    out_starts = jnp.where(unique_seg_ids == -1, -1, out_starts)
    out_starts = typing.cast(jnp.ndarray, out_starts)

    def _process_segment(key: PRNGKeyArray, id: int) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
        """
        Applies UL2R denoising to a single segment.
        Returns `(input_mask, denoising_tokens, out_seg_ids)`.
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
        # TODO this should return the actual length, not just out_length which
        # might include an extra token? Or we could just use padding. Loss
        # shouldn't be compute don padding anyways.
        inputs_len, denoising_tokens = to_ul2r_tokens(
            key, task_params[task_idx], segment, in_length, pad_token_id, sentinel_token_ids, QPos.size
        )

        n_tokens = tokens.array.shape[0]
        input_mask = jnp.arange(n_tokens) < inputs_len
        input_mask = jnp.roll(input_mask, out_start)
        denoising_tokens = jnp.roll(denoising_tokens, out_start)
        return (input_mask, denoising_tokens, out_start, out_length)

    def _should_loop(
        acc: tuple[PRNGKeyArray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> bool:
        (_, i, _, _, _) = acc
        return (i < max_segments_per_example) & (unique_seg_ids[i] != -1)

    def _loop(
        acc: tuple[PRNGKeyArray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[PRNGKeyArray, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        (key, i, input_mask, denoising_tokens, out_seg_ids) = acc
        process_key, key = jax.random.split(key)
        (seg_input_mask, seg_denoising_tokens, out_start, out_length) = _process_segment(process_key, unique_seg_ids[i]) # type: ignore
        input_mask = input_mask | seg_input_mask
        denoising_tokens = denoising_tokens | seg_denoising_tokens

        indices = jnp.arange(out_seg_ids.shape[0])
        seg_mask = (indices >= out_start) & (indices < out_start + out_length)
        out_seg_ids = jnp.where(seg_mask, unique_seg_ids[i], out_seg_ids)
        out_seg_ids = typing.cast(jnp.ndarray, out_seg_ids)

        return (key, i + 1, input_mask, denoising_tokens, out_seg_ids)

    input_mask = jnp.zeros_like(tokens.array, dtype=jnp.bool_)
    denoising_tokens = jnp.zeros_like(tokens.array)
    out_seg_ids = jnp.full(tokens.array.shape, -1)
    acc = (key, 0, input_mask, denoising_tokens, out_seg_ids)

    # jax.debug.print("create_ul2r_example loop")

    (_, _, input_mask, denoising_tokens, out_seg_ids) = jax.lax.while_loop(_should_loop, _loop, acc)

    # jax.debug.print("create_ul2r_example loss_mask")
    # TODO GreedyPrepackedDataset pads w/ zeros so can we end up with two
    # padding token IDs?
    loss_mask = ul2r_loss_mask(input_mask, out_seg_ids, denoising_tokens, pad_token_id)
    loss_mask = hax.named(loss_mask, QPos)

    # TODO Do we not need KPos? LmExample.causal() just reuses the QPos seg_ids...
    out_seg_ids = hax.named(out_seg_ids, [QPos])
    attn_mask = AttentionMask(
        is_causal=True,
        prefix_lm_mask=hax.named(input_mask, [QPos]),
        segment_ids=(out_seg_ids, out_seg_ids),
    )

    denoising_tokens = hax.named(denoising_tokens, QPos)
    return LmExample(tokens=denoising_tokens, loss_mask=loss_mask, attn_mask=attn_mask)


TokenizedDict = typing.TypedDict("TokenizedDict", {"input_ids": np.ndarray})


# [example, seg_ids]: tuple[TokenizedDict, TokenizedDict]
class Ul2rDataset(MappedAsyncDataset[tuple[TokenizedDict, TokenizedDict], LmExample]):
    def __init__(
        self,
        cache: TreeCache[TokenizedDict],
        QPos: Axis,
        KPos: Axis,
        task_configs: typing.Dict[str, DenoisingConfig],
        task_probs: Dict[str, float],
        key: PRNGKeyArray,
        pad_token_id: int,
        sentinel_token_ids: jnp.ndarray,
        max_segments_per_example: int = 64,
        slice_strategy: Literal["left", "right", "raise"] = "left",
    ):
        cache.await_finished()

        # Copied from GreedyPrepackedDataset.__init__
        # TODO factor out?
        # TODO avoid reading store.offsets twice (here and in GreedyPrepackedDataset)
        offsets = jax.tree.map(lambda store: store.offsets[0 : store.num_rows + 1].read(), cache.store.tree)
        offsets = jax.tree.map(lambda fut: fut.result(), offsets)

        def _diff_offsets(offsets: np.ndarray):
            # fine to mutate since we have a copy
            # the array store has the number of rows in the 0th offset
            offsets[0] = 0
            return offsets[1:] - offsets[:-1]

        in_lengths = jax.tree.map(_diff_offsets, offsets)
        in_token_counts = jnp.array(in_lengths["input_ids"])
        n_docs = in_token_counts.shape[0]

        # print("Ul2rDataset after in_lengths")

        task_items = [(config, task_probs[name]) for name, config in task_configs.items()]
        n_tasks = len(task_items)
        task_probs_arr = jnp.array([prob for _, prob in task_items])
        task_indices_key, key = jax.random.split(key)
        task_indices = jax.random.choice(task_indices_key, n_tasks, shape=(n_docs,), p=task_probs_arr)

        # shape (n_tasks, A)
        # where A = max(num_args(to_ul2r_rx_tokens), num_args(to_ul2r_s_tokens))
        # e.g. [[task_kind, pad_token_id, mask_prob, mean_span_length]]
        # task_kind = 0 for R/X-denoising, 1 for S-denoising
        task_params = jnp.array([task.to_task_params() for task, _prob in task_items])

        # We compute the length of the tokens after denoising because we want
        # to turn each input batch into a denoising batch while still staying
        # under the max sequence length for the model.
        def _compute_length(task_idx: jnp.ndarray, length: jnp.ndarray) -> int:
            return compute_denoising_length(task_params[task_idx], length)

        out_token_counts = jax.vmap(_compute_length)(task_indices, in_token_counts)
        out_lengths = {**in_lengths, "input_ids": out_token_counts}

        # NB the GreedyPackedDataset returns a tuple, where the first has the
        # packed leaves and the second has the segment ids
        self.packed: GreedyPrepackedDataset[TokenizedDict] = GreedyPrepackedDataset(
            cache.store.tree,
            QPos.size,
            max_segments_per_example=max_segments_per_example,
            slice_strategy=slice_strategy,
            packing_lengths=out_lengths,
            # Reserve space for UL2R; denoising examples increase in length.
            pad_with_zeros=True,
        )
        self.QPos = QPos
        self.KPos = KPos
        self.pad_token_id = pad_token_id

        @functools.partial(eqx.filter_jit)
        def _create_lm_example(e: tuple[TokenizedDict, TokenizedDict]) -> LmExample:
            example, seg_ids = e
            tokens = hax.named(example["input_ids"], self.QPos)
            segment_ids = hax.named(seg_ids["input_ids"], self.QPos)
            return create_ul2r_example(
                key,
                self.QPos,
                self.KPos,
                self.pad_token_id,
                sentinel_token_ids,
                max_segments_per_example,
                task_params,
                task_indices,
                tokens,
                segment_ids,
            )

        super().__init__(self.packed, _create_lm_example)
