# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-loss probe primitives for the candidate 11 port of Noise.

These are building blocks, not a complete Boost/Cut estimator. No autodiff is
used. The rectangular direction frame supports Belay's non-power-of-two widths.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import hadamard

PROBE_RMS_FLOOR = 1e-6


def direction_frame(width: int, key: jax.Array) -> jax.Array:
    """Return a signed frame H with H.T @ H = order * I.

    At power-of-two widths this is Noise's full Hadamard cycle. At other
    widths, retain the first width columns of the next larger cycle. The
    decoder must divide by the number of rows, not the model width.
    """
    order = 1 << (width - 1).bit_length()
    frame = jnp.asarray(hadamard(order, dtype=np.int8)[:, :width], jnp.float32)
    signs = jax.random.rademacher(key, (width,), dtype=jnp.float32)
    return frame * signs


def token_readout_credit(
    loss: Callable[[jax.Array], jax.Array],
    point: jax.Array,
    directions: jax.Array,
    *,
    relative_step: float,
    chunk_size: int,
) -> jax.Array:
    """Measure local readout credit by a full cycle of central forward probes.

    ``loss`` maps [tokens, width] to [tokens] and must not mix tokens. This
    function is not a probe of a sequence mixer such as KDA or attention.
    """
    point = point.astype(jnp.float32)
    count = directions.shape[0]
    step = relative_step * jnp.maximum(jnp.sqrt(jnp.mean(point**2, axis=-1)), PROBE_RMS_FLOOR)
    padding = (-count) % chunk_size
    chunks = jnp.pad(directions, ((0, padding), (0, 0))).reshape(-1, chunk_size, point.shape[-1])

    def probe(total, rows):
        moves = rows[:, None, :] * step[None, :, None]
        upper = jax.vmap(loss)(point[None, :, :] + moves)
        lower = jax.vmap(loss)(point[None, :, :] - moves)
        slopes = (upper - lower) / (2 * step[None, :])
        return total + slopes.T @ rows / count, None

    credit, _ = jax.lax.scan(probe, jnp.zeros_like(point), chunks)
    return credit


def head_coordinate_credit(
    raw_logits: jax.Array,
    labels: jax.Array,
    *,
    sigma: float,
    soft_cap: float,
    z_loss_weight: float,
) -> jax.Array:
    """Evaluate coordinate finite differences of capped CE plus Belay's z-loss.

    Updating one logit changes only one term of the partition function, so
    these are exact moved forward losses without materializing a vocabulary
    squared tensor. Labels below zero denote ignored positions.
    """
    raw = raw_logits.astype(jnp.float32)
    capped = soft_cap * jnp.tanh(raw / soft_cap)
    upper = soft_cap * jnp.tanh((raw + sigma) / soft_cap)
    lower = soft_cap * jnp.tanh((raw - sigma) / soft_cap)
    log_partition = jax.scipy.special.logsumexp(capped, axis=-1, keepdims=True)
    probability = jnp.exp(capped - log_partition)

    def moved_loss(moved):
        change = jnp.log1p(probability * jnp.expm1(moved - capped))
        return change + z_loss_weight * change * (2 * log_partition + change)

    change = moved_loss(upper) - moved_loss(lower)
    target = jax.nn.one_hot(jnp.maximum(labels, 0), raw.shape[-1])
    change = change - target * (upper - lower)
    return jnp.where(labels[..., None] >= 0, change / (2 * sigma), 0)


def capped_token_loss(raw_logits: jax.Array, labels: jax.Array, soft_cap: float, z_loss_weight: float) -> jax.Array:
    capped = soft_cap * jnp.tanh(raw_logits / soft_cap)
    partition = jax.scipy.special.logsumexp(capped, axis=-1)
    target = jnp.take_along_axis(capped, jnp.maximum(labels, 0)[..., None], axis=-1)[..., 0]
    return jnp.where(labels >= 0, partition - target + z_loss_weight * partition**2, 0)


def attnres_readout_credit(
    partial: jax.Array,
    fixed_sources: jax.Array,
    query: jax.Array,
    head: jax.Array,
    norm_gain: jax.Array,
    labels: jax.Array,
    directions: jax.Array,
    *,
    relative_step: float,
    chunk_size: int,
    epsilon: float,
    soft_cap: float,
    z_loss_weight: float,
) -> jax.Array:
    """Probe the candidate 11 depth-attention readout without repeated head GEMMs.

    Inputs are one token tile: partial [T,D], fixed sources [N,T,D], head
    [D,V], and directions [K,D]. Every probe recomputes depth attention, final
    RMS normalization, capped cross entropy and z-loss by forward evaluation.
    Linear head projections and dot products are shared across the probes.
    """
    partial, sources = partial.astype(jnp.float32), fixed_sources.astype(jnp.float32)
    query, head = query.astype(jnp.float32), head.astype(jnp.float32)
    width = partial.shape[-1]
    fixed_logits = jnp.einsum("ntd,d->nt", sources, query) * jax.lax.rsqrt(jnp.mean(sources**2, axis=-1) + epsilon)
    fixed_partition = jax.scipy.special.logsumexp(fixed_logits, axis=0)
    fixed = jnp.einsum("nt,ntd->td", jax.nn.softmax(fixed_logits, axis=0), sources)
    weighted_head = norm_gain.astype(jnp.float32)[:, None] * head
    fixed_head, partial_head = fixed @ weighted_head, partial @ weighted_head
    partial_square = jnp.sum(partial**2, axis=-1)
    partial_query = partial @ query
    partial_norm = jnp.sqrt(partial_square / width + epsilon)
    partial_logit = partial_query / partial_norm
    clean_weight = jax.nn.sigmoid(partial_logit - fixed_partition)
    clean_mix = (1 - clean_weight)[:, None] * fixed + clean_weight[:, None] * partial
    clean_norm = jnp.sqrt(jnp.mean(clean_mix**2, axis=-1) + epsilon)
    clean_head = (1 - clean_weight)[:, None] * fixed_head + clean_weight[:, None] * partial_head
    clean_raw = clean_head / clean_norm[:, None]
    clean_tanh = jnp.tanh(clean_raw / soft_cap)
    clean_capped = soft_cap * clean_tanh
    clean_partition = jax.scipy.special.logsumexp(clean_capped, axis=-1)
    clean_probability = jax.nn.softmax(clean_capped, axis=-1)
    step = relative_step * jnp.maximum(jnp.sqrt(partial_square / width), PROBE_RMS_FLOOR)
    count = directions.shape[0]
    padding = (-count) % chunk_size
    chunks = jnp.pad(directions, ((0, padding), (0, 0))).reshape(-1, chunk_size, width)

    def probe(total, rows):
        row_head = rows @ weighted_head
        row_query = rows @ query

        def loss_changes(sign):
            displacement = sign * step[:, None]
            move = displacement[..., None] * rows[None, :, :]
            moved_norm = jnp.sqrt(jnp.mean((partial[:, None, :] + move) ** 2, axis=-1) + epsilon)
            norm_change = jnp.mean(2 * partial[:, None, :] * move + move**2, axis=-1)
            inverse_change = -norm_change / (moved_norm * partial_norm[:, None] * (moved_norm + partial_norm[:, None]))
            logit_change = displacement * row_query / moved_norm + partial_query[:, None] * inverse_change
            half_tanh = jnp.tanh(logit_change / 2)
            alpha = clean_weight[:, None]
            alpha_change = 2 * alpha * (1 - alpha) * half_tanh / (1 + (2 * alpha - 1) * half_tanh)
            moved_alpha = alpha + alpha_change
            mix_move = alpha_change[..., None] * (partial - fixed)[:, None, :] + moved_alpha[..., None] * move
            mix_norm = jnp.sqrt(jnp.mean((clean_mix[:, None, :] + mix_move) ** 2, axis=-1) + epsilon)
            mix_norm_change = jnp.mean(2 * clean_mix[:, None, :] * mix_move + mix_move**2, axis=-1)
            mix_inverse_change = -mix_norm_change / (mix_norm * clean_norm[:, None] * (mix_norm + clean_norm[:, None]))
            head_move = (
                alpha_change[..., None] * (partial_head - fixed_head)[:, None, :]
                + moved_alpha[..., None] * displacement[..., None] * row_head[None, :, :]
            )
            raw_change = head_move / mix_norm[..., None] + clean_head[:, None, :] * mix_inverse_change[..., None]
            # Evaluate tanh(z + delta) - tanh(z) without subtracting nearby logits.
            tanh_move = jnp.tanh(raw_change / soft_cap)
            capped_change = (
                soft_cap * tanh_move * (1 - clean_tanh[:, None, :] ** 2) / (1 + clean_tanh[:, None, :] * tanh_move)
            )
            partition_change = jnp.log1p(jnp.sum(clean_probability[:, None, :] * jnp.expm1(capped_change), axis=-1))
            target = jnp.broadcast_to(jnp.maximum(labels, 0)[:, None, None], (*partition_change.shape, 1))
            target_change = jnp.take_along_axis(capped_change, target, axis=-1)[..., 0]
            change = (
                partition_change
                - target_change
                + z_loss_weight * partition_change * (2 * clean_partition[:, None] + partition_change)
            )
            return jnp.where(labels[:, None] >= 0, change, 0)

        slopes = (loss_changes(1) - loss_changes(-1)) / (2 * step[:, None])
        return total + slopes @ rows / count, None

    credit, _ = jax.lax.scan(probe, jnp.zeros_like(partial), chunks)
    return credit


class ReadoutParameterCredit(NamedTuple):
    head: jax.Array
    norm_gain: jax.Array
    query: jax.Array


def capped_loss_change(
    clean_raw: jax.Array,
    raw_change: jax.Array,
    labels: jax.Array,
    *,
    soft_cap: float,
    z_loss_weight: float,
) -> jax.Array:
    """Evaluate a finite readout loss change without subtracting nearby losses.

    The first dimensions broadcast: callers insert a probe dimension into
    ``clean_raw`` and ``labels`` when evaluating several moves together.
    """
    clean_tanh = jnp.tanh(clean_raw / soft_cap)
    clean_capped = soft_cap * clean_tanh
    partition = jax.scipy.special.logsumexp(clean_capped, axis=-1)
    probability = jax.nn.softmax(clean_capped, axis=-1)
    tanh_move = jnp.tanh(raw_change / soft_cap)
    capped_change = soft_cap * tanh_move * (1 - clean_tanh**2) / (1 + clean_tanh * tanh_move)
    partition_change = jnp.log1p(jnp.sum(probability * jnp.expm1(capped_change), axis=-1))
    target = jnp.broadcast_to(jnp.maximum(labels, 0)[..., None], (*partition_change.shape, 1))
    target_change = jnp.take_along_axis(capped_change, target, axis=-1)[..., 0]
    loss_change = (
        partition_change - target_change + z_loss_weight * partition_change * (2 * partition + partition_change)
    )
    return jnp.where(labels >= 0, loss_change, 0)


def readout_parameter_credit(
    sources: jax.Array,
    query: jax.Array,
    head: jax.Array,
    norm_gain: jax.Array,
    labels: jax.Array,
    loss_weight: jax.Array,
    directions: jax.Array,
    *,
    head_step: float,
    query_step: float,
    direction_chunk: int,
    epsilon: float,
    soft_cap: float,
    z_loss_weight: float,
) -> ReadoutParameterCredit:
    """Measure shared readout parameter updates on a token tile, without AD.

    ``sources`` has shape [sources, tokens, width]. Head and learned RMS gain
    updates transport directly measured logit credit through linear maps.
    The depth query uses its own complete finite forward-loss cycle through
    depth mixing, final normalization, head and capped loss. Query moves use
    an absolute step because candidate 11 initializes this query at zero.
    Returns unnormalized weighted sums; the caller owns batch normalization.
    """
    sources, query, head, gain = (array.astype(jnp.float32) for array in (sources, query, head, norm_gain))
    normalized_sources = sources * jax.lax.rsqrt(jnp.mean(sources**2, axis=-1, keepdims=True) + epsilon)
    logits = normalized_sources @ query
    weights = jax.nn.softmax(logits, axis=0)
    mixed = jnp.sum(weights[..., None] * sources, axis=0)
    norm = jnp.sqrt(jnp.mean(mixed**2, axis=-1) + epsilon)
    normalized = mixed / norm[:, None]
    hidden = normalized * gain
    clean_head = (mixed * gain) @ head
    raw = clean_head / norm[:, None]
    logit_credit = (
        head_coordinate_credit(raw, labels, sigma=head_step, soft_cap=soft_cap, z_loss_weight=z_loss_weight)
        * loss_weight[:, None]
    )
    head_credit = hidden.T @ logit_credit
    gain_credit = jnp.sum(normalized * (logit_credit @ head.T), axis=0)
    count, width = directions.shape
    padding = (-count) % direction_chunk
    chunks = jnp.pad(directions, ((0, padding), (0, 0))).reshape(-1, direction_chunk, width)

    def probe(total, rows):
        logit_moves = query_step * jnp.einsum("ntd,kd->ntk", normalized_sources, rows)

        def loss_changes(sign):
            exp_change = jnp.expm1(sign * logit_moves)
            partition_change = jnp.sum(weights[..., None] * exp_change, axis=0)
            weight_change = weights[..., None] * (exp_change - partition_change[None]) / (1 + partition_change[None])
            mix_change = jnp.einsum("ntk,ntd->tkd", weight_change, sources)
            moved_norm = jnp.sqrt(jnp.mean((mixed[:, None] + mix_change) ** 2, axis=-1) + epsilon)
            square_change = jnp.mean(2 * mixed[:, None] * mix_change + mix_change**2, axis=-1)
            inverse_change = -square_change / (moved_norm * norm[:, None] * (moved_norm + norm[:, None]))
            raw_change = (mix_change * gain) @ head / moved_norm[..., None] + clean_head[:, None] * inverse_change[
                ..., None
            ]
            return capped_loss_change(
                raw[:, None],
                raw_change,
                labels[:, None],
                soft_cap=soft_cap,
                z_loss_weight=z_loss_weight,
            )

        slopes = (loss_changes(1) - loss_changes(-1)) / (2 * query_step)
        return (
            total + jnp.sum(slopes * loss_weight[:, None], axis=0) @ rows / count,
            None,
        )

    query_credit, _ = jax.lax.scan(probe, jnp.zeros_like(query), chunks)
    return ReadoutParameterCredit(head_credit, gain_credit, query_credit)


class Relu2ParameterCredit(NamedTuple):
    up: jax.Array
    down: jax.Array


def relu2_parameter_credit(
    inputs: jax.Array,
    preactivation: jax.Array,
    output_weight: jax.Array,
    output_credit: jax.Array,
    *,
    relative_step: float,
) -> Relu2ParameterCredit:
    """Reconstruct an ungated ReLU-squared expert's two matrix updates.

    The caller supplies independently measured output credit, already weighted
    by the objective. The hidden response is Noise's collapsed full central
    Hadamard cycle, retaining finite-step ReLU crossings. It must not be reused
    as credit through a further nonlinear operation. Return unnormalized sums.
    """
    # Preserve the clean forward's activation rounding for its linear readout.
    hidden = jnp.square(jax.nn.relu(preactivation))
    if preactivation.dtype == jnp.bfloat16:
        # A cast alone can be removed when XLA fuses this with the f32 readout.
        hidden = jax.lax.reduce_precision(hidden.astype(jnp.float32), exponent_bits=8, mantissa_bits=7)
    hidden = hidden.astype(jnp.float32)
    inputs = inputs.astype(jnp.float32)
    point = preactivation.astype(jnp.float32)
    weight = output_weight.astype(jnp.float32)
    credit = output_credit.astype(jnp.float32)
    step = relative_step * jnp.maximum(jnp.sqrt(jnp.mean(point**2, axis=-1, keepdims=True)), PROBE_RMS_FLOOR)
    upper = jax.nn.relu(point + step)
    # Exact piecewise central difference: avoid subtracting nearby positive
    # preactivations, which loses precision at small finite probe steps.
    finite_slope = jnp.where(point >= step, 2 * point, upper**2 / (2 * step))
    hidden_credit = (credit @ weight.T) * finite_slope
    return Relu2ParameterCredit(inputs.T @ hidden_credit, hidden.T @ credit)


def causal_convolution_credit(
    output_credit: jax.Array,
    weight: jax.Array,
    segment_ids: jax.Array | None,
) -> jax.Array:
    """Transport measured credit through a fixed linear causal convolution.

    This is an exact linear contraction, with future responses restricted to
    the same packed document. It neither estimates a nonlinear derivative nor
    invokes the model's backward kernels.
    """
    credit = output_credit.astype(jnp.float32)
    weight = weight.astype(jnp.float32)
    result = jnp.zeros_like(credit)
    length = credit.shape[1]
    for lag in range(min(weight.shape[0], length)):
        term = credit[:, lag:] * weight[lag]
        if segment_ids is not None:
            valid = segment_ids[:, lag:] == segment_ids[:, : length - lag]
            term = jnp.where(valid[..., None], term, 0)
        result = result + jnp.pad(term, ((0, 0), (0, lag), (0, 0)))
    return result


def causal_convolution_parameter_credit(
    inputs: jax.Array,
    output_credit: jax.Array,
    segment_ids: jax.Array | None,
    *,
    kernel_size: int,
) -> jax.Array:
    """Contract independent output credit into causal depthwise kernel weights.

    Inputs and credits are [batch, sequence, channels]. Packed-document edges
    stop each lag, matching native ShortConv. Return unnormalized [lag, channel]
    sums; the caller combines sums across batch shards.
    """
    inputs = inputs.astype(jnp.float32)
    credit = output_credit.astype(jnp.float32)
    length = inputs.shape[1]
    rows = []
    for lag in range(kernel_size):
        if lag >= length:
            rows.append(jnp.zeros((inputs.shape[-1],), jnp.float32))
            continue
        term = inputs[:, : length - lag] * credit[:, lag:]
        if segment_ids is not None:
            valid = segment_ids[:, lag:] == segment_ids[:, : length - lag]
            term = jnp.where(valid[..., None], term, 0)
        rows.append(jnp.sum(term, axis=(0, 1)))
    return jnp.stack(rows)


def sqrt_softplus_router_credit(
    logits: jax.Array,
    expert_scores: jax.Array,
    kept: jax.Array,
    *,
    sigma: float,
    renorm_sum: float,
) -> jax.Array:
    """Measure selected-router logits through the original measured output hub.

    All arguments have shape [tokens, selected experts]. ``expert_scores`` is
    each frozen expert output contracted with independently measured hub credit.
    Each coordinate gets its own positive/negative forward probe. Selection and
    capacity masks stay fixed, while sqrt-softplus combine weights and native
    post-drop renormalization are recomputed. Return unnormalized token credits.
    """
    logits = logits.astype(jnp.float32)
    scores = expert_scores.astype(jnp.float32)
    softplus = jax.nn.softplus(logits)
    mass = jnp.sqrt(jnp.maximum(softplus, 1e-30))
    total = jnp.sum(mass, axis=-1, keepdims=True)
    surviving = jnp.sum(jnp.where(kept, mass, 0), axis=-1, keepdims=True)
    numerator = jnp.sum(jnp.where(kept, mass * scores, 0), axis=-1, keepdims=True)
    average = numerator / jnp.where(surviving > 0, surviving, 1)
    scale = renorm_sum * total / (total + 1e-9)

    def moved_loss_change(step):
        # log(1 + exp(x + step)) - log(1 + exp(x)), evaluated stably.
        softplus_change = jnp.log1p(jax.nn.sigmoid(logits) * jnp.expm1(step))
        moved = jnp.maximum(softplus + softplus_change, 1e-30)
        change = jnp.where(
            (softplus > 1e-30) & (softplus + softplus_change > 1e-30),
            softplus_change,
            moved - jnp.maximum(softplus, 1e-30),
        ) / (jnp.sqrt(moved) + mass)
        average_change = jnp.where(
            kept,
            change * (scores - average) / jnp.where(kept, surviving + change, 1),
            0,
        )
        scale_change = renorm_sum * 1e-9 * change / ((total + 1e-9) * (total + change + 1e-9))
        return scale * average_change + scale_change * (average + average_change)

    return (moved_loss_change(sigma) - moved_loss_change(-sigma)) / (2 * sigma)
