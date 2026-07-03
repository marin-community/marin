# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Hashed multi-gram INPUT embedding (Over-Tokenized / LongCat n-gram).

Adds a causal, hashed n-gram embedding to the ordinary unigram token embedding on
the input side only. For each n-gram order and each hash function the last ``n``
tokens (looking backward only) are folded into a rolling polynomial hash and used
to gather a row from a fixed-size table; the gathered vectors are summed into the
base embedding. The output head and vocabulary are untouched, so this composes
with any base tokenizer and changes neither BPB accounting nor LM-head FLOPs.

The hash is strictly causal: the id at position ``i`` depends only on tokens at
positions ``<= i`` (front-padded with a zero token), so it never leaks future
tokens into an autoregressive model.
"""

import math
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from levanter.grug.sharding import Pembed_vocab, _batch_spec, _current_mesh, _reshard_for_init

# Rolling-hash accumulator arithmetic stays in int32 so it is correct whether or
# not JAX x64 is enabled. With the accumulator and the (reduced) token both below
# ``hash_buckets``, the intermediate ``acc * mult + token`` is bounded by
# ``hash_buckets * (mult + 1)``; capping ``mult`` below this keeps it under 2**31.
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class NgramEmbedConfig:
    """Configuration for the hashed multi-gram input embedding.

    Attributes:
        orders: N-gram orders to hash (e.g. ``(2, 3)`` for bigrams and trigrams).
        num_hashes: Number of independent hash functions per order.
        hash_buckets: Row count of every hash table; a large prime keeps
            collisions generic.
        rank: Low-rank width of the per-pair sub-embedder. ``None`` uses the full
            hidden dimension and no up-projection.
        combine: How the n-gram terms are folded into the base embedding.
            ``"sum"`` adds them; ``"mean"`` averages base plus all n-gram terms
            (divisor ``len(orders) * num_hashes + 1``), matching LongCat.
        init_std_scale: Scale of the truncated-normal table init (std
            ``init_std_scale / sqrt(width)``). ``0.0`` starts the tables at exactly
            zero, making the n-gram contribution zero. With ``combine="sum"`` the
            model is then bit-identical to the no-n-gram baseline (a clean ablation
            start); with ``combine="mean"`` the base embedding is still rescaled by
            ``1/(len(orders)*num_hashes + 1)``, which the following embedding RMS
            norm absorbs only up to its ``eps`` term.
    """

    orders: tuple[int, ...] = (2, 3)
    num_hashes: int = 2
    hash_buckets: int = 1048573
    rank: int | None = None
    combine: Literal["sum", "mean"] = "mean"
    init_std_scale: float = 0.0

    def __post_init__(self) -> None:
        if len(self.orders) == 0:
            raise ValueError("orders must contain at least one n-gram order")
        if any(order < 1 for order in self.orders):
            raise ValueError(f"orders must be >= 1, got {self.orders}")
        if self.num_hashes < 1:
            raise ValueError(f"num_hashes must be >= 1, got {self.num_hashes}")
        if self.hash_buckets < 2:
            raise ValueError(f"hash_buckets must be >= 2, got {self.hash_buckets}")
        if self.rank is not None and self.rank < 1:
            raise ValueError(f"rank must be >= 1 when set, got {self.rank}")
        if self.combine not in ("sum", "mean"):
            raise ValueError(f"combine must be 'sum' or 'mean', got {self.combine!r}")

    @property
    def num_pairs(self) -> int:
        """Number of (order, hash) tables held by the embedder."""
        return len(self.orders) * self.num_hashes


def _primes_below(limit: int) -> tuple[int, ...]:
    """Return all primes strictly below ``limit`` in ascending order."""
    if limit < 3:
        return ()
    sieve = bytearray([1]) * limit
    sieve[0] = 0
    sieve[1] = 0
    for candidate in range(2, int(limit**0.5) + 1):
        if sieve[candidate]:
            sieve[candidate * candidate :: candidate] = b"\x00" * len(sieve[candidate * candidate :: candidate])
    return tuple(index for index in range(2, limit) if sieve[index])


def _hash_multipliers(num_pairs: int, hash_buckets: int) -> tuple[int, ...]:
    """Pick ``num_pairs`` distinct prime multipliers, one per (order, hash) pair.

    The multipliers are capped so the rolling hash never overflows int32 (see the
    module note); the largest primes under that cap are chosen for the best
    mixing. They are prime and far below any real vocabulary, so they are never a
    multiple of ``vocab_size`` and stay coprime to ``hash_buckets``.
    """
    mult_cap = _INT32_MAX // hash_buckets - 1
    if mult_cap < 3:
        raise ValueError(f"hash_buckets={hash_buckets} is too large for int32-safe hashing")
    primes = _primes_below(mult_cap + 1)
    if len(primes) < num_pairs:
        raise ValueError(
            f"need {num_pairs} distinct hash multipliers below {mult_cap + 1} but only {len(primes)} primes exist; "
            "reduce orders/num_hashes or hash_buckets"
        )
    return primes[-num_pairs:]


def _shift_back(token_ids: Int[Array, "B S"], offset: int) -> Int[Array, "B S"]:
    """Shift tokens toward higher indices by ``offset``, front-padding with zeros.

    ``result[:, i] == token_ids[:, i - offset]`` for ``i >= offset`` and ``0``
    otherwise, so gathering at ``offset`` positions back never reads the future.
    """
    if offset == 0:
        return token_ids
    seq_len = token_ids.shape[1]
    # Build the zero pad from a token_ids slice (× 0) rather than jnp.zeros so it inherits
    # token_ids' batch sharding. Under the model's explicit mesh axes a freshly created array is
    # committed to a replicated sharding, and concatenating it with the batch-sharded token slice
    # raises ShardingTypeError; two slices of token_ids share one sharding and concatenate cleanly.
    pad = token_ids[:, :offset] * 0
    return jnp.concatenate([pad, token_ids[:, : seq_len - offset]], axis=1)


def _causal_ngram_hash(
    token_ids: Int[Array, "B S"], order: int, multiplier: int, hash_buckets: int
) -> Int[Array, "B S"]:
    """Rolling polynomial hash of the causal length-``order`` window ending at each position.

    Uses Horner's method with per-step modular reduction, keeping every
    intermediate below ``2**31`` so the result is correct without x64.
    """
    reduced = jnp.mod(token_ids.astype(jnp.int32), hash_buckets)
    acc = jnp.zeros_like(reduced)
    for offset in range(order - 1, -1, -1):
        acc = jnp.mod(acc * multiplier + _shift_back(reduced, offset), hash_buckets)
    return acc


class NgramInputEmbed(eqx.Module):
    """Additive hashed multi-gram embedding for the model input.

    Holds one ``(hash_buckets, width)`` table per (order, hash) pair, where
    ``width`` is ``rank`` (with a per-pair up-projection to hidden) or the full
    hidden dimension. Tables are row-sharded on the model axis like the base token
    embedding.
    """

    tables: tuple[jax.Array, ...]
    up_projs: tuple[jax.Array, ...] | None
    orders: tuple[int, ...] = eqx.field(static=True)
    num_hashes: int = eqx.field(static=True)
    hash_buckets: int = eqx.field(static=True)
    multipliers: tuple[int, ...] = eqx.field(static=True)
    combine: Literal["sum", "mean"] = eqx.field(static=True)

    @staticmethod
    def init(config: NgramEmbedConfig, hidden_dim: int, *, key: PRNGKeyArray) -> "NgramInputEmbed":
        """Build the embedder, mirroring the base token-embedding init and sharding."""
        num_pairs = config.num_pairs
        width = config.rank if config.rank is not None else hidden_dim
        multipliers = _hash_multipliers(num_pairs, config.hash_buckets)

        table_std = config.init_std_scale / math.sqrt(width)
        keys = random.split(key, 2 * num_pairs)
        tables: list[jax.Array] = []
        up_projs: list[jax.Array] | None = [] if config.rank is not None else None
        for pair in range(num_pairs):
            table = table_std * random.truncated_normal(keys[pair], -3, 3, (config.hash_buckets, width))
            tables.append(_reshard_for_init(table, Pembed_vocab))
            if config.rank is not None:
                up = (1.0 / math.sqrt(config.rank)) * random.truncated_normal(
                    keys[num_pairs + pair], -3, 3, (config.rank, hidden_dim)
                )
                # up_projs is a concrete list on this branch; the guard is for the type checker.
                assert up_projs is not None
                up_projs.append(_reshard_for_init(up, P(None, None)))

        return NgramInputEmbed(
            tables=tuple(tables),
            up_projs=tuple(up_projs) if up_projs is not None else None,
            orders=tuple(config.orders),
            num_hashes=config.num_hashes,
            hash_buckets=config.hash_buckets,
            multipliers=multipliers,
            combine=config.combine,
        )

    def _extra_embedding(self, token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        """Sum of the hashed n-gram embeddings at each position (no base term)."""
        mesh = _current_mesh()
        out_spec = _batch_spec(mesh) if mesh is not None and not mesh.empty else None
        total: jax.Array | None = None
        pair = 0
        for order in self.orders:
            for _ in range(self.num_hashes):
                idx = _causal_ngram_hash(token_ids, order, self.multipliers[pair], self.hash_buckets)
                gathered = _gather_rows(self.tables[pair], idx, out_spec)
                if self.up_projs is not None:
                    gathered = _up_project(gathered, self.up_projs[pair], out_spec)
                total = gathered if total is None else total + gathered
                pair += 1
        assert total is not None
        return total

    def combine_into(self, token_hidden: Float[Array, "B S D"], token_ids: Int[Array, "B S"]) -> Float[Array, "B S D"]:
        """Fold the hashed n-gram embeddings into the base token embedding.

        Args:
            token_hidden: The base unigram embedding, shape ``(B, S, D)``.
            token_ids: The input token ids, shape ``(B, S)``.

        Returns:
            The combined ``(B, S, D)`` embedding. With ``combine="mean"`` the base
            plus all n-gram terms are averaged; with ``"sum"`` they are added.
        """
        combined = token_hidden + self._extra_embedding(token_ids)
        if self.combine == "mean":
            combined = combined / (len(self.orders) * self.num_hashes + 1)
        return combined


def _gather_rows(table: jax.Array, indices: Int[Array, "B S"], out_spec: P | None) -> Float[Array, "B S W"]:
    if out_spec is None:
        return table[indices]
    return table.at[indices].get(out_sharding=out_spec)


def _up_project(gathered: Float[Array, "B S R"], up_proj: jax.Array, out_spec: P | None) -> Float[Array, "B S D"]:
    if out_spec is None:
        return jnp.einsum("bsr,rd->bsd", gathered, up_proj)
    return jnp.einsum("bsr,rd->bsd", gathered, up_proj, out_sharding=out_spec)


__all__ = ["NgramEmbedConfig", "NgramInputEmbed"]
