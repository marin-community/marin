# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jaxtyping
from haliax import NamedArray
from haliax import haxtyping as ht
from haliax.jax_utils import ensure_scalar
from jax import numpy as jnp

from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID, is_stop_signal, is_valid, masked_set, purge


class PackedSequence(eqx.Module):
    """
    A sequence of tokens packed into a single array, with
    This is used to pack sequences into a single array for efficient processing.

    Boundaries for sampling are now computed using PageBatchInfo.seq_lens and these pos_ids in the generation loop.
    """

    tokens: ht.i32[NamedArray, "position"]  # packed tokens
    slot_ids: ht.i32[NamedArray, "position"]  # local slot ids for each token
    pos_ids: ht.i32[NamedArray, "position"]  # position ids for each token
    num_tokens: jax.Array  # number of tokens in the packed sequence

    def token_counts_per_slot(self, max_slots: int) -> ht.i32[NamedArray, "seq"]:  # type: ignore[name-defined]
        """
        Returns the number of tokens per slot in the packed sequence.
        The result is a vector of size `max_slots`, where each entry corresponds to a slot ID.
        """
        raw_slot_ids = self.slot_ids.array
        weights = jnp.where(jnp.arange(len(raw_slot_ids)) < self.num_tokens, 1, 0)
        counts = jnp.bincount(raw_slot_ids, weights=weights, length=max_slots)

        return hax.named(counts, axis=("seq",))


class SeqDecodingParams(eqx.Module):
    """Per-sequence decoding parameters."""

    max_num_tokens: jnp.ndarray
    stop_tokens: ht.i32[NamedArray, "stop_seq position"] | None
    temperature: jnp.ndarray
    key: jaxtyping.PRNGKeyArray

    @staticmethod
    def default() -> "SeqDecodingParams":
        """
        Returns a default SeqDecodingParams with the given number of stop sequences and maximum stop tokens.
        """
        max_int_jnp = jnp.iinfo(jnp.int32).max
        return SeqDecodingParams(
            max_num_tokens=jnp.array(max_int_jnp - 100000, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(0),
        )


class DecodeState(eqx.Module):
    """
    State of sequences during decoding. This manages a "hot set" of sequences that are currently being decoded.

    * `tokens` is a buffer of tokens for each sequence. It includes any prompt/prefix.
    * `seq_lens` is a buffer of sequence lengths for each sequence. This is the number of tokens in the `tokens` buffer that have been generated so far.
    * `logprobs` is an optional buffer of log probabilities for the tokens. If not None, it should have the same shape
       as `tokens`, i.e. `logprobs["seq", i, "position", j]` is the log probability of the token at position `j` in
       sequence `i`. It is kept in sync with `tokens`, i.e. if a token is generated, its log probability is also
       generated. We don't currently compute log probabilities for the prefix tokens, so `logprobs` is set to nan for
       those positions.
    """

    tokens: ht.i32[NamedArray, "seq position"]
    """ most recent tokens generated for each sequence. Should always start at a page boundary. """
    logprobs: ht.Float[NamedArray, "seq position"] | None  # log probabilities of the tokens
    seq_lens: ht.i32[NamedArray, "seq"]
    """Sequence length for each sequence. This is the number of tokens currently in the sequence"""
    clone_sources: ht.i32[NamedArray, "seq"]
    """
    For each local sequence slot, the local source id it should be cloned from, or INVALID if it's either already
    been cloned or is an original sequence. This is used to implement efficient cloning of sequences for multi-sample
    decoding or potentially beam search / particle filtering.
    """

    # TODO: these aren't actually used anywhere. (We currently only use them from PageTable)
    # This is a better place for them, so we should move them here and update the code to use them.
    kv_pages: ht.i32[NamedArray, "seq page"]
    """Key-value pages for each sequence. This is used to store the key-value pairs for the sequences."""
    page_size: int = eqx.field(static=True)

    # Page table for KV page allocation and per-sequence lengths/usage
    page_table: PageTable

    # Per sequence sampling parameters
    max_num_tokens: ht.i32[NamedArray, "seq"]
    """
    Maximum number of tokens for each sequence. This is used to limit the number of tokens generated.
    This is inclusive of the prefix length, i.e. the total number of tokens that can be generated for each sequence.
    """
    stop_tokens: ht.i32[NamedArray, "seq stop_seq position"] | None
    """Stop sequences for each sequence. If None, no stop sequences are used. **Left padded** with pad_token_id."""
    temperature: ht.Float[NamedArray, "seq"]
    """temperature for sampling. 0 means greedy sampling"""
    prng_keys: jaxtyping.PRNGKeyArray
    """one per sequence, used for sampling. This is a JAX PRNG key, so it can be split to get new keys."""

    # Token queue for pending decode work
    tqueue: "TokenQueue"

    # Cached finished flags per sequence (updated when tokens are enqueued)
    finished: ht.bool_[NamedArray, "seq"]

    @eqx.filter_jit(donate="all")
    def invalidate_finished(self) -> "DecodeState":
        """Invalidate metadata for sequences marked finished by ``finished_mask``.

        - Sets ``seq_lens`` to INVALID for finished slots
        - Resets ``clone_sources`` to INVALID
        - Clears ``kv_pages`` rows for finished slots to INVALID
        """
        mask = self.finished
        new_seq_lens = hax.where(mask, INVALID, self.seq_lens)
        new_clone_sources = hax.where(mask, INVALID, self.clone_sources)
        new_kv_pages = hax.where(mask, INVALID, self.kv_pages)
        finished = hax.zeros_like(self.finished)  # reset finished flags

        return dataclasses.replace(
            self,
            seq_lens=new_seq_lens,
            clone_sources=new_clone_sources,
            kv_pages=new_kv_pages,
            finished=finished,
        )

    def prng_key_for(self, slot_id: int, pos_id: int) -> jaxtyping.PRNGKeyArray:
        """
        Get the PRNG key for the given slot ID and position.
        This is used to sample new tokens for the given slot ID and position.
        """
        per_pos_key = self.prng_keys[ensure_scalar(slot_id)]
        return jax.random.fold_in(per_pos_key, ensure_scalar(pos_id))

    def prng_keys_for(self, slot_ids: ht.i32[NamedArray, "position"], pos_ids: ht.i32[NamedArray, "position"]) -> jaxtyping.PRNGKeyArray:  # type: ignore[name-defined]
        """
        Get the PRNG keys for the given slot IDs and positions.
        This is used to sample new tokens for the given slot IDs and positions.
        """
        # We assume that slot_ids and pos_ids are aligned
        per_pos_keys = self.prng_keys[slot_ids.array]
        return jax.vmap(jax.random.fold_in)(per_pos_keys, pos_ids.array)

    def enqueue_tokens(
        self,
        new_tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_slot_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_pos_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        num_new_tokens: int,
    ) -> "DecodeState":
        """Forward ``enqueue_tokens`` to the underlying ``TokenQueue`` and return an updated ``DecodeState``."""
        new_tqueue = self.tqueue.enqueue_tokens(new_tokens, new_slot_ids, new_pos_ids, num_new_tokens)
        return dataclasses.replace(self, tqueue=new_tqueue)

    def purge_queue_of_slot(self, slot_id: hax.NamedArray | int) -> "DecodeState":
        """Forward ``purge_queue_of_slot`` to ``TokenQueue`` and return an updated ``DecodeState``."""
        new_tqueue = self.tqueue.purge_queue_of_slot(slot_id)
        return dataclasses.replace(self, tqueue=new_tqueue)

    def pack_next_sequence(self, max_tokens: int) -> tuple["DecodeState", PackedSequence]:  # type: ignore[name-defined]
        """Forward ``pack_next_sequence`` to ``TokenQueue`` and return updated ``DecodeState`` plus the ``PackedSequence``."""
        new_tqueue, packed = self.tqueue.pack_next_sequence(max_tokens)
        return dataclasses.replace(self, tqueue=new_tqueue), packed

    @eqx.filter_jit
    def discharge_clone(
        self,
        target_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        num_targets: jnp.ndarray | int,
    ) -> "DecodeState":
        """
        Mark the given target local slot ids as no longer pending clones by setting ``clone_sources`` to INVALID
        for the first ``num_targets`` entries of ``target_slot_ids``.

        JIT-safe: uses a bounded fori_loop over ``num_targets``.
        """
        clone_map = self.clone_sources

        def body(i, cmap):
            tid = target_slot_ids["position", i].scalar()

            def do(c):
                return c.at["seq", tid].set(INVALID)

            return jax.lax.cond(is_valid(tid), do, lambda c: c, cmap)

        new_map = jax.lax.fori_loop(0, num_targets, body, clone_map)
        return dataclasses.replace(self, clone_sources=new_map)

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        """Expose remaining queue capacity from ``TokenQueue``."""
        return self.tqueue.empty_queue_space

    @property
    def max_queued_tokens(self) -> int:
        """Expose queue capacity from ``TokenQueue``."""
        return self.tqueue.max_queued_tokens

    @property
    def num_queued_tokens(self) -> jax.Array:
        """Expose current queued token count from ``TokenQueue``."""
        return self.tqueue.num_queued_tokens

    @property
    def max_seqs(self) -> int:
        """Number of sequences in the buffer."""
        return self.tokens.axis_size("seq")

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens that can be generated for each sequence, including any prefix tokens."""
        return self.tokens.axis_size("position")

    @property
    def max_stop_seq_len(self) -> int:
        """Maximum number of stop sequences for each sequence."""
        if self.stop_tokens is None:
            return 0
        return self.stop_tokens.axis_size("position")

    @eqx.filter_jit
    def assign_seq(
        self,
        local_slot_id: int,
        tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        seq_len: jnp.ndarray,
        kv_pages: ht.i32[NamedArray, "page"] | None = None,  # type: ignore[name-defined]
        seq_params: SeqDecodingParams | None = None,
    ) -> "DecodeState":
        """Assign a new sequence to the given local slot."""

        new_tokens = self.tokens.at["seq", local_slot_id, "position", 0 : tokens.axis_size("position")].set(tokens)

        if kv_pages is None:
            kv_pages = hax.full_like(self.kv_pages["seq", local_slot_id], INVALID)

        new_state = dataclasses.replace(
            self,
            kv_pages=self.kv_pages.at["seq", local_slot_id].set(kv_pages),
            tokens=new_tokens,
            # set log probs to nan for the prefix tokens
            logprobs=(
                self.logprobs.at["seq", local_slot_id, "position", :].set(jnp.nan)
                if self.logprobs is not None
                else None
            ),
            seq_lens=self.seq_lens.at["seq", local_slot_id].set(seq_len),
            finished=self.finished.at["seq", local_slot_id].set(False),
        )

        if seq_params is not None:
            new_state = dataclasses.replace(
                new_state,
                max_num_tokens=new_state.max_num_tokens.at["seq", local_slot_id].set(seq_params.max_num_tokens),
                temperature=new_state.temperature.at["seq", local_slot_id].set(seq_params.temperature),
                prng_keys=self.prng_keys.at[local_slot_id].set(seq_params.key),  # type: ignore[name-defined]
            )
            match (new_state.stop_tokens, seq_params.stop_tokens):
                case (None, None):
                    pass
                case (None, _):
                    raise ValueError("DecodeState was initialized without stop token storage")
                case (stops, None):
                    # this is fine, just fill this sequence with the pad token
                    assert stops is not None  # make mypy happy
                    new_stop_tokens = stops.at["seq", local_slot_id].set(INVALID)
                    new_state = dataclasses.replace(new_state, stop_tokens=new_stop_tokens)
                case (stops, seq_stops):
                    # too fancy, but we allow for different stop sequences per sequence etc.
                    # Probably better to do this in python outside of the jit loop
                    assert stops is not None  # make mypy happy
                    assert seq_stops is not None  # make mypy happy
                    seq_num_stops = seq_stops.axis_size("stop_seq")
                    seq_stop_len = seq_stops.axis_size("position")
                    this_row_full = hax.full_like(stops["seq", local_slot_id], INVALID)
                    this_row_full = this_row_full.at["stop_seq", 0:seq_num_stops, "position", -seq_stop_len:].set(
                        seq_stops
                    )
                    new_stops = stops.at["seq", local_slot_id].set(this_row_full)
                    new_state = dataclasses.replace(new_state, stop_tokens=new_stops)

        return new_state

    def update_tokens(
        self,
        new_tokens: ht.i32[NamedArray, " position"],  # type: ignore
        local_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore
        new_log_probs: ht.Float[NamedArray, " position"],  # type: ignore
        num_new_tokens: jnp.ndarray | int,
    ) -> "DecodeState":  # type: ignore
        """
        Update the tokens and (optional) log probabilities for the given local slot IDs,
        and enqueue these tokens onto the pending TokenQueue.
        """
        tokens = self.tokens
        logprobs = self.logprobs
        counts = self.seq_lens
        fins = self.finished

        # We'll also compute per-token absolute position ids to feed into the TokenQueue.
        pos_ids = hax.full_like(new_tokens, INVALID)
        should_purge = hax.full_like(new_tokens, True, dtype=bool)

        def body(i, state):
            sid = local_slot_ids["position", i].scalar()

            def update(state):
                tkns, lps, cnts, pids, f, should_purge = state
                pos = cnts["seq", sid].scalar()
                tkns = tkns.at["seq", sid, "position", pos].set(new_tokens["position", i])
                if lps is not None:
                    lps = lps.at["seq", sid, "position", pos].set(new_log_probs["position", i])
                cnts = cnts.at["seq", sid].add(1)
                # completion checks
                max_allowed = self.max_num_tokens["seq", sid].scalar()
                len_done = (pos + 1) >= max_allowed
                stop_done = False
                if self.stop_tokens is not None:
                    stop_len = self.stop_tokens.axis_size("position")
                    row = tkns["seq", sid].array
                    padded = jnp.concatenate([jnp.full((stop_len,), INVALID, dtype=jnp.int32), row])
                    tail = jax.lax.dynamic_slice(padded, (pos + 1,), (stop_len,))
                    stop_done = is_stop_signal(hax.named(tail, axis=("position",)), self.stop_tokens["seq", sid]).array
                f = f.at["seq", sid].set(len_done | stop_done)
                should_purge = should_purge.at["position", i].set(len_done | stop_done)

                # record position id for this token in the outgoing queue payload
                # pos here is the absolute position of this token in the sequence buffer
                pids = pids.at["position", i].set(pos)
                return tkns, lps, cnts, pids, f, should_purge

            return jax.lax.cond(is_valid(sid), update, lambda s: s, state)

        tokens, logprobs, counts, pos_ids, fins, should_purge = jax.lax.fori_loop(
            0, num_new_tokens, body, (tokens, logprobs, counts, pos_ids, fins, should_purge)
        )

        # TODO: we want to purge new_tokens of any sequences that have finished, to avoid re-processing them
        # easiest is to set the purge mask inside the loop above (based on fins)
        # jax.debug.print("should_purge: {}", should_purge)
        # jax.debug.print("before {} {} {} {}", local_slot_ids, new_tokens, pos_ids, num_new_tokens)
        local_slot_ids = purge(local_slot_ids, should_purge)
        new_tokens = purge(new_tokens, should_purge)
        pos_ids = purge(pos_ids, should_purge)
        num_new_tokens_to_queue = hax.sum((~should_purge).astype(jnp.int32)).scalar()
        # jax.debug.print("after {} {} {} {}", local_slot_ids, new_tokens, pos_ids, num_new_tokens_to_queue)

        # Enqueue tokens and their corresponding position ids into the queue
        new_tqueue = self.tqueue.enqueue_tokens(new_tokens, local_slot_ids, pos_ids, num_new_tokens_to_queue)

        return dataclasses.replace(
            self, tokens=tokens, logprobs=logprobs, seq_lens=counts, tqueue=new_tqueue, finished=fins
        )

    def is_finished(self, slot_id: jnp.ndarray) -> jnp.ndarray:
        """
        Check if the sequence or sequences with the given local ID is finished.
        A sequence is finished if it has reached its maximum number of tokens or hit its stop sequence.

        See is_stop_signal for stop sequence checking.

        Returns jnp.ndarray with the same shape as slot_id, where each entry is True if the sequence is finished.
        """

        if slot_id.ndim == 0:
            slot_id = jnp.expand_dims(slot_id, axis=0)
        return self.finished.array[slot_id]

    def debug_print(self):
        jax.debug.print(
            """
DecodeState:
num_tokens: {num_tokens}
finished: {finished}
tokens: {tokens}
stop_tokens: {stop_tokens}
kv_pages: {kv_pages}
logprobs: {logprobs}
max_num_tokens: {max_num_tokens}
""",
            num_tokens=self.seq_lens,
            finished=self.finished,
            tokens=self.tokens,
            stop_tokens=self.stop_tokens,
            kv_pages=self.kv_pages,
            logprobs=self.logprobs if self.logprobs is not None else "None",
            max_num_tokens=self.max_num_tokens,
        )

    @staticmethod
    def init(
        page_table: PageTable,
        pad_token_id: int = INVALID,
        max_stop_seqs: int = 0,
        max_stop_tokens: int = 16,
        max_queued_tokens: int = 0,
        enable_logprobs: bool = False,
    ) -> "DecodeState":
        """
        Initialize a DecodeState with empty buffers.
        """
        max_seqs = page_table.max_seqs
        pages_per_seq = page_table.pages_per_seq
        page_size = page_table.page_size
        max_seq_len = page_table.max_len_per_seq

        return DecodeState(
            kv_pages=hax.full({"seq": max_seqs, "page": pages_per_seq}, INVALID, dtype=jnp.int32),
            page_size=page_size,
            page_table=page_table,
            tokens=hax.full({"seq": max_seqs, "position": max_seq_len}, pad_token_id, dtype=jnp.int32),
            logprobs=(
                None
                if not enable_logprobs
                else hax.full({"seq": max_seqs, "position": max_seq_len}, jnp.nan, dtype=jnp.float32)
            ),
            seq_lens=hax.zeros({"seq": max_seqs}, dtype=jnp.int32),
            clone_sources=hax.full({"seq": max_seqs}, INVALID, dtype=jnp.int32),
            max_num_tokens=hax.full({"seq": max_seqs}, 0, dtype=jnp.int32),
            stop_tokens=(
                hax.full(
                    {"seq": max_seqs, "stop_seq": max_stop_seqs, "position": max_stop_tokens},
                    INVALID,
                    dtype=jnp.int32,
                )
                if max_stop_tokens > 0
                else None
            ),
            temperature=hax.ones({"seq": max_seqs}, dtype=jnp.float32),
            prng_keys=jax.vmap(jax.random.PRNGKey, axis_size=max_seqs, in_axes=None)(0),
            tqueue=TokenQueue.init(max_queued_tokens) if max_queued_tokens > 0 else TokenQueue.init(0),
            finished=hax.zeros({"seq": max_seqs}, dtype=bool),
        )


class TokenQueue(eqx.Module):
    """
    Manages a queue of tokens that are waiting to be processed. These are tokens that have been generated (or requestd for prefill)
    but have not yet been consumed by the decoding process.
    """

    # Notes:
    # - ``queued_tokens`` are stored "flat" with accompanying ``queued_slot_ids``
    queued_tokens: ht.i32[NamedArray, "position"]  # tokens queued for decoding
    queued_slot_ids: ht.i32[NamedArray, "position"]
    queued_pos_ids: ht.i32[NamedArray, "position"]  # absolute position id for each queued token
    num_queued_tokens: jax.Array

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        """How many tokens can be enqueued in the queue."""
        return self.queued_tokens.axis_size("position") - self.num_queued_tokens

    @property
    def max_queued_tokens(self) -> int:
        """Maximum number of tokens that can be buffered in the queue."""
        return self.queued_tokens.axis_size("position")

    @staticmethod
    def init(max_queued_tokens: int) -> "TokenQueue":
        """Create a ``JitScheduler`` with empty buffers."""
        return TokenQueue(
            queued_tokens=hax.full({"position": max_queued_tokens}, INVALID, dtype=jnp.int32),
            queued_slot_ids=hax.full({"position": max_queued_tokens}, INVALID, dtype=jnp.int32),
            queued_pos_ids=hax.full({"position": max_queued_tokens}, INVALID, dtype=jnp.int32),
            num_queued_tokens=jnp.array(0, dtype=jnp.int32),
        )

    def enqueue_tokens(
        self,
        new_tokens: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_slot_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        new_pos_ids: ht.i32[NamedArray, "position"],  # type: ignore[name-defined]
        num_new_tokens: int,
    ) -> "TokenQueue":
        """Append ``new_tokens`` and ``new_slot_ids`` to the queue."""
        # jax.debug.print("Enqueueing tokens {} {} {} {}", new_tokens, new_slot_ids, new_pos_ids, num_new_tokens)

        new_q_tokens = masked_set(
            self.queued_tokens,
            "position",
            self.num_queued_tokens,
            new_tokens,
            num_new_tokens,
        )
        new_q_slot_ids = masked_set(
            self.queued_slot_ids,
            "position",
            self.num_queued_tokens,
            new_slot_ids,
            num_new_tokens,
        )
        new_q_pos_ids = masked_set(
            self.queued_pos_ids,
            "position",
            self.num_queued_tokens,
            new_pos_ids,
            num_new_tokens,
        )

        return dataclasses.replace(
            self,
            queued_tokens=new_q_tokens,
            queued_slot_ids=new_q_slot_ids,
            queued_pos_ids=new_q_pos_ids,
            num_queued_tokens=self.num_queued_tokens + num_new_tokens,
        )

    def pack_next_sequence(self, max_tokens: int) -> tuple["TokenQueue", PackedSequence]:  # type: ignore[name-defined]
        """
        Dequeue up to ``max_tokens`` tokens from the queue and return them.

        Returns the updated scheduler, the tokens, slot ids and number of actual tokens that were dequeued.
        """

        pos_axis = self.queued_tokens.resolve_axis("position")
        num = jnp.minimum(self.num_queued_tokens, max_tokens)

        tokens = self.queued_tokens["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]
        slot_ids = self.queued_slot_ids["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]
        pos_ids = self.queued_pos_ids["position", hax.ds(0, max_tokens)]  # type: ignore[name-defined]

        rolled_tokens = hax.roll(self.queued_tokens, -num, "position")
        rolled_slot_ids = hax.roll(self.queued_slot_ids, -num, "position")
        rolled_pos_ids = hax.roll(self.queued_pos_ids, -num, "position")
        idx = hax.arange(pos_axis)
        mask = idx >= (pos_axis.size - num)
        filler_tokens = hax.where(mask, hax.full_like(idx, INVALID), rolled_tokens)
        filler_slot_ids = hax.where(mask, hax.full_like(idx, INVALID), rolled_slot_ids)
        filler_pos_ids = hax.where(mask, hax.full_like(idx, INVALID), rolled_pos_ids)

        new_q_tokens = filler_tokens
        new_q_slot_ids = filler_slot_ids
        new_q_pos_ids = filler_pos_ids

        new_scheduler = dataclasses.replace(
            self,
            queued_tokens=new_q_tokens,
            queued_slot_ids=new_q_slot_ids,
            queued_pos_ids=new_q_pos_ids,
            num_queued_tokens=self.num_queued_tokens - num,
        )

        # now ensure slot ids are sorted

        position_axis = slot_ids.axis_indices("position")
        assert position_axis is not None

        # TODO: add stable arg to argsort in haliax
        slot_ids_sort_order = jnp.argsort(slot_ids.array, axis=position_axis, stable=True)
        tokens = tokens["position", slot_ids_sort_order]
        slot_ids = slot_ids["position", slot_ids_sort_order]
        pos_ids = pos_ids["position", slot_ids_sort_order]

        sequence = PackedSequence(
            tokens=tokens,
            slot_ids=slot_ids,
            pos_ids=pos_ids,
            num_tokens=num,
        )

        return new_scheduler, sequence

    def purge_queue_of_slot(self, slot_id: hax.NamedArray | int) -> "TokenQueue":
        """
        Remove all tokens from the queue that belong to the given slot IDs.
        Slides remaining tokens to the front of the queue.
        """

        if isinstance(slot_id, hax.NamedArray):
            is_slot_id = hax.einsum(" -> position", self.queued_slot_ids.broadcast_axis(slot_id.axes) == slot_id)
        else:
            is_slot_id = self.queued_slot_ids == slot_id
        new_slot_ids = purge(self.queued_slot_ids, is_slot_id)
        new_tokens = purge(self.queued_tokens, is_slot_id)
        new_pos_ids = purge(self.queued_pos_ids, is_slot_id)
        new_queued = hax.sum(new_slot_ids != INVALID).scalar()

        return dataclasses.replace(
            self,
            queued_tokens=new_tokens,
            queued_slot_ids=new_slot_ids,
            queued_pos_ids=new_pos_ids,
            num_queued_tokens=new_queued,
        )

    def cleared(self) -> "TokenQueue":
        """
        Returns a new JitScheduler with all buffers cleared.
        This is useful for resetting the scheduler state.
        """
        return TokenQueue.init(
            max_queued_tokens=self.queued_tokens.axis_size("position"),
        )

    def debug_print(self, prefix: str = ""):

        def callback(self):
            print(f"{prefix}JitScheduler State:")
            print(f"{prefix}Queued Tokens: {self.queued_tokens}")
            print(f"{prefix}Queued Slot IDs: {self.queued_slot_ids}")
            print(f"{prefix}Num Queued Tokens: {self.num_queued_tokens}")

        jax.experimental.io_callback(callback, None, ordered=True, self=self)


class _DecodeOutputs(eqx.Module):
    """
    A simple queue-like buffer for outputs emitted by the decode generation loop.

    Stores the flat stream of sampled token IDs and their corresponding local slot IDs, with an
    optional logprob stream. Also carries a copy of the latest `finished` flags from `DecodeState`.

    This mirrors the behavior of `TokenQueue` but is for host-side consumption of outputs rather than
    feeding work to the device.
    """

    tokens: ht.i32[NamedArray, "position"]
    slot_ids: ht.i32[NamedArray, "position"]
    logprobs: ht.Float[NamedArray, "position"] | None
    num_tokens: jax.Array
    finished: ht.bool_[NamedArray, "seq"]

    @property
    def max_queued_tokens(self) -> int:
        return self.tokens.axis_size("position")

    @property
    def empty_queue_space(self) -> jnp.ndarray:
        return self.tokens.axis_size("position") - self.num_tokens

    @staticmethod
    def init(max_tokens: int, max_seqs: int, with_logprobs: bool = True) -> "_DecodeOutputs":
        return _DecodeOutputs(
            tokens=hax.full({"position": max_tokens}, INVALID, dtype=jnp.int32),
            slot_ids=hax.full({"position": max_tokens}, INVALID, dtype=jnp.int32),
            logprobs=(hax.full({"position": max_tokens}, jnp.nan, dtype=jnp.float32) if with_logprobs else None),
            num_tokens=jnp.array(0, dtype=jnp.int32),
            finished=hax.zeros({"seq": max_seqs}, dtype=bool),
        )

    def append(
        self,
        new_tokens: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        new_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        new_logprobs: ht.Float[NamedArray, " position"],  # type: ignore[name-defined]
        num_new_tokens: int,
        finished_snapshot: ht.bool_[NamedArray, "seq"],  # type: ignore[name-defined]
    ) -> "_DecodeOutputs":
        """Append a batch of outputs and update the finished flags snapshot."""

        new_tok_buf = masked_set(self.tokens, "position", self.num_tokens, new_tokens, num_new_tokens)
        new_sid_buf = masked_set(self.slot_ids, "position", self.num_tokens, new_slot_ids, num_new_tokens)
        if self.logprobs is not None:
            new_lp_buf = masked_set(self.logprobs, "position", self.num_tokens, new_logprobs, num_new_tokens)
        else:
            new_lp_buf = None
        # Keep finished flags monotonic (once finished, always finished)
        new_finished = self.finished | finished_snapshot
        return dataclasses.replace(
            self,
            tokens=new_tok_buf,
            slot_ids=new_sid_buf,
            logprobs=new_lp_buf,
            num_tokens=self.num_tokens + num_new_tokens,
            finished=new_finished,
        )
