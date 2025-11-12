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

from levanter.inference.page_table import PageTable, PageBatchInfo
from levanter.inference.utils import (
    INVALID,
    get_unique_in_order,
    is_stop_signal,
    is_valid,
    masked_set,
    purge,
    is_invalid,
)


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


class SequenceTable(eqx.Module):
    """Compact view over per-sequence metadata tracked by :class:`DecodeState`."""

    seq_lens: ht.i32[NamedArray, "seq"]
    clone_sources: ht.i32[NamedArray, "seq"]
    kv_pages: ht.i32[NamedArray, "seq page"]
    page_indices: ht.i32[NamedArray, "seq page"]
    used_mask: ht.bool_[NamedArray, "seq"]
    page_size: int = eqx.field(static=True)

    @staticmethod
    def init(max_seqs: int, pages_per_seq: int, page_size: int) -> "SequenceTable":
        """Create an empty sequence table."""
        return SequenceTable(
            seq_lens=hax.zeros({"seq": max_seqs}, dtype=jnp.int32),
            clone_sources=hax.full({"seq": max_seqs}, INVALID, dtype=jnp.int32),
            kv_pages=hax.full({"seq": max_seqs, "page": pages_per_seq}, INVALID, dtype=jnp.int32),
            page_indices=hax.full({"seq": max_seqs, "page": pages_per_seq}, INVALID, dtype=jnp.int32),
            used_mask=hax.full({"seq": max_seqs}, False, dtype=bool),
            page_size=page_size,
        )

    @property
    def max_seqs(self) -> int:
        return self.seq_lens.axis_size("seq")

    @property
    def pages_per_seq(self) -> int:
        return self.page_indices.axis_size("page")

    def reserve_slot(self, slot_id: int | jnp.ndarray | None = None) -> tuple["SequenceTable", int]:
        """Reserve a free sequence slot and return its ID.

        DONATES self

        If ``seq_id`` is provided and valid (0 <= seq_id < max_seqs), it is used directly and no search is done.
        If ``seq_id`` is None or invalid (<0 or >= max_seqs), a free slot is searched for and assigned.

        If no free slots are available, returns INVALID (-1) as the seq_id and does not modify the table.

        Args:
            seq_id: Optional specific sequence ID to assign. If None or invalid, a free slot is searched for.

        Returns:
            A tuple of (new PageTable with updated metadata, assigned sequence ID or INVALID if none available).
        """
        if isinstance(slot_id, int):
            slot_id = jnp.array(slot_id, dtype=jnp.int32)
        return self._reserve_slot(slot_id)

    @eqx.filter_jit(donate="all")
    def _reserve_slot(self, slot_id: jnp.ndarray | None = None) -> tuple["SequenceTable", int]:
        if slot_id is None:
            slot_id = INVALID

        def validate(candidate):
            return hax.where(self.used_mask["seq", candidate], INVALID, candidate)

        def find_free(_: jnp.ndarray):
            free_flags = ~self.used_mask
            maybe = hax.argmax(free_flags, "seq").scalar()
            available = (~self.used_mask["seq", maybe]).scalar()
            return hax.where(available, maybe, INVALID)

        slot = jax.lax.cond(is_valid(slot_id), validate, find_free, slot_id)

        def do_assign(table):
            seq_lens = table.seq_lens.at["seq", slot].set(0)
            clone_sources = table.clone_sources.at["seq", slot].set(INVALID)
            kv_pages = table.kv_pages.at["seq", slot].set(hax.full_like(table.kv_pages["seq", slot], INVALID))
            page_indices = table.page_indices.at["seq", slot].set(
                hax.full_like(table.page_indices["seq", slot], INVALID)
            )
            used_mask = table.used_mask.at["seq", slot].set(True)
            return dataclasses.replace(
                table,
                seq_lens=seq_lens,
                clone_sources=clone_sources,
                kv_pages=kv_pages,
                page_indices=page_indices,
                used_mask=used_mask,
            )

        table = jax.lax.cond(is_valid(slot), do_assign, lambda t: t, self)
        return table, slot

    def assign_slot(
        self,
        slot_id: int,
        *,
        seq_len: jnp.ndarray,
        kv_pages: ht.i32[NamedArray, "page"],  # type: ignore[name-defined]
        page_indices: ht.i32[NamedArray, "page"] | None = None,  # type: ignore[name-defined]
        clone_source: jnp.ndarray | int = INVALID,
    ) -> "SequenceTable":
        seq_lens = self.seq_lens.at["seq", slot_id].set(seq_len)
        clone_sources = self.clone_sources.at["seq", slot_id].set(clone_source)
        kv_pages_arr = self.kv_pages.at["seq", slot_id].set(kv_pages)
        indices_row = (
            page_indices if page_indices is not None else hax.full_like(self.page_indices["seq", slot_id], INVALID)
        )
        page_indices_arr = self.page_indices.at["seq", slot_id].set(indices_row)
        used_mask = self.used_mask.at["seq", slot_id].set(True)
        return dataclasses.replace(
            self,
            seq_lens=seq_lens,
            clone_sources=clone_sources,
            kv_pages=kv_pages_arr,
            page_indices=page_indices_arr,
            used_mask=used_mask,
        )

    def set_clone_source(self, slot_id: int, clone_source: jnp.ndarray | int) -> "SequenceTable":
        clone_sources = self.clone_sources.at["seq", slot_id].set(clone_source)
        return dataclasses.replace(self, clone_sources=clone_sources)

    def clear_slots(self, mask: ht.bool_[NamedArray, "seq"]) -> "SequenceTable":  # type: ignore[name-defined]
        new_seq_lens = hax.where(mask, INVALID, self.seq_lens)
        new_clone_sources = hax.where(mask, INVALID, self.clone_sources)
        new_kv_pages = hax.where(mask, INVALID, self.kv_pages)
        new_page_indices = hax.where(mask, INVALID, self.page_indices)
        new_used_mask = hax.where(mask, False, self.used_mask)
        return SequenceTable(
            new_seq_lens,
            new_clone_sources,
            new_kv_pages,
            new_page_indices,
            new_used_mask,
            self.page_size,
        )

    def release_slot(self, slot_id: int) -> "SequenceTable":
        mask = hax.zeros_like(self.used_mask)
        mask = mask.at["seq", slot_id].set(True)
        return self.clear_slots(mask)

    @eqx.filter_jit
    def allocate_for_seq(
        self,
        page_table: "PageTable",
        token_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        token_pos_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
    ) -> tuple["SequenceTable", "PageTable", "PageBatchInfo"]:
        """
        Allocate pages from PageTable for new sequences and update ``seq_lens``.

        **ASSUMES** that the ``token_slot_ids`` are already grouped by sequence ID, i.e. all tokens for a given sequence
        are contiguous in the input. The order of sequences in the input does not matter.
        """
        token_slot_ids = hax.where(token_slot_ids < 0, self.max_seqs, token_slot_ids)
        # CAREFUL: we don't assume slot_ids are sorted, just contiguous. segment_sum is our friend
        # NB: segment_sum assumes that segment ids are in the range [0, num_segments)
        # and returns an array of that length
        # essentially this means segment_sum et al require dense segment ids so we have to denseify the slot ids first
        unique_ids, dense_ids = get_unique_in_order(
            token_slot_ids.array,
            size=self.max_seqs + 1,  # +1 for INVALID
            fill_value=INVALID,
        )
        segment_lengths = jax.ops.segment_sum(
            data=jnp.ones_like(token_slot_ids.array, dtype=jnp.int32),
            segment_ids=dense_ids,
            num_segments=self.max_seqs,
        )
        # now we need to know the segment_ids
        segment_ids = jax.ops.segment_max(
            data=token_slot_ids.array,
            segment_ids=dense_ids,
            num_segments=self.max_seqs,
        )
        # and then the maximum position within each segment
        max_pos_per_seq = jax.ops.segment_max(
            data=token_pos_ids.array,
            segment_ids=dense_ids,
            num_segments=self.max_seqs,
        )

        updated_seqs = hax.named(segment_ids, axis="seq")
        new_counts = hax.named(segment_lengths, axis="seq")
        new_max_pos = hax.named(max_pos_per_seq, axis="seq")
        # jax.debug.print("segment_lengths={sl} segment_ids={sid}, tsi={tsi}", sl=new_counts, sid=updated_seqs, tsi=token_slot_ids)

        valid_updated = is_valid(updated_seqs) & (updated_seqs < self.max_seqs)
        safe_updated = hax.where(valid_updated, updated_seqs, 0)

        masked_max_pos = hax.where(valid_updated, new_max_pos, -1)

        cu_new_counts = hax.concatenate(
            "seq",
            [
                hax.zeros({"seq": 1}, dtype=jnp.int32),
                hax.cumsum(new_counts, "seq", dtype=jnp.int32),
            ],
        )

        new_counts = hax.where(valid_updated, new_counts, 0)

        current_lens = self.seq_lens
        active_mask_for_updated = hax.where(valid_updated, self.used_mask["seq", safe_updated], False)

        num_updated_seqs = hax.sum(valid_updated).scalar().astype(jnp.int32)

        masked_seq_len = (masked_max_pos + 1) * active_mask_for_updated.astype(new_counts.dtype)
        new_lens = current_lens.at["seq", safe_updated].max(masked_seq_len, mode="drop")
        # jax.debug.print("masked_seq_len={msl} new_lens={nl} safe_updated={su}", msl=masked_seq_len, nl=new_lens, su=safe_updated)

        new_num_pages_needed = (new_lens + self.page_size - 1) // self.page_size
        # Count how many pages are actually allocated (valid page_indices), not based on seq_lens
        # This handles the case where seq_len is set but pages haven't been allocated yet
        num_allocated_pages = hax.sum(is_valid(self.page_indices), axis="page")
        old_num_pages_needed = num_allocated_pages
        old_num_pages_needed = hax.where(valid_updated, old_num_pages_needed["seq", safe_updated], 0)
        new_num_pages_needed = hax.where(valid_updated, new_num_pages_needed["seq", safe_updated], 0)
        # jax.debug.print("new_num_pages_needed={nnp} old_num_pages_needed={onp}", nnp=new_num_pages_needed, onp=old_num_pages_needed)
        # jax.debug.print(
        #     "nnp per seq: seq0={s0} seq1={s1} seq2={s2}",
        #     s0=new_num_pages_needed["seq", 0],
        #     s1=new_num_pages_needed["seq", 1],
        #     s2=new_num_pages_needed["seq", 2],
        # )

        page_indices = self.page_indices
        page_ref_counts = page_table.page_ref_counts

        # NB seq_offset refers to the offset into updated_seqs, not the actual seq_id
        def _alloc_pages_for_seq(seq_offset, carry):
            indices, ref_counts = carry
            num_needed = new_num_pages_needed["seq", seq_offset].scalar()
            old_needed = old_num_pages_needed["seq", seq_offset].scalar()
            seq_id = safe_updated["seq", seq_offset].scalar()
            # jax.debug.print(
            #     "alloc seq {seq_id} lookup nnp={nnp}", seq_id=seq_id, nnp=new_num_pages_needed["seq", seq_id]
            # )
            # jax.debug.print("alloc seq {seq_id} old_needed={old} num_needed={new}", seq_id=seq_id, old=old_needed, new=num_needed)

            def body(page_idx, state):
                indices, ref_counts = state
                has_free = hax.any(ref_counts == 0).scalar()

                ref_counts = eqx.error_if(ref_counts, ~has_free, "Out of free pages during allocation")

                def do_alloc(state):
                    indices, ref_counts = state
                    # choose a page with the smallest ref count; when has_free, argmin will pick a zero-ref page
                    free_page_idx = hax.argmin(ref_counts, "page")
                    ref_counts = ref_counts.at["page", free_page_idx].add(1)
                    indices = indices.at["seq", seq_id, "page", page_idx].set(free_page_idx)
                    return indices, ref_counts

                def no_alloc(state):
                    # No-op; leave index INVALID so downstream gets INVALID destinations
                    state = eqx.error_if(state, jnp.zeros(()) < 4, "INVALID!")
                    return state

                return jax.lax.cond(has_free, do_alloc, no_alloc, (indices, ref_counts))

            return jax.lax.fori_loop(old_needed, num_needed, body, (indices, ref_counts))

        def outer(i, carry):
            page_indices, page_ref_counts = carry

            def do_alloc(carry):
                return _alloc_pages_for_seq(i, carry)

            seq_id = updated_seqs["seq", i].scalar()
            cond = is_valid(seq_id)
            # jax.debug.print("outer iter {i}: seq_id={seq} is_valid={iv}", i=i, seq=seq_id, iv=is_valid(seq_id))

            page_indices, page_ref_counts = jax.lax.cond(cond, do_alloc, lambda c: c, (page_indices, page_ref_counts))
            return page_indices, page_ref_counts

        page_indices, page_ref_counts = jax.lax.fori_loop(0, num_updated_seqs, outer, (page_indices, page_ref_counts))

        batch_info = self._create_batch_info(
            updated_seqs,
            page_indices,
            cu_new_counts,
            new_lens,
            token_slot_ids,
            token_pos_ids,
        )

        kv_pages = self.kv_pages.at["seq", safe_updated].set(page_indices["seq", safe_updated])
        new_sequences = dataclasses.replace(self, seq_lens=new_lens, page_indices=page_indices, kv_pages=kv_pages)
        new_page_table = dataclasses.replace(page_table, page_ref_counts=page_ref_counts)

        return new_sequences, new_page_table, batch_info

    def _create_batch_info(self, updated_seqs, page_indices, cu_new_counts, new_lens, token_slot_ids, token_pos_ids):
        mask = is_valid(updated_seqs) & (updated_seqs < self.max_seqs)
        safe_updated = hax.where(mask, updated_seqs, 0)

        gathered_page_indices = page_indices["seq", safe_updated]
        batch_page_indices = hax.where(mask, gathered_page_indices, INVALID)

        seq_lens = new_lens["seq", safe_updated]
        seq_lens = hax.where(mask, seq_lens, INVALID)

        token_dests = hax.full(token_slot_ids.shape, INVALID, dtype=jnp.int32)

        def token_body(i, token_dests):
            seq_id = token_slot_ids["position", i].scalar()
            pos_id = token_pos_ids["position", i].scalar()

            def assign(dest):
                page_idx = pos_id // self.page_size
                page_offset = pos_id % self.page_size
                page = page_indices["seq", seq_id, "page", page_idx]
                dest_value = hax.where(is_invalid(page), INVALID, page * self.page_size + page_offset)
                return dest.at["position", i].set(dest_value)

            seq_valid = (seq_id >= 0) & (seq_id < self.max_seqs) & is_valid(seq_id)
            pos_valid = is_valid(pos_id)
            return jax.lax.cond(seq_valid & pos_valid, assign, lambda d: d, token_dests)

        token_dests = jax.lax.fori_loop(0, token_slot_ids.axis_size("position"), token_body, token_dests)

        batch_info = PageBatchInfo(
            slot_ids=updated_seqs,
            page_indices=batch_page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_new_counts,
            num_seqs=hax.sum(mask).scalar().astype(jnp.int32),
            new_token_dests=token_dests,
            page_size=self.page_size,
        )

        return batch_info

    @eqx.filter_jit(donate="all")
    def free_pages(self, page_table: "PageTable", seq_id: int) -> tuple["SequenceTable", "PageTable"]:
        seq_pages = self.page_indices["seq", seq_id]
        is_valid_page = is_valid(seq_pages)

        def body(i, ref_counts):
            def dec(rc):
                page = seq_pages["page", i].scalar()
                return rc.at["page", page].add(-1)

            return jax.lax.cond(is_valid_page["page", i].scalar(), dec, lambda x: x, ref_counts)

        new_ref_counts = jax.lax.fori_loop(0, seq_pages.axis_size("page"), body, page_table.page_ref_counts)
        new_ref_counts = hax.maximum(new_ref_counts, hax.zeros_like(new_ref_counts))

        new_seq_lens = self.seq_lens.at["seq", seq_id].set(0)
        new_clone_sources = self.clone_sources.at["seq", seq_id].set(INVALID)
        new_kv_pages = self.kv_pages.at["seq", seq_id].set(hax.full_like(self.kv_pages["seq", seq_id], INVALID))
        new_page_indices = self.page_indices.at["seq", seq_id].set(
            hax.full_like(self.page_indices["seq", seq_id], INVALID)
        )
        new_used_mask = self.used_mask.at["seq", seq_id].set(False)

        new_sequences = SequenceTable(
            new_seq_lens,
            new_clone_sources,
            new_kv_pages,
            new_page_indices,
            new_used_mask,
            self.page_size,
        )
        new_page_table = dataclasses.replace(page_table, page_ref_counts=new_ref_counts)
        return new_sequences, new_page_table

    @eqx.filter_jit(donate="all")
    def free_pages_for_finished(
        self,
        page_table: "PageTable",
        finished_mask: jnp.ndarray,
    ) -> tuple["SequenceTable", "PageTable"]:
        assert finished_mask.ndim == 1

        def dec_refcounts_for_seq(pages_row, ref_counts):
            valid = is_valid(pages_row)

            def body(i, rc):
                def dec(rc):
                    page = pages_row["page", i].scalar()
                    return rc.at["page", page].add(-1)

                return jax.lax.cond(valid["page", i].scalar(), dec, lambda x: x, rc)

            updated = jax.lax.fori_loop(0, pages_row.axis_size("page"), body, ref_counts)
            return hax.maximum(updated, hax.zeros_like(updated))

        def body(i, state):
            ref_counts, sequences = state

            def do(state):
                ref_counts, sequences = state
                pages_row = sequences.page_indices["seq", i]
                ref_counts = dec_refcounts_for_seq(pages_row, ref_counts)
                sequences = sequences.release_slot(i)
                return ref_counts, sequences

            return jax.lax.cond(finished_mask[i], do, lambda s: s, (ref_counts, sequences))

        ref_counts, sequences = jax.lax.fori_loop(0, self.max_seqs, body, (page_table.page_ref_counts, self))
        new_page_table = dataclasses.replace(page_table, page_ref_counts=ref_counts)
        return sequences, new_page_table

    @eqx.filter_jit(donate="all")
    def clone_pages_from(
        self,
        page_table: "PageTable",
        src_seq_id: int,
        dst_seq_id: int,
    ) -> tuple["SequenceTable", "PageTable"]:
        # jax.debug.print("clone {} -> {}", src_seq_id, dst_seq_id)
        src_pages = self.page_indices["seq", src_seq_id]
        src_len = self.seq_lens["seq", src_seq_id].scalar()
        size = self.page_size

        used_pages = (src_len + size - 1) // size
        last_idx = hax.maximum(used_pages - 1, 0)
        is_boundary = (src_len % size) == 0

        page_indices = self.page_indices.at["seq", dst_seq_id].set(src_pages)

        def inc_shared(ref_counts):
            def body(i, rc):
                page = src_pages["page", i]

                def inc(rc):
                    return rc.at["page", page].add(1)

                return jax.lax.cond(is_valid(page).scalar(), inc, lambda x: x, rc)

            limit = used_pages - jnp.where(is_boundary, 0, 1)
            return jax.lax.fori_loop(0, limit, body, ref_counts)

        ref_counts = inc_shared(page_table.page_ref_counts)

        def handle_partial(state):
            ref_counts, indices = state
            has_free = hax.any(ref_counts == 0).scalar()
            ref_counts = eqx.error_if(ref_counts, ~has_free, "Out of free pages during clone_pages_from")
            free_idx = hax.argmax(ref_counts == 0, "page")
            indices = indices.at["seq", dst_seq_id, "page", last_idx].set(free_idx)
            ref_counts = ref_counts.at["page", free_idx].add(1)
            return ref_counts, indices

        ref_counts, page_indices = jax.lax.cond(
            is_boundary,
            lambda s: s,
            handle_partial,
            (ref_counts, page_indices),
        )

        seq_lens = self.seq_lens.at["seq", dst_seq_id].set(src_len)
        used_mask = self.used_mask.at["seq", dst_seq_id].set(True)

        kv_pages = self.kv_pages.at["seq", dst_seq_id].set(page_indices["seq", dst_seq_id])
        sequences = dataclasses.replace(
            self,
            seq_lens=seq_lens,
            page_indices=page_indices,
            used_mask=used_mask,
            kv_pages=kv_pages,
        )
        page_table = dataclasses.replace(page_table, page_ref_counts=ref_counts)
        return sequences, page_table

    def bump_seq_len_to_next_page(self, seq_id: int) -> "SequenceTable":
        cur = self.seq_lens["seq", seq_id]
        size = jnp.array(self.page_size, dtype=jnp.int32)
        next_page = ((cur + size - 1) // size) * size
        new_seq_lens = self.seq_lens.at["seq", seq_id].set(next_page)
        return dataclasses.replace(self, seq_lens=new_seq_lens)


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
    sequences: SequenceTable
    """Aggregated per-sequence state such as lengths, clone metadata, and KV page assignments."""
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

    tqueue: "TokenQueue"
    """token queue for pending decode work"""

    # Cached finished flags per sequence (updated when tokens are enqueued)
    finished: ht.bool_[NamedArray, "seq"]

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

        sequence_table = SequenceTable.init(max_seqs, pages_per_seq, page_size)

        return DecodeState(
            sequences=sequence_table,
            page_size=page_size,
            page_table=page_table,
            tokens=hax.full({"seq": max_seqs, "position": max_seq_len}, pad_token_id, dtype=jnp.int32),
            logprobs=(
                None
                if not enable_logprobs
                else hax.full({"seq": max_seqs, "position": max_seq_len}, jnp.nan, dtype=jnp.float32)
            ),
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
            tqueue=TokenQueue.init(max_queued_tokens),
            finished=hax.zeros({"seq": max_seqs}, dtype=bool),
        )

    @property
    def seq_lens(self) -> ht.i32[NamedArray, "seq"]:  # type: ignore[name-defined]
        """Current logical length for each active sequence."""
        return self.sequences.seq_lens

    @property
    def clone_sources(self) -> ht.i32[NamedArray, "seq"]:  # type: ignore[name-defined]
        """Mapping from clone targets to their parent sequences."""
        return self.sequences.clone_sources

    @property
    def kv_pages(self) -> ht.i32[NamedArray, "seq page"]:  # type: ignore[name-defined]
        """KV page assignments per sequence."""
        return self.sequences.kv_pages

    @eqx.filter_jit(donate="all")
    def invalidate_finished(self) -> "DecodeState":
        """Invalidate metadata for sequences marked finished by ``finished_mask``.

        - Sets ``seq_lens`` to INVALID for finished slots
        - Resets ``clone_sources`` to INVALID
        - Clears ``kv_pages`` rows for finished slots to INVALID
        """
        mask = self.finished
        finished = hax.zeros_like(self.finished)
        new_sequences = self.sequences.clear_slots(mask)
        return dataclasses.replace(self, sequences=new_sequences, finished=finished)

    def prng_key_for(self, slot_id: int, pos_id: int) -> jaxtyping.PRNGKeyArray:
        """
        Get the PRNG key for the given slot ID and position.
        This is used to sample new tokens for the given slot ID and position.
        """
        per_pos_key = self.prng_keys[ensure_scalar(slot_id)]
        return jax.random.fold_in(per_pos_key, ensure_scalar(pos_id))

    def reserve_slot(self, slot_id: int | jnp.ndarray | None = None) -> tuple["DecodeState", int]:
        sequences, slot = self.sequences.reserve_slot(slot_id)
        return dataclasses.replace(self, sequences=sequences), slot

    def release_slot(self, slot_id: int) -> "DecodeState":
        sequences = self.sequences.release_slot(slot_id)
        return dataclasses.replace(self, sequences=sequences)

    def allocate_for_seq(
        self,
        token_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        token_pos_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
    ) -> tuple["DecodeState", PageBatchInfo]:
        sequences, page_table, batch_info = self.sequences.allocate_for_seq(
            self.page_table, token_slot_ids, token_pos_ids
        )
        return dataclasses.replace(self, sequences=sequences, page_table=page_table), batch_info

    def free_pages(self, seq_id: int) -> "DecodeState":
        sequences, page_table = self.sequences.free_pages(self.page_table, seq_id)
        return dataclasses.replace(self, sequences=sequences, page_table=page_table)

    def free_pages_for_finished(self, finished_mask: jnp.ndarray) -> "DecodeState":
        sequences, page_table = self.sequences.free_pages_for_finished(self.page_table, finished_mask)
        return dataclasses.replace(self, sequences=sequences, page_table=page_table)

    def bump_seq_len_to_next_page(self, seq_id: int) -> "DecodeState":
        sequences = self.sequences.bump_seq_len_to_next_page(seq_id)
        return dataclasses.replace(self, sequences=sequences)

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
        clone_map = self.sequences.clone_sources

        def body(i, cmap):
            tid = target_slot_ids["position", i].scalar()

            def do(c):
                return c.at["seq", tid].set(INVALID)

            return jax.lax.cond(is_valid(tid), do, lambda c: c, cmap)

        new_map = jax.lax.fori_loop(0, num_targets, body, clone_map)
        table = self.sequences
        new_sequences = dataclasses.replace(table, clone_sources=new_map)
        return dataclasses.replace(self, sequences=new_sequences)

    def clone_pages_from(self, src, dest) -> "DecodeState":
        """
        Clone kv_pages from src slot to dest slot.
        """
        sequences, page_table = self.sequences.clone_pages_from(self.page_table, src, dest)
        return dataclasses.replace(self, sequences=sequences, page_table=page_table)

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
        seq_len: jnp.ndarray | int = 0,
        kv_pages: ht.i32[NamedArray, "page"] | None = None,  # type: ignore[name-defined]
        page_indices: ht.i32[NamedArray, "page"] | None = None,  # type: ignore[name-defined]
        seq_params: SeqDecodingParams | None = None,
    ) -> "DecodeState":
        """Assign a new sequence to the given local slot."""

        new_tokens = self.tokens.at["seq", local_slot_id, "position", 0 : tokens.axis_size("position")].set(tokens)

        sequences = self.sequences
        if kv_pages is None:
            kv_pages = hax.full_like(sequences.kv_pages["seq", local_slot_id], INVALID)

        new_sequences = sequences.assign_slot(
            local_slot_id,
            seq_len=seq_len,
            kv_pages=kv_pages,
            page_indices=page_indices,
        )

        new_state = dataclasses.replace(
            self,
            sequences=new_sequences,
            tokens=new_tokens,
            # set log probs to nan for the prefix tokens
            logprobs=(
                self.logprobs.at["seq", local_slot_id, "position", :].set(jnp.nan)
                if self.logprobs is not None
                else None
            ),
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
        sequences = self.sequences
        counts = sequences.seq_lens
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

        # Enqueue tokens and their corresponding position ids into the queue
        new_tqueue = self.tqueue.enqueue_tokens(new_tokens, local_slot_ids, pos_ids, num_new_tokens_to_queue)

        new_sequences = dataclasses.replace(sequences, seq_lens=counts)

        return dataclasses.replace(
            self, tokens=tokens, logprobs=logprobs, sequences=new_sequences, tqueue=new_tqueue, finished=fins
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
            num_tokens=self.sequences.seq_lens,
            finished=self.finished,
            tokens=self.tokens,
            stop_tokens=self.stop_tokens,
            kv_pages=self.sequences.kv_pages,
            logprobs=self.logprobs if self.logprobs is not None else "None",
            max_num_tokens=self.max_num_tokens,
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
        assert (
            new_tokens.axis_size("position") <= self.max_queued_tokens
        ), f"Too many new tokens to enqueue {new_tokens.axis_size('position')} > {self.max_queued_tokens}"

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
