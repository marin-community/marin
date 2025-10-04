# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.haxtyping as ht
from haliax import NamedArray

__all__ = ["PageTable", "PageBatchInfo"]

from levanter.inference.utils import INVALID, is_invalid, is_valid, get_unique_in_order


def _relative_positions(seg_ids: jnp.ndarray):
    idx = jnp.arange(seg_ids.shape[0])
    is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
    start_idx = idx * is_start.astype(idx.dtype)
    seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
    return idx - seg_start  # 0,1,2,… inside each segment


class PageTable(eqx.Module):
    """
    Tracks which pages are allocated to which sequences.

    This page table data structure is pretty dumb. It doesn't support automatic prefix caching or any
    of that fanciness.

    """

    page_indices: NamedArray  # i32[Seq, PagePerSeq]
    page_ref_counts: NamedArray  # i32[Page]
    seq_lens: NamedArray  # i32[Seq]
    used_mask: NamedArray  # bool[Seq] — True if slot is in use
    page_size: int = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_pages(self) -> int:
        return self.page_ref_counts.axis_size("page")

    @property
    def pages_per_seq(self) -> int:
        return self.page_indices.axis_size("page")

    @property
    def max_seqs(self) -> int:
        return self.page_indices.axis_size("seq")

    @property
    def max_len_per_seq(self) -> int:
        return self.page_size * self.pages_per_seq

    @property
    def current_num_seqs(self) -> int:
        # Count currently used sequence slots
        return hax.sum(self.used_mask).scalar()

    @property
    def max_Seq(self) -> hax.Axis:
        """Axis representing the maximum number of sequences."""
        return hax.Axis("seq", self.max_seqs)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def init(max_pages: int, max_seqs: int, page_size: int, max_pages_per_seq: int) -> "PageTable":
        page_indices = hax.full({"seq": max_seqs, "page": max_pages_per_seq}, INVALID, dtype=jnp.int32)
        page_ref_counts = hax.full({"page": max_pages}, 0, dtype=jnp.int32)
        # With a separate used/free mask, keep seq_lens as lengths and default to 0
        seq_lens = hax.full({"seq": max_seqs}, 0, dtype=jnp.int32)
        used_mask = hax.full({"seq": max_seqs}, False)
        return PageTable(page_indices, page_ref_counts, seq_lens, used_mask, page_size)

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------

    def assign_seq_id_to_seq(self, seq_id: int | jnp.ndarray | None = None) -> tuple["PageTable", int]:
        """Assign a free sequence slot and return its ID.

        DONATES self

        If ``seq_id`` is provided and valid (0 <= seq_id < max_seqs), it is used directly and no search is done.
        If ``seq_id`` is None or invalid (<0 or >= max_seqs), a free slot is searched for and assigned.

        If no free slots are available, returns INVALID (-1) as the seq_id and does not modify the table.

        Args:
            seq_id: Optional specific sequence ID to assign. If None or invalid, a free slot is searched for.

        Returns:
            A tuple of (new PageTable with updated metadata, assigned sequence ID or INVALID if none available).
        """
        # JIT compile the inner function to avoid recompiling the whole function on different seq_id inputs
        if isinstance(seq_id, int):
            seq_id = jnp.array(seq_id, dtype=jnp.int32)
        return self._assign_seq_id_to_seq_jit(seq_id)

    @eqx.filter_jit(donate="all")
    def _assign_seq_id_to_seq_jit(self, seq_id: jnp.ndarray | None = None) -> tuple["PageTable", int]:
        # Find a free slot using the used_mask
        if seq_id is None:
            seq_id = INVALID

        def validate(seq_id):
            return hax.where(self.used_mask["seq", seq_id], INVALID, seq_id)

        def _find_free(seq_id):
            free_flags = ~self.used_mask
            maybe_seq_id = hax.argmax(free_flags, "seq").scalar()
            available = (~self.used_mask["seq", maybe_seq_id]).scalar()
            maybe_seq_id = hax.where(available, maybe_seq_id, INVALID)

            return maybe_seq_id

        seq_id = jax.lax.cond(
            is_valid(seq_id),
            validate,
            _find_free,
            seq_id,
        )

        def do_assign(self_):
            new_seq_lens = self_.seq_lens.at["seq", seq_id].set(0)
            new_used = self_.used_mask.at["seq", seq_id].set(True)
            return dataclasses.replace(self_, seq_lens=new_seq_lens, used_mask=new_used)

        def no_op(self_):
            return self_

        new_self = jax.lax.cond(is_valid(seq_id), do_assign, no_op, self)
        return new_self, seq_id

    @eqx.filter_jit
    def allocate_for_seq(
        self,
        token_slot_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
        token_pos_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
    ) -> tuple["PageTable", "PageBatchInfo"]:
        """
        Allocate pages for new sequences and update ``seq_lens``.

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

        cu_new_counts = hax.concatenate(
            "seq",
            [
                hax.zeros({"seq": 1}, dtype=jnp.int32),
                hax.cumsum(new_counts, "seq", dtype=jnp.int32),
            ],
        )

        new_counts = hax.where(updated_seqs >= self.max_seqs, 0, new_counts)

        # Only update lengths for active sequences
        current_lens = self.seq_lens
        # Mask counts for sequences not currently used
        active_mask_for_updated = self.used_mask["seq", updated_seqs]
        active_mask_for_updated = active_mask_for_updated & is_valid(updated_seqs)

        masked_seq_len = (new_max_pos + 1) * active_mask_for_updated.astype(new_counts.dtype)
        new_lens = current_lens.at["seq", updated_seqs].max(masked_seq_len, mode="drop")

        new_num_pages_needed = (new_lens + self.page_size - 1) // self.page_size
        old_num_pages_needed = (self.seq_lens + self.page_size - 1) // self.page_size

        def _alloc_pages_for_seq(seq_id, carry):
            page_indices, page_ref_counts = carry
            num_needed = new_num_pages_needed["seq", seq_id].scalar()
            old_needed = old_num_pages_needed["seq", seq_id].scalar()

            def body(page_idx, state):
                page_indices, page_ref_counts = state
                has_free = hax.any(page_ref_counts == 0).scalar()

                # Emit a runtime error if we are out of free pages
                page_ref_counts = eqx.error_if(
                    page_ref_counts,
                    ~has_free,
                    "Out of free pages during allocation",
                )

                def do_alloc(state):
                    pic, prc = state
                    # choose a page with the smallest ref count; when has_free, argmin will pick a zero-ref page
                    free_page_idx = hax.argmin(prc, "page")
                    prc = prc.at["page", free_page_idx].add(1)
                    pic = pic.at["seq", seq_id, "page", page_idx].set(free_page_idx)
                    return pic, prc

                def no_alloc(state):
                    # No-op; leave index INVALID so downstream gets INVALID destinations
                    return state

                return jax.lax.cond(has_free, do_alloc, no_alloc, (page_indices, page_ref_counts))

            new_page_indices, new_page_ref_counts = jax.lax.fori_loop(
                old_needed, num_needed, body, (page_indices, page_ref_counts)
            )
            return new_page_indices, new_page_ref_counts

        def outer(i, carry):
            page_indices, page_ref_counts = carry
            seq_id = updated_seqs["seq", i].scalar()

            def do_alloc(carry):
                return _alloc_pages_for_seq(seq_id, carry)

            # cond = jnp.logical_and(seq_id >= 0, seq_id < self.max_seqs)
            cond = is_valid(seq_id)
            page_indices, page_ref_counts = jax.lax.cond(cond, do_alloc, lambda c: c, (page_indices, page_ref_counts))
            return page_indices, page_ref_counts

        page_indices, page_ref_counts = jax.lax.fori_loop(
            0, updated_seqs.axis_size("seq"), outer, ((self.page_indices), (self.page_ref_counts))
        )

        new_table = dataclasses.replace(
            self,
            page_indices=page_indices,
            page_ref_counts=page_ref_counts,
            seq_lens=new_lens,
        )

        batch_info = self._slice_batch_info(
            updated_seqs,
            cu_new_counts,
            new_table,
            token_slot_ids,
            token_pos_ids,
        )

        return new_table, batch_info

    def _slice_batch_info(self, updated_seqs, cu_q_lens, new_table, slot_ids, pos_ids):
        mask = is_valid(updated_seqs)
        safe_updated = hax.where(mask, updated_seqs, 0)

        gathered_page_indices = new_table.page_indices["seq", safe_updated]
        page_indices = hax.where(mask, gathered_page_indices, INVALID)

        seq_lens = new_table.seq_lens["seq", safe_updated]
        seq_lens = hax.where(mask, seq_lens, INVALID)

        num_seqs = hax.sum(mask).scalar()

        token_dests = hax.full(slot_ids.shape, INVALID, dtype=jnp.int32)

        def token_body(i, token_dests):
            seq_id = slot_ids["position", i].scalar()
            pos_id = pos_ids["position", i].scalar()

            def assign(token_dests):
                page_idx = pos_id // self.page_size
                page_offset = pos_id % self.page_size
                page = new_table.page_indices["seq", seq_id, "page", page_idx]
                dest = hax.where(is_invalid(page), INVALID, page * self.page_size + page_offset)
                return token_dests.at["position", i].set(dest)

            return jax.lax.cond(is_valid(seq_id) & is_valid(pos_id), assign, lambda t: t, token_dests)

        token_dests = jax.lax.fori_loop(0, slot_ids.axis_size("position"), token_body, token_dests)

        # jax.debug.print(
        #     "[allocate_for_seq] slots={slots} pos={pos} dest={dest} cu_q_lens={cu_q_lens} n_seqs={n_seqs}",
        #     slots=slot_ids.array,
        #     pos=pos_ids.array,
        #     dest=token_dests.array,
        #     cu_q_lens=cu_q_lens,
        #     n_seqs=num_seqs,
        # )

        batch_info = PageBatchInfo(
            slot_ids=updated_seqs,
            page_indices=page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            num_seqs=num_seqs,
            new_token_dests=token_dests,
            page_size=self.page_size,
        )
        return batch_info

    @eqx.filter_jit(donate="all")
    def free_pages(self, seq_id: int) -> "PageTable":
        # No-op for invalid sequence ids
        if seq_id < 0 or seq_id >= self.max_seqs:
            return self

        # For the given sequence, decrement ref counts for all valid pages it used
        seq_pages = self.page_indices["seq", seq_id]
        is_valid_page = is_valid(seq_pages)

        def body(i, ref_counts):
            def dec(rc):
                page = seq_pages["page", i].scalar()
                return rc.at["page", page].add(-1)

            return jax.lax.cond(is_valid_page["page", i].scalar(), dec, lambda x: x, ref_counts)

        new_ref_counts = jax.lax.fori_loop(0, seq_pages.axis_size("page"), body, self.page_ref_counts)
        # Clamp at zero to be safe
        new_ref_counts = hax.maximum(new_ref_counts, hax.zeros_like(new_ref_counts))

        new_page_indices = self.page_indices.at["seq", seq_id].set(INVALID)
        new_seq_lens = self.seq_lens.at["seq", seq_id].set(0)
        new_used = self.used_mask.at["seq", seq_id].set(False)

        return dataclasses.replace(
            self,
            page_ref_counts=new_ref_counts,
            page_indices=new_page_indices,
            seq_lens=new_seq_lens,
            used_mask=new_used,
        )

    @eqx.filter_jit(donate="all")
    def free_pages_for_finished(self, finished_mask: jnp.ndarray) -> "PageTable":
        """Free pages and clear metadata for all sequences where ``finished_mask[seq]`` is True.

        Processes all sequences in a single JAX program to avoid per-sequence dispatch overhead.
        """
        assert finished_mask.ndim == 1

        def dec_refcounts_for_seq(pages_row, ref_counts):
            is_valid_page = is_valid(pages_row)

            def body(i, rc):
                def dec(rc):
                    page = pages_row["page", i].scalar()
                    return rc.at["page", page].add(-1)

                return jax.lax.cond(is_valid_page["page", i].scalar(), dec, lambda x: x, rc)

            return jax.lax.fori_loop(0, pages_row.axis_size("page"), body, ref_counts)

        def body(i, state):
            rc, indices, seq_lens, used_mask = state

            def do(rc_ind):
                rc, indices, seq_lens, used_mask = rc_ind
                pages_row = indices["seq", i]
                rc = dec_refcounts_for_seq(pages_row, rc)
                rc = hax.maximum(rc, hax.zeros_like(rc))
                indices = indices.at["seq", i].set(INVALID)
                seq_lens = seq_lens.at["seq", i].set(0)
                used_mask = used_mask.at["seq", i].set(False)
                return rc, indices, seq_lens, used_mask

            return jax.lax.cond(finished_mask[i], do, lambda x: x, (rc, indices, seq_lens, used_mask))

        rc, indices, seq_lens, used_mask = jax.lax.fori_loop(
            0,
            self.max_seqs,
            body,
            (self.page_ref_counts, self.page_indices, self.seq_lens, self.used_mask),
        )

        return dataclasses.replace(
            self, page_ref_counts=rc, page_indices=indices, seq_lens=seq_lens, used_mask=used_mask
        )

    # ------------------------------------------------------------------
    # Page sharing / refcount helpers
    # ------------------------------------------------------------------

    def increment_ref_counts_for_pages(self, pages: NamedArray, count: int = 1) -> "PageTable":
        """
        Increment ref counts for the given page index row, ignoring INVALID entries.

        Args:
            pages: NamedArray of shape [page] containing page indices (or INVALID) for a single sequence.
            count: amount to add to the ref count for each valid page (default 1).
        """
        is_valid_page = is_valid(pages)

        def body(i, ref_counts):
            def inc(rc):
                page = pages["page", i].scalar()
                return rc.at["page", page].add(count)

            return jax.lax.cond(is_valid_page["page", i].scalar(), inc, lambda x: x, ref_counts)

        new_ref_counts = jax.lax.fori_loop(0, pages.axis_size("page"), body, self.page_ref_counts)
        return dataclasses.replace(self, page_ref_counts=new_ref_counts)

    def clone_pages_from(self, src_seq_id: int, dst_seq_id: int) -> "PageTable":
        """
        Make ``dst_seq_id`` reference the same pages as ``src_seq_id`` for all fully used pages, and handle the
        last (possibly partial) page as follows:

        - If the source's last page is full (length divisible by ``page_size``), share it as well and increment
          its refcount.
        - If the source's last page is partial (length not divisible by ``page_size"), allocate a fresh page for the
          clone's last page (do not increment the parent's last-page refcount). This ensures subsequent writes by the
          clone do not affect the parent's partial page. The caller should copy the KV contents for the used slots
          from the parent's last page to the clone's new page if needed.

        The destination's ``seq_lens`` is set equal to the source's ``seq_lens`` (no rounding).
        """
        src_pages = self.page_indices["seq", src_seq_id]
        src_len = self.seq_lens["seq", src_seq_id].scalar()
        size = self.page_size

        # How many pages are used by the source (ceil division)
        used_pages = (src_len + size - 1) // size
        last_idx = hax.maximum(used_pages - 1, 0)
        is_boundary = (src_len % size) == 0

        # Start by copying page mappings from source to dest
        new_indices = self.page_indices.at["seq", dst_seq_id].set(src_pages)

        # Increment refcounts for all fully used pages; if partial, exclude the last page from sharing
        def inc_shared(ref_counts):
            def body(i, rc):
                page = src_pages["page", i]

                def inc(rc):
                    return rc.at["page", page].add(1)

                return jax.lax.cond(is_valid(page).scalar(), inc, lambda x: x, rc)

            # limit = used_pages if boundary else used_pages - 1
            limit = used_pages - jnp.where(is_boundary, 0, 1)
            return jax.lax.fori_loop(0, limit, body, ref_counts)

        ref_counts = inc_shared(self.page_ref_counts)

        # If partial, allocate a fresh page for the last slot
        def handle_partial(state):
            rc, indices = state
            has_free = hax.any(rc == 0).scalar()
            rc = eqx.error_if(rc, ~has_free, "Out of free pages during clone_pages_from")
            free_idx = hax.argmax(rc == 0, "page")
            indices = indices.at["seq", dst_seq_id, "page", last_idx].set(free_idx)
            rc = rc.at["page", free_idx].add(1)
            return rc, indices

        ref_counts, new_indices = jax.lax.cond(
            is_boundary,
            lambda s: s,
            handle_partial,
            (ref_counts, new_indices),
        )

        # Keep destination length equal to source length
        new_seq_lens = self.seq_lens.at["seq", dst_seq_id].set(src_len)
        new_used = self.used_mask.at["seq", dst_seq_id].set(True)

        return dataclasses.replace(
            self, page_indices=new_indices, page_ref_counts=ref_counts, seq_lens=new_seq_lens, used_mask=new_used
        )

    def bump_seq_len_to_next_page(self, seq_id: int) -> "PageTable":
        """
        Increase ``seq_lens[seq_id]`` to the start of the next page to force the next token onto a fresh page.
        Useful when creating clones that should start writing on a different page from the source.
        """
        cur = self.seq_lens["seq", seq_id]
        size = jnp.array(self.page_size, dtype=jnp.int32)
        next_page = ((cur + size - 1) // size) * size
        new_seq_lens = self.seq_lens.at["seq", seq_id].set(next_page)
        return dataclasses.replace(self, seq_lens=new_seq_lens)


class PageBatchInfo(eqx.Module):
    """
    Page and length information for a batch of sequences.

    NOTE: the "sequence" indices here are not the same as the sequence indices in DecodeState. That is,
    page_indices[0] does not in general correspond to the first sequence in DecodeState, but rather the first sequence
    that has tokens **in this batch**.

    To recover the mapping, use slot_ids to map from batch sequence index to DecodeState sequence index.
    """

    slot_ids: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    page_indices: ht.i32[NamedArray, " seq page"]  # type: ignore[name-defined]
    seq_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    cu_q_lens: ht.i32[NamedArray, " seq"]  # type: ignore[name-defined]
    num_seqs: ht.i32[jnp.ndarray, ""]
    new_token_dests: ht.i32[NamedArray, "position"]  # type: ignore[name-defined]
    page_size: int = eqx.field(static=True)

    def __post_init__(self):
        assert isinstance(self.num_seqs, jnp.ndarray), "num_seqs must be a JAX ndarray"

    def pages_and_slots(self):
        token_dests = self.new_token_dests

        t_pages = hax.where(is_valid(token_dests), token_dests // self.page_size, INVALID)
        t_slots = hax.where(is_valid(token_dests), token_dests % self.page_size, INVALID)
        # jax.debug.print(
        #     "[pages_and_slots] dest={dest} page={page} slot={slot}",
        #     dest=token_dests.array,
        #     page=t_pages.array,
        #     slot=t_slots.array,
        # )

        return t_pages, t_slots
