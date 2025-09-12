# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence
from collections import deque

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
import haliax.haxtyping as ht
from haliax import Axis, NamedArray

from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID, is_valid
from levanter.layers.attention import KvPageCache
from levanter.layers.sampler import Sampler
from levanter.models.lm_model import LmHeadModel
from levanter.inference.jit_scheduler import DecodeState, SeqDecodingParams, TokenQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    """A request for generation of a single sequence."""

    prompt_tokens: list[int]
    request_id: int
    decode_params: SeqDecodingParams
    n_generations: int


@dataclass(frozen=True)
class GenerationServiceConfig:
    """Configuration for GenerationService memory/layout knobs.

    Exposes key buffer sizes and limits controlling prefill, decode queueing, and page table capacity.
    """

    max_pages: Optional[int] = None
    """Total number of KV pages available. If None, computed as ``max_seqs * max_pages_per_seq``."""
    max_seqs: int = 16
    """Maximum concurrent sequences (local slots)."""
    page_size: int = 128
    """Tokens per KV page."""
    max_pages_per_seq: int = 64
    """Maximum pages a single sequence may use (max sequence length = page_size * max_pages_per_seq)."""
    compute_dtype: jnp.dtype = jnp.bfloat16
    """KV cache dtype. Default bfloat16 for performance/accuracy balance."""
    max_queued_tokens: int = 512
    """Capacity of the token queue used between sampling and decode packing."""
    max_seqs_in_prefill: int = 16
    """Maximum number of sequences to batch in prefill before flushing."""

    # Decode loop knobs
    max_tokens_per_round: int = 8
    """Pack size for each decode loop iteration."""
    max_rounds: int = 8
    """Maximum number of while-loop iterations per decode call."""

    # Default PRNG seed for building per-request keys (optional convenience)
    seed: int = 0

    @property
    def imputed_max_pages(self) -> int:
        """Return explicit `max_pages` or compute `max_seqs * max_pages_per_seq` when unset."""
        return int(self.max_pages) if self.max_pages is not None else int(self.max_seqs * self.max_pages_per_seq)


class GenState(eqx.Module):
    """Container for generation state used during decoding.

    Holds the page table, KV cache, and `DecodeState`. Provides `clone_sequence` to
    efficiently support multi-sample generation by sharing fully used pages.
    """

    cache: KvPageCache
    page_table: PageTable
    decode_state: DecodeState

    def clone_sequence(
        self,
        parent_local_id: int,
        child_local_id: int | None = None,
        *,
        global_id: int | None = None,
        seq_params: SeqDecodingParams | None = None,
    ) -> tuple["GenState", int]:
        """Clone a sequence into a new local slot, sharing full pages and using a fresh page for the last partial page.

        Args:
            parent_local_id: Local sequence id to clone from.
            child_local_id: Optional local id to clone into; allocated if None.
            global_id: Global id to assign to the clone in `DecodeState`.
            seq_params: Per-sequence decoding parameters for the clone.

        Returns:
            updated GenState
        """
        page_table = self.page_table
        if child_local_id is None:
            page_table, new_child = page_table.assign_seq_id_to_seq()
            child_local_id = int(jax.device_get(new_child))

        decode_state = self.decode_state
        prefix_len = int(decode_state.seq_lens["seq", parent_local_id].scalar())
        parent_prefix = decode_state.tokens["seq", parent_local_id, "position", 0:prefix_len]
        parent_kv_pages_row = decode_state.kv_pages["seq", parent_local_id]
        gid = int(decode_state.seq_id["seq", parent_local_id].scalar()) if global_id is None else global_id

        # Assign child sequence state (copies tokens up to prefix and kv_pages row)
        decode_state = decode_state.assign_seq(
            local_seq_id=child_local_id,
            global_seq_id=gid,
            tokens=parent_prefix,
            prefix_len=prefix_len,
            kv_pages=parent_kv_pages_row,
            seq_params=seq_params,
        )
        # Record clone mapping on the child slot
        decode_state = dataclasses.replace(
            decode_state,
            clone_sources=decode_state.clone_sources.at["seq", child_local_id].set(parent_local_id),
        )

        page_table = page_table.clone_pages_from(parent_local_id, child_local_id)

        page_size = page_table.page_size
        src_len = int(page_table.seq_lens["seq", parent_local_id].scalar())

        def _copy(_):
            last_idx = (src_len + page_size - 1) // page_size - 1
            src_page = page_table.page_indices["seq", parent_local_id, "page", last_idx].scalar()
            dst_page = page_table.page_indices["seq", child_local_id, "page", last_idx].scalar()
            return self.cache.copy_page(src_page, dst_page)

        def _identity(_):
            return self.cache

        cache = jax.lax.cond((src_len % page_size != 0) and (src_len > 0), _copy, _identity, None)

        new_state = dataclasses.replace(self, page_table=page_table, decode_state=decode_state, cache=cache)
        return new_state, child_local_id


def _compute_sample_indices(pos_ids, seq_ids, seq_lens, max_sample_indices):
    """
    Compute positions of last tokens per sequence inside a packed slice.

    Boundary when absolute pos_id equals the post-allocation seq_len - 1 for that sequence.
    """
    seq_lens_per_seq = seq_lens["seq", seq_ids]
    boundary_mask = pos_ids == (seq_lens_per_seq - 1)
    sample_indices = hax.where(
        boundary_mask,
        fill_value=INVALID,
        new_axis=pos_ids.resolve_axis("position").resize(max_sample_indices),
    )[0]
    return sample_indices


@hax.named_jit(donate_args=(True, False, False, False, False))
def _run_prefill(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    queue: TokenQueue,
    max_seqs_in_prefill: int,  # static
) -> GenState:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    tokens = queue.queued_tokens
    pos_ids = queue.queued_pos_ids
    seq_ids = queue.queued_seq_ids
    seq_lens = gen_state.decode_state.seq_lens

    page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=seq_ids)

    sample_indices = _compute_sample_indices(pos_ids, seq_ids, seq_lens, max_seqs_in_prefill)

    logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
    logits_at_samples = logits["position", sample_indices]

    num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
    new_seq_ids = seq_ids["position", sample_indices]
    new_pos_ids = pos_ids["position", sample_indices]
    prng_keys = gen_state.decode_state.prng_keys_for(new_seq_ids, new_pos_ids)

    temps = gen_state.decode_state.temperature["seq", new_seq_ids]

    new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

    # Update decode_state (also enqueues into the main decode queue)
    decode_state = gen_state.decode_state.update_tokens(new_tokens, new_seq_ids, log_probs, num_new_tokens)

    gen_state = dataclasses.replace(gen_state, page_table=page_table, cache=cache, decode_state=decode_state)

    # If clone targets specified, sample alternative tokens for clones using the same logits slice
    if decode_state.clone_sources is not None:
        gen_state, _covered_clones = _handle_clones(
            gen_state,
            logits_at_samples,
            new_seq_ids,
            new_pos_ids,
            sampler,
        )
    return gen_state


def _handle_clones(
    gen_state: GenState,
    logits: ht.Float[NamedArray, " position vocab"],  # type: ignore
    seq_ids: ht.Int[NamedArray, " position"],  # type: ignore
    pos_ids: ht.Int[NamedArray, " position"],  # type: ignore
    sampler: Sampler,
) -> tuple[GenState, ht.bool_[NamedArray, " seq"]]:  # type: ignore
    """
    Sample alternative tokens for the given logits, seq_ids, pos_ids, and clone_targets.
    This is used for the `n>1` case of `n_generations` in the `Request` class.

    Uses ``gen_state.decode_state.clone_sources`` as a mapping from target local ids to source local ids.

    It's assumed that:
      1. gen_state already has the appropriate page table and decode state.
      2. logits/seq_ids/pos_ids are already sliced

    Returns the updated gen_state and a boolean array indicating which ids from `clone_targets` were sampled.
    """
    # Resolve axes
    CloneSeq = gen_state.decode_state.clone_sources.resolve_axis("seq")

    # For each clone source, find its index in the provided seq_ids (within this packed/sliced batch).
    # If not present, mark as INVALID.
    def find_src(i):
        src = gen_state.decode_state.clone_sources["seq", i].scalar()

        def do(src):
            # match positions where seq_ids == src; take first
            eq = (seq_ids == src).array
            idx = jnp.nonzero(eq, size=1, fill_value=INVALID)[0][0]
            return idx

        return jax.lax.cond(is_valid(src), do, lambda x: x, src)

    # source_indices tells us, for each sequence that is a clone target, the index in the
    # logits/seq_ids/pos_ids arrays of its source sequence.
    # INVALID if either no source or source not in this batch.
    source_indices = hax.named(hax.vmap(find_src, "seq")(jnp.arange(CloneSeq.size)), axis="seq")

    # Determine which clone targets can be sampled this step:
    # need a valid source index and a valid target id
    can_sample = source_indices != INVALID

    # Build a compact position index list of clones to process this time
    selected = hax.where(can_sample, fill_value=INVALID, new_axis=CloneSeq)[0]
    selected = selected.rename({"seq": "position"})

    num_new = hax.sum(selected != INVALID).scalar().astype(jnp.int32)

    # Gather per-clone data
    # Use a masked/guarded gather to keep shapes static. First entries are valid clones.
    selected_safe = hax.where(selected != INVALID, selected, 0)
    tgt_ids = selected_safe
    src_pos = source_indices["seq", selected_safe]
    src_ids = seq_ids["position", src_pos]
    logits_this_time = logits["position", src_pos]
    pos_ids_this_time = pos_ids["position", src_pos]

    # Sample clones from the same boundary logits as their sources
    temps = gen_state.decode_state.temperature["seq", tgt_ids]
    prng_keys = gen_state.decode_state.prng_keys_for(tgt_ids, pos_ids_this_time)

    new_tokens, log_probs = hax.vmap(sampler, "position")(logits_this_time, temps, key=prng_keys)

    # update page table and cache for the clone targets
    page_table = gen_state.page_table
    cache = gen_state.cache
    size = page_table.page_size

    def copy_pages_for_updated_seq(
        i,
        state: tuple[PageTable, KvPageCache],
    ) -> tuple[PageTable, KvPageCache]:
        page_table, cache = state
        src_seq_id = src_ids["position", i].scalar()
        dst_seq_id = tgt_ids["position", i].scalar()
        page_table = page_table.clone_pages_from(src_seq_id, dst_seq_id)

        src_len = page_table.seq_lens["seq", src_seq_id].scalar()
        used_pages = (src_len + size - 1) // size
        last_idx = jnp.maximum(used_pages - 1, 0)

        def _copy(_):
            src_page = page_table.page_indices["seq", src_seq_id, "page", last_idx].scalar()
            dst_page = page_table.page_indices["seq", dst_seq_id, "page", last_idx].scalar()
            return cache.copy_page(src_page, dst_page)

        def _identity(_):
            return cache

        cache = jax.lax.cond((src_len % size != 0) & (src_len > 0), _copy, _identity, None)
        return page_table, cache

    page_table, cache = jax.lax.fori_loop(0, num_new, copy_pages_for_updated_seq, (page_table, cache))

    # Enqueue/update tokens for the clone targets (only the first num_new entries will be used)
    decode_state = gen_state.decode_state.update_tokens(new_tokens, tgt_ids, log_probs, num_new)
    # Discharge processed clones so they are not reprocessed in subsequent flushes
    decode_state = decode_state.discharge_clone(tgt_ids, num_new)
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state, page_table=page_table, cache=cache)

    # Return which clones we sampled this time (mask over seq axis)
    return gen_state, can_sample


@hax.named_jit(donate_args=(True, False, False))
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
) -> GenState:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached."""

    def cond(state: tuple[GenState, jax.Array, jax.Array]):
        _gen_state, finished, step = state
        return (step < max_rounds) & (_gen_state.decode_state.num_queued_tokens > 0) & (~jnp.all(finished))

    def body(state: tuple[GenState, jax.Array, jax.Array]) -> tuple[GenState, jax.Array, jax.Array]:
        gen_state, has_finished, step = state

        # Pack the next chunk from the queue via DecodeState
        decode_state, packed_seq = gen_state.decode_state.pack_next_sequence(max_tokens_per_round)

        tokens = packed_seq.tokens
        pos_ids = packed_seq.pos_ids
        seq_ids = packed_seq.seq_ids
        # NB: use decode_state.num_tokens to determine the number of tokens in each sequence, not what's in page table
        seq_lens = decode_state.seq_lens

        page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=seq_ids)

        max_sample_indices = min(page_table.max_seqs, max_tokens_per_round)
        sample_indices = _compute_sample_indices(pos_ids, seq_ids, seq_lens, max_sample_indices)

        # Decode logits and sample new tokens
        logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
        logits_at_samples = logits["position", sample_indices]

        num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
        new_seq_ids = seq_ids["position", sample_indices]
        new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_seq_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_seq_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_seq_ids, log_probs, num_new_tokens)
        new_finished = decode_state.is_finished(jnp.arange(decode_state.max_seqs))
        has_finished = has_finished | new_finished

        # purge any finished sequencse
        finished_sequences = jnp.nonzero(new_finished, size=gen_state.page_table.max_seqs, fill_value=INVALID)[0]
        finished_sequences = hax.named(finished_sequences, axis="seq")
        decode_state = decode_state.purge_queue_of_seq(finished_sequences)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(
            gen_state,
            page_table=page_table,
            cache=cache,
            decode_state=decode_state,
        )
        return new_gen_state, has_finished, step + 1

    has_finished = gen_state.decode_state.is_finished(jnp.arange(gen_state.decode_state.max_seqs))
    init_state = (gen_state, has_finished, jnp.array(0, dtype=jnp.int32))
    final_gen_state, _, _ = jax.lax.while_loop(cond, body, init_state)
    return final_gen_state


class GenerationService:
    """Encapsulates batch inference: prefill + decode + output extraction.

    Typical usage:

        svc = GenerationService.from_model(model, tokenizer, Vocab, max_seqs, max_pages, page_size, max_pages_per_seq, compute_dtype)
        texts = svc.generate(requests)
    """

    def __init__(
        self,
        *,
        model: LmHeadModel,
        tokenizer,
        table: PageTable,
        cache: KvPageCache,
        decode_state: DecodeState,
        sampler: Sampler,
        config: GenerationServiceConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.gen_state: GenState = GenState(cache=cache, page_table=table, decode_state=decode_state)
        self._initial_decode_state = decode_state
        self.config = config
        # Track free local sequence slots
        self.free_slots: int = int(table.max_seqs)
        # Mapping structures for active sequences
        # local_map: local seq id -> (request id, child id)
        # sequences: request id -> {child id -> local seq id}
        self.local_map: dict[int, tuple[int, int]] = {}
        self.sequences: dict[int, dict[int, int]] = {}
        # FIFO request queue for incremental admission
        self.request_queue: deque[Request] = deque()

    @classmethod
    def from_model(
        cls,
        *,
        model: LmHeadModel,
        tokenizer,
        vocab_axis: Axis,
        max_pages: int,
        max_seqs: int,
        page_size: int,
        max_pages_per_seq: int,
        compute_dtype,
        max_queued_tokens: int = 32,
        max_seqs_in_prefill: int = 16,
    ) -> "GenerationService":
        """Build a service with fresh PageTable/KV cache/DecodeState and a `Sampler` for `vocab_axis`."""
        table = PageTable.init(max_pages, max_seqs, page_size, max_pages_per_seq)
        cache = hax.named_jit(model.initial_cache)(table, dtype=compute_dtype)
        decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
            max_stop_seqs=1,
            max_queued_tokens=max_queued_tokens,
        )
        sampler = Sampler(vocab_axis)
        cfg = GenerationServiceConfig(
            max_pages=max_pages,
            max_seqs=max_seqs,
            page_size=page_size,
            max_pages_per_seq=max_pages_per_seq,
            compute_dtype=compute_dtype,
            max_queued_tokens=max_queued_tokens,
            max_seqs_in_prefill=max_seqs_in_prefill,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            table=table,
            cache=cache,
            decode_state=decode_state,
            sampler=sampler,
            config=cfg,
        )

    @classmethod
    def from_model_with_config(
        cls,
        *,
        model: LmHeadModel,
        tokenizer,
        config: GenerationServiceConfig,
    ) -> "GenerationService":
        """Build a service using a GenerationServiceConfig for sizing knobs."""
        table = PageTable.init(config.imputed_max_pages, config.max_seqs, config.page_size, config.max_pages_per_seq)
        cache = hax.named_jit(model.initial_cache)(table, dtype=config.compute_dtype)
        decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
            max_stop_seqs=1,
            max_queued_tokens=config.max_queued_tokens,
        )
        sampler = Sampler(model.Vocab)
        return cls(
            model=model,
            tokenizer=tokenizer,
            table=table,
            cache=cache,
            decode_state=decode_state,
            sampler=sampler,
            config=config,
        )

    def reset(self) -> None:
        """Free all local sequence slots and reset to the initial `DecodeState`.

        Keeps the KV cache memory allocated. Reuses current `PageTable` object with pages freed.
        """
        page_table = self.gen_state.page_table
        for seq_id in range(page_table.max_seqs):
            page_table = page_table.free_pages(seq_id)
        self.gen_state = dataclasses.replace(
            self.gen_state,
            page_table=page_table,
            decode_state=self._initial_decode_state,
        )
        # All local slots are free after reset
        self.free_slots = int(page_table.max_seqs)
        self.local_map.clear()
        self.sequences.clear()

    def _release_finished_sequences(self) -> None:
        """Release resources for any sequences that have finished.

        - Frees pages in the PageTable for finished local sequence ids.
        - Clears DecodeState bookkeeping so the local slots can be reused.

        This makes explicit `reset()` calls generally unnecessary between batches.
        """
        ds = self.gen_state.decode_state
        pt = self.gen_state.page_table

        # Determine which local sequence slots are finished
        finished_mask = jax.device_get(ds.is_finished(jnp.arange(ds.max_seqs))).astype(bool)
        finished_locals = [i for i, f in enumerate(finished_mask) if bool(f)]

        if len(finished_locals) == 0:
            return

        # Capture current global ids for logging before we clear them
        try:
            seq_ids_host = jax.device_get(ds.seq_id.array)
            finished_globals = [int(seq_ids_host[i]) for i in finished_locals]
        except Exception:
            finished_globals = []
        logger.info(f"Releasing finished sequences: locals={finished_locals}, globals={finished_globals}")

        # Free pages for finished sequences
        for local_seq in finished_locals:
            pt = pt.free_pages(local_seq)
            # Maintain request/child mapping
            info = self.local_map.pop(local_seq, None)
            if info is not None:
                rid, cid = info
                cmap = self.sequences.get(rid)
                if cmap is not None:
                    cmap.pop(cid, None)
                    if len(cmap) == 0:
                        self.sequences.pop(rid, None)
            # Freed one local slot
            self.free_slots += 1

        # Clear DecodeState slot metadata for finished sequences
        # Build a boolean NamedArray mask over the seq axis
        Seq = ds.seq_id.resolve_axis("seq")
        mask = hax.named(jnp.asarray(finished_mask, dtype=bool), axis=Seq)

        # Invalidate ids and lengths for finished slots
        new_seq_id = hax.where(mask, hax.full_like(ds.seq_id, INVALID), ds.seq_id)
        new_seq_lens = hax.where(mask, hax.full_like(ds.seq_lens, INVALID), ds.seq_lens)
        new_prefix_len = hax.where(mask, hax.full_like(ds.prefix_len, 0), ds.prefix_len)
        new_clone_sources = hax.where(mask, hax.full_like(ds.clone_sources, INVALID), ds.clone_sources)

        # Invalidate kv_pages rows for finished slots
        new_kv_pages = ds.kv_pages
        for local_seq in finished_locals:
            new_kv_pages = new_kv_pages.at["seq", local_seq].set(INVALID)

        self.gen_state = dataclasses.replace(
            self.gen_state,
            page_table=pt,
            decode_state=dataclasses.replace(
                ds,
                seq_id=new_seq_id,
                seq_lens=new_seq_lens,
                prefix_len=new_prefix_len,
                clone_sources=new_clone_sources,
                kv_pages=new_kv_pages,
            ),
        )

    # ------------------------------- Queue helpers -------------------------------
    def enqueue_requests(self, requests: Sequence[Request]) -> None:
        for r in requests:
            self.request_queue.append(r)

    def _admit_from_queue(self) -> int:
        """Admit a batch from the head of the queue that fits in free slots/pages.

        Returns the number of admitted requests.
        """
        if not self.request_queue:
            return 0
        sim_slots = int(self.free_slots)
        sim_pages = self._free_page_count()
        batch: list[Request] = []
        while self.request_queue:
            nxt = self.request_queue[0]
            need_slots = int(nxt.n_generations)
            need_pages = self._pages_needed_for_prompt(len(nxt.prompt_tokens))
            if sim_slots < need_slots or sim_pages < need_pages:
                break
            batch.append(self.request_queue.popleft())
            sim_slots -= need_slots
            sim_pages -= need_pages
        if not batch:
            return 0
        _ = self._prefill_prompts(batch, primary_global_ids=list(range(len(batch))))
        self.gen_state = jax.block_until_ready(self.gen_state)
        return len(batch)

    def _free_page_count(self) -> int:
        """Return number of free KV pages in the PageTable."""
        prc = jax.device_get(self.gen_state.page_table.page_ref_counts.array)
        return int((prc == 0).sum())

    def _pages_needed_for_prompt(self, prompt_len: int) -> int:
        size = int(self.gen_state.page_table.page_size)
        return (int(prompt_len) + size - 1) // size

    def _prefill_prompts(
        self,
        requests: Sequence[Request],
        *,
        primary_global_ids: Optional[Sequence[int]] = None,
    ) -> list[int]:
        """Assign sequence ids, set per-seq params, and run prefill using a local queue."""
        MAX_SEQS_IN_PREFILL = int(self.config.max_seqs_in_prefill)
        tokens = jnp.full((128,), INVALID, dtype=jnp.int32)
        seq_ids = jnp.full((128,), INVALID, dtype=jnp.int32)
        pos_ids = jnp.full((128,), INVALID, dtype=jnp.int32)
        offset = 0
        num_seqs_in_prefill = 0

        primary_local_ids: list[int] = []

        for idx, request in enumerate(requests):
            seq_tokens = request.prompt_tokens
            seq_params = request.decode_params

            page_table, seq_id = self.gen_state.page_table.assign_seq_id_to_seq()
            seq_id = int(seq_id)

            if not is_valid(seq_id):
                raise RuntimeError("Ran out of sequence IDs in the page table during prefill.")

            self.gen_state = dataclasses.replace(self.gen_state, page_table=page_table)

            this_tokens = jnp.asarray(seq_tokens, dtype=jnp.int32)
            if len(seq_tokens) + offset > tokens.shape[0] or num_seqs_in_prefill >= MAX_SEQS_IN_PREFILL:
                token_queue = TokenQueue(
                    queued_tokens=hax.named(tokens, axis="position"),
                    queued_seq_ids=hax.named(seq_ids, axis="position"),
                    queued_pos_ids=hax.named(pos_ids, axis="position"),
                    num_queued_tokens=jnp.array(offset, dtype=jnp.int32),
                )
                self.gen_state = _run_prefill(
                    self.gen_state, self.model, self.sampler, token_queue, MAX_SEQS_IN_PREFILL
                )
                tokens = jnp.full((128,), INVALID, dtype=jnp.int32)
                seq_ids = jnp.full((128,), INVALID, dtype=jnp.int32)
                pos_ids = jnp.full((128,), INVALID, dtype=jnp.int32)
                offset = 0
                num_seqs_in_prefill = 0

            tokens = tokens.at[offset : offset + len(seq_tokens)].set(this_tokens)
            seq_ids = seq_ids.at[offset : offset + len(seq_tokens)].set(seq_id)
            pos_ids = pos_ids.at[offset : offset + len(seq_tokens)].set(jnp.arange(len(seq_tokens), dtype=jnp.int32))
            offset += len(seq_tokens)
            num_seqs_in_prefill += 1

            self.gen_state = dataclasses.replace(
                self.gen_state,
                decode_state=self.gen_state.decode_state.assign_seq(
                    local_seq_id=seq_id,
                    global_seq_id=(primary_global_ids[idx] if primary_global_ids is not None else request.request_id),
                    tokens=hax.named(this_tokens, axis="position"),
                    prefix_len=len(seq_tokens),
                    kv_pages=None,
                    seq_params=seq_params,
                ),
            )
            # Consume one free slot for this primary
            self.free_slots -= 1
            # Record mapping: primary child id is 0
            rid = int(request.request_id)
            self.local_map[seq_id] = (rid, 0)
            self.sequences.setdefault(rid, {})[0] = seq_id
            primary_local_ids.append(seq_id)

            if request.n_generations > 1:
                for k in range(1, request.n_generations):
                    self.gen_state, child_local_id = self.gen_state.clone_sequence(
                        seq_id,
                        global_id=(
                            primary_global_ids[idx] + k if primary_global_ids is not None else request.request_id
                        ),
                        seq_params=dataclasses.replace(seq_params, key=jax.random.fold_in(seq_params.key, k)),
                    )
                    # Consume one free slot for the clone
                    self.free_slots -= 1
                    # Record mapping for clone: child id is k
                    self.local_map[child_local_id] = (rid, k)
                    self.sequences.setdefault(rid, {})[k] = child_local_id

        if offset > 0:
            token_queue = TokenQueue(
                queued_tokens=hax.named(tokens, axis="position"),
                queued_seq_ids=hax.named(seq_ids, axis="position"),
                queued_pos_ids=hax.named(pos_ids, axis="position"),
                num_queued_tokens=jnp.array(offset, dtype=jnp.int32),
            )
            self.gen_state = _run_prefill(self.gen_state, self.model, self.sampler, token_queue, MAX_SEQS_IN_PREFILL)

        return primary_local_ids

    def _extract_outputs(self, outputs: list[list[int]], finished: list[bool]) -> int:
        """Drain generated tokens from `DecodeState` and append into `outputs`.

        Returns the number of new tokens extracted this call.
        """
        total_this_time = 0
        decode_state = self.gen_state.decode_state
        num_seqs = decode_state.max_seqs
        tokens = jax.device_get(decode_state.tokens.array)
        num_tokens = jax.device_get(decode_state.seq_lens.array)
        seq_ids = jax.device_get(decode_state.seq_id.array)
        this_finished = jax.device_get(decode_state.is_finished(jnp.arange(num_seqs)))

        for local_seq in range(num_seqs):
            global_id = int(seq_ids[local_seq])
            if global_id < 0 or global_id == INVALID:
                continue

            current_num_tokens = len(outputs[global_id])
            new_num_tokens = int(num_tokens[local_seq])
            count_to_extract = new_num_tokens - current_num_tokens

            if finished[global_id]:
                continue

            total_this_time += max(0, count_to_extract)
            if bool(this_finished[local_seq]):
                finished[global_id] = True

            if count_to_extract <= 0:
                continue

            seq_tokens = tokens[local_seq, current_num_tokens:new_num_tokens]
            outputs[global_id].extend(int(x) for x in seq_tokens)

        return total_this_time

    def generate(self, requests: Sequence[Request]) -> tuple[list[list[int]], int]:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).
        """
        # Enqueue incoming requests to internal queue
        self.enqueue_requests(requests)
        # Track outputs and finished flags for only this call's requests
        call_rids = [int(r.request_id) for r in requests]
        expected_children: dict[int, int] = {rid: int(r.n_generations) for rid, r in zip(call_rids, requests)}
        outputs_for: dict[int, dict[int, list[int]]] = {
            rid: {k: [] for k in range(expected_children[rid])} for rid in expected_children
        }
        finished_for: dict[int, dict[int, bool]] = {
            rid: {k: False for k in range(expected_children[rid])} for rid in expected_children
        }

        # Ensure stop-token buffer capacity is sufficient based on the longest stop sequence requested
        desired_stop_len = 0
        for req in requests:
            if req.decode_params.stop_tokens is not None:
                desired_stop_len = max(desired_stop_len, int(req.decode_params.stop_tokens.axis_size("position")))
        ds = self.gen_state.decode_state
        current_stop_len = 0 if ds.stop_tokens is None else ds.stop_tokens.axis_size("position")
        if desired_stop_len > current_stop_len:
            # Reinitialize an empty DecodeState with larger stop-token capacity
            new_ds = DecodeState.init(
                ds.max_seqs,
                ds.kv_pages.axis_size("page"),
                ds.page_size,
                ds.tokens.axis_size("position"),
                max_stop_seqs=1 if desired_stop_len > 0 else 0,
                max_stop_tokens=desired_stop_len,
                max_queued_tokens=int(ds.tqueue.max_queued_tokens),
            )
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=new_ds)
            self._initial_decode_state = new_ds

        time_in = time.time()
        # Try initial admission from queue and extract prompt tokens
        _ = self._admit_from_queue()
        _ = self._extract_outputs_for(outputs_for, finished_for)
        self._release_finished_sequences()

        # Autoregressive generation loop with periodic extraction
        def _all_done() -> bool:
            for rid, kids in finished_for.items():
                if any(not v for v in kids.values()):
                    return False
            return True

        stagnant_iters = 0
        while not _all_done():
            t0 = time.time()
            self.gen_state = _run_generation_loop(
                self.gen_state,
                self.model,
                self.sampler,
                self.config.max_tokens_per_round,
                self.config.max_rounds,
            )
            loop_time = time.time() - t0
            self.gen_state = jax.block_until_ready(self.gen_state)
            new_tokens = self._extract_outputs_for(outputs_for, finished_for)
            # Release any sequences that finished in this step
            self._release_finished_sequences()
            # Admit more if capacity allows
            _ = self._admit_from_queue()
            _ = self._extract_outputs_for(outputs_for, finished_for)
            self._release_finished_sequences()
            if loop_time > 0:
                tps = new_tokens / loop_time
                logger.info(f"Decode iter: {loop_time:.3f}s, {tps:.2f} tok/s, {new_tokens} new")
            # Safety: if nothing new was produced and queue is empty, avoid infinite loop
            if (
                new_tokens == 0
                and int(jax.device_get(self.gen_state.decode_state.num_queued_tokens)) == 0
                and not self.request_queue
            ):
                stagnant_iters += 1
            else:
                stagnant_iters = 0
            if stagnant_iters >= 2:
                logger.warning("No progress in decoding for 2 consecutive iterations; breaking to avoid hang.")
                break

        # Assemble outputs in the order of the requests for this call
        outputs_list: list[list[int]] = []
        total_prompt_tokens = 0
        for r in requests:
            rid = int(r.request_id)
            total_prompt_tokens += len(r.prompt_tokens) * int(r.n_generations)
            for k in range(int(r.n_generations)):
                outputs_list.append(outputs_for[rid][k])
        total_generated = sum(len(seq_outputs) for seq_outputs in outputs_list) - total_prompt_tokens
        logger.info(f"Batch generated in {time.time() - time_in:.2f}s, {total_generated} tokens")
        return outputs_list, total_generated

    def _extract_outputs_for(
        self,
        outputs: dict[int, dict[int, list[int]]],
        finished: dict[int, dict[int, bool]],
    ) -> int:
        """Append newly available tokens into outputs per (request_id, child_id).

        Returns number of new tokens appended.
        """
        total_this_time = 0
        decode_state = self.gen_state.decode_state
        num_seqs = decode_state.max_seqs
        tokens = jax.device_get(decode_state.tokens.array)
        num_tokens = jax.device_get(decode_state.seq_lens.array)
        this_finished = jax.device_get(decode_state.is_finished(jnp.arange(num_seqs)))

        for local_seq in range(num_seqs):
            info = self.local_map.get(local_seq)
            if info is None:
                continue
            rid, cid = info
            if rid not in outputs:
                continue
            current_num_tokens = len(outputs[rid][cid])
            new_num_tokens = int(num_tokens[local_seq])
            count_to_extract = new_num_tokens - current_num_tokens
            if count_to_extract <= 0:
                if bool(this_finished[local_seq]):
                    finished[rid][cid] = True
                continue
            total_this_time += count_to_extract if count_to_extract > 0 else 0
            if bool(this_finished[local_seq]):
                finished[rid][cid] = True
            seq_tokens = tokens[local_seq, current_num_tokens:new_num_tokens]
            outputs[rid][cid].extend(int(x) for x in seq_tokens)

        return total_this_time
