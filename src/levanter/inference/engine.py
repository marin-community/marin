# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import equinox as eqx
import haliax as hax
import haliax.haxtyping as ht
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from haliax.jax_utils import is_jax_array_like

from levanter.inference.jit_scheduler import (
    DecodeState,
    SeqDecodingParams,
    TokenQueue,
    _DecodeOutputs,
)
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID, is_valid
from levanter.layers.attention import KvPageCache
from levanter.layers.sampler import Sampler
from levanter.models.lm_model import LmHeadModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    """A request for generation of a single sequence."""

    prompt_tokens: list[int]
    request_id: int
    decode_params: SeqDecodingParams
    n_generations: int
    enable_logprobs: bool = False


@dataclasses.dataclass
class DecodeResult:
    """Holds per-(request, choice) decode outputs and status."""

    id: int
    choice: int
    token_list: list[int]
    # Count of newly appended tokens (includes prompt tokens as extracted)
    tokens_decoded: int = 0
    done: bool = False
    logprobs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class InferenceEngineConfig:
    """Configuration for Engine memory/layout knobs.

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

    # Prefill buffer sizing
    max_prefill_size: Optional[int] = None
    """Maximum number of tokens packed into the prefill buffer before a flush.

    If None, inferred at construction time from `tokenizer.model_max_length` when available; otherwise
    falls back to the page table's `max_len_per_seq` or 4096 as a final default.
    """

    # Decode loop knobs
    max_tokens_per_round: int | None = None
    """Pack size for each decode loop iteration."""
    max_rounds: int = 8
    """Maximum number of while-loop iterations per decode call."""

    # Stop-token capacity (used for validation and buffer sizing at init)
    max_stop_seqs: int = 4
    """Maximum number of stop sequences per active sequence. 0 disables stop tokens."""
    max_stop_tokens: int = 16
    """Maximum tokens per stop sequence (position axis length)."""

    # Default PRNG seed for building per-request keys (optional convenience)
    seed: int = 0

    enable_logprobs: bool = False
    """Enable computing logprobs for generated tokens."""

    @property
    def imputed_max_pages(self) -> int:
        """Return explicit `max_pages` or compute `max_seqs * max_pages_per_seq` when unset."""
        return int(self.max_pages) if self.max_pages is not None else int(self.max_seqs * self.max_pages_per_seq)


class GenState(eqx.Module):
    """Container for generation state used during decoding.

    Holds the KV cache and `DecodeState` (which itself owns the `PageTable`).
    Provides `clone_sequence` to efficiently support multi-sample generation by
    sharing fully used pages.
    """

    cache: KvPageCache
    decode_state: DecodeState

    def clone_sequence(
        self, parent_local_id: int, child_local_id: int | None = None, seq_params: SeqDecodingParams | None = None
    ) -> tuple["GenState", int]:
        """Clone a sequence into a new local slot, sharing full pages and using a fresh page for the last partial page.

        DONATES self.

        Args:
            parent_local_id: Local slot id to clone from.
            child_local_id: Optional local slot id to clone into; allocated if None.
            seq_params: Per-sequence decoding parameters for the clone.

        Returns:
            updated GenState, child_local_id (which will be INVALID if allocation failed).
        """
        if isinstance(parent_local_id, int):
            parent_local_id = jnp.asarray(parent_local_id, dtype=jnp.int32)
        if child_local_id is not None and isinstance(child_local_id, int):
            child_local_id = jnp.asarray(child_local_id, dtype=jnp.int32)

        new_state, child_local_id = _clone_sequence(
            self,
            parent_local_id,
            child_local_id,
            seq_params=seq_params,
        )

        return new_state, child_local_id  # type: ignore


@functools.partial(jax.jit, donate_argnums=0)
def _clone_sequence(
    state,
    parent_local_id: jnp.ndarray,
    child_local_id: jnp.ndarray | None = None,
    *,
    seq_params: SeqDecodingParams | None = None,
) -> tuple["GenState", int]:

    page_table = state.decode_state.page_table
    if child_local_id is None:
        page_table, new_child = page_table.assign_seq_id_to_seq()
        child_local_id = eqx.error_if(
            new_child, ~is_valid(new_child), "No free local slots available to clone sequence."
        )
    else:
        page_table, assigned_id = page_table.assign_seq_id_to_seq(child_local_id)
        child_local_id = eqx.error_if(
            child_local_id, assigned_id != child_local_id, "Requested clone slot already in use."
        )
    # Important: assign_seq_id_to_seq donates its input; update decode_state.page_table immediately
    decode_state = dataclasses.replace(state.decode_state, page_table=page_table)

    # Assign child sequence state (copies tokens up to prefix and kv_pages row)
    decode_state = decode_state.assign_seq(
        local_slot_id=child_local_id,
        tokens=decode_state.tokens["seq", parent_local_id],
        seq_len=decode_state.seq_lens["seq", parent_local_id],
        kv_pages=decode_state.kv_pages["seq", parent_local_id],
        seq_params=seq_params,
    )
    # Record clone mapping on the child slot
    decode_state = dataclasses.replace(
        decode_state,
        clone_sources=decode_state.clone_sources.at["seq", child_local_id].set(parent_local_id),
    )

    page_table = page_table.clone_pages_from(parent_local_id, child_local_id)

    page_size = page_table.page_size
    src_len = page_table.seq_lens["seq", parent_local_id].scalar()

    def _copy(_):
        last_idx = (src_len + page_size - 1) // page_size - 1
        src_page = page_table.page_indices["seq", parent_local_id, "page", last_idx].scalar()
        dst_page = page_table.page_indices["seq", child_local_id, "page", last_idx].scalar()
        return state.cache.copy_page(src_page, dst_page)

    def _identity(_):
        return state.cache

    cache = jax.lax.cond((src_len % page_size != 0) & (src_len > 0), _copy, _identity, None)

    # persist the updated page table inside the decode state
    decode_state = dataclasses.replace(decode_state, page_table=page_table)
    new_state = dataclasses.replace(state, decode_state=decode_state, cache=cache)
    return new_state, child_local_id


class PrefillWork(eqx.Module):
    """Plain data container describing host-side work required for a prefill flush."""

    queue: TokenQueue
    new_num_seqs: jnp.ndarray
    new_slot_ids: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    clone_targets: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    prompt_tokens: ht.i32[NamedArray, "seq position"]  # type: ignore[name-defined]
    prompt_lengths: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    seq_params: SeqDecodingParams


def _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices):
    """
    Compute positions of last tokens per sequence inside a packed slice.

    Boundary when absolute pos_id equals the post-allocation seq_len - 1 for that sequence.
    """
    seq_lens_per_seq = seq_lens["seq", slot_ids]
    boundary_mask = pos_ids == (seq_lens_per_seq - 1)
    sample_indices = hax.where(
        boundary_mask,
        fill_value=INVALID,
        new_axis=pos_ids.resolve_axis("position").resize(max_sample_indices),
    )[0]
    return sample_indices


@hax.named_jit(donate_args=(True,))
def _release_finished_device(gen_state: GenState, finished_mask: ht.bool_[NamedArray, "seq"] | None = None):  # type: ignore
    """JIT-safe release of finished sequences on device.

    Frees pages in the PageTable for finished slots and invalidates those slots in DecodeState.
    If ``finished_mask`` is None, uses ``gen_state.decode_state.finished``.
    """
    ds = gen_state.decode_state
    pt = ds.page_table

    mask = finished_mask if finished_mask is not None else ds.finished
    new_pt = pt.free_pages_for_finished(mask.array)
    new_ds = ds.invalidate_finished()
    new_ds = dataclasses.replace(new_ds, page_table=new_pt)
    return dataclasses.replace(gen_state, decode_state=new_ds)


def _prefill_kernel(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    queue: TokenQueue,
    max_seqs_in_prefill: int,  # static
) -> tuple[GenState, _DecodeOutputs]:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    tokens = queue.queued_tokens
    pos_ids = queue.queued_pos_ids
    slot_ids = queue.queued_slot_ids
    seq_lens = gen_state.decode_state.seq_lens
    page_table, binfo = gen_state.decode_state.page_table.allocate_for_seq(
        token_slot_ids=slot_ids, token_pos_ids=pos_ids
    )

    sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_seqs_in_prefill)

    # jax.debug.print(
    #     "[_run_prefill] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
    #     tokens=tokens.array,
    #     slots=slot_ids.array,
    #     pos=pos_ids.array,
    #     lens=gen_state.decode_state.seq_lens.array,
    # )
    logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
    logits_at_samples = logits["position", sample_indices]

    num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
    # jax.debug.print(
    #     "[prefill] sample_count={num} queued_before={queued}",
    #     num=num_new_tokens,
    #     queued=gen_state.decode_state.num_queued_tokens,
    # )
    new_slot_ids = slot_ids["position", sample_indices]
    new_pos_ids = pos_ids["position", sample_indices]
    prng_keys = gen_state.decode_state.prng_keys_for(new_slot_ids, new_pos_ids)
    # jax.debug.print(
    #     "[_run_prefill] sample_indices={} new_slots={} new_pos={} prng_keys={}",
    #     sample_indices.array,
    #     new_slot_ids.array,
    #     new_pos_ids.array,
    #     prng_keys,
    # )

    temps = gen_state.decode_state.temperature["seq", new_slot_ids]

    new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

    # Update decode_state (also enqueues into the main decode queue)
    decode_state = gen_state.decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

    # Initialize outputs buffer and append prefill-sampled tokens
    outputs = _DecodeOutputs.init(
        max_tokens=gen_state.decode_state.max_seqs * 2,
        max_seqs=gen_state.decode_state.max_seqs,
        with_logprobs=True,
    )
    outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)

    decode_state = dataclasses.replace(decode_state, page_table=page_table)
    gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)

    # If clone targets specified, sample alternative tokens for clones using the same logits slice
    if decode_state.clone_sources is not None:
        gen_state, outputs = _handle_clones(
            gen_state,
            logits_at_samples,
            new_slot_ids,
            new_pos_ids,
            sampler,
            outputs,
        )

    # Device-side release of finished sequences (jit-safe)
    gen_state = _release_finished_device(gen_state)
    # jax.debug.print("[_run_prefill] output_tokens={} output_slots={}", outputs.tokens, outputs.slot_ids)

    # jax.debug.print(
    #     "[prefill] outputs_size={size} queued_after={queued}",
    #     size=outputs.num_tokens,
    #     queued=gen_state.decode_state.num_queued_tokens,
    # )
    return gen_state, outputs


def _stop_tokens_from_work(work: PrefillWork, idx: int) -> ht.i32[NamedArray, "stop_seq position"] | None:  # type: ignore[name-defined]
    stop_tokens = work.seq_params.stop_tokens
    if stop_tokens is None:
        return None
    return stop_tokens["seq", idx]


def _seq_params_from_work(work: PrefillWork, idx: int) -> SeqDecodingParams:
    def select(x):
        if isinstance(x, NamedArray):
            return x["seq", idx]
        elif is_jax_array_like(x):
            return x[idx]
        else:
            raise TypeError(f"Unexpected type in seq_params: {type(x)}")

    return hax.tree_util.tree_map(select, work.seq_params)


def _apply_prefill_work(gen_state: GenState, work: PrefillWork) -> GenState:
    num_new = work.new_num_seqs.astype(jnp.int32)
    max_slots = work.new_slot_ids.array.shape[0]

    def body(i: int, state: GenState) -> GenState:
        slot_val = work.new_slot_ids.array[i]

        def process(gs: GenState) -> GenState:
            parent_val = work.clone_targets.array[i]
            seq_params = _seq_params_from_work(work, i)

            def do_clone(gs_clone: GenState) -> GenState:
                new_state, _ = gs_clone.clone_sequence(
                    parent_val,
                    child_local_id=slot_val,
                    seq_params=seq_params,
                )
                return new_state

            def do_primary(gs_primary: GenState) -> GenState:
                decode_state = gs_primary.decode_state
                page_table, assigned = decode_state.page_table.assign_seq_id_to_seq(slot_val)
                assigned = eqx.error_if(
                    assigned,
                    assigned != slot_val,
                    "Requested local slot mismatch during prefill.",
                )
                decode_state = dataclasses.replace(decode_state, page_table=page_table)
                prompt_len = work.prompt_lengths.array[i].astype(jnp.int32)
                decode_state = decode_state.assign_seq(
                    local_slot_id=slot_val,
                    tokens=work.prompt_tokens["seq", i],
                    seq_len=prompt_len,
                    kv_pages=None,
                    seq_params=seq_params,
                )
                return dataclasses.replace(gs_primary, decode_state=decode_state)

            return jax.lax.cond(is_valid(parent_val), do_clone, do_primary, gs)

        should_process = (i < num_new) & is_valid(slot_val)
        return jax.lax.cond(should_process, process, lambda gs: gs, state)

    return jax.lax.fori_loop(0, max_slots, body, gen_state)


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("max_seqs_in_prefill",))
def _run_prefill(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    work: PrefillWork,
    max_seqs_in_prefill: int,
) -> tuple[GenState, _DecodeOutputs]:
    gen_state = _apply_prefill_work(gen_state, work)
    return _prefill_kernel(gen_state, model, sampler, work.queue, max_seqs_in_prefill)


def _handle_clones(
    gen_state: GenState,
    logits: ht.Float[NamedArray, " position vocab"],  # type: ignore
    slot_ids: ht.Int[NamedArray, " position"],  # type: ignore
    pos_ids: ht.Int[NamedArray, " position"],  # type: ignore
    sampler: Sampler,
    outputs: _DecodeOutputs,
) -> tuple[GenState, _DecodeOutputs]:  # type: ignore
    """
    Sample alternative tokens for the given logits, slot_ids, pos_ids, and clone_targets.
    This is used for the `n>1` case of `n_generations` in the `Request` class.

    Uses ``gen_state.decode_state.clone_sources`` as a mapping from target local ids to source local ids.

    It's assumed that:
      1. gen_state already has the appropriate page table and decode state.
      2. logits/slot_ids/pos_ids are already sliced

    Returns the updated gen_state and a boolean array indicating which ids from `clone_targets` were sampled.
    """
    # Resolve axes
    CloneSeq = gen_state.decode_state.clone_sources.resolve_axis("seq")

    # For each clone source, find its index in the provided slot_ids (within this packed/sliced batch).
    # If not present, mark as INVALID.
    def find_src(i):
        src = gen_state.decode_state.clone_sources["seq", i].scalar()

        def do(src):
            # match positions where slot_ids == src; take first
            eq = (slot_ids == src).array
            idx = jnp.nonzero(eq, size=1, fill_value=INVALID)[0][0]
            return idx

        return jax.lax.cond(is_valid(src), do, lambda x: x, src)

    # source_indices tells us, for each sequence that is a clone target, the index in the
    # logits/slot_ids/pos_ids arrays of its source sequence.
    # INVALID if either no source or source not in this batch.
    source_indices = hax.named(hax.vmap(find_src, "seq")(jnp.arange(CloneSeq.size)), axis="seq")

    # Determine which clone targets can be sampled this step:
    # need a valid source index and a valid target id
    can_sample = source_indices != INVALID

    # Build a compact position index list of clones to process this time
    selected = hax.where(can_sample, fill_value=INVALID, new_axis=CloneSeq)[0]
    selected = selected.rename({"seq": "position"})

    num_new = hax.sum(selected != INVALID).scalar().astype(jnp.int32)
    # jax.debug.print("[prefill clones] clone_count={num}", num=num_new)

    # Gather per-clone data
    # Use a masked/guarded gather to keep shapes static. First entries are valid clones.
    selected_safe = hax.where(selected != INVALID, selected, 0)
    tgt_ids = selected_safe
    src_pos = source_indices["seq", selected_safe]
    src_ids = slot_ids["position", src_pos]
    logits_this_time = logits["position", src_pos]
    pos_ids_this_time = pos_ids["position", src_pos]

    # Sample clones from the same boundary logits as their sources
    temps = gen_state.decode_state.temperature["seq", tgt_ids]
    prng_keys = gen_state.decode_state.prng_keys_for(tgt_ids, pos_ids_this_time)

    new_tokens, log_probs = hax.vmap(sampler, "position")(logits_this_time, temps, key=prng_keys)

    # update page table and cache for the clone targets
    page_table = gen_state.decode_state.page_table
    cache = gen_state.cache
    size = page_table.page_size

    def copy_pages_for_updated_seq(
        i,
        state: tuple[PageTable, KvPageCache],
    ) -> tuple[PageTable, KvPageCache]:
        page_table, cache = state
        src_slot_id = src_ids["position", i].scalar()
        dst_slot_id = tgt_ids["position", i].scalar()
        page_table = page_table.clone_pages_from(src_slot_id, dst_slot_id)

        src_len = page_table.seq_lens["seq", src_slot_id].scalar()
        used_pages = (src_len + size - 1) // size
        last_idx = jnp.maximum(used_pages - 1, 0)

        def _copy(_):
            src_page = page_table.page_indices["seq", src_slot_id, "page", last_idx].scalar()
            dst_page = page_table.page_indices["seq", dst_slot_id, "page", last_idx].scalar()
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
    # persist page table into decode state
    decode_state = dataclasses.replace(decode_state, page_table=page_table)
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state, cache=cache)

    # Append clone outputs
    outputs = outputs.append(new_tokens, tgt_ids, log_probs, num_new, gen_state.decode_state.finished)

    # Device-side release of finished sequences (jit-safe)
    gen_state = _release_finished_device(gen_state)

    return gen_state, outputs


# @hax.named_jit(donate_args=(True, False, False))
@functools.partial(jax.jit, static_argnums=(3, 4), donate_argnames=("gen_state",))
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
) -> tuple[GenState, _DecodeOutputs]:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached."""

    def cond(state: tuple[GenState, _DecodeOutputs, jax.Array]):
        _gen_state, _outputs, step = state
        return (
            (step < max_rounds)
            & (_gen_state.decode_state.num_queued_tokens > 0)
            & (~hax.all(_gen_state.decode_state.finished)).scalar()
        )

    def body(state: tuple[GenState, _DecodeOutputs, jax.Array]) -> tuple[GenState, _DecodeOutputs, jax.Array]:
        gen_state, outputs, step = state

        # Pack the next chunk from the queue via DecodeState
        decode_state, packed_seq = gen_state.decode_state.pack_next_sequence(max_tokens_per_round)

        tokens = packed_seq.tokens
        pos_ids = packed_seq.pos_ids
        slot_ids = packed_seq.slot_ids
        # NB: use decode_state.seq_lens to determine the number of tokens in each sequence, not what's in page table
        seq_lens = decode_state.seq_lens

        # jax.debug.print(
        #     "[_run_gen_loop] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
        #     tokens=tokens.array,
        #     slots=slot_ids.array,
        #     pos=pos_ids.array,
        #     lens=gen_state.decode_state.seq_lens.array,
        # )

        page_table, binfo = gen_state.decode_state.page_table.allocate_for_seq(
            token_slot_ids=slot_ids, token_pos_ids=pos_ids
        )

        max_sample_indices = min(page_table.max_seqs, max_tokens_per_round)
        sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices)

        # jax.debug.print("[_run_gen_loop] sample_indices={}", sample_indices.array)

        # Decode logits and sample new tokens
        logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
        logits_at_samples = logits["position", sample_indices]

        num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
        new_slot_ids = slot_ids["position", sample_indices]
        new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_slot_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)
        # jax.debug.print(
        #     "[gen] step={step} packed={packed} sample_count={num} queued_before={queued}",
        #     step=step,
        #     packed=packed_seq.num_tokens,
        #     num=num_new_tokens,
        #     queued=gen_state.decode_state.num_queued_tokens,
        # )

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

        # Update the gen_state with all the new components
        decode_state = dataclasses.replace(decode_state, page_table=page_table)
        new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
        # Append non-stateful outputs for host-side extraction
        outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)

        new_gen_state = _release_finished_device(new_gen_state)
        # jax.debug.print(
        #     "[gen] step={step} outputs_size={size} queued_after={queued}",
        #     step=step,
        #     size=outputs.num_tokens,
        #     queued=new_gen_state.decode_state.num_queued_tokens,
        # )
        return new_gen_state, outputs, step + 1

    # Allocate an outputs buffer sized for this run
    outputs_buf = _DecodeOutputs.init(
        max_tokens=max(max_tokens_per_round * max_rounds, 1),
        max_seqs=gen_state.decode_state.max_seqs,
        with_logprobs=True,
    )
    init_state = (gen_state, outputs_buf, jnp.array(0, dtype=jnp.int32))
    final_gen_state, final_outputs, _ = jax.lax.while_loop(cond, body, init_state)
    # jax.debug.print("[gen] final outputs_size={size}", size=final_outputs.num_tokens)
    return final_gen_state, final_outputs


@dataclass
class GenerationResult:
    tokens: list[list[int]]
    logprobs: list[list[float]] | None
    total_generated: int


class InferenceEngine:
    """Encapsulates batch inference: prefill + decode + output extraction.

    Typical usage:

        svc = Engine.from_model(model, tokenizer, Vocab, max_seqs, max_pages, page_size, max_pages_per_seq, compute_dtype)
        texts = svc.generate(requests)
    """

    def __init__(
        self,
        *,
        model: LmHeadModel,
        tokenizer,
        cache: KvPageCache,
        decode_state: DecodeState,
        sampler: Sampler,
        config: InferenceEngineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.gen_state: GenState = GenState(cache=cache, decode_state=decode_state)
        self._initial_decode_state = decode_state
        # Impute max_prefill_size if not set
        if config.max_prefill_size is None:
            config = dataclasses.replace(config, max_prefill_size=decode_state.page_table.max_len_per_seq)
        self.config = config
        # Track free local sequence slots as explicit ids (smallest id first to match allocator expectations).
        # Respect any pre-populated allocations in the provided PageTable.
        page_table = decode_state.page_table
        used_mask = np.asarray(jax.device_get(page_table.used_mask.array))
        free_slot_ids = [idx for idx, used in enumerate(used_mask) if not bool(used)]
        # Maintain free slots in ascending order to mirror PageTable's allocation policy.
        self.free_slots: list[int] = free_slot_ids
        # Mapping structures for active sequences
        # local_map: local slot id -> (request id, child id)
        # sequences: request id -> {child id -> local slot id}
        self.local_map: dict[int, tuple[int, int]] = {}
        self.sequences: dict[int, dict[int, int]] = {}
        # FIFO request queue for incremental admission
        self.request_queue: deque[Request] = deque()
        # Results by request id -> choice -> DecodeResult
        self.results: dict[int, dict[int, DecodeResult]] = {}

    def _verify_free_slot_view(self, *, context: str) -> None:
        """Ensure host free-list matches the device page-table used mask."""

        page_table = self.gen_state.decode_state.page_table
        used_mask = np.asarray(jax.device_get(page_table.used_mask.array)).astype(bool)
        free_set = set(self.free_slots)

        for slot_id, is_used in enumerate(used_mask):
            if is_used and slot_id in free_set:
                raise RuntimeError(
                    f"[free slot invariant] slot {slot_id} marked used but present in free list during {context}"
                )
            if not is_used and slot_id not in free_set:
                raise RuntimeError(
                    f"[free slot invariant] slot {slot_id} free in page table but missing from free list during {context}"
                )

    @classmethod
    def from_model(
        cls,
        model: LmHeadModel,
        tokenizer,
        *,
        max_pages: int,
        max_seqs: int,
        page_size: int,
        max_pages_per_seq: int,
        compute_dtype,
        max_queued_tokens: int = 32,
        max_seqs_in_prefill: int = 16,
        max_prefill_size: Optional[int] = None,
    ) -> "InferenceEngine":
        """Build an engine using basic sizing knobs. Uses defaults for stop-token capacity."""
        cfg = InferenceEngineConfig(
            max_pages=max_pages,
            max_seqs=max_seqs,
            page_size=page_size,
            max_pages_per_seq=max_pages_per_seq,
            compute_dtype=compute_dtype,
            max_queued_tokens=max_queued_tokens,
            max_seqs_in_prefill=max_seqs_in_prefill,
            max_prefill_size=max_prefill_size,
        )
        return cls.from_model_with_config(model=model, tokenizer=tokenizer, config=cfg)

    @classmethod
    def from_model_with_config(
        cls,
        model: LmHeadModel,
        tokenizer,
        config: InferenceEngineConfig,
    ) -> "InferenceEngine":
        """Build an engine using a EngineConfig for sizing knobs."""
        table = PageTable.init(config.imputed_max_pages, config.max_seqs, config.page_size, config.max_pages_per_seq)
        cache = hax.named_jit(model.initial_cache)(table, dtype=config.compute_dtype)
        decode_state = DecodeState.init(
            table,
            max_stop_seqs=config.max_stop_seqs,
            max_stop_tokens=config.max_stop_tokens,
            max_queued_tokens=config.max_queued_tokens,
            enable_logprobs=config.enable_logprobs,
        )
        vocab_axis = model.Vocab
        sampler = Sampler(vocab_axis)
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            decode_state=decode_state,
            sampler=sampler,
            config=config,
        )

    def reset(self) -> None:
        """Free all local sequence slots and reset to the initial `DecodeState`.

        Keeps the KV cache memory allocated. Reuses current `PageTable` object with pages freed.
        """
        page_table = self.gen_state.decode_state.page_table
        for slot_id in range(page_table.max_seqs):
            page_table = page_table.free_pages(slot_id)
        # persist into decode state
        new_decode_state = dataclasses.replace(self._initial_decode_state, page_table=page_table)
        self.gen_state = dataclasses.replace(self.gen_state, decode_state=new_decode_state)
        # All local slots are free after reset (ascending order to keep parents before clones)
        self.free_slots = list(range(int(page_table.max_seqs)))
        self.local_map.clear()
        self.sequences.clear()
        self._verify_free_slot_view(context="reset")

    def _release_finished_sequences(self, outputs: _DecodeOutputs) -> None:
        """Host-side bookkeeping for finished sequences.

        Device-side page freeing and decode-state invalidation occur inside the JIT loops.
        Here we update host maps and free-slot counters using the finished mask from outputs when provided.
        """
        finished_mask = jax.device_get(outputs.finished.array).astype(bool)
        finished_locals = [i for i, f in enumerate(finished_mask) if bool(f)]
        if not finished_locals:
            return
        logger.info(f"Releasing finished sequences: locals={finished_locals}")
        # Maintain request/child mappings and slot counts on host
        for local_slot in finished_locals:
            info = self.local_map.pop(local_slot, None)
            if info is not None:
                rid, cid = info
                cmap = self.sequences.get(rid)
                if cmap is not None:
                    cmap.pop(cid, None)
                    if len(cmap) == 0:
                        self.sequences.pop(rid, None)
            # Ensure any residual tokens/logprobs for this slot are dropped from the pending queue
            self.free_slots.append(local_slot)

    # ------------------------------- Queue helpers -------------------------------
    def enqueue_requests(self, requests: Sequence[Request]) -> None:
        for r in requests:
            self.request_queue.append(r)

    def _admit_from_queue(self) -> _DecodeOutputs | None:
        """Admit a batch from the head of the queue that fits in free slots/pages.

        Returns the decode outputs for the admitted prefill batch, or None if no work was admitted.
        """
        if not self.request_queue:
            return None

        sim_slots = len(self.free_slots)
        sim_pages = self._free_page_count()
        max_prefill_size = int(self.config.max_prefill_size or self.gen_state.decode_state.page_table.max_len_per_seq)
        max_seqs_in_prefill = int(self.config.max_seqs_in_prefill)
        sim_tokens = 0
        primaries_in_batch = 0

        batch: list[Request] = []
        while self.request_queue:
            nxt = self.request_queue[0]
            need_slots = int(nxt.n_generations)
            need_pages = self._pages_needed_for_prompt(len(nxt.prompt_tokens))
            # Check capacity constraints: slots (including clones), pages, token buffer, prefill batch size
            if (
                sim_slots < need_slots
                or sim_pages < need_pages
                or sim_tokens + len(nxt.prompt_tokens) > max_prefill_size
                or primaries_in_batch >= max_seqs_in_prefill
            ):
                break
            batch.append(self.request_queue.popleft())
            sim_slots -= need_slots
            sim_pages -= need_pages
            sim_tokens += len(nxt.prompt_tokens)
            primaries_in_batch += 1

        if not batch:
            return None

        # Build a single PrefillWork description and run prefill exactly once
        prefill_work = self._prefill_prompts(batch)
        if prefill_work is None:
            return None
        new_state = _run_prefill(
            self.gen_state, self.model, self.sampler, prefill_work, self.config.max_seqs_in_prefill
        )

        # _run_prefill returns (GenState, _DecodeOutputs)
        self.gen_state, outputs = new_state
        return outputs

    def _free_page_count(self) -> int:
        """Return number of free KV pages in the PageTable."""
        prc = jax.device_get(self.gen_state.decode_state.page_table.page_ref_counts.array)
        return int((prc == 0).sum())

    def _pages_needed_for_prompt(self, prompt_len: int) -> int:
        size = int(self.gen_state.decode_state.page_table.page_size)
        return (int(prompt_len) + size - 1) // size

    def _prefill_prompts(
        self,
        requests: Sequence[Request],
    ) -> PrefillWork | None:
        """Pack prompt work into a single PrefillWork structure for downstream device execution."""

        decode_state = self.gen_state.decode_state
        max_seqs_in_prefill = self.config.max_seqs_in_prefill
        max_prefill_size = self.config.max_prefill_size or self.model.Pos.size
        max_seq_len = decode_state.tokens.axis_size("position")
        max_slots = decode_state.max_seqs

        queue_tokens = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)
        queue_slot_ids = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)
        queue_pos_ids = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)

        work_slot_ids = np.full((max_slots,), INVALID, dtype=np.int32)
        clone_targets = np.full((max_slots,), INVALID, dtype=np.int32)
        prompt_tokens = np.full((max_slots, max_seq_len), INVALID, dtype=np.int32)
        prompt_lengths = np.zeros((max_slots,), dtype=np.int32)

        stop_tokens_template = decode_state.stop_tokens
        max_num_tokens = np.zeros((max_slots,), dtype=np.int32)
        temperatures = np.zeros((max_slots,), dtype=np.float32)
        prng_keys = np.zeros((max_slots, 2), dtype=np.uint32)
        if stop_tokens_template is not None:
            stop_tokens = np.full(
                (
                    max_slots,
                    stop_tokens_template.axis_size("stop_seq"),
                    stop_tokens_template.axis_size("position"),
                ),
                INVALID,
                dtype=np.int32,
            )
        else:
            stop_tokens = None

        offset = 0
        num_primary = 0
        total_new = 0

        for request in requests:
            seq_tokens = request.prompt_tokens
            seq_params = request.decode_params

            if len(seq_tokens) + offset > queue_tokens.shape[0] or num_primary >= max_seqs_in_prefill:
                break

            if len(self.free_slots) < request.n_generations:
                if max_seqs_in_prefill < request.n_generations:
                    raise RuntimeError(
                        f"Request {request.request_id} asked for {request.n_generations} generations, "
                        f"but max_seqs_in_prefill={max_seqs_in_prefill} is too small to accommodate. "
                        "Increase max_seqs_in_prefill or reduce n_generations."
                    )
                break

            requested_slot = self.free_slots.pop()
            slot_id = int(requested_slot)

            this_tokens = np.asarray(seq_tokens, dtype=np.int32)
            queue_tokens[offset : offset + len(seq_tokens)] = this_tokens
            queue_slot_ids[offset : offset + len(seq_tokens)] = slot_id
            queue_pos_ids[offset : offset + len(seq_tokens)] = np.arange(len(seq_tokens), dtype=np.int32)

            prefill_idx = total_new
            if prefill_idx >= max_slots:
                raise RuntimeError("Exceeded maximum slot instructions while building prefill work.")

            work_slot_ids[prefill_idx] = slot_id
            clone_targets[prefill_idx] = INVALID
            prompt_lengths[prefill_idx] = len(seq_tokens)
            prompt_tokens[prefill_idx, : len(seq_tokens)] = this_tokens

            max_num_tokens[prefill_idx] = np.asarray(seq_params.max_num_tokens, dtype=np.int32).item()
            temperatures[prefill_idx] = np.asarray(seq_params.temperature, dtype=np.float32).item()
            prng_keys[prefill_idx] = np.asarray(seq_params.key, dtype=np.uint32)
            if stop_tokens is not None:
                if seq_params.stop_tokens is None:
                    stop_tokens[prefill_idx].fill(INVALID)
                else:
                    row = stop_tokens[prefill_idx]
                    row.fill(INVALID)
                    seq_stop = np.asarray(seq_params.stop_tokens.array)
                    seq_num_stops, seq_stop_len = seq_stop.shape
                    row[:seq_num_stops, -seq_stop_len:] = seq_stop

            rid = int(request.request_id)
            self.local_map[slot_id] = (rid, 0)
            self.sequences.setdefault(rid, {})[0] = slot_id

            offset += len(seq_tokens)
            num_primary += 1
            total_new += 1

            if request.n_generations > 1:
                parent_length = len(seq_tokens)
                for k in range(1, request.n_generations):
                    if not self.free_slots:
                        raise RuntimeError("Clone requested but no free local slots remained.")

                    requested_child_slot = self.free_slots.pop()
                    child_slot_id = int(requested_child_slot)
                    clone_idx = total_new
                    if clone_idx >= max_slots:
                        raise RuntimeError("Exceeded maximum slot instructions while adding clones.")

                    child_params = dataclasses.replace(seq_params, key=jax.random.fold_in(seq_params.key, k))

                    work_slot_ids[clone_idx] = child_slot_id
                    clone_targets[clone_idx] = slot_id
                    prompt_lengths[clone_idx] = parent_length
                    # Clones reuse prompt tokens from their parent; no need to copy here.
                    max_num_tokens[clone_idx] = np.asarray(child_params.max_num_tokens, dtype=np.int32).item()
                    temperatures[clone_idx] = np.asarray(child_params.temperature, dtype=np.float32).item()
                    prng_keys[clone_idx] = np.asarray(child_params.key, dtype=np.uint32)
                    if stop_tokens is not None:
                        stop_tokens[clone_idx] = stop_tokens[prefill_idx]

                    self.local_map[child_slot_id] = (rid, k)
                    self.sequences.setdefault(rid, {})[k] = child_slot_id

                    total_new += 1

        if offset == 0:
            return None

        prefill_queue = TokenQueue(
            queued_tokens=hax.named(jnp.asarray(queue_tokens, dtype=jnp.int32), axis="position"),
            queued_slot_ids=hax.named(jnp.asarray(queue_slot_ids, dtype=jnp.int32), axis="position"),
            queued_pos_ids=hax.named(jnp.asarray(queue_pos_ids, dtype=jnp.int32), axis="position"),
            num_queued_tokens=jnp.array(offset, dtype=jnp.int32),
        )

        return PrefillWork(
            queue=prefill_queue,
            new_num_seqs=jnp.array(total_new, dtype=jnp.int32),
            new_slot_ids=hax.named(jnp.asarray(work_slot_ids, dtype=jnp.int32), axis="seq"),
            clone_targets=hax.named(jnp.asarray(clone_targets, dtype=jnp.int32), axis="seq"),
            prompt_tokens=hax.named(jnp.asarray(prompt_tokens, dtype=jnp.int32), axis=("seq", "position")),
            prompt_lengths=hax.named(jnp.asarray(prompt_lengths, dtype=jnp.int32), axis="seq"),
            seq_params=SeqDecodingParams(
                max_num_tokens=jnp.asarray(max_num_tokens, dtype=jnp.int32),
                stop_tokens=(
                    None
                    if stop_tokens is None
                    else hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("seq", "stop_seq", "position"))
                ),
                temperature=jnp.asarray(temperatures, dtype=jnp.float32),
                key=jnp.asarray(prng_keys, dtype=jnp.uint32),
            ),
        )

    def generate(self, requests: Sequence[Request]) -> GenerationResult:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).
        """
        # validate we don't have any sequences with n_generations exceeding max_seqs
        max_needed = max(int(r.n_generations) for r in requests)
        if max_needed > int(self.gen_state.decode_state.page_table.max_seqs):
            raise ValueError(
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.gen_state.decode_state.page_table.max_seqs})."
                "Decompose your request into smaller batches or increase max_seqs when building the service."
            )

        needs_logprobs = any(r.enable_logprobs for r in requests)
        # if we need logprobs but decode state doesn't have logprobs enabled, re-init
        if needs_logprobs and self.gen_state.decode_state.logprobs is None:
            logger.info("Re-initializing decode state with logprobs enabled.")
            max_seqs = int(self.gen_state.decode_state.max_seqs)
            max_seq_len = int(self.gen_state.decode_state.page_table.max_len_per_seq)
            new_decode_state = dataclasses.replace(
                self.gen_state.decode_state,
                logprobs=hax.full({"seq": max_seqs, "position": max_seq_len}, jnp.nan, dtype=jnp.float32),
            )
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=new_decode_state)

        # Enqueue incoming requests to internal queue
        self.enqueue_requests(requests)
        # Track outputs and finished flags using self.results for only this call's requests
        call_rids = [int(r.request_id) for r in requests]
        expected_children: dict[int, int] = {rid: int(r.n_generations) for rid, r in zip(call_rids, requests)}
        # Initialize fresh result buckets for this call
        for rid in call_rids:
            self.results[rid] = {
                k: DecodeResult(id=rid, choice=k, token_list=[]) for k in range(expected_children[rid])
            }

        # Validate requested stop-token shapes against configured capacity; do not resize dynamically
        ds = self.gen_state.decode_state
        cur_stop_seqs = 0 if ds.stop_tokens is None else ds.stop_tokens.axis_size("stop_seq")
        cur_stop_len = 0 if ds.stop_tokens is None else ds.stop_tokens.axis_size("position")
        req_stop_seqs = 0
        req_stop_len = 0
        for req in requests:
            st = req.decode_params.stop_tokens
            if st is None:
                continue
            req_stop_seqs = max(req_stop_seqs, int(st.axis_size("stop_seq")))
            req_stop_len = max(req_stop_len, int(st.axis_size("position")))
        if req_stop_seqs > 0 or req_stop_len > 0:
            if ds.stop_tokens is None:
                raise ValueError(
                    f"Requested stop tokens (seqs={req_stop_seqs}, len={req_stop_len}) but service was initialized "
                    f"without stop-token capacity. Recreate service with nonzero max_stop_seqs/max_stop_tokens."
                )
            if req_stop_seqs > cur_stop_seqs or req_stop_len > cur_stop_len:
                raise ValueError(
                    "Requested stop-token configuration exceeds service capacity: "
                    f"required (seqs={req_stop_seqs}, len={req_stop_len}) > "
                    f"configured (seqs={cur_stop_seqs}, len={cur_stop_len}). "
                    "Increase max_stop_seqs/max_stop_tokens when constructing the service."
                )

        time_in = time.time()
        # Try initial admission from queue and extract prompt tokens
        decode_outputs = self._admit_from_queue()
        if decode_outputs:
            _ = self._ingest_outputs(decode_outputs)
        initial_prefill_out = time.time()
        logger.info(f"Initial prefill and extraction took {initial_prefill_out - time_in:.3f}s")

        # Autoregressive generation loop with periodic extraction
        def _all_done() -> bool:
            for rid, n_kids in expected_children.items():
                kid_map = self.results.get(rid, {})
                for cid in range(n_kids):
                    dr = kid_map.get(cid)
                    if dr is None or not dr.done:
                        return False
            return True

        stagnant_iters = 0
        while not _all_done():
            iter_start = time.time()

            fake_submit_start = time.time()
            # future_state, decode_outputs = _run_generation_loop(
            jax.tree.flatten(
                (
                    self.gen_state,
                    self.model,
                    self.sampler,
                    1,
                    0,
                )
            )
            fake_submit_done = time.time()

            submit_start = iter_start
            future_state, decode_outputs = _run_generation_loop(
                self.gen_state,
                self.model,
                self.sampler,
                # TODO: tune max_tokens_per_round
                self.config.max_tokens_per_round or self.config.max_seqs,
                self.config.max_rounds,
            )
            submit_done = time.time()
            # Time spent with device executing (and the host thread waiting)
            self.gen_state = future_state
            device_time = time.time() - submit_done

            extract_start = time.time()
            new_tokens = self._ingest_outputs(decode_outputs)
            extract_time = time.time() - extract_start

            # Release any sequences that finished in this step
            release_start = time.time()
            # Admit more if capacity allows
            admit_outputs = self._admit_from_queue()
            if admit_outputs is not None:
                mid_tokens = self._ingest_outputs(admit_outputs)
            else:
                mid_tokens = 0
            new_tokens += mid_tokens
            release_time = time.time() - release_start

            iter_end = time.time()
            iter_time = iter_end - iter_start
            # Host time is everything except the device execution wait
            host_time = max(iter_time - device_time, 0.0)
            submit_time = submit_done - submit_start
            if iter_time > 0:
                tps_total = new_tokens / iter_time
                logger.info(
                    f"Decode iter: total {iter_time:.3f}s (device {device_time:.3f}s, host {host_time:.3f}s, "
                    f"submit {submit_time:.3f}s), "
                    f"fake_submit {fake_submit_done - fake_submit_start:.3f}s, "
                    f"{tps_total:.2f} tok/s, {new_tokens} new"
                    f" (extract {extract_time:.3f}s, release {release_time:.3f}s)"
                )
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
        logprobs_list: list[list[float]] = []
        total_prompt_tokens = 0
        for r in requests:
            rid = int(r.request_id)
            total_prompt_tokens += len(r.prompt_tokens) * int(r.n_generations)
            # Initialize result buckets for this rid if not present
            kid_map = self.results.get(rid, {})
            for k in range(int(r.n_generations)):
                dr = kid_map.get(k)
                if dr is None:
                    # Ensure a placeholder exists to avoid KeyErrors
                    kid_map[k] = DecodeResult(id=rid, choice=k, token_list=[])
                    dr = kid_map[k]
                outputs_list.append(dr.token_list)
                logprobs_list.append(dr.logprobs if dr.logprobs is not None else [])
            self.results[rid] = kid_map
        total_generated = sum(len(seq_outputs) for seq_outputs in outputs_list)
        total_time = time.time() - time_in
        tps_overall = (total_generated / total_time) if total_time > 0 else 0.0
        logger.info(f"Batch generated in {total_time:.2f}s, {total_generated} tokens, {tps_overall:.2f} tok/s")
        # Clear results for these requests now that we've assembled outputs
        for rid in call_rids:
            if rid in self.results:
                self.results.pop(rid, None)
        return GenerationResult(tokens=outputs_list, logprobs=logprobs_list, total_generated=total_generated)

    def _extract_outputs(self, pending_outputs) -> int:
        """Append newly available tokens into outputs per (request_id, child_id).

        Returns number of new tokens appended.
        """
        if pending_outputs is None:
            return 0

        # Pull the entire buffer in one host op
        pending_outputs = jax.device_get(pending_outputs)
        n = int(pending_outputs.num_tokens)
        fins = pending_outputs.finished.array
        toks_arr = pending_outputs.tokens.array
        sids_arr = pending_outputs.slot_ids.array

        appended = 0
        unmapped = 0
        for i in range(n):
            local_slot = int(sids_arr[i])
            tok = int(toks_arr[i])
            info = self.local_map.get(local_slot)
            if info is None:
                unmapped += 1
                continue
            rid, cid = info
            dr = self.results.setdefault(rid, {}).setdefault(cid, DecodeResult(id=rid, choice=cid, token_list=[]))
            dr.token_list.append(tok)
            if pending_outputs.logprobs is not None:
                dr.logprobs.append(float(pending_outputs.logprobs.array[i]))
            dr.tokens_decoded += 1
            appended += 1

        # Update done flags based on snapshot
        for local_slot, is_done in enumerate(fins):
            if not bool(is_done):
                continue
            info = self.local_map.get(local_slot)
            if info is None:
                continue
            rid, cid = info
            dr = self.results.setdefault(rid, {}).setdefault(cid, DecodeResult(id=rid, choice=cid, token_list=[]))
            dr.done = True

        num_finished = int(fins.sum()) if hasattr(fins, "sum") else 0
        logger.info(f"extract: appended={appended} (drained={n}) unmapped={unmapped} finished_count={num_finished}")

        return appended

    def _ingest_outputs(self, outputs: _DecodeOutputs | None) -> int:
        """Drain device outputs into host results and apply host-side release.

        Returns the number of tokens appended to results. No-op if outputs is None.
        """
        if outputs is None:
            return 0
        appended = self._extract_outputs(outputs)
        self._release_finished_sequences(outputs)
        return appended
