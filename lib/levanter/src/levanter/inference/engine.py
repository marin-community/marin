# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import equinox as eqx
import fsspec
import haliax as hax
import haliax.haxtyping as ht
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from haliax.jax_utils import is_jax_array_like

import levanter.tracker
from levanter.inference.jit_scheduler import (
    DecodeState,
    SeqDecodingParams,
    TokenQueue,
    _DecodeOutputs,
)
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID, is_valid
from levanter.layers.kv_cache import PageCache
from levanter.layers.sampler import Sampler
from levanter.models.lm_model import LmHeadModel
from levanter.utils.jax_utils import estimated_free_device_memory, sharded_tree_size

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceEngineConfig:
    """Configuration for Engine memory/layout knobs.

    Exposes key buffer sizes and limits controlling prefill, decode queueing, and page table capacity.
    """

    max_seq_len: int
    """
    Maximum sequence length (including prompt). Used for validation and buffer sizing at init.
    """

    hbm_utilization: float = 0.9
    """Fraction of device HBM to reserve for the KV cache when :attr:`max_pages` is ``None``."""

    page_size: int = 128
    """Tokens per KV page."""

    max_rounds: int = 32
    """Maximum number of while-loop iterations per decode call. Higher values increase throughput but also latency."""

    # Stop-token capacity (used for validation and buffer sizing at init)
    max_stop_seqs: int = 4
    """Maximum number of stop sequences per active sequence. 0 disables stop tokens."""
    max_stop_tokens: int = 16
    """Maximum tokens per stop sequence (position axis length)."""

    # Default PRNG seed for building per-request keys (optional convenience)
    seed: int = 0

    # You probably don't need to tune the knobs below

    max_seqs: int = 256
    """Maximum concurrent sequences (local slots)."""

    max_pages: Optional[int] = None
    """Total number of KV pages available. If None, inferred from :attr:`hbm_utilization`."""

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
    """Pack size for each decode loop iteration. If None, set to max_seqs """

    # Device override for multi-host inference
    devices: Optional[Sequence] = None
    """Devices to use for inference. If None, uses jax.devices(). For multi-host inference,
    pass jax.local_devices() to run inference on a single host's devices only."""

    # Memory management options
    incremental_cleanup: bool = False
    """When True, call free_pages_for_finished() after each decode round to incrementally
    release pages for finished sequences. Default False to preserve existing behavior."""

    use_logical_reset: bool = False
    """When True, skip physical zeroing of KV cache during reset. The attention mask
    ensures stale data is never read. Default False to preserve existing behavior."""

    def __post_init__(self):
        # this one is only required because of clones. If we really care, we could relax this
        if self.max_queued_tokens < self.max_seqs:
            raise ValueError("max_queued_tokens must be >= max_seqs")

        if self.max_queued_tokens < self.imputed_max_tokens_per_round:
            raise ValueError("max_queued_tokens must be >= max_tokens_per_round")

        if self.max_queued_tokens < self.max_seqs_in_prefill:
            raise ValueError("max_queued_tokens must be >= max_seqs_in_prefill")

    @property
    def imputed_max_tokens_per_round(self) -> int:
        """Return explicit `max_tokens_per_round` or default to `max_seqs` when unset."""
        return self.max_tokens_per_round if self.max_tokens_per_round is not None else self.max_seqs

    @property
    def max_pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size


def _tree_byte_size(tree) -> int:
    """Return the per-device number of bytes represented by ``tree``."""

    return sharded_tree_size(tree)


def _available_hbm_budget_bytes(hbm_utilization: float, devices: Sequence | None = None) -> int:
    """Estimate the per-device HBM budget available to the KV cache.

    Args:
        hbm_utilization: Fraction of HBM to use (0, 1].
        devices: Devices to consider. If None, uses jax.devices().
    """

    if not (0.0 < hbm_utilization <= 1.0):
        raise ValueError("hbm_utilization must be in the interval (0, 1].")

    if devices is None:
        devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available for inference.")

    budgets: list[int] = []
    bytes_per_gib = 1024**3
    for device in devices:
        free_gib = estimated_free_device_memory(device)
        if free_gib is None:
            raise RuntimeError(f"Device {device} does not expose memory statistics.")
        free_bytes = max(int(free_gib * bytes_per_gib), 0)
        budgets.append(int(free_bytes * hbm_utilization))

    if not budgets:
        raise RuntimeError("Unable to determine device HBM budget.")

    return min(budgets)


def _infer_max_pages_from_hbm(model: LmHeadModel, config: InferenceEngineConfig) -> int:
    """Infer a KV-page budget using HBM utilization targets."""

    max_pages_per_seq = config.max_pages_per_seq

    try:
        budget = _available_hbm_budget_bytes(config.hbm_utilization, devices=config.devices)
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        logger.warning(
            "Falling back to max_seqs * max_pages_per_seq for KV cache sizing because HBM budget "
            "could not be determined: %s",
            exc,
        )
        return int(config.max_seqs * max_pages_per_seq)

    @functools.lru_cache(maxsize=None)
    def cache_bytes(num_pages: int) -> int:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive when sizing the KV cache.")

        def initial_cache(pages: int) -> int:
            table = PageTable.init(pages, config.max_seqs, config.page_size, max_pages_per_seq)
            cache_shape = model.initial_cache(table.spec(), dtype=config.compute_dtype)
            return cache_shape

        cache_shape = eqx.filter_eval_shape(initial_cache, num_pages)

        return _tree_byte_size(cache_shape)

    bytes_one = cache_bytes(1)
    if bytes_one > budget:
        raise ValueError(
            "HBM budget insufficient to allocate even a single KV cache page. "
            "Provide `max_pages` explicitly or increase `hbm_utilization`."
        )

    # Use the previous heuristic as the initial guess before expanding.
    guess = max(int(config.max_seqs * max_pages_per_seq), 1)

    low = 1
    high = guess
    high_bytes = cache_bytes(high)

    if high_bytes <= budget:
        low = high
        while True:
            high *= 2
            if high > (1 << 20):
                warnings.warn(
                    "KV cache size exceeded 1M pages during budget inference; "
                    "aborting search and using current estimate."
                )
                high = 1 << 20
                break
            high_bytes = cache_bytes(high)
            if high_bytes > budget:
                break
            low = high

    # Binary search between the known-good lower bound and the first oversized bound.
    while low + 1 < high:
        mid = (low + high) // 2
        mid_bytes = cache_bytes(mid)
        if mid_bytes <= budget:
            low = mid
        else:
            high = mid

    max_pages = low

    bytes_at_max = cache_bytes(max_pages)
    next_bytes = cache_bytes(high)
    per_page = bytes_at_max if max_pages == 1 else bytes_at_max - cache_bytes(max_pages - 1)
    base_bytes = max(bytes_at_max - per_page * max_pages, 0)

    import humanfriendly as hly

    logger.info(
        "Auto-computed KV cache budget: base=%s, per_page=%s, budget=%s, used=%s, next=%s -> max_pages=%d",
        hly.format_size(base_bytes),
        hly.format_size(per_page),
        hly.format_size(budget),
        hly.format_size(bytes_at_max),
        hly.format_size(next_bytes),
        max_pages,
    )

    return max_pages


@dataclass(frozen=True)
class Request:
    """A request for generation of a single sequence."""

    prompt_tokens: list[int]
    request_id: int
    decode_params: SeqDecodingParams
    n_generations: int


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


class GenState(eqx.Module):
    """Container for generation state used during decoding.

    Holds the KV cache and `DecodeState` (which itself owns the `PageTable`).
    Provides `clone_sequence` to efficiently support multi-sample generation by
    sharing fully used pages.
    """

    cache: PageCache
    decode_state: DecodeState

    def reset(self):
        """Reset with physical cache zeroing."""
        return GenState(
            cache=self.cache.reset(),
            decode_state=self.decode_state.reset(),
        )

    def reset_logical(self):
        """Reset without physical cache zeroing.

        Uses logical reset which skips the expensive zeroing of KV pages.
        This is safe because attention masks by kv_len and new allocations
        overwrite stale data before it can be used.
        """
        return GenState(
            cache=self.cache.reset_logical(),
            decode_state=self.decode_state.reset(),
        )

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
    decode_state = state.decode_state
    if child_local_id is None:
        decode_state, new_child = decode_state.reserve_slot()
        child_local_id = eqx.error_if(
            new_child, ~is_valid(new_child), "No free local slots available to clone sequence."
        )
    else:
        decode_state, assigned_id = decode_state.reserve_slot(child_local_id)
        child_local_id = eqx.error_if(
            child_local_id, assigned_id != child_local_id, "Requested clone slot already in use."
        )

    # Assign child sequence state (copies tokens up to prefix and kv_pages row)
    parent_kv_pages = decode_state.kv_pages["seq", parent_local_id]
    parent_page_indices = decode_state.sequences.page_indices["seq", parent_local_id]
    decode_state = decode_state.assign_seq(
        local_slot_id=child_local_id,
        tokens=decode_state.tokens["seq", parent_local_id],
        seq_len=decode_state.seq_lens["seq", parent_local_id],
        kv_pages=parent_kv_pages,
        page_indices=parent_page_indices,
        seq_params=seq_params,
    )
    # Record clone mapping on the child slot
    new_sequences = decode_state.sequences.set_clone_source(child_local_id, parent_local_id)
    decode_state = dataclasses.replace(decode_state, sequences=new_sequences)

    decode_state = decode_state.clone_pages_from(parent_local_id, child_local_id)

    page_size = decode_state.page_table.page_size
    src_len = decode_state.seq_lens["seq", parent_local_id].scalar()

    def _copy(_):
        last_idx = (src_len + page_size - 1) // page_size - 1
        src_page = decode_state.sequences.page_indices["seq", parent_local_id, "page", last_idx].scalar()
        dst_page = decode_state.sequences.page_indices["seq", child_local_id, "page", last_idx].scalar()
        return state.cache.copy_page(src_page, dst_page)

    def _identity(_):
        return state.cache

    cache = jax.lax.cond((src_len % page_size != 0) & (src_len > 0), _copy, _identity, None)

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
    # jax.debug.print("pos_ids={pos} seq_lens={lens} boundary={b}", pos=pos_ids.array, lens=seq_lens_per_seq.array, b=boundary_mask.array)
    sample_indices = hax.where(
        boundary_mask,
        fill_value=INVALID,
        new_axis=pos_ids.resolve_axis("position").resize(max_sample_indices),
    )[0]
    return sample_indices


@functools.partial(hax.named_jit, static_argnums=(4,))
def _prefill_kernel(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    queue: TokenQueue,
    max_seqs_in_prefill: int,  # static
) -> tuple[GenState, _DecodeOutputs]:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    jax.debug.print("[_prefill_kernel] === ENTERED prefill_kernel ===")

    tokens = queue.queued_tokens
    pos_ids = queue.queued_pos_ids
    slot_ids = queue.queued_slot_ids

    jax.debug.print("[_prefill_kernel] === got tokens/pos/slots ===")

    decode_state, binfo = gen_state.decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)

    jax.debug.print("[_prefill_kernel] === allocate_for_seq done ===")

    seq_lens = decode_state.seq_lens

    sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_seqs_in_prefill)

    jax.debug.print("[_prefill_kernel] === about to call model.decode ===")
    logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
    jax.debug.print("[_prefill_kernel] === model.decode returned ===")
    jax.debug.print("[_prefill_kernel] logits_shape={shape}", shape=logits.array.shape)
    logits_at_samples = logits["position", sample_indices]

    num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
    jax.debug.print("[_prefill_kernel] === num_new_tokens={num} ===", num=num_new_tokens)

    new_slot_ids = slot_ids["position", sample_indices]
    new_pos_ids = pos_ids["position", sample_indices]
    prng_keys = gen_state.decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

    jax.debug.print("[_prefill_kernel] === about to sample ===")

    temps = decode_state.temperature["seq", new_slot_ids]

    new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

    jax.debug.print("[_prefill_kernel] === sampling done ===")

    # Update decode_state (also enqueues into the main decode queue)
    decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

    jax.debug.print("[_prefill_kernel] === update_tokens done ===")

    # Initialize outputs buffer and append prefill-sampled tokens
    outputs = _DecodeOutputs.init(
        max_tokens=decode_state.max_seqs * 2,
        max_seqs=decode_state.max_seqs,
        with_logprobs=True,
    )
    outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finished)
    gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)

    jax.debug.print("[_prefill_kernel] === outputs created ===")

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

    jax.debug.print("[_prefill_kernel] === returning from prefill_kernel ===")

    return gen_state, outputs


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
                decode_state, assigned = decode_state.reserve_slot(slot_val)
                # Get the prompt length for this sequence
                prompt_len = work.prompt_lengths.array[i].astype(jnp.int32)
                decode_state = decode_state.assign_seq(
                    local_slot_id=slot_val,
                    tokens=work.prompt_tokens["seq", i],
                    seq_len=prompt_len,
                    kv_pages=None,  # Will be allocated later in allocate_for_seq
                    page_indices=None,  # Will be set during page allocation
                    seq_params=seq_params,
                )
                return dataclasses.replace(gs_primary, decode_state=decode_state)

            return jax.lax.cond(is_valid(parent_val), do_clone, do_primary, gs)

        should_process = (i < num_new) & is_valid(slot_val)
        return jax.lax.cond(should_process, process, lambda gs: gs, state)

    return jax.lax.fori_loop(0, max_slots, body, gen_state)


def _run_prefill(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    work: PrefillWork,
    max_seqs_in_prefill: int,
) -> tuple[GenState, _DecodeOutputs]:
    """Run prefill by applying work and then calling the module-level _prefill_kernel JIT."""
    import sys

    print(f"[_run_prefill P{jax.process_index()}] === ENTERED ===", flush=True)
    sys.stdout.flush()

    # Step 1: Apply prefill work (updates gen_state with new sequences)
    # This needs to run first to set up the decode_state with the new sequences
    print(f"[_run_prefill P{jax.process_index()}] === Step 1: _apply_prefill_work ===", flush=True)
    sys.stdout.flush()

    @hax.named_jit
    def _jit_apply_prefill_work(gs, w):
        return _apply_prefill_work(gs, w)

    gen_state = _jit_apply_prefill_work(gen_state, work)
    print(f"[_run_prefill P{jax.process_index()}] === Step 1 DONE ===", flush=True)
    sys.stdout.flush()

    # Step 2: Call the module-level _prefill_kernel JIT (now decorated at module level)
    print(f"[_run_prefill P{jax.process_index()}] === Step 2: _prefill_kernel (module-level JIT) ===", flush=True)
    sys.stdout.flush()

    result = _prefill_kernel(gen_state, model, sampler, work.queue, max_seqs_in_prefill)

    print(f"[_run_prefill P{jax.process_index()}] === DONE ===", flush=True)
    sys.stdout.flush()
    return result


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
    decode_state = gen_state.decode_state
    cache = gen_state.cache
    size = decode_state.sequences.page_size

    def copy_pages_for_updated_seq(
        i,
        state: tuple[DecodeState, PageCache],
    ) -> tuple[DecodeState, PageCache]:
        decode_state, cache = state
        src_slot_id = src_ids["position", i].scalar()
        dst_slot_id = tgt_ids["position", i].scalar()

        src_len = decode_state.seq_lens["seq", dst_slot_id].scalar()
        used_pages = (src_len + size - 1) // size
        last_idx = jnp.maximum(used_pages - 1, 0)

        def _copy(_):
            src_page = decode_state.sequences.page_indices["seq", src_slot_id, "page", last_idx].scalar()
            dst_page = decode_state.sequences.page_indices["seq", dst_slot_id, "page", last_idx].scalar()
            return cache.copy_page(src_page, dst_page)

        def _identity(_):
            return cache

        cache = jax.lax.cond((src_len % size != 0) & (src_len > 0), _copy, _identity, None)
        return decode_state, cache

    decode_state, cache = jax.lax.fori_loop(0, num_new, copy_pages_for_updated_seq, (decode_state, cache))

    # Enqueue/update tokens for the clone targets (only the first num_new entries will be used)
    decode_state = decode_state.update_tokens(new_tokens, tgt_ids, log_probs, num_new)
    # Discharge processed clones so they are not reprocessed in subsequent flushes
    decode_state = decode_state.discharge_clone(tgt_ids, num_new)
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state, cache=cache)

    # Append clone outputs
    outputs = outputs.append(new_tokens, tgt_ids, log_probs, num_new, gen_state.decode_state.finished)

    # Device-side release of finished sequences (jit-safe)
    return gen_state, outputs


# @hax.named_jit(donate_args=(True, False, False))
@hax.named_jit(
    donate_args=(True, False, False, False, False, False),
)
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
    incremental_cleanup: bool = False,
) -> tuple[GenState, _DecodeOutputs]:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached.

    Args:
        gen_state: Current generation state with cache and decode state.
        model: Language model for decoding.
        sampler: Sampler for token selection.
        max_tokens_per_round: Maximum tokens to process per iteration.
        max_rounds: Maximum number of iterations.
        incremental_cleanup: When True, free pages for finished sequences after each round.
    """

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

        # jax.debug.print(
        #     "[_run_gen_loop] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
        #     tokens=tokens.array,
        #     slots=slot_ids.array,
        #     pos=pos_ids.array,
        #     lens=gen_state.decode_state.seq_lens.array,
        # )

        decode_state, binfo = decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)

        seq_lens = decode_state.seq_lens

        max_sample_indices = min(decode_state.page_table.max_seqs, max_tokens_per_round)
        sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices)

        # Decode logits and sample new tokens
        logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
        logits_at_samples = logits["position", sample_indices]

        num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
        new_slot_ids = slot_ids["position", sample_indices]
        new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_slot_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits_at_samples, temps, key=prng_keys)

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

        # Snapshot finished flags BEFORE cleanup for output extraction
        finished_snapshot = decode_state.finished

        # Incremental cleanup: free pages and invalidate metadata for finished sequences
        if incremental_cleanup:
            finished_mask = finished_snapshot.array
            # Free pages to reclaim memory
            decode_state = decode_state.free_pages_for_finished(finished_mask)
            # Invalidate sequence metadata and clear finished flags
            decode_state = decode_state.invalidate_finished()

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
        # Append non-stateful outputs for host-side extraction (use snapshot from before cleanup)
        outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, finished_snapshot)

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
        cache: PageCache,
        decode_state: DecodeState,
        sampler: Sampler,
        config: InferenceEngineConfig,
    ) -> None:
        """
        Args:
            model: Language model with :meth:`decode` and :meth:`initial_cache`.
            tokenizer: Tokenizer with `encode` and `decode` methods.
            cache: Pre-allocated KV cache matching the model and page table. **DONATED**
            decode_state: Initial decode state matching the cache's page table. **DONATED**
            sampler: Sampler instance for decoding.
            config: Engine configuration with sizing and decode parameters.
        """
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
        sequences = decode_state.sequences
        used_mask = np.asarray(jax.device_get(sequences.used_mask.array))
        free_slot_ids = [idx for idx, used in enumerate(used_mask) if not bool(used)]
        # Maintain free slots in ascending order to mirror PageTable's allocation policy.
        self.free_slots: list[int] = free_slot_ids
        # Mapping structures for active sequences
        # local_map: local slot id -> (request id, child id)
        # sequences: request id -> {child id -> local slot id}
        self.local_map: dict[int, tuple[int, int]] = {}
        self.sequences: dict[int, dict[int, int]] = {}
        # Results by request id -> choice -> DecodeResult
        self.results: dict[int, dict[int, DecodeResult]] = {}

    @classmethod
    def from_model_with_config(
        cls,
        model: LmHeadModel,
        tokenizer,
        config: InferenceEngineConfig,
    ) -> "InferenceEngine":
        """Build an engine using a EngineConfig for sizing knobs."""
        if config.max_pages is None:
            inferred_pages = _infer_max_pages_from_hbm(model, config)
            config = dataclasses.replace(config, max_pages=int(inferred_pages))

        max_pages_per_seq = config.max_pages_per_seq

        assert config.max_pages is not None

        table = PageTable.init(config.max_pages, config.max_seqs, config.page_size, max_pages_per_seq)
        cache = hax.named_jit(model.initial_cache)(table.spec(), dtype=config.compute_dtype)
        decode_state = DecodeState.init(
            table,
            max_stop_seqs=config.max_stop_seqs,
            max_stop_tokens=config.max_stop_tokens,
            max_queued_tokens=config.max_queued_tokens,
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
        When `use_logical_reset` is True in config, skips expensive cache zeroing.
        """
        import sys
        print(f"[DEBUG InferenceEngine.reset P{jax.process_index()}] === Starting reset, use_logical_reset={self.config.use_logical_reset} ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # In multi-host mode, skip the reset of JAX arrays entirely on first call
        # since the arrays were just initialized in from_model_with_config
        # Instead, just clear the Python-side state
        print(f"[DEBUG InferenceEngine.reset P{jax.process_index()}] === Skipping array reset for multi-host, just clearing Python state ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        self.free_slots = list(range(int(self.gen_state.decode_state.max_seqs)))
        self.local_map.clear()
        self.sequences.clear()
        self.results = {}
        print(f"[DEBUG InferenceEngine.reset P{jax.process_index()}] === Reset done ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

    def _prefill_batch(self, batch: Sequence[Request]) -> _DecodeOutputs | None:
        """Admit a batch from the head of the queue that fits in free slots/pages.

        Returns the decode outputs for the admitted prefill batch, or None if no work was admitted.
        """
        import sys
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === Starting _prefill_batch with {len(batch)} requests ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Build a single PrefillWork description and run prefill exactly once
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === About to call _prefill_prompts ===", flush=True)
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === free_slots before: {self.free_slots} ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        prefill_work = self._prefill_prompts(batch)
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === free_slots after: {self.free_slots} ===", flush=True)
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === _prefill_prompts done, work={prefill_work is not None} ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        if prefill_work is not None:
            print(f"[DEBUG _prefill_batch P{jax.process_index()}] === prefill_work created ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        if prefill_work is None:
            return None
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === About to call _run_prefill ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Add barrier sync before JIT call to ensure all hosts are aligned
        from levanter.utils.jax_utils import barrier_sync_with_tag
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === Calling barrier_sync_with_tag before _run_prefill ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        barrier_sync_with_tag("prefill_start")
        print(f"[DEBUG _prefill_batch P{jax.process_index()}] === barrier_sync done, now calling _run_prefill ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            new_state = _run_prefill(
                self.gen_state,
                self.model,
                self.sampler,
                prefill_work,
                int(self.config.max_seqs_in_prefill),
            )
            print(f"[DEBUG _prefill_batch P{jax.process_index()}] === _run_prefill done ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            import traceback
            print(f"[DEBUG _prefill_batch P{jax.process_index()}] === _run_prefill FAILED: {e} ===", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise

        # _run_prefill returns (GenState, _DecodeOutputs)
        self.gen_state, outputs = new_state
        return outputs

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

        # In multi-host mode, each host creates local arrays which are then combined
        # during JIT compilation. Since all hosts have identical data, this should work.
        import sys

        # Barrier sync to ensure all hosts are ready to build PrefillWork at the same time
        if jax.process_count() > 1:
            from levanter.utils.jax_utils import barrier_sync_with_tag
            print(f"[_prefill_prompts P{jax.process_index()}] === Waiting at barrier before building PrefillWork ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            barrier_sync_with_tag("prefill_prompts_build", timeout=60.0)

        print(f"[_prefill_prompts P{jax.process_index()}] === Building PrefillWork ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        # Create proper global arrays using jax.make_array_from_callback.
        # For replicated data, we use PartitionSpec(None, None, ...) with one None per dimension.
        # IMPORTANT: PartitionSpec() (empty) is different from PartitionSpec(None,) for 1D arrays!
        from jax.sharding import NamedSharding, PartitionSpec
        mesh = hax.partitioning._get_mesh()

        print(f"[_prefill_prompts P{jax.process_index()}] === Building PrefillWork with global arrays ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        def _to_jax_arr(arr, dtype):
            """Convert numpy array to a properly sharded global JAX array."""
            np_arr = np.asarray(arr, dtype=dtype)
            # Create a PartitionSpec with one None per dimension - this is fully replicated
            # but with correct dimensionality for the array.
            spec = PartitionSpec(*([None] * np_arr.ndim))
            replicated_sharding = NamedSharding(mesh, spec)
            # Use make_array_from_callback with replicated sharding.
            # The callback receives index tuples but for replicated sharding,
            # each shard is the full array.
            return jax.make_array_from_callback(
                np_arr.shape,
                replicated_sharding,
                lambda _: np_arr
            )

        # Create the PrefillWork with local arrays first
        prefill_queue = TokenQueue(
            queued_tokens=hax.named(_to_jax_arr(queue_tokens, jnp.int32), axis="position"),
            queued_slot_ids=hax.named(_to_jax_arr(queue_slot_ids, jnp.int32), axis="position"),
            queued_pos_ids=hax.named(_to_jax_arr(queue_pos_ids, jnp.int32), axis="position"),
            num_queued_tokens=_to_jax_arr(offset, jnp.int32),
        )

        prefill_work = PrefillWork(
            queue=prefill_queue,
            new_num_seqs=_to_jax_arr(total_new, jnp.int32),
            new_slot_ids=hax.named(_to_jax_arr(work_slot_ids, jnp.int32), axis="seq"),
            clone_targets=hax.named(_to_jax_arr(clone_targets, jnp.int32), axis="seq"),
            prompt_tokens=hax.named(_to_jax_arr(prompt_tokens, jnp.int32), axis=("seq", "position")),
            prompt_lengths=hax.named(_to_jax_arr(prompt_lengths, jnp.int32), axis="seq"),
            seq_params=SeqDecodingParams(
                max_num_tokens=_to_jax_arr(max_num_tokens, jnp.int32),
                stop_tokens=(
                    None
                    if stop_tokens is None
                    else hax.named(_to_jax_arr(stop_tokens, jnp.int32), axis=("seq", "stop_seq", "position"))
                ),
                temperature=_to_jax_arr(temperatures, jnp.float32),
                key=_to_jax_arr(prng_keys, jnp.uint32),
            ),
        )

        print(f"[_prefill_prompts P{jax.process_index()}] === PrefillWork created successfully ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        return prefill_work

    def generate(self, requests: Sequence[Request], step_callback=None) -> GenerationResult:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).

        Args:
            requests: Sequence of generation requests
            step_callback: Optional callback function called at each decode iteration with iteration number
        """
        # validate we don't have any sequences with n_generations exceeding max_seqs
        max_needed = max(int(r.n_generations) for r in requests)
        if max_needed > int(self.gen_state.decode_state.page_table.max_seqs):
            raise ValueError(
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.gen_state.decode_state.page_table.max_seqs})."
                "Decompose your request into smaller batches or increase max_seqs when building the service."
            )

        # for now, reset the engine state between each batch - the engine cannot be called with
        # parallel batches.
        import sys
        print(f"[DEBUG engine.generate P{jax.process_index()}] === About to call reset() ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        self.reset()
        print(f"[DEBUG engine.generate P{jax.process_index()}] === reset() done ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

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
        # Initial admission from queue and extract prompt tokens
        print(f"[DEBUG engine.generate P{jax.process_index()}] === About to call _prefill_batch() ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        decode_outputs = self._prefill_batch(requests)
        print(f"[DEBUG engine.generate P{jax.process_index()}] === _prefill_batch() done ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        self._ingest_outputs(decode_outputs)
        print(f"[DEBUG engine.generate P{jax.process_index()}] === _ingest_outputs() done ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        initial_prefill_out = time.time()
        logger.info(f"Initial prefill and extraction took {initial_prefill_out - time_in:.3f}s")

        def _all_done() -> bool:
            for rid, n_kids in expected_children.items():
                kid_map = self.results.get(rid, {})
                for cid in range(n_kids):
                    dr = kid_map.get(cid)
                    if dr is None or not dr.done:
                        return False
            return True

        stagnant_iters = 0
        decode_iteration = 0
        print(f"[DEBUG engine.generate P{jax.process_index()}] === Entering decode loop, max_iters={self.config.max_seq_len // self.config.max_rounds} ===", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        for loop_iter in range(self.config.max_seq_len // self.config.max_rounds):
            print(f"[DEBUG engine.generate P{jax.process_index()}] === Loop iteration {loop_iter} ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            if _all_done():
                break
            # Call step callback if provided
            if step_callback is not None:
                step_callback(decode_iteration)

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
            print(f"[DEBUG engine.generate P{jax.process_index()}] === About to call _run_generation_loop() ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            future_state, decode_outputs = _run_generation_loop(
                self.gen_state,
                self.model,
                self.sampler,
                # TODO: tune max_tokens_per_round
                int(self.config.imputed_max_tokens_per_round),
                int(self.config.max_rounds),
                bool(self.config.incremental_cleanup),
            )
            print(f"[DEBUG engine.generate P{jax.process_index()}] === _run_generation_loop() returned ===", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            submit_done = time.time()
            # Time spent with device executing (and the host thread waiting)
            self.gen_state = future_state
            device_time = time.time() - submit_done

            extract_start = time.time()
            new_tokens = self._ingest_outputs(decode_outputs)
            extract_time = time.time() - extract_start

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
                    f" (extract {extract_time:.3f}s"
                )
                # Log decode metrics to W&B
                levanter.tracker.log(
                    {
                        "decode/tokens_per_sec": tps_total,
                        "decode/iter_time_sec": iter_time,
                        "decode/device_time_sec": device_time,
                        "decode/host_time_sec": host_time,
                        "decode/submit_time_sec": submit_time,
                        "decode/extract_time_sec": extract_time,
                        "decode/new_tokens": new_tokens,
                    },
                    step=decode_iteration,
                )

            decode_iteration += 1

            # Safety: if nothing new was produced, avoid infinite loop
            if new_tokens == 0 and int(jax.device_get(self.gen_state.decode_state.num_queued_tokens)) == 0:
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
        logger.debug(f"Batch generated in {total_time:.2f}s, {total_generated} tokens, {tps_overall:.2f} tok/s")
        # Clear results for these requests now that we've assembled outputs
        for rid in call_rids:
            if rid in self.results:
                self.results.pop(rid, None)
        return GenerationResult(tokens=outputs_list, logprobs=logprobs_list, total_generated=total_generated)

    def write_kernel_jaxprs(self, path, log_artifacts: bool = True):
        """
        Write out jaxpr and hlo for the generation loop to the given path.
        """
        traced = jax.make_jaxpr(_run_generation_loop.__wrapped__)(
            self.gen_state,
            self.model,
            self.sampler,
            # TODO: tune max_tokens_per_round
            int(self.config.imputed_max_tokens_per_round),
            int(self.config.max_rounds),
            bool(self.config.incremental_cleanup),
        )
        with fsspec.open(os.path.join(path, "gen_loop.jaxpr.txt.gz"), "w", compression="infer") as f:
            f.write(str(traced.jaxpr))
        with fsspec.open(os.path.join(path, "gen_loop.hlo.txt.gz"), "w", compression="infer") as f:
            f.write(
                _run_generation_loop.lower(
                    self.gen_state,
                    self.model,
                    self.sampler,
                    int(self.config.imputed_max_tokens_per_round),
                    int(self.config.max_rounds),
                    bool(self.config.incremental_cleanup),
                ).as_text()
            )

        def _create_dummy_work():
            max_slots = self.config.max_seqs_in_prefill
            max_len = self.config.max_prefill_size
            prefill_queue = TokenQueue(
                queued_tokens=hax.zeros({"position": max_len}, dtype=jnp.int32),
                queued_slot_ids=hax.zeros({"position": max_len}, dtype=jnp.int32),
                queued_pos_ids=hax.zeros({"position": max_len}, dtype=jnp.int32),
                num_queued_tokens=jnp.zeros((), dtype=jnp.int32),
            )

            return PrefillWork(
                queue=prefill_queue,
                new_num_seqs=jnp.array(0, dtype=jnp.int32),
                new_slot_ids=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                clone_targets=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                prompt_tokens=hax.zeros({"seq": max_slots, "position": max_len}, dtype=jnp.int32),
                prompt_lengths=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                seq_params=SeqDecodingParams(
                    max_num_tokens=jnp.zeros(max_slots, dtype=jnp.int32),
                    stop_tokens=None,
                    temperature=jnp.zeros(max_slots, dtype=jnp.float32),
                    key=jnp.zeros((max_slots, 2), dtype=jnp.uint32),
                ),
            )

        prefill_traced = jax.make_jaxpr(_run_prefill.__wrapped__)(
            self.gen_state,
            self.model,
            self.sampler,
            eqx.filter_eval_shape(_create_dummy_work),
            int(self.config.max_seqs_in_prefill),
        )
        with fsspec.open(os.path.join(path, "run_prefill.jaxpr.txt.gz"), "w", compression="infer") as f:
            f.write(str(prefill_traced.jaxpr))
        with fsspec.open(os.path.join(path, "run_prefill.hlo.txt.gz"), "w", compression="infer") as f:
            f.write(
                _run_prefill.lower(
                    self.gen_state,
                    self.model,
                    self.sampler,
                    eqx.filter_eval_shape(_create_dummy_work),
                    int(self.config.max_seqs_in_prefill),
                ).as_text()
            )

        if log_artifacts:
            levanter.tracker.current_tracker().log_artifact(path, name="generation_kernels")
            logger.info(f"Written trace info to {path} and logged artifacts")
        else:
            logger.info(f"Written trace info to {path}")

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

            # # Print accumulated decoded text as it is generated -- For debugging
            # print_every_n = 10
            # if dr.tokens_decoded % print_every_n == 0:
            #     try:
            #         # Decode the full sequence so far
            #         full_text = self.tokenizer.decode(dr.token_list, skip_special_tokens=False)
            #         logger.info(f"[Request {rid}, Choice {cid}] Tokens {dr.tokens_decoded}: '{full_text}'")
            #     except Exception as e:
            #         logger.info(f"[Request {rid}, Choice {cid}] Tokens {dr.tokens_decoded}: <decode_error: {e}>")

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

            # Print final complete text when sequence is finished
            try:
                full_text = self.tokenizer.decode(dr.token_list, skip_special_tokens=False)
                logger.debug(f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): '{full_text}'")
            except Exception as e:
                logger.error(f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): <decode_error: {e}>")

        num_finished = int(fins.sum()) if hasattr(fins, "sum") else 0
        logger.debug(f"extract: appended={appended} (drained={n}) unmapped={unmapped} finished_count={num_finished}")

        return appended

    def _ingest_outputs(self, outputs: _DecodeOutputs | None) -> int:
        """Drain device outputs into host results and apply host-side release.

        Returns the number of tokens appended to results. No-op if outputs is None.
        """
        if outputs is None:
            return 0
        appended = self._extract_outputs(outputs)
        return appended
