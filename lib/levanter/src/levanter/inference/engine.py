# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import equinox as eqx
import haliax as hax
import humanfriendly as hly
from rigging.filesystem.factory import open_url
import haliax.haxtyping as ht
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceMapping

from jax.sharding import NamedSharding, PartitionSpec

import levanter.tracker
from levanter.inference.jit_scheduler import (
    DecodeState,
    FinishReason,
    SeqDecodingParams,
    TokenQueue,
    _CapturedRoutes,
    _DecodeOutputs,
)
from levanter.inference.page_table import PageBatchInfo, PageTable, kv_pages_for_request
from levanter.inference.utils import INVALID, is_valid
from levanter.inference.window_page_map import WindowPageMap
from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.layers.attention import AttentionMask
from levanter.layers.kv_cache import ListCache, PageCache
from levanter.layers.sampler import FULL_DISTRIBUTION_TOP_P, LogprobMode, Sampler
from levanter.models.lm_model import LmHeadModel
from levanter.models.snowball import SnowballLMHeadModel
from levanter.tokenizers import MarinTokenizer
from levanter.utils.jax_utils import estimated_free_device_memory, sharded_tree_size

logger = logging.getLogger(__name__)

_release_finished_slots = jax.jit(DecodeState.free_pages_for_finished, donate_argnums=0)
_abort_slots = jax.jit(DecodeState.abort_slots, donate_argnums=0)


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

    window_cache_max_pages: int | None = None
    """Optional separate page capacity for Snowball's sliding-window layers."""

    compute_dtype: jnp.dtype = jnp.bfloat16
    """KV cache dtype. Default bfloat16 for performance/accuracy balance."""

    logprob_mode: LogprobMode = LogprobMode.BEHAVIOR
    """Distribution used to score generated tokens."""

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

    prefill_bucket_sizes: tuple[int, ...] = ()
    """Additional compiled prompt token capacities below max_prefill_size."""

    decode_bucket_sizes: tuple[int, ...] = ()
    """Additional compiled token capacities below max_tokens_per_round."""

    max_retained_prompts: int = 0
    """Completed prompts whose full pages may be retained for Snowball prefix reuse."""

    window_cache_retained_spans: int = 1
    """Sliding-window spans to retain per cached prompt, in order of proximity to its end."""

    def __post_init__(self):
        if self.max_retained_prompts < 0 or self.window_cache_retained_spans < 1:
            raise ValueError("Prompt retention requires a nonnegative prompt limit and positive window span limit")
        prefill_limit = self.max_prefill_size or self.max_pages_per_seq * self.page_size
        _validate_bucket_sizes(self.prefill_bucket_sizes, prefill_limit, "prefill")
        _validate_bucket_sizes(self.decode_bucket_sizes, self.imputed_max_tokens_per_round, "decode")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")

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


def _validate_bucket_sizes(sizes: tuple[int, ...], limit: int, phase: str) -> None:
    if sizes != tuple(sorted(set(sizes))) or any(not 0 < size < limit for size in sizes):
        raise ValueError(f"{phase} bucket sizes must be positive, strictly increasing, and below {limit}")


def _token_bucket(needed: int, sizes: tuple[int, ...], limit: int) -> int:
    return next((size for size in sizes if size >= needed), limit)


def _tree_byte_size(tree, axis_resources: ResourceMapping | None = None) -> int:
    """Return the per-device number of bytes represented by ``tree``."""

    return sharded_tree_size(tree, mapping=axis_resources)


def _available_hbm_budget_bytes(hbm_utilization: float) -> int:
    """Estimate the per-device HBM budget available to the KV cache."""

    if not (0.0 < hbm_utilization <= 1.0):
        raise ValueError("hbm_utilization must be in the interval (0, 1].")

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


def _infer_max_pages_from_hbm(
    model: LmHeadModel,
    config: InferenceEngineConfig,
    axis_resources: ResourceMapping | None = None,
) -> int:
    """Infer a KV-page budget using HBM utilization targets."""

    max_pages_per_seq = config.max_pages_per_seq

    try:
        budget = _available_hbm_budget_bytes(config.hbm_utilization)
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

        return _tree_byte_size(cache_shape, axis_resources)

    bytes_one = cache_bytes(1)
    if bytes_one > budget:
        raise ValueError(
            "HBM budget insufficient to allocate even a single KV cache page. "
            "Provide `max_pages` explicitly or increase `hbm_utilization`."
        )

    # No engine state can use more than max_seqs fully populated sequences. Allocating pages beyond
    # this bound only consumes the HBM needed by compilation and transient execution buffers.
    page_capacity = max(int(config.max_seqs * max_pages_per_seq), 1)
    low = 1
    high = page_capacity
    high_bytes = cache_bytes(high)

    if high_bytes <= budget:
        low = page_capacity

    # Binary search between one known-good page and the capacity bound when the full cache is too large.
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
    finish_reason: FinishReason = FinishReason.RUNNING
    logprobs: list[float] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.finish_reason != FinishReason.RUNNING


class GenState(eqx.Module):
    """Container for generation state used during decoding.

    Holds the KV cache and `DecodeState` (which itself owns the `PageTable`).
    Provides `clone_sequence` to efficiently support multi-sample generation by
    sharing fully used pages.
    """

    cache: PageCache
    decode_state: DecodeState
    window_pages: WindowPageMap | None = None

    def reset(self):
        return GenState(
            cache=self.cache.reset(),
            decode_state=self.decode_state.reset(),
            window_pages=None if self.window_pages is None else self.window_pages.reset(),
        )

    def remap_window_pages(
        self, decode_state: DecodeState, batch: PageBatchInfo
    ) -> tuple["GenState", PageBatchInfo | None]:
        if self.window_pages is None:
            return self, None
        window_pages, window_batch = self.window_pages.remap(decode_state.sequences, batch)
        return dataclasses.replace(self, window_pages=window_pages), window_batch

    def clone_sequence(
        self,
        parent_local_id: int | jax.Array,
        child_local_id: int | jax.Array | None = None,
        seq_params: SeqDecodingParams | None = None,
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
    has_clones: bool = eqx.field(static=True)
    cached_pages: NamedArray | None = None
    prompt_rows: NamedArray | None = None


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


def _gather_logits(logits: NamedArray, indices: NamedArray) -> NamedArray:
    """Gather only sampling rows, replicated for vocabulary-wide sampling."""
    logits = logits.rearrange(("position", ...))
    sharding = named_sharding_of(logits.array)
    out_sharding = None
    if sharding is not None and not sharding.mesh.empty:
        # The sampler sorts across the vocabulary for nucleus sampling. Gather after
        # selecting rows so prefill does not replicate every token's vocabulary logits.
        out_sharding = NamedSharding(sharding.mesh, PartitionSpec())
    values = logits.array.at[indices.array].get(out_sharding=out_sharding)
    return hax.named(values, (*indices.axes, *logits.axes[1:]))


@runtime_checkable
class _SelectedRowDecoder(Protocol):
    def decode_for_sampling(
        self,
        input_ids: NamedArray,
        kv_cache: PageCache,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        sample_indices: NamedArray,
    ) -> tuple[NamedArray, PageCache]: ...


def _route_dtype(model: SnowballLMHeadModel) -> jnp.dtype:
    return jnp.dtype(jnp.uint8 if model.config.num_experts <= 256 else jnp.int32)


def _capture_buffer(model: LmHeadModel, capacity: int) -> _CapturedRoutes:
    if not isinstance(model, SnowballLMHeadModel):
        raise ValueError("Expert route capture requires a Snowball model")
    config = model.config
    return _CapturedRoutes.init(capacity, config.num_layers, config.num_experts_per_token, _route_dtype(model))


def _decode_for_sampling(
    model: LmHeadModel,
    cache: PageCache,
    binfo: PageBatchInfo,
    tokens: NamedArray,
    positions: NamedArray,
    slots: NamedArray,
    sample_indices: NamedArray | None,
    routes: _CapturedRoutes | None,
    *,
    window_batch: PageBatchInfo | None,
    prefill: bool = False,
) -> tuple[NamedArray, PageCache, _CapturedRoutes | None]:
    if routes is None:
        if window_batch is not None:
            assert isinstance(model, SnowballLMHeadModel)
            if prefill and sample_indices is not None:
                logits, cache = model.decode_for_sampling(
                    tokens, cache, binfo, positions, sample_indices, window_batch=window_batch
                )
                return logits, cache, None
            logits, cache = model.decode(tokens, cache, binfo, positions, window_batch=window_batch)
        elif prefill and sample_indices is not None and isinstance(model, _SelectedRowDecoder):
            logits, cache = model.decode_for_sampling(tokens, cache, binfo, positions, sample_indices)
            return logits, cache, None
        else:
            logits, cache = model.decode(tokens, cache, binfo, positions)
        return (logits if sample_indices is None else _gather_logits(logits, sample_indices)), cache, None
    if not isinstance(model, SnowballLMHeadModel):
        raise ValueError("Expert route capture requires a Snowball model")
    result = model.decode_with_routes(
        tokens, cache, binfo, positions, sample_indices=sample_indices, window_batch=window_batch, prefill=prefill
    )
    assert result.routes is not None
    routes = routes.append(result.routes.expert_ids, slots, positions, binfo.num_new_tokens)
    return result.logits, result.cache, routes


def _sample_sharded_positions(
    sampler: Sampler,
    logits: NamedArray,
    temperatures: NamedArray,
    top_ps: NamedArray,
    num_samples: jax.Array,
    keys: jax.Array,
) -> tuple[NamedArray, NamedArray]:
    sharding = named_sharding_of(logits.array)
    if sharding is None or sharding.mesh.empty or not sharding.spec or sharding.spec[0] is None:
        return sampler.sample_positions(logits, temperatures, top_ps=top_ps, num_samples=num_samples, key=keys)

    row_spec = PartitionSpec(sharding.spec[0])
    logits_spec = PartitionSpec(sharding.spec[0], None)
    positions = hax.arange(logits.resolve_axis("position"))
    top_ps = hax.where(positions < num_samples, top_ps, FULL_DISTRIBUTION_TOP_P)

    def local_sample(local_logits, local_temps, local_top_ps, local_keys):
        rows = hax.Axis("position", local_logits.shape[0])
        tokens, scores = sampler.sample_positions(
            hax.named(local_logits, (rows, sampler.Vocab)),
            hax.named(local_temps, rows),
            top_ps=hax.named(local_top_ps, rows),
            num_samples=jnp.int32(rows.size),
            key=local_keys,
        )
        return tokens.array, scores.array

    sample = jax.shard_map(
        local_sample,
        mesh=sharding.mesh,
        in_specs=(logits_spec, row_spec, row_spec, row_spec),
        out_specs=(row_spec, row_spec),
        check_vma=False,
    )
    row_sharding = sharding.update(spec=row_spec)
    tokens, scores = sample(
        jax.reshard(logits.array, sharding.update(spec=logits_spec)),
        jax.reshard(temperatures.array, row_sharding),
        jax.reshard(top_ps.array, row_sharding),
        jax.reshard(keys, row_sharding),
    )
    gathered = sharding.update(spec=PartitionSpec())
    return hax.named(jax.reshard(tokens, gathered), "position"), hax.named(jax.reshard(scores, gathered), "position")


def _prefill_kernel(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    queue: TokenQueue,
    binfo: PageBatchInfo,
    max_seqs_in_prefill: int,  # static
    capture_routes: bool,
    *,
    has_clones: bool,
) -> tuple[GenState, _DecodeOutputs]:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    tokens = queue.queued_tokens
    pos_ids = queue.queued_pos_ids
    slot_ids = queue.queued_slot_ids

    decode_state = gen_state.decode_state

    seq_lens = decode_state.seq_lens

    sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_seqs_in_prefill)

    # jax.debug.print(
    #     "[_run_prefill] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
    #     tokens=tokens.array,
    #     slots=slot_ids.array,
    #     pos=pos_ids.array,
    #     lens=decode_state.seq_lens.array,
    # )
    if gen_state.window_pages is not None:
        binfo = binfo.trim_sequence_padding(max_seqs_in_prefill)
    gen_state, window_batch = gen_state.remap_window_pages(decode_state, binfo)
    route_buffer = _capture_buffer(model, tokens.size) if capture_routes else None
    logits_at_samples, cache, routes = _decode_for_sampling(
        model,
        gen_state.cache,
        binfo,
        tokens,
        pos_ids,
        slot_ids,
        sample_indices,
        route_buffer,
        window_batch=window_batch,
        prefill=True,
    )

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

    temps = decode_state.temperature["seq", new_slot_ids]
    top_ps = decode_state.top_p["seq", new_slot_ids]

    new_tokens, log_probs = sampler.sample_positions(
        logits_at_samples, temps, top_ps=top_ps, num_samples=num_new_tokens, key=prng_keys
    )

    # Update decode_state (also enqueues into the main decode queue)
    decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

    # Initialize outputs buffer and append prefill-sampled tokens
    outputs = _DecodeOutputs.init(
        max_tokens=decode_state.max_seqs * 2,
        max_seqs=decode_state.max_seqs,
        with_logprobs=True,
    )
    outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finish_reasons)
    gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)

    # If clone targets specified, sample alternative tokens for clones using the same logits slice
    if has_clones:
        gen_state, outputs = _handle_clones(
            gen_state,
            logits_at_samples,
            new_slot_ids,
            new_pos_ids,
            sampler,
            outputs,
        )

    # Device-side release of finished sequences (jit-safe)
    # jax.debug.print("[_run_prefill] output_tokens={} output_slots={}", outputs.tokens, outputs.slot_ids)

    # jax.debug.print(
    #     "[prefill] outputs_size={size} queued_after={queued}",
    #     size=outputs.num_tokens,
    #     queued=gen_state.decode_state.num_queued_tokens,
    # )
    return gen_state, dataclasses.replace(outputs, routes=routes)


def _seq_params_from_work(work: PrefillWork, idx: int) -> SeqDecodingParams:
    def select(x):
        if isinstance(x, NamedArray):
            return x["seq", idx]
        elif is_jax_array_like(x):
            return x[idx]
        else:
            raise TypeError(f"Unexpected type in seq_params: {type(x)}")

    return hax.tree_util.tree_map(select, work.seq_params)


def _apply_prefill_work(gen_state: GenState, work: PrefillWork) -> tuple[GenState, PageBatchInfo]:
    num_new = work.new_num_seqs.astype(jnp.int32)

    def primary(i: int, state: GenState) -> GenState:
        slot = work.new_slot_ids.array[i]

        def assign(gs: GenState) -> GenState:
            decode_state, _ = gs.decode_state.reserve_slot(slot)
            pages = None
            if work.cached_pages is not None:
                pages = work.cached_pages["seq", i]
                table = decode_state.page_table
                refs = table.page_ref_counts.at["page", pages].add(1, mode="drop")
                decode_state = dataclasses.replace(
                    decode_state, page_table=dataclasses.replace(table, page_ref_counts=refs)
                )
            prompt_row = i if work.prompt_rows is None else work.prompt_rows.array[i]
            decode_state = decode_state.assign_seq(
                local_slot_id=slot,
                tokens=work.prompt_tokens["seq", prompt_row],
                seq_len=work.prompt_lengths.array[i].astype(jnp.int32),
                kv_pages=pages,
                page_indices=pages,
                seq_params=_seq_params_from_work(work, i),
            )
            return dataclasses.replace(gs, decode_state=decode_state)

        is_primary = is_valid(slot) & ~is_valid(work.clone_targets.array[i])
        return jax.lax.cond(is_primary, assign, lambda gs: gs, state)

    def clone(i: int, state: GenState) -> GenState:
        slot = work.new_slot_ids.array[i]
        parent = work.clone_targets.array[i]

        def assign(gs: GenState) -> GenState:
            cloned, _ = gs.clone_sequence(parent, child_local_id=slot, seq_params=_seq_params_from_work(work, i))
            return cloned

        is_clone = is_valid(slot) & is_valid(parent)
        return jax.lax.cond(is_clone, assign, lambda gs: gs, state)

    gen_state = jax.lax.fori_loop(0, num_new, primary, gen_state)
    # Full prefix pages must exist before cloning, so parent prefill fills the shared pages.
    decode_state, binfo = gen_state.decode_state.allocate_for_seq(
        token_slot_ids=work.queue.queued_slot_ids, token_pos_ids=work.queue.queued_pos_ids
    )
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state)
    if work.has_clones:
        gen_state = jax.lax.fori_loop(0, num_new, clone, gen_state)
    return gen_state, binfo


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("max_seqs_in_prefill", "capture_routes"))
def _run_prefill(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    work: PrefillWork,
    max_seqs_in_prefill: int,
    capture_routes: bool = False,
) -> tuple[GenState, _DecodeOutputs]:
    gen_state, binfo = _apply_prefill_work(gen_state, work)
    return _prefill_kernel(
        gen_state, model, sampler, work.queue, binfo, max_seqs_in_prefill, capture_routes, has_clones=work.has_clones
    )


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
    logits_this_time = _gather_logits(logits, src_pos)
    pos_ids_this_time = pos_ids["position", src_pos]

    # Sample clones from the same boundary logits as their sources
    temps = gen_state.decode_state.temperature["seq", tgt_ids]
    top_ps = gen_state.decode_state.top_p["seq", tgt_ids]
    prng_keys = gen_state.decode_state.prng_keys_for(tgt_ids, pos_ids_this_time)

    new_tokens, log_probs = sampler.sample_positions(
        logits_this_time, temps, top_ps=top_ps, num_samples=num_new, key=prng_keys
    )

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
    outputs = outputs.append(new_tokens, tgt_ids, log_probs, num_new, gen_state.decode_state.finish_reasons)

    # Device-side release of finished sequences (jit-safe)
    return gen_state, outputs


@functools.partial(jax.jit, static_argnums=(3, 4, 5), donate_argnames=("gen_state",))
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
    capture_routes: bool = False,
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

        # jax.debug.print(
        #     "[_run_gen_loop] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
        #     tokens=tokens.array,
        #     slots=slot_ids.array,
        #     pos=pos_ids.array,
        #     lens=gen_state.decode_state.seq_lens.array,
        # )

        decode_state, binfo = decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)

        max_sample_indices = min(decode_state.page_table.max_seqs, max_tokens_per_round)
        if gen_state.window_pages is not None:
            # Every participating sequence contributes at least one token row.
            binfo = binfo.trim_sequence_padding(max_sample_indices)
        sample_locally = (
            isinstance(model, SnowballLMHeadModel)
            and model.has_expert_sharded_kv
            and decode_state.max_seqs > 1
            and max_sample_indices == max_tokens_per_round
        )
        sample_indices = None
        if not sample_locally:
            sample_indices = _compute_sample_indices(pos_ids, slot_ids, decode_state.seq_lens, max_sample_indices)

        # Decode logits and sample new tokens
        gen_state, window_batch = gen_state.remap_window_pages(decode_state, binfo)
        logits_at_samples, cache, routes = _decode_for_sampling(
            model,
            gen_state.cache,
            binfo,
            tokens,
            pos_ids,
            slot_ids,
            sample_indices,
            outputs.routes,
            window_batch=window_batch,
        )
        outputs = dataclasses.replace(outputs, routes=routes)

        if sample_indices is None:
            num_new_tokens = binfo.num_new_tokens
            new_slot_ids, new_pos_ids = slot_ids, pos_ids
        else:
            num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
            new_slot_ids = slot_ids["position", sample_indices]
            new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_slot_ids]
        top_ps = decode_state.top_p["seq", new_slot_ids]

        if sample_locally:
            new_tokens, log_probs = _sample_sharded_positions(
                sampler, logits_at_samples, temps, top_ps, num_new_tokens, prng_keys
            )
        else:
            new_tokens, log_probs = sampler.sample_positions(
                logits_at_samples, temps, top_ps=top_ps, num_samples=num_new_tokens, key=prng_keys
            )

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
        # Append non-stateful outputs for host-side extraction
        outputs = outputs.append(new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finish_reasons)

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
    if capture_routes:
        outputs_buf = dataclasses.replace(outputs_buf, routes=_capture_buffer(model, outputs_buf.max_queued_tokens))
    init_state = (gen_state, outputs_buf, jnp.array(0, dtype=jnp.int32))
    final_gen_state, final_outputs, _ = jax.lax.while_loop(cond, body, init_state)
    # jax.debug.print("[gen] final outputs_size={size}", size=final_outputs.num_tokens)
    return final_gen_state, final_outputs


@dataclass(frozen=True)
class GeneratedExpertRoutes:
    expert_ids: np.ndarray  # [layer, position, expert_slot]
    valid: np.ndarray  # [position], terminal token may have no route yet


@dataclass
class GenerationResult:
    tokens: list[list[int]]
    logprobs: list[list[float]] | None
    total_generated: int
    finish_reasons: list[FinishReason]
    expert_routes: list[GeneratedExpertRoutes] | None = None


FIRST_TOKEN_LOGPROB = 0.0


@dataclass(frozen=True)
class TokenSequenceLogprobs:
    """Causal logprobs aligned with an input token sequence.

    The first token has no preceding context, so its logprob is the synthetic
    ``FIRST_TOKEN_LOGPROB`` sentinel and its top-token map contains only itself.
    """

    token_logprobs: list[float]
    top_token_logprobs: list[dict[int, float]]


def score_token_sequence_logprobs(
    model: LmHeadModel,
    token_ids: Sequence[int],
    top_k: int,
) -> TokenSequenceLogprobs:
    """Score a token sequence with a causal full forward pass."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    token_id_list = [int(token_id) for token_id in token_ids]
    if not token_id_list:
        return TokenSequenceLogprobs(token_logprobs=[], top_token_logprobs=[])

    token_logprobs = [FIRST_TOKEN_LOGPROB]
    top_token_logprobs = [{token_id_list[0]: FIRST_TOKEN_LOGPROB}]
    if len(token_id_list) == 1:
        return TokenSequenceLogprobs(token_logprobs=token_logprobs, top_token_logprobs=top_token_logprobs)

    Pos = hax.Axis("position", len(token_id_list))
    input_ids = hax.named(jnp.array(token_id_list, dtype=jnp.int32), Pos)
    pos_ids = hax.named(jnp.arange(len(token_id_list), dtype=jnp.int32), Pos)
    logits = model(input_ids=input_ids, attn_mask=AttentionMask.causal(), pos_ids=pos_ids, key=None)

    logits_array = logits.astype(jnp.float32).rearrange((Pos, model.Vocab)).array
    next_token_logprobs = jax.nn.log_softmax(logits_array[:-1], axis=-1)
    target_ids = jnp.array(token_id_list[1:], dtype=jnp.int32)
    scored_logprobs = next_token_logprobs[jnp.arange(len(target_ids)), target_ids]
    token_logprobs.extend(float(logprob) for logprob in jax.device_get(scored_logprobs))

    vocab_top_k = min(top_k, next_token_logprobs.shape[-1])
    top_values, top_indices = jax.lax.top_k(next_token_logprobs, vocab_top_k)
    top_values = np.asarray(jax.device_get(top_values))
    top_indices = np.asarray(jax.device_get(top_indices))
    for values, indices in zip(top_values, top_indices, strict=True):
        top_token_logprobs.append(
            {int(token_id): float(logprob) for token_id, logprob in zip(indices, values, strict=True)}
        )

    return TokenSequenceLogprobs(token_logprobs=token_logprobs, top_token_logprobs=top_token_logprobs)


@dataclass(frozen=True)
class _CachedPrompt:
    tokens: np.ndarray
    pages: np.ndarray
    routes: list[tuple[np.ndarray, np.ndarray]] | None


@dataclass(frozen=True)
class _PageReservation:
    private_pages: int
    shared_pages: np.ndarray


def _window_working_set_pages(
    config: InferenceEngineConfig, table: PageTable, window: int, sequence_count: int
) -> int:
    # Budget every live tail and all new pages in the largest packed prefill.
    prefill_capacity = config.max_prefill_size or table.max_len_per_seq
    return (
        sequence_count * ((window + table.page_size - 1) // table.page_size + 1)
        + (prefill_capacity + table.page_size - 1) // table.page_size
        + config.max_seqs_in_prefill
    )


def _route_prefix(chunks: Sequence[tuple[np.ndarray, np.ndarray]], length: int) -> list[tuple[np.ndarray, np.ndarray]]:
    routes = []
    for positions, ids in chunks:
        keep = positions < length
        if np.any(keep):
            routes.append((positions[keep], ids[keep]))
    return routes


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
        tokenizer: MarinTokenizer | None,
        cache: PageCache,
        decode_state: DecodeState,
        sampler: Sampler,
        config: InferenceEngineConfig,
        window_pages: WindowPageMap | None = None,
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
        self.gen_state: GenState = GenState(cache=cache, decode_state=decode_state, window_pages=window_pages)
        self._initial_decode_state = decode_state
        # Impute max_prefill_size if not set
        if config.max_prefill_size is None:
            config = dataclasses.replace(config, max_prefill_size=decode_state.page_table.max_len_per_seq)
        if config.max_retained_prompts and not isinstance(model, SnowballLMHeadModel):
            raise ValueError("Prompt prefix retention requires Snowball")
        self._cached_prompts: list[_CachedPrompt] = []
        self._page_reservations: dict[int, _PageReservation] = {}
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
        tokenizer: MarinTokenizer | None,
        config: InferenceEngineConfig,
        axis_resources: ResourceMapping | None = None,
    ) -> "InferenceEngine":
        """Build an engine using a EngineConfig for sizing knobs."""
        if config.max_pages is None:
            inferred_pages = _infer_max_pages_from_hbm(model, config, axis_resources)
            config = dataclasses.replace(config, max_pages=int(inferred_pages))

        max_pages_per_seq = config.max_pages_per_seq

        assert config.max_pages is not None

        if isinstance(model, SnowballLMHeadModel) and model.config.replay_precision == "zero_kl":
            if (config.max_seqs, config.max_seq_len, config.page_size, config.max_pages) != (4, 256, 128, 8):
                raise ValueError("Zero-KL replay requires four 256-token slots and eight 128-token pages")
            if config.logprob_mode != LogprobMode.POLICY:
                raise ValueError("Zero-KL replay requires raw policy log probabilities")
        table = PageTable.init(config.max_pages, config.max_seqs, config.page_size, max_pages_per_seq)
        window_pages = None
        window_capacity = config.window_cache_max_pages
        if window_capacity is not None:
            if not isinstance(model, SnowballLMHeadModel):
                raise ValueError("A compact window cache requires a Snowball model")
            page_size = table.page_size
            window = model.config.sliding_window
            prefill_size = config.max_prefill_size or table.max_len_per_seq
            # Pin one tail per active request and budget all writes during the largest prefill.
            tail_pages = (window + page_size - 1) // page_size + 1
            prefill_pages = (prefill_size + page_size - 1) // page_size + config.max_seqs_in_prefill
            min_pages = config.max_seqs * tail_pages + prefill_pages
            if not min_pages <= window_capacity < table.num_pages:
                raise ValueError(f"Compact window cache capacity must be in [{min_pages}, {table.num_pages})")
            window_pages = WindowPageMap.init(table.num_pages, window_capacity, window)
            # Allocate one layer at a time and keep the KV head placement specified by Snowball.
            initialize_layer = eqx.filter_jit(model.initial_layer_cache)
            specs = model.layer_cache_specs(table.spec(), window_capacity)
            cache = ListCache(tuple(initialize_layer(spec, dtype=config.compute_dtype) for spec in specs))
        else:
            cache = hax.named_jit(model.initial_cache, axis_resources=axis_resources)(
                table.spec(), dtype=config.compute_dtype
            )
        decode_state = DecodeState.init(
            table,
            eos_token_id=tokenizer.eos_token_id if tokenizer is not None else None,
            max_stop_seqs=config.max_stop_seqs,
            max_stop_tokens=config.max_stop_tokens,
            max_queued_tokens=config.max_queued_tokens,
        )
        vocab_axis = model.Vocab
        sampler = Sampler(vocab_axis, logprob_mode=config.logprob_mode)
        return cls(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            decode_state=decode_state,
            sampler=sampler,
            config=config,
            window_pages=window_pages,
        )

    def reset(self) -> None:
        """Free all local sequence slots and reset to the initial `DecodeState`.

        Keeps the KV cache memory allocated. Reuses current `PageTable` object with pages freed.
        """
        self.gen_state = eqx.filter_jit(GenState.reset, donate="all", keep_unused=True)(self.gen_state)
        self.free_slots = list(range(int(self.gen_state.decode_state.max_seqs)))
        self.local_map.clear()
        self.sequences.clear()
        self.results = {}
        self._cached_prompts.clear()
        self._page_reservations.clear()

    def can_reuse_cache(self, model: LmHeadModel) -> bool:
        """Check configuration, shapes, dtypes, and placements before reusing cache storage."""
        old_leaves, old_structure = jax.tree.flatten(self.model)
        new_leaves, new_structure = jax.tree.flatten(model)
        if old_structure != new_structure:
            return False
        for old, new in zip(old_leaves, new_leaves, strict=True):
            if isinstance(old, jax.Array):
                if (
                    not isinstance(new, jax.Array)
                    or old.shape != new.shape
                    or old.dtype != new.dtype
                    or old.sharding != new.sharding
                ):
                    return False
            elif old != new:
                return False
        return True

    def score_token_logprobs(self, token_ids: Sequence[int], top_k: int) -> TokenSequenceLogprobs:
        """Score a token sequence under this engine's model."""
        return score_token_sequence_logprobs(self.model, token_ids, top_k)

    def _request_pages(self, request: Request) -> int:
        return kv_pages_for_request(
            len(request.prompt_tokens),
            min(int(request.decode_params.max_num_tokens), self.config.max_seq_len),
            request.n_generations,
            self.config.page_size,
        )

    def _retained_page_refs(self, table: PageTable) -> np.ndarray:
        if not self._cached_prompts:
            return np.zeros(table.num_pages, dtype=np.int32)
        pages = np.concatenate([cached.pages for cached in self._cached_prompts])
        return np.bincount(pages, minlength=table.num_pages).astype(np.int32)

    def _reserved_page_usage(self, table: PageTable) -> tuple[int, np.ndarray]:
        shared = np.zeros(table.num_pages, dtype=bool)
        private = 0
        for reservation in self._page_reservations.values():
            private += reservation.private_pages
            shared[reservation.shared_pages] = True
        return private + int(np.count_nonzero(shared)), shared

    def _retain_batch_prompts(
        self,
        requests: Sequence[Request],
        route_chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] | None,
    ) -> None:
        state = self.gen_state.decode_state
        before = self._retained_page_refs(state.page_table)
        page_indices = jax.device_get(state.sequences.page_indices.array)
        for request in requests:
            rid = request.request_id
            page_count = len(request.prompt_tokens) // self.config.page_size
            if not page_count:
                continue
            pages = page_indices[self.sequences[rid][0], :page_count].copy()
            reservation = self._page_reservations[rid]
            private_pages = reservation.private_pages + len(reservation.shared_pages) - len(pages)
            self._page_reservations[rid] = _PageReservation(private_pages, pages)
            routes = None
            if route_chunks is not None:
                full_length = page_count * self.config.page_size
                routes = _route_prefix(route_chunks.get((rid, 0), ()), full_length)
            self._cached_prompts.append(
                _CachedPrompt(np.asarray(request.prompt_tokens, dtype=np.int32), pages, routes)
            )
        self._trim_retained_prompts(before)

    def _trim_retained_prompts(self, before: np.ndarray) -> None:
        state = self.gen_state.decode_state
        table = state.page_table
        after = self._retained_page_refs(table)
        reserved, shared = self._reserved_page_usage(table)
        available = table.num_pages - reserved
        retained = np.count_nonzero((after > 0) & ~shared)
        while retained > available or len(self._cached_prompts) > self.config.max_retained_prompts:
            evict = 0
            if retained > available:
                evict = next(i for i, cached in enumerate(self._cached_prompts) if not np.all(shared[cached.pages]))
            pages = self._cached_prompts.pop(evict).pages
            np.add.at(after, pages, -1)
            retained -= np.count_nonzero((after[pages] == 0) & ~shared[pages])
        self._refresh_window_prompt_tails()
        change = after - before
        if np.any(change):
            delta = jax.device_put(change, table.page_ref_counts.array.sharding)
            table = dataclasses.replace(table, page_ref_counts=table.page_ref_counts + hax.named(delta, "page"))
            self.gen_state = dataclasses.replace(
                self.gen_state, decode_state=dataclasses.replace(state, page_table=table)
            )

    def _refresh_window_prompt_tails(self) -> None:
        window_pages = self.gen_state.window_pages
        if window_pages is None:
            return

        page_size = self.config.page_size
        retained_tail = (window_pages.window + page_size - 1) // page_size + 1
        mask = np.zeros(window_pages.logical_to_window.size, dtype=np.bool_)
        # Admission refreshes pins with all newly reserved slots before launching prefill.
        working_set = _window_working_set_pages(
            self.config, self.gen_state.decode_state.page_table, window_pages.window, len(self.local_map)
        )
        free = window_pages.window_to_logical.size - working_set
        # Give recent prompts their terminal window before retaining earlier spans.
        for distance in range(0, self.config.window_cache_retained_spans * retained_tail, retained_tail):
            start = -(distance + retained_tail)
            stop = -distance if distance else None
            for prompt in reversed(self._cached_prompts):
                if distance and not np.all(mask[prompt.pages[-retained_tail:]]):
                    break
                span = prompt.pages[start:stop]
                cost = int(np.count_nonzero(~mask[span]))
                if cost > free:
                    break
                mask[span] = True
                free -= cost
        pins = jax.device_put(mask, window_pages.terminal_pins.sharding)
        window_pages = dataclasses.replace(window_pages, terminal_pins=pins)
        self.gen_state = dataclasses.replace(self.gen_state, window_pages=window_pages)

    def _matching_prompt_pages(
        self, tokens: np.ndarray, window_residency: np.ndarray | None, *, require_routes: bool
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]] | None]:
        pages = np.empty(0, dtype=np.int32)
        routes = None
        window_pages = self.gen_state.window_pages
        page_size = self.config.page_size
        tail_pages = 0 if window_pages is None else (window_pages.window + page_size - 1) // page_size
        for cached in reversed(self._cached_prompts):
            if require_routes and cached.routes is None:
                continue
            # Leave one query so prefill can compute next-token logits.
            limit = min(len(tokens) - 1, len(cached.tokens))
            next_page_end = (len(pages) + 1) * page_size - 1
            if next_page_end >= limit or tokens[next_page_end] != cached.tokens[next_page_end]:
                continue
            mismatches = tokens[:limit] != cached.tokens[:limit]
            first = int(np.argmax(mismatches))
            count = (first if mismatches[first] else limit) // page_size
            if window_residency is not None and count > len(pages):
                # A textual match can resume only where the previous window is resident.
                resident = window_residency[cached.pages[:count]] != INVALID
                if not resident[-tail_pages:].all():
                    boundaries = np.arange(1, count + 1)
                    unavailable = np.pad(np.cumsum(~resident, dtype=np.int32), (1, 0))
                    valid = unavailable[boundaries] == unavailable[np.maximum(boundaries - tail_pages, 0)]
                    eligible = boundaries[valid]
                    count = int(eligible[-1]) if eligible.size else 0
            if count > len(pages):
                pages = cached.pages[:count]
                routes = cached.routes
        return pages, routes

    def _prefill_batch(
        self,
        batch: Sequence[Request],
        capture_routes: bool = False,
        route_chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> _DecodeOutputs | None:
        """Admit a batch from the head of the queue that fits in free slots/pages.

        Returns the decode outputs for the admitted prefill batch, or None if no work was admitted.
        """
        # Build a single PrefillWork description and run prefill exactly once
        prefill_work = self._prefill_prompts(batch, route_chunks)
        if prefill_work is None:
            return None
        new_state = _run_prefill(
            self.gen_state, self.model, self.sampler, prefill_work, self.config.max_seqs_in_prefill, capture_routes
        )

        # _run_prefill returns (GenState, _DecodeOutputs)
        self.gen_state, outputs = new_state
        return outputs

    def _prefill_prompts(
        self,
        requests: Sequence[Request],
        route_chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> PrefillWork | None:
        """Pack prompt work into a single PrefillWork structure for downstream device execution."""

        decode_state = self.gen_state.decode_state
        max_seqs_in_prefill = self.config.max_seqs_in_prefill
        max_prefill_size = self.config.max_prefill_size or self.model.Pos.size
        max_seq_len = decode_state.tokens.axis_size("position")
        max_slots = decode_state.max_seqs
        if self.gen_state.window_pages is not None:
            max_slots = min(max_slots, max_seqs_in_prefill)

        queue_tokens = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)
        queue_slot_ids = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)
        queue_pos_ids = np.full((max_prefill_size,), INVALID, dtype=jnp.int32)

        work_slot_ids = np.full((max_slots,), INVALID, dtype=np.int32)
        clone_targets = np.full((max_slots,), INVALID, dtype=np.int32)
        prompt_rows = np.full((max_slots,), INVALID, dtype=np.int32)
        prompt_tokens = np.full((min(max_slots, max_seqs_in_prefill), max_seq_len), INVALID, dtype=np.int32)
        prompt_lengths = np.zeros((max_slots,), dtype=np.int32)

        prefix_pages = None
        table = decode_state.page_table
        if self.config.max_retained_prompts:
            prefix_pages = np.full((max_slots, table.pages_per_seq), INVALID, dtype=np.int32)
            reserved, shared = self._reserved_page_usage(table)
            pages_left = table.num_pages - reserved
        window_pages = self.gen_state.window_pages
        residency = (
            None
            if window_pages is None or not self._cached_prompts
            else np.asarray(jax.device_get(window_pages.logical_to_window))
        )

        stop_tokens_template = decode_state.stop_tokens
        max_num_tokens = np.zeros((max_slots,), dtype=np.int32)
        temperatures = np.zeros((max_slots,), dtype=np.float32)
        top_ps = np.ones((max_slots,), dtype=np.float32)
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

            if num_primary >= max_seqs_in_prefill:
                break

            if len(self.free_slots) < request.n_generations:
                if max_seqs_in_prefill < request.n_generations:
                    raise RuntimeError(
                        f"Request {request.request_id} asked for {request.n_generations} generations, "
                        f"but max_seqs_in_prefill={max_seqs_in_prefill} is too small to accommodate. "
                        "Increase max_seqs_in_prefill or reduce n_generations."
                    )
                break

            this_tokens = np.asarray(seq_tokens, dtype=np.int32)
            reused = 0
            if prefix_pages is not None:
                pages, reused_routes = self._matching_prompt_pages(
                    this_tokens, residency, require_routes=route_chunks is not None
                )
                reused = len(pages) * self.config.page_size
                private = self._request_pages(request) - len(pages)
                needed = private + int(np.count_nonzero(~shared[pages]))
                if needed > pages_left:
                    break
            end = offset + len(seq_tokens) - reused
            if end > max_prefill_size:
                break

            slot_id = int(self.free_slots.pop())
            if prefix_pages is not None:
                self._page_reservations[request.request_id] = _PageReservation(private, pages)
                shared[pages] = True
                pages_left -= needed
                prefix_pages[total_new, : len(pages)] = pages
                if route_chunks is not None and reused_routes is not None:
                    route_chunks[request.request_id, 0] = _route_prefix(reused_routes, reused)
            queue_tokens[offset:end] = this_tokens[reused:]
            queue_slot_ids[offset:end] = slot_id
            queue_pos_ids[offset:end] = np.arange(reused, len(seq_tokens), dtype=np.int32)

            prefill_idx = total_new
            if prefill_idx >= max_slots:
                raise RuntimeError("Exceeded maximum slot instructions while building prefill work.")

            work_slot_ids[prefill_idx] = slot_id
            clone_targets[prefill_idx] = INVALID
            prompt_lengths[prefill_idx] = len(seq_tokens)
            prompt_rows[prefill_idx] = num_primary
            prompt_tokens[num_primary, : len(seq_tokens)] = this_tokens

            max_num_tokens[prefill_idx] = min(int(seq_params.max_num_tokens), self.config.max_seq_len)
            temperatures[prefill_idx] = np.asarray(seq_params.temperature, dtype=np.float32).item()
            top_ps[prefill_idx] = np.asarray(seq_params.top_p, dtype=np.float32).item()
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

            offset = end
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
                    max_num_tokens[clone_idx] = max_num_tokens[prefill_idx]
                    temperatures[clone_idx] = np.asarray(child_params.temperature, dtype=np.float32).item()
                    top_ps[clone_idx] = np.asarray(child_params.top_p, dtype=np.float32).item()
                    prng_keys[clone_idx] = np.asarray(child_params.key, dtype=np.uint32)
                    if stop_tokens is not None:
                        stop_tokens[clone_idx] = stop_tokens[prefill_idx]

                    self.local_map[child_slot_id] = (rid, k)
                    self.sequences.setdefault(rid, {})[k] = child_slot_id

                    total_new += 1

        if offset == 0:
            return None

        if self._cached_prompts:
            self._trim_retained_prompts(self._retained_page_refs(table))
        capacity = _token_bucket(offset, self.config.prefill_bucket_sizes, max_prefill_size)
        prefill_queue = TokenQueue(
            queued_tokens=hax.named(jnp.asarray(queue_tokens[:capacity], dtype=jnp.int32), axis="position"),
            queued_slot_ids=hax.named(jnp.asarray(queue_slot_ids[:capacity], dtype=jnp.int32), axis="position"),
            queued_pos_ids=hax.named(jnp.asarray(queue_pos_ids[:capacity], dtype=jnp.int32), axis="position"),
            num_queued_tokens=jnp.array(offset, dtype=jnp.int32),
        )

        return PrefillWork(
            queue=prefill_queue,
            new_num_seqs=jnp.array(total_new, dtype=jnp.int32),
            has_clones=total_new > num_primary,
            new_slot_ids=hax.named(jnp.asarray(work_slot_ids, dtype=jnp.int32), axis="seq"),
            clone_targets=hax.named(jnp.asarray(clone_targets, dtype=jnp.int32), axis="seq"),
            prompt_tokens=hax.named(jnp.asarray(prompt_tokens, dtype=jnp.int32), axis=("seq", "position")),
            prompt_lengths=hax.named(jnp.asarray(prompt_lengths, dtype=jnp.int32), axis="seq"),
            prompt_rows=hax.named(jnp.asarray(prompt_rows), axis="seq"),
            cached_pages=None if prefix_pages is None else hax.named(jnp.asarray(prefix_pages), ("seq", "page")),
            seq_params=SeqDecodingParams(
                max_num_tokens=jnp.asarray(max_num_tokens, dtype=jnp.int32),
                stop_tokens=(
                    None
                    if stop_tokens is None
                    else hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("seq", "stop_seq", "position"))
                ),
                temperature=jnp.asarray(temperatures, dtype=jnp.float32),
                top_p=jnp.asarray(top_ps, dtype=jnp.float32),
                key=jnp.asarray(prng_keys, dtype=jnp.uint32),
            ),
        )

    @property
    def supports_route_capture(self) -> bool:
        """Whether generation can retain this model's selected expert IDs."""
        return isinstance(self.model, SnowballLMHeadModel)

    def _validate_generation_requests(self, requests: Sequence[Request]) -> None:
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Request IDs must be unique within a generation batch")
        max_prefill = self.config.max_prefill_size or self.model.Pos.size
        if any(not 0 < len(request.prompt_tokens) <= max_prefill for request in requests):
            raise ValueError("Prompts must be nonempty and fit within max_prefill_size")
        if any(len(request.prompt_tokens) >= self.config.max_seq_len for request in requests):
            raise ValueError("Prompts must leave room for generation within max_seq_len")
        for request in requests:
            stops = request.decode_params.stop_tokens
            if stops is not None and (
                stops.axis_size("stop_seq") > self.config.max_stop_seqs
                or stops.axis_size("position") > self.config.max_stop_tokens
            ):
                raise ValueError("Requested stop-token configuration exceeds service capacity")
        if any(request.n_generations < 1 for request in requests):
            raise ValueError("Every request must ask for at least one generation")

        if self.config.max_retained_prompts:
            pages = self.gen_state.decode_state.page_table.num_pages
            if any(self._request_pages(request) > pages for request in requests):
                raise ValueError("A request exceeds the full KV page budget")
        max_needed = max(int(r.n_generations) for r in requests)
        if self.gen_state.window_pages is not None and max_needed != 1:
            raise ValueError("A compact window cache requires one generation per request")
        if max_needed > int(self.gen_state.decode_state.page_table.max_seqs):
            raise ValueError(
                f"Total sequences needed ({max_needed}) exceeds max_seqs ({self.gen_state.decode_state.page_table.max_seqs})."
                "Decompose your request into smaller batches or increase max_seqs when building the service."
            )

    def generate(
        self, requests: Sequence[Request], step_callback=None, *, capture_routes: bool = False
    ) -> GenerationResult:
        """Generate each request and return choices in input order."""
        results: dict[int, GenerationResult] = {}
        self.generate_continuously(
            requests,
            on_complete=results.__setitem__,
            step_callback=step_callback,
            capture_routes=capture_routes,
        )
        tokens = []
        logprobs = []
        finish_reasons: list[FinishReason] = []
        expert_routes = [] if capture_routes else None
        for request in requests:
            result = results[request.request_id]
            tokens.extend(result.tokens)
            assert result.logprobs is not None
            logprobs.extend(result.logprobs)
            finish_reasons.extend(result.finish_reasons)
            if expert_routes is not None:
                assert result.expert_routes is not None
                expert_routes.extend(result.expert_routes)
        return GenerationResult(tokens, logprobs, sum(map(len, tokens)), finish_reasons, expert_routes)

    def generate_continuously(
        self,
        requests: Sequence[Request],
        *,
        on_complete: Callable[[int, GenerationResult], None],
        poll_requests: Callable[[int], Sequence[Request]] | None = None,
        on_tokens: Callable[[int, int, list[int]], None] | None = None,
        step_callback: Callable[[int], None] | None = None,
        should_abort: Callable[[int], bool] | None = None,
        capture_routes: bool = False,
    ) -> None:
        """Admit more requests between rounds and publish each response when it finishes.

        The nonblocking poll receives free sequence capacity. It returns requests
        with unique IDs and at most that many choices in total. When no work
        remains, the loop returns and the caller can wait for another request.
        The token callback receives the request ID, choice, and new tokens in order.
        At each scheduling boundary, should_abort can terminate one request
        and publish its partial choices while leaving peers in the decode queue.
        """
        if not requests:
            return
        if capture_routes and not self.supports_route_capture:
            raise ValueError("Expert route capture requires a Snowball model")

        self._validate_generation_requests(requests)
        if not self.config.max_retained_prompts or self.local_map:
            self.reset()
        route_chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] | None = (
            {} if capture_routes else None
        )
        pending: list[Request] = []
        outstanding: dict[int, Request] = {}
        seen_ids: set[int] = set()

        def admit(incoming: Sequence[Request]) -> None:
            for request in incoming:
                request_id = request.request_id
                if request_id in seen_ids:
                    raise ValueError("Request IDs must be unique during continuous generation")
                seen_ids.add(request_id)
                outstanding[request_id] = request
                pending.append(request)
                self.results[request_id] = {
                    choice: DecodeResult(id=request_id, choice=choice, token_list=[])
                    for choice in range(request.n_generations)
                }

        def publish_finished() -> None:
            finished = [
                request_id
                for request_id in outstanding
                if all(result.done for result in self.results[request_id].values())
            ]
            for request_id in finished:
                request = outstanding.pop(request_id)
                choices = self.results.pop(request_id)
                self.sequences.pop(request_id, None)
                self._page_reservations.pop(request_id, None)
                tokens = [choices[i].token_list for i in range(request.n_generations)]
                logprobs = [choices[i].logprobs or [] for i in range(request.n_generations)]
                routes = None
                if route_chunks is not None:
                    routes = [
                        self._assemble_routes(route_chunks, request, choice, len(choice_tokens))
                        for choice, choice_tokens in enumerate(tokens)
                    ]
                    for choice in range(request.n_generations):
                        route_chunks.pop((request_id, choice), None)
                finish_reasons = [choices[i].finish_reason for i in range(request.n_generations)]
                on_complete(
                    request_id, GenerationResult(tokens, logprobs, sum(map(len, tokens)), finish_reasons, routes)
                )

        def abort_requested() -> bool:
            if should_abort is None:
                return False
            cancelled = {request_id for request_id in outstanding if should_abort(request_id)}
            if not cancelled:
                return False

            pending[:] = [request for request in pending if request.request_id not in cancelled]
            self._abort_requests(cancelled)
            publish_finished()
            return True

        admit(requests)
        decode_iteration = 0
        while outstanding:
            refill_pending = self.config.max_retained_prompts > 0 and decode_iteration > 0
            if poll_requests is not None and (not pending or refill_pending):
                # Pending choices already claim slots even though their prompts have not run.
                available = len(self.free_slots) - sum(request.n_generations for request in pending)
                incoming = poll_requests(available) if available > 0 else ()
                if incoming:
                    self._validate_generation_requests(incoming)
                    if sum(request.n_generations for request in incoming) > available:
                        raise ValueError("The request poll returned more choices than available slots")
                    admit(incoming)
            abort_requested()
            if not outstanding:
                break
            prefill_outputs = None
            if pending and self.free_slots:
                prefill_outputs = self._prefill_batch(pending, capture_routes, route_chunks)
                if prefill_outputs is not None:
                    admitted = [request for request in pending if request.request_id in self.sequences]
                    self._extract_outputs(prefill_outputs, route_chunks, on_tokens, admitted)
                    pending = [request for request in pending if request.request_id not in self.sequences]
                    publish_finished()
                    if poll_requests is None or not self.local_map:
                        continue

            if step_callback is not None:
                step_callback(decode_iteration)
            if (prefill_outputs is not None or step_callback is not None) and abort_requested():
                continue
            token_capacity = _token_bucket(
                len(self.local_map), self.config.decode_bucket_sizes, self.config.imputed_max_tokens_per_round
            )
            started = time.monotonic()
            self.gen_state, outputs = _run_generation_loop(
                self.gen_state,
                self.model,
                self.sampler,
                token_capacity,
                self.config.max_rounds,
                capture_routes,
            )
            generated = self._extract_outputs(outputs, route_chunks, on_tokens)
            publish_finished()
            logger.debug("Decode iteration generated %d tokens in %.3fs", generated, time.monotonic() - started)
            decode_iteration += 1
            if generated == 0 and outstanding:
                raise RuntimeError("Inference made no progress with unfinished requests")

    def _abort_requests(self, cancelled: set[int]) -> None:
        slots = [slot for slot, (request_id, _) in self.local_map.items() if request_id in cancelled]
        if slots:
            mask = np.zeros(self.gen_state.decode_state.max_seqs, dtype=bool)
            mask[slots] = True
            decode_state = _abort_slots(self.gen_state.decode_state, jnp.asarray(mask))
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)
            for slot in slots:
                del self.local_map[slot]
            self.free_slots.extend(slots)

        for request_id in cancelled:
            for choice in self.results[request_id].values():
                if not choice.done:
                    choice.finish_reason = FinishReason.ABORT

    def write_kernel_jaxprs(self, path, log_artifacts: bool = True):
        """
        Write out jaxpr and hlo for the generation loop to the given path.
        """
        traced = _run_generation_loop.trace(
            self.gen_state,
            self.model,
            self.sampler,
            # TODO: tune max_tokens_per_round
            self.config.imputed_max_tokens_per_round,
            self.config.max_rounds,
        )
        with open_url(os.path.join(path, "gen_loop.jaxpr.txt.gz"), "w", compression="infer") as f:
            f.write(str(traced.jaxpr))
        with open_url(os.path.join(path, "gen_loop.hlo.txt.gz"), "w", compression="infer") as f:
            f.write(traced.lower().as_text())

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
                has_clones=False,
                new_slot_ids=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                clone_targets=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                prompt_tokens=hax.zeros({"seq": max_slots, "position": max_len}, dtype=jnp.int32),
                prompt_lengths=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
                seq_params=SeqDecodingParams(
                    max_num_tokens=jnp.zeros(max_slots, dtype=jnp.int32),
                    stop_tokens=None,
                    temperature=jnp.zeros(max_slots, dtype=jnp.float32),
                    top_p=jnp.ones(max_slots, dtype=jnp.float32),
                    key=jnp.zeros((max_slots, 2), dtype=jnp.uint32),
                ),
            )

        prefill_traced = _run_prefill.trace(
            self.gen_state,
            self.model,
            self.sampler,
            eqx.filter_eval_shape(_create_dummy_work),
            self.config.max_seqs_in_prefill,
        )
        with open_url(os.path.join(path, "run_prefill.jaxpr.txt.gz"), "w", compression="infer") as f:
            f.write(str(prefill_traced.jaxpr))
        with open_url(os.path.join(path, "run_prefill.hlo.txt.gz"), "w", compression="infer") as f:
            f.write(prefill_traced.lower().as_text())

        if log_artifacts:
            levanter.tracker.current_tracker().log_artifact(path, name="generation_kernels")
            logger.info(f"Written trace info to {path} and logged artifacts")
        else:
            logger.info(f"Written trace info to {path}")

    def _assemble_routes(
        self,
        chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]],
        request: Request,
        choice: int,
        generated: int,
    ) -> GeneratedExpertRoutes:
        assert isinstance(self.model, SnowballLMHeadModel)
        config = self.model.config
        prompt_length = len(request.prompt_tokens)
        length = prompt_length + generated
        dtype = _route_dtype(self.model)
        experts = np.zeros((length, config.num_layers, config.num_experts_per_token), dtype=dtype)
        valid = np.zeros(length, dtype=bool)
        for positions, ids in chunks.get((request.request_id, 0), ()):
            prompt = positions < prompt_length
            experts[positions[prompt]] = ids[prompt]
            valid[positions[prompt]] = True
        for positions, ids in chunks.get((request.request_id, choice), ()):
            experts[positions] = ids
            valid[positions] = True
        return GeneratedExpertRoutes(experts.transpose(1, 0, 2), valid)

    def _extract_outputs(
        self,
        pending_outputs: _DecodeOutputs | None,
        route_chunks: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] | None = None,
        on_tokens: Callable[[int, int, list[int]], None] | None = None,
        new_prompts: Sequence[Request] = (),
    ) -> int:
        """Append newly available tokens into outputs per (request_id, child_id).

        Returns number of new tokens appended.
        """
        if pending_outputs is None:
            return 0

        # Pull the entire buffer in one host op
        pending_outputs = jax.device_get(pending_outputs)
        if pending_outputs.routes is not None and route_chunks is not None:
            routes = pending_outputs.routes
            count = int(routes.count)
            slots = routes.slots.array[:count]
            for slot in np.unique(slots):
                request = self.local_map[int(slot)]
                take = slots == slot
                chunk = (routes.positions.array[:count][take], routes.experts.array[:count][take])
                route_chunks.setdefault(request, []).append(chunk)
        n = int(pending_outputs.num_tokens)
        reasons = pending_outputs.finish_reasons.array
        finished_slots = reasons != FinishReason.RUNNING
        toks_arr = pending_outputs.tokens.array
        sids_arr = pending_outputs.slot_ids.array

        token_updates: dict[tuple[int, int], list[int]] = {}
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
            if on_tokens is not None:
                token_updates.setdefault(info, []).append(tok)
            if pending_outputs.logprobs is not None:
                dr.logprobs.append(float(pending_outputs.logprobs.array[i]))
            dr.tokens_decoded += 1
            appended += 1

        if on_tokens is not None:
            for (request_id, choice), tokens in token_updates.items():
                on_tokens(request_id, choice, tokens)

        if new_prompts and self.config.max_retained_prompts:
            self._retain_batch_prompts(new_prompts, route_chunks)

        # Finish outputs before releasing slots, so a new request can reuse them.
        released_slots = []
        for local_slot, is_done in enumerate(finished_slots):
            if not bool(is_done):
                continue
            info = self.local_map.get(local_slot)
            if info is None:
                continue
            rid, cid = info
            dr = self.results.setdefault(rid, {}).setdefault(cid, DecodeResult(id=rid, choice=cid, token_list=[]))
            dr.finish_reason = FinishReason(int(reasons[local_slot]))
            del self.local_map[local_slot]
            released_slots.append(local_slot)

            if self.tokenizer is not None and logger.isEnabledFor(logging.DEBUG):
                try:
                    full_text = self.tokenizer.decode(dr.token_list, skip_special_tokens=False)
                    logger.debug(f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): '{full_text}'")
                except Exception as e:
                    logger.error(
                        f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): <decode_error: {e}>"
                    )

        if released_slots:
            decode_state = _release_finished_slots(self.gen_state.decode_state, jnp.asarray(finished_slots))
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)
            self.free_slots.extend(released_slots)
        logger.debug(
            "extract: appended=%d (drained=%d) unmapped=%d finished=%d", appended, n, unmapped, len(released_slots)
        )

        return appended
