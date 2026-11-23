# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

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

from levanter.kernels.pallas.autotune_utils import named_sharding_of

import levanter.tracker
from levanter.inference.jit_scheduler import (
    DecodeState,
    FinishReason,
    SeqDecodingParams,
    SequenceTable,
    TokenQueue,
    _DecodeOutputs,
)
from levanter.inference.page_table import PageBatchInfo, PageTable, kv_pages_for_request
from levanter.inference.snowball_replay_payload import (
    _RequestRows,
    _SnowballDecodeOutputs,
    _SnowballReplayCollector,
    _SnowballRouteBuffer,
)
from levanter.inference.utils import INVALID, is_valid
from levanter.inference.window_page_map import WindowPageMap
from levanter.layers.attention import AttentionMask
from levanter.layers.kv_cache import KvPageCache, ListCache, PageCache, StackedKvPageCache
from levanter.kernels.logsumexp import batch_invariant_logsumexp
from levanter.layers.sampler import Sampler
from levanter.models.lm_model import LmHeadModel
from levanter.models.snowball import is_long_attention_layer, SnowballLMHeadModel
from levanter.utils.jax_utils import estimated_free_device_memory, sharded_tree_size

logger = logging.getLogger(__name__)

TokenCallback = Callable[[int, int, list[int]], None]


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

    return_policy_logprobs: bool = False
    """Collect untransformed scores for sampled tokens alongside their sampling scores."""

    # You probably don't need to tune the knobs below

    max_seqs: int = 256
    """Maximum concurrent sequences (local slots)."""

    max_pages: Optional[int] = None
    """Total number of KV pages available. If None, inferred from :attr:`hbm_utilization`."""

    window_cache_max_pages: int | None = None
    """Separate page capacity for Snowball sliding-window layers, in the configured KV dtype."""

    window_cache_retained_spans: int = 1
    """Sliding-window spans to retain per cached prompt, within the available page budget."""

    compute_dtype: jnp.dtype = jnp.bfloat16
    """KV cache dtype. Default bfloat16 for performance/accuracy balance."""

    max_queued_tokens: int = 512
    """Capacity of the token queue used between sampling and decode packing."""

    max_seqs_in_prefill: int = 16
    """Maximum number of sequences to batch in prefill before flushing."""

    # Prefill buffer sizing
    max_prefill_size: Optional[int] = None
    """Maximum tokens packed into a prefill batch. Defaults to the page table's sequence limit."""

    prefill_bucket_sizes: tuple[int, ...] = ()
    """Optional, increasing prefill buffer sizes below max_prefill_size.

    Each packed batch uses the smallest fitting size. Include sizes divisible by the model's
    token mesh size. The largest prefill always uses max_prefill_size.
    """
    prefix_cache: Literal["disabled", "last_prompt", "recent_prompts"] = "disabled"
    """Reuse full prompt pages for Snowball serving.

    With one sequence slot, recent_prompts reserves one context for generation. With multiple
    slots it evicts the oldest prefixes as live requests reserve pages.
    """

    max_retained_prompts: int | None = None
    """Packed prefix index capacity, independent of the KV page budget. Defaults to max_seqs."""

    # Decode loop knobs
    max_tokens_per_round: int | None = None
    """Pack size for each decode loop iteration. If None, set to max_seqs """
    decode_bucket_sizes: tuple[int, ...] = ()
    """Increasing decode capacities below max_tokens_per_round, aligned to the token mesh."""

    def __post_init__(self):
        # this one is only required because of clones. If we really care, we could relax this
        if self.max_queued_tokens < self.max_seqs:
            raise ValueError("max_queued_tokens must be >= max_seqs")

        if self.max_queued_tokens < self.imputed_max_tokens_per_round:
            raise ValueError("max_queued_tokens must be >= max_tokens_per_round")

        if self.max_queued_tokens < self.max_seqs_in_prefill:
            raise ValueError("max_queued_tokens must be >= max_seqs_in_prefill")

        if any(size <= 0 for size in self.prefill_bucket_sizes) or self.prefill_bucket_sizes != tuple(
            sorted(set(self.prefill_bucket_sizes))
        ):
            raise ValueError("prefill_bucket_sizes must be positive and strictly increasing")
        if (
            self.prefill_bucket_sizes
            and self.max_prefill_size is not None
            and self.prefill_bucket_sizes[-1] >= self.max_prefill_size
        ):
            raise ValueError("prefill_bucket_sizes must be below max_prefill_size")
        if self.prefix_cache == "last_prompt" and self.max_seqs != 1:
            raise ValueError("last_prompt requires max_seqs=1")
        if self.max_retained_prompts is not None and (
            self.max_retained_prompts < 1 or self.prefix_cache != "recent_prompts" or self.max_seqs <= 1
        ):
            raise ValueError("max_retained_prompts requires a positive count and packed recent_prompts caching")
        if self.window_cache_retained_spans < 1:
            raise ValueError("window_cache_retained_spans must be positive")
        if self.decode_bucket_sizes != tuple(sorted(set(self.decode_bucket_sizes))) or any(
            not 0 < size < self.imputed_max_tokens_per_round for size in self.decode_bucket_sizes
        ):
            raise ValueError("decode_bucket_sizes must be increasing and below max_tokens_per_round")

    @property
    def imputed_max_tokens_per_round(self) -> int:
        """Return explicit `max_tokens_per_round` or default to `max_seqs` when unset."""
        return self.max_tokens_per_round if self.max_tokens_per_round is not None else self.max_seqs

    @property
    def max_pages_per_seq(self) -> int:
        return (self.max_seq_len + self.page_size - 1) // self.page_size


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
    policy_logprobs: list[float] = field(default_factory=list)

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
        self, sequences: SequenceTable, batch: PageBatchInfo
    ) -> tuple["GenState", PageBatchInfo | None]:
        """Allocate and translate sliding-window pages when a separate pool is configured."""
        if self.window_pages is None:
            return self, None
        window_pages, window_batch = self.window_pages.remap(sequences, batch)
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
    cache = cache.copy_sequence(parent_local_id, child_local_id)

    new_state = dataclasses.replace(state, decode_state=decode_state, cache=cache)
    return new_state, child_local_id


class PrefillWork(eqx.Module):
    """Plain data container describing host-side work required for a prefill flush."""

    queue: TokenQueue
    new_num_seqs: jnp.ndarray
    new_slot_ids: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    clone_targets: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    prompt_tokens: ht.i32[NamedArray, "prompt position"]  # type: ignore[name-defined]
    prompt_rows: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    prompt_lengths: ht.i32[NamedArray, "seq"]  # type: ignore[name-defined]
    seq_params: SeqDecodingParams
    cached_pages: NamedArray | None = None
    has_clones: bool = eqx.field(static=True, default=False)


def _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_sample_indices):
    """
    Compute positions of last tokens per sequence inside a packed slice.

    Boundary when absolute pos_id equals the post-allocation seq_len - 1 for that sequence.
    """
    seq_lens_per_seq = seq_lens["seq", slot_ids]
    boundary_mask = is_valid(slot_ids) & is_valid(pos_ids) & (pos_ids == (seq_lens_per_seq - 1))
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


def _sample_tokens(
    sampler: Sampler,
    logits: NamedArray,
    temperatures: NamedArray,
    top_ps: NamedArray,
    keys: jax.Array,
    return_policy_logprobs: bool,
) -> tuple[NamedArray, NamedArray, NamedArray | None]:
    tokens, sampled_logprobs = sampler.sample_positions(logits, temperatures, top_ps=top_ps, key=keys)
    if not return_policy_logprobs:
        return tokens, sampled_logprobs, None

    def raw_logprobs():
        scores = logits.rearrange(("position", sampler.Vocab)).array.astype(jnp.float32)
        selected = jnp.take_along_axis(scores, tokens.array[:, None], axis=-1)[:, 0]
        return selected - batch_invariant_logsumexp(scores)

    # Untransformed sampling already computed the raw scores.
    unscaled = (temperatures == 1.0) | (temperatures == 0.0)
    same_distribution = hax.all(unscaled & (top_ps >= 1.0)).scalar()
    policy_logprobs = jax.lax.cond(same_distribution, lambda: sampled_logprobs.array, raw_logprobs)
    return tokens, sampled_logprobs, hax.named(policy_logprobs, tokens.axes)


def _sample_expert_shards(
    sampler: Sampler,
    logits: NamedArray,
    temperatures: NamedArray,
    top_ps: NamedArray,
    keys: jax.Array,
    return_policy_logprobs: bool,
) -> tuple[NamedArray, NamedArray, NamedArray | None]:
    sharding = named_sharding_of(logits.array)
    if sharding is None or sharding.mesh.empty or not sharding.spec or sharding.spec[0] is None:
        return _sample_tokens(sampler, logits, temperatures, top_ps, keys, return_policy_logprobs)

    row_spec = PartitionSpec(sharding.spec[0])
    logits_spec = PartitionSpec(sharding.spec[0], None)

    def sample(local_logits, local_temperatures, local_top_ps, local_keys):
        rows = hax.Axis("position", local_logits.shape[0])
        sampled, logprobs, policy = _sample_tokens(
            sampler,
            hax.named(local_logits, (rows, logits.resolve_axis(sampler.Vocab))),
            hax.named(local_temperatures, rows),
            hax.named(local_top_ps, rows),
            local_keys,
            return_policy_logprobs,
        )
        return sampled.array, logprobs.array, None if policy is None else policy.array

    sample_rows = jax.shard_map(
        sample,
        mesh=sharding.mesh,
        in_specs=(logits_spec, row_spec, row_spec, row_spec),
        out_specs=(row_spec, row_spec, row_spec if return_policy_logprobs else None),
        check_vma=False,
    )
    row_sharding = sharding.update(spec=row_spec)
    sampled, logprobs, policy = sample_rows(
        jax.reshard(logits.array, sharding.update(spec=logits_spec)),
        jax.reshard(temperatures.array, row_sharding),
        jax.reshard(top_ps.array, row_sharding),
        jax.reshard(keys, row_sharding),
    )
    replicated = sharding.update(spec=PartitionSpec())
    return (
        hax.named(jax.reshard(sampled, replicated), "position"),
        hax.named(jax.reshard(logprobs, replicated), "position"),
        None if policy is None else hax.named(jax.reshard(policy, replicated), "position"),
    )


def _capture_terminal_inputs(
    gen_state: GenState, model: SnowballLMHeadModel, routes: _SnowballRouteBuffer
) -> tuple[GenState, _SnowballRouteBuffer]:
    """Decode finished inputs once, without sampling, while their pages still exist."""

    def capture(state):
        gen_state, routes = state
        ds = gen_state.decode_state
        finished = ds.finished.array & ds.sequences.used_mask.array
        slots = jnp.nonzero(finished, size=ds.max_seqs, fill_value=INVALID)[0].astype(jnp.int32)
        safe_slots = jnp.maximum(slots, 0)
        positions = jnp.where(slots >= 0, ds.seq_lens.array[safe_slots] - 1, INVALID)
        tokens = jnp.where(slots >= 0, ds.tokens.array[safe_slots, jnp.maximum(positions, 0)], INVALID)
        slot_ids = hax.named(slots, "position")
        pos_ids = hax.named(positions, "position")
        ds, info = ds.allocate_for_seq(slot_ids, pos_ids)
        gen_state, window_info = gen_state.remap_window_pages(ds.sequences, info)
        _, cache, captured = model.decode_with_route_captures(
            hax.named(tokens, "position"), gen_state.cache, info, pos_ids, window_batch_info=window_info
        )
        return dataclasses.replace(gen_state, decode_state=ds, cache=cache), routes.append(captured, slot_ids)

    finished = gen_state.decode_state.finished & gen_state.decode_state.sequences.used_mask
    return jax.lax.cond(hax.any(finished).scalar(), capture, lambda state: state, (gen_state, routes))


def _prefill_kernel(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    queue: TokenQueue,
    max_seqs_in_prefill: int,  # static
    capture_routes: bool = False,
    has_clones: bool = False,
    return_policy_logprobs: bool = False,
) -> tuple[GenState, _DecodeOutputs | _SnowballDecodeOutputs]:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    tokens = queue.queued_tokens
    pos_ids = queue.queued_pos_ids
    slot_ids = queue.queued_slot_ids

    decode_state, binfo = gen_state.decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)
    binfo = binfo.with_seq_capacity(max_seqs_in_prefill)

    gen_state, window_info = gen_state.remap_window_pages(decode_state.sequences, binfo)

    seq_lens = decode_state.seq_lens

    sample_indices = _compute_sample_indices(pos_ids, slot_ids, seq_lens, max_seqs_in_prefill)

    # jax.debug.print(
    #     "[_run_prefill] tokens={tokens} slots={slots} pos={pos} seq_lens={lens}",
    #     tokens=tokens.array,
    #     slots=slot_ids.array,
    #     pos=pos_ids.array,
    #     lens=decode_state.seq_lens.array,
    # )
    routes = None
    if capture_routes:
        assert isinstance(model, SnowballLMHeadModel)
        routes = _SnowballRouteBuffer.init(model, tokens.axis_size("position") + decode_state.max_seqs)
        use_selected_rows = model.config.projection_implementation == "native"
        logits, cache, captured = model.decode_with_route_captures(
            tokens,
            gen_state.cache,
            binfo,
            pos_ids,
            sample_indices=sample_indices if use_selected_rows else None,
            window_batch_info=window_info,
        )
        routes = routes.append(captured, slot_ids)
        logits_at_samples = logits if use_selected_rows else _gather_logits(logits, sample_indices)
    elif isinstance(model, SnowballLMHeadModel) and model.config.projection_implementation == "native":
        logits_at_samples, cache = model.decode_for_sampling(
            tokens, gen_state.cache, binfo, pos_ids, sample_indices, window_batch_info=window_info
        )
    else:
        logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)
        logits_at_samples = _gather_logits(logits, sample_indices)

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

    new_tokens, log_probs, policy_logprobs = _sample_tokens(
        sampler, logits_at_samples, temps, top_ps, prng_keys, return_policy_logprobs
    )

    # Update decode_state (also enqueues into the main decode queue)
    decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

    # Initialize outputs buffer and append prefill-sampled tokens
    outputs = _DecodeOutputs.init(
        max_tokens=decode_state.max_seqs * 2,
        max_seqs=decode_state.max_seqs,
        with_logprobs=True,
        with_policy_logprobs=return_policy_logprobs,
    )
    outputs = outputs.append(
        new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finish_reasons, policy_logprobs
    )
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
    if routes is not None:
        assert isinstance(model, SnowballLMHeadModel)
        gen_state, routes = _capture_terminal_inputs(gen_state, model, routes)
        return gen_state, _SnowballDecodeOutputs(outputs, routes)
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

    def primary(i: int, state: GenState) -> GenState:
        slot = work.new_slot_ids.array[i]

        def assign(gs: GenState) -> GenState:
            decode_state, _ = gs.decode_state.reserve_slot(slot)
            pages = None
            if work.cached_pages is not None:
                pages = work.cached_pages["seq", i]
                table = decode_state.page_table
                valid_pages = hax.where(is_valid(pages), pages, table.num_pages)
                refs = table.page_ref_counts.at["page", valid_pages].add(1, mode="drop")
                decode_state = dataclasses.replace(
                    decode_state, page_table=dataclasses.replace(table, page_ref_counts=refs)
                )
            decode_state = decode_state.assign_seq(
                local_slot_id=slot,
                tokens=work.prompt_tokens["prompt", work.prompt_rows.array[i]],
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
    if not work.has_clones:
        return gen_state
    # Full prefix pages must exist before cloning, so parent prefill fills the shared pages.
    decode_state, _ = gen_state.decode_state.allocate_for_seq(
        token_slot_ids=work.queue.queued_slot_ids, token_pos_ids=work.queue.queued_pos_ids
    )
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state)
    return jax.lax.fori_loop(0, num_new, clone, gen_state)


@functools.partial(jax.jit, donate_argnums=(0,))
def _stack_snowball_cache(layers: ListCache[KvPageCache]) -> StackedKvPageCache:
    last = layers[-1].kv_pages
    sharding = named_sharding_of(last.array)

    def restore_heads(layer):
        pages = layer.kv_pages.array
        missing = last.shape["kv_head"] - layer.kv_pages.shape["kv_head"]
        if missing:
            pages = jnp.pad(pages, ((0, 0), (0, 0), (0, missing), (0, 0)))
        if sharding is not None and not sharding.mesh.empty:
            pages = jax.reshard(pages, sharding)
        return pages

    pages = jnp.stack([restore_heads(layer) for layer in layers])
    if sharding is not None and not sharding.mesh.empty:
        pages = jax.reshard(pages, sharding.update(spec=PartitionSpec(None, *sharding.spec)))
    return StackedKvPageCache(hax.named(pages, (hax.Axis("layer", len(layers)), *last.axes)))


@functools.partial(
    jax.jit, donate_argnums=0, static_argnames=("max_seqs_in_prefill", "capture_routes", "return_policy_logprobs")
)
def _run_prefill(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    work: PrefillWork,
    max_seqs_in_prefill: int,
    capture_routes: bool = False,
    return_policy_logprobs: bool = False,
) -> tuple[GenState, _DecodeOutputs | _SnowballDecodeOutputs]:
    gen_state = _apply_prefill_work(gen_state, work)
    return _prefill_kernel(
        gen_state,
        model,
        sampler,
        work.queue,
        max_seqs_in_prefill,
        capture_routes=capture_routes,
        has_clones=work.has_clones,
        return_policy_logprobs=return_policy_logprobs,
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

    new_tokens, log_probs, policy_logprobs = _sample_tokens(
        sampler, logits_this_time, temps, top_ps, prng_keys, outputs.policy_logprobs is not None
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
        cache = cache.copy_sequence(src_slot_id, dst_slot_id)
        return decode_state, cache

    decode_state, cache = jax.lax.fori_loop(0, num_new, copy_pages_for_updated_seq, (decode_state, cache))

    # Enqueue/update tokens for the clone targets (only the first num_new entries will be used)
    decode_state = decode_state.update_tokens(new_tokens, tgt_ids, log_probs, num_new)
    # Discharge processed clones so they are not reprocessed in subsequent flushes
    decode_state = decode_state.discharge_clone(tgt_ids, num_new)
    gen_state = dataclasses.replace(gen_state, decode_state=decode_state, cache=cache)

    # Append clone outputs
    outputs = outputs.append(
        new_tokens, tgt_ids, log_probs, num_new, gen_state.decode_state.finish_reasons, policy_logprobs
    )

    # Device-side release of finished sequences (jit-safe)
    return gen_state, outputs


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_tokens_per_round",
        "max_rounds",
        "capture_routes",
        "replicated_single_token",
        "return_policy_logprobs",
    ),
    donate_argnames=("gen_state",),
)
def _run_generation_loop(
    gen_state: GenState,
    model: LmHeadModel,
    sampler: Sampler,
    max_tokens_per_round: int,
    max_rounds: int,
    capture_routes: bool = False,
    replicated_single_token: bool = False,
    return_policy_logprobs: bool = False,
) -> tuple[GenState, _DecodeOutputs | _SnowballDecodeOutputs]:
    """Run autoregressive generation until all sequences finish or `max_rounds` reached."""

    decode_model = model
    if replicated_single_token:
        # The existing singleton kernels accept the same expert-sharded KV pages.
        assert isinstance(model, SnowballLMHeadModel)
        config = dataclasses.replace(
            model.config, inference_decode_token_layout="replicated", inference_expert_implementation="split"
        )
        decode_model = SnowballLMHeadModel(model.transformer, config)

    def cond(state):
        _gen_state, _outputs, _routes, step = state
        return (
            (step < max_rounds)
            & (_gen_state.decode_state.num_queued_tokens > 0)
            & (~hax.all(_gen_state.decode_state.finished)).scalar()
        )

    def body(state):
        gen_state, outputs, routes, step = state

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

        max_sample_indices = (
            1 if replicated_single_token else min(decode_state.page_table.max_seqs, max_tokens_per_round)
        )
        if isinstance(model, SnowballLMHeadModel):
            # A packed round cannot touch more sequences than it has token rows.
            binfo = binfo.with_seq_capacity(max_sample_indices)

        gen_state, window_info = gen_state.remap_window_pages(decode_state.sequences, binfo)

        # Decode logits and sample new tokens
        if routes is not None:
            assert isinstance(model, SnowballLMHeadModel)
            logits, cache, captured = model.decode_with_route_captures(
                tokens,
                gen_state.cache,
                binfo,
                pos_ids,
                max_active_tokens=1 if decode_state.max_seqs == 1 else None,
                window_batch_info=window_info,
            )
            routes = routes.append(captured, slot_ids)
        elif replicated_single_token:
            logits, cache = decode_model.decode(tokens, gen_state.cache, binfo, pos_ids, max_active_tokens=1)
        elif (
            isinstance(model, SnowballLMHeadModel)
            and (
                model.config.inference_attention_implementation == "gpu"
                or model.config.inference_decode_token_layout == "replicated"
            )
            and decode_state.max_seqs == 1
        ):
            # The generation queue holds at most one sampled token per active sequence.
            logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids, max_active_tokens=1)
        elif window_info is not None:
            assert isinstance(model, SnowballLMHeadModel)
            logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids, window_batch_info=window_info)
        else:
            logits, cache = model.decode(tokens, gen_state.cache, binfo, pos_ids)

        sample_packed_rows = (
            isinstance(model, SnowballLMHeadModel)
            and model.config.projection_implementation == "native"
            and model.config.inference_kv_sharding == "expert"
            and decode_state.max_seqs > 1
            and max_sample_indices == max_tokens_per_round
        )
        if sample_packed_rows:
            # Decode packs one token per sequence. Sample on the owning shard and
            # exchange only sampled tokens and scores instead of vocabulary rows.
            logits_at_samples = logits
            num_new_tokens = binfo.num_new_tokens
            new_slot_ids, new_pos_ids = slot_ids, pos_ids
        else:
            sample_indices = _compute_sample_indices(pos_ids, slot_ids, decode_state.seq_lens, max_sample_indices)
            logits_at_samples = _gather_logits(logits, sample_indices)
            num_new_tokens = hax.sum(sample_indices != INVALID).scalar().astype(jnp.int32)
            new_slot_ids = slot_ids["position", sample_indices]
            new_pos_ids = pos_ids["position", sample_indices]
        prng_keys = decode_state.prng_keys_for(new_slot_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_slot_ids]
        top_ps = decode_state.top_p["seq", new_slot_ids]

        if sample_packed_rows:
            new_tokens, log_probs, policy_logprobs = _sample_expert_shards(
                sampler, logits_at_samples, temps, top_ps, prng_keys, return_policy_logprobs
            )
        else:
            new_tokens, log_probs, policy_logprobs = _sample_tokens(
                sampler, logits_at_samples, temps, top_ps, prng_keys, return_policy_logprobs
            )

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(gen_state, cache=cache, decode_state=decode_state)
        # Append non-stateful outputs for host-side extraction
        outputs = outputs.append(
            new_tokens, new_slot_ids, log_probs, num_new_tokens, decode_state.finish_reasons, policy_logprobs
        )

        # jax.debug.print(
        #     "[gen] step={step} outputs_size={size} queued_after={queued}",
        #     step=step,
        #     size=outputs.num_tokens,
        #     queued=new_gen_state.decode_state.num_queued_tokens,
        # )
        return new_gen_state, outputs, routes, step + 1

    # Allocate an outputs buffer sized for this run
    outputs_buf = _DecodeOutputs.init(
        max_tokens=max(max_tokens_per_round * max_rounds, 1),
        max_seqs=gen_state.decode_state.max_seqs,
        with_logprobs=True,
        with_policy_logprobs=return_policy_logprobs,
    )
    routes = None
    if capture_routes:
        assert isinstance(model, SnowballLMHeadModel)
        routes = _SnowballRouteBuffer.init(model, max_tokens_per_round * max_rounds + gen_state.decode_state.max_seqs)
    split_cache = False
    keep_layers = isinstance(model, SnowballLMHeadModel) and model.config.inference_decode_token_layout == "replicated"
    if (
        isinstance(model, SnowballLMHeadModel)
        and model.config.inference_decode_implementation == "unrolled_split_cache"
        and isinstance(gen_state.cache, StackedKvPageCache)
    ):
        # Update each layer directly through a whole decode round.
        split_cache = True
        stacked = gen_state.cache.kv_pages
        stacked_sharding = named_sharding_of(stacked.array)
        layers = jnp.unstack(stacked.array, axis=0)
        if stacked_sharding is not None and not stacked_sharding.mesh.empty:
            layer_sharding = stacked_sharding.update(spec=PartitionSpec(*stacked_sharding.spec[1:]))
            layers = tuple(jax.reshard(layer, layer_sharding) for layer in layers)
        if keep_layers and model.config.inference_replicate_window_kv:
            heads = 2 * model.config.num_kv_heads
            page, slot, _, embed = stacked.axes[1:]

            def layer_cache(layer, pages):
                if is_long_attention_layer(layer, len(layers)):
                    return KvPageCache(hax.named(pages, stacked.axes[1:]))
                pages = jax.reshard(pages, PartitionSpec(None, None, "model", None))[:, :, :heads]
                return KvPageCache(hax.named(pages, (page, slot, hax.Axis("kv_head", heads), embed)))

            caches = ListCache(tuple(layer_cache(i, pages) for i, pages in enumerate(layers)))
        else:
            caches = ListCache(tuple(KvPageCache(hax.named(layer, stacked.axes[1:])) for layer in layers))
        gen_state = dataclasses.replace(gen_state, cache=caches)
    init_state = (gen_state, outputs_buf, routes, jnp.array(0, dtype=jnp.int32))
    final_gen_state, final_outputs, routes, _ = jax.lax.while_loop(cond, body, init_state)
    if split_cache and not keep_layers:
        assert isinstance(final_gen_state.cache, ListCache)
        final_gen_state = dataclasses.replace(final_gen_state, cache=_stack_snowball_cache(final_gen_state.cache))
    if routes is not None:
        assert isinstance(model, SnowballLMHeadModel)
        final_gen_state, routes = _capture_terminal_inputs(final_gen_state, model, routes)
        return final_gen_state, _SnowballDecodeOutputs(final_outputs, routes)
    return final_gen_state, final_outputs


@dataclass
class GenerationResult:
    tokens: list[list[int]]
    logprobs: list[list[float]] | None
    total_generated: int
    finish_reasons: list[FinishReason]
    policy_logprobs: list[list[float]] | None = None


CompletionCallback = Callable[[int, GenerationResult], None]
RequestSource = Callable[[int], Sequence[Request]]


def _result_for_choices(choices: Sequence[DecodeResult]) -> GenerationResult:
    return GenerationResult(
        tokens=[choice.token_list for choice in choices],
        logprobs=[choice.logprobs for choice in choices],
        total_generated=sum(len(choice.token_list) for choice in choices),
        finish_reasons=[choice.finish_reason for choice in choices],
        policy_logprobs=(
            [choice.policy_logprobs for choice in choices]
            if any(choice.policy_logprobs for choice in choices)
            else None
        ),
    )


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
    model: LmHeadModel
    tokens: np.ndarray
    pages: np.ndarray
    route_rows: tuple[_RequestRows, ...] | None = None


@dataclass(frozen=True)
class _PageReservation:
    private_pages: int
    shared_pages: np.ndarray | None = None


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
        window_pages = None
        if config.window_cache_max_pages is not None:
            assert isinstance(model, SnowballLMHeadModel)
            window_pages = WindowPageMap.init(
                decode_state.page_table.num_pages, config.window_cache_max_pages, model.config.sliding_window
            )
        self.gen_state = GenState(cache=cache, decode_state=decode_state, window_pages=window_pages)
        self._initial_decode_state = decode_state
        # Impute max_prefill_size if not set
        if config.max_prefill_size is None:
            config = dataclasses.replace(config, max_prefill_size=decode_state.page_table.max_len_per_seq)
        if config.prefix_cache != "disabled" and not isinstance(model, SnowballLMHeadModel):
            raise ValueError("The prompt cache requires Snowball")
        self._retained_prompt_limit = 1
        self._retain_packed_prompts = config.max_seqs > 1 and config.prefix_cache == "recent_prompts"
        table = decode_state.page_table
        if config.prefix_cache == "recent_prompts":
            self._retained_prompt_limit = table.num_pages // table.pages_per_seq - 1
            if self._retained_prompt_limit < 1:
                raise ValueError("The recent-prompts cache requires max_pages for at least two full sequences")
            if self._retain_packed_prompts:
                self._retained_prompt_limit = config.max_retained_prompts or config.max_seqs
        self.config = config
        self._cached_prompts: list[_CachedPrompt] = []
        self._page_reservations: dict[int, _PageReservation] = {}
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
        axis_resources: ResourceMapping | None = None,
    ) -> "InferenceEngine":
        """Build an engine using a EngineConfig for sizing knobs."""
        if isinstance(model, SnowballLMHeadModel) and model.config.inference_kv_sharding == "expert":
            axis_resources = {**(axis_resources or {}), "kv_head": "expert"}
        if config.max_pages is None:
            inferred_pages = _infer_max_pages_from_hbm(model, config, axis_resources)
            config = dataclasses.replace(config, max_pages=int(inferred_pages))

        max_pages_per_seq = config.max_pages_per_seq

        assert config.max_pages is not None

        table = PageTable.init(config.max_pages, config.max_seqs, config.page_size, max_pages_per_seq)
        cache: PageCache
        prefill_capacity = config.max_prefill_size or table.max_len_per_seq
        use_layer_caches = (
            isinstance(model, SnowballLMHeadModel)
            and config.max_seqs > 1
            and model.config.inference_decode_implementation == "unrolled_split_cache"
            and prefill_capacity <= model.config.inference_max_unrolled_prefill_tokens
        )
        window_capacity = config.window_cache_max_pages
        if window_capacity is not None:
            if (
                not use_layer_caches
                or not isinstance(model, SnowballLMHeadModel)
                or model.config.projection_implementation != "native"
            ):
                raise ValueError("Sliding-window KV requires split per-layer Snowball buffers and native projection")
            min_window_pages = _window_working_set_pages(config, table, model.config.sliding_window, config.max_seqs)
            if not min_window_pages <= window_capacity < table.num_pages:
                raise ValueError(f"Sliding-window KV capacity must be in [{min_window_pages}, {table.num_pages})")
        if use_layer_caches:
            assert isinstance(model, SnowballLMHeadModel)
            allocate_layer = hax.named_jit(model.initial_layer_cache, axis_resources=axis_resources)
            layers = len(model.transformer.blocks)
            spec = table.spec()
            window_spec = spec if window_capacity is None else dataclasses.replace(spec, num_pages=window_capacity)
            cache = ListCache(
                tuple(
                    allocate_layer(
                        spec if is_long_attention_layer(layer, layers) else window_spec,
                        dtype=config.compute_dtype,
                    )
                    for layer in range(layers)
                )
            )
        else:
            cache = hax.named_jit(model.initial_cache, axis_resources=axis_resources)(
                table.spec(), dtype=config.compute_dtype
            )
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
        """
        self.gen_state = eqx.filter_jit(GenState.reset, donate="all", keep_unused=True)(self.gen_state)
        self._cached_prompts.clear()
        self._clear_requests()

    def _clear_requests(self) -> None:
        self.free_slots = list(range(int(self.gen_state.decode_state.max_seqs)))
        self.local_map.clear()
        self.sequences.clear()
        self.results = {}
        self._page_reservations.clear()

    def _request_pages(self, request: Request) -> int:
        return kv_pages_for_request(
            len(request.prompt_tokens),
            int(request.decode_params.max_num_tokens),
            request.n_generations,
            self.config.page_size,
        )

    def score_token_logprobs(self, token_ids: Sequence[int], top_k: int) -> TokenSequenceLogprobs:
        """Score a token sequence under this engine's model."""
        return score_token_sequence_logprobs(self.model, token_ids, top_k)

    def _prefill_batch(
        self, batch: Sequence[Request], replay: _SnowballReplayCollector | None = None
    ) -> _DecodeOutputs | _SnowballDecodeOutputs | None:
        """Admit a batch from the head of the queue that fits in free slots/pages.

        Returns the decode outputs for the admitted prefill batch, or None if no work was admitted.
        """
        # Build a single PrefillWork description and run prefill exactly once
        prefill_work = self._prefill_prompts(batch, replay)
        if prefill_work is None:
            return None
        if (
            isinstance(self.model, SnowballLMHeadModel)
            and isinstance(self.gen_state.cache, ListCache)
            and (
                self.config.max_seqs == 1
                or prefill_work.queue.queued_tokens.axis_size("position")
                > self.model.config.inference_max_unrolled_prefill_tokens
            )
        ):
            self.gen_state = dataclasses.replace(self.gen_state, cache=_stack_snowball_cache(self.gen_state.cache))
        new_state, outputs = _run_prefill(
            self.gen_state,
            self.model,
            self.sampler,
            prefill_work,
            self.config.max_seqs_in_prefill,
            capture_routes=replay is not None,
            return_policy_logprobs=self.config.return_policy_logprobs,
        )

        self.gen_state = new_state
        if replay is not None and self.config.prefix_cache != "disabled":
            outputs = jax.device_get(outputs)
            assert isinstance(outputs, _SnowballDecodeOutputs)
            replay.append(outputs.routes, self.local_map)
            outputs = outputs.outputs
        if self._retain_packed_prompts:
            self._retain_batch_prompts(batch, replay)
        elif self.config.prefix_cache != "disabled":
            request = batch[0]
            tokens = np.asarray(request.prompt_tokens, dtype=np.int32)
            count = len(tokens) // self.config.page_size
            pages = np.array(jax.device_get(new_state.decode_state.sequences.page_indices.array[0, :count]))
            if np.any((pages < 0) | (pages >= new_state.decode_state.page_table.num_pages)):
                raise ValueError("Prefill returned an invalid full prompt page")
            rows = None if replay is None else replay.prefix(request.request_id, count * self.config.page_size)
            self._cached_prompts.append(_CachedPrompt(self.model, tokens, pages, rows))
            del self._cached_prompts[: -self._retained_prompt_limit]
        return outputs

    def _retained_page_refs(self, table: PageTable) -> np.ndarray:
        refs = np.zeros(table.num_pages, dtype=np.int32)
        for cached in self._cached_prompts:
            np.add.at(refs, cached.pages, 1)
        return refs

    def _reserved_page_usage(self, table: PageTable) -> tuple[int, np.ndarray]:
        shared = np.zeros(table.num_pages, dtype=bool)
        private = 0
        for reservation in self._page_reservations.values():
            private += reservation.private_pages
            if reservation.shared_pages is not None:
                shared[reservation.shared_pages] = True
        return private + int(np.count_nonzero(shared)), shared

    def _retain_batch_prompts(self, requests: Sequence[Request], replay: _SnowballReplayCollector | None) -> None:
        completed_prompts = []
        for request in requests:
            slots = self.sequences.get(request.request_id)
            if slots is None:
                break
            if len(request.prompt_tokens) >= self.config.page_size:
                completed_prompts.append((request.request_id, slots[0], request.prompt_tokens))
        if not completed_prompts:
            return
        state = self.gen_state.decode_state
        before = self._retained_page_refs(state.page_table)
        page_indices = jax.device_get(state.sequences.page_indices.array)
        for rid, slot, prompt in completed_prompts:
            tokens = np.asarray(prompt, dtype=np.int32)
            pages = page_indices[slot, : len(tokens) // self.config.page_size].copy()
            reservation = self._page_reservations[rid]
            num_reused = 0 if reservation.shared_pages is None else len(reservation.shared_pages)
            self._page_reservations[rid] = _PageReservation(reservation.private_pages + num_reused - len(pages), pages)
            rows = None if replay is None else replay.prefix(rid, len(pages) * self.config.page_size)
            self._cached_prompts.append(_CachedPrompt(self.model, tokens, pages, rows))
        self._trim_retained_prompts(before)

    def _trim_retained_prompts(self, before: np.ndarray) -> None:
        state = self.gen_state.decode_state
        table = state.page_table
        # Track ownership by entry so shared physical pages survive eviction of either prefix.
        after = self._retained_page_refs(table)
        reserved, shared = self._reserved_page_usage(table)
        page_budget = table.num_pages - reserved
        pinned = np.count_nonzero((after > 0) & ~shared)
        while pinned > page_budget or len(self._cached_prompts) > self._retained_prompt_limit:
            evict = 0
            if pinned > page_budget:
                # Prompts backed entirely by live pages consume no retention budget.
                evict = next(i for i, cached in enumerate(self._cached_prompts) if not np.all(shared[cached.pages]))
            pages = self._cached_prompts.pop(evict).pages
            np.add.at(after, pages, -1)
            pinned -= np.count_nonzero((after[pages] == 0) & ~shared[pages])
        self._refresh_window_prompt_tails()
        change = after - before
        if not np.any(change):
            return
        delta = jax.device_put(change, table.page_ref_counts.array.sharding)
        table = dataclasses.replace(table, page_ref_counts=table.page_ref_counts + hax.named(delta, "page"))
        self.gen_state = dataclasses.replace(self.gen_state, decode_state=dataclasses.replace(state, page_table=table))

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
    ) -> tuple[np.ndarray, tuple[_RequestRows, ...] | None]:
        pages = np.empty(0, dtype=np.int32)
        routes = None
        window_pages = self.gen_state.window_pages
        page_size = self.config.page_size
        tail_pages = 0 if window_pages is None else (window_pages.window + page_size - 1) // page_size
        for cached in reversed(self._cached_prompts):
            if require_routes and cached.route_rows is None:
                continue
            # Leave one query so prefill can compute next-token logits.
            limit = min(len(tokens) - 1, len(cached.tokens))
            if limit // page_size <= len(pages):
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
                routes = cached.route_rows
        return pages, routes

    def _prefill_prompts(
        self,
        requests: Sequence[Request],
        replay: _SnowballReplayCollector | None,
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
        prompt_rows = np.full((max_slots,), INVALID, dtype=np.int32)
        prompt_tokens = np.full((min(max_slots, max_seqs_in_prefill), max_seq_len), INVALID, dtype=np.int32)
        prompt_lengths = np.zeros((max_slots,), dtype=np.int32)
        prefix_pages = (
            np.full((max_slots, decode_state.page_table.pages_per_seq), INVALID, dtype=np.int32)
            if self.config.prefix_cache != "disabled"
            else None
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
        reserved, shared = self._reserved_page_usage(decode_state.page_table)
        pages_left = decode_state.page_table.num_pages - reserved
        window_pages = self.gen_state.window_pages
        window_residency = (
            None
            if window_pages is None or not self._cached_prompts
            else np.asarray(jax.device_get(window_pages.logical_to_window))
        )

        for request in requests:
            seq_tokens = request.prompt_tokens
            seq_params = request.decode_params
            rid = request.request_id

            if num_primary >= max_seqs_in_prefill or len(self.free_slots) < request.n_generations:
                break

            this_tokens = np.asarray(seq_tokens, dtype=np.int32)
            pages = None
            reused_routes = None
            if prefix_pages is not None and self._cached_prompts:
                pages, reused_routes = self._matching_prompt_pages(
                    this_tokens, window_residency, require_routes=replay is not None
                )
            reused_pages = 0 if pages is None else len(pages)
            private_pages = self._request_pages(request) - reused_pages if max_slots > 1 else 0
            pages_needed = private_pages
            if pages is not None:
                pages_needed += int(np.count_nonzero(~shared[pages]))
            if max_slots > 1 and pages_needed > pages_left:
                break
            reused = reused_pages * self.config.page_size
            end = offset + len(seq_tokens) - reused
            if end > max_prefill_size:
                break

            slot_id = self.free_slots.pop()
            queue_tokens[offset:end] = this_tokens[reused:]
            queue_slot_ids[offset:end] = slot_id
            queue_pos_ids[offset:end] = np.arange(reused, len(seq_tokens), dtype=np.int32)
            offset = end

            if max_slots > 1:
                self._page_reservations[rid] = _PageReservation(private_pages, pages)
                if pages is not None:
                    shared[pages] = True
                pages_left -= pages_needed
            if prefix_pages is not None and pages is not None and reused:
                prefix_pages[total_new, : len(pages)] = pages
                if replay is not None:
                    assert reused_routes is not None
                    replay.reuse_prefix(rid, reused_routes, reused)
                logger.info("Reused %d prompt tokens", reused)
            prefill_idx = total_new
            if prefill_idx >= max_slots:
                raise RuntimeError("Exceeded maximum slot instructions while building prefill work.")

            work_slot_ids[prefill_idx] = slot_id
            clone_targets[prefill_idx] = INVALID
            prompt_lengths[prefill_idx] = len(seq_tokens)
            prompt_rows[prefill_idx] = num_primary
            prompt_tokens[num_primary, : len(seq_tokens)] = this_tokens

            max_num_tokens[prefill_idx] = np.asarray(seq_params.max_num_tokens, dtype=np.int32).item()
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

            self.local_map[slot_id] = (rid, 0)
            self.sequences.setdefault(rid, {})[0] = slot_id

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
                    top_ps[clone_idx] = np.asarray(child_params.top_p, dtype=np.float32).item()
                    prng_keys[clone_idx] = np.asarray(child_params.key, dtype=np.uint32)
                    if stop_tokens is not None:
                        stop_tokens[clone_idx] = stop_tokens[prefill_idx]

                    self.local_map[child_slot_id] = (rid, k)
                    self.sequences.setdefault(rid, {})[k] = child_slot_id

                    total_new += 1

        if offset == 0:
            return None

        if self._retain_packed_prompts and self._cached_prompts:
            # The device attaches all matched pages before allocating any new pages.
            # A matching prefix selected in this work item remains safe to reuse
            # even when its cache entry is evicted here.
            self._trim_retained_prompts(self._retained_page_refs(decode_state.page_table))

        bucket_size = next((size for size in self.config.prefill_bucket_sizes if size >= offset), max_prefill_size)
        prefill_queue = TokenQueue(
            queued_tokens=hax.named(jnp.asarray(queue_tokens[:bucket_size], dtype=jnp.int32), axis="position"),
            queued_slot_ids=hax.named(jnp.asarray(queue_slot_ids[:bucket_size], dtype=jnp.int32), axis="position"),
            queued_pos_ids=hax.named(jnp.asarray(queue_pos_ids[:bucket_size], dtype=jnp.int32), axis="position"),
            num_queued_tokens=jnp.array(offset, dtype=jnp.int32),
        )

        return PrefillWork(
            queue=prefill_queue,
            new_num_seqs=jnp.array(total_new, dtype=jnp.int32),
            new_slot_ids=hax.named(jnp.asarray(work_slot_ids, dtype=jnp.int32), axis="seq"),
            clone_targets=hax.named(jnp.asarray(clone_targets, dtype=jnp.int32), axis="seq"),
            prompt_tokens=hax.named(jnp.asarray(prompt_tokens, dtype=jnp.int32), axis=("prompt", "position")),
            prompt_rows=hax.named(jnp.asarray(prompt_rows), "seq"),
            prompt_lengths=hax.named(jnp.asarray(prompt_lengths, dtype=jnp.int32), axis="seq"),
            cached_pages=None if prefix_pages is None else hax.named(jnp.asarray(prefix_pages), ("seq", "page")),
            has_clones=bool(np.any(clone_targets[:total_new] != INVALID)),
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

    def generate(
        self,
        requests: Sequence[Request],
        step_callback=None,
        *,
        should_abort: Callable[[], bool] | None = None,
        on_tokens: TokenCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> GenerationResult:
        """Generate tokens for a batch of Requests.

        Each Request provides prompt_tokens, decode_params, and n_generations (clones).
        Returns (outputs_per_sequence, total_generated_tokens).

        Args:
            requests: Sequence of generation requests
            step_callback: Optional callback function called at each decode iteration with iteration number
            should_abort: Host predicate checked before prefill and between decode rounds.
                Unfinished choices return their exact partial output with an ABORT reason.
            on_tokens: Called on the host with a request ID, choice index, and newly sampled tokens.
            on_complete: Called once when all choices for a request complete or abort.
        """
        return self._generate(
            requests, step_callback, should_abort=should_abort, on_tokens=on_tokens, on_complete=on_complete
        )

    def _generate(
        self,
        requests: Sequence[Request],
        step_callback=None,
        *,
        should_abort: Callable[[], bool] | None = None,
        replay: _SnowballReplayCollector | None = None,
        on_tokens: TokenCallback | None = None,
        on_complete: CompletionCallback | None = None,
        poll_requests: RequestSource | None = None,
    ) -> GenerationResult:
        if not requests:
            return GenerationResult(tokens=[], logprobs=[], total_generated=0, finish_reasons=[])
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("Request IDs must be unique within a generation batch.")
        requests = [self._prepare_request(request) for request in requests]

        if self.config.prefix_cache != "disabled" and self.config.max_seqs == 1 and len(requests) != 1:
            raise ValueError("Single-slot prefix caching requires one request")

        self._start_generation()
        call_rids = [request.request_id for request in requests]
        for request in requests:
            self._begin_request(request)

        completed: list[int] = []
        streamed_tokens = 0

        def complete_live_request(rid: int, result: GenerationResult) -> None:
            nonlocal streamed_tokens
            assert on_complete is not None
            on_complete(rid, result)
            streamed_tokens += result.total_generated
            completed.append(rid)

        complete = complete_live_request if poll_requests is not None else on_complete
        time_in = time.time()
        aborted = should_abort is not None and should_abort()
        if not aborted:
            decode_outputs = self._prefill_batch(requests, replay)
            self._extract_outputs(decode_outputs, replay, on_tokens, complete)
        initial_prefill_out = time.time()
        logger.info(f"Initial prefill and extraction took {initial_prefill_out - time_in:.3f}s")

        pending = [request for request in requests if request.request_id not in self.sequences]
        decode_iteration = 0
        while not aborted:
            if should_abort is not None and should_abort():
                aborted = True
                break
            if poll_requests is None:
                if all(choice.done for choices in self.results.values() for choice in choices.values()):
                    break
            else:
                for rid in completed:
                    del self.results[rid]
                    del self.sequences[rid]
                completed.clear()
                capacity = self.config.max_seqs - len(self.results)
                if capacity > 0:
                    for request in poll_requests(capacity):
                        request = self._prepare_request(request)
                        self._begin_request(request)
                        pending.append(request)
                if not self.results:
                    break

            if pending and self.free_slots:
                prefill_outputs = self._prefill_batch(pending, replay)
                self._extract_outputs(prefill_outputs, replay, on_tokens, complete)
                pending = [request for request in pending if request.request_id not in self.sequences]
                if prefill_outputs is not None and (poll_requests is None or not self.local_map):
                    continue

            if step_callback is not None:
                step_callback(decode_iteration)
            if should_abort is not None and should_abort():
                aborted = True
                break

            iter_start = time.time()
            # Each active sequence contributes at most one queued token. Avoid running
            # the full decode graph when the live batch fits in a smaller bucket.
            active = len(self.local_map)
            round_tokens = next(
                (size for size in self.config.decode_bucket_sizes if size >= active),
                self.config.imputed_max_tokens_per_round,
            )
            replicated_single_token = (
                active == 1
                and self.config.max_seqs > 1
                and self.gen_state.window_pages is None
                and replay is None
                and isinstance(self.model, SnowballLMHeadModel)
                and self.model.config.inference_attention_implementation == "gpu"
                and self.model.config.inference_decode_implementation == "unrolled_split_cache"
                and self.model.config.inference_kv_sharding == "expert"
                and self.model.config.inference_decode_token_layout == "packed"
                and self.model.config.inference_expert_implementation == "separate"
                and self.model.config.projection_implementation == "native"
                and self.model.config.shared_attention_implementation == "native"
                and "H100" in jax.devices()[0].device_kind
            )
            future_state, decode_outputs = _run_generation_loop(
                self.gen_state,
                self.model,
                self.sampler,
                round_tokens,
                self.config.max_rounds,
                capture_routes=replay is not None,
                replicated_single_token=replicated_single_token,
                return_policy_logprobs=self.config.return_policy_logprobs,
            )
            submit_done = time.time()
            self.gen_state = future_state
            new_tokens = self._extract_outputs(decode_outputs, replay, on_tokens, complete)

            iter_time = time.time() - iter_start
            if iter_time > 0:
                logger.info(
                    "Decode iter: total %.3fs (submit %.3fs), %.2f tok/s, %d new",
                    iter_time,
                    submit_done - iter_start,
                    new_tokens / iter_time,
                    new_tokens,
                )
            decode_iteration += 1
            if new_tokens == 0:
                raise RuntimeError("Inference made no progress with unfinished requests.")

        if poll_requests is not None:
            call_rids = list(self.results)
        ordered_choices = []
        for rid in call_rids:
            choices = list(self.results.pop(rid).values())
            if aborted and any(not choice.done for choice in choices):
                for choice in choices:
                    if not choice.done:
                        choice.finish_reason = FinishReason.ABORT
                if complete is not None:
                    complete(rid, _result_for_choices(choices))
            if poll_requests is None:
                ordered_choices.extend(choices)
        result = _result_for_choices(ordered_choices)
        total_generated = streamed_tokens if poll_requests is not None else result.total_generated
        total_time = time.time() - time_in
        tps_overall = (total_generated / total_time) if total_time > 0 else 0.0
        logger.debug(f"Batch generated in {total_time:.2f}s, {total_generated} tokens, {tps_overall:.2f} tok/s")
        if aborted:
            self.reset()
        return result

    def generate_continuously(
        self,
        requests: Sequence[Request],
        poll_requests: RequestSource,
        on_complete: CompletionCallback,
        *,
        should_abort: Callable[[], bool],
        on_tokens: TokenCallback,
    ) -> None:
        """Serve requests until idle, admitting more work between device rounds."""
        if self.config.max_seqs < 2:
            raise ValueError("Continuous generation requires multiple sequence slots")
        self._generate(
            requests,
            should_abort=should_abort,
            on_tokens=on_tokens,
            on_complete=on_complete,
            poll_requests=poll_requests,
        )

    def _prepare_request(self, request: Request) -> Request:
        assert self.config.max_prefill_size is not None
        if not request.prompt_tokens or len(request.prompt_tokens) > self.config.max_prefill_size:
            raise ValueError("Each prompt must be nonempty and fit within max_prefill_size.")
        if len(request.prompt_tokens) >= self.config.max_seq_len:
            raise ValueError("Each prompt must leave room for generation within max_seq_len.")
        if request.n_generations != 1 and self.gen_state.window_pages is not None:
            raise ValueError("Sliding-window KV requires one generation per request")
        if request.n_generations < 1 or request.n_generations > self.config.max_seqs:
            raise ValueError(
                f"Requested {request.n_generations} sequences for a service with {self.config.max_seqs} slots"
            )
        max_tokens = min(int(request.decode_params.max_num_tokens), self.config.max_seq_len)
        request = dataclasses.replace(
            request, decode_params=dataclasses.replace(request.decode_params, max_num_tokens=jnp.asarray(max_tokens))
        )
        if (
            self.config.max_seqs > 1
            and self._request_pages(request) > self.gen_state.decode_state.page_table.num_pages
        ):
            raise ValueError("Request exceeds the KV pages available for generation")
        stop = request.decode_params.stop_tokens
        if stop is not None and (
            stop.axis_size("stop_seq") > self.config.max_stop_seqs
            or stop.axis_size("position") > self.config.max_stop_tokens
        ):
            raise ValueError("Requested stop tokens exceed the configured stop-token capacity")
        return request

    def _begin_request(self, request: Request) -> None:
        rid = request.request_id
        if rid in self.results:
            raise ValueError("Request IDs must be unique among active requests")
        self.results[rid] = {k: DecodeResult(id=rid, choice=k, token_list=[]) for k in range(request.n_generations)}

    def _start_generation(self) -> None:
        if not isinstance(self.model, SnowballLMHeadModel):
            self.reset()
        else:
            if self._cached_prompts and self._cached_prompts[-1].model is not self.model:
                self._cached_prompts.clear()
            # Snowball writes each new token's KV before attention. Free pages need no
            # device-wide memset between batches, while explicit reset still clears them.
            decode_state = eqx.filter_jit(DecodeState.reset)(self.gen_state.decode_state)
            if self.config.prefix_cache == "recent_prompts" and self._cached_prompts:
                table = decode_state.page_table
                refs = self._retained_page_refs(table)
                pages = jax.device_put(refs, table.page_ref_counts.array.sharding)
                table = dataclasses.replace(table, page_ref_counts=hax.named(pages, table.page_ref_counts.axes))
                decode_state = dataclasses.replace(decode_state, page_table=table)
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)
            self._clear_requests()

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
            assert max_len is not None
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
                prompt_tokens=hax.zeros(
                    {"prompt": min(max_slots, self.config.max_seqs_in_prefill), "position": max_len}, dtype=jnp.int32
                ),
                prompt_rows=hax.zeros({"seq": max_slots}, dtype=jnp.int32),
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

    def _extract_outputs(
        self,
        pending_outputs: _DecodeOutputs | _SnowballDecodeOutputs | None,
        replay: _SnowballReplayCollector | None = None,
        on_tokens: TokenCallback | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> int:
        """Append newly available tokens into outputs per (request_id, child_id).

        Returns number of new tokens appended.
        """
        if pending_outputs is None:
            return 0

        # Pull the entire buffer in one host op
        pending_outputs = jax.device_get(pending_outputs)
        if isinstance(pending_outputs, _SnowballDecodeOutputs):
            assert replay is not None
            replay.append(pending_outputs.routes, self.local_map)
            pending_outputs = pending_outputs.outputs
        n = int(pending_outputs.num_tokens)
        finish_reasons = np.asarray(pending_outputs.finish_reasons.array)
        fins = finish_reasons != FinishReason.RUNNING
        toks_arr = pending_outputs.tokens.array
        sids_arr = pending_outputs.slot_ids.array
        probs_arr = pending_outputs.logprobs.array if pending_outputs.logprobs is not None else None
        policy_probs = pending_outputs.policy_logprobs
        policy_arr = None if policy_probs is None else policy_probs.array

        appended = 0
        unmapped = 0
        new_tokens: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            local_slot = int(sids_arr[i])
            tok = int(toks_arr[i])
            info = self.local_map.get(local_slot)
            if info is None:
                unmapped += 1
                continue
            rid, cid = info
            dr = self.results[rid][cid]
            dr.token_list.append(tok)
            if probs_arr is not None:
                dr.logprobs.append(float(probs_arr[i]))
            if policy_arr is not None:
                dr.policy_logprobs.append(float(policy_arr[i]))
            dr.tokens_decoded += 1
            appended += 1
            if on_tokens is not None:
                new_tokens.setdefault((rid, cid), []).append(tok)

        if on_tokens is not None:
            for (rid, cid), tokens in new_tokens.items():
                on_tokens(rid, cid, tokens)

        # Update done flags based on snapshot
        finished_slots = sorted(slot for slot in self.local_map if fins[slot])
        for local_slot in finished_slots:
            rid, cid = self.local_map[local_slot]
            dr = self.results[rid][cid]
            dr.finish_reason = FinishReason(int(finish_reasons[local_slot]))
            choices = self.results[rid].values()
            if all(choice.done for choice in choices):
                self._page_reservations.pop(rid, None)
                if on_complete is not None:
                    on_complete(rid, _result_for_choices(list(choices)))

            del self.local_map[local_slot]
            self.free_slots.append(local_slot)

            # Detokenization is only needed for diagnostic logging.
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    full_text = self.tokenizer.decode(dr.token_list, skip_special_tokens=False)
                    logger.debug(f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): '{full_text}'")
                except Exception as e:
                    logger.error(
                        f"[Request {rid}, Choice {cid}] FINAL ({dr.tokens_decoded} tokens): <decode_error: {e}>"
                    )

        num_finished = len(finished_slots)
        if num_finished:
            released = np.zeros_like(fins)
            released[finished_slots] = True
            decode_state = eqx.filter_jit(DecodeState.free_pages_for_finished)(
                self.gen_state.decode_state, jnp.asarray(released)
            )
            self.gen_state = dataclasses.replace(self.gen_state, decode_state=decode_state)
            self.free_slots.sort()
        logger.debug(f"extract: appended={appended} (drained={n}) unmapped={unmapped} finished_count={num_finished}")

        return appended
