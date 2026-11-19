# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified teacher-matching and cached top-k KL objectives for long-loop SFT.

Supports two forward-KL objectives behind one trainer interface, optimizing

    L_step = ce_weight * L_CE(all M microbatches) + weight * L_KL(K of M microbatches)

inside a single optimizer step:

1. ``full_vocab_temperature`` (Ambient SFT, arXiv 2611.23229, Eq. 4):
   Full-vocabulary forward KL against a frozen teacher checkpoint at temperature ``T``:

       KL_i = sum_v softmax(z_{teacher, i} / T)_v
                    * (log_softmax(z_{teacher, i} / T)_v - log_softmax(z_{student, i} / T)_v)

   Hinton's T^2 prefactor is intentionally omitted (arXiv 2611.23229 Sec. 5, Prop. E.1),
   so d(KL_i)/d(z_{student, i}) = (p_{T, student} - p_{T, teacher}) / T.

2. ``cached_topk_kl`` (Beyer et al. arXiv 2106.05237 / Busbridge et al. arXiv 2502.08606 App. G.4):
   Coarse-grained ``{top-k} U {actual token} + residual tail bin`` forward KL at ``T = 1.0``
   strictly from cached full-softmax teacher log-probabilities loaded via ``CachedTopKMixtureReader``,
   using exact complement log-sum-exp over non-explicit student logits and rejecting invalid rows.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int

from levanter.data.mixture import MixtureDataset
from levanter.grug.sharding import _axis_names, _current_mesh, _reshard_for_shard_map
from levanter.sharding import partition_spec_of
from levanter.utils.jax_utils import local_cpu_mesh


class TeacherMatchingMode(StrEnum):
    """Supported teacher-matching KL objectives behind the unified SFT trainer interface."""

    # Full-vocabulary forward KL to a frozen on-host teacher at temperature T (arXiv 2611.23229, no T^2).
    FULL_VOCAB_TEMPERATURE = "full_vocab_temperature"
    # Coarse-grained {top-k} U {actual} + residual tail-bin forward KL at T=1 from cached teacher logprobs
    # (Beyer et al. arXiv:2106.05237 + Busbridge et al. arXiv:2502.08606 App. G.4).
    CACHED_TOPK_KL = "cached_topk_kl"


@register_dataclass
@dataclass(frozen=True)
class CachedTeacherTopK:
    """Per-position cached teacher top-k (plus optional actual token) natural log-probabilities at T=1.

    Alignment contract (critical — avoids off-by-1 teacher supervision):
      ``Transformer.next_token_loss_from_hidden`` shifts ``labels = pad(token_ids[:, 1:])``, so
      hidden state ``hidden[:, i, :]`` predicts ``token_ids[:, i + 1]`` with loss weight
      ``loss_weight[:, i]``. Accordingly, ``ids[:, i, :]`` and ``logprobs[:, i, :]`` MUST store
      the teacher's distribution for **``token_ids[:, i + 1]``** (i.e., vLLM's
      ``prompt_logprobs[i + 1]``, left-shifted by 1 to align with ``loss_weight[:, i]``).

    Shapes match batch tokens with a trailing slot dimension K (for example K=21 for top-20 plus
    the actual next token when it falls outside top-20). Invalid or padded slots may be marked
    with ``ids < 0`` and/or ``valid=False``; duplicate token IDs within a position are deduplicated
    automatically inside the KL kernel.
    """

    ids: Int[Array, "B S K"]
    logprobs: Float[Array, "B S K"]
    valid: jax.Array | None = None


@dataclass(frozen=True)
class TemperatureMatchingConfig:
    """Unified configuration for same-step teacher matching / distillation in long-loop SFT.

    Optimizes ``ce_weight * L_CE(all M microbatches) + weight * L_KL(K of M microbatches)``
    inside a single optimizer step, supporting both:
    - ``full_vocab_temperature``: frozen BF16 start checkpoint on ``pinned_host``, chunked full-vocab
      forward KL at temperature ``T`` without Hinton's ``T^2`` prefactor (arXiv 2611.23229).
    - ``cached_topk_kl``: cached teacher top-k (+ actual token) log-probabilities + residual tail bin
      at ``T=1.0`` strictly (Beyer et al. 2106.05237 / Busbridge et al. 2502.08606 App. G.4), loaded
      per mixture source from ``cached_topk_paths`` via ``CachedTopKMixtureReader``.

    When ``enabled=True``, ``weight``, ``temperature``, and ``microbatch_count`` must be supplied
    explicitly (no silent defaults on critical hyperparameters).
    """

    enabled: bool = False
    mode: TeacherMatchingMode = TeacherMatchingMode.FULL_VOCAB_TEMPERATURE
    # Weight on L_CE in ce_weight * L_CE + weight * L_KL (1.0 for Ambient SFT; e.g. 0.5 for top-k KD).
    ce_weight: float = 1.0
    # Required when enabled=True: weight on L_KL in ce_weight * L_CE + weight * L_KL.
    weight: float | None = None
    # Required when enabled=True: softmax temperature T > 0 (no Hinton T^2). Must be 1.0 for cached_topk_kl.
    temperature: float | None = None
    # Required when enabled=True: number of microbatches K (out of M) on which teacher matching runs.
    microbatch_count: int | None = None
    # Token block size for chunked full-vocabulary / top-k KL scan (avoids [B, S, V] materialization).
    token_block_size: int = 1024
    # Keep frozen teacher weights on pinned_host and stream layer-by-layer (full_vocab_temperature only).
    offload_teacher: bool = True
    # Explicit checkpoint path for the frozen teacher (full_vocab_temperature). Defaults to initialize_from.
    teacher_checkpoint_path: str | None = None
    # Mapping of mixture source name -> cached_teacher_topk_v1 directory URI (required for cached_topk_kl).
    cached_topk_paths: dict[str, str] = field(default_factory=dict)
    # Verify per-row token-ID SHA-256 against cached manifest hashes when present (cached_topk_kl).
    verify_cached_topk_token_hashes: bool = True
    # If True, fail if a configured cached source does not cover a sampled source row index.
    require_full_cached_source_coverage: bool = False
    # Per-source switches (mutually exclusive): turn TM off on chosen MixtureDataset sources.
    disabled_sources: tuple[str, ...] = ()
    enabled_sources: tuple[str, ...] | None = None
    # Log full-parameter ||grad_KL|| / ||grad_CE|| watch metrics without adding HBM to train_step.
    # Runs one extra full CE + KL forward/backward pass sequentially on watch steps only.
    log_grad_norm_ratio: bool = True
    # Cadence (in steps) for ||grad_KL|| / ||grad_CE|| watch metrics. None follows watch.interval.
    grad_norm_ratio_interval: int | None = None
    # Expert-parallel runs (expert_axis_size > 1) use capacity-limited MoE dispatch that can drop
    # different tokens for student and teacher; disallow by default so both see identical tokens.
    allow_ep_capacity_drops: bool = False

    def __post_init__(self) -> None:
        if self.disabled_sources and self.enabled_sources is not None:
            raise ValueError("Set at most one of disabled_sources and enabled_sources")
        if self.grad_norm_ratio_interval is not None and self.grad_norm_ratio_interval <= 0:
            raise ValueError("grad_norm_ratio_interval must be positive")
        if not self.enabled:
            return
        if not math.isfinite(self.ce_weight) or self.ce_weight < 0.0:
            raise ValueError(f"temperature_matching.ce_weight must be finite and non-negative, got {self.ce_weight}")
        if self.weight is None or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"temperature_matching.weight must be explicitly set (> 0) when enabled, got {self.weight}")
        if self.temperature is None or not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(
                f"temperature_matching.temperature must be explicitly set (> 0) when enabled, got {self.temperature}"
            )
        if self.mode == TeacherMatchingMode.CACHED_TOPK_KL:
            if not math.isclose(self.temperature, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "cached_topk_kl requires temperature == 1.0; rescaling truncated top-k logprobs to T != 1 "
                    f"is invalid because tail mass cannot be reconstructed (got temperature={self.temperature})"
                )
            if not self.cached_topk_paths:
                raise ValueError(
                    "cached_topk_kl requires non-empty temperature_matching.cached_topk_paths mapping "
                    "mixture source names to cached_teacher_topk_v1 directories"
                )
        if self.microbatch_count is None or self.microbatch_count <= 0:
            raise ValueError(
                "temperature_matching.microbatch_count must be explicitly set (> 0) when enabled, "
                f"got {self.microbatch_count}"
            )
        if self.token_block_size <= 0:
            raise ValueError(f"temperature_matching.token_block_size must be positive, got {self.token_block_size}")


def validate_temperature_matching_trainer_config(
    tm_config: TemperatureMatchingConfig,
    *,
    gradient_accumulation_steps: int,
    expert_axis_size: int,
) -> None:
    """Validate temperature-matching invariants that depend on trainer mesh/accumulation settings."""
    if not tm_config.enabled:
        return
    assert tm_config.microbatch_count is not None
    if tm_config.microbatch_count > gradient_accumulation_steps:
        raise ValueError(
            f"temperature_matching.microbatch_count ({tm_config.microbatch_count}) must not exceed "
            f"gradient_accumulation_steps ({gradient_accumulation_steps})"
        )
    if expert_axis_size > 1 and not tm_config.allow_ep_capacity_drops:
        raise ValueError(
            "Temperature matching requires dropless MoE (expert_axis_size == 1) so student and teacher see "
            f"the exact same tokens; got expert_axis_size={expert_axis_size}. Set allow_ep_capacity_drops=True "
            "only if capacity-limited EP token drops are explicitly intended."
        )


def validate_tm_source_names(dataset: MixtureDataset, tm_config: TemperatureMatchingConfig) -> None:
    """Fail fast if any configured source name is absent from the training mixture."""
    if not tm_config.enabled:
        return
    available = set(dataset.dataset_index)
    configured: tuple[str, ...] = tm_config.disabled_sources
    label = "disabled_sources"
    if tm_config.enabled_sources is not None:
        configured = tm_config.enabled_sources
        label = "enabled_sources"
    missing = sorted(set(configured) - available)
    if missing:
        raise ValueError(
            f"temperature_matching.{label} references unknown mixture sources {missing}; "
            f"available sources are {sorted(available)}"
        )


def source_names_for_mixture_indices(dataset: MixtureDataset, indices: Sequence[int]) -> list[str]:
    """Return the MixtureDataset source name for each global row index without reading caches."""
    names: list[str] = []
    block_size = dataset.block_size
    dataset_index = dataset.dataset_index
    # MixtureDataset.__init__ places self.key under local_cpu_mesh() (Auto); match that mesh
    # so cold-cache _get_block() calls work inside an active Explicit training mesh.
    with local_cpu_mesh():
        for raw_index in indices:
            rel_index = dataset._relative_index(int(raw_index))
            block_id = rel_index // block_size
            block = dataset._get_block(int(block_id))
            packed_id = int(block[rel_index % block_size])
            dataset_id = packed_id >> 16
            names.append(dataset_index[dataset_id])
    return names


def tm_row_mask_for_mixture_indices(
    dataset: MixtureDataset,
    indices: Sequence[int],
    tm_config: TemperatureMatchingConfig,
) -> np.ndarray:
    """Build a per-row float32 mask in {0.0, 1.0} for the configured per-source TM switch."""
    if not tm_config.disabled_sources and tm_config.enabled_sources is None:
        return np.ones((len(indices),), dtype=np.float32)
    names = source_names_for_mixture_indices(dataset, indices)
    if tm_config.enabled_sources is not None:
        allow = set(tm_config.enabled_sources)
        return np.array([1.0 if name in allow else 0.0 for name in names], dtype=np.float32)
    deny = set(tm_config.disabled_sources)
    return np.array([0.0 if name in deny else 1.0 for name in names], dtype=np.float32)


def global_l2_norm(tree) -> jax.Array:
    """Compute the FP32 global L2 norm across all floating-point array leaves of a PyTree."""
    sq_sum = jnp.zeros((), dtype=jnp.float32)
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            leaf_f32 = leaf.astype(jnp.float32)
            sq_sum = sq_sum + jnp.sum(leaf_f32 * leaf_f32)
    return jnp.sqrt(sq_sum)


def _token_dim_specs(x: jax.Array) -> tuple:
    """Preserve all token dimensions so sequence shards need no activation all-gather."""
    spec = partition_spec_of(x)
    if spec is not None:
        padded = tuple(spec) + (None,) * (x.ndim - len(spec))
        token_dims = padded[: x.ndim - 1]
        if any(entry is not None for entry in token_dims):
            return token_dims
    return ("data",) + (None,) * (x.ndim - 2)


def _psum_over_axes(x: jax.Array, axis_names: tuple[str, ...]) -> jax.Array:
    if len(axis_names) == 0:
        return x
    if len(axis_names) == 1:
        return jax.lax.psum(x, axis_names[0])
    return jax.lax.psum(x, axis_names)


def _pad_rows(a: jax.Array, padding: int) -> jax.Array:
    return jnp.pad(a, ((0, padding),) + ((0, 0),) * (a.ndim - 1)) if padding else a


def _row_blocks(a: jax.Array, block_size: int) -> jax.Array:
    return a.reshape((a.shape[0] // block_size, block_size) + a.shape[1:])


def _block_logits(x_block: jax.Array, w: jax.Array, precision: jax.lax.PrecisionLike) -> jax.Array:
    return jax.lax.dot_general(
        x_block, w, (((1,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
    ).astype(jnp.float32)


def _blocked_inputs(
    student_hidden: jax.Array,
    teacher_hidden: jax.Array,
    weight: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    padding = (-student_hidden.shape[0]) % block_size
    s_h, t_h, w = (_pad_rows(a, padding) for a in (student_hidden, teacher_hidden, weight))
    return _row_blocks(s_h, block_size), _row_blocks(t_h, block_size), _row_blocks(w, block_size)


def _kl_weighted_sum_primal(
    student_hidden: Float[Array, "T H"],
    student_lm_head: Float[Array, "H V"],
    teacher_hidden: Float[Array, "T H"],
    teacher_lm_head: Float[Array, "H V"],
    weight: Float[Array, "T"],
    temperature: float,
    block_size: int,
    precision: jax.lax.PrecisionLike,
    teacher_hidden_dtype: jnp.dtype,
    teacher_lm_head_dtype: jnp.dtype,
) -> Float[Array, ""]:
    del teacher_hidden_dtype, teacher_lm_head_dtype
    inv_temp = jnp.asarray(1.0 / temperature, dtype=jnp.float32)
    teacher_hidden = jax.lax.stop_gradient(teacher_hidden)
    teacher_lm_head = jax.lax.stop_gradient(teacher_lm_head)

    def body(total, block):
        s_h_block, t_h_block, w_block = block
        s_logits = _block_logits(s_h_block, student_lm_head, precision) * inv_temp
        t_logits = _block_logits(t_h_block, teacher_lm_head, precision) * inv_temp
        log_q = jax.nn.log_softmax(t_logits, axis=-1)
        q = jnp.exp(log_q)
        log_p = jax.nn.log_softmax(s_logits, axis=-1)
        kl_block = jnp.sum(q * (log_q - log_p), axis=-1)
        return total + jnp.sum(kl_block * w_block.astype(jnp.float32)), None

    total, _ = jax.lax.scan(
        body,
        jnp.zeros((), dtype=jnp.float32),
        _blocked_inputs(student_hidden, teacher_hidden, weight, block_size),
    )
    return total


def _kl_weighted_sum_fwd(
    student_hidden: Float[Array, "T H"],
    student_lm_head: Float[Array, "H V"],
    teacher_hidden: Float[Array, "T H"],
    teacher_lm_head: Float[Array, "H V"],
    weight: Float[Array, "T"],
    temperature: float,
    block_size: int,
    precision: jax.lax.PrecisionLike,
    teacher_hidden_dtype: jnp.dtype,
    teacher_lm_head_dtype: jnp.dtype,
):
    del teacher_hidden_dtype, teacher_lm_head_dtype
    rows, hidden_dim = student_hidden.shape
    gemm_dtype = jnp.result_type(student_hidden.dtype, student_lm_head.dtype)
    inv_temp = jnp.asarray(1.0 / temperature, dtype=jnp.float32)
    teacher_hidden = jax.lax.stop_gradient(teacher_hidden)
    teacher_lm_head = jax.lax.stop_gradient(teacher_lm_head)

    def body(grad_s_w, block):
        s_h_block, t_h_block, w_block = block
        s_logits = _block_logits(s_h_block, student_lm_head, precision) * inv_temp
        t_logits = _block_logits(t_h_block, teacher_lm_head, precision) * inv_temp
        log_q = jax.nn.log_softmax(t_logits, axis=-1)
        q = jnp.exp(log_q)
        log_p = jax.nn.log_softmax(s_logits, axis=-1)
        p = jnp.exp(log_p)
        kl_block = jnp.sum(q * (log_q - log_p), axis=-1)

        # Exact derivative without Hinton's T^2 factor:
        # d(weight * KL(q || p)) / d(z_student) = (weight / T) * (p - q).
        w_scaled = w_block.astype(jnp.float32) * inv_temp
        dlogits = ((p - q) * w_scaled[:, None]).astype(gemm_dtype)
        grad_s_h = jax.lax.dot_general(
            dlogits, student_lm_head, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        grad_s_w = grad_s_w + jax.lax.dot_general(
            s_h_block, dlogits, (((0,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        return grad_s_w, (kl_block, grad_s_h.astype(student_hidden.dtype))

    grad_s_w, (kl_blocks, grad_s_h_blocks) = jax.lax.scan(
        body,
        jnp.zeros(student_lm_head.shape, dtype=jnp.float32),
        _blocked_inputs(student_hidden, teacher_hidden, weight, block_size),
    )
    kl_per_row = kl_blocks.reshape(-1)[:rows]
    grad_s_h = grad_s_h_blocks.reshape(-1, hidden_dim)[:rows]
    total = jnp.sum(kl_per_row * weight.astype(jnp.float32))
    return total, (grad_s_h, grad_s_w.astype(student_lm_head.dtype), kl_per_row.astype(weight.dtype))


def _kl_weighted_sum_bwd(temperature, block_size, precision, teacher_hidden_dtype, teacher_lm_head_dtype, residuals, g):
    del temperature, block_size, precision
    grad_s_h, grad_s_w, kl_per_row = residuals
    g = g.astype(jnp.float32)
    return (
        (g * grad_s_h).astype(grad_s_h.dtype),
        (g * grad_s_w).astype(grad_s_w.dtype),
        jnp.zeros(grad_s_h.shape, dtype=teacher_hidden_dtype),
        jnp.zeros(grad_s_w.shape, dtype=teacher_lm_head_dtype),
        (g * kl_per_row).astype(kl_per_row.dtype),
    )


_kl_weighted_sum = jax.custom_vjp(_kl_weighted_sum_primal, nondiff_argnums=(5, 6, 7, 8, 9))
_kl_weighted_sum.defvjp(_kl_weighted_sum_fwd, _kl_weighted_sum_bwd)


def chunked_linear_tempered_kl_weighted_sum(
    student_hidden: jax.Array,
    student_lm_head: jax.Array,
    teacher_hidden: jax.Array,
    teacher_lm_head: jax.Array,
    weight: jax.Array,
    *,
    temperature: float,
    token_block_size: int = 1024,
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
    """Compute ``sum_i weight_i * KL(p_{T, teacher} || p_{T, student})`` without materializing [B, S, V].

    Mirrors ``levanter.grug.loss.fused_linear_softmax_cross_entropy_loss`` sharding:
    preserves token sharding across data/expert/context axes inside ``shard_map``, scans
    in blocks of ``token_block_size`` tokens using ``custom_vjp`` grad-in-forward, and
    returns the globally summed scalar KL across all token shards.
    """
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if token_block_size <= 0:
        raise ValueError(f"token_block_size must be positive, got {token_block_size}")
    if student_lm_head.ndim != 2 or teacher_lm_head.ndim != 2:
        raise ValueError(f"lm_head tensors must be 2D; got {student_lm_head.shape} and {teacher_lm_head.shape}")
    hidden_dim = student_hidden.shape[-1]
    if student_lm_head.shape != teacher_lm_head.shape or student_lm_head.shape[0] != hidden_dim:
        raise ValueError(
            f"Incompatible shapes: student_hidden={student_hidden.shape}, "
            f"student_lm_head={student_lm_head.shape}, teacher_lm_head={teacher_lm_head.shape}"
        )

    mesh = _current_mesh()
    has_mesh = mesh is not None and not mesh.empty
    token_dim_specs = _token_dim_specs(student_hidden) if has_mesh else ()
    token_sharding_axes = tuple(name for entry in token_dim_specs for name in _axis_names(entry))

    def _kl_shard(
        shard_s_h: jax.Array,
        shard_s_w: jax.Array,
        shard_t_h: jax.Array,
        shard_t_w: jax.Array,
        shard_w: jax.Array,
    ) -> jax.Array:
        flat_s_h = shard_s_h.reshape((-1, hidden_dim))
        flat_t_h = shard_t_h.reshape((-1, hidden_dim))
        flat_w = shard_w.reshape((-1,)).astype(jnp.float32)
        rows = flat_s_h.shape[0]
        if rows == 0:
            local_sum = jnp.zeros((), dtype=jnp.float32)
        else:
            effective_block = min(token_block_size, rows)
            local_sum = _kl_weighted_sum(
                flat_s_h,
                shard_s_w,
                jax.lax.stop_gradient(flat_t_h),
                jax.lax.stop_gradient(shard_t_w),
                flat_w,
                float(temperature),
                int(effective_block),
                precision,
                flat_t_h.dtype,
                shard_t_w.dtype,
            )
        return _psum_over_axes(local_sum, token_sharding_axes)

    if not has_mesh:
        return _kl_shard(student_hidden, student_lm_head, teacher_hidden, teacher_lm_head, weight)

    hidden_spec = P(*token_dim_specs, None)
    lm_head_spec = P(None, None)
    weight_spec = P(*token_dim_specs)
    student_hidden = _reshard_for_shard_map(student_hidden, mesh, hidden_spec)
    student_lm_head = _reshard_for_shard_map(student_lm_head, mesh, lm_head_spec)
    teacher_hidden = _reshard_for_shard_map(teacher_hidden, mesh, hidden_spec)
    teacher_lm_head = _reshard_for_shard_map(teacher_lm_head, mesh, lm_head_spec)
    weight = _reshard_for_shard_map(weight, mesh, weight_spec)

    return jax.shard_map(
        _kl_shard,
        mesh=mesh,
        in_specs=(hidden_spec, lm_head_spec, hidden_spec, lm_head_spec, weight_spec),
        out_specs=P(),
        check_vma=False,
    )(student_hidden, student_lm_head, teacher_hidden, teacher_lm_head, weight)


_CACHED_LOGPROB_UPPER_TOL = 1e-3
_CACHED_MASS_CLOSURE_TOL = 1e-2


def _validate_and_deduplicate_topk_slots(
    ids: Int[Array, "... K"],
    logprobs: Float[Array, "... K"],
    valid: jax.Array | None,
    vocab_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Validate full-softmax cached logprobs, deduplicate explicit slots, and check mass closure.

    Unused padding slots are marked with ``ids < 0`` and/or ``valid=False``. Any row containing an
    active slot with out-of-range ID (``>= vocab_size``), non-finite logprob, positive logprob above
    ``_CACHED_LOGPROB_UPPER_TOL``, or deduplicated explicit probability mass exceeding
    ``1.0 + _CACHED_MASS_CLOSURE_TOL`` is rejected as invalid (``row_valid=False``).
    """
    valid_mask = jnp.ones(ids.shape, dtype=jnp.bool_) if valid is None else valid.astype(jnp.bool_)
    lp_f32 = logprobs.astype(jnp.float32)
    slot_present = (ids >= 0) & valid_mask
    slot_in_vocab = slot_present & (ids < vocab_size)
    slot_corrupt = (slot_present & (ids >= vocab_size)) | (
        slot_in_vocab & ((~jnp.isfinite(lp_f32)) | (lp_f32 > _CACHED_LOGPROB_UPPER_TOL))
    )
    row_corrupt = jnp.any(slot_corrupt, axis=-1)

    base_valid = slot_in_vocab & jnp.isfinite(lp_f32) & (lp_f32 <= _CACHED_LOGPROB_UPPER_TOL)
    k = ids.shape[-1]
    same_id = (ids[..., :, None] == ids[..., None, :]) & base_valid[..., :, None] & base_valid[..., None, :]
    earlier = jnp.tril(jnp.ones((k, k), dtype=jnp.bool_), k=-1)
    is_dup = jnp.any(same_id & earlier, axis=-1)
    unique_valid = base_valid & (~is_dup)
    safe_ids = jnp.where(unique_valid, ids, 0).astype(jnp.int32)

    safe_lp = jnp.where(unique_valid, jnp.minimum(lp_f32, 0.0), 0.0)
    raw_p_exp = jnp.where(unique_valid, jnp.exp(safe_lp), 0.0)
    raw_sum = jnp.sum(raw_p_exp, axis=-1)
    row_valid = (~row_corrupt) & jnp.any(unique_valid, axis=-1) & (raw_sum <= 1.0 + _CACHED_MASS_CLOSURE_TOL)

    unique_valid = unique_valid & row_valid[..., None]
    scale_down = jnp.where(raw_sum > 1.0, 1.0 / jnp.maximum(raw_sum, 1.0), 1.0)
    p_exp = jnp.where(unique_valid, raw_p_exp * scale_down[..., None], 0.0)
    log_p_exp = jnp.where(unique_valid, safe_lp + jnp.log(scale_down)[..., None], 0.0)
    p_tail = jnp.where(row_valid, jnp.maximum(1.0 - jnp.sum(p_exp, axis=-1), 0.0), 0.0)
    return unique_valid, row_valid, safe_ids, p_exp, log_p_exp, p_tail


def cached_topk_has_valid_entries(teacher_topk: CachedTeacherTopK, vocab_size: int) -> jax.Array:
    """Return a float32 ``[B, S]`` indicator (1.0 where cached top-k passes validation & mass closure)."""
    _, row_valid, _, _, _, _ = _validate_and_deduplicate_topk_slots(
        teacher_topk.ids, teacher_topk.logprobs, teacher_topk.valid, vocab_size
    )
    return row_valid.astype(jnp.float32)


def _blocked_topk_inputs(
    student_hidden: jax.Array,
    ids: jax.Array,
    logprobs: jax.Array,
    valid: jax.Array,
    weight: jax.Array,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    padding = (-student_hidden.shape[0]) % block_size
    s_h = _pad_rows(student_hidden, padding)
    ids_p = jnp.pad(ids, ((0, padding), (0, 0)), constant_values=-1) if padding else ids
    lp_p = _pad_rows(logprobs, padding)
    valid_p = _pad_rows(valid, padding)
    w_p = _pad_rows(weight, padding)
    return (
        _row_blocks(s_h, block_size),
        _row_blocks(ids_p, block_size),
        _row_blocks(lp_p, block_size),
        _row_blocks(valid_p, block_size),
        _row_blocks(w_p, block_size),
    )


def _topk_block_kl_and_target(
    s_logits: Float[Array, "B V"],
    ids_block: Int[Array, "B K"],
    lp_block: Float[Array, "B K"],
    valid_block: jax.Array,
) -> tuple[Float[Array, "B"], jax.Array, Float[Array, "B V"], Float[Array, "B V"], Float[Array, "B"]]:
    """Compute exact 21-bin coarse-grained KL at T=1 and target vocab distribution in FP32.

    Uses complement log-sum-exp over non-explicit student logits so ``log(q_tail)`` and its
    gradient remain exact and finite even when ``q_tail`` underflows FP32 subtraction ``1 - sum(q_exp)``.
    """
    block_rows, vocab_size = s_logits.shape
    unique_valid, row_valid, safe_ids, p_exp, log_p_exp, p_tail = _validate_and_deduplicate_topk_slots(
        ids_block, lp_block, valid_block, vocab_size
    )
    row_idx = jnp.arange(block_rows, dtype=jnp.int32)[:, None]

    # Explicit support mask via bool scatter-max (1 byte/entry, never clears token 0 on invalid slots).
    exp_mask = jnp.zeros((block_rows, vocab_size), dtype=jnp.bool_).at[row_idx, safe_ids].max(unique_valid)
    tail_active = row_valid & (p_tail > 1e-12)

    # Full student distribution and exact complement log-sum-exp over tail R = V \ E.
    lse_full = jax.scipy.special.logsumexp(s_logits, axis=-1)
    log_q_full = s_logits - lse_full[:, None]
    log_q_exp = jnp.take_along_axis(log_q_full, safe_ids, axis=-1)
    q_full = jnp.exp(log_q_full)

    tail_has_tokens = jnp.any(~exp_mask, axis=-1)
    tail_logits = jnp.where(exp_mask, -jnp.inf, s_logits)
    lse_tail_raw = jax.scipy.special.logsumexp(tail_logits, axis=-1)
    lse_tail = jnp.where(tail_has_tokens, lse_tail_raw, 0.0)
    log_q_tail = jnp.where(tail_has_tokens, lse_tail - lse_full, 0.0)

    kl_exp = jnp.sum(jnp.where(unique_valid, p_exp * (log_p_exp - log_q_exp), 0.0), axis=-1)
    include_tail = tail_active & tail_has_tokens
    kl_tail = jnp.where(
        include_tail,
        p_tail * (jnp.log(jnp.maximum(p_tail, 1e-30)) - log_q_tail),
        0.0,
    )
    kl_block = jnp.where(row_valid, kl_exp + kl_tail, 0.0)

    # Effective target distribution over V: p_k on E, p_tail * q(k | R) on R.
    # Retaining exact target_mass ensures d(kl_block)/d(s_logits) = target_mass * q_full - target_vocab
    # matches autodiff even when include_tail is False (zero-mass limit or full-vocab explicit support).
    tail_mass = jnp.where(include_tail, p_tail, 0.0)
    tail_target = jnp.where(
        (~exp_mask) & include_tail[:, None],
        tail_mass[:, None] * jnp.exp(tail_logits - lse_tail[:, None]),
        0.0,
    )
    target_vocab = tail_target.at[row_idx, safe_ids].add(p_exp)
    target_mass = jnp.sum(p_exp, axis=-1) + tail_mass
    return kl_block, row_valid, q_full, target_vocab, target_mass


def _topk_kl_weighted_sum_primal(
    student_hidden: Float[Array, "T H"],
    student_lm_head: Float[Array, "H V"],
    ids: Int[Array, "T K"],
    logprobs: Float[Array, "T K"],
    valid: jax.Array,
    weight: Float[Array, "T"],
    block_size: int,
    precision: jax.lax.PrecisionLike,
    lp_dtype: jnp.dtype,
    k_slots: int,
) -> Float[Array, ""]:
    del lp_dtype, k_slots

    def body(total, block):
        s_h_block, ids_block, lp_block, valid_block, w_block = block
        s_logits = _block_logits(s_h_block, student_lm_head, precision)
        kl_block, row_valid, _, _, _ = _topk_block_kl_and_target(s_logits, ids_block, lp_block, valid_block)
        eff_w = jnp.where(row_valid, w_block.astype(jnp.float32), 0.0)
        return total + jnp.sum(kl_block * eff_w), None

    total, _ = jax.lax.scan(
        body,
        jnp.zeros((), dtype=jnp.float32),
        _blocked_topk_inputs(student_hidden, ids, logprobs, valid, weight, block_size),
    )
    return total


def _topk_kl_weighted_sum_fwd(
    student_hidden: Float[Array, "T H"],
    student_lm_head: Float[Array, "H V"],
    ids: Int[Array, "T K"],
    logprobs: Float[Array, "T K"],
    valid: jax.Array,
    weight: Float[Array, "T"],
    block_size: int,
    precision: jax.lax.PrecisionLike,
    lp_dtype: jnp.dtype,
    k_slots: int,
):
    del lp_dtype, k_slots
    rows, hidden_dim = student_hidden.shape
    gemm_dtype = jnp.result_type(student_hidden.dtype, student_lm_head.dtype)

    def body(grad_s_w, block):
        s_h_block, ids_block, lp_block, valid_block, w_block = block
        s_logits = _block_logits(s_h_block, student_lm_head, precision)
        kl_block, row_valid, q_full, target_vocab, target_mass = _topk_block_kl_and_target(
            s_logits, ids_block, lp_block, valid_block
        )
        eff_w = jnp.where(row_valid, w_block.astype(jnp.float32), 0.0)
        dlogits = ((target_mass[:, None] * q_full - target_vocab) * eff_w[:, None]).astype(gemm_dtype)
        grad_s_h = jax.lax.dot_general(
            dlogits, student_lm_head, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        grad_s_w = grad_s_w + jax.lax.dot_general(
            s_h_block, dlogits, (((0,), (0,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
        )
        return grad_s_w, (kl_block, eff_w, grad_s_h.astype(student_hidden.dtype))

    grad_s_w, (kl_blocks, eff_w_blocks, grad_s_h_blocks) = jax.lax.scan(
        body,
        jnp.zeros(student_lm_head.shape, dtype=jnp.float32),
        _blocked_topk_inputs(student_hidden, ids, logprobs, valid, weight, block_size),
    )
    kl_per_row = kl_blocks.reshape(-1)[:rows]
    eff_w = eff_w_blocks.reshape(-1)[:rows]
    grad_s_h = grad_s_h_blocks.reshape(-1, hidden_dim)[:rows]
    total = jnp.sum(kl_per_row * eff_w)
    return total, (grad_s_h, grad_s_w.astype(student_lm_head.dtype), kl_per_row.astype(weight.dtype))


def _topk_kl_weighted_sum_bwd(block_size, precision, lp_dtype, k_slots, residuals, g):
    del block_size, precision
    grad_s_h, grad_s_w, kl_per_row = residuals
    ids_shape = (grad_s_h.shape[0], k_slots)
    g = g.astype(jnp.float32)
    return (
        (g * grad_s_h).astype(grad_s_h.dtype),
        (g * grad_s_w).astype(grad_s_w.dtype),
        np.zeros(ids_shape, dtype=jax.dtypes.float0),
        jnp.zeros(ids_shape, dtype=lp_dtype),
        np.zeros(ids_shape, dtype=jax.dtypes.float0),
        (g * kl_per_row).astype(kl_per_row.dtype),
    )


_topk_kl_weighted_sum = jax.custom_vjp(_topk_kl_weighted_sum_primal, nondiff_argnums=(6, 7, 8, 9))
_topk_kl_weighted_sum.defvjp(_topk_kl_weighted_sum_fwd, _topk_kl_weighted_sum_bwd)


def chunked_linear_cached_topk_kl_weighted_sum(
    student_hidden: jax.Array,
    student_lm_head: jax.Array,
    teacher_topk: CachedTeacherTopK,
    weight: jax.Array,
    *,
    token_block_size: int = 1024,
    precision: jax.lax.PrecisionLike = None,
) -> jax.Array:
    """Compute coarse-grained ``{top-k} U {actual} + residual tail`` forward KL at T=1 in chunks.

    Preserves token sharding across data/expert/context axes inside ``shard_map``, never
    materializes ``[B, S, V]``, deduplicates explicit slots per token, and computes ``log(q_tail)``
    via exact complement log-sum-exp over non-explicit logits so near-zero tail mass never NaNs.
    """
    if token_block_size <= 0:
        raise ValueError(f"token_block_size must be positive, got {token_block_size}")
    if student_lm_head.ndim != 2:
        raise ValueError(f"student_lm_head must be 2D (H, V), got {student_lm_head.shape}")
    hidden_dim = student_hidden.shape[-1]
    if student_lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={student_lm_head.shape[0]}")
    if teacher_topk.ids.shape != teacher_topk.logprobs.shape:
        raise ValueError(
            f"teacher_topk ids {teacher_topk.ids.shape} and logprobs {teacher_topk.logprobs.shape} must match"
        )
    if teacher_topk.ids.shape[:-1] != weight.shape:
        raise ValueError(
            f"teacher_topk leading shape {teacher_topk.ids.shape[:-1]} must match weight shape {weight.shape}"
        )

    valid_arr = (
        jnp.ones(teacher_topk.ids.shape, dtype=jnp.bool_)
        if teacher_topk.valid is None
        else teacher_topk.valid.astype(jnp.bool_)
    )
    k_slots = teacher_topk.ids.shape[-1]
    mesh = _current_mesh()
    has_mesh = mesh is not None and not mesh.empty
    token_dim_specs = _token_dim_specs(student_hidden) if has_mesh else ()
    token_sharding_axes = tuple(name for entry in token_dim_specs for name in _axis_names(entry))

    def _shard_fn(
        shard_s_h: jax.Array,
        shard_s_w: jax.Array,
        shard_ids: jax.Array,
        shard_lp: jax.Array,
        shard_valid: jax.Array,
        shard_w: jax.Array,
    ) -> jax.Array:
        flat_s_h = shard_s_h.reshape((-1, hidden_dim))
        flat_ids = shard_ids.reshape((-1, k_slots)).astype(jnp.int32)
        flat_lp = shard_lp.reshape((-1, k_slots))
        flat_valid = shard_valid.reshape((-1, k_slots)).astype(jnp.bool_)
        flat_w = shard_w.reshape((-1,)).astype(jnp.float32)
        rows = flat_s_h.shape[0]
        if rows == 0:
            local_sum = jnp.zeros((), dtype=jnp.float32)
        else:
            effective_block = min(token_block_size, rows)
            local_sum = _topk_kl_weighted_sum(
                flat_s_h,
                shard_s_w,
                flat_ids,
                jax.lax.stop_gradient(flat_lp),
                flat_valid,
                flat_w,
                int(effective_block),
                precision,
                flat_lp.dtype,
                int(k_slots),
            )
        return _psum_over_axes(local_sum, token_sharding_axes)

    if not has_mesh:
        return _shard_fn(student_hidden, student_lm_head, teacher_topk.ids, teacher_topk.logprobs, valid_arr, weight)

    hidden_spec = P(*token_dim_specs, None)
    lm_head_spec = P(None, None)
    slot_spec = P(*token_dim_specs, None)
    weight_spec = P(*token_dim_specs)
    student_hidden = _reshard_for_shard_map(student_hidden, mesh, hidden_spec)
    student_lm_head = _reshard_for_shard_map(student_lm_head, mesh, lm_head_spec)
    ids = _reshard_for_shard_map(teacher_topk.ids, mesh, slot_spec)
    logprobs = _reshard_for_shard_map(teacher_topk.logprobs, mesh, slot_spec)
    valid_arr = _reshard_for_shard_map(valid_arr, mesh, slot_spec)
    weight = _reshard_for_shard_map(weight, mesh, weight_spec)

    return jax.shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(hidden_spec, lm_head_spec, slot_spec, slot_spec, slot_spec, weight_spec),
        out_specs=P(),
        check_vma=False,
    )(student_hidden, student_lm_head, ids, logprobs, valid_arr, weight)


__all__ = [
    "CachedTeacherTopK",
    "TeacherMatchingMode",
    "TemperatureMatchingConfig",
    "cached_topk_has_valid_entries",
    "chunked_linear_cached_topk_kl_weighted_sum",
    "chunked_linear_tempered_kl_weighted_sum",
    "global_l2_norm",
    "source_names_for_mixture_indices",
    "tm_row_mask_for_mixture_indices",
    "validate_temperature_matching_trainer_config",
    "validate_tm_source_names",
]
