# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ambient SFT full-vocabulary temperature-matching KL regularizer (arXiv 2611.23229, Eq. 4).

Implements memory-safe, full-vocabulary forward KL matching between a frozen
teacher checkpoint and the live SFT student inside a single optimizer step:

    L_step = ce_weight * L_CE(all M microbatches) + weight * L_TM(K of M microbatches)

where per token i:

    KL_i = sum_v softmax(z_{teacher, i} / T)_v
                 * (log_softmax(z_{teacher, i} / T)_v - log_softmax(z_{student, i} / T)_v)

Hinton's T^2 prefactor is intentionally omitted (arXiv 2611.23229 Sec. 5,
Prop. E.1), so d(KL_i)/d(z_{student, i}) = (p_{T, student} - p_{T, teacher}) / T
scales as 1/T and the Hessian scales as 1/T^2.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float

from levanter.data.mixture import MixtureDataset
from levanter.grug.sharding import _axis_names, _current_mesh, _reshard_for_shard_map
from levanter.sharding import partition_spec_of
from levanter.utils.jax_utils import local_cpu_mesh


@dataclass(frozen=True)
class TemperatureMatchingConfig:
    """Configuration for same-step full-vocabulary temperature matching in long-loop SFT.

    Optimizes ``ce_weight * L_CE(all M microbatches) + weight * L_TM(K of M microbatches)``
    inside a single optimizer step against a frozen BF16 teacher checkpoint on ``pinned_host``,
    using a chunked full-vocabulary forward-KL scan without Hinton's ``T^2`` prefactor
    (arXiv 2611.23229).

    When ``enabled=True``, ``weight``, ``temperature``, and ``microbatch_count`` must be supplied
    explicitly (no silent defaults on critical hyperparameters).
    """

    enabled: bool = False
    # Weight on L_CE in ce_weight * L_CE + weight * L_TM (1.0 for Ambient SFT).
    ce_weight: float = 1.0
    # Required when enabled=True: weight on L_TM in ce_weight * L_CE + weight * L_TM.
    weight: float | None = None
    # Required when enabled=True: softmax temperature T > 0 (no Hinton T^2 factor).
    temperature: float | None = None
    # Required when enabled=True: number of microbatches K (out of M) on which TM runs.
    microbatch_count: int | None = None
    # Token block size for chunked full-vocabulary KL scan (avoids [B, S, V] materialization).
    token_block_size: int = 1024
    # Keep frozen teacher weights on pinned_host and stream layer-by-layer during decoupled forward.
    offload_teacher: bool = True
    # Explicit checkpoint path for the frozen teacher. Defaults to trainer.initialize_from.
    teacher_checkpoint_path: str | None = None
    # Per-source switches (mutually exclusive): turn TM off on chosen MixtureDataset sources.
    disabled_sources: tuple[str, ...] = ()
    enabled_sources: tuple[str, ...] | None = None
    # Log full-parameter ||grad_TM|| / ||grad_CE|| watch metrics without adding HBM to train_step.
    # Runs one extra full CE + TM forward/backward pass sequentially on watch steps only.
    log_grad_norm_ratio: bool = True
    # Cadence (in steps) for ||grad_TM|| / ||grad_CE|| watch metrics. None follows watch.interval.
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


__all__ = [
    "TemperatureMatchingConfig",
    "chunked_linear_tempered_kl_weighted_sum",
    "global_l2_norm",
    "source_names_for_mixture_indices",
    "tm_row_mask_for_mixture_indices",
    "validate_temperature_matching_trainer_config",
    "validate_tm_source_names",
]
