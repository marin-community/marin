# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter binding for paired and spectral RandRAND.

Matrix axes are flattened as (Out, In), preserving a whole attention projection.
Only leading axes outside Linear.In/Out (e.g. the scanned layer axis) are split.
All other parameters use AdamW; embedding and output-head routes are explicit.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import NamedTuple
from enum import StrEnum

import haliax as hax
import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig

from levanter.optim import randrand_core as rr
from levanter.optim import randrand_spectral as spectral_rr


class RandRANDVariant(StrEnum):
    PAIRED = "paired"
    SPECTRAL = "spectral"


class NamedRandRANDState(NamedTuple):
    count: jax.Array
    first: object
    second: object
    matrices: tuple
    key: jax.Array


@dataclass(frozen=True)
class LeafLayout:
    kind: str
    permutation: tuple[int, ...]
    ordered_shape: tuple[int, ...]
    batch_size: int
    rows: int
    columns: int


def _is_parameter(x):
    return isinstance(x, hax.NamedArray) and x.array is not None


def _axes(spec):
    return (spec,) if isinstance(spec, hax.Axis) else tuple(spec)


def parameter_layout(params):
    """Derive per-array roles and matrix layouts from actual Haliax modules."""

    def is_module(x):
        return isinstance(x, (hax.nn.Linear, hax.nn.Embedding))

    modules = jax.tree_util.tree_leaves_with_path(params, is_leaf=is_module)
    metadata: dict[tuple, tuple[str, tuple[hax.Axis, ...], tuple[hax.Axis, ...]]] = {}
    for path, module in modules:
        if not is_module(module):
            continue
        weight_path = (*path, jax.tree_util.GetAttrKey("weight"))
        if isinstance(module, hax.nn.Embedding):
            metadata[weight_path] = ("embed", (), ())
        elif any(isinstance(key, jax.tree_util.GetAttrKey) and key.name == "lm_head" for key in path):
            metadata[weight_path] = ("head", (), ())
        else:
            metadata[weight_path] = ("matrix", _axes(module.Out), _axes(module.In))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params, is_leaf=_is_parameter)
    layouts = []
    for path, leaf in leaves:
        if not isinstance(leaf, hax.NamedArray):
            raise TypeError(f"Expected filtered Haliax parameters, got {type(leaf)} at {path}")
        kind, out_axes, in_axes = metadata.get(path, ("vector", (), ()))
        if kind != "matrix":
            # Scanned normalization vectors may have multiple storage axes.
            # Their update is still elementwise AdamW, with one common vector LR.
            layouts.append(LeafLayout(kind, tuple(range(leaf.ndim)), leaf.array.shape, 1, 0, 0))
            continue
        batch_axes = tuple(a for a in leaf.axes if a not in (*out_axes, *in_axes))
        order = (*batch_axes, *out_axes, *in_axes)
        if len(set(order)) != len(leaf.axes) or set(order) != set(leaf.axes):
            raise ValueError(f"Linear In/Out axes do not partition {path}")
        permutation = tuple(leaf.axes.index(a) for a in order)
        layouts.append(
            LeafLayout(
                kind,
                permutation,
                tuple(a.size for a in order),
                math.prod(a.size for a in batch_axes),
                math.prod(a.size for a in out_axes),
                math.prod(a.size for a in in_axes),
            )
        )
    if not any(x.kind == "embed" for x in layouts) or not any(x.kind == "head" for x in layouts):
        raise ValueError("This Qwen3 adapter requires explicit, untied embedding and lm_head modules")
    return tuple(layouts), treedef


def _pack(params, layouts):
    leaves = jax.tree_util.tree_leaves(params, is_leaf=_is_parameter)
    arrays = []
    for leaf, layout in zip(leaves, layouts, strict=True):
        if layout.kind == "matrix":
            batch = jnp.transpose(leaf.array, layout.permutation).reshape(
                layout.batch_size, layout.rows, layout.columns
            )
            arrays.extend(batch[i] for i in range(layout.batch_size))
        else:
            arrays.append(leaf.array)
    return tuple(arrays)


def _unpack(arrays, template, layouts, treedef):
    leaves = jax.tree_util.tree_leaves(template, is_leaf=_is_parameter)
    rebuilt, cursor = [], 0
    for leaf, layout in zip(leaves, layouts, strict=True):
        if layout.kind == "matrix":
            array = jnp.stack(arrays[cursor : cursor + layout.batch_size]).reshape(layout.ordered_shape)
            inverse = tuple(layout.permutation.index(i) for i in range(len(layout.permutation)))
            array = jnp.transpose(array, inverse)
            cursor += layout.batch_size
        else:
            array = arrays[cursor]
            cursor += 1
        rebuilt.append(hax.NamedArray(array, leaf.axes))
    if cursor != len(arrays):
        raise ValueError("Optimizer update count does not match parameter layout")
    return treedef.unflatten(rebuilt)


def named_randrand(config, *, total_steps, schedule, spectral=None):
    """Wrap the plain-array optimizer without losing Haliax parameter axes."""

    def optimizer_for_layouts(layouts):
        labels = tuple(layout.kind for layout in layouts for _ in range(layout.batch_size))
        if spectral is not None:
            return spectral_rr.randrand(labels, config, spectral, total_steps=total_steps, schedule=schedule)
        return rr.randrand(labels, config, total_steps=total_steps, schedule=schedule)

    def name_state(raw, params, layouts, treedef):
        # Preserve parameter axes on the large Adam buffers so Levanter's FSDP
        # mapping shards them just like the weights. Gram/sketch/basis state is replicated.
        return NamedRandRANDState(
            raw.count,
            _unpack(raw.first, params, layouts, treedef),
            _unpack(raw.second, params, layouts, treedef),
            raw.matrices,
            raw.key,
        )

    def init_fn(params):
        layouts, treedef = parameter_layout(params)
        tx = optimizer_for_layouts(layouts)
        return name_state(tx.init(_pack(params, layouts)), params, layouts, treedef)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("RandRAND requires params")
        layouts, treedef = parameter_layout(params)
        tx = optimizer_for_layouts(layouts)
        if jax.tree_util.tree_structure(updates, is_leaf=_is_parameter) != treedef:
            raise ValueError("Gradient and parameter structures differ")
        raw_state = rr.RandRANDState(
            state.count, _pack(state.first, layouts), _pack(state.second, layouts), state.matrices, state.key
        )
        raw_updates, raw_state = tx.update(_pack(updates, layouts), raw_state, _pack(params, layouts))
        return _unpack(raw_updates, params, layouts, treedef), name_state(raw_state, params, layouts, treedef)

    # Trainer passes objective/loss kwargs for optimizers that need them.
    # RandRAND only consumes gradients; use Optax's standard extra-argument adapter.
    return optax.with_extra_args_support(optax.GradientTransformation(init_fn, update_fn))


@OptimizerConfig.register_subclass("randrand")
@dataclass(frozen=True)
class RandRANDConfig(OptimizerConfig):
    """Marin optimizer config. learning_rate and weight_decay refer to body matrices.

    Route LRs are peak rates. The inherited Levanter scheduler supplies one shared
    multiplier; clipping occurs inside RandRAND, once per logical optimizer step.
    """

    learning_rate: float = rr.MATRIX_LR
    weight_decay: float = rr.MATRIX_WEIGHT_DECAY
    warmup: int | float = rr.WARM_UP
    min_lr_ratio: float = rr.MIN_LR_RATIO
    matrix_betas: tuple[float, float] = rr.MATRIX_BETAS
    matrix_eps: float = rr.MATRIX_EPS
    embed_lr: float = rr.EMBED_LR
    embed_betas: tuple[float, float] = rr.EMBED_BETAS
    embed_weight_decay: float = rr.EMBED_WEIGHT_DECAY
    head_lr: float = rr.HEAD_LR
    head_betas: tuple[float, float] = rr.HEAD_BETAS
    head_weight_decay: float = rr.HEAD_WEIGHT_DECAY
    vector_lr: float = rr.VECTOR_LR
    vector_betas: tuple[float, float] = rr.VECTOR_BETAS
    vector_weight_decay: float = rr.VECTOR_WEIGHT_DECAY
    adam_eps: float = rr.ADAM_EPS
    rank: int = rr.DEFLATION_RANK
    rank_sketch: int = rr.DEFLATION_SKETCH_DIM
    randrand_warmup: int = rr.DEFLATION_WARMUP
    basis_every: int = rr.BASIS_EVERY
    basis_ema_beta: float = rr.BASIS_EMA_BETA
    max_gram_dim: int = rr.MAX_GRAM_DIM
    min_dim_for_deflation: int = rr.MIN_DIM_FOR_DEFLATION
    boost_factor: float = rr.BOOST_FACTOR
    low_rank_correction: bool = rr.LOW_RANK_CORRECTION
    correction_scale: float = rr.CORRECTION_SCALE
    pair_k: int = rr.PAIRED_K
    max_grad_norm: float = rr.GRAD_CLIP
    basis_seed: int = 42
    variant: RandRANDVariant = RandRANDVariant.PAIRED
    spectral: spectral_rr.SpectralSettings = field(default_factory=spectral_rr.SpectralSettings)

    def algorithm_config(self):
        """Translate explicit route controls without applying inherited WD masks."""
        if self.weight_decay_modules is not None or self.default_weight_decay_mask is not None:
            raise ValueError("RandRAND uses explicit route weight decay; inherited WD masks are unsupported")
        fields = {field.name for field in dataclasses.fields(rr.RandRANDConfig)}
        direct = fields - {"matrix_lr", "matrix_weight_decay", "warmup_steps", "min_lr_ratio"}
        values = {name: getattr(self, name) for name in direct}
        return rr.RandRANDConfig(**values, matrix_lr=self.learning_rate, matrix_weight_decay=self.weight_decay)

    def spectral_settings(self):
        if self.variant == RandRANDVariant.SPECTRAL:
            return self.spectral
        if self.variant == RandRANDVariant.PAIRED:
            return None
        raise ValueError(f"Unknown RandRAND variant: {self.variant}")

    def build(self, num_train_steps):
        return named_randrand(
            self.algorithm_config(),
            spectral=self.spectral_settings(),
            total_steps=num_train_steps,
            schedule=self.lr_scheduler(num_train_steps, override_lr=1.0),
        )
