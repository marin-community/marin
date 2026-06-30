# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Mudam — Shampoo-Muon: a block-partitioned Shampoo preconditioner composed with Muon.

For each matrix (Linear) weight the update is, on one side (input "right" or output "left"):

    M  = EMA_b1(G)                                   # first moment (momentum, Nesterov optional)
    H  = EMA_b2(G Gᵀ  or  Gᵀ G)                       # Shampoo one-sided second moment (shampoo_beta)
    Ĝ  = H^{-1/4} M    (or H^{-1/2} M)                # Shampoo whitening on the preferred side
    U  = muon_normalize(Ĝ)                            # Newton-Schulz orthogonalization (+ Kimi scaling)
    W ← W − η · U

i.e. a one-sided Shampoo preconditioner feeding a Muon orthogonalization — hence "Shampoo-Muon".
Large matrices are **block-partitioned** (``block_size``, ``partition_grads_into_blocks``) and small
dims optionally **merged** (``merge_small_dims``) to bound the eigendecomposition cost, à la
distributed Shampoo / SOAP. ``prefer_input_side`` chooses the right (input) vs left (output) factor;
``o_proj`` can be flipped to the embedding side. Embeddings / lm_head / biases use AdamW.

Knobs of note: ``shampoo_beta`` (H EMA), ``block_size``/``merge_small_dims`` (cost control),
``normalization`` (``"muon"`` orthogonalization), ``use_scaling`` (``"kimi"`` LR scaling),
``interpolation_to_muon`` / ``debug_to_muon`` (ablate back to plain Muon), and momentum/Nesterov
placement in the first/second moments. Authored by Kaiyue Wen; ported from a standalone levanter
checkout. Variants ``Mudam2Config`` / ``Mudam3Config`` are later iterations of the same idea.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Any, List, Optional, Tuple, Union
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu
import chex
from jax import vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from jaxtyping import Array
from optax import GradientTransformation, Updates
from optax._src.utils import canonicalize_dtype

import haliax as hax

from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths
from haliax.nn import Linear

jax.config.update("jax_enable_x64", False)


@OptimizerConfig.register_subclass("mudam")
@dataclass(frozen=True)
class MudamConfig(OptimizerConfig):
    weight_decay: float = 0.0
    beta1: float = 0.95
    momentum: float = 0.95
    momentum_2: float = 1.0
    momentum_out: float = 1.0
    shampoo_beta: float = 0.95
    muon_epsilon: float = 1e-5
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    haps: Optional[list[int]] = None
    schedule_list: Optional[list[str]] = None
    max_precond_dim: int = 10000
    merge_small_dims: bool = True
    target_merged_dim_size: int = 2048
    mu_dtype: Optional[str] = None
    precond_dtype: Optional[str] = None
    partition_grads_into_blocks: bool = False
    adam_lr: float = 6e-4
    block_size: int = 256
    steps: int = 5
    force_full_rank: bool = False
    normalize_gradient: bool = False
    use_mudam_for_embedding: bool = False
    use_nesterov_in_second_moment: bool = False
    use_momentum_in_second_moment: bool = False
    prefer_embedding_side: bool = False
    prefer_input_side: bool = True
    update_normed: bool = False
    use_nesterov_in_first_moment: bool = True
    nesterov_adam: bool = True
    another_muon: Optional[bool] = None
    normalization: Optional[str] = "muon"
    use_scaling: Optional[str] = "kimi"
    interpolation_to_muon: float = 0.0
    debug_to_muon: bool = False
    spectral_normalize: bool = False

    def build(self, num_train_steps):
        """Creates the optimizer"""
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def optimizer(learning_rate, adam_lr):

            def muon_transform(prefer_input_side: bool = True):
                components = []
                components.append(
                    scale_by_mudam(
                        b1=self.momentum,
                        b1_2=self.momentum_2,
                        b1_out=self.momentum_out,
                        b2=self.shampoo_beta,
                        steps=self.steps,
                        epsilon=self.epsilon,
                        muon_epsilon=self.muon_epsilon,
                        max_precond_dim=self.max_precond_dim,
                        merge_small_dims=self.merge_small_dims,
                        target_merged_dim_size=self.target_merged_dim_size,
                        mu_dtype=self.mu_dtype,
                        interpolation_to_muon=self.interpolation_to_muon,
                        precond_dtype=self.precond_dtype,
                        partition_grads_into_blocks=self.partition_grads_into_blocks,
                        normalize_gradient=self.normalize_gradient,
                        block_size=self.block_size,
                        use_nesterov_in_second_moment=self.use_nesterov_in_second_moment,
                        use_nesterov_in_first_moment=self.use_nesterov_in_first_moment,
                        use_momentum_in_second_moment=self.use_momentum_in_second_moment,
                        prefer_input_side=prefer_input_side,
                        force_full_rank=self.force_full_rank,
                        normalization=self.normalization,
                        another_muon=self.another_muon,
                        use_scaling=self.use_scaling,
                        update_normed=self.update_normed,
                        debug_to_muon=self.debug_to_muon,
                        spectral_normalize=self.spectral_normalize,
                    )
                )
                if self.weight_decay > 0:
                    # No mask: the muon group is only Linear weights (norms/biases are routed to
                    # adamw), so decay applies to all of them — and build_weight_decay_mask() crashes
                    # on the masked qk_norm NamedArrays (MaskedNode) that land in this group.
                    components.append(optax.add_decayed_weights(self.weight_decay))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    optax.scale_by_adam(self.beta1, self.beta2, self.epsilon, nesterov=self.nesterov_adam)
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            # Only include the muon group(s) the mask actually routes to — otherwise
            # multi_transform inits the unused group on an all-masked tree, which crashes
            # Mudam's raw shape inference on MaskedNode leaves.
            transformations = {"adamw": adamw_transform()}
            if self.prefer_embedding_side:
                transformations["muon_left"] = muon_transform(prefer_input_side=False)
                transformations["muon_right"] = muon_transform(prefer_input_side=True)
            elif self.prefer_input_side:
                transformations["muon_right"] = muon_transform(prefer_input_side=True)
            else:
                transformations["muon_left"] = muon_transform(prefer_input_side=False)

            return optax.multi_transform(
                transformations,
                partial(
                    self.create_mask,
                    use_mudam_for_embedding=self.use_mudam_for_embedding,
                    prefer_embedding_side=self.prefer_embedding_side,
                    prefer_input_side=self.prefer_input_side,
                ),
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(
        self,
        params,
        use_mudam_for_embedding: bool = False,
        prefer_embedding_side: bool = False,
        prefer_input_side: bool = True,
    ):
        """
        Creates a mask that labels parameters as 'muon' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)
        assert not (prefer_embedding_side and prefer_input_side), "Cannot have two preferences"

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                if use_mudam_for_embedding:
                    return "muon_right"
                else:
                    return "adamw"
            elif isinstance(param, Linear):
                # muon for linear layers
                if prefer_input_side:
                    optim_type = "muon_right"
                elif prefer_embedding_side:
                    if "o_proj" in path_str:
                        optim_type = "muon_left"
                    else:
                        optim_type = "muon_right"
                else:
                    optim_type = "muon_left"
                return dataclasses.replace(param, weight=optim_type, bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


def _safe_sharding_constraint(x, sharding):
    if sharding is None:
        return x
    else:
        return with_sharding_constraint(x, sharding)


def p_over_x(b1, b2, use_nesterov_in_first_moment):
    if not use_nesterov_in_first_moment:
        # denominator is N = (1 - b2) * (g_t^2 + b2 * g_{t-1}^2 + b2^2 * g_{t-2}^2 + ...) / (1 - b2^t)
        # nominator is M = (g_t +  b1 * g_{t-1} +  b1^2 * g_{t-2} + ...) * (1 - b1) / (1 - b1^t)
        # output a number c such that c^2 N >= M^2 for all t
        return jnp.sqrt(1 / ((1 - b2) * (1 - b1**2 / b2))) * (1 - b1)
    else:
        # denominator is N = (1 - b2) * (g_t^2 + b2 * g_{t-1}^2 + b2^2 * g_{t-2}^2 + ...) / (1 - b2^t)
        # nominator is M = (g_t + b1 * g_t +  b1^2 * g_{t-1} +  b1^3 * g_{t-2} + ...) * (1 - b1) / (1 - b1^{t + 1})
        # output a number c such that c^2 N >= M^2 for all t
        term = (1 + b1) ** 2 + b1**4 / (1 - b1**2 / b2)
        return jnp.sqrt(1 / (1 - b2) * term) * (1 - b1)


def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Maybe map a fn along multiple leading axes."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        mapped_fn = lambda xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.lax.map(mapped_fn, xs=args, batch_size=bs if bs > 1 else None)
    else:
        mapped_fn = lambda *xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return vmap(mapped_fn)(*args)


def scale_by_mudam(
    b1: float = 0.95,
    b1_out: float = 1.0,
    b1_2: float = 1.0,
    b2: float = 0.95,
    steps: int = 5,
    epsilon: float = 1e-8,
    interpolation_to_muon: float = 0.0,
    muon_epsilon: float = 1e-10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Any = None,
    precond_dtype: Any = None,
    partition_grads_into_blocks: Optional[bool] = True,
    normalize_gradient: Optional[bool] = True,
    block_size: Optional[int] = 256,
    lax_map_scanned_layers: Optional[bool] = True,
    lax_map_batch_size: Optional[int] = 4,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    use_nesterov_in_second_moment: bool = False,
    use_nesterov_in_first_moment: bool = True,
    use_momentum_in_second_moment: bool = False,
    prefer_input_side: bool = True,
    force_full_rank: bool = True,
    normalization: Optional[str] = "muon",
    another_muon: Optional[bool] = None,
    use_scaling: Optional[str] = "kimi",
    update_normed: bool = False,
    debug_to_muon: bool = False,
    spectral_normalize: bool = False,
) -> GradientTransformation:
    mu_dtype = canonicalize_dtype(mu_dtype) if mu_dtype is not None else None
    precond_dtype = canonicalize_dtype(precond_dtype) if precond_dtype is not None else None

    def init_fn(params: Updates) -> dict:
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            params,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )
        params_sharding_ = hax.partitioning.infer_resource_partitions(params)
        params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
        shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], params, scanned_layers_)

        shapes_leaf = jax.tree.map(
            lambda p, s: p.shape[int(s) :], jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
        )

        exp_avg = otu.tree_zeros_like(params, dtype=mu_dtype)
        scanned_dim_sharding = [
            PartitionSpec(sh[0]) if s else None
            for sh, s in zip(jax.tree.leaves(params_sharding_), jax.tree.leaves(scanned_layers_))
        ]

        merged_shapes = shapes
        merged_shapes_leaf = shapes_leaf
        if merge_small_dims:
            merged_shapes = jax.tree.map(
                lambda p, s: _merge_small_dims(p.shape[int(s) :], target_merged_dim_size),
                params,
                scanned_layers_,
            )
            merged_shapes_leaf = [
                _merge_small_dims(p.shape[int(s) :], target_merged_dim_size)
                for p, s in zip(
                    jax.tree.leaves(params),
                    jax.tree.leaves(scanned_layers_),
                )
            ]
        partitioned_shapes = merged_shapes
        partitioned_shapes_leaf = merged_shapes_leaf
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps: BlockPartitioner(ps, block_size),
                params,
                partitioned_shapes,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners)
            partitioned_shapes_leaf = [p_cls._padded_stacked_shape for p_cls in jax.tree.leaves(partitioners)]
            print("Block partitioned shapes: ", partitioned_shapes_leaf)

        def broadcast_qs(_, ps, q, s):
            stack_n = ps[0]
            if partition_grads_into_blocks:
                # add leading dim for stacked partitions
                q = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), stack_n, axis=0), q)
            if s > 0:
                # add leading dim if we're scanning this layer
                q = jax.tree.map(lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), q)
            return q

        def add_dims_to_spec(qss, sds):
            if partition_grads_into_blocks:
                qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
            if sds is not None:
                qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
            return qss

        scanned_sizes = jax.tree.map(lambda p, s: p.shape[0] if s else 0, params, scanned_layers_)

        GG_and_sharding = [
            init_conditioner(
                t[1:] if partition_grads_into_blocks else t,
                max_precond_dim,
                precond_dtype,
                prefer_input_side,
                force_full_rank,
            )
            for t in partitioned_shapes_leaf
        ]
        GG = [x[0] for x in GG_and_sharding]
        GG_sharding_without_scan = [x[1] for x in GG_and_sharding]
        GG = [
            broadcast_qs(None, shape, gg, size)
            for shape, gg, size in zip(partitioned_shapes_leaf, GG, jax.tree.leaves(scanned_sizes))
        ]
        GG_sharding = [add_dims_to_spec(qss, sds) for qss, sds in zip(GG_sharding_without_scan, scanned_dim_sharding)]
        GG = _safe_sharding_constraint(GG, GG_sharding)

        state = {
            "count": jnp.zeros([], jnp.int32),
            "exp_avg": exp_avg,
            "GG": GG,
        }
        if b1_2 < 1:
            last_gradient = otu.tree_zeros_like(params, dtype=mu_dtype)
            state["last_gradient"] = last_gradient
        if b1_out < 1:
            update_momentum = otu.tree_zeros_like(params, dtype=mu_dtype)
            state["update_momentum"] = update_momentum
        return state

    def update_step(updates: Updates, state: dict, scanned_layers_: Updates) -> tuple[Updates, dict]:
        # Update moments
        _, grads_structure = jax.tree.flatten(updates, is_leaf=lambda x: isinstance(x, jax.Array))

        if normalize_gradient:
            updates = jax.tree.map(lambda g: g / jnp.linalg.norm(g), updates)

        exp_avg = state["exp_avg"]
        if b1_2 < 1:
            last_gradient = state["last_gradient"]
            current_gradient = updates
            updates = jax.tree.map(lambda g, g_old: g + (1 - b1_2) * (g - g_old), updates, last_gradient)
        gradient = updates
        exp_avg = jax.tree.map(lambda m, g: None if g is None else b1 * m + g, exp_avg, updates)
        updates = jax.tree.map(lambda m, g: None if g is None else b1 * m + g, exp_avg, updates)
        shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], updates, scanned_layers_)
        # bias correction for nominator
        if use_nesterov_in_first_moment:
            nominator = jax.tree.map(
                lambda g: (1 - b1) * g / (1 - b1 ** (state["count"] + 1)) if g is not None else None, updates
            )
        else:
            nominator = jax.tree.map(
                lambda g: (1 - b1) * g / (1 - b1 ** state["count"]) if g is not None else None, exp_avg
            )

        # bias correction for denominator
        if use_nesterov_in_second_moment:
            denominator_diff = jax.tree.map(
                lambda g: (1 - b1) * g / (1 - b1 ** (state["count"] + 1)) if g is not None else None, updates
            )
        elif use_momentum_in_second_moment:
            denominator_diff = jax.tree.map(
                lambda g: (1 - b1) * g / (1 - b1 ** state["count"]) if g is not None else None, exp_avg
            )
        else:
            denominator_diff = gradient

        # block gradients, exp_avg
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        # merge small dims
        merged_shapes = shapes
        if merge_small_dims:
            original_shapes = shapes
            merged_shapes = jax.tree.map(
                lambda g, s: _merge_small_dims(g.shape[int(s) :], target_merged_dim_size),
                nominator,
                scanned_layers_,
            )
            # reshape
            nominator = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                nominator,
                scanned_layers_,
                merged_shapes,
            )
            denominator_diff = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                denominator_diff,
                scanned_layers_,
                merged_shapes,
            )

        # partition
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps: BlockPartitioner(ps, block_size),
                nominator,
                partitioned_shapes,
            )
            blocked_nominator = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                nominator,
                partitioners,
                scanned_layers_,
            )
            blocked_denominator_diff = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                denominator_diff,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                blocked_nominator,
                scanned_layers_,
            )
            blocked_nominator = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_nominator,
                scanned_layers_,
            )
            blocked_denominator_diff = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_denominator_diff,
                scanned_layers_,
            )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        else:
            blocked_nominator = nominator
            blocked_denominator_diff = denominator_diff

        # first update denominator
        new_GG = jax.tree.map(
            lambda nm, grad, gg: _map_fn(False, 0, nm, partial(update_preconditioner, beta=b2), grad, gg),
            jax.tree.leaves(n_dims_to_map),
            jax.tree.leaves(blocked_denominator_diff),
            state["GG"],
        )

        assert b2 > b1**2, "b2 must be greater than b1^2 for the shampoo to have bounded norm update"
        P_over_X = p_over_x(b1, b2, use_nesterov_in_first_moment)
        denominator = jax.tree.map(lambda gg: (1 - b2 ** state["count"]) / (1 - b2) * gg, new_GG)

        if not debug_to_muon:
            blocked_norm_updates_leaves = jax.tree.map(
                lambda _, nm, e, gg: _map_fn(
                    False,
                    0,
                    nm,
                    partial(
                        ns_generalized,
                        b2=b2,
                        steps=steps,
                        eps=muon_epsilon,
                        P_over_X=P_over_X,
                        normalization=normalization,
                        another_muon=another_muon,
                        use_scaling=use_scaling,
                        update_normed=update_normed,
                        interpolation_to_muon=interpolation_to_muon,
                        spectral_normalize=spectral_normalize,
                    ),
                    e,
                    gg,
                ),
                jax.tree.leaves(dummy_updates_tree),
                jax.tree.leaves(n_dims_to_map),
                jax.tree.leaves(blocked_nominator),
                denominator,
            )
        else:
            blocked_norm_updates_leaves = jax.tree.map(
                lambda _, nm, e: _map_fn(
                    False,
                    0,
                    nm,
                    partial(
                        ns_for_debug,
                        steps=steps,
                        eps=muon_epsilon,
                        use_scaling=use_scaling,
                        normalization=normalization,
                    ),
                    e,
                ),
                jax.tree.leaves(dummy_updates_tree),
                jax.tree.leaves(n_dims_to_map),
                jax.tree.leaves(blocked_nominator),
            )
        blocked_norm_updates = grads_structure.unflatten(blocked_norm_updates_leaves)

        # revert blocking of everything
        if partition_grads_into_blocks:
            norm_updates = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                blocked_norm_updates,
                scanned_layers_,
                partitioned_shapes,
            )
            norm_updates = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                norm_updates,
                scanned_layers_,
                partitioners,
            )
        else:
            norm_updates = blocked_norm_updates

        # unmerge
        if merge_small_dims:
            norm_updates = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                norm_updates,
                scanned_layers_,
                original_shapes,
            )

        # precision
        new_GG = otu.tree_cast(new_GG, precond_dtype)
        exp_avg = otu.tree_cast(exp_avg, mu_dtype)

        new_state = {
            "GG": new_GG,
            "exp_avg": exp_avg,
            "count": state["count"],
        }

        if b1_2 < 1:
            new_state["last_gradient"] = current_gradient
        if b1_out < 1:
            new_state["update_momentum"] = jax.tree.map(
                lambda g, m: g + (1 - b1_out) * m, norm_updates, state["update_momentum"]
            )
            # bias correction for update
            norm_updates = jax.tree.map(
                lambda m: (1 - b1_out) * m / (1 - b1_out ** state["count"]) if m is not None else None,
                new_state["update_momentum"],
            )

        return norm_updates, new_state

    def update_fn(updates: Updates, state: dict, params: Optional[Updates] = None) -> tuple[Updates, dict]:
        count_inc = jnp.asarray(optax.safe_int32_increment(state["count"]))
        state["count"] = count_inc
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            params,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )

        updates, new_state = update_step(updates, state, scanned_layers_)

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def update_preconditioner(
    grad: Array,
    GG: List[Union[Array, None]],
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> List[Union[Array, None]]:
    if grad.ndim == 1:
        return [jnp.matmul(grad[:, None], grad[None, :], precision=precision) + beta * GG[0]]  # type: ignore

    def update_gg(idx, gg):
        if gg is None:
            return None
        outer_product = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        return outer_product + beta * gg

    new_GG = jax.tree.map(update_gg, list(range(len(GG))), GG)

    return new_GG


def ns_for_debug(X, steps=5, eps=1e-7, normalization="muon", use_scaling="kimi"):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    if normalization == "muon":
        a, b, c = (3.4445, -4.7750, 2.0315)
        hs = [(a, b, c)] * steps
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    # assert X.shape[0] <= X.shape[1], "X should have more columns than rows for this implementation."
    # TODO: we should be smarter and also transpose if they're ~the same and X is already sharded along first axis
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    if len(hax.partitioning._get_mesh().devices):
        X = jax.lax.with_sharding_constraint(X, PartitionSpec(None, ("data", "model")))

    for i in range(len(hs)):
        a, b, c = hs[i]
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T
    if use_scaling == "kimi":
        scale = jnp.sqrt(jnp.maximum(X.shape[0], X.shape[1]))
    elif use_scaling == "zhiyuan":
        scale = jnp.sqrt(jnp.minimum(X.shape[0], X.shape[1]))
    elif use_scaling == "muon":
        scale = jnp.sqrt(jnp.maximum(1, X.shape[0] / X.shape[1]))
    else:
        scale = jnp.sqrt(X.shape[0])
    return X * scale


def approximated_spectral_norm(X, num_iterations=10):
    """
    Approximates the spectral norm of a matrix X using power iteration.
    """
    # Initialize a random vector with the correct shape.
    # The vector v should have shape (X.shape[1], 1)
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (X.shape[1],))  # use a fixed probing vector for now
    v = v / jnp.linalg.norm(v)

    # Power iteration loop
    for _ in range(num_iterations):
        # This is more numerically stable and efficient than forming X.T @ X
        u = X @ v
        v = X.T @ u
        v = v / jnp.linalg.norm(v)

    # The spectral norm is the L2 norm of (X @ v)
    final_norm = jnp.linalg.norm(X @ v)
    return final_norm


def ns_generalized(
    X: Array,
    GG: List[Union[Array, None]],
    steps: int = 5,
    b2: float = 0.0,
    eps: float = 1e-7,
    P_over_X: Any = 1.0,
    normalization: Optional[str] = "muon",
    interpolation_to_muon: float = 0.0,
    another_muon: Optional[bool] = None,
    spectral_normalize: bool = False,
    use_scaling: Optional[str] = "kimi",
    update_normed: bool = False,
) -> Array:
    idx = 0
    for i, mat in enumerate(GG):
        if mat is not None:  # noqa: SIM108
            idx = i
    chex.assert_rank(X, 2)

    transpose = False
    if idx == 1:
        X = X.T
        transpose = True

    P_raw = GG[idx]

    # if len(hax.partitioning._get_mesh().devices):
    # X = jax.lax.with_sharding_constraint(X, PartitionSpec(None, ("data", "model")))
    # P_raw = jax.lax.with_sharding_constraint(P_raw, PartitionSpec(None, None))

    # for nominator X, need to subtract XX.T from the denominator
    P = interpolation_to_muon * X @ X.T + (1 - interpolation_to_muon) * P_raw  # pyrefly: ignore[unsupported-operation]

    if normalization == "svd":
        s, u = jnp.linalg.eigh(P + X @ X.T)
        s_inv_sqrt = jnp.where(s > eps, 1.0 / jnp.sqrt(s), 0.0)
        P_inv_sqrt = u @ jnp.diag(s_inv_sqrt) @ u.T
        X = P_inv_sqrt @ X
    elif normalization == "jianlin":
        I1 = jnp.eye(P.shape[0])
        P = P + eps * I1
        normalized_factor_p = jnp.sqrt(jnp.trace(P @ P))
        normalized_factor_x = jnp.linalg.norm(X) + eps
        P = P / normalized_factor_p
        X = X / normalized_factor_x
        hs = [
            [2.003698915546185, -0.12738228464323412, -0.9913739512092223],
            [1.7312012486502126, 0.11574802804004483, -0.650104836053695],
            [1.756732832712327, 0.08760436701325444, -0.6062859774838089],
            [1.6774879041707145, 0.1265063444437542, -0.5850694621322253],
            [1.5240377698859244, 0.16539061780947165, -0.5799808499244595],
            [1.5389518470783554, 0.15096635863051178, -0.5781799066369828],
            [1.5230508848208697, 0.18063663836127025, -0.5433476761697408],
            [0.8831254701177772, 0.6900069105312058, -0.5045468048486641],
            [0.8885959911978427, 0.5789969826532673, -0.4718140813257857],
            [1.0230225077411645, 0.4597415301949877, -0.48280094211975133],
        ]
        for a, b, c in hs:
            W1 = a * I1 + b * P + c * P @ P
            X = W1 @ X
            P = 0.5 * (W1 @ (P + P.T) @ W1.T)
        X = X * normalized_factor_x / (normalized_factor_p**0.5)
    else:
        # calling the subprocess to compute
        P_over_X = jnp.sqrt(interpolation_to_muon + P_over_X**2 * (1 - interpolation_to_muon))
        X = X / P_over_X
        P = P - X @ X.T
        # when not including nominator, need to add eps * I to ensure P is positive definite
        normalized_factor = jnp.sqrt(jnp.sqrt(jnp.trace(P @ P)) + eps + jnp.linalg.norm(X) ** 2)
        # jax.debug.print("X norm {x}, P norm {p}, Normalized factor {n}", x = jnp.linalg.norm(X), p = jnp.sqrt(jnp.trace(P)), n = normalized_factor)
        P = P + eps * jnp.eye(P.shape[0])  # avoid non positive definite
        X /= normalized_factor
        P /= normalized_factor**2
        if normalization == "polar":
            coeffs_list_polar = [
                (8.28721201814563, -23.595886519098837, 17.300387312530933),
                (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
                (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
                (3.3184196573706015, -2.488488024314874, 0.51004894012372),
                (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
                (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
                (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
                (1.875, -1.25, 0.375),
            ]  # subsequent coeffs equal this numerically
            # safety factor for numerical stability (but exclude last polynomial)
            coeffs_list_polar = [(a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list_polar[:-1]] + [
                coeffs_list_polar[-1]
            ]
            hs = coeffs_list_polar[:steps] + [coeffs_list_polar[-1]] * max(0, steps - len(coeffs_list_polar))
        elif normalization == "heavy":
            coefficient_list = [
                [2.7016027669794043, -1.602970321850751, 0.39423894803566567],
                [2.8401746851601453, -1.6053566347080421, 0.3840046561658029],
                [2.7156109765394056, -1.6695204205086023, 0.32209664174118957],
                [2.601508909818141, -1.6226872362221807, 0.3657957045623002],
                [2.4751052423152244, -1.5953618300406027, 0.394339195930196],
                [2.585463724527698, -1.5768104299646881, 0.41509996062370064],
                [2.6521988290513394, -1.6717314166731414, 0.3240032693041669],
                [2.0441429057969813, -1.5153187043371619, 0.47478569113020946],
                [1.9630651968491937, -1.5499418485715168, 0.4516820484751629],
                [2.008830255477319, -1.5030574864379898, 0.48563512296169237],
            ]
            hs = coefficient_list  # for now ignore steps, assume it always equals to 10
        elif normalization == "muon":
            a, b, c = (3.4445, -4.7750, 2.0315)
            hs = [(a, b, c)] * steps
        else:
            a, b, c = (2.0, -1.5, 0.5)
            hs = [(a, b, c)] * steps
        for a, b, c in hs:
            A = X @ X.T + P
            B = b * A + c * A @ A
            X = a * X + B @ X
            P = a * a * P + a * (B @ P + P @ B) + B @ P @ B
        X = X * P_over_X

    if another_muon:
        X = X / (jnp.linalg.norm(X) + eps)
        for a, b, c in hs:
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    if spectral_normalize:
        # spectral_norm = jnp.linalg.norm(X, ord=2)
        spectral_norm = approximated_spectral_norm(X)
        X = X / spectral_norm

    if transpose:
        X = X.T

    if update_normed:
        X = X / jnp.linalg.norm(X)

    if use_scaling == "kimi":
        scale = jnp.sqrt(jnp.maximum(X.shape[0], X.shape[1]))
    elif use_scaling == "zhiyuan":
        scale = jnp.sqrt(jnp.minimum(X.shape[0], X.shape[1]))
    elif use_scaling == "muon":
        scale = jnp.sqrt(jnp.maximum(1, X.shape[0] / X.shape[1]))
    elif use_scaling == "interpolation":
        scale = jnp.sqrt(
            jnp.maximum(1, X.shape[0] / X.shape[1]) * interpolation_to_muon + X.shape[0] * (1 - interpolation_to_muon)
        )
    elif use_scaling == "interpolation2":
        scale = jnp.sqrt(
            jnp.maximum(1, X.shape[0] / X.shape[1]) * interpolation_to_muon
            + 16 * X.shape[0] * (1 - interpolation_to_muon)
        )
    elif use_scaling == "interpolation3":
        scale = jnp.sqrt(
            jnp.maximum(1, X.shape[0] / X.shape[1]) * interpolation_to_muon
            + 64 * X.shape[0] * (1 - interpolation_to_muon)
        )
    elif use_scaling == "interpolation4":
        scale = jnp.sqrt(
            jnp.maximum(1, X.shape[0] / X.shape[1]) * interpolation_to_muon
            + 32 * X.shape[0] * (1 - interpolation_to_muon)
        )
    else:
        scale = jnp.sqrt(X.shape[0])
    return X * scale


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_precond_dim: int, prefer_input_side: bool = True, force_full_rank: bool = True
) -> List[bool]:
    if len(shape) == 0:
        return [False]

    if len(shape) == 1:
        return [False]

    min_dim = min(shape)
    new_result = []
    flag = True
    if prefer_input_side:
        shape = shape[::-1]
    for i in range(len(shape)):
        if (shape[i] == min_dim or (not force_full_rank)) and flag:
            flag = False
            new_result.append(False)
        else:
            new_result.append(True)
    if prefer_input_side:
        new_result = new_result[::-1]
    return new_result


def init_conditioner(
    p_shape,
    max_precond_dim: int,
    dtype: Optional[Union[str, jnp.dtype]],
    prefer_input_side: bool = True,
    force_full_rank: bool = True,
):
    if len(p_shape) == 1:
        return ([jnp.zeros((p_shape[0], p_shape[0]), dtype=dtype)], [PartitionSpec()])

    # sharding purpose. NOTE: replicate the (small) GG conditioners — robust to JAX AbstractMesh
    # during optimizer init (mesh.devices is unavailable on AbstractMesh). Fine at 130m.
    mesh = None
    fsdp_axis_name = hax.partitioning.ResourceAxis.DATA

    sharding_out = [PartitionSpec(None)] * len(p_shape)
    preconditioner_types = _get_preconditioner_types(p_shape, max_precond_dim, prefer_input_side, force_full_rank)
    output = []
    for i in range(len(p_shape)):
        s = p_shape[i]
        if not preconditioner_types[i]:
            output.append(jnp.zeros((s, s), dtype=dtype))
            if mesh is not None:
                if s % fsdp_size == 0:
                    q_sharding = PartitionSpec(fsdp_axis_name, None)
                else:
                    q_sharding = PartitionSpec(None, None)
            else:
                q_sharding = PartitionSpec(None, None)
            sharding_out[i] = q_sharding
        else:
            output.append(None)
    return (output, sharding_out)


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size):
        self._shape = param_shape
        self._shape = tuple(int(_) for _ in self._shape)  # jnp value refuse to be equal to integer, manually convert
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes

        # TODO (evanatyourservice)
        # this might fail with scalar params but for now we're reshaping those
        single_shape = [a[0] for a in split_sizes]
        padded_single_shape = [-(-dim // block_size) * block_size for dim in single_shape]
        stack_size = max(1, np.prod([max(1, len(s)) for s in split_sizes]))
        self._padded_stacked_shape = tuple([stack_size] + padded_single_shape)

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Partition tensor into blocks."""
        assert tensor.shape == self._shape
        tensors = [tensor]
        for i, indices in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tuple(tensors)

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for i, indices in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(jnp.concatenate(partitions[ind : ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def _partitions(lst):
    """Generate all partitions of a list."""
    if not lst:
        yield [[]]
    else:
        for i in range(len(lst)):
            for part in _partitions(lst[i + 1 :]):
                yield [lst[: i + 1]] + part


def _pad_and_stack_matrices(array_list, block_size):
    # Handle scalar arrays by adding a dummy dimension
    is_scalar = len(array_list[0].shape) == 0
    if is_scalar:
        array_list = [arr[None] for arr in array_list]

    shapes = [arr.shape for arr in array_list]
    max_dims = [max(shape[i] for shape in shapes) for i in range(len(shapes[0]))]
    padded_shape = [-(-dim // block_size) * block_size for dim in max_dims]
    padded_arrays = []
    for arr in array_list:
        pad_width = [(0, padded_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        padded = jnp.pad(arr, pad_width)
        padded_arrays.append(padded)

    stacked = jnp.stack(padded_arrays)
    return stacked


def _unstack_and_unpad_matrices(stacked_array, shapes):
    # Handle scalar arrays
    is_scalar = len(shapes[0]) == 0

    unstacked = jnp.split(stacked_array, stacked_array.shape[0], axis=0)
    unpadded = []
    for arr, orig_shape in zip(unstacked, shapes):
        arr = jnp.squeeze(arr, axis=0)
        if is_scalar:
            # For scalars, just take the first element
            arr = arr[0]
        else:
            # For non-scalars, slice to original shape
            slices = tuple(slice(0, dim) for dim in orig_shape)
            arr = arr[slices]
        unpadded.append(arr)
    return tuple(unpadded)


# unused fns (can be used for stacking partitions without padding):
def _sort_and_group_matrices(matrix_shapes: List[Tuple[int, ...]]):
    indexed_list = list(enumerate(matrix_shapes))
    sorted_indexed = sorted(indexed_list, key=lambda x: x[1])
    sorted_shapes = [shape for _, shape in sorted_indexed]
    change_indices = [original_index for original_index, _ in sorted_indexed]
    revert_indices = [0] * len(matrix_shapes)
    for new_pos, (original_index, _) in enumerate(sorted_indexed):
        revert_indices[original_index] = new_pos
    shape_groups = defaultdict(list)
    for i, shape in enumerate(sorted_shapes):
        shape_groups[shape].append(i)
    unique_sorted_shapes = list(shape_groups.keys())
    return unique_sorted_shapes, dict(shape_groups), change_indices, revert_indices


def _stack_matrices(array_list):
    in_tuple = isinstance(array_list, tuple)
    shapes = [arr.shape for arr in array_list]
    unique_shapes, shape_groups, change_indices, _ = _sort_and_group_matrices(shapes)
    sorted_arrays = [array_list[i] for i in change_indices]
    stacked_arrays = []
    for shape in unique_shapes:
        indices = shape_groups[shape]
        stacked = jnp.stack([sorted_arrays[i] for i in indices])
        stacked_arrays.append(stacked)
    if in_tuple:
        return tuple(stacked_arrays)
    return stacked_arrays


def _unstack_matrices(stacked_arrays, revert_indices):
    in_tuple = isinstance(stacked_arrays, tuple)
    unstacked = []
    for arr in stacked_arrays:
        unstacked.extend(jnp.split(arr, arr.shape[0]))
    array_list = [jnp.squeeze(unstacked[i], axis=0) for i in revert_indices]
    if in_tuple:
        return tuple(array_list)
    return array_list


def _merge_small_dims(
    shape_to_merge, max_dim
) -> Tuple[List[int], List[bool], Optional[Tuple]] | Tuple[List[int], List[bool]] | List[int]:
    if not shape_to_merge:  # handles scalar shape ()
        return [], [True]
    if np.all(np.array(shape_to_merge) == 1):  # handles shape (1,)
        return (
            [1],
            [True],
        )

    def dim2loss(d, dim0=max_dim):
        """A heuristic map from dim to loss with the least loss occurs at dim0."""
        loss = 0
        if d < dim0:
            loss += np.log2(dim0 / d)
            too_small = dim0 / 8
            if d < too_small:
                loss += 100 * np.log2(too_small / d)
        else:
            loss += 10 * np.log2(d / dim0)
            too_large = 8 * dim0
            if d > too_large:
                loss += 1000 * np.log2(d / too_large)
        return loss

    best_loss = float("inf")
    best_partition = []

    for p in _partitions(list(range(len(shape_to_merge)))):
        loss = 0
        merged = []
        for group in p:
            if not group:
                continue
            d = np.prod([shape_to_merge[i] for i in group])
            loss += dim2loss(d)
            merged.append(group)

        if loss < best_loss:
            best_loss = loss
            best_partition = merged

    merged_shape = []
    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))

    return merged_shape
