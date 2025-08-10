from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import List, Optional, Tuple, Union
import dataclasses
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import optax.tree_utils as otu
import chex
from chex import Numeric
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


@OptimizerConfig.register_subclass("shampoo3")
@dataclass
class Shampoo3Config(OptimizerConfig):
    weight_decay: float = 0.0
    beta1: float = 0.95
    momentum: float = 0.95
    shampoo_beta: float = 0.95
    beta2: float = 0.95
    adam_beta: float = 0.95
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
    grafting: bool = True
    

    def build(self, num_train_steps):
        """Creates the optimizer"""
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def optimizer(learning_rate, adam_lr):

            def muon_transform():
                components = []
                components.append(scale_by_shampoo3(
                    b1=self.momentum,
                    b2=self.shampoo_beta,
                    b_adam = self.adam_beta,
                    steps=self.steps,
                    epsilon=self.epsilon,
                    max_precond_dim=self.max_precond_dim,
                    merge_small_dims=self.merge_small_dims,
                    target_merged_dim_size=self.target_merged_dim_size,
                    mu_dtype=self.mu_dtype,
                    precond_dtype=self.precond_dtype,
                    partition_grads_into_blocks=self.partition_grads_into_blocks,
                    block_size=self.block_size,
                    grafting=self.grafting
                ))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'muon' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                # muon for linear layers
                return dataclasses.replace(param, weight="muon", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


def _safe_sharding_constraint(x, sharding):
    if sharding is None:
        return x
    else:
        return with_sharding_constraint(x, sharding)


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


def scale_by_shampoo3(
    b1: float = 0.95,
    b2: float = 0.95,
    b_adam: float = 0.95,
    steps: int = 5,
    epsilon: float = 1e-8,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    partition_grads_into_blocks: Optional[bool] = True,
    block_size: Optional[int] = 256,
    lax_map_scanned_layers: Optional[bool] = True,
    lax_map_batch_size: Optional[int] = 4,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    grafting: bool = True,
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
        exp_avg_sq = otu.tree_zeros_like(params, dtype=mu_dtype)
        null_dims = jax.tree.map(
            lambda p, s: _get_preconditioner_types(p.shape[int(s) :], max_precond_dim),
            params,
            scanned_layers_,
        )
        null_dims_leaf = [
            _get_preconditioner_types(p.shape[int(s) :], max_precond_dim)
            for p, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]
        scanned_dim_sharding = [
            PartitionSpec(sh[0]) if s else None
            for sh, s in zip(jax.tree.leaves(params_sharding_), jax.tree.leaves(scanned_layers_))
        ]

        merged_shapes = shapes
        merged_shapes_leaf = shapes_leaf
        if merge_small_dims:
            output = jax.tree.map(
                lambda p, s, dd: _merge_small_dims(p.shape[int(s) :], target_merged_dim_size, dd),
                params,
                scanned_layers_,
                null_dims,
            )
            merged_shapes, null_dims = [jax.tree.map(lambda _, x: x[i], params, output) for i in range(2)]
            merged_shapes_leaf = [
                _merge_small_dims(p.shape[int(s) :], target_merged_dim_size, dd)[0]
                for p, s, dd in zip(
                    jax.tree.leaves(params),
                    jax.tree.leaves(scanned_layers_),
                    null_dims_leaf,
                )
            ]
            print("Shapes after merge small dims: ", merged_shapes_leaf)
            print("Null dims: ", jax.tree.leaves(null_dims))
        partitioned_shapes = merged_shapes
        partitioned_shapes_leaf = merged_shapes_leaf
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                params,
                partitioned_shapes,
                null_dims,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners)
            partitioned_shapes_leaf = [p_cls._padded_stacked_shape for p_cls in jax.tree.leaves(partitioners)]

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
            init_conditioner(t[1:] if partition_grads_into_blocks else t, max_precond_dim, precond_dtype)
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

        return {
            "count": jnp.zeros([], jnp.int32),
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "GG": GG,
        }


    def update_step(updates: Updates, state: dict, scanned_layers_: Updates) -> tuple[Updates, dict]:
        # Update moments
        _, grads_structure = jax.tree.flatten(updates, is_leaf=lambda x: isinstance(x, jax.Array))

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg = jax.tree.map(lambda m, g: None if g is None else b1 * m + (1 - b1) * g, exp_avg, updates)
        exp_avg_sq = jax.tree.map(lambda m, g: None if g is None else b_adam * m + (1 - b_adam) * g * g, exp_avg_sq, updates)
        shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], updates, scanned_layers_)
        # block gradients, exp_avg
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        null_dims = jax.tree.map(
            lambda p, s: _get_preconditioner_types(p.shape[int(s) :], max_precond_dim),
            updates,
            scanned_layers_,
        )
        # merge small dims
        merged_shapes = shapes
        if merge_small_dims:
            original_shapes = shapes
            output = jax.tree.map(
                lambda g, dd, s: _merge_small_dims(g.shape[int(s) :], target_merged_dim_size, dd),
                updates,
                null_dims,
                scanned_layers_,
            )
            merged_shapes, null_dims = [jax.tree.map(lambda _, x: x[i], updates, output) for i in range(2)]
            # reshape
            updates = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                updates,
                scanned_layers_,
                merged_shapes,
            )
            exp_avg = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                exp_avg,
                scanned_layers_,
                merged_shapes,
            )
            exp_avg_sq = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                exp_avg_sq,
                scanned_layers_,
                merged_shapes,
            )

        # partition
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            null_dims = jax.tree.map(
                lambda p, s: _get_preconditioner_types(p.shape[int(s) :], max_precond_dim),
                updates,
                scanned_layers_,
            )
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                updates,
                partitioned_shapes,
                null_dims,
            )
            blocked_exp_avg = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                exp_avg,
                partitioners,
                scanned_layers_,
            )
            blocked_exp_avg_sq = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                exp_avg_sq,
                partitioners,
                scanned_layers_,
            )
            blocked_updates = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                updates,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                blocked_exp_avg,
                scanned_layers_,
            )
            blocked_exp_avg = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_exp_avg,
                scanned_layers_,
            )
            blocked_exp_avg_sq = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_exp_avg_sq,
                scanned_layers_,
            )
            blocked_updates = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_updates,
                scanned_layers_,
            )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        else:
            blocked_exp_avg = exp_avg
            blocked_exp_avg_sq = exp_avg_sq
            blocked_updates = updates

        # Project gradients
        # TBD: bias correction for b1 and b2, leave it for now
        new_GG = jax.tree.map(
            lambda nm, grad, gg: _map_fn(False, 0, nm, partial(update_preconditioner, beta=b2), grad, gg),
            jax.tree.leaves(n_dims_to_map),
            jax.tree.leaves(blocked_updates),
            state["GG"],
        )
        blocked_norm_updates_leaves = jax.tree.map(
            lambda _, nm, e, gg: _map_fn(False, 0, nm, partial(shampoo_update, precision=precision, steps=steps, eps=epsilon), e, gg),
            jax.tree.leaves(dummy_updates_tree),
            jax.tree.leaves(n_dims_to_map),
            jax.tree.leaves(blocked_exp_avg),
            new_GG,
        )
        blocked_adam_updates_leaves = jax.tree.map(
            lambda _, nm, e, v: _map_fn(False, 0, nm, lambda e, v: e / (jnp.sqrt(v) + epsilon), e, v),
            jax.tree.leaves(dummy_updates_tree),
            jax.tree.leaves(n_dims_to_map),
            jax.tree.leaves(blocked_exp_avg),
            jax.tree.leaves(blocked_exp_avg_sq),
        )

        blocked_norm_updates_leaves = jax.tree.map(lambda g: g * jnp.sqrt(1 - b2 ** state["count"]) / ((1 - b1 ** state["count"])) , blocked_norm_updates_leaves)
        blocked_adam_updates_leaves = jax.tree.map(lambda g: g * jnp.sqrt(1 - b_adam ** state["count"]) / ((1 - b1 ** state["count"])) , blocked_adam_updates_leaves)


        blocked_norm_updates_leaves_norm = jax.tree.map(lambda g: jnp.linalg.norm(g) if g is not None else None, blocked_norm_updates_leaves)
        blocked_adam_updates_leaves_norm = jax.tree.map(lambda g: jnp.linalg.norm(g) if g is not None else None, blocked_adam_updates_leaves)
        # jax.debug.print("blocked_norm_updates_leaves_norm: {}", blocked_norm_updates_leaves_norm)
        # jax.debug.print("blocked_adam_updates_leaves_norm: {}", blocked_adam_updates_leaves_norm)


        # grafting, use adam's norm and shampoo's update
        if grafting:
            blocked_norm_updates_leaves = jax.tree.map(
                lambda _, nm, u_1, u_2: _map_fn(False, 0, nm, lambda u_1, u_2: jnp.linalg.norm(u_2) / jnp.linalg.norm(u_1) * u_1, u_1, u_2),
                jax.tree.leaves(dummy_updates_tree),
                jax.tree.leaves(n_dims_to_map),
                jax.tree.leaves(blocked_norm_updates_leaves),
                jax.tree.leaves(blocked_adam_updates_leaves),
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
            exp_avg = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                blocked_exp_avg,
                scanned_layers_,
                partitioned_shapes,
            )
            exp_avg = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                exp_avg,
                scanned_layers_,
                partitioners,
            )
            exp_avg_sq = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                blocked_exp_avg_sq,
                scanned_layers_,
                partitioners,
            )
            exp_avg_sq = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                exp_avg_sq,
                scanned_layers_,
                partitioners,
            )
        else:
            norm_updates = blocked_norm_updates
            exp_avg = blocked_exp_avg
            exp_avg_sq = blocked_exp_avg_sq

        # unmerge
        if merge_small_dims:
            norm_updates = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                norm_updates,
                scanned_layers_,
                original_shapes,
            )
            exp_avg = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                exp_avg,
                scanned_layers_,
                original_shapes,
            )
            exp_avg_sq = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                exp_avg_sq,
                scanned_layers_,
                original_shapes,
            )

        # precision
        new_GG = otu.tree_cast(new_GG, precond_dtype)
        exp_avg = otu.tree_cast(exp_avg, mu_dtype)
        exp_avg_sq = otu.tree_cast(exp_avg_sq, mu_dtype)
        new_state = {
            "GG": new_GG,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "count": state["count"],
        }

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
        return [(1 - beta) * jnp.matmul(grad[:, None], grad[None, :], precision=precision) + beta * GG[0]]  # type: ignore

    def update_gg(idx, gg):
        if gg is None:
            return None
        outer_product = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        return (1 - beta) * outer_product + beta * gg

    new_GG = jax.tree.map(update_gg, list(range(len(GG))), GG)

    return new_GG




def shampoo_update(
    X: Array,
    GG: List[Union[Array, None]],
    steps: int = 5,
    eps: float = 1e-7,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    idx = 0
    for i, mat in enumerate(GG):
        if mat is not None:  # noqa: SIM108
            idx = i
    P = GG[idx] + eps * jnp.eye(GG[idx].shape[0])
    chex.assert_rank(X, 2)
    # a, b, c = (3.4445, -4.7750, 2.0315)
    a, b, c = (2.0, -1.5, 0.5)
    transpose = False
    # if X.shape[0] > X.shape[1]:
    #     X = X.T
    #     transpose = True
    if idx == 1:
        X = X.T
        transpose = True
    P_over_X_factor = 2.0
    X = X / P_over_X_factor
    P = P - X @ X.T
    normalized_factor = (jnp.linalg.norm(X) + eps + jnp.sqrt(jnp.trace(P)))
    X /= normalized_factor # Ensure top singular value <= 1
    P /= (normalized_factor ** 2)
    for i in range(steps):
        A = X @ X.T + P
        B = b * A + c * A @ A
        X = a * X + B @ X
        P = a * a * P + a * (B @ P + P @ B) + B @ P @ B
    if transpose:
        X = X.T
    scale = jnp.sqrt(jnp.maximum(1, X.shape[0] / X.shape[1]))
    return X * scale * P_over_X_factor







def _get_preconditioner_types(shape: Tuple[int, ...], max_precond_dim: int) -> List[bool]:
    if len(shape) == 0:
        return [False]

    if len(shape) == 1:
        return [False]

    min_dim = min(shape)
    new_result = []
    flag = True
    for i in range(len(shape)):
        if shape[i] == min_dim and flag:
            flag = False
            new_result.append(False)
        else:
            new_result.append(True)
    return new_result


def infer_conditioner_sharding(p_shape, max_precond_dim: int):
    if len(p_shape) == 1:
        return [PartitionSpec()]

    # sharding purpose
    mesh = hax.partitioning._get_mesh()
    if mesh.devices.shape == ():
        mesh = None
    # get fsdp mesh axis
    if mesh is not None:
        fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
        fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
        fsdp_size = mesh.devices.shape[fsdp_axis]

    sharding_out = [PartitionSpec(None)] * len(p_shape)
    preconditioner_types = _get_preconditioner_types(p_shape, max_precond_dim)
    for i in range(len(preconditioner_types)):
        s = p_shape[i]
        if not preconditioner_types[i]:
            if mesh is not None:
                if s % fsdp_size == 0:
                    q_sharding = PartitionSpec(fsdp_axis_name, None)
                else:
                    q_sharding = PartitionSpec(None, None)
            else:
                q_sharding = PartitionSpec(None, None)
            sharding_out[i] = q_sharding
    return sharding_out


def init_conditioner(p_shape, max_precond_dim: int, dtype: Optional[Union[str, jnp.dtype]]):
    if len(p_shape) == 1:
        return ([jnp.zeros((p_shape[0], p_shape[0]), dtype=dtype)], [PartitionSpec()])

    # sharding purpose
    mesh = hax.partitioning._get_mesh()
    if mesh.devices.shape == ():
        mesh = None
    # get fsdp mesh axis
    if mesh is not None:
        fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
        fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
        fsdp_size = mesh.devices.shape[fsdp_axis]

    sharding_out = [PartitionSpec(None)] * len(p_shape)
    preconditioner_types = _get_preconditioner_types(p_shape, max_precond_dim)
    print("Preconditioner types: ", preconditioner_types)
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

    def __init__(self, param_shape, block_size, null_dims):
        self._shape = param_shape
        self._shape = tuple(int(_) for _ in self._shape)  # jnp value refuse to be equal to integer, manually convert
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d and not null_dims[i]:
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
    shape_to_merge, max_dim, null_dims
) -> Tuple[List[int], List[bool], Optional[Tuple]] | Tuple[List[int], List[bool]]:
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
    merged_diag = []
    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))
        merged_diag.append(all(null_dims[i] for i in group))

    return (
        merged_shape,
        merged_diag,
    )
