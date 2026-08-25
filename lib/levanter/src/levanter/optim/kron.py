# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import string
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union, cast

import chex
import haliax as hax
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import vmap
from jax.sharding import PartitionSpec
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

from levanter.optim._block_partition import (
    BlockPartitioner,
    is_scanned_stack,
    map_fn,
    merge_small_dimensions,
    pad_and_stack_matrices,
    safe_sharding_constraint,
    unstack_and_unpad_matrices,
)
from levanter.optim.config import OptimizerConfig

# Define type variables for the pytree structure
T = TypeVar("T")
PartitionSpecTree = TypeVar(
    "PartitionSpecTree", bound=Union[PartitionSpec, List[PartitionSpec], Tuple[PartitionSpec, ...], dict, list, tuple]
)

# Type for the update probability schedule
UpdateProbSchedule = Union[float, Callable[[int], float]]


@OptimizerConfig.register_subclass("kron")
@dataclass(frozen=True)
class KronConfig(OptimizerConfig, Generic[PartitionSpecTree]):
    """Configuration for PSGD Kron optimizer.

    Attributes:
        beta1: Momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Optional gradient norm clipping value.
        normalize_grads: Whether to normalize the incoming gradients to unit norm layer-wise.
            Can help with stability.
        preconditioner_update_probability: Final probability of updating the preconditioner. Default
            is 0.05 (update every 20 steps). The `precond_update_prob_schedule` holds probability at
            1.0 for `update_prob_flat_start` steps before annealing exponentially down to this
            value within ~3000 steps. Training is slower while updates are done every step, but
            training speeds up after update probability decays.
        update_prob_flat_start: Number of steps to keep update probability at 1.0 before annealing.
            Default value of 500 works well, but increasing this to 1000 or 2000 can benefit training.
            However, this slows down training. A good balance is to keep update probability at 1.0 during
            initial loss drop, then when you notice loss start to plateau, the preconditioner is mostly
            learned and update probability can be decayed for faster training.
        max_size_triangular: Max size for dim's preconditioner to be triangular.
        min_ndim_triangular: Minimum number of dimensions a layer needs to have triangular preconditioners.
        memory_save_mode: Memory saving mode for preconditioners. Options:
            - None: All preconditioners are triangular (default)
            - 'one_diag': Largest/last dim per layer uses diagonal preconditioner
            - 'all_diag': All preconditioners are diagonal
        preconditioner_lr: Learning rate for preconditioner.
        preconditioner_init_scale: Scale for preconditioner initialization.
        mu_dtype: Dtype of the momentum buffer. Defaults to same dtype as parameters.
        precond_dtype: Dtype of the preconditioners. Defaults to 'float32'.
        precond_update_precision: Precision for matmul during preconditioner update.
            Options: 'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: Precision for matmul during preconditioning grads.
            Options: 'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: Tree of booleans same structure as params indicating scanned dimensions
            for each layer. PSGD will vmap over leading dimension.
        lax_map_scanned_layers: Whether to use lax.map for scanned layers instead of vmap.
            Useful to save memory with large models.
        lax_map_batch_size: Batch size for lax.map, see JAX docs for more info.
        merge_small_dims: Whether to merge small dimensions to improve preconditioner efficiency.
        target_merged_dim_size: Target size of merged dimensions.
        partition_grads_into_blocks: Whether to partition grads into chunks of size block_size
            for efficiency.
        block_size: Block size to use for partitioning grads.
        params_sharding: Pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: PartitionSpec for preconditioner matrices. Best practice is to
            shard first dimension across fsdp-like mesh axis, or largest/most common axis in params.
            Example: PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').
    """

    # some of these are changed from kron defaults to better suit levanter
    beta1: float = 0.9
    weight_decay: float = 0.1
    max_grad_norm: Optional[float] = 1.0
    normalize_grads: bool = False
    preconditioner_update_probability: UpdateProbSchedule = 0.05
    update_prob_flat_start: int = 500
    max_size_triangular: int = 25000
    min_ndim_triangular: int = 2
    memory_save_mode: Optional[str] = None
    preconditioner_lr: float = 0.1
    preconditioner_init_scale: float = 1.0
    mu_dtype: Optional[Union[str, jnp.dtype]] = None
    precond_dtype: Optional[Union[str, jnp.dtype]] = None
    precond_update_precision: Optional[str] = "tensorfloat32"
    precond_grads_precision: Optional[str] = None
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8
    merge_small_dims: bool = True
    target_merged_dim_size: int = 8192
    partition_grads_into_blocks: bool = True
    block_size: int = 256
    params_sharding: Optional[PartitionSpecTree] = None
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None

    def build(self, num_train_steps):
        """Creates the optimizer."""

        def _optimizer(learning_rate) -> optax.GradientTransformation:
            precond_partition_spec = (
                PartitionSpec(*self.preconditioner_sharding) if self.preconditioner_sharding is not None else None
            )
            components = []
            if self.max_grad_norm and not self.normalize_grads:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                scale_by_kron(
                    b1=self.beta1,
                    normalize_grads=self.normalize_grads,
                    preconditioner_update_probability=precond_update_prob_schedule(
                        min_prob=self.preconditioner_update_probability,
                        flat_start=self.update_prob_flat_start,
                    ),
                    max_size_triangular=self.max_size_triangular,
                    min_ndim_triangular=self.min_ndim_triangular,
                    memory_save_mode=self.memory_save_mode,
                    preconditioner_lr=self.preconditioner_lr,
                    preconditioner_init_scale=self.preconditioner_init_scale,
                    mu_dtype=self.mu_dtype,
                    precond_dtype=self.precond_dtype,
                    precond_update_precision=self.precond_update_precision,
                    precond_grads_precision=self.precond_grads_precision,
                    lax_map_scanned_layers=self.lax_map_scanned_layers,
                    lax_map_batch_size=self.lax_map_batch_size,
                    merge_small_dims=self.merge_small_dims,
                    target_merged_dim_size=self.target_merged_dim_size,
                    partition_grads_into_blocks=self.partition_grads_into_blocks,
                    block_size=self.block_size,
                    params_sharding=self.params_sharding,
                    preconditioner_sharding=precond_partition_spec,
                )
            )
            components.append(optax.clip_by_block_rms(1.1))
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale_by_learning_rate(learning_rate))
            return optax.chain(*components)

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


"""PSGD Kron"""


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500) -> Callable[[int], float]:
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n: int) -> float:
        """Exponential anneal with flat start."""
        # jnp.clip yields a scalar Array; it is consumed as a float probability.
        return cast(float, jnp.clip(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob, max_prob))

    return _schedule


def scale_by_kron(
    b1: float = 0.9,
    normalize_grads: bool = False,
    preconditioner_update_probability: UpdateProbSchedule = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    lax_map_scanned_layers: bool = True,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 256,
    params_sharding: Optional[PartitionSpecTree] = None,
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None,
    **kwargs,
) -> base.GradientTransformation:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.
    Author: https://github.com/evanatyourservice

    Args:
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformation
    """
    # optax.canonicalize_dtype is stubbed with a broad return; it normalizes to a concrete dtype here.
    mu_dtype = cast(Optional[Union[str, jnp.dtype]], canonicalize_dtype(mu_dtype))
    precond_dtype = cast(Optional[Union[str, jnp.dtype]], canonicalize_dtype(precond_dtype or jnp.float32))
    lax_map = lax_map_scanned_layers
    bs = lax_map_batch_size
    scanned_layers = None

    def add_dims_to_spec(_, qss, sds):
        """Prepend the stack and scan dims to a leaf's Q partition specs."""
        if partition_grads_into_blocks:
            qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
        if sds is not None:
            qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
        return qss

    def init_fn(params, return_partition_specs_only=False):
        # flatten haliax trees so the rest of the algorithm sees plain arrays
        scanned_layers_ = scanned_layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(
                lambda x: (jax.tree.map(lambda _: True, x) if is_scanned_stack(x) else False),
                params,
                is_leaf=is_scanned_stack,
            )
        params_sharding_ = params_sharding
        if any(
            isinstance(x, hax.NamedArray)
            for x in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, hax.NamedArray))
        ):
            # if in haliax, we can grab scanned_layers and params_sharding from params
            # this does not support nested stacks
            if params_sharding_ is None:
                params_sharding_ = hax.partitioning.infer_resource_partitions(params)
                params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
            params = jax.tree.leaves(params)
            scanned_layers_ = jax.tree.leaves(scanned_layers_)
            params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None

        # check that there is a PartitionSpec for every param
        if params_sharding_ is not None:
            assert len(jax.tree.leaves(params_sharding_)) == len(
                jax.tree.leaves(params)
            ), "There must be a PartitionSpec for every parameter in PSGD Kron."
        # check that preconditioner sharding length is at least 1
        if preconditioner_sharding is not None:
            assert len(preconditioner_sharding) > 0, (
                "preconditioner_sharding must have length > 0. For example, "
                "PartitionSpec(None) or PartitionSpec('fsdp', None) are valid."
            )

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda p, sh: PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh)))),
                params,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        params = jax.tree.map(lambda p: p[None] if len(p.shape) == 0 else p, params)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        scanned_sizes = jax.tree.map(lambda p, s: p.shape[0] if s else 0, params, scanned_layers_)

        # momentum
        mu = None
        mu_sharding = params_sharding_
        if b1 > 0 and not return_partition_specs_only:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)
            # apply params sharding to momentum buffer
            if have_params_sharding:
                mu = safe_sharding_constraint(mu, params_sharding_)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda p, s: _get_preconditioner_types(
                p.shape[int(s) :],
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
            ),
            params,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, params)
        merged_shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], params, scanned_layers_)
        if merge_small_dims:
            output = jax.tree.map(
                lambda p, s, dd, sh: merge_small_dimensions(p.shape[int(s) :], target_merged_dim_size, dd, sh),
                params,
                scanned_layers_,
                dim_diag,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]

        # partition grads into blocks
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                params,
                merged_shapes,
                dim_diag,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners)

        # initialize preconditioners
        output = jax.tree.map(
            lambda _, ps, dd, sh: list(
                _init_Q_exprs(
                    ps[1:] if partition_grads_into_blocks else ps,
                    preconditioner_init_scale,
                    dd,
                    precond_dtype,
                    existing_Q=True if return_partition_specs_only else None,
                    precond_sharding=preconditioner_sharding_,
                    param_sharding=sh,
                )
            ),
            params,
            partitioned_shapes,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
        )
        if return_partition_specs_only:
            exprs, Qs_sharding_no_leading_dims = [jax.tree.map(lambda _, x: x[i], params, output) for i in range(2)]
        else:
            Qs, exprs, Qs_sharding_no_leading_dims = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]

        Qs_sharding = jax.tree.map(
            add_dims_to_spec,
            params,
            Qs_sharding_no_leading_dims,
            scanned_dim_sharding if scanned_dim_sharding is not None else nones,
        )

        if not return_partition_specs_only:
            # broadcast Qs for stacks and scans
            def broadcast_qs(_, ps, q, s):
                stack_n = ps[0]
                if partition_grads_into_blocks:
                    # add leading dim for stacked partitions
                    q = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), stack_n, axis=0), q)
                if s > 0:
                    # add leading dim if we're scanning this layer
                    q = jax.tree.map(lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), q)
                return q

            Qs = jax.tree.map(broadcast_qs, params, partitioned_shapes, Qs, scanned_sizes)
            Qs = safe_sharding_constraint(Qs, Qs_sharding)

        if return_partition_specs_only:
            return dict(
                key=PartitionSpec(),
                count=PartitionSpec(),
                mu=mu_sharding,
                Qs_preconditioners=Qs_sharding,
                update_counter=PartitionSpec(),
                balance_counter=PartitionSpec(),
            )

        return dict(
            key=jax.random.PRNGKey(0),
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=jnp.zeros([], jnp.int32),
            balance_counter=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates: base.Updates, state: dict, params: Optional[base.Params] = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key, subkey = jax.random.split(state["key"])

        # flatten haliax trees so the rest of the algorithm sees plain arrays
        scanned_layers_ = scanned_layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(
                lambda x: (jax.tree.map(lambda _: True, x) if is_scanned_stack(x) else False),
                updates,
                is_leaf=is_scanned_stack,
            )
        params_sharding_ = params_sharding
        updates_struct = None
        if any(
            isinstance(x, hax.NamedArray)
            for x in jax.tree.leaves(updates, is_leaf=lambda x: isinstance(x, hax.NamedArray))
        ):
            # if in haliax, we can grab scanned_layers and params_sharding from params
            # this does not support nested stacks
            if params_sharding_ is None:
                params_sharding_ = hax.partitioning.infer_resource_partitions(updates)
                params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
            updates, updates_struct = jax.tree.flatten(updates)
            scanned_layers_ = jax.tree.leaves(scanned_layers_)
            params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None
        if have_params_sharding:
            original_params_sharding_ = params_sharding_

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda g, sh: PartitionSpec(*(sh + (None,) * (len(g.shape) - len(sh)))),
                updates,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        input_shapes = jax.tree.map(lambda g: g.shape, updates)
        updates = jax.tree.map(lambda g: g[None] if len(g.shape) == 0 else g, updates)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        # update probability can be scheduled

        if callable(preconditioner_update_probability):
            update_prob_in = cast(Callable[[int], float], preconditioner_update_probability)(count_inc)
        else:
            update_prob_in = float(preconditioner_update_probability)

        # normalize grads
        def norm_grads(g):
            return g / (jnp.linalg.norm(g) + 1e-16)

        if normalize_grads:
            updates = jax.tree.map(norm_grads, updates)

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            if have_params_sharding:
                mu = safe_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda g, s: _get_preconditioner_types(
                g.shape[int(s) :],
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
            ),
            momentum_updates,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, momentum_updates)
        merged_params_sharding = params_sharding_
        original_shapes = None
        if merge_small_dims:
            original_shapes = jax.tree.map(lambda g, s: g.shape[int(s) :], momentum_updates, scanned_layers_)
            output = jax.tree.map(
                lambda g, dd, s, sh: merge_small_dimensions(g.shape[int(s) :], target_merged_dim_size, dd, sh),
                momentum_updates,
                dim_diag,
                scanned_layers_,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], momentum_updates, output) for i in range(3)
            ]
            # reshape
            momentum_updates = jax.tree.map(
                lambda g, s, ns: map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                momentum_updates,
                scanned_layers_,
                merged_shapes,
            )
            if have_params_sharding:
                # scanned dim sharding + new merged sharding
                merged_params_sharding = jax.tree.map(
                    lambda sws, sds: PartitionSpec(*(sds + sws if sds is not None else sws)),
                    sharding_without_scan,
                    scanned_dim_sharding,
                )
        # constrain sharding
        if have_params_sharding:
            momentum_updates = safe_sharding_constraint(momentum_updates, merged_params_sharding)

        # partition grads into blocks
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        partitioned_sharding = merged_params_sharding
        partitioners = None
        partitioned_shapes = None
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda g, dd, s: BlockPartitioner(g.shape[int(s) :], block_size, dd),
                momentum_updates,
                dim_diag,
                scanned_layers_,
            )
            # layers become tuples each containing layer's partitions
            momentum_updates = jax.tree.map(
                lambda g, p_cls, s: map_fn(False, 0, int(s), p_cls.partition, g),
                momentum_updates,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # constrain partitions to same sharding as entire layer
                momentum_updates = jax.tree.map(
                    lambda _, g, mps: jax.tree.map(lambda x: safe_sharding_constraint(x, mps), g),
                    dummy_updates_tree,
                    momentum_updates,
                    merged_params_sharding,
                )
            # pad and stack partitions, tuples become arrays with new leading dim
            momentum_updates = jax.tree.map(
                lambda _, g, s: map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # add dim to sharding specs for new stacked dim
                partitioned_sharding = jax.tree.map(
                    lambda mps, s: PartitionSpec(*(mps[: int(s)] + (None,) + mps[1:])),
                    merged_params_sharding,
                    scanned_layers_,
                )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        # constrain sharding
        if have_params_sharding:
            momentum_updates = safe_sharding_constraint(momentum_updates, partitioned_sharding)

        # get einsum expressions and Qs sharding
        Qs = state["Qs_preconditioners"]
        Qs_sharding = None
        exprs_and_sharding = jax.tree.map(
            lambda g, dd, sh, nm: _init_Q_exprs(
                g.shape[nm:],
                preconditioner_init_scale,
                dd,
                precond_dtype,
                existing_Q=True,
                precond_sharding=preconditioner_sharding_,
                param_sharding=sh,
            ),
            momentum_updates,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
            n_dims_to_map,
        )
        exprs, Qs_sharding_no_leading_dims = [
            jax.tree.map(lambda _, x: x[i], dummy_updates_tree, exprs_and_sharding) for i in range(2)
        ]

        Qs_sharding = jax.tree.map(
            add_dims_to_spec,
            dummy_updates_tree,
            Qs_sharding_no_leading_dims,
            scanned_dim_sharding if scanned_dim_sharding is not None else nones,
        )

        # maybe update preconditioner
        def update_preconditioner_fn(rngkey, Qs, grads_in, bal_counter):
            with jax.default_matmul_precision(precond_update_precision):
                # balance preconditioners about every 100 updates
                def balance_Qs(Qs_to_bal):
                    def _balance_Q(Q):
                        norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)
                        gmean = jnp.exp(jnp.mean(jnp.log(norms)))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return jax.tree.map(
                        lambda _, Q, nm: map_fn(False, 0, nm, _balance_Q, Q),
                        dummy_updates_tree,
                        Qs_to_bal,
                        n_dims_to_map,
                    )

                balance_counter_inc = safe_int32_increment(bal_counter)
                do_balances = balance_counter_inc >= 100
                balance_counter_inc = jnp.where(do_balances, 0, balance_counter_inc)
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)
                Qs = safe_sharding_constraint(Qs, Qs_sharding)

                # create random vectors
                Vs = _tree_random_like(rngkey, grads_in)
                # apply params sharding to random vectors
                if have_params_sharding:
                    Vs = safe_sharding_constraint(Vs, partitioned_sharding)

                # damp based on machine precision
                damp_eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)  # bf16 eps too large
                grads_in = jax.tree.map(
                    lambda g, v: g + damp_eps.astype(g.dtype) * jnp.mean(jnp.abs(g)) * v,
                    grads_in,
                    Vs,
                )

                # form conjB
                conjBs = jax.tree.map(
                    lambda g, Q, v, nm: map_fn(lax_map, bs, nm, _conjB, Q, g, v),
                    grads_in,
                    Qs,
                    Vs,
                    n_dims_to_map,
                )
                if have_params_sharding:
                    conjBs = safe_sharding_constraint(conjBs, partitioned_sharding)

                # update Qs and constrain sharding
                new_Qs = jax.tree.map(
                    lambda g, Q, conjb, expr, nm, qss, sh: map_fn(
                        lax_map,
                        bs,
                        nm,
                        partial(
                            _update_precond,
                            exprs=expr,
                            precond_lr=preconditioner_lr,
                            qs_sharding=qss,
                            params_sharding=sh,
                        ),
                        Q,
                        g,
                        conjb,
                    ),
                    grads_in,
                    Qs,
                    conjBs,
                    exprs,
                    n_dims_to_map,
                    Qs_sharding_no_leading_dims,
                    sharding_without_scan if have_params_sharding else nones,
                )
                new_Qs = safe_sharding_constraint(new_Qs, Qs_sharding)

                new_Qs = otu.tree_cast(new_Qs, precond_dtype)
                return new_Qs, balance_counter_inc

        def pass_through_fn(rngkey, qs, grads_in, bal_counter):
            return safe_sharding_constraint(qs, Qs_sharding), bal_counter

        # update preconditioner deterministically
        update_counter_inc = safe_int32_increment(state["update_counter"])
        do_update = update_counter_inc >= 1 / update_prob_in
        update_counter_inc = jnp.where(do_update, 0, update_counter_inc)
        Qs, balance_counter_inc = jax.lax.cond(
            do_update,
            update_preconditioner_fn,
            pass_through_fn,
            subkey,
            Qs,
            momentum_updates,
            state["balance_counter"],
        )
        Qs = safe_sharding_constraint(Qs, Qs_sharding)

        # precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            precond_gs = jax.tree.map(
                lambda g, Q, expr, nm: map_fn(lax_map, bs, nm, partial(_precond_grad, exprs=expr), Q, g),
                momentum_updates,
                Qs,
                exprs,
                n_dims_to_map,
            )
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, partitioned_sharding)

        # unpartition grads
        if partition_grads_into_blocks:
            precond_gs = jax.tree.map(
                lambda g, s, ps: map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                precond_gs,
                scanned_layers_,
                partitioned_shapes,
            )
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, merged_params_sharding)
            precond_gs = jax.tree.map(
                lambda _, g, s, p_cls: map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                precond_gs,
                scanned_layers_,
                partitioners,
            )
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, merged_params_sharding)

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = jax.tree.map(
                lambda g, s, os: map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                precond_gs,
                scanned_layers_,
                original_shapes,
            )
            if have_params_sharding:
                precond_gs = safe_sharding_constraint(precond_gs, params_sharding_)

        # return scalars to original shape
        precond_gs = jax.tree.map(lambda g, s: jnp.reshape(g, s), precond_gs, input_shapes)

        # final constraint for good measure
        if have_params_sharding:
            precond_gs = safe_sharding_constraint(precond_gs, original_params_sharding_)

        # rebuild the haliax tree we flattened on the way in
        if updates_struct is not None:
            precond_gs = updates_struct.unflatten(precond_gs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(
            key=key,
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=update_counter_inc,
            balance_counter=balance_counter_inc,
        )

        return precond_gs, state

    return base.GradientTransformation(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    preconditioner_update_probability: UpdateProbSchedule = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 256,
    params_sharding: Optional[PartitionSpecTree] = None,
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None,
) -> base.GradientTransformation:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate schedule.
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: float, weight decay coefficient.
        weight_decay_mask: optional pytree same structure as params, or callable
            returning a pytree, that masks weight decay. Weight decay is applied to
            leaves that are True.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformation
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
            normalize_grads=normalize_grads,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            merge_small_dims=merge_small_dims,
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_size: int, min_ndim: int, mem_save_mode: Optional[str]
) -> List[bool]:
    if len(shape) == 0:
        return [True]

    if mem_save_mode is None:
        dim_diag = [False for _ in shape]
    elif mem_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        dim_diag[rev_sorted_dims[0]] = True
    elif mem_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(f"Invalid mem_save_mode: {mem_save_mode}, must be one of [None, 'one_diag', 'all_diag']")

    for i, size in enumerate(shape):
        if size == 1 or size > max_size or len(shape) < min_ndim:
            dim_diag[i] = True

    return dim_diag


def _init_Q_exprs(
    t_shape,
    scale,
    dim_diag,
    dtype,
    existing_Q=None,
    precond_sharding=None,
    param_sharding=None,
):
    have_qs_sharding = precond_sharding is not None or param_sharding is not None
    letters = string.ascii_lowercase + string.ascii_uppercase
    if len(t_shape) == 0:  # scalar
        Q = [scale * jnp.ones(t_shape, dtype=dtype)] if existing_Q is None else existing_Q
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"

        sharding_out = [None]
        if have_qs_sharding:
            sharding_out = [PartitionSpec()]
    else:  # tensor
        if len(t_shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t_shape.shape)}; Einstein runs out of letters!")
        scale = scale ** (1 / len(t_shape))
        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

        sharding_out: list[PartitionSpec | None] = [None] * len(t_shape)
        if have_qs_sharding:
            sharding_out = [PartitionSpec(None)] * len(t_shape)

        for i, (size, dim_d) in enumerate(zip(t_shape, dim_diag)):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    q = scale * jnp.ones(size, dtype=dtype)
                    Q.append(q)

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(t_shape))])
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                q_sharding = None
                if have_qs_sharding:
                    # grab the fsdp axis from the current mesh and shard Q accordingly
                    mesh = jax._src.mesh.get_concrete_mesh()
                    if mesh.devices.shape == ():
                        mesh = None
                    if mesh is not None:
                        fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
                        fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
                        fsdp_size = mesh.devices.shape[fsdp_axis]
                        if size % fsdp_size == 0:
                            q_sharding = PartitionSpec(fsdp_axis_name, None)
                        else:
                            q_sharding = PartitionSpec(None, None)
                    else:
                        q_sharding = PartitionSpec(None, None)
                    sharding_out[i] = q_sharding

                if existing_Q is None:
                    q = scale * jnp.eye(size, dtype=dtype)
                    if have_qs_sharding:
                        q = safe_sharding_constraint(q, q_sharding)
                    Q.append(q)

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(t_shape))])
                piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(t_shape))])
                exprGs.append(piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26])

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return (exprA, exprGs, exprP), sharding_out
    return Q, (exprA, exprGs, exprP), sharding_out


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and
    sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A). Looks to be a very
    tight lower bound.

    A is hermitian so we can always use dim 0 and not have to compare to dim 1.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        aa = A * A
        aa_sum0 = jnp.sum(aa, axis=0)
        i = jnp.argmax(aa_sum0, 0)
        x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
        x = x @ A
        return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A.T)

    return jnp.where(max_abs > 0, calc(A), max_abs)


def _solve_triangular_right(X, A):
    """Compute X @ inv(A).

    A triangular solve has roughly the same complexity as a matmul.
    """
    X_ndim = X.ndim
    if X_ndim < 2:
        X = X[None, :]

    dtype_in = jnp.promote_types(A.dtype, X.dtype)
    A, X = A.astype(dtype_in), X.astype(dtype_in)
    leading_dims = 0
    if X.ndim > 2:
        leading_dims = X.ndim - 2
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=False, lower=False)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    solution = solve_fn(A, X)

    if X_ndim < 2:
        return solution[0]
    return solution


def _conjB(Q, G, V):
    """Compute conjB."""
    order = G.ndim
    p = list(range(order))
    conjB = jnp.transpose(V, p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.ndim < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = jnp.swapaxes(conjB, i, order - 1)
    return conjB


def _update_precond(Q, G, conjB, exprs, precond_lr, qs_sharding, params_sharding):
    """Compute A and update Q."""
    exprA, exprGs, _ = exprs

    A = jnp.einsum(exprA, *Q, G)

    def _update_single_q(i, q):
        term1 = jnp.einsum(exprGs[i], A, A)
        term2 = jnp.einsum(exprGs[i], conjB, conjB)

        if q.ndim < 2:
            q -= precond_lr / _add_tiny(jnp.max(jnp.abs(term1 + term2))) * (term1 - term2) * q
        else:
            if qs_sharding is not None:
                sharding = qs_sharding[i]
                # transpose q sharding for terms
                if len(sharding) < 2:
                    sharding = PartitionSpec(*((None,) + sharding))
                else:
                    assert len(sharding) == 2
                    sharding = PartitionSpec(*(sharding[1:] + sharding[:1]))
                term1 = safe_sharding_constraint(term1, sharding)
                term2 = safe_sharding_constraint(term2, sharding)
            q -= precond_lr / _add_tiny(_norm_lower_bound(term1 + term2)) * jnp.triu(term1 - term2) @ q
        return q

    return [_update_single_q(i, q) for i, q in enumerate(Q)]


def _precond_grad(Q, G, exprs):
    """Precondition gradient G with preconditioner Q."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *Q, *Q, G)


def _add_tiny(x):
    return x + jnp.finfo(x.dtype).tiny


def _tree_random_like(rng_key: chex.PRNGKey, target_tree: chex.ArrayTree, dtype=None) -> chex.ArrayTree:
    # adopted from optax
    tree_def = jax.tree.structure(target_tree)
    keys = jax.random.split(rng_key, tree_def.num_leaves)
    keys_tree = jax.tree.unflatten(tree_def, keys)
    return jax.tree.map(
        lambda target_array, key: jax.random.normal(
            key, target_array.shape, dtype if dtype is not None else target_array.dtype
        ),
        target_tree,
        keys_tree,
    )
