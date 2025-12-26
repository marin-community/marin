# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import atexit
import copy
import functools
import logging as pylogging
import os
import sys
import time
import typing
import warnings
import dataclasses
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import equinox as eqx
import fsspec
import haliax as hax
import haliax.tree_util
import jax
import jax.debug as debug
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import wandb
from draccus import field
from haliax import Axis
from haliax.partitioning import ResourceMapping, named_jit
from haliax.quantization import QuantizationConfig, apply_updates, partition_for_grad_overwrite
from haliax.types import Scalar
from jax._src import mesh as mesh_lib
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation
from tqdm import tqdm

import levanter.callbacks._metrics
import levanter.checkpoint
import levanter.tracker
import levanter.tracker.wandb
import levanter.utils.logging
from levanter.callbacks import Callback, CBInfo, JitCallback, LambdaCallback, StepInfo
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, is_checkpoint_path, load_checkpoint_or_initialize
from levanter.config import JsonAtom
from levanter.data import AsyncDataset, DataLoader
from levanter.data.loader import _round_to_nearest_multiple
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grad_accum import microbatched
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import compute_next_token_loss
from levanter.metrics import Metric, auto_metric_from_name, unwrap_metrics
from levanter.optim.model_averaging import ModelAveragingConfig
from levanter.schedule import BatchSchedule, IntSchedule, ScheduleStep, distinct_values, value_at_step
from levanter.tracker import TrackerConfig, capture_time
from levanter.tracker.wandb import WandbConfig
from levanter.trainer_state import InsideJitInfo, TrainerState, saveable_training_mask, trainables_only
from levanter.utils import cloud_utils, fsspec_utils
from levanter.utils.jax_utils import barrier_sync, best_effort_sharding, zeros_like_tree
from levanter.utils.mesh import MeshConfig, create_mesh_from_axis_specs
from levanter.utils.tree_utils import inference_mode, tree_statistics
from levanter.utils.types import ComputeLossFunction, FilterSpec, FilterTree

logger = pylogging.getLogger(__name__)

X = TypeVar("X")  # Input
M = TypeVar("M")  # Model
S = TypeVar("S", bound=TrainerState)  # State

DEFAULT_JAX_CONFIG: Dict[str, JsonAtom] = {
    "jax_threefry_partitionable": True,
    "jax_softmax_custom_jvp": True,
    "jax_explain_cache_misses": True,
}
from jax.tree_util import tree_map_with_path
import haliax as hax

def _is_leaf(x):
    # Don't descend into NamedArray internals; also treat raw JAX arrays as leaves.
    return isinstance(x, hax.NamedArray) or (hasattr(x, "shape") and hasattr(x, "dtype"))

def _path_to_str(path):
    parts = []
    for k in path:
        if hasattr(k, "name"):
            parts.append(k.name)       # GetAttrKey
        elif hasattr(k, "key"):
            parts.append(str(k.key))   # DictKey
        elif hasattr(k, "idx"):
            parts.append(str(k.idx))   # SequenceKey / FlattenedIndexKey
        else:
            parts.append(str(k))
    return ".".join(parts)

def list_weight_paths(tree, limit=30):
    paths = []
    def _collect(path, x):
        name = _path_to_str(path)
        if "weight" in name:          # loose and simple
            paths.append(name)
    tree_map_with_path(_collect, tree, is_leaf=_is_leaf)
    for p in paths[:limit]:
        print(p)
    print(f"... ({len(paths)} total)")

def print_one_sharding(tree, match: str):
    found = False
    def _visit(path, x):
        nonlocal found
        if found: return
        name = _path_to_str(path)
        if match in name:
            arr = x.array if hasattr(x, "array") else x
            sh  = getattr(arr, "sharding", None)
            spec = getattr(sh, "spec", None)
            print(f"[Shard] {name}\n        shape={getattr(arr,'shape',None)} sharding={sh} pspec={spec}")
            found = True
    tree_map_with_path(_visit, tree, is_leaf=_is_leaf)
    if not found:
        print(f"[Shard] no match for '{match}'")

def log_layer_sharding(label: str, tree, match: str):
    if jax.process_index() == 0:
        print(f"[Shard] {label} -> {match}")
        print_one_sharding(tree, match)

# Representative transformer linear weights we try to match for sharding debug prints
TARGET_MATCHES = (
    "attn.c_attn.weight",
    "mlp.c_fc.weight",
    "attn.c_proj.weight",
    "embeddings.token_embeddings.weight",
)

def _select_weight_array(tree):
    selected = {"arr": None, "name": None, "axes": None, "named_sharding": None}

    def _visit(path, x):
        if selected["arr"] is not None:
            return
        name = _path_to_str(path)
        for m in TARGET_MATCHES:
            if m in name:
                arr = x.array if isinstance(x, hax.NamedArray) else x
                if hasattr(arr, "shape") and hasattr(arr, "dtype"):
                    selected["arr"] = arr
                    selected["name"] = name
                    if isinstance(x, hax.NamedArray):
                        try:
                            selected["axes"] = [ax.name for ax in x.axes]
                        except Exception:
                            selected["axes"] = None
                        try:
                            selected["named_sharding"] = getattr(x, "sharding", None)
                        except Exception:
                            selected["named_sharding"] = None
                    break

    tree_map_with_path(_visit, tree, is_leaf=_is_leaf)
    return selected["name"], selected["arr"], selected["axes"], selected["named_sharding"]

def _debug_weight_sharding(label, tree):
    name, arr, axes_names, named_sharding = _select_weight_array(tree)
    if arr is None:
        debug.print("[ShardDBG] {}: no match in tree", label)
    else:
        shape = getattr(arr, "shape", None)

        def _cb(sh):
            try:
                spec = getattr(sh, "spec", None)
            except Exception:
                spec = None
            try:
                n_spec = getattr(named_sharding, "spec", None)
            except Exception:
                n_spec = None
            print(f"\n[ShardDBG] {label}: {name}\n\tshape={shape}\n\taxes={axes_names}\n\tnamed_sharding={named_sharding}\n\tnamed_pspec={n_spec}\n\tarray_sharding={sh}\n\tarray_pspec={spec}", flush=True)

    debug.inspect_array_sharding(arr, callback=_cb)

def _debug_array_sharding(label, arr):
    try:
        shape = getattr(arr, "shape", None)
        def _cb(sh):
            try:
                spec = getattr(sh, "spec", None)
            except Exception:
                spec = None
            print(f"\n[ShardDBG] {label}: shape={shape}\n\tarray_sharding={sh}\n\tarray_pspec={spec}", flush=True)
        debug.inspect_array_sharding(arr, callback=_cb)
    except Exception:
        pass

def _constrain_array_to_fsdp(x, mesh):
    """Attach NamedSharding derived from best_effort_sharding to a raw JAX array."""
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        try:
            sh = best_effort_sharding(x, mesh=mesh)
            return jax.lax.with_sharding_constraint(x, sh)
        except Exception:
            return x
    return x

def _constrain_named_or_array(x, mesh):
    """Same as above, but preserves/rewraps Haliax NamedArray leaves."""
    if isinstance(x, hax.NamedArray):
        arr = _constrain_array_to_fsdp(x.array, mesh)
        return hax.named(arr, x.axes)
    return _constrain_array_to_fsdp(x, mesh)


# A note on the semantics of "step" vs "next_step":
# The "step" of a TrainerState is the state after `step` steps have been taken.
# A "StepInfo"'s step is the step that was just completed. If you want the next step, use `next_step`.


@dataclass
class _Hook:
    fn: Callback
    every: int


@dataclass
class _JitHook:
    fn: JitCallback
    every: int


@dataclass
class TrainStepResult(typing.Generic[S]):
    """Result of a training step, returned from the JIT-compiled _train_step function."""

    loss: Scalar
    new_state: S
    loss_metrics: Dict[str, jax.Array]
    hook_infos: Optional[Sequence[CBInfo]]  # type: ignore


register_dataclass(TrainStepResult)


class TrainerHooks:
    hooks: List[_Hook]
    jit_hooks: List[_JitHook]

    def __init__(self):
        self.hooks = []
        self.jit_hooks = []

    def run_hooks(self, info: StepInfo, force: bool = False):
        for hook in self.hooks:
            if force or info.step % hook.every == 0:
                hook.fn.on_step(info, force=force)

    def run_jit_hooks_outside_step(self, info: StepInfo, cb_infos: Sequence[PyTree], force: bool = False):
        for s_hook, cb_info in zip(self.jit_hooks, cb_infos):
            if force or (info.step % s_hook.every == 0):
                s_hook.fn.on_step(info, cb_info)

    def run_jit_hooks(self, state: TrainerState, jit_info: InsideJitInfo, force: bool = False) -> tuple[PyTree, ...]:
        hook: _JitHook
        hook_infos = []
        for hook in self.jit_hooks:
            hook_shape = eqx.filter_eval_shape(hook.fn.inside_step, state, jit_info)
            new_s = jax.lax.cond(
                force or (state.step % hook.every == 0),
                lambda: hook.fn.inside_step(state, jit_info),
                lambda: zeros_like_tree(hook_shape),
            )
            hook_infos.append(new_s)

        return tuple(hook_infos)

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any] | JitCallback | Callback] = None, *, every: int = 1):
        def decorator(fn):
            is_something = False

            if isinstance(fn, Callback):
                self.hooks.append(_Hook(fn, every))
                is_something = True

            if isinstance(fn, JitCallback):
                self.jit_hooks.append(_JitHook(fn, every))
                is_something = True

            if not is_something:
                if not callable(fn):
                    raise ValueError(f"fn must be callable, got {fn}")
                self.hooks.append(_Hook(LambdaCallback(fn), every))

        if fn is None:
            return decorator
        else:
            return decorator(fn)


def _unify_model_and_model_init(model: Optional[M], model_init: Optional[Callable[[], M]]) -> Callable[[], M]:
    if model is not None:
        if model_init is not None:
            raise ValueError("only one of model and model_init should be specified")

        if model is not None:
            # we can't just use `lambda: model` because JAX jit can't see captures, but it can see jax partials
            model_init = jax.tree_util.Partial(lambda m: m, model)
    elif model_init is None:
        raise ValueError("one of model and model_init must be specified")

    assert model_init is not None
    return model_init


def _make_opt_sharding(x, dim, mesh: Mesh | None = None):
    current_mesh = mesh or mesh_lib.get_concrete_mesh()
    if current_mesh is None or not getattr(current_mesh, "axis_names", ()):
        return None  # no named mesh active; don't attach NamedSharding

    data_axis = ResourceAxis.DATA if ResourceAxis.DATA in current_mesh.axis_names else "data"

    if x.size > 1024:
        if x.shape[0] % dim == 0:
            return jax.NamedSharding(current_mesh, P(data_axis))
        if x.shape[1] % dim == 0:
            return jax.NamedSharding(current_mesh, P(None, data_axis))
        if x.shape[2] % dim == 0:
            return jax.NamedSharding(current_mesh, P(None, None, data_axis))
        raise ValueError(f"Neither first, second, nor third dim of tensor divides {dim}")
    return None


def lmexample_to_weighted_batch(
    example_batch: levanter.models.lm_model.LmExample, grad_accum_size: int, global_idx_start: int, pad_token_id: int
):
    """
    Convert a Levanter LmExample batch into the dict expected by the
    gradient-accumulation training loop.

    Returns a dict of NamedArrays:
        {
          "input_ids":  NamedArray with axes ("microbatch", batch, Pos),
          "labels":     NamedArray with axes ("microbatch", batch, Pos),
          "index":      NamedArray with axes ("microbatch", batch),
        }
    """
    # 1 – Get Haliax arrays and axes
    tokens = example_batch.tokens

    # We need to figure out which axis is Batch and which is Position.
    # The DataLoader is responsible for adding the batch axis. The model defines the position axis.
    # A bit of a hack, but we'll assume the position axis is the one with "pos" in its name.
    pos_axis = None
    batch_axis = None
    for axis in tokens.axes:
        if "pos" in axis.name.lower():
            pos_axis = axis
        else:
            batch_axis = axis

    if pos_axis is None or batch_axis is None:
        # Fallback to assuming (Batch, Pos) order
        batch_axis, pos_axis = tokens.axes

    B = batch_axis.size

    # 2 – targets are next-token ids; last position is padding
    labels = hax.roll(tokens, -1, axis=pos_axis)
    labels = labels.at[pos_axis, -1].set(pad_token_id)

    # 3 – construct indices/ids
    # Global train-order index for this batch
    global_index_vec_jnp = jnp.arange(B, dtype=jnp.int32) + jnp.int32(global_idx_start)
    # Per-dataset local index (may collide across datasets)
    local_index_vec_jnp = example_batch.index if example_batch.index is not None else jnp.full((B,), -1, dtype=jnp.int32)
    # Dataset id if provided by mixture, else -1
    dataset_id_vec_jnp = (
        example_batch.dataset_id if getattr(example_batch, "dataset_id", None) is not None else jnp.full((B,), -1, dtype=jnp.int32)
    )

    # 4 – reshape for micro-batching
    batch_size = B // grad_accum_size
    Microbatch = hax.Axis("microbatch", grad_accum_size)
    MiniBatch = hax.Axis(batch_axis.name, batch_size)

    batched_input_ids = tokens.unflatten_axis(batch_axis, (Microbatch, MiniBatch))
    batched_labels = labels.unflatten_axis(batch_axis, (Microbatch, MiniBatch))

    # Named and unflatten
    global_index_vec = hax.named(global_index_vec_jnp, batch_axis)
    batched_index = global_index_vec.unflatten_axis(batch_axis, (Microbatch, MiniBatch))

    local_index_vec = hax.named(local_index_vec_jnp, batch_axis)
    batched_local_index = local_index_vec.unflatten_axis(batch_axis, (Microbatch, MiniBatch))

    dataset_id_vec = hax.named(dataset_id_vec_jnp, batch_axis)
    batched_dataset_id = dataset_id_vec.unflatten_axis(batch_axis, (Microbatch, MiniBatch))

    batched = {
        "input_ids": batched_input_ids,
        "labels": batched_labels,
        # index is the global train-order index used to index weight vectors and scatter metagrads
        "index": batched_index,
        # additional diagnostics for analysis
        "dataset_id": batched_dataset_id,
        "local_index": batched_local_index,
    }
    return batched


def compute_weighted_loss(
    logits: hax.NamedArray,
    targets: hax.NamedArray,
    pad_token_id: int,
    weights: Optional[hax.NamedArray] = None,
    base_loss_fn: Callable[
        [jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = optax.softmax_cross_entropy_with_integer_labels,
    reduce: bool = True,
) -> Union[hax.NamedArray, Scalar]:
    """Compute weighted loss.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target tokens [batch, seq_len]
        weights: Per-sample weights [batch] or None for uniform weighting
    """
    vocab_axis = logits.axes[-1]
    # flatten all axes except vocab
    batch_axes = logits.axes[:-1]
    flat_logits = logits.flatten_axes(batch_axes, "flat_batch")  # shape (flat_batch, vocab)
    flat_targets = targets.flatten_axes(targets.axes, "flat_batch")  # shape (flat_batch,)

    # these are now named arrays, but base_loss_fn expects jnp arrays
    losses_flat = base_loss_fn(flat_logits.array, flat_targets.array)  # shape (flat_batch,)

    # un-flatten
    losses = hax.named(losses_flat, flat_targets.axes).unflatten_axis("flat_batch", targets.axes)

    if pad_token_id is not None:
        mask = targets != pad_token_id
        losses = losses * mask

    pos_axis = targets.axes[-1]  # assuming position is last axis of targets
    losses = hax.mean(losses, axis=pos_axis)

    if weights is not None:
        # losses is a NamedArray, but weights might be a raw jax array tracer.
        # we handle both cases by getting the raw array out of weights if it's a NamedArray.
        weights_array = weights.array if isinstance(weights, hax.NamedArray) else weights
        weighted_losses_array = losses.array * weights_array
        losses = hax.named(weighted_losses_array, losses.axes)

    if reduce:
        return hax.mean(losses).scalar()
    else:
        return losses


def make_train_functions(
    optimizer: optax.GradientTransformation,
    pad_token_id: int,
    is_trainable: FilterTree = True,
    compute_axis_mapping: ResourceMapping = None,
    parameter_axis_mapping: ResourceMapping = None,
):
    """Takes in the static arguments and returns training functions.

    Training operates over single batches, where each batch consists of grad_accum_size microbatches.
    We checkpoint the inside of the batch step, so that during the VJP we only materialize each microbatch worth of data gradients at once.

    Note how we do the VJP in two steps - first propagating back to the summed gradient, and then back to the individual exampeles.
    For the second step, using a JVP is faster for batched inner products with per-example gradients so we have a manual version of that.
    """

    def microbatch_step_full(data_weights, train_state, carry, microbatch_data):
        """ runs a single microbatch step, accumulating the gradients and the loss """
        grad_buffer, loss_sum = carry
        model, _ = train_state
        inputs = microbatch_data["input_ids"]
        targets = microbatch_data["labels"]
        index = microbatch_data["index"]
        weights = data_weights[index.array]

        def weighted_loss(model):
            logits = model(inputs, attn_mask=AttentionMask.causal())
            return compute_weighted_loss(logits, targets, pad_token_id, weights)

        #loss, grads = eqx.filter_checkpoint(eqx.filter_value_and_grad(weighted_loss))(model)
        differentiable_model, static_model = eqx.partition(model, eqx.is_inexact_array)
        loss, grads = eqx.filter_checkpoint(
            lambda m: eqx.filter_value_and_grad(weighted_loss)(eqx.combine(m, static_model))
        )(differentiable_model)
        #loss, grads = eqx.filter_value_and_grad(weighted_loss)(
        #    eqx.combine(differentiable_model, static_model)
        #)

        grad_buffer = jax.tree.map(lambda g_new, g: g + g_new, grads, grad_buffer)
        return (grad_buffer, loss_sum + loss), loss


    def _make_zero_grad_tree(model):
        """Arrays → zeros, everything else → None."""
        def _zero_or_none(leaf):
            return jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else None
        return jax.tree.map(_zero_or_none, model)


    def compute_grads(data_weights, train_state, batch_data):
        """
        Accumulate gradients over all micro-batches; return (grad_buffer, mean_loss).
        `train_state` is (model, opt_state) as elsewhere.
        """
        with hax.axis_mapping(compute_axis_mapping):
            model, _ = train_state

            # 1️⃣  zero-initialised buffer with full model structure
            #grad_buffer = _make_zero_grad_tree(model)
            diff_model, _ = eqx.partition(model, eqx.is_inexact_array)
            grad_buffer = jax.tree.map(jnp.zeros_like, diff_model)
            loss_sum    = 0.0

            debug.print("compute_grads > batch_data: {}", batch_data['input_ids'].shape)

            # 2️⃣  partially-applied micro-batch step
            micro_step = functools.partial(
                microbatch_step_full,      # ← uses the _add_or_keep helper above
                data_weights,
                train_state,
            )

            # 3️⃣  scan over micro-batches
            (grad_buffer, loss_sum), _ = hax.scan(
                eqx.filter_checkpoint(micro_step), 'microbatch',
                unroll=1,
            )((grad_buffer, loss_sum), batch_data)

            # 4️⃣  average loss and gradients
            #import pdb; pdb.set_trace()
            num_mb   = batch_data["input_ids"].shape['microbatch']
            mean_loss = loss_sum / num_mb
            grad_buffer = jax.tree.map(lambda g: g / num_mb, grad_buffer)

            debug.print("compute_grads > mean_loss: {}", mean_loss)
            #debug.print("compute_grads > grad_buffer: {}", grad_buffer.embedding.weight)
            return grad_buffer, mean_loss


    def update_with_grads(avg_grad: jnp.ndarray, train_state):
        """Updates the model parameters with the average gradient (FSDP-safe)."""
        model, opt_state = train_state

        # 1) Filter to trainable parts and compute 'overwrites' (unchanged logic)
        train_grads = trainables_only(avg_grad, is_trainable)
        overwrites, train_grads = partition_for_grad_overwrite(train_grads)

        trainable_model = trainables_only(model, is_trainable)
        _ignored, trainable_model = partition_for_grad_overwrite(trainable_model)

        # 2) Get the current mesh (works inside/outside jit)
        current_mesh = (
            mesh_lib.get_abstract_mesh()
            if mesh_lib.get_concrete_mesh() is not None
            else mesh_lib.thread_resources.env.physical_mesh
        )

        # 3) Re-assert FSDP sharding precisely on the objects passed to the optimizer
        #    (prevents XLA from up-gathering full-width params for the update)
        '''
        train_grads = jax.tree.map(
            lambda x: _constrain_array_to_fsdp(x, current_mesh),
            train_grads,
            is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
        )

        trainable_model = jax.tree.map(
            lambda x: _constrain_named_or_array(x, current_mesh),
            trainable_model,
            is_leaf=lambda x: isinstance(x, hax.NamedArray) or (hasattr(x, "shape") and hasattr(x, "dtype")),
        )
        '''

        # 4) Optimizer step (unchanged)
        updates, opt_state = optimizer.update(train_grads, opt_state, params=trainable_model)

        '''
        # 5) (Optional) Constrain updates as well (can reduce incidental gathers/copies)
        updates = jax.tree.map(
            lambda x: _constrain_array_to_fsdp(x, current_mesh),
            updates,
            is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
        '''

        # 6) Apply updates with your existing 'overwrites' mechanism
        new_model = apply_updates(model, updates, overwrites)
        return new_model, opt_state



    def single_batch_step(data_weights: jnp.ndarray, train_state: Tuple[Any, Any], batch_data: Dict[str, jnp.ndarray]) -> Tuple[Tuple[Any, Any], jnp.ndarray]:
        """Runs for 1 global batch, which is grad_accum_steps * batch_size."""
        grad_buffer, loss = compute_grads(data_weights, train_state, batch_data)
        model_params, opt_params = update_with_grads(grad_buffer, train_state)
        return (model_params, opt_params), loss


    def unreduced_microbatch_losses(model, data_weights, microbatch_data):
        """ Computes the loss of a single microbatch """
        inputs = microbatch_data["input_ids"]
        targets = microbatch_data["labels"]
        index = microbatch_data["index"]
        weights = data_weights[index.array]

        def weighted_loss(model):
            with hax.axis_mapping(compute_axis_mapping):
                logits = model(inputs, attn_mask=AttentionMask.causal()) #, training=True)
                return compute_weighted_loss(logits, targets, pad_token_id, weights, reduce=False)

        example_losses = weighted_loss(model)

        return example_losses


    def microbatch_vjp_grad_fun(avg_grad_grad: jnp.ndarray, data_weights, train_state, batch_data):
        """ /Manually/ computes the per-example metagrad VJP.
        Any time we make changes to the updates, we have to check this against the true VJP.

        The logic is simple - if we know the gradient wrt the average gradient, then upweighting any individual example's gradient
        has the derivative which is just the inner product of the example's gradient with the average gradient.
        """

        model_params, opt_params = train_state
        def jvp_microbatch(microbatch_data):
            recued_microbatch_fn = jax.tree_util.Partial(unreduced_microbatch_losses, data_weights=data_weights, microbatch_data=microbatch_data)
            return eqx.filter_jvp(recued_microbatch_fn, (model_params,), (avg_grad_grad,))[1]
        data_weights_list = hax.scan(lambda carry, x: (None, jvp_microbatch(x)), "microbatch")(None, batch_data)[1]
        # data_weights_list is now of size (grad_accum_size, batch_size) and need to flatten it.
        # The ordering matches that of flattening the microbatch, but we want to reorder it to match the data_weights, which is controlled by the index
        #import pdb; pdb.set_trace()
        #debug.print(">>>>> batch_data: {} | data_weights_list: {}", batch_data["input_ids"].array, data_weights_list.array)
        data_weights_list = data_weights_list.flatten("batch")
        #debug.print(f">>>>> data_weights_list: {data_weights_list.axes}")
        inverse_index = jnp.argsort(batch_data["index"].flatten("batch").array)
        data_weights_list = data_weights_list.array[inverse_index]

        return data_weights_list, None, None


    def run_vjp_update(non_donated_args, params_grad, opt_grad, grad_sharding):
        """computes the backwards functions that reverse state->compute_grads->update_with_grads->new_state.
        This computes the branch that does not involve the metagrads themselves.
        Mirrors forward sharding: params/opt state use parameter_axis_mapping, compute uses compute_axis_mapping."""
        initial_params, initial_opt_state, data_weights, batch_data = non_donated_args
        train_state = (initial_params, initial_opt_state)

        # 1) Forward leg: state -> avg grads; respect parameter sharding layout if provided
        def train_state_to_grads_with_sharding(state):
            with jax.named_scope("train_state_to_grads_with_fsdp_sharding"):
                model, opt_state = state
                #if parameter_axis_mapping is not None:
                #    with hax.axis_mapping(parameter_axis_mapping):
                #        grad_buffer, loss = compute_grads(data_weights, (model, opt_state), batch_data)
                #else:
                grad_buffer, loss = compute_grads(data_weights, (model, opt_state), batch_data)
                return grad_buffer, loss
        (grad_buffer, loss), vjp_grad_state_fun = eqx.filter_vjp(train_state_to_grads_with_sharding, train_state)
        # Debug: sharding of avg grad buffer produced by state→grads leg
        #_debug_weight_sharding("vjp_update/grad_buffer_post_state_to_grads", grad_buffer)

        # 2) Forward leg: (avg grads, state) -> new state; apply grad sharding if provided
        #if grad_sharding is not None:
        grad_buffer = hax.shard(grad_buffer, parameter_axis_mapping)
        # Debug: sharding after applying grad_sharding constraint
        #_debug_weight_sharding("vjp_update/grad_buffer_post_grad_sharding", grad_buffer)
        _, vjp_update_fun = eqx.filter_vjp(update_with_grads, grad_buffer, train_state)
        avg_grad_grad, train_state_grad = vjp_update_fun((params_grad, opt_grad))

        # 3) Backward through grads path and sum contributions
        avg_grad_grad = hax.shard(avg_grad_grad, parameter_axis_mapping)
        # Debug: sharding of gradient wrt avg grads and wrt train_state from update leg
        #_debug_weight_sharding("vjp_update/avg_grad_grad_from_update", avg_grad_grad)
        #_debug_weight_sharding("vjp_update/train_state_grad_from_update", train_state_grad)

        train_state_grad_through_grads = vjp_grad_state_fun((avg_grad_grad, 0.0))[0]
        train_state_grad = jax.tree.map(
            lambda x, y: x + y if x.dtype is not jax.dtypes.float0 else x,
            train_state_grad,
            train_state_grad_through_grads,
        )
        # Debug: final sharding after summing both gradient paths
        #_debug_weight_sharding("vjp_update/train_state_grad_post_sum", train_state_grad)
        return avg_grad_grad, train_state_grad, loss


    def run_vjp_grad_manual(initial_model, initial_opt_state, data_weights, batch_data, avg_grad_grad):
        """ computes the backwards functions that reverse data_weights -> compute grads
        this is the second part that propagates gradients at the batch level to each example and uses a hand computed VJP """
        train_state = (initial_model, initial_opt_state)
        # Debug: inspect sharding of the incoming avg_grad_grad tree
        #_debug_weight_sharding("vjp_grad_manual/avg_grad_grad_input", avg_grad_grad)

        manual_vjp_grad_fun = jax.tree_util.Partial(
            microbatch_vjp_grad_fun,
            data_weights=data_weights,
            train_state=train_state,
            batch_data=batch_data,
        )
        if parameter_axis_mapping is not None:
            with hax.axis_mapping(parameter_axis_mapping):
                res = manual_vjp_grad_fun(avg_grad_grad)
        else:
            res = manual_vjp_grad_fun(avg_grad_grad)

        # Debug: inspect sharding of gradient wrt data_weights (first return)
        try:
            d_dw = res[0]
            _debug_array_sharding("vjp_grad_manual/d_data_weights", d_dw)
        except Exception:
            pass

        return res


    def run_vjp_grad_jax(initial_model, initial_opt_state, data_weights, batch_data, avg_grad_grad):
        """ computes the backwards functions that reverse data_weights -> compute grads
        this is the second part that propagates gradients at the batch level to each example. this is a slower, jax computed version to be used for a reference in tests"""
        train_state = (initial_model, initial_opt_state)
        # Debug: inspect sharding of the incoming avg_grad_grad tree
        #_debug_weight_sharding("vjp_grad_jax/avg_grad_grad_input", avg_grad_grad)

        if parameter_axis_mapping is not None:
            with hax.axis_mapping(parameter_axis_mapping):
                _, vjp_grad_fun, loss = eqx.filter_vjp(
                    compute_grads, data_weights, train_state, batch_data, has_aux=True
                )
        else:
            _, vjp_grad_fun, loss = eqx.filter_vjp(
                compute_grads, data_weights, train_state, batch_data, has_aux=True
            )
        res = vjp_grad_fun(avg_grad_grad)

        # Debug: inspect sharding of gradient wrt data_weights (first return)
        try:
            d_dw = res[0]
            _debug_array_sharding("vjp_grad_jax/d_data_weights", d_dw)
        except Exception:
            pass

        return res

    return single_batch_step, run_vjp_update, run_vjp_grad_manual, run_vjp_grad_jax


class WrappedLossFunction:
    """
    Wrapper around a loss function that provides a uniform interface.

    Handles casting & executing in the proper axis mapping.

    Always returns (loss, wrapped_metrics_dict). The user loss function
    may return `Metric` objects or floats. Floats are automatically coerced to Metrics
    based on their name.
    """

    _raw_fn: ComputeLossFunction
    _mp: jmp.Policy
    _compute_axis_mapping: ResourceMapping

    def __init__(
        self,
        raw_fn: ComputeLossFunction,
        mp: jmp.Policy,
        compute_axis_mapping: ResourceMapping,
    ):
        """
        Args:
            raw_fn: The underlying loss function
            mp: Mixed precision policy for casting
            compute_axis_mapping: Axis mapping for compute
        """
        self._raw_fn = raw_fn
        self._mp = mp
        self._compute_axis_mapping = compute_axis_mapping

    def __call__(self, model, *batch, **batch_kwargs) -> Tuple[Scalar, Dict[str, Metric]]:
        """
        Call the loss function with model casting and axis mapping.
        Always returns (loss, wrapped_metrics) where metrics are Metric objects.
        """
        with hax.axis_mapping(self._compute_axis_mapping):
            model = self._mp.cast_to_compute(model)
            result = self._raw_fn(model, *batch, **batch_kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            loss, metrics = result
        else:
            # Treat scalar return as (loss, {})
            loss = result
            metrics = {}

        if not isinstance(metrics, dict):
            raise ValueError(f"Expected metrics to be dict, got {type(metrics)}")

        # Auto-wrap plain values into Metric objects
        wrapped_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, Metric):
                wrapped_metrics[key] = value
            else:
                # Infer type from name and wrap
                wrapped_metrics[key] = auto_metric_from_name(key, value)

        return _ensure_scalar(loss.mean()), wrapped_metrics


class Trainer:
    config: "TrainerConfig"
    optimizer: GradientTransformation
    hooks: TrainerHooks
    tracker: levanter.tracker.Tracker
    is_trainable_param: PyTree[FilterSpec]
    _raw_loss_function: Callable
    _cmanagers: List[typing.ContextManager] = []
    consumed_tokens: int = 0

    def __init__(
        self,
        config: "TrainerConfig",
        optimizer: GradientTransformation,
        loss_fn: ComputeLossFunction,
        *,
        add_default_hooks: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            config:  the trainer config
            optimizer: the optimizer, e.g. `optax.adam(1e-3)` or produced by [levanter.optim.OptimizerConfig][]
            loss_fn (Callable): the loss function. This should be a function that takes a model and some inputs and returns
                either a scalar loss, or a tuple of (scalar loss, metrics_dict). The metrics dict will be automatically
                logged to the tracker with appropriate prefixes (e.g., "train/accuracy", "eval/perplexity").
                The function should be jit-able and should not have any side effects.
        """
        self.hooks = TrainerHooks()
        self.config = config
        self.optimizer = optimizer
        self._raw_loss_function = loss_fn
        self.data_weight_vector: Optional[jnp.ndarray] = None
        self.consumed_tokens = 0
        self.debug = debug
        if isinstance(config.tracker, Sequence):
            self.tracker = levanter.tracker.CompositeTracker([c.init(self.run_id) for c in config.tracker])
        else:
            self.tracker = config.tracker.init(self.run_id)

        self._cmanagers = []
        self.profiler_running = False
        # When set, only allow saving checkpoints at these steps during forward train.
        # None means use default checkpointer policy (no gating).
        self._checkpoint_allowed_steps: Optional[set[int]] = None
        # Keep a handle to the checkpointer used by default hooks so we can invoke
        # it manually (e.g., during segment retraining).
        self._checkpointer = None
        # Global loss_masks array filled during training (replaces old list-based collection)
        self.loss_masks_global: Optional[jnp.ndarray] = None

        if add_default_hooks:
            self._add_default_hooks()

        self._cmanagers = []
        self._logged_jaxprs: set[str] = set()


    @cached_property
    def loss_fn(self) -> WrappedLossFunction:
        """
        Wrapped loss function that always returns (loss, metrics_dict).
        Casts the model to compute precision and sets the context axis mapping to compute.
        """
        return WrappedLossFunction(
            self._raw_loss_function,
            self.mp,
            self.compute_axis_mapping,
        )

    @property
    def run_id(self) -> str:
        """Returns the run id"""
        assert self.config.id is not None
        return self.config.id

    @property
    def mp(self) -> jmp.Policy:
        """Returns the mixed precision policy"""
        return self.config.mp

    @property
    def num_train_steps(self) -> int:
        return self.config.num_train_steps

    @typing.overload
    def add_hook(self, fn: Callable[[StepInfo], Any], *, every: int = 1): ...

    @typing.overload
    def add_hook(self, fn: JitCallback, *, every: int = 1): ...

    @typing.overload
    def add_hook(self, fn: Callback, *, every: int = 1): ...

    @typing.overload
    def add_hook(self, *, every: int = 1): ...

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any] | Callback | JitCallback] = None, *, every: int = 1):
        return self.hooks.add_hook(fn, every=every)

    def run_hooks(self, info: StepInfo, force: bool = False):
        self.hooks.run_hooks(info, force=force)

    @property
    def parameter_axis_mapping(self) -> ResourceMapping:
        return self.config.parameter_axis_mapping

    @property
    def compute_axis_mapping(self) -> ResourceMapping:
        return self.config.compute_axis_mapping

    @property
    def device_mesh(self) -> Mesh:
        return self.config.device_mesh

    def get_collected_loss_masks(self) -> Optional[jnp.ndarray]:
        """
        Returns the global loss_masks indicator array filled during training.
        Returns None if no loss_masks were collected.

        The returned array is a 1D boolean array of shape (num_train_steps * train_batch_size,)
        where each entry is 1 if that example had any non-zero loss_weight, 0 otherwise.
        Indexed by sequential train order (first example seen → index 0, etc.).

        In distributed training, this is a sharded JAX array that can be saved
        with _save_array_tpu_safe which handles the sharding correctly.
        """
        return self.loss_masks_global

    def clear_collected_loss_masks(self):
        """Clears the global loss_masks array from memory."""
        self.loss_masks_global = None

    @property
    def TrainBatch(self):
        return self.config.TrainBatch

    @property
    def EvalBatch(self):
        return self.config.EvalBatch

    def __enter__(self):
        if len(self._cmanagers) > 0:
            raise RuntimeError("Trainer is already entered")

        # Debug: device mesh summary
        try:
            mesh = self.device_mesh
            mesh_shape = getattr(mesh.devices, "shape", None)
            devices_total = getattr(mesh.devices, "size", None)
            print(
                f"[Mesh DBG] axis_names={mesh.axis_names} shape={mesh_shape} devices_total={devices_total}",
                flush=True,
            )
            print(
                f"[AxisMapping DBG] compute={self.compute_axis_mapping} params={self.parameter_axis_mapping}",
                flush=True,
            )
        except Exception:
            pass

        self._cmanagers = [
            levanter.current_tracker(self.tracker),
            haliax.partitioning.set_mesh(self.device_mesh),
            hax.axis_mapping(self.parameter_axis_mapping),
        ]

        for cmanager in self._cmanagers:
            cmanager.__enter__()

        return self

    def __exit__(self, *args):
        problems = []
        for cmanager in reversed(self._cmanagers):
            try:
                cmanager.__exit__(*args)
            except Exception as e:
                problems.append(e)

        self._cmanagers = []

        if len(problems) > 0:
            raise RuntimeError("Exception(s) occurred while exiting trainer", problems) from problems[0]

    def initial_state(
        self,
        training_key: PRNGKeyArray,
        model: Optional[M] = None,
        model_init: Optional[Callable[[], M]] = None,
        *,
        is_trainable: PyTree[FilterSpec] = True,
    ) -> TrainerState[M]:
        """
        Either loads a checkpoint or initializes a fresh trainer state. This is the recommended way to initialize
        a trainer state.

        This method is smart enough to handle subclasses of TrainerState. If you want to extend TrainerState, you
        can override _initialize_state_from_scratch

        Args
            is_trainable: optional filter spec for the trainable parameters. This is used to filter out non-trainable
                parameters for the optimizer state and for computing gradients. Non-trainable parameters are also
                not checkpointed. If you don't specify this, all parameters are assumed to be trainable.

        Returns:
            TrainerState: the initial state,
        """
        model_init = _unify_model_and_model_init(model, model_init)

        del model
        assert model_init is not None

        # first try to load a full trainer state checkpoint
        checkpoint_path = self.checkpoint_path

        load_checkpoint = self.config.load_checkpoint
        # we don't save the full trainer state, so we need to filter out the non-trainable parameters
        if load_checkpoint is True and not fsspec_utils.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
        elif load_checkpoint is None:
            load_checkpoint = levanter.checkpoint.is_checkpoint_path(checkpoint_path)

        if load_checkpoint is False and self.config.initialize_from is not None:
            # we're not going to load a checkpoint from this run, so instead we can initialize from a different run
            logger.info(f"Initializing from {self.config.initialize_from}")
            load_checkpoint = True
            checkpoint_path = self.config.initialize_from
            if not is_checkpoint_path(checkpoint_path):
                raise ValueError(f"initialize_from must be a checkpoint path, got {checkpoint_path}")

        def init_state_and_model(model_init, training_key):
            model = model_init()
            # only force trainable params to param precision. Other params are cast to compute precision
            state = TrainerState.init(
                self.optimizer,
                model,
                key=training_key,
                is_trainable=is_trainable,
                mp=self.mp,
                quantization=self.config.quantization,
                model_averaging=self.config.model_averaging,
            )
            return state

        trainer_state_shape = eqx.filter_eval_shape(init_state_and_model, model_init, training_key)
        saveable_train_state = saveable_training_mask(trainer_state_shape, is_trainable)

        state = load_checkpoint_or_initialize(
            init_state_and_model,
            checkpoint_path,
            axis_mapping=self.parameter_axis_mapping,
            mesh=self.device_mesh,
            is_checkpointed=saveable_train_state,
            do_load=load_checkpoint,
            allow_partial=self.config.allow_partial_checkpoint,
        )(model_init, training_key)

        # If resuming from a checkpoint at startup, log the forward checkpoint step
        #if load_checkpoint:
        try:
            resumed_step = int(state.step)
        except Exception:
            # Fallback in case state.step is a device array type
            resumed_step = int(jax.device_get(state.step))
        logger.info(f"> Resuming forward checkpoint: step-{resumed_step} from {checkpoint_path}")

        # NOTE: Disabled FSDP constraint block - was causing resume non-determinism
        # The checkpoint restoration should already provide correct sharding.
        '''
        # --- Force FSDP sharding on state once, pre-JIT ---
        with hax.axis_mapping(self.parameter_axis_mapping):
            mesh = self.device_mesh

            # Constrain model parameters (handles NamedArray leaves)
            constrained_model = jax.tree.map(
                lambda x: _constrain_named_or_array(x, mesh),
                state.model,
                is_leaf=lambda z: isinstance(z, hax.NamedArray) or (hasattr(z, "shape") and hasattr(z, "dtype")),
            )

            # Constrain optimizer state (raw arrays only)
            constrained_opt = jax.tree.map(
                lambda x: _constrain_array_to_fsdp(x, mesh),
                state.opt_state,
                is_leaf=lambda z: hasattr(z, "shape") and hasattr(z, "dtype"),
            )

            state = eqx.tree_at(lambda s: (s.model, s.opt_state), state, (constrained_model, constrained_opt))
        '''
        return state


    @property
    def checkpoint_path(self) -> str:
        checkpoint_path = self.config.load_checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpointer.expanded_path(self.run_id)
        return checkpoint_path

    def train_step(self, state: S, *batch: X, **batch_kwargs) -> StepInfo[S]:
        """
        Performs a single training step.
        """
        # jit hooks impose a nontrivial cost even when they're not run (since they defeat some compiler optimizations)
        # so we avoid running them when they're not needed
        # this results in two compiles, but the cost of the second compile is worth it
        hooks_this_time = any(state.step % h.every == 0 for h in self.hooks.jit_hooks)

        with capture_time() as step_time:
            if hooks_this_time:
                result = self._maybe_save_jaxpr("train_step", self._jit_train_step_fn, state, batch, batch_kwargs)
                # force the loss so timing numbers are accurate. laziness isn't going to help here (i think?)
            else:
                result = self._maybe_save_jaxpr(
                    "train_step_hooks", self._jit_train_step_fn_no_hook, state, batch, batch_kwargs
                )

            loss = result.loss.item()

            if self.config.crash_on_nan and jnp.isnan(loss):
                raise RuntimeError("Loss is NaN")

            if self.config.crash_on_inf and jnp.isinf(loss):
                raise RuntimeError("Loss is Inf")

            info = StepInfo(result.new_state, loss, step_time())

            with capture_time() as hook_time:
                self.run_hooks(info)
                if hooks_this_time:
                    self.hooks.run_jit_hooks_outside_step(info, result.hook_infos)

            # Log metrics from loss function and throughput metrics
            metrics_to_log = {**result.loss_metrics, "throughput/hook_time": hook_time()}
            levanter.tracker.log(metrics_to_log, step=info.step)

        print('[Timing Info] > step time:', step_time(), 'hook time:', hook_time())
        return info

    def training_steps(self, state: S, train_loader) -> typing.Iterator[StepInfo[S]]:
        """
        Generator that yields training steps and runs hooks.
        """
        iter_data = iter(train_loader)

        # print optimizer info
        #print('$$$$ Optimizer info:', state.opt_state)

        self.dataset_ids = []

        while int(state.step) < self.num_train_steps:
            print(f'\n\n------------------------------- Forward it: {state.step} -------------------------------')
            with capture_time() as loading_time:
                try:
                    example = next(iter_data)
                    self.dataset_ids.append(example.dataset_id)

                    # Fill in loss_masks_global indicator array in sequential train order
                    if self.loss_masks_global is not None:
                        if hasattr(example, "loss_weight") and example.loss_weight is not None:
                            loss_weight = example.loss_weight
                            # Extract underlying array if it's a NamedArray
                            if hasattr(loss_weight, "array"):
                                loss_weight = loss_weight.array

                            # Calculate sequential indices based on step and batch size
                            # loss_masks_global is sharded the same way as the batch data,
                            # so each process writes to its corresponding shard automatically
                            batch_size = loss_weight.shape[0]
                            start_idx = int(state.step) * batch_size
                            end_idx = start_idx + batch_size

                            # Compute indicator: 1 if row sum > 0, else 0
                            # Sum over sequence dimension (axis=1 or axis=-1)
                            row_sums = loss_weight.sum(axis=-1)
                            mask_indicators = (row_sums > 0).astype(jnp.int32)

                            # Debug: warn if all indicators are zero but mask exists
                            num_active = int(mask_indicators.sum())
                            if num_active == 0:
                                logger.warning(
                                    f"[LossMask] Step {state.step}: All indicators are ZERO! loss_weight shape={loss_weight.shape}"
                                )

                            # Update the global indicator array at sequential indices
                            self.loss_masks_global = self.loss_masks_global.at[start_idx:end_idx].set(mask_indicators)

                            if int(state.step) % 100 == 0:
                                logger.info(
                                    f"[LossMask] Step {state.step}: Filled loss_weight indicators at [{start_idx}:{end_idx}], active: {num_active}/{batch_size}"
                                )
                        else:
                            # Missing loss_weight - log more frequently to catch this
                            if int(state.step) % 10 == 0:
                                logger.warning(
                                    f"[LossMask] Step {state.step}: Example missing loss_weight! has_attr={hasattr(example, 'loss_weight')}, is_none={getattr(example, 'loss_weight', None) is None}"
                                )

                    #if int(state.step) == 0:
                        # NOTE(safety): The preview decode below fetches a global jax.Array to host, which
                        # fails on multi-host TPU pods ("non-addressable devices"). Commented out for now.
                        # If needed later, gate on jax.process_count()==1 or gather via multihost_utils.
                        #
                        # from transformers import GPT2Tokenizer
                        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                        # input_ids = np.asarray(example.tokens.array).astype(int)
                        # texts = tokenizer.batch_decode(
                        #     input_ids.tolist(),
                        #     skip_special_tokens=True,
                        #     clean_up_tokenization_spaces=False,
                        # )
                        # for j, text in enumerate(texts):
                        #     print(f"Example {j}:")
                        #     print(text)
                        #     print("-" * 20)
                        #     if j == 5:
                        #         break

                    # Optional: print loss weight information for each batch
                    if getattr(self.config, "debug_print_loss_mask", False) and example.loss_weight is not None:
                        # and hasattr(example, "loss_weight") and example.loss_weight is not None:
                        try:
                            # Convert to host scalar safely
                            mask_sum = int(np.asarray(hax.sum(example.loss_weight).array))
                        except Exception:
                            mask_sum = -1
                        print(
                            f'> step: {int(state.step)} | ds_id: {getattr(example, "dataset_id", None)} | '
                            f'idx: {getattr(example, "index", None)} | mask_sum: {mask_sum}',
                            flush=True,
                        )
                        if True:
                            try:
                                m = np.asarray(example.loss_weight.array)
                                if m.ndim >= 2:
                                    # Print per-example sums across the last axis (positions)
                                    per_example = m.sum(axis=-1)
                                    print(f'  mask_sums_per_example: {per_example.tolist()}', flush=True)
                                else:
                                    print(f'  mask: {m.tolist()}', flush=True)
                            except Exception as e:
                                print(f'  [mask values unavailable: {e}]', flush=True)
                    self.consumed_tokens += example.tokens.size
                except StopIteration:
                    logger.info("Reached end of training data loader")
                    break

            # print('[Timing Info] > data loading time:', loading_time(), flush=True)

            # Optionally persist batch inputs for exact reproducibility. We avoid fetching global arrays
            # here; store the batch object for a later save hook executed outside JIT in train_lm.main.
            try:
                self._last_batch_example = example  # type: ignore[attr-defined]
            except Exception:
                pass

            info = self.train_step(state, example)
            state = info.state

            levanter.tracker.log(
                {"throughput/loading_time": loading_time(), "metrics/consumed_tokens": self.consumed_tokens},
                step=info.step,
            )

            yield info


    def train(
        self,
        state: S,
        train_loader: Iterable[X],
        data_weight_vector: Optional[jnp.ndarray] = None,
        *,
        checkpoint_steps: Optional[Iterable[int]] = None,
        loss_masks_global: Optional[jnp.ndarray] = None,
    ) -> StepInfo[S]:
        """
        Performs training until the number of steps is reached.
        """
        info: Optional[StepInfo[S]] = None
        self.data_weight_vector = data_weight_vector
        self.loss_masks_global = loss_masks_global

        # Gate checkpoint saving to specified steps if provided
        prev_allowed = self._checkpoint_allowed_steps
        try:
            self._checkpoint_allowed_steps = (
                set(int(s) for s in checkpoint_steps) if checkpoint_steps is not None else None
            )

            # initialize consumed_tokens
            if int(state.step) > 0:
                # TODO: this is not exactly correct if the batch size has changed during training
                # a better way would be to save this in the checkpoint
                loader = getattr(train_loader, "dl", train_loader)
                if hasattr(loader.data_store, "seq_len"):
                    seq_len = loader.data_store.seq_len
                    batch_size = self.config.batch_schedule.value_at_step(state.step)
                    self.consumed_tokens = int(state.step) * batch_size * seq_len
            else:
                self.consumed_tokens = 0

            # force hooks to run at the beginning
            if state.step == 0:
                self.run_hooks(StepInfo(state, jnp.array(0.0), 0.0), force=True)

            last_info: Optional[StepInfo[S]] = None
            for info in self.training_steps(state, train_loader):
                last_info = info

            # if we didn't train, info is None, so we should use the initial state
            if last_info is None:
                print("No training steps were executed. The dataset may be empty or there are no steps left to run.")
                time.sleep(10)
                last_info = StepInfo(state, jnp.array(float("nan")), 0.0)

            # force hooks to run at the end
            self.run_hooks(last_info, force=True)

            return last_info
        finally:
            # Restore checkpoint gating
            self._checkpoint_allowed_steps = prev_allowed

    def construct_metagrad_helpers(self, use_manual_vjp=True):
        pad_token_id = -100
        single_batch_step, run_vjp_update, run_vjp_grad_manual, run_vjp_grad_jax = make_train_functions(
            self.optimizer,
            pad_token_id,
            compute_axis_mapping=self.compute_axis_mapping,
            parameter_axis_mapping=self.parameter_axis_mapping,
        )
        if use_manual_vjp: # this should be the faster manual VJP that uses JVPs internally.
            self.run_vjp_grad = run_vjp_grad_manual
        else: # this is the 'exact' VJP chaining approach. used to check correctness.
            self.run_vjp_grad = run_vjp_grad_jax
        self.single_batch_step = single_batch_step
        self.run_vjp_update = run_vjp_update


    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)
        else:
            pass

    def jax_debug_print(self, *args, **kwargs):
        if self.debug:
            debug.print(*args, **kwargs)
        else:
            pass


    def train_and_replay(self, state: S, train_loader: Iterable[X], reversed_train_loader: Iterable[X],
                         val_loader: Iterable[X],
                         data_weight_vector: jnp.ndarray,
                         segment_starts,
                         train_only=False,
                         loss_masks_global: Optional[jnp.ndarray] = None) -> StepInfo[S]:
        init_state = self._duplicate_pytree_arrays(state) # TODO: check this
        forward_ckpt_steps = set(int(s) - 1 for s in segment_starts if int(s) > 0)
        metagrad_ckpt_dir = fsspec_utils.join_path(self.checkpoint_path, "metagrad_checkpoints")
        print(f"[DBG] Metagrad checkpoint directory: {metagrad_ckpt_dir}", flush=True)
        # Determine resume only if at least one valid step-* checkpoint exists
        resumed_from_metagrad = False
        if fsspec_utils.exists(metagrad_ckpt_dir):
            try:
                fs, _, paths = fsspec.get_fs_token_paths(metagrad_ckpt_dir)
                files = fs.ls(paths[0])
                for entry in files:
                    name = entry["name"] if isinstance(entry, dict) else str(entry)
                    base = os.path.basename(name)
                    if base.startswith("step-"):
                        meta_path = fsspec_utils.join_path(metagrad_ckpt_dir, base)
                        meta_json = fsspec_utils.join_path(meta_path, "metadata.json")
                        if fsspec_utils.exists(meta_json):
                            resumed_from_metagrad = True
                            break
            except (FileNotFoundError, IndexError):
                resumed_from_metagrad = False
        if resumed_from_metagrad:
            logger.info(f"[DBG] Resumed from backward; skipping forward training and reward eval")
        # Always preserve the very last forward checkpoint (step-(num_train_steps - 1)) during backward cleanup.
        # This ensures the final checkpoint is retained regardless of which steps were saved by policy.
        self._last_forward_ckpt_to_keep = self.config.num_train_steps - 1

        # Ensure reward is defined even if we skip reward evaluation
        reward = 0.0
        metagrads = jnp.zeros_like(data_weight_vector)
        # Global accumulators for dataset id and local (per-dataset) index, aligned to train-order indices
        dataset_ids_global = jnp.full(metagrads.shape, -1, dtype=jnp.int32)
        local_indices_global = jnp.full(metagrads.shape, -1, dtype=jnp.int32)

        if not resumed_from_metagrad:
            info = self.train(state, train_loader, data_weight_vector, checkpoint_steps=forward_ckpt_steps, loss_masks_global=loss_masks_global)
            final_model, final_opt_state = info.state.model, info.state.opt_state
            final_model = inference_mode(final_model, True)
            final_model = self.mp.cast_to_compute(final_model)

            #self.dataset_ids = jnp.concatenate(self.dataset_ids)
            #print('dataset_ids_global_fwd:', jnp.concatenate(self.dataset_ids))

            # 1) Define a scalar reward loss (same as you already do, just as a callable)
            def reward_loss_fn(model, batch):
                with hax.axis_mapping(self.compute_axis_mapping):
                    # Per-token losses (NamedArray with Pos axis)
                    losses = compute_next_token_loss(model, batch, reduction=None, reduction_axis=())
                    weight = batch.loss_weight
                    # Negative average log-likelihood over masked tokens
                    loss_scalar = (-hax.einsum("->", losses, weight) / hax.sum(weight)).scalar()
                    # Return aux metrics as empty dict to satisfy has_aux=True in gradient code
                    return loss_scalar, {}

            # JIT a plain two-arg function that calls your microbatched gradient helper
            def _reward_grad_step_impl(model, batch):
                return self._compute_gradients_microbatched(reward_loss_fn, model, batch)

            reward_grad_step = eqx.filter_jit(_reward_grad_step_impl)

            # 2) Init a device-resident, sharded accumulator that matches param sharding
            differentiable_final_model, _ = eqx.partition(final_model, eqx.is_inexact_array)
            differentiable_final_model = hax.shard(differentiable_final_model, self.parameter_axis_mapping)
            #with hax.axis_mapping(self.parameter_axis_mapping):

            @eqx.filter_jit
            def zeros_like(m):
                return hax.shard(hax.tree_util.tree_map(hax.zeros_like, m), self.parameter_axis_mapping)

            params_grad = zeros_like(differentiable_final_model)
            #params_grad = hax.tree_util.tree_map(hax.zeros_like, differentiable_final_model)
            #params_grad = hax.shard(params_grad, self.parameter_axis_mapping)

            # (optional but recommended) constrain to best-effort FSDP sharding
            #_debug_weight_sharding("reward_grad_init/params_grad", params_grad)

            total_weights = 0.0
            total_batches = 0

            # 3) Use the SAME microbatched machinery you use for training to compute reward grads
            reward = 0.0
            for i, val_batch in tqdm(enumerate(val_loader), desc="Computing reward"):
                loss, p_grad, _metrics = reward_grad_step(  # <- microbatched + checkpointed
                    final_model, val_batch
                )
                # Accumulate on device (params_grad is sharded, so we don't replicate)
                params_grad = jax.tree_util.tree_map(jnp.add, params_grad, p_grad)
                total_weights += val_batch.tokens.array.shape[0]
                total_batches += 1
                reward += loss

                # NOTE(safety): The validation preview decode fetches a global jax.Array to host, which
                # breaks on multi-host TPU pods. Commented out for now; re-enable with proper multihost gather.
                # if i == 0:
                #     from transformers import GPT2Tokenizer
                #     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                #     input_ids = np.asarray(val_batch.tokens.array).astype(int)
                #     texts = tokenizer.batch_decode(
                #         input_ids.tolist(),
                #         skip_special_tokens=True,
                #         clean_up_tokenization_spaces=False,
                #     )
                #     for j, text in enumerate(texts):
                #         print(f"Example {i*input_ids.shape[0] + j}:")
                #         print(text)
                #         print("-" * 20)
                #         if j == 5:
                #             break

            print(f'Total weights: {total_weights}', flush=True)
            print(f'Total batches: {total_batches}', flush=True)

            # 4) Normalize once at the end
            if total_weights > 0:
                params_grad = jax.tree_util.tree_map(lambda g: g / total_weights, params_grad)

            # The original code differentiated wrt (final_model, final_opt_state).
            # final_opt_state isn't used in reward computation, so its gradient is zero.
            differentiable_opt_state, _ = eqx.partition(final_opt_state, eqx.is_inexact_array)
            opt_grad = jax.tree_util.tree_map(jnp.zeros_like, differentiable_opt_state)

            #_debug_weight_sharding("reward_grad_init/params_grad", params_grad)
            #_debug_weight_sharding("reward_grad_init/opt_grad", opt_grad)
            print(f"Final reward: {reward}")

            if train_only:
                return reward, None, self.dataset_ids, None, self.loss_masks_global

            print('***** COMPUTED FINAL STATE GRADIENTS *****\n\n\n', flush=True)

            ckpt_dir = fsspec_utils.join_path(metagrad_ckpt_dir, f"step-{self.config.num_train_steps}")

            # Save using TensorStore-based checkpointing to preserve sharding (TPU-safe, supports gs://)
            save_tree = {
                "params_grad": params_grad,
                "opt_grad": opt_grad,
                "metagrads": metagrads,
            }
            levanter.checkpoint.save_checkpoint(
                save_tree,
                step=self.config.num_train_steps,
                checkpoint_path=ckpt_dir,
                manager=None,
                is_temporary=False,
            )

        else:
            # Initialize final state and gradient buffers when resuming from metagrad checkpoints
            final_model, final_opt_state = state.model, state.opt_state
            final_model = inference_mode(final_model, True)
            final_model = self.mp.cast_to_compute(final_model)

            differentiable_final_model, _ = eqx.partition(final_model, eqx.is_inexact_array)
            differentiable_final_model = hax.shard(differentiable_final_model, self.parameter_axis_mapping)

            @eqx.filter_jit
            def zeros_like(m):
                return hax.shard(hax.tree_util.tree_map(hax.zeros_like, m), self.parameter_axis_mapping)

            params_grad = zeros_like(differentiable_final_model)

            differentiable_opt_state, _ = eqx.partition(final_opt_state, eqx.is_inexact_array)
            opt_grad = jax.tree_util.tree_map(jnp.zeros_like, differentiable_opt_state)

        self.construct_metagrad_helpers()

        # define custom x-axis for metagrads in wandb
        if wandb.run is not None:
            wandb.define_metric("rev_it")
            wandb.define_metric("metagrads/*", step_metric="rev_it")

        # TODO: get this from actual config
        dim = 32 # self.config.model.hidden_dim
        #grad_sharding = jax.tree.map(lambda x: _make_opt_sharding(x, dim, self.device_mesh), params_grad)
        grad_sharding = None
        #print('opt grad_sharding:', grad_sharding, flush=True)
        sharded_update = jax.tree_util.Partial(self.run_vjp_update, grad_sharding=grad_sharding)

        rev_it = self.config.num_train_steps - 1

        # Metagrad checkpointing setup
        print("[DBG] Creating metagrad checkpointer...", flush=True)
        checkpointer = self.config.checkpointer.create(
            self.run_id, axis_mapping=self.parameter_axis_mapping, mesh=self.device_mesh
        )
        print("[DBG] Metagrad checkpointer created", flush=True)
        #metagrad_ckpt_dir = fsspec_utils.join_path(self.checkpoint_path, "metagrad_checkpoints")
        #print(f"[DBG] Metagrad checkpoint directory: {metagrad_ckpt_dir}", flush=True)
        if (not resumed_from_metagrad) and self.config.metagrad_checkpoint_frequency > 0:
            print("[DBG] Ensuring metagrad checkpoint directory exists...", flush=True)
            fsspec_utils.mkdirs(metagrad_ckpt_dir)
            print("[DBG] mkdirs completed", flush=True)

        # Backward profiler setup
        if self.config.backward_profiler:
            print("[DBG] Backward profiler enabled", flush=True)
            backward_profile_path = str(self.config.log_dir / self.run_id / "backward_profiler")
            start_step = self.config.backward_profiler_start_step
            num_steps = self.config.backward_profiler_num_steps

            # The profiler will start when rev_it is at start_step and run for num_steps.
            # Since rev_it is decreasing, the profiler will stop at start_step - num_steps.
            # We need to ensure that this range is valid.
            if start_step - num_steps < -1:
                logger.warning(
                    f"Adjusting backward_profiler_num_steps from {num_steps} to {start_step + 1}"
                )
                num_steps = start_step + 1

        backward_profiler_started_by_us = False

        # Cache of batches for the currently reconstructed segment, if any
        data_batches_cache: Optional[dict[int, Any]] = None

        # Resume from metagrad checkpoint if available
        print(f"[DBG] Checking for existing metagrad checkpoints at {metagrad_ckpt_dir}...", flush=True)
        _has_metagrad_ckpt_dir = fsspec_utils.exists(metagrad_ckpt_dir)
        print(f"[DBG] metagrad_ckpt_dir exists: {_has_metagrad_ckpt_dir}", flush=True)
        if self.config.metagrad_checkpoint_frequency > 0 and _has_metagrad_ckpt_dir:
            try:
                # with fsspec, we can't glob, so we list and filter
                print("[DBG] Listing checkpoint directory...", flush=True)
                fs, _, paths = fsspec.get_fs_token_paths(metagrad_ckpt_dir)
                files = fs.ls(paths[0])
                print(f"[DBG] Found {len(files)} entries in checkpoint directory", flush=True)

                # Collect TensorStore dir checkpoints (step-<n>/metadata.json)
                dir_steps: list[int] = []
                for entry in files:
                    name = entry["name"] if isinstance(entry, dict) else str(entry)
                    base = os.path.basename(name)
                    if base.startswith("step-"):
                        meta_path = fsspec_utils.join_path(metagrad_ckpt_dir, base)
                        meta_json = fsspec_utils.join_path(meta_path, "metadata.json")
                        if fsspec_utils.exists(meta_json):
                            try:
                                dir_steps.append(int(base.split("-")[1]))
                            except Exception:
                                pass
            except (FileNotFoundError, IndexError):
                dir_steps = []

            if dir_steps:
                latest_step = min(dir_steps)
                logger.info(f"Resuming metagrad computation from step {latest_step}")
                ckpt_dir = fsspec_utils.join_path(metagrad_ckpt_dir, f"step-{latest_step}")
                print(f"[DBG] Loading metagrad checkpoint from {ckpt_dir}", flush=True)

                exemplar = {
                    "params_grad": params_grad,
                    "opt_grad": opt_grad,
                    "metagrads": metagrads,
                }
                loaded = levanter.checkpoint.load_checkpoint(
                    exemplar,
                    ckpt_dir,
                    discover_latest=False,
                    axis_mapping=self.parameter_axis_mapping,
                    mesh=self.device_mesh,
                )
                print("[DBG] Metagrad checkpoint load completed", flush=True)
                params_grad = loaded["params_grad"]
                opt_grad = loaded["opt_grad"]
                # Normalize metagrads to single-device array for consistent scatter updates
                metagrads = jnp.asarray(jax.device_get(loaded["metagrads"]))

                # Set rev_it to resume point directly (no dataloader fast-forward)
                steps_to_skip = (self.config.num_train_steps - 1) - (latest_step - 1) - 1
                if steps_to_skip > 0:
                    logger.info(f"Skipping {steps_to_skip} backward iterations to resume.")
                rev_it = latest_step - 1

                if rev_it == -1:
                    # Already fully computed
                    return 0., None, None, None

                # prefetch data_batches_cache since lost on resume
                segment_start = max([s for s in segment_starts if s <= rev_it])
                segment_loader = reversed_train_loader.iter_segment(segment_start)
                data_batches_cache = {}
                for i, example in enumerate(segment_loader):
                    data_batches_cache[segment_start + i] = example
                print(f"[DBG] rev_it set to {rev_it} after resume", flush=True)

        global_idx_start = (rev_it) * self.config.train_batch_size
        print(self.config)
        print(f"[DBG] global_idx_start initialized to {global_idx_start}", flush=True)

        # Prefetch the first checkpoint
        if rev_it > 0:
            print(f"[DBG] Prefetching checkpoint for step {rev_it - 1}...", flush=True)
            checkpointer.begin_checkpoint_loading(rev_it - 1, init_state)
            print("[DBG] Prefetch request issued", flush=True)

        while rev_it >= 0:
            print(f'------------------------------- Backward it: {rev_it} -------------------------------', flush=True)
            it_start_time = time.time()
            if self.config.backward_profiler:
                if rev_it == start_step:
                    try:
                        jax.profiler.stop_trace()
                    except RuntimeError:
                        # it's okay if the profiler was not running
                        pass
                    # Reset the trainer's profiler state to prevent forward profiler interference
                    self.profiler_running = False
                    _create_perfetto_link = self.config.profiler_perfetto_link and jax.process_index() == 0
                    logger.info(f"Starting backward profiler at rev_it {rev_it} for {num_steps} steps.")
                    _tpu_trace_mode = getattr(self.config, "profiler_tpu_trace_mode", None)
                    if _tpu_trace_mode:
                        try:
                            options = jax.profiler.ProfileOptions()
                            options.advanced_configuration = {"tpu_trace_mode": _tpu_trace_mode}
                            jax.profiler.start_trace(
                                backward_profile_path,
                                create_perfetto_link=_create_perfetto_link,
                                create_perfetto_trace=True,
                                options=options,
                            )
                        except TypeError:
                            jax.profiler.start_trace(
                                backward_profile_path,
                                create_perfetto_link=_create_perfetto_link,
                                create_perfetto_trace=True,
                            )
                    else:
                        jax.profiler.start_trace(
                            backward_profile_path,
                            create_perfetto_link=_create_perfetto_link,
                            create_perfetto_trace=True,
                        )
                    backward_profiler_started_by_us = True
                elif rev_it == start_step - num_steps:
                    if backward_profiler_started_by_us:
                        logger.info(f"Stopping backward profiler at rev_it {rev_it}.")
                        jax.profiler.stop_trace()
                        levanter.tracker.current_tracker().log_artifact(backward_profile_path, type="jax_profile")
                        barrier_sync()
                        backward_profiler_started_by_us = False

            #  rev_it is the iteration (step) to rewind to; we need the state prior to it (rev_it-1)
            # Ensure we have the segment's batches cached; if not, retrain the segment now.
            segment_start = max([s for s in segment_starts if s <= rev_it])

            checkpoint_start_time = time.time()
            state_pre = self._restore_step(init_state, rev_it - 1, checkpointer)
            if state_pre is not None:
                # ensure identical sharding as the forward path
                state_pre = hax.shard(state_pre, self.parameter_axis_mapping)
            jax.block_until_ready(state_pre)
            checkpoint_end_time = time.time()
            checkpoint_duration = checkpoint_end_time - checkpoint_start_time
            print(f"[Timing] Checkpoint restoration for step {rev_it-1} took {checkpoint_duration:.3f}s", flush=True)

            need_retrain = (state_pre is None) or (data_batches_cache is None) or (
                (data_batches_cache is not None) and (rev_it not in data_batches_cache)
            )
            safe_missing_flag = (False if data_batches_cache is None else (rev_it not in data_batches_cache))
            if need_retrain:
                logger.info(
                    f"[REPLAY] segment missing---state_pre is None: {state_pre is None}, "
                    f"data_batches_cache is None: {data_batches_cache is None}, "
                    f"rev_it not in data_batches_cache: {safe_missing_flag}"
                )
                print(f"[REPLAY] rev_it: {rev_it}, retraining segment from step {segment_start - 1}...", flush=True)
                restored = self._restore_step(init_state, segment_start - 1, checkpointer)
                if restored is None:
                    raise RuntimeError(f"Missing checkpoint at segment start-1 ({segment_start - 1}); cannot retrain segment.")

                state_pre, data_batches_cache = self.retrain_segment(restored, reversed_train_loader, segment_start, rev_it-1)
                #jax.block_until_ready(state_pre)

            # Use the restored pre-state for VJP computation
            state = state_pre

            # Retrieve the batch from cache for this step
            example = data_batches_cache[rev_it]
            # Safe multi-host logging: avoid embedding global jax.Arrays in f-strings.
            # Print metadata only (host step, shapes, sharding), which does not trigger host fetches.
            try:
                step_host = int(jax.device_get(state.step)) if hasattr(state.step, "dtype") else int(state.step)
            except Exception:
                step_host = int(state.step)
            tokens_arr = getattr(getattr(example, "tokens", None), "array", None)
            tokens_shape = getattr(tokens_arr, "shape", None)
            tokens_sharding = getattr(tokens_arr, "sharding", None)
            index_shape = getattr(getattr(example, "index", None), "shape", None)
            dsid_shape = getattr(getattr(example, "dataset_id", None), "shape", None)
            print(
                f"> backward step: {rev_it} | step: {step_host} | "
                f"tokens shape={tokens_shape} sharding={tokens_sharding} | "
                f"index shape={index_shape} | dataset_id shape={dsid_shape}",
                flush=True,
            )

            # Prefetch the next checkpoint while we compute the current one
            if rev_it > 1:
                checkpointer.begin_checkpoint_loading(rev_it - 2, state)

            batch_kwargs = {}
            Batch = _resolve_axis_in_tree((example, batch_kwargs), self.config.batch_axis_name)
            bs = Batch.size
            mbs = self.config.microbatch_size
            grad_accum_size = bs // mbs
            print(f"bs: {bs}, mbs: {mbs}, grad_accum_size: {grad_accum_size}", flush=True)

            pad_token_id = -100
            batch = lmexample_to_weighted_batch(example, grad_accum_size, global_idx_start, pad_token_id)
            global_idx_start -= bs

            self.rev_it = rev_it

            with jax.profiler.StepTraceAnnotation("backward_step", step_num=rev_it):
                params_grad, opt_grad, metagrads_for_batch_local, metagrads_indices, ds_ids_flat, local_idx_flat = self._vjp_update_step(
                    batch,
                    data_weight_vector,
                    inference_mode(state.model, True),
                    state.opt_state,
                    params_grad,
                    opt_grad,
                    sharded_update,
                )
                jax.block_until_ready((params_grad, opt_grad, metagrads_for_batch_local, metagrads_indices, ds_ids_flat, local_idx_flat))

            # Accumulate metagrads by scattering in true train order using global indices
            metagrads = metagrads.at[metagrads_indices].add(metagrads_for_batch_local)
            # Record dataset id and local index aligned to global positions
            print('len dataset_ids_global:', len(dataset_ids_global))
            print('len metagrads_indices:', len(metagrads_indices))
            print('len ds_ids_flat:', len(ds_ids_flat), flush=True)
            dataset_ids_global = dataset_ids_global.at[metagrads_indices].set(ds_ids_flat)
            local_indices_global = local_indices_global.at[metagrads_indices].set(local_idx_flat)

            # if metagrads exceed threshold / blow up, terminate early
            if jnp.any(jnp.abs(metagrads) > 1e6):
                print('Metagrads blew up! Terminating...', flush=True)
                raise ValueError('Metagrads blew up!')

            if self.config.metagrad_checkpoint_frequency > 0 and (rev_it % self.config.metagrad_checkpoint_frequency == 0):
                logger.info(f"Checkpointing metagrad computation at step {rev_it}")
                ckpt_dir = fsspec_utils.join_path(metagrad_ckpt_dir, f"step-{rev_it}")

                # Save using TensorStore-based checkpointing to preserve sharding (TPU-safe, supports gs://)
                save_tree = {
                    "params_grad": params_grad,
                    "opt_grad": opt_grad,
                    "metagrads": metagrads,
                }
                levanter.checkpoint.save_checkpoint(
                    save_tree,
                    step=rev_it,
                    checkpoint_path=ckpt_dir,
                    manager=None,
                    is_temporary=False,
                )

                # After saving the current checkpoint, delete the previous one to save space.
                # The reverse iteration goes down, so the previous checkpoint is at a higher step number.
                prev_ckpt_step = rev_it + self.config.metagrad_checkpoint_frequency
                if prev_ckpt_step < self.config.num_train_steps:
                    prev_ckpt_path = fsspec_utils.join_path(metagrad_ckpt_dir, f"step-{prev_ckpt_step}")
                    try:
                        fs, _, (base_path,) = fsspec.get_fs_token_paths(str(metagrad_ckpt_dir))
                        if fs.exists(prev_ckpt_path):
                            logger.info(f"Deleting old metagrad checkpoint at step {prev_ckpt_step}")
                            fs.rm(prev_ckpt_path, recursive=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete old metagrad checkpoint {prev_ckpt_path}: {e}")

            # After finishing rev_it, optionally delete the forward training checkpoint for step-{rev_it}
            # to save space as we move backward. Keep the last forward checkpoint (step-{num_train_steps-1}).
            last_keep = getattr(self, "_last_forward_ckpt_to_keep", self.config.num_train_steps - 1)
            if 0 <= rev_it and rev_it != last_keep:
                try:
                    checkpointer._rm_checkpoint(f"step-{rev_it}")
                except Exception as e:
                    logger.warning(f"Failed to schedule deletion for forward checkpoint step-{rev_it}: {e}")

            iter_end_time = time.time()
            iter_duration = iter_end_time - it_start_time
            print(f"[Timing] Iteration {rev_it} took {iter_duration:.3f}s", flush=True)

            #self.debug_print('> iter:', rev_it, 'global_idx_start:', global_idx_start, 'example.tokens:', example.tokens)
            rev_it -= 1

        print(f"metagrads: {metagrads[:200]}")
        print(f"dataset_ids_global: {dataset_ids_global[:200]}")
        print(f"local_indices_global: {local_indices_global[:200]}")

        return reward, metagrads, dataset_ids_global, local_indices_global, self.loss_masks_global


    def retrain_segment(self, state, reversed_train_loader, segment_start, target_step):
        """Retrain forward over a segment and save a checkpoint for each step.

        Only checkpoint saving is performed (no other hooks). Also caches the
        data batches for this segment for later reuse during backward VJP.
        """
        if self._checkpointer is None:
            # Should be set in _add_default_hooks; fail loudly if missing.
            raise RuntimeError("Internal error: checkpointer handle is not initialized.")

        # The iterator returned by reversed_train_loader.iter_segment already
        # performs background prefetching via DataLoaderIterator; do not wrap it
        # in a new DataLoader (which expects an AsyncDataset).
        segment_loader = reversed_train_loader.iter_segment(segment_start)
        data_batches_cache: dict[int, Any] = {}

        # Iterate forward within the segment and reproduce training steps without running hooks.
        current_step = int(segment_start) - 1

        for segment_idx, example in enumerate(segment_loader):
            '''
            # Refresh the step scalar to avoid reusing a potentially donated int32[] from prior calls
            try:
                step_dtype = state.step.dtype  # type: ignore[attr-defined]
            except Exception:
                step_dtype = jnp.int32
            state = dataclasses.replace(state, step=jnp.array(current_step, dtype=step_dtype))
            # Run the JITed train step without hooks
            example_for_train = self._duplicate_pytree_arrays(example)
            loss, state, _metrics, _ = self._jit_train_step_fn_no_hook_nodonate(state, (example_for_train,), {})
            # Ensure previous async saves are finished, then remove any existing checkpoint dir for this step
            try:
                self._checkpointer.wait_until_finished()
                try:
                    save_step = int(jax.device_get(state.step))  # type: ignore[attr-defined]
                except Exception:
                    save_step = current_step + 1
                from levanter.checkpoint import _get_fs_and_plain_path  # local import to avoid cycles
                fs, plain_base = _get_fs_and_plain_path(self._checkpointer.base_path)
                ckpt_dir_plain = os.path.join(plain_base, f"step-{save_step}")
                if fs.exists(ckpt_dir_plain):
                    fs.rm(ckpt_dir_plain, recursive=True)
            except Exception:
                # Best-effort cleanup; proceed even if removal fails
                pass
            '''
            data_batches_cache[segment_start + segment_idx] = self._duplicate_pytree_arrays(example)
            if segment_start + segment_idx > target_step:
                break
            result = self._jit_train_step_fn_no_hook(state, (example,), {})
            state = result.new_state
            print(' > retrained step:', segment_start + segment_idx, flush=True)
            # Force-save a permanent checkpoint for this step only
            info = StepInfo(state, float(result.loss), 0.0)
            self._checkpointer.on_step(info, force=True)
            # Cache the batch by its global step index
            current_step += 1

        return state, data_batches_cache


    def _reindex_interval_data(
        self,
        batch_data: Dict[str, hax.NamedArray]
    ) -> Tuple[Dict[str, hax.NamedArray], jnp.ndarray, jnp.ndarray]:
        """Re-indexes batch data for data weights"""
        original_indices = batch_data["index"]
        # Perform unique/inverse on-device to avoid fetching non-addressable shards to host
        indices_jax = original_indices.array
        indices_flat = jnp.ravel(indices_jax)
        unique_indices, inverse_indices = jnp.unique(indices_flat, return_inverse=True)

        local_indices_jnp = jnp.reshape(inverse_indices, indices_jax.shape).astype(original_indices.dtype)
        local_indices = hax.named(local_indices_jnp, original_indices.axes)

        batch_data_local = batch_data.copy()
        batch_data_local["index"] = local_indices
        return batch_data_local, unique_indices, inverse_indices


    def _vjp_update_step(
        self,
        batch_data: Dict[str, jnp.ndarray],
        data_weight_vector: jnp.ndarray,
        initial_params: Any,
        initial_opt_state: Any,
        params_grad: Any,
        opt_grad: Any,
        sharded_update: Any,
        use_wandb: bool = True,
    ) -> Tuple[Any, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Performs a single VJP update step, walking backwards from the final iteration to the first, passing partial derivatives backward and computing metagrads for the batch.

        Returns:
            Tuple of (updated_params_grad, updated_opt_grad, metagrads_for_batch, unique_indices)
        """

        # sanity: list what's matchable
        #if jax.process_index() == 0:
        #    list_weight_paths(initial_params, limit=50)

        data_processing_start = time.time()
        # Use global train-order indices (already in batch_data['index']) to align with data_weight_vector
        batch_data_local = batch_data
        flat_indices = jnp.ravel(batch_data_local["index"].array)
        ds_ids_flat = jnp.ravel(batch_data_local.get("dataset_id").array) if "dataset_id" in batch_data_local else jnp.full_like(flat_indices, -1)
        local_idx_flat = jnp.ravel(batch_data_local.get("local_index").array) if "local_index" in batch_data_local else jnp.full_like(flat_indices, -1)
        # Gather weights in the exact (global) order, no unique/aggregation
        interval_weights = data_weight_vector.at[flat_indices].get()
        #print(f"Prefetch and reindex took {time.time() - data_processing_start:.3f}s")

        vjp_start = time.time()

        # choose which layer to inspect
        LAYER = "embeddings.token_embeddings.weight" #"attn.c_attn.weight"   # or "attn.c_proj.weight", "mlp.c_fc.weight", "token_embeddings.weight"

        # --- BEFORE sharded_update ---
        #log_layer_sharding("pre/initial_params", initial_params, LAYER)
        #log_layer_sharding("pre/params_grad",    params_grad,     LAYER)
        #log_layer_sharding("pre/opt_grad",       opt_grad,        LAYER)

        avg_grad_grad, train_state_grad, loss = eqx.filter_jit(sharded_update, donate='all-except-first')( # donate_argnums=(4, 5))(
            (initial_params, initial_opt_state, interval_weights,
            batch_data_local), params_grad, opt_grad
        )

        params_grad_new, opt_grad_new = train_state_grad

        metagrads_for_batch_local, _, _ = eqx.filter_jit(self.run_vjp_grad)( #), donate_argnums=(0, 1, 4))(
            initial_params, initial_opt_state, interval_weights,
            batch_data_local, avg_grad_grad
        )

        # We now have per-example metagrads in exact train order (flattened batch order).
        # Scatter-add directly into the global metagrads vector using global indices.
        if metagrads_for_batch_local.ndim == 1:
            metagrads_indices = flat_indices
        else:
            # fallback: flatten any trailing dims
            metagrads_for_batch_local = metagrads_for_batch_local.reshape((-1,))
            metagrads_indices = flat_indices

        tree_statistics(params_grad_new, "params_grad")

        if use_wandb and wandb.run is not None:
            params_grad_norm = optax.global_norm(params_grad_new)
            opt_grad_norm = optax.global_norm(opt_grad_new)
            wandb.log({
                "rev_it": self.rev_it,
                "metagrads/mean": jnp.mean(metagrads_for_batch_local),
                "metagrads/std": jnp.std(metagrads_for_batch_local),
                "metagrads/max": jnp.max(metagrads_for_batch_local),
                "metagrads/min": jnp.min(metagrads_for_batch_local),
                "metagrads/params_grad_norm": params_grad_norm,
                "metagrads/opt_grad_norm": opt_grad_norm,
                "metagrads/processing_time": time.time() - data_processing_start,
                #"metagrads/examples_per_second": (self.config.train_batch_size * self.config.grad_accum_size) / (time.time() - data_processing_start),
                #"metagrads/tokens_per_second": (self.config.train_batch_size * self.config.grad_accum_size * batch_data_local["input_ids"].shape[-1]) / (time.time() - data_processing_start)
            }, commit=True)

        return params_grad_new, opt_grad_new, metagrads_for_batch_local, metagrads_indices, ds_ids_flat, local_idx_flat


    def _add_default_hooks(self):
        from levanter import callbacks

        self.add_hook(levanter.callbacks.pbar_logger(total=self.config.num_train_steps), every=1)
        self.add_hook(levanter.callbacks.log_step_info(self.config.num_train_steps), every=1)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = self.config.checkpointer.create(
            self.run_id, axis_mapping=self.parameter_axis_mapping, mesh=self.device_mesh
        )
        # Save handle for manual invocation during segment retraining
        self._checkpointer = checkpointer

        # Gate checkpointer to only run at allowed steps if provided.
        # Always honor force=True (e.g., initial step -1), regardless of gating.
        from levanter.callbacks._core import Callback
        class _GatedCheckpointerCallback(Callback):
            def __init__(self, trainer, checkpointer):
                self._trainer = trainer
                self._checkpointer = checkpointer
            def on_step(self, info: StepInfo, force: bool = False):
                allowed = self._trainer._checkpoint_allowed_steps
                step = int(info.step)
                if force:
                    self._checkpointer.on_step(info, force=True)
                elif allowed is None or step in allowed:
                    self._checkpointer.on_step(info, force=False)

        self.add_hook(_GatedCheckpointerCallback(self, checkpointer), every=1)  # checkpointer manages its own frequency

        # Add watch callback if configured
        if self.config.watch.is_enabled:
            watch_callback = self.config.watch.build()
            self.add_hook(watch_callback, every=self.config.watch.interval)
            #print(f"*** DEBUG: Added watch callback with interval: {self.config.watch.interval}")

        if self.config.profiler:
            profile_path = self.config.log_dir / self.run_id / "profiler"
            total_prof_steps = self.config.profiler_num_steps
            if total_prof_steps + self.config.profiler_start_step > self.config.num_train_steps:
                logger.warning(
                    f"Adjusting profiler_total_steps from {total_prof_steps} to"
                    f" {self.config.num_train_steps - self.config.profiler_start_step}"
                )
                total_prof_steps = self.config.num_train_steps - self.config.profiler_start_step
            self.add_hook(
                callbacks.profile(
                    self,
                    str(profile_path),
                    self.config.profiler_start_step,
                    total_prof_steps,
                    self.config.profiler_perfetto_link,
                ),
                every=1,
            )


    def add_eval_hook(self, eval_dataset, name: Optional[str] = None):
        from levanter import callbacks

        eval_loader = self.data_loader(eval_dataset, self.EvalBatch)

        if eval_loader and (self.config.max_eval_batches is None or self.config.max_eval_batches > 0):

            @eqx.filter_jit
            def eval_loss(model, *batch, **batch_kwargs):
                # Normalize eval pathway to avoid changing cache keys across steps.
                # Set inference mode and dtype inside the JIT rather than swapping models outside.
                m = inference_mode(model, True)
                m = self.mp.cast_to_compute(m)
                return self._raw_loss_function(m, *batch, **batch_kwargs).mean()

            self.add_hook(
                callbacks.compute_validation_loss(
                    eval_loss,
                    eval_loader,
                    max_batches=self.config.max_eval_batches,
                    name=name,
                ),
                every=self.config.steps_per_eval,
            )

    def data_loader(self, dataset: AsyncDataset[X], batch: Optional[hax.Axis] = None) -> DataLoader[X]:
        """Creates a data loader for the given dataset and batch axis.

        Args:
            dataset (AsyncDataset): the dataset to load
            batch (Optional[hax.Axis]): the batch axis. If None, uses the trainer batch axis (and schedule, if applicable)

        Returns:
            DataLoader: the data loader
        """
        if batch is not None:
            batch_name = batch.name
            batch_size = batch.size
        else:
            batch_name = self.config.batch_axis_name
            batch_size = self.config.train_batch_size

        return DataLoader(
            dataset,
            batch_size=batch_size,
            max_buffered_batches=128,
            mesh=self.device_mesh,
            axis_resources=self.compute_axis_mapping,
            prefetch_size=32,
            batch_axis_name=batch_name,
            allow_nondivisible_batch_size=self.config.allow_nondivisible_batch_size,
            #num_train_steps=self.config.num_train_steps,
        )

    @cached_property
    def _jit_train_step_fn(self):
        return named_jit(
            self._train_step,
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(True,),
        )

    @cached_property
    def _jit_train_step_fn_no_hook(self):
        return named_jit(
            functools.partial(self._train_step, _no_hooks=True),
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(True,),
        )

    @cached_property
    def _jit_train_step_fn_no_hook_nodonate(self):
        """Non-donating variant for segment retraining to avoid reusing donated arrays across calls."""
        return named_jit(
            functools.partial(self._train_step, _no_hooks=True),
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(),
        )

    def _train_step(self, state: S, batch, batch_kwargs, _no_hooks=False) -> TrainStepResult[S]:
        key, new_key = jax.random.split(state.training_key)
        model = inference_mode(state.model, False)

        def loss_fn(model, *batch, key, **kwargs):
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                # we have to here because of how microbatching works
                unreduced_loss = self._raw_loss_function(
                    model, *batch, key=key, reduction=None, reduction_axis=(), **kwargs
                )
                if self.data_weight_vector is not None:
                    # assume the batch has an "index" field
                    batch_example = batch[0]
                    #print(f"Batch example: {batch_example}")
                    weights = self.data_weight_vector[batch_example.index]
                    #debug.print("batch_example.index: {}", batch_example.index)
                    #print(unreduced_loss.shape)
                    unreduced_loss = hax.mean(unreduced_loss, axis="position")
                    # multiply on the raw arrays and then wrap back in a NamedArray.
                    # this is necessary because haliax doesn't support this kind of indexing on a NamedArray,
                    # and also doesn't support broadcasting from a raw jax array in a jitted context.
                    weighted_loss_array = unreduced_loss.array * weights
                    unreduced_loss = hax.named(weighted_loss_array, unreduced_loss.axes)

                return hax.mean(unreduced_loss).scalar(), {}

        # Returns (loss, grads, wrapped_metrics) where wrapped_metrics is Dict[str, Metric]
        loss, grads, wrapped_metrics = self._compute_gradients_microbatched(
            loss_fn, model, *batch, **batch_kwargs, key=key
        )

        # Sophia needs to be able to access the loss function in the optimizer
        def obj_fun(trainable_model):
            model = eqx.combine(trainable_model, state.model)
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                result = self._raw_loss_function(model, *batch, **batch_kwargs, key=key)
                # result is (loss, metrics) tuple
                loss_for_opt, _metrics = result
                return loss_for_opt.scalar()

        new_state, updates = state.take_step(grads, obj_fun=obj_fun, loss=loss, key=new_key)
        #metrics["update_norm"] = optax.global_norm(updates)
        new_state = hax.shard(new_state, self.parameter_axis_mapping)

        hook_infos = None
        if not _no_hooks:
            with hax.axis_mapping(self.parameter_axis_mapping):
                jit_info: InsideJitInfo = InsideJitInfo(grads=grads, updates=updates)
                hook_infos = self.hooks.run_jit_hooks(state, jit_info, force=False)

        # extract plain metrics and prefix their keys
        plain_metrics = unwrap_metrics(wrapped_metrics)
        train_metrics = {f"train/{k}": v for k, v in plain_metrics.items()}

        result = TrainStepResult(
            loss=loss,
            new_state=new_state,
            loss_metrics=train_metrics,
            hook_infos=hook_infos,
        )
        return hax.shard_with_axis_mapping(result, self.parameter_axis_mapping)

    def _compute_gradients_microbatched(
        self, loss_fn: WrappedLossFunction, model: M, *batch, **batch_kwargs
    ) -> Tuple[Scalar, M, Dict[str, Metric]]:
        """
        Compute gradients, optionally with microbatching.
        Returns (loss, grads, dict[str, Metric]).
        """
        Batch = _resolve_axis_in_tree((batch, batch_kwargs), self.config.batch_axis_name)

        # loss_fn always returns (loss, metrics), so has_aux=True
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        mbs = self.config.microbatch_size
        if mbs is not None:
            grad_fn = microbatched(
                grad_fn,
                Batch,
                mbs,
                self.parameter_axis_mapping,
                self.compute_axis_mapping,
            )

        with hax.axis_mapping(self.compute_axis_mapping):
            (loss, metrics), grads = grad_fn(model, *batch, **batch_kwargs)

        return loss, grads, metrics

    def write_artifact(self, name: str, artifact: Any, type: Optional[str] = None):
        """Saves an artifact to disk (in the run dir) and logs it to the tracker."""
        dir = self.config.log_dir / self.run_id / "artifacts"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        artifact_path = dir / name

        if isinstance(artifact, str):
            with fsspec.open(str(artifact_path), "w", compression="infer") as f:
                f.write(artifact)
        else:
            with fsspec.open(str(artifact_path), "wb", compression="infer") as f:
                f.write(artifact)

        self.tracker.log_artifact(artifact_path, name=name, type=type)

    def _maybe_save_jaxpr(self, name: str, fn, *args, **kwargs):
        logged = False
        if self.config.log_jaxprs and name not in self._logged_jaxprs:
            jaxpr, _, _ = eqx.filter_make_jaxpr(fn)(*args, **kwargs)
            pretty = jaxpr.pretty_print(name_stack=True, use_color=False)
            self.write_artifact(f"{name}.jaxpr.txt.gz", pretty, type="jaxpr")
            logged = True

        if self.config.log_xla_hlo and name not in self._logged_jaxprs:
            hlo = fn.lower(*args, **kwargs).as_text("stablehlo")
            self.write_artifact(f"{name}.hlo.txt", hlo, type="hlo")
            logged = True

        if logged:
            self._logged_jaxprs.add(name)

        return fn(*args, **kwargs)

    def _duplicate_pytree_arrays(self, tree):
        """Deep-copies JAX arrays within a pytree to ensure fresh device buffers.

        This avoids reusing inputs that may have been previously donated across JIT calls.
        """
        def _copy_leaf(x):
            # jax.Array supports .copy() which preserves sharding and device placement.
            try:
                import jax
                import jax.numpy as jnp  # noqa: F401
                import numpy as np  # noqa: F401
                import haliax as hax  # local import to avoid cycle at module import time
            except Exception:
                # Should never happen here, but safeguard to avoid NameError during teardown.
                return x

            # NamedArray: copy underlying array and re-wrap to preserve axes
            if isinstance(x, hax.NamedArray):
                array_copied = x.array.copy() if isinstance(x.array, jax.Array) else jnp.array(x.array, copy=True)
                return hax.named(array_copied, x.axes)

            if isinstance(x, jax.Array):
                return x.copy()
            # Fall back to JAX array copy if it behaves like an ndarray
            if isinstance(x, np.ndarray):
                return jnp.array(x, copy=True)
            return x

        return jax.tree_util.tree_map(_copy_leaf, tree)

    def _restore_step(self, exemplar_state, step_to_load: int, checkpointer: "Checkpointer"):
        #if step_to_load < 0:
        #    # Returning a deep copy avoids using potentially donated buffers from the exemplar state.
        #    return self._duplicate_pytree_arrays(exemplar_state)

        # Try to retrieve a prefetched checkpoint if available. If the prefetch failed because
        # the checkpoint doesn't exist, treat it as a cache miss and fall back to direct load.
        try:
            restored_state = checkpointer.get_loaded_checkpoint()
        except FileNotFoundError:
            restored_state = None

        if restored_state is None:
            # Direct load of the specific step; if it doesn't exist, return None so caller can retrain.
            try:
                return checkpointer.load_checkpoint(
                    exemplar_state,
                    path=os.path.join(checkpointer.base_path, f"step-{step_to_load}"),
                    discover_latest=False,
                )
            except FileNotFoundError:
                return None

        return restored_state


def _initialize_global_tracker(config, run_id):
    if isinstance(config, Sequence):
        tracker = levanter.tracker.CompositeTracker([c.init(run_id) for c in config])
    else:
        tracker = config.init(run_id)

    levanter.tracker.set_global_tracker(tracker)


@dataclass
class TrainerConfig:
    seed: int = 0  # random seed
    mp: jmp.Policy = jmp.get_policy("f32")  # mixed precision policy
    quantization: Optional[QuantizationConfig] = None
    model_averaging: ModelAveragingConfig | None = None

    wandb: Optional[WandbConfig] = None
    log_dir: Path = Path("logs/")
    id: Optional[str] = None  # run id. if None, will be set to a random string

    tracker: TrackerConfig | Tuple[TrackerConfig, ...] = field(default_factory=WandbConfig)
    watch: WatchConfig = WatchConfig()

    # TODO: refactor callbacks
    profiler: bool = False
    profiler_start_step: int = 3
    profiler_num_steps: int = 2
    profiler_perfetto_link: bool = False
    profiler_tpu_trace_mode: Optional[str] = None

    # A separate profiler for the backward pass
    backward_profiler: bool = False
    backward_profiler_start_step: int = 3
    backward_profiler_num_steps: int = 2

    log_jaxprs: bool = True
    """Whether to log the jaxpr of the training step. This is useful for debugging and understanding the model."""
    log_xla_hlo: bool = True
    """Whether to log the XLA HLO of the training step. This is useful for debugging and understanding the model."""

    # Debug printing controls
    debug_print_loss_mask: bool = False
    """If True, print loss mask stats for each example (host-side) and inside JIT (device-side)."""
    debug_print_loss_mask_values: bool = False
    """If True, also print the full loss mask values on the host. Potentially very verbose."""

    # helpful checks
    crash_on_nan: bool = True
    crash_on_inf: bool = True

    # config related to partitioning
    mesh: MeshConfig = MeshConfig()

    @property
    def batch_axis_name(self) -> str | None:
        return self.mesh.batch_axis_name

    # Config related to batch sizes
    train_batch_size: int | IntSchedule = 512
    per_device_parallelism: int = -1
    """how many examples to process in parallel on each device. -1 (default) means train_batch_size/num_devices"""

    per_device_eval_parallelism: int = -1
    """how many examples to process in parallel on each device. -1 (default) means same as per_device_parallelism"""

    allow_nondivisible_batch_size: bool = False
    """
    Allow batch sizes to be non-divisible by the number of devices (or data axis size).

    This is typically used when you want a specific batch size but have a weird number of devices.
    """

    # Config related to duration
    num_train_steps: int = 400_000  # number of training steps
    steps_per_eval: int = 1_000  # how often to evaluate
    max_eval_batches: Optional[int] = None  # max number of batches to evaluate on. None means all batches

    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    load_checkpoint: Optional[bool] = None
    """if None (default), we'll load a checkpoint if it exists. If true, we must load a checkpoint"""
    load_checkpoint_path: Optional[str] = None
    """can be a parent (to find latest) or a specific checkpoint. if None, will set to checkpointer.base_path."""
    initialize_from: Optional[str] = None  # Levanter trainer checkpoint to initialize from
    """Load and continue training from a checkpoint. If None, will initialize from model_init."""
    allow_partial_checkpoint: bool = False
    """If True, we allow loading a checkpoint that doesn't have all the parameters in the model.
        Missing parameters are initialized from the model_init function."""

    load_debug_weights: bool = False

    metagrad_checkpoint_frequency: int = 0
    """Frequency to checkpoint metagrad computation. 0 to disable."""

    # Size of segments for metagrad replay. If > 0, reversed replay will
    # treat training as segments starting at [0, S, 2*S, ...], up to
    # num_train_steps. This is separate from checkpoint cadence.
    metagrad_segment_size: int = 0

    jax_config: Mapping[str, JsonAtom] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_JAX_CONFIG)
    )  # config to pass to jax.config.update
    jax_compilation_cache_dir: Optional[str] = None

    distributed: DistributedConfig = DistributedConfig()
    ray: RayConfig = field(default_factory=RayConfig)

    # whether or not to require an accelerator (e.g. TPU or GPU).
    # default depends on the platform: on macos False, else True
    require_accelerator: Optional[bool] = None

    # whether or not to shutdown the tpu at exit. If a float, shutdown after that many seconds. True = 5 minutes
    shutdown_at_exit: Union[bool, float] = False

    @property
    def TrainBatch(self):
        if not isinstance(self.train_batch_size, int):
            raise ValueError("TrainBatch is only valid for a single batch size. Use batch_axis_at_step instead")
        return Axis(self.batch_axis_name, self.train_batch_size)

    @cached_property
    def batch_schedule(self):
        return BatchSchedule(self.train_batch_size)

    def batch_axis_at_step(self, step: int) -> Axis:
        bs = value_at_step(self.train_batch_size, step)
        return Axis(self.batch_axis_name, bs)

    @property
    def EvalBatch(self):
        return Axis(self.batch_axis_name, self.eval_batch_size)

    @property
    def microbatch_size(self) -> int | None:
        if self.per_device_parallelism < 0:
            return None
        return self.per_device_parallelism * self.data_axis_size

    def __post_init__(self):
        if self.wandb is not None:
            warnings.warn(
                "wandb is deprecated. use tracker with type wandb instead",
                DeprecationWarning,
            )
            self.tracker = self.wandb

    def initialize(self):
        """Initializes jax, logging, setting the run name/id in the process"""
        self._initialize_jax_config()
        # Can't do full logging setup until we've initialized jax b/c we use jax for rank id
        pylogging.basicConfig(level=pylogging.WARNING)
        self.distributed.initialize()
        self._validate_and_set_defaults()

        # Debug: summarize partitioning topology and mappings
        try:
            def _mapping_contains(mapping, axis_name):
                if isinstance(mapping, tuple):
                    return axis_name in mapping
                return mapping == axis_name

            model_axis = "model"
            embed_mapping = self.parameter_axis_mapping.get("embed")
            tensor_parallel_axes = sorted(
                logical
                for logical, physical in self.compute_axis_mapping.items()
                if _mapping_contains(physical, model_axis)
            )
            print(
                f"[TrainerConfig DBG] axes: batch_axis_name={self.batch_axis_name} "
                f"fsdp_param_axis={embed_mapping} tensor_parallel_axes={tensor_parallel_axes}",
                flush=True,
            )
            print(
                f"[TrainerConfig DBG] sizes: replica_ici={self.replica_ici_axis_size} data_ici={self.data_ici_axis_size} "
                f"model={self.model_axis_size} replica_dcn={self.replica_dcn_axis_size} data_dcn={self.data_dcn_axis_size} "
                f"devices={jax.device_count()}",
                flush=True,
            )
            print(
                f"[TrainerConfig DBG] mappings: compute={self.compute_axis_mapping} params={self.parameter_axis_mapping}",
                flush=True,
            )
        except Exception:
            pass

        id = self._maybe_set_id()
        levanter.utils.logging.init_logging(self.log_dir, f"{id}.log")
        _initialize_global_tracker(self.tracker, id)

        self.ray.initialize()

        if self.require_accelerator is None:
            self.require_accelerator = not sys.platform.startswith("darwin")

        if self.require_accelerator:
            if jax.default_backend() == "cpu":
                raise RuntimeError("No accelerator found. Please run on a TPU or GPU.")

        if self.shutdown_at_exit is not False:
            if isinstance(self.shutdown_at_exit, bool):
                self.shutdown_at_exit = 5.0 * 60
            logger.info(f"At end of run, shutting down TPU VM in {self.shutdown_at_exit} seconds")
            atexit.register(cloud_utils.shutdown_tpu_vm, self.shutdown_at_exit)

        # Ensure the profiler is stopped at exit.
        def safe_stop_trace():
            try:
                jax.profiler.stop_trace()
            except RuntimeError:
                pass

        atexit.register(safe_stop_trace)

    @property
    def device_mesh(self) -> Mesh:
        ici, dcn = self.mesh.axis_shapes(jax.device_count(), self.num_slices)
        return create_mesh_from_axis_specs(ici_axes=ici, dcn_axes=dcn)

    def use_device_mesh(self) -> ContextManager[None]:
        """
        Context manager that sets the device mesh for jax, using Haliax's wrapper.

        In recent jax, this is the same as `jax.set_mesh(self.device_mesh)`, but we use Haliax's wrapper for
        compatibility with older jax versions.
        """
        return haliax.partitioning.set_mesh(self.device_mesh)

    @property
    def eval_batch_size(self):
        return self.per_device_eval_parallelism * self.data_axis_size

    @cached_property
    def num_slices(self):
        """number of nodes"""
        return max(getattr(device, "slice_index", 0) for device in jax.devices()) + 1

    @property
    def num_devices_per_slice(self):
        """number of devices within a slice"""
        return jax.device_count() // self.num_slices

    @cached_property
    def mesh_axis_specs(self) -> List[str]:
        """Materialized mesh axis names; validates mesh config."""
        ici, dcn = self.mesh.axis_shapes(jax.device_count(), self.num_slices)
        return list(ici.keys() | dcn.keys())

    @cached_property
    def _mesh_axis_totals(self) -> Dict[str, int]:
        ici, dcn = self.mesh.axis_shapes(jax.device_count(), self.num_slices)
        return {name: ici.get(name, 1) * dcn.get(name, 1) for name in set(ici) | set(dcn)}

    def _axis_size(self, axis: str) -> int:
        return self._mesh_axis_totals.get(axis, 1)

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        batch_map = self.compute_axis_mapping.get(self.batch_axis_name, self.compute_axis_mapping.get("batch"))
        if batch_map is None:
            raise ValueError(
                f"No mapping found for batch axis {self.batch_axis_name} in compute axis mapping."
                f"In Levanter, you must specify a mapping for the batch axis. For instance, the default is:"
                """
                             mesh:
                               compute_mapping:
                                 batch: [replica_dcn, replica, data]
                             """
            )
        axes = batch_map if isinstance(batch_map, tuple) else (batch_map,)
        prod_size = 1
        for ax in axes:
            prod_size *= self._axis_size(ax)
        return prod_size

    @property
    def compute_axis_mapping(self) -> ResourceMapping:
        """Mapping from logical axis to physical axis for compute."""
        return self.mesh.resolved_compute_mapping

    @property
    def parameter_axis_mapping(self) -> ResourceMapping:
        return self.mesh.resolved_param_mapping

    def _initialize_jax_config(self):
        for key, value in self.jax_config.items():
            jax.config.update(key, value)

        if self.jax_compilation_cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", self.jax_compilation_cache_dir)

    def _maybe_set_id(self):
        # always do this so we don't get weird hangs if the id isn't set right
        # for random ids, we want to ensure that all hosts have the same id
        # NB: do NOT use the run seed here. we want the run id to be independent of the seed
        seed = np.random.randint(0, 2**31 - 1)
        seed = multihost_utils.broadcast_one_to_all(jax.numpy.array(seed, dtype=np.int32)).item()

        # RUN ID comes from a few places: the config, the environment, or wandb, or a random string
        if self.id is None:
            # TODO: this doesn't work with wandb sweeps. need to reconcile when we merge
            if "RUN_ID" in os.environ:
                self.id = os.environ["RUN_ID"]
            elif self.wandb is not None and self.wandb.id is not None:
                self.id = self.wandb.id
            else:
                # wandb run ids are 8 characters [a-z0-9], which we'll emulate here
                # we also want to ensure that all hosts have the same run id
                # we do this by syncing a random seed across all hosts and then using that to generate the run id
                gen = np.random.default_rng(seed)
                self.id = "".join(gen.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 8))

            logger.info(f"Setting run id to {self.id}")

        return self.id

    # we can't do this in post_init because we don't want to call jax.device_count before calling distributed.initialize
    def _validate_and_set_defaults(self):
        # Validate mesh early
        _ = self.mesh_axis_specs

        if self.train_batch_size == -1 and self.per_device_parallelism == -1:
            raise ValueError("either train_batch_size or per_device_parallelism must be specified (not -1)")

        if self.per_device_parallelism == -1:
            if isinstance(self.train_batch_size, int):
                self.per_device_parallelism = self.train_batch_size // self.data_axis_size
            else:
                logger.info(
                    "per_device_parallelism is not set and train_batch_size is not an int. "
                    "Not using microbatching and just maxing out the per_device_parallelism."
                )

        if self.train_batch_size == -1:
            self.train_batch_size = self.per_device_parallelism * self.data_axis_size

        # validate size of per_device_parallelism
        if self.per_device_parallelism != -1:
            if isinstance(self.train_batch_size, Sequence):
                for phase in self.train_batch_size:
                    assert isinstance(phase, ScheduleStep)
                    if phase.value % (self.per_device_parallelism * self.data_axis_size) != 0:
                        raise ValueError(
                            f"At step {phase.start}, train_batch_size ({phase.value}) must be divisible by "
                            "per_device_parallelism * data_axis_size "
                            f"({self.per_device_parallelism}, {self.data_axis_size})"
                        )
            elif self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
                raise ValueError(
                    f"train_batch_size ({self.train_batch_size}) must be divisible by per_device_parallelism *"
                    f" data_axis_size ({self.per_device_parallelism}, {self.data_axis_size})"
                )

        if self.per_device_eval_parallelism == -1:
            if self.per_device_parallelism == -1:
                tbs = max(distinct_values(self.train_batch_size))
                self.per_device_eval_parallelism = (
                    _round_to_nearest_multiple(tbs, self.data_axis_size) // self.data_axis_size
                )
            else:
                self.per_device_eval_parallelism = self.per_device_parallelism

            logger.info(f"Setting per_device_eval_parallelism to {self.per_device_eval_parallelism}")


class AllConfig(Protocol):
    trainer: TrainerConfig


def initialize(config: TrainerConfig | AllConfig):
    """Initializes jax, logging, setting the run name/id in the process. Also initializes tracking and saves config
    as hyperparameters and an artifact"""
    if isinstance(config, TrainerConfig):
        trainer_config = config
    else:
        trainer_config = config.trainer

    trainer_config.initialize()
    levanter.tracker.log_configuration(config)


def _ensure_scalar(x: hax.types.Scalar | hax.NamedArray) -> hax.types.Scalar:
    if isinstance(x, hax.NamedArray):
        return x.scalar()
    else:
        return x


def _resolve_axis_in_tree(tree, axis):
    """
    Resolves an axis in a tree of NamedArrays. This is useful for finding the batch axis in a batch of data.
    """
    for leaf in haliax.tree_util.tree_leaves(tree):
        if isinstance(leaf, haliax.NamedArray):
            try:
                return leaf.resolve_axis(axis)
            except ValueError:
                pass

    raise ValueError(f"Could not find axis {axis} in tree {tree}")
