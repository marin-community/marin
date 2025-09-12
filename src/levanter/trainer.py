# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import atexit
import copy
import functools
import logging as pylogging
import os
import sys
import typing
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, Union

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from draccus import field
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation

import haliax as hax
import haliax.tree_util
from haliax import Axis
from haliax.partitioning import ResourceAxis, ResourceMapping, named_jit
from haliax.quantization import QuantizationConfig
from haliax.types import Scalar

import levanter.callbacks._metrics
import levanter.checkpoint
import levanter.tracker
import levanter.tracker.wandb
import levanter.utils.logging
from levanter import tracker
from levanter.callbacks import Callback, CBInfo, JitCallback, LambdaCallback, StepInfo
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, is_checkpoint_path, load_checkpoint_or_initialize
from levanter.config import JsonAtom
from levanter.data import AsyncDataset, DataLoader
from levanter.data.loader import _round_to_nearest_multiple
from levanter.distributed import DistributedConfig, RayConfig
from levanter.grad_accum import microbatched
from levanter.optim.model_averaging import ModelAveragingConfig
from levanter.schedule import BatchSchedule, IntSchedule, ScheduleStep, value_at_step
from levanter.tracker import TrackerConfig, capture_time
from levanter.trainer_state import InsideJitInfo, TrainerState, saveable_training_mask
from levanter.utils import cloud_utils, fsspec_utils
from levanter.utils.jax_utils import create_fsdp_mesh, zeros_like_tree
from levanter.utils.tree_utils import inference_mode
from levanter.utils.types import ComputeLossFunction, FilterSpec


logger = pylogging.getLogger(__name__)

X = TypeVar("X")  # Input
M = TypeVar("M")  # Model
S = TypeVar("S", bound=TrainerState)  # State

DEFAULT_JAX_CONFIG: Dict[str, JsonAtom] = {
    "jax_threefry_partitionable": True,
    "jax_softmax_custom_jvp": True,
}


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

    return model_init


class Trainer:
    config: "TrainerConfig"
    optimizer: GradientTransformation
    hooks: TrainerHooks
    tracker: levanter.tracker.Tracker
    is_trainable_param: PyTree[FilterSpec]
    _raw_loss_function: Callable
    _cmanagers: List[typing.ContextManager] = []

    def __init__(
        self,
        config: "TrainerConfig",
        optimizer: GradientTransformation,
        loss_fn: ComputeLossFunction,
        *,
        add_default_hooks: bool = True,
    ):
        """

        Args:
            config:  the trainer config
            optimizer: the optimizer, e.g. `optax.adam(1e-3)` or produced by [levanter.optim.OptimizerConfig][]
            loss_fn (Callable): the loss function. This should be a function that takes a model and some inputs and returns a
                scalar loss. It should be jit-able and should not have any side effects.
        """
        self.hooks = TrainerHooks()
        self.config = config
        self.optimizer = optimizer
        self._raw_loss_function = loss_fn
        if isinstance(config.tracker, Sequence):
            self.tracker = levanter.tracker.CompositeTracker([c.init(self.run_id) for c in config.tracker])
        else:
            self.tracker = config.tracker.init(self.run_id)

        self._cmanagers = []

        if add_default_hooks:
            self._add_default_hooks()

        self._cmanagers = []
        self._logged_jaxprs: set[str] = set()

    @cached_property
    def loss_fn(self):
        """
        Wrapped loss function that casts the model to compute precision and sets the context axis mapping to compute
        """

        @functools.wraps(self._raw_loss_function)
        def fn(model, *batch, **batch_kwargs):
            with hax.axis_mapping(self.compute_axis_mapping):
                model = self.mp.cast_to_compute(model)
                return _ensure_scalar(self._raw_loss_function(model, *batch, **batch_kwargs).mean())

        return fn

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

    @property
    def TrainBatch(self):
        return self.config.TrainBatch

    @property
    def EvalBatch(self):
        return self.config.EvalBatch

    def __enter__(self):
        if len(self._cmanagers) > 0:
            raise RuntimeError("Trainer is already entered")

        self._cmanagers = [
            levanter.current_tracker(self.tracker),
            self.device_mesh,
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
                loss, new_state, metrics, cb_states = self._maybe_save_jaxpr(
                    "train_step", self._jit_train_step_fn, state, batch, batch_kwargs
                )
                # force the loss so timing numbers are accurate. laziness isn't going to help here (i think?)
            else:
                loss, new_state, metrics, _ = self._maybe_save_jaxpr(
                    "train_step_hooks", self._jit_train_step_fn_no_hook, state, batch, batch_kwargs
                )
            loss = loss.item()  # type: ignore

            if self.config.crash_on_nan and jnp.isnan(loss):
                raise RuntimeError("Loss is NaN")

            if self.config.crash_on_inf and jnp.isinf(loss):
                raise RuntimeError("Loss is Inf")

            info = StepInfo(new_state, loss, step_time())

            with capture_time() as hook_time:
                self.run_hooks(info)
                if hooks_this_time:
                    self.hooks.run_jit_hooks_outside_step(info, cb_states)

            levanter.tracker.log({**metrics, "throughput/hook_time": hook_time()}, step=info.step)

        return info

    def training_steps(self, state: S, train_loader) -> typing.Iterator[StepInfo[S]]:
        """
        Generator that yields training steps and runs hooks.
        """
        iter_data = iter(train_loader)

        while int(state.step) < self.num_train_steps:
            with capture_time() as loading_time:
                try:
                    example = next(iter_data)
                except StopIteration:
                    logger.info("Reached end of training data loader")
                    break
            info = self.train_step(state, example)
            state = info.state

            levanter.tracker.log({"throughput/loading_time": loading_time()}, step=info.step)

            yield info

    def train(self, state: S, train_loader: Iterable[X]) -> StepInfo[S]:
        """
        Performs training until the number of steps is reached.
        """
        info: Optional[StepInfo[S]] = None
        for info in self.training_steps(state, train_loader):
            pass

        if info is None:
            raise RuntimeError(
                "No training steps were executed. The dataset may be empty or there are no steps left to run."
            )

        # force hooks to run at the end
        self.run_hooks(info, force=True)

        return info

    def _add_default_hooks(self):
        from levanter import callbacks

        self.add_hook(levanter.callbacks.pbar_logger(total=self.config.num_train_steps), every=1)
        self.add_hook(levanter.callbacks.log_step_info(self.config.num_train_steps), every=1)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = self.config.checkpointer.create(self.run_id)
        self.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency

        # Add watch callback if configured
        if self.config.watch.is_enabled:
            self.add_hook(self.config.watch.build(), every=self.config.watch.interval)

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
                model = self.mp.cast_to_compute(model)
                return self.loss_fn(model, *batch, **batch_kwargs, key=None)

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
            batch_name = self.config.batch_axis
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

    def _train_step(
        self, state: S, batch, batch_kwargs, _no_hooks=False
    ) -> tuple[Scalar, S, dict[str, Any], Sequence[CBInfo] | None]:
        with levanter.tracker.defer_tracker_for_jit() as metrics:
            key, new_key = jax.random.split(state.training_key)
            model = inference_mode(state.model, False)

            loss, grads = self._compute_gradients_microbatched(self.loss_fn, model, *batch, **batch_kwargs, key=key)

            # Sophia needs to be able to access the loss function in the optimizer
            def obj_fun(trainable_model):
                model = eqx.combine(trainable_model, state.model)
                with hax.axis_mapping(self.compute_axis_mapping):
                    model = self.mp.cast_to_compute(model)
                    return self._raw_loss_function(model, *batch, **batch_kwargs, key=key).scalar()

            new_state, updates = state.take_step(grads, obj_fun=obj_fun, loss=loss, key=new_key)
            new_state = hax.shard(new_state, self.parameter_axis_mapping)

            if not _no_hooks:
                with hax.axis_mapping(self.parameter_axis_mapping):
                    jit_info: InsideJitInfo = InsideJitInfo(grads=grads, updates=updates)
                    hook_infos = self.hooks.run_jit_hooks(state, jit_info, force=False)

        if _no_hooks:
            return hax.shard_with_axis_mapping((loss, new_state, metrics, None), self.parameter_axis_mapping)
        else:
            # return loss, new_state, metrics, hook_infos
            return hax.shard_with_axis_mapping((loss, new_state, metrics, hook_infos), self.parameter_axis_mapping)

    def _compute_gradients_microbatched(self, loss_fn, model: M, *batch, **batch_kwargs) -> tuple[Scalar, M]:
        Batch = _resolve_axis_in_tree((batch, batch_kwargs), self.config.batch_axis)

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=False)

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
            return grad_fn(model, *batch, **batch_kwargs)

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

    wandb: Optional[tracker.wandb.WandbConfig] = None
    log_dir: Path = Path("logs/")
    id: Optional[str] = None  # run id. if None, will be set to a random string

    tracker: TrackerConfig | Tuple[TrackerConfig, ...] = field(default_factory=tracker.wandb.WandbConfig)
    watch: WatchConfig = WatchConfig()

    # TODO: refactor callbacks
    profiler: bool = False
    profiler_start_step: int = 5
    profiler_num_steps: int = 100
    profiler_perfetto_link: bool = False

    log_jaxprs: bool = True
    """Whether to log the jaxpr of the training step. This is useful for debugging and understanding the model."""
    log_xla_hlo: bool = True
    """Whether to log the XLA HLO of the training step. This is useful for debugging and understanding the model."""

    # helpful checks
    crash_on_nan: bool = True
    crash_on_inf: bool = True

    # config related to partitioning

    batch_axis: str = "batch"  # Batch axis for data parallel.
    fsdp_axis: Optional[Union[str, List[str]]] = "embed"  # Axis/Axes to use for FSDP
    tensor_parallel_axes: Optional[List[str]] = None  # Axes, if any, to use for tensor parallelism

    axis_resources: Mapping[str, Union[Tuple[str], str]] = field(default_factory=dict)
    """mapping from logical axis to physical axis. batch_axis, fsdp_axis, and tensor_parallel_axes are preferred"""
    parameter_axis_resources: Mapping[str, Union[Tuple[str], str]] = field(
        default_factory=dict
    )  # overrides axis_mapping for parameter
    """logical->physical mapping for parameter/optimizer sharding. fsdp_axis and tensor_parallel_axes are preferred"""

    """Interchip Interconnect (ICI) & Data Center Networking (DCN) shardings https://cloud.google.com/tpu/docs/multislice-introduction"""
    replica_ici_axis_size: int = 1
    model_axis_size: int = 1
    """how many devices within each slice for sharding with DP. Fix TP=1, the rest of the devices is for FSDP."""
    replica_dcn_axis_size: int = 1
    """how many slices in the multislice scheme for sharding with DP and TP. The rest of the devices is for FSDP."""

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
        return Axis(self.batch_axis, self.train_batch_size)

    @cached_property
    def batch_schedule(self):
        return BatchSchedule(self.train_batch_size)

    def batch_axis_at_step(self, step: int) -> Axis:
        bs = value_at_step(self.train_batch_size, step)
        return Axis(self.batch_axis, bs)

    @property
    def EvalBatch(self):
        return Axis(self.batch_axis, self.eval_batch_size)

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

    @cached_property
    def device_mesh(self) -> Mesh:
        return create_fsdp_mesh(
            self.replica_ici_axis_size,
            self.data_ici_axis_size,
            self.model_axis_size,
            self.replica_dcn_axis_size,
            self.data_dcn_axis_size,
        )

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

    @property
    def data_ici_axis_size(self):
        """size of the FSDP axis within slices"""
        assert self.num_devices_per_slice % (self.replica_ici_axis_size * self.model_axis_size) == 0
        return self.num_devices_per_slice // (self.replica_ici_axis_size * self.model_axis_size)

    @property
    def data_dcn_axis_size(self):
        """size of the FSDP axis across slices"""
        assert self.num_slices % self.replica_dcn_axis_size == 0
        return self.num_slices // self.replica_dcn_axis_size

    @property
    def data_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        return (
            self.data_dcn_axis_size * self.data_ici_axis_size * self.replica_dcn_axis_size * self.replica_ici_axis_size
        )

    @property
    def replica_axis_size(self):
        """size of the data parallel/batch parallel axis."""
        return self.replica_dcn_axis_size * self.replica_ici_axis_size

    @cached_property
    def compute_axis_mapping(self) -> ResourceMapping:
        """Mapping from logical axis to physical axis for compute."""
        axes_to_return = dict(self.axis_resources)

        tp_axes = self.tensor_parallel_axes or []
        if tp_axes and len(axes_to_return) > 0:
            logger.warning(f"tensor parallelism axes {tp_axes} will override axis_resources {axes_to_return}")
        for axis in tp_axes:
            axes_to_return[axis] = ResourceAxis.MODEL

        if self.batch_axis is not None:
            axes_to_return[self.batch_axis] = (ResourceAxis.REPLICA, ResourceAxis.DATA)  # type: ignore

        return axes_to_return

    @cached_property
    def parameter_axis_mapping(self) -> ResourceMapping:
        mapping = dict(self.compute_axis_mapping)

        for axis, resource in self.parameter_axis_resources.items():
            mapping[axis] = resource

        if isinstance(self.fsdp_axis, str):
            mapping[self.fsdp_axis] = ResourceAxis.DATA
        elif isinstance(self.fsdp_axis, list):
            for axis in self.fsdp_axis:
                mapping[axis] = ResourceAxis.DATA

        return mapping

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
        if jax.device_count() % self.model_axis_size != 0:
            raise ValueError(
                f"num_devices ({jax.device_count()}) is not divisible by model_axis_size ({self.model_axis_size})"
            )

        if (
            jax.local_device_count() % self.model_axis_size != 0
            and self.model_axis_size % jax.local_device_count() != 0
        ):
            raise ValueError("either model_axis_size or local_device_count must be divisible by the other")

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
                tbs = max(levanter.schedule.distinct_values(self.train_batch_size))
                self.per_device_eval_parallelism = (
                    _round_to_nearest_multiple(tbs, self.data_axis_size) // self.data_axis_size
                )
            else:
                self.per_device_eval_parallelism = self.per_device_parallelism

            logger.info(f"Setting per_device_eval_parallelism to {self.per_device_eval_parallelism}")

        if self.replica_dcn_axis_size == -1:
            self.replica_dcn_axis_size = self.num_slices
            logger.info(f"Setting replica_dcn_axis_size to {self.replica_dcn_axis_size}")


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
