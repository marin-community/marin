# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded JaxPP reproducer for an FP8 expert-GEMM backward compile stall.

The production failure occurs while JaxPP compiles the last pipeline stage's
first loss-and-backward task. The ``*_ring`` kernels keep that task's expert
sharding boundary: a production-named stage mesh, sharded expert weights and
activations, ring collectives inside ``shard_map``, FP8 delayed-scaling state
returned as overwrite gradients, and optional microbatch gradient reduction.
They remove attention, learned routing, optimizer state, and pipeline
scheduling.

``--runtime direct`` compiles each task with ordinary ``jax.jit`` on one stage.
``--runtime distributed_direct`` initializes the same two stage processes as
JaxPP but runs ordinary ``jax.jit`` only on the compute stage. ``--runtime
jaxpp`` wraps that computation in a task on a two-stage MPMD mesh. Each stage
owns ``--devices-per-stage`` devices. Every compiler boundary emits a JSON
event and a watchdog terminates a stuck worker with exit status 124.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.quantization import Fp8RaggedDotOp, OverwriteWithGradient, apply_updates, partition_for_grad_overwrite
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp.experimental import mpmd as jaxpp_mpmd
from levanter.grug.grug_moe import MoeRaggedDotOps, moe_mlp

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
Kernel = Literal["bf16", "fp8", "bf16_ring", "fp8_ring"]
Runtime = Literal["direct", "distributed_direct", "jaxpp"]
StopAfter = Literal["lower", "execute"]
_FP8_ALIGNMENT = 128
_STAGE_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
_BATCH_AXES = ("replica_dcn", "data", "expert")


def event(name: str, **fields: Any) -> None:
    """Emit one machine-readable diagnostic event."""
    print(json.dumps({"time": time.time(), "event": name, **fields}, default=str), flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def jaxpp_revision() -> str:
    try:
        distribution = importlib.metadata.distribution("jaxpp")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


@dataclass(frozen=True)
class Config:
    runtime: Runtime
    kernel: Kernel
    layers: int
    experts: int
    tokens: int
    hidden: int
    intermediate: int
    top_k: int
    devices_per_stage: int
    microbatches: int
    amax_history: int
    timeout: int
    stack_after: int
    coordinator_port: int
    stop_after: StopAfter
    dump_dir: str | None

    @property
    def tokens_per_expert(self) -> int:
        return self.tokens // self.experts

    def validate(self) -> None:
        positive = {
            "layers": self.layers,
            "experts": self.experts,
            "tokens": self.tokens,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "top_k": self.top_k,
            "devices_per_stage": self.devices_per_stage,
            "microbatches": self.microbatches,
            "amax_history": self.amax_history,
            "timeout": self.timeout,
            "stack_after": self.stack_after,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.top_k > self.experts:
            raise ValueError(f"top_k={self.top_k} must be <= experts={self.experts}")
        if self.kernel.endswith("_ring"):
            if self.devices_per_stage <= 1:
                raise ValueError(f"{self.kernel} requires devices_per_stage greater than 1")
            if self.experts % self.devices_per_stage:
                raise ValueError(
                    f"experts={self.experts} must be divisible by devices_per_stage={self.devices_per_stage}"
                )
            if self.tokens % self.devices_per_stage:
                raise ValueError(f"tokens={self.tokens} must be divisible by devices_per_stage={self.devices_per_stage}")
            assignments = self.tokens * self.top_k
            if assignments % self.experts:
                raise ValueError(
                    "balanced ring routing requires tokens * top_k to be divisible by experts; "
                    f"got tokens={self.tokens}, top_k={self.top_k}, experts={self.experts}"
                )
        elif self.tokens % self.experts:
            raise ValueError(f"tokens={self.tokens} must be divisible by experts={self.experts}")
        if self.kernel.startswith("fp8"):
            aligned = {
                "hidden": self.hidden,
                "intermediate": self.intermediate,
            }
            if self.kernel == "fp8":
                aligned["tokens"] = self.tokens
            else:
                assignments_per_device, remainder = divmod(
                    self.tokens * self.top_k,
                    self.devices_per_stage,
                )
                if remainder:
                    raise ValueError(
                        "ring assignments must divide evenly across devices_per_stage; "
                        f"got tokens={self.tokens}, top_k={self.top_k}, "
                        f"devices_per_stage={self.devices_per_stage}"
                    )
                aligned["assignments_per_device"] = assignments_per_device
            for name, value in aligned.items():
                if value % _FP8_ALIGNMENT:
                    raise ValueError(f"FP8 {name}={value} must be divisible by {_FP8_ALIGNMENT}")


@dataclass(frozen=True)
class StagePartitionSpecs:
    activation: P
    weight: P
    state: P


@dataclass(frozen=True)
class StageShardings:
    activation: NamedSharding
    weight: NamedSharding
    state: NamedSharding


def stage_partition_specs(config: Config) -> StagePartitionSpecs:
    if config.kernel.endswith("_ring"):
        return StagePartitionSpecs(
            activation=P(_BATCH_AXES, None),
            weight=P("expert", None, None),
            state=P(),
        )
    return StagePartitionSpecs(activation=P(), weight=P(), state=P())


def stage_shardings(config: Config, mesh: Mesh) -> StageShardings:
    specs = stage_partition_specs(config)
    return StageShardings(
        activation=NamedSharding(mesh, specs.activation),
        weight=NamedSharding(mesh, specs.weight),
        state=NamedSharding(mesh, specs.state),
    )


class Bf16ExpertLayer(eqx.Module):
    w13: jax.Array
    w2: jax.Array


class Fp8ExpertLayer(eqx.Module):
    w13: jax.Array
    w2: jax.Array
    w13_op: Fp8RaggedDotOp
    w2_op: Fp8RaggedDotOp


ExpertLayer = Bf16ExpertLayer | Fp8ExpertLayer
Parameters = tuple[ExpertLayer, ...]


def _fp8_op(amax_history: int, array: Any) -> Fp8RaggedDotOp:
    return Fp8RaggedDotOp(
        input_scale=array((1,), jnp.float32, 1.0),
        output_grad_scale=array((1,), jnp.float32, 1.0),
        kernel_scale=array((1,), jnp.float32, 1.0),
        input_amax_history=array((amax_history,), jnp.float32, 0.0),
        output_grad_amax_history=array((amax_history,), jnp.float32, 0.0),
        kernel_amax_history=array((amax_history,), jnp.float32, 0.0),
        compute_dtype=None,
        fwd_dtype=jnp.float8_e4m3fn,
        rev_dtype=jnp.float8_e4m3fn,
    )


def abstract_parameters(config: Config, shardings: StageShardings) -> Parameters:
    def weight_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.weight)

    def state_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.state)

    layers: list[ExpertLayer] = []
    for _ in range(config.layers):
        w13 = weight_array((config.experts, config.hidden, 2 * config.intermediate), jnp.bfloat16, 0.0)
        w2 = weight_array((config.experts, config.intermediate, config.hidden), jnp.bfloat16, 0.0)
        if config.kernel.startswith("fp8"):
            layers.append(
                Fp8ExpertLayer(
                    w13=w13,
                    w2=w2,
                    w13_op=_fp8_op(config.amax_history, state_array),
                    w2_op=_fp8_op(config.amax_history, state_array),
                )
            )
        else:
            layers.append(Bf16ExpertLayer(w13=w13, w2=w2))
    return tuple(layers)


def _make_array(shape: tuple[int, ...], dtype: jnp.dtype, fill: float, sharding: NamedSharding) -> jax.Array:
    if not any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat):
        return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)
    with jax.set_mesh(sharding.mesh):
        return jax.jit(
            lambda: jnp.full(shape, fill, dtype),
            out_shardings=sharding,
        )()


def materialize_parameters(config: Config, shardings: StageShardings) -> Parameters:
    def weight_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.weight)

    def state_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.state)

    layers: list[ExpertLayer] = []
    for _ in range(config.layers):
        w13 = weight_array((config.experts, config.hidden, 2 * config.intermediate), jnp.bfloat16, 0.01)
        w2 = weight_array((config.experts, config.intermediate, config.hidden), jnp.bfloat16, 0.01)
        if config.kernel.startswith("fp8"):
            layers.append(
                Fp8ExpertLayer(
                    w13=w13,
                    w2=w2,
                    w13_op=_fp8_op(config.amax_history, state_array),
                    w2_op=_fp8_op(config.amax_history, state_array),
                )
            )
        else:
            layers.append(Bf16ExpertLayer(w13=w13, w2=w2))
    return tuple(layers)


def _bf16_grouped_dot(lhs: jax.Array, rhs: jax.Array, config: Config) -> jax.Array:
    grouped = lhs.reshape(config.experts, config.tokens_per_expert, lhs.shape[-1])
    return jnp.einsum("etk,ekn->etn", grouped, rhs).reshape(config.tokens, rhs.shape[-1])


def _ring_routing(config: Config) -> tuple[jax.Array, jax.Array]:
    token = jnp.arange(config.tokens, dtype=jnp.int32)[:, None]
    route = jnp.arange(config.top_k, dtype=jnp.int32)[None, :]
    selected_experts = (token * config.top_k + route) % config.experts
    combine_weights = jnp.full(selected_experts.shape, 1.0 / config.top_k, dtype=jnp.bfloat16)
    return selected_experts, combine_weights


def _layer_forward(
    layer: ExpertLayer,
    x: jax.Array,
    group_sizes: jax.Array,
    config: Config,
    mesh: Mesh,
) -> jax.Array:
    if config.kernel.endswith("_ring"):
        selected_experts, combine_weights = _ring_routing(config)
        ragged_dot_ops = None
        if isinstance(layer, Fp8ExpertLayer):
            ragged_dot_ops = MoeRaggedDotOps(w13=layer.w13_op, w2=layer.w2_op)
        return cast(
            jax.Array,
            moe_mlp(
                x,
                selected_experts,
                combine_weights,
                layer.w13,
                layer.w2,
                activation=jax.nn.silu,
                implementation="ring",
                mesh=mesh,
                capacity_factor=1.0,
                ragged_dot_ops=ragged_dot_ops,
            ),
        )
    if isinstance(layer, Fp8ExpertLayer):
        hidden = layer.w13_op(x, layer.w13, group_sizes)
        gate, up = jnp.split(hidden, 2, axis=-1)
        return layer.w2_op(jax.nn.silu(gate) * up, layer.w2, group_sizes)
    hidden = _bf16_grouped_dot(x, layer.w13, config)
    gate, up = jnp.split(hidden, 2, axis=-1)
    return _bf16_grouped_dot(jax.nn.silu(gate) * up, layer.w2, config)


def loss_and_gradients(
    params: Parameters,
    x: jax.Array,
    dependency: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, Parameters, jax.Array]:
    """Return the scalar loss, parameter gradients, and activation gradient."""
    group_sizes = jnp.full((config.experts,), config.tokens_per_expert, dtype=jnp.int32)

    def loss_fn(stage_params, activation):
        activation = activation + dependency.astype(activation.dtype)
        for layer in stage_params:
            activation = activation + _layer_forward(layer, activation, group_sizes, config, mesh)
        return jnp.mean(jnp.square(activation.astype(jnp.float32)))

    loss, (grads, dx) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params, x)
    return loss, grads, dx


def _is_overwrite(value: Any) -> bool:
    return isinstance(value, OverwriteWithGradient)


def accumulate_gradients(accumulated: Parameters, value: Parameters) -> Parameters:
    """Add trainable gradients and max delayed-scaling overwrite state."""
    accumulated_overwrites, _ = partition_for_grad_overwrite(accumulated)
    overwrites, ordinary = partition_for_grad_overwrite(value)
    max_overwrites = jax.tree.map(jnp.maximum, accumulated_overwrites, overwrites)
    return apply_updates(accumulated, ordinary, max_overwrites)


def average_gradients(grads: Parameters, microbatches: int) -> Parameters:
    """Average trainable gradients without scaling overwrite state."""
    overwrites, ordinary = partition_for_grad_overwrite(grads)
    scale = jnp.asarray(1.0 / microbatches, dtype=jnp.float32)
    averaged = jax.tree.map(lambda value: value * scale, ordinary)
    return eqx.combine(overwrites, averaged, is_leaf=_is_overwrite)


def _configure_worker(config: Config, process_id: int) -> None:
    if config.dump_dir is not None:
        dump_dir = os.path.join(config.dump_dir, f"rank-{process_id}")
        os.makedirs(dump_dir, exist_ok=True)
        dump_flags = (
            f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text " "--xla_dump_hlo_as_proto --xla_dump_hlo_as_long_text"
        )
        os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {dump_flags}".strip()
    faulthandler.enable()
    faulthandler.dump_traceback_later(config.stack_after, repeat=True)

    def hard_stop() -> None:
        event("watchdog_timeout", process_id=process_id, timeout=config.timeout)
        os._exit(124)

    timer = threading.Timer(config.timeout, hard_stop)
    timer.daemon = True
    timer.start()


def _environment() -> dict[str, Any]:
    backend = jax.extend.backend.get_backend()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "jaxpp": package_version("jaxpp"),
        "jaxpp_revision": jaxpp_revision(),
        "backend_platform": backend.platform,
        "backend_platform_version": backend.platform_version,
        "devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
    }


def _stage_mesh(config: Config, devices: list[jax.Device]) -> Mesh:
    if len(devices) != config.devices_per_stage:
        raise ValueError(
            f"{config.runtime} runtime requires {config.devices_per_stage} compute devices, got {len(devices)}"
        )
    shaped = np.asarray(devices, dtype=object).reshape(1, 1, config.devices_per_stage, 1)
    return Mesh(shaped, _STAGE_AXIS_NAMES, axis_types=(AxisType.Explicit,) * len(_STAGE_AXIS_NAMES))


def _tree_shardings(tree: Any) -> Any:
    return jax.tree.map(lambda value: value.sharding, tree)


def _run_direct(config: Config, process_id: int, devices: list[jax.Device], event_prefix: str) -> None:
    mesh = _stage_mesh(config, devices)
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    xs = tuple(
        _make_array(
            (config.tokens, config.hidden),
            jnp.bfloat16,
            0.02 + index * 0.001,
            shardings.activation,
        )
        for index in range(config.microbatches)
    )
    dependency = _make_array((), jnp.float32, 0.0, shardings.state)
    backward = jax.jit(lambda p, x, d: loss_and_gradients(p, x, d, config, mesh))
    event(f"{event_prefix}_loss_backward_lower_entered", process_id=process_id)
    started = time.perf_counter()
    lowered = backward.lower(params, xs[0], dependency)
    event(
        f"{event_prefix}_loss_backward_lower_returned",
        process_id=process_id,
        elapsed=time.perf_counter() - started,
    )
    if config.stop_after == "lower":
        return

    event(f"{event_prefix}_loss_backward_compile_entered", process_id=process_id)
    started = time.perf_counter()
    compiled_backward = lowered.compile()
    event(
        f"{event_prefix}_loss_backward_compile_returned",
        process_id=process_id,
        elapsed=time.perf_counter() - started,
    )

    accumulated = None
    loss_sum = jnp.asarray(0.0, jnp.float32)
    for microbatch_index, x in enumerate(xs):
        event(f"{event_prefix}_loss_backward_execute_entered", process_id=process_id, microbatch=microbatch_index)
        started = time.perf_counter()
        loss, grads, dx = compiled_backward(params, x, dependency)
        jax.block_until_ready((loss, grads, dx))
        event(
            f"{event_prefix}_loss_backward_execute_returned",
            process_id=process_id,
            microbatch=microbatch_index,
            elapsed=time.perf_counter() - started,
        )
        loss_sum = loss_sum + loss
        accumulated = grads if accumulated is None else accumulate_gradients(accumulated, grads)
    assert accumulated is not None
    averaged = jax.jit(average_gradients)(accumulated, config.microbatches)
    jax.block_until_ready((loss_sum, averaged))
    event(
        f"{event_prefix}_training_step_returned",
        process_id=process_id,
        loss=float(loss_sum / config.microbatches),
    )


def run_direct_worker(config: Config, process_id: int) -> None:
    _configure_worker(config, process_id)
    if len(jax.devices()) != config.devices_per_stage:
        raise ValueError(
            f"direct runtime requires exactly {config.devices_per_stage} visible devices, got {len(jax.devices())}"
        )
    event("environment", process_id=process_id, config=asdict(config), environment=_environment())
    _run_direct(config, process_id, jax.devices(), "direct")


def _distributed_local_device_ids(config: Config, process_id: int) -> list[int]:
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        return list(range(config.devices_per_stage))
    start = process_id * config.devices_per_stage
    return list(range(start, start + config.devices_per_stage))


def run_distributed_direct_worker(config: Config, process_id: int) -> None:
    _configure_worker(config, process_id)
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{config.coordinator_port}",
        num_processes=2,
        process_id=process_id,
        local_device_ids=_distributed_local_device_ids(config, process_id),
        cluster_detection_method="deactivate",
    )
    try:
        expected_devices = 2 * config.devices_per_stage
        if len(jax.devices()) != expected_devices:
            actual_devices = len(jax.devices())
            raise ValueError(
                f"distributed_direct runtime requires exactly {expected_devices} global devices, got {actual_devices}"
            )
        event("environment", process_id=process_id, config=asdict(config), environment=_environment())
        if process_id == 1:
            _run_direct(config, process_id, jax.local_devices(), "distributed_direct")
        multihost_utils.sync_global_devices("distributed_direct_fp8_repro_complete")
        event("distributed_direct_completion_barrier_returned", process_id=process_id)
    finally:
        jax.distributed.shutdown()


def run_jaxpp_worker(config: Config, process_id: int) -> None:
    _configure_worker(config, process_id)
    actual_revision = jaxpp_revision()
    if actual_revision != JAXPP_REVISION:
        raise RuntimeError(f"expected JaxPP revision {JAXPP_REVISION}, got {actual_revision}")
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{config.coordinator_port}",
        num_processes=2,
        process_id=process_id,
        local_device_ids=_distributed_local_device_ids(config, process_id),
        cluster_detection_method="deactivate",
    )
    try:
        expected_devices = 2 * config.devices_per_stage
        if len(jax.devices()) != expected_devices:
            raise ValueError(
                f"jaxpp runtime requires exactly {expected_devices} global devices, got {len(jax.devices())}"
            )
        devices = np.asarray(jax.devices(), dtype=object).reshape(2, 1, 1, config.devices_per_stage, 1)
        mpmd_mesh = jaxpp_mpmd.MpmdMesh(
            Mesh(
                devices,
                ("pp", *_STAGE_AXIS_NAMES),
                axis_types=(AxisType.Explicit,) * (1 + len(_STAGE_AXIS_NAMES)),
            ),
            "pp",
        )
        source = NamedSharding(mpmd_mesh.unstack[0], P())
        compute_mesh = mpmd_mesh.unstack[1]
        compute_shardings = stage_shardings(config, compute_mesh)
        compute = compute_shardings.state
        param_shapes = abstract_parameters(config, compute_shardings)
        param_shardings = _tree_shardings(param_shapes)
        x_shapes = tuple(
            jax.ShapeDtypeStruct(
                (config.tokens, config.hidden),
                jnp.bfloat16,
                sharding=compute_shardings.activation,
            )
            for _ in range(config.microbatches)
        )
        x_shardings = tuple(compute_shardings.activation for _ in x_shapes)
        scalar_shape = jax.ShapeDtypeStruct((), jnp.float32, sharding=source)

        @jaxpp_mpmd.mpmd(
            mpmd_mesh,
            in_shardings=(source, param_shardings, x_shardings),
            infer_donation=False,
        )
        def program(seed, params, xs):
            seed = jaxpp_mpmd.task(lambda value: value + 1, name="repro_source", out_shardings=source)(seed)
            dependency = jaxpp_mpmd.transfer(seed, out_shardings=compute).done()
            accumulated = None
            loss_sum = None
            for microbatch_index, x in enumerate(xs):
                loss, grads, _dx = jaxpp_mpmd.task(
                    lambda p, value, dep: loss_and_gradients(p, value, dep, config, compute_mesh),
                    name=f"repro_mb{microbatch_index}_loss_backward",
                    out_shardings=(compute, param_shardings, compute_shardings.activation),
                )(params, x, dependency)
                if accumulated is None:
                    accumulated = grads
                    loss_sum = loss
                else:
                    accumulated = jaxpp_mpmd.task(
                        accumulate_gradients,
                        name=f"repro_mb{microbatch_index}_accumulate_grads",
                        out_shardings=param_shardings,
                    )(accumulated, grads)
                    loss_sum = jaxpp_mpmd.task(
                        lambda left, right: left + right,
                        name=f"repro_mb{microbatch_index}_accumulate_loss",
                        out_shardings=compute,
                    )(loss_sum, loss)
            assert accumulated is not None
            assert loss_sum is not None
            averaged = jaxpp_mpmd.task(
                lambda value: average_gradients(value, config.microbatches),
                name="repro_average_grads",
                out_shardings=param_shardings,
            )(accumulated)
            mean_loss = jaxpp_mpmd.task(
                lambda value: value / config.microbatches,
                name="repro_average_loss",
                out_shardings=compute,
            )(loss_sum)
            return mean_loss, averaged

        event("environment", process_id=process_id, config=asdict(config), environment=_environment())
        event("jaxpp_lower_entered", process_id=process_id)
        started = time.perf_counter()
        lowered = program.lower(scalar_shape, param_shapes, x_shapes)
        event("jaxpp_lower_returned", process_id=process_id, elapsed=time.perf_counter() - started)
        if config.stop_after == "lower":
            multihost_utils.sync_global_devices("jaxpp_fp8_repro_lower_complete")
            return

        seed = _make_array((), jnp.float32, 0.0, source)
        params = materialize_parameters(config, compute_shardings)
        xs = tuple(
            _make_array(
                (config.tokens, config.hidden),
                jnp.bfloat16,
                0.02 + index * 0.001,
                compute_shardings.activation,
            )
            for index in range(config.microbatches)
        )
        flat_args, _ = jax.tree_util.tree_flatten((seed, params, xs))
        local_jaxpr = lowered._local_jaxpr
        local_args = [flat_args[index] for index in local_jaxpr.global_invar_indices]
        event(
            "jaxpp_eval_local_compile_execute_entered",
            process_id=process_id,
            local_inputs=len(local_args),
        )
        started = time.perf_counter()
        result = lowered.eval_local(*local_args)
        jax.block_until_ready(result)
        event(
            "jaxpp_eval_local_compile_execute_returned",
            process_id=process_id,
            elapsed=time.perf_counter() - started,
        )
        multihost_utils.sync_global_devices("jaxpp_fp8_repro_complete")
        event("jaxpp_completion_barrier_returned", process_id=process_id)
    finally:
        jax.distributed.shutdown()


def run_supervised(config: Config) -> int:
    context = mp.get_context("spawn")
    workers = {
        "direct": run_direct_worker,
        "distributed_direct": run_distributed_direct_worker,
        "jaxpp": run_jaxpp_worker,
    }
    worker = workers[config.runtime]
    process_count = 1 if config.runtime == "direct" else 2
    processes = [context.Process(target=worker, args=(config, process_id)) for process_id in range(process_count)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + config.timeout + 30
    try:
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            if any(process.exitcode not in (None, 0) for process in processes):
                break
            time.sleep(0.25)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
    exit_codes = [process.exitcode for process in processes]
    if all(code == 0 for code in exit_codes):
        event("verdict", verdict="pass", exit_codes=exit_codes)
        return 0
    timed_out = any(code == 124 for code in exit_codes)
    event("verdict", verdict="compile_stall" if timed_out else "error", exit_codes=exit_codes)
    if timed_out:
        return 124
    return next((code for code in exit_codes if code not in (None, 0)), 1)


def parse_config(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("direct", "distributed_direct", "jaxpp"), required=True)
    parser.add_argument("--kernel", choices=("bf16", "fp8", "bf16_ring", "fp8_ring"), default="fp8")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--devices-per-stage", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--amax-history", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--stack-after", type=int, default=120)
    parser.add_argument("--coordinator-port", type=int, default=5793)
    parser.add_argument("--stop-after", choices=("lower", "execute"), default="execute")
    parser.add_argument("--dump-dir")
    args = parser.parse_args(argv)
    config = Config(**vars(args))
    config.validate()
    return config


def main() -> None:
    config = parse_config()
    raise SystemExit(run_supervised(config))


if __name__ == "__main__":
    main()
