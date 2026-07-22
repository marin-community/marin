# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded JaxPP reproducer for an FP8 expert-GEMM backward compile stall.

The production failure occurs while JaxPP compiles the last pipeline stage's
first loss-and-backward task. The ``*_ring`` kernels keep that task's expert
sharding boundary: a production-named stage mesh, sharded expert weights and
activations, ring collectives inside ``shard_map``, FP8 delayed-scaling state
returned as overwrite gradients, and optional microbatch gradient reduction.
Opt-in boundaries add the production loss, rematerialization, learned routing,
and full transformer block while retaining the same bounded task wrapper.

``--runtime direct`` compiles each task with ordinary ``jax.jit`` on one stage.
``--runtime distributed_direct`` initializes the same two stage processes as
JaxPP but runs ordinary ``jax.jit`` only on the compute stage. ``--runtime
jaxpp`` wraps that computation in a task on a two-stage MPMD mesh. Each stage
owns ``--devices-per-stage`` devices. ``--worker-mode external`` joins two
Iris tasks instead of spawning local processes. Every compiler boundary emits
a JSON event and a watchdog terminates a stuck worker with exit status 124.
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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.quantization import Fp8RaggedDotOp, OverwriteWithGradient, apply_updates, partition_for_grad_overwrite
from iris.cluster.client import get_job_info
from iris.runtime.jax_init import initialize_jax as initialize_iris_jax
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp.experimental import mpmd as jaxpp_mpmd
from levanter.grug._moe.common import _DEFAULT_EP_CAPACITY_FACTOR
from levanter.grug.attention import AttentionMask, GrugAttentionImplementation
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES, MoEExpertMlp, MoeRaggedDotOps, moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import (
    _DEFAULT_EP_CAPACITY_FACTOR as _GRUG_MOE_EP_CAPACITY_FACTOR,
)
from experiments.grug.moe.model import (
    Block,
    CausalSelfAttention,
    GatedNorm,
    GrugModelConfig,
    MoEMLP,
    RMSNorm,
    TransformerPipelineStage,
    _stack_router_metrics,
)

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
Kernel = Literal["bf16", "fp8", "bf16_ring", "fp8_ring"]
Runtime = Literal["direct", "distributed_direct", "jaxpp"]
WorkerMode = Literal["local", "external"]
StopAfter = Literal["lower", "execute"]
Bootstrap = Literal["jax_environment", "iris_job_info"]
LossBoundary = Literal["mse", "next_token"]
RematMode = Literal["none", "recompute_all", "save_moe"]
RoutingMode = Literal["fixed", "learned_qb"]
BlockBoundary = Literal["moe_only", "full"]
AttentionImplementation = Literal["reference", "gpu_fa4_cute"]
_FP8_ALIGNMENT = 128
_GATED_NORM_RANK = 128
_RMS_NORM_EPS = 1e-5
_PRODUCTION_PIPELINE_STAGES = 4
_STAGE_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")
_BATCH_AXES = ("replica_dcn", "data", "expert")
_JAX_DISTRIBUTED_ENV_KEYS = (
    "JAX_COORDINATOR_ADDRESS",
    "JAX_NUM_PROCESSES",
    "JAX_PROCESS_ID",
)


class _IrisJobInfo(Protocol):
    num_tasks: int

    @property
    def task_index(self) -> int: ...


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
    worker_mode: WorkerMode
    kernel: Kernel
    layers: int
    experts: int
    tokens: int
    hidden: int
    intermediate: int
    loss_boundary: LossBoundary
    remat_mode: RematMode
    routing_mode: RoutingMode
    block_boundary: BlockBoundary
    attention_implementation: AttentionImplementation
    num_heads: int
    num_kv_heads: int
    total_layers: int
    sliding_window: int
    sequence_length: int
    vocab_size: int
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
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "top_k": self.top_k,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "total_layers": self.total_layers,
            "sliding_window": self.sliding_window,
            "devices_per_stage": self.devices_per_stage,
            "microbatches": self.microbatches,
            "amax_history": self.amax_history,
            "timeout": self.timeout,
            "stack_after": self.stack_after,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.worker_mode == "external" and self.runtime == "direct":
            raise ValueError("external worker mode supports only distributed_direct or jaxpp runtime")
        if self.loss_boundary == "next_token":
            if self.tokens % self.sequence_length:
                raise ValueError(f"tokens={self.tokens} must be divisible by sequence_length={self.sequence_length}")
            if self.batch_size % self.devices_per_stage:
                raise ValueError(
                    f"next-token batch size={self.batch_size} must be divisible by "
                    f"devices_per_stage={self.devices_per_stage}"
                )
        if self.routing_mode == "learned_qb":
            if self.loss_boundary != "next_token":
                raise ValueError("learned_qb routing requires --loss-boundary next_token")
            if self.top_k >= self.experts:
                raise ValueError(
                    f"learned_qb routing requires top_k < experts for top-(K+1); got {self.top_k} >= {self.experts}"
                )
            if self.kernel == "fp8":
                raise ValueError("learned_qb FP8 routing requires the expert-parallel fp8_ring kernel")
        if self.block_boundary == "full":
            if self.loss_boundary != "next_token" or self.routing_mode != "learned_qb":
                raise ValueError("full block boundary requires --loss-boundary next_token and --routing-mode learned_qb")
            if self.remat_mode != "save_moe":
                raise ValueError("full block boundary requires --remat-mode save_moe")
            if self.layers > self.total_layers:
                raise ValueError(f"local layers={self.layers} must be <= total_layers={self.total_layers}")
            if self.total_layers - self.layers < _PRODUCTION_PIPELINE_STAGES - 1:
                raise ValueError(
                    "full block boundary models pipeline stage 3 and requires at least one layer in each prior stage; "
                    f"got total_layers={self.total_layers}, local layers={self.layers}"
                )
            if self.hidden % self.num_heads:
                raise ValueError(f"hidden={self.hidden} must be divisible by num_heads={self.num_heads}")
            if self.num_heads % self.num_kv_heads:
                raise ValueError(f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}")
            head_dim = self.hidden // self.num_heads
            if head_dim % 4:
                raise ValueError(f"full block attention requires head dimension divisible by 4, got {head_dim}")
            if self.attention_implementation == "gpu_fa4_cute" and head_dim != 128:
                raise ValueError(f"gpu_fa4_cute requires head dimension 128, got {head_dim}")
        elif self.attention_implementation != "reference":
            raise ValueError("non-reference attention requires --block-boundary full")
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

    @property
    def batch_size(self) -> int:
        return self.tokens // self.sequence_length


@dataclass(frozen=True)
class ExternalDistributedContext:
    """Resolved rank and bootstrap source for one external Iris task."""

    coordinator_address: str | None
    num_processes: int
    process_id: int
    bootstrap: Bootstrap


def _parse_environment_integer(environment: Mapping[str, str], key: str) -> int:
    raw = environment[key]
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from error


def external_distributed_context(
    environment: Mapping[str, str],
    job_info: _IrisJobInfo | None,
) -> ExternalDistributedContext:
    """Resolve and validate the two-task distributed launch boundary."""
    present = tuple(key for key in _JAX_DISTRIBUTED_ENV_KEYS if environment.get(key, "").strip())
    if present and len(present) != len(_JAX_DISTRIBUTED_ENV_KEYS):
        missing = tuple(key for key in _JAX_DISTRIBUTED_ENV_KEYS if key not in present)
        raise ValueError(f"external worker requires all JAX distributed environment variables; missing {missing}")

    if present:
        coordinator_address = environment["JAX_COORDINATOR_ADDRESS"]
        host, separator, raw_port = coordinator_address.rpartition(":")
        if not separator or not host or not raw_port:
            raise ValueError(f"JAX_COORDINATOR_ADDRESS must have host:port form, got {coordinator_address!r}")
        try:
            port = int(raw_port)
        except ValueError as error:
            raise ValueError(f"JAX_COORDINATOR_ADDRESS port must be an integer, got {raw_port!r}") from error
        if not 1 <= port <= 65535:
            raise ValueError(f"JAX_COORDINATOR_ADDRESS port must be in [1, 65535], got {port}")
        context = ExternalDistributedContext(
            coordinator_address=coordinator_address,
            num_processes=_parse_environment_integer(environment, "JAX_NUM_PROCESSES"),
            process_id=_parse_environment_integer(environment, "JAX_PROCESS_ID"),
            bootstrap="jax_environment",
        )
    elif job_info is not None:
        context = ExternalDistributedContext(
            coordinator_address=None,
            num_processes=job_info.num_tasks,
            process_id=job_info.task_index,
            bootstrap="iris_job_info",
        )
    else:
        raise ValueError("external worker requires JAX distributed environment variables or an Iris job-info context")

    if context.num_processes != 2:
        raise ValueError(f"external worker requires exactly 2 processes, got {context.num_processes}")
    if not 0 <= context.process_id < context.num_processes:
        raise ValueError(f"JAX process id {context.process_id} is outside num_processes={context.num_processes}")
    return context


@dataclass(frozen=True)
class StagePartitionSpecs:
    activation: P
    sequence_activation: P
    token: P
    weight: P
    attention_input_weight: P
    attention_output_weight: P
    lm_head: P
    qb_beta: P
    state: P


@dataclass(frozen=True)
class StageShardings:
    activation: NamedSharding
    sequence_activation: NamedSharding
    token: NamedSharding
    weight: NamedSharding
    attention_input_weight: NamedSharding
    attention_output_weight: NamedSharding
    lm_head: NamedSharding
    qb_beta: NamedSharding
    state: NamedSharding


def stage_partition_specs(config: Config) -> StagePartitionSpecs:
    if config.kernel.endswith("_ring"):
        return StagePartitionSpecs(
            activation=P(_BATCH_AXES, None),
            sequence_activation=P(_BATCH_AXES, None, None),
            token=P(_BATCH_AXES, None),
            weight=P("expert", None, None),
            attention_input_weight=P("data", "model"),
            attention_output_weight=P("model", "data"),
            lm_head=P(("replica_dcn", "data"), "model"),
            qb_beta=P(None, None),
            state=P(),
        )
    return StagePartitionSpecs(
        activation=P(),
        sequence_activation=P(),
        token=P(),
        weight=P(),
        attention_input_weight=P(),
        attention_output_weight=P(),
        lm_head=P(),
        qb_beta=P(),
        state=P(),
    )


def stage_shardings(config: Config, mesh: Mesh) -> StageShardings:
    specs = stage_partition_specs(config)
    return StageShardings(
        activation=NamedSharding(mesh, specs.activation),
        sequence_activation=NamedSharding(mesh, specs.sequence_activation),
        token=NamedSharding(mesh, specs.token),
        weight=NamedSharding(mesh, specs.weight),
        attention_input_weight=NamedSharding(mesh, specs.attention_input_weight),
        attention_output_weight=NamedSharding(mesh, specs.attention_output_weight),
        lm_head=NamedSharding(mesh, specs.lm_head),
        qb_beta=NamedSharding(mesh, specs.qb_beta),
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


ExpertLayer = Bf16ExpertLayer | Fp8ExpertLayer | MoEMLP


class LastStageParameters(eqx.Module):
    expert_layers: tuple[ExpertLayer, ...]
    final_norm_weight: jax.Array
    final_gate_down: jax.Array
    final_gate_up: jax.Array
    lm_head: jax.Array


Parameters = tuple[ExpertLayer, ...] | LastStageParameters | TransformerPipelineStage


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


def _routing_model_config(config: Config) -> GrugModelConfig:
    remat_mode = "save_moe" if config.remat_mode == "save_moe" else "recompute_all"
    full_block = config.block_boundary == "full"
    return GrugModelConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden,
        intermediate_dim=config.intermediate,
        shared_expert_intermediate_dim=0,
        num_layers=config.total_layers if full_block else config.layers,
        num_heads=config.num_heads if full_block else 1,
        num_kv_heads=config.num_kv_heads if full_block else 1,
        max_seq_len=config.sequence_length,
        sliding_window=config.sliding_window,
        num_experts=config.experts,
        num_experts_per_token=config.top_k,
        router_z_loss_coef=0.0,
        attention_implementation=cast(GrugAttentionImplementation, config.attention_implementation),
        moe_implementation="ring" if config.kernel.endswith("_ring") else "scatter",
        loss_implementation="xla",
        remat_mode=remat_mode,
    )


def _routed_expert_layer(
    config: Config,
    weight_array: Any,
    state_array: Any,
    model_config: GrugModelConfig | None = None,
) -> MoEMLP:
    ragged_dot_ops = None
    if config.kernel.startswith("fp8"):
        ragged_dot_ops = MoeRaggedDotOps(
            w13=_fp8_op(config.amax_history, state_array),
            w2=_fp8_op(config.amax_history, state_array),
        )
    return MoEMLP(
        router=state_array((config.hidden, config.experts), jnp.bfloat16, 0.01),
        router_bias=state_array((config.experts,), jnp.bfloat16, 0.0),
        expert_mlp=MoEExpertMlp(
            w_gate=weight_array((config.experts, config.hidden, config.intermediate), jnp.bfloat16, 0.01),
            w_up=weight_array((config.experts, config.hidden, config.intermediate), jnp.bfloat16, 0.01),
            w_down=weight_array((config.experts, config.intermediate, config.hidden), jnp.bfloat16, 0.01),
            implementation="ring" if config.kernel.endswith("_ring") else "scatter",
            activation=ActivationFunctionEnum.silu,
            capacity_factor=(
                _GRUG_MOE_EP_CAPACITY_FACTOR
                if model_config is not None and config.block_boundary == "full"
                else _DEFAULT_EP_CAPACITY_FACTOR
            ),
            ragged_dot_ops=ragged_dot_ops,
        ),
        cfg=model_config or _routing_model_config(config),
    )


def _full_block_stage(
    config: Config,
    weight_array: Any,
    state_array: Any,
    attention_input_array: Any,
    attention_output_array: Any,
    lm_head_array: Any,
) -> TransformerPipelineStage:
    model_config = _routing_model_config(config)
    head_dim = model_config.inferred_head_dim

    def gated_norm() -> GatedNorm:
        return GatedNorm(
            w_down=state_array((config.hidden, _GATED_NORM_RANK), jnp.bfloat16, 0.01),
            w_up=state_array((_GATED_NORM_RANK, config.hidden), jnp.bfloat16, 0.01),
        )

    blocks = []
    for _ in range(config.layers):
        attention = CausalSelfAttention(
            w_q=attention_input_array((config.hidden, config.num_heads * head_dim), jnp.bfloat16, 0.01),
            w_k=attention_input_array((config.hidden, config.num_kv_heads * head_dim), jnp.bfloat16, 0.02),
            w_v=attention_input_array((config.hidden, config.num_kv_heads * head_dim), jnp.bfloat16, 0.03),
            w_o=attention_output_array((config.num_heads * head_dim, config.hidden), jnp.bfloat16, 0.04),
            attn_gate=state_array((config.hidden, config.num_heads), jnp.bfloat16, 0.01),
            cfg=model_config,
        )
        blocks.append(
            Block(
                rms_attn=RMSNorm(
                    weight=state_array((config.hidden,), jnp.float32, 1.0),
                    eps=model_config.layer_norm_eps,
                ),
                attn_gated_norm=gated_norm(),
                attn=attention,
                rms_mlp=RMSNorm(
                    weight=state_array((config.hidden,), jnp.float32, 1.0),
                    eps=model_config.layer_norm_eps,
                ),
                mlp_gated_norm=gated_norm(),
                mlp=_routed_expert_layer(config, weight_array, state_array, model_config),
                shared=None,
            )
        )

    return TransformerPipelineStage(
        token_embed=None,
        embed_norm=None,
        embed_gated_norm=None,
        output_proj=lm_head_array((config.hidden, config.vocab_size), jnp.bfloat16, 0.01),
        blocks=tuple(blocks),
        final_norm=RMSNorm(
            weight=state_array((config.hidden,), jnp.float32, 1.0),
            eps=model_config.layer_norm_eps,
        ),
        final_gated_norm=gated_norm(),
        config=model_config,
        stage_index=_PRODUCTION_PIPELINE_STAGES - 1,
        start_layer=config.total_layers - config.layers,
        end_layer=config.total_layers,
        pipeline_stages=_PRODUCTION_PIPELINE_STAGES,
    )


def abstract_parameters(config: Config, shardings: StageShardings) -> Parameters:
    def weight_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.weight)

    def state_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.state)

    def attention_input_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.attention_input_weight)

    def attention_output_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.attention_output_weight)

    def lm_head_array(shape, dtype, _fill):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=shardings.lm_head)

    if config.block_boundary == "full":
        return _full_block_stage(
            config,
            weight_array,
            state_array,
            attention_input_array,
            attention_output_array,
            lm_head_array,
        )

    layers: list[ExpertLayer] = []
    for _ in range(config.layers):
        if config.routing_mode == "learned_qb":
            layers.append(_routed_expert_layer(config, weight_array, state_array))
            continue
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
    expert_layers = tuple(layers)
    if config.loss_boundary == "mse":
        return expert_layers
    return LastStageParameters(
        expert_layers=expert_layers,
        final_norm_weight=state_array((config.hidden,), jnp.float32, 1.0),
        final_gate_down=state_array((config.hidden, _GATED_NORM_RANK), jnp.bfloat16, 0.0),
        final_gate_up=state_array((_GATED_NORM_RANK, config.hidden), jnp.bfloat16, 0.0),
        lm_head=jax.ShapeDtypeStruct(
            (config.hidden, config.vocab_size),
            jnp.bfloat16,
            sharding=shardings.lm_head,
        ),
    )


def _make_array(shape: tuple[int, ...], dtype: jnp.dtype, fill: float, sharding: NamedSharding) -> jax.Array:
    if not any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat):
        return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)
    with jax.set_mesh(sharding.mesh):
        return jax.jit(
            lambda: jnp.full(shape, fill, dtype),
            out_shardings=sharding,
        )()


def _make_lm_head_array(shape: tuple[int, int], dtype: jnp.dtype, sharding: NamedSharding) -> jax.Array:
    if not any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat):
        return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)
    with jax.set_mesh(sharding.mesh):
        return jax.jit(
            lambda: (((jnp.arange(shape[0] * shape[1]).reshape(shape) % 17) - 8) * 0.01).astype(dtype),
            out_shardings=sharding,
        )()


def materialize_parameters(config: Config, shardings: StageShardings) -> Parameters:
    def weight_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.weight)

    def state_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.state)

    def attention_input_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.attention_input_weight)

    def attention_output_array(shape, dtype, fill):
        return _make_array(shape, dtype, fill, shardings.attention_output_weight)

    def lm_head_array(shape, dtype, _fill):
        return _make_lm_head_array(shape, dtype, shardings.lm_head)

    if config.block_boundary == "full":
        return _full_block_stage(
            config,
            weight_array,
            state_array,
            attention_input_array,
            attention_output_array,
            lm_head_array,
        )

    layers: list[ExpertLayer] = []
    for _ in range(config.layers):
        if config.routing_mode == "learned_qb":
            layers.append(_routed_expert_layer(config, weight_array, state_array))
            continue
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
    expert_layers = tuple(layers)
    if config.loss_boundary == "mse":
        return expert_layers
    return LastStageParameters(
        expert_layers=expert_layers,
        final_norm_weight=state_array((config.hidden,), jnp.float32, 1.0),
        final_gate_down=state_array((config.hidden, _GATED_NORM_RANK), jnp.bfloat16, 0.01),
        final_gate_up=state_array((_GATED_NORM_RANK, config.hidden), jnp.bfloat16, 0.01),
        lm_head=_make_array(
            (config.hidden, config.vocab_size),
            jnp.bfloat16,
            0.01,
            shardings.lm_head,
        ),
    )


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
    if isinstance(layer, MoEMLP):
        raise ValueError("learned_qb routing must use the production MoEMLP call path")
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


def _learned_qb_stack_forward(
    layers: tuple[ExpertLayer, ...],
    activation: jax.Array,
    qb_betas: jax.Array | None,
    config: Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if activation.ndim != 3:
        raise ValueError(f"learned_qb routing requires rank-3 [B, S, D] activations, got {activation.shape}")
    if qb_betas is None or qb_betas.shape != (config.layers, config.experts):
        actual_shape = None if qb_betas is None else qb_betas.shape
        raise ValueError(
            f"learned_qb routing requires qb_betas shape {(config.layers, config.experts)}, got {actual_shape}"
        )

    router_stats_by_layer = []

    def isolated_block(block_layer: MoEMLP, block_hidden: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        routed, router_stats = block_layer(block_hidden)
        return block_hidden + routed, router_stats

    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, MoEMLP):
            raise ValueError("learned_qb routing requires MoEMLP parameter layers")
        router_bias = -qb_betas[layer_index]
        router_bias = (router_bias - jnp.mean(router_bias)).astype(layer.router_bias.dtype)
        compute_layer = eqx.tree_at(lambda module: module.router_bias, layer, router_bias)
        if config.remat_mode == "none":
            activation, router_stats = isolated_block(compute_layer, activation)
        else:
            remat_policy = None
            if config.remat_mode == "save_moe":
                remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
            activation, router_stats = eqx.filter_checkpoint(isolated_block, policy=remat_policy)(
                compute_layer,
                activation,
            )
        router_stats_by_layer.append(router_stats)
    return activation, _stack_router_metrics(router_stats_by_layer)


def _expert_stack_forward(
    layers: tuple[ExpertLayer, ...],
    activation: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    original_shape = activation.shape
    if activation.ndim == 3:
        activation = jnp.reshape(
            activation,
            (config.tokens, config.hidden),
            out_sharding=NamedSharding(mesh, stage_partition_specs(config).activation),
        )
    group_sizes = jnp.full((config.experts,), config.tokens_per_expert, dtype=jnp.int32)
    proxy_qb_betas = []

    def isolated_block(block_layer: ExpertLayer, block_hidden: jax.Array) -> tuple[jax.Array, jax.Array]:
        update = _layer_forward(block_layer, block_hidden, group_sizes, config, mesh)
        block_hidden = block_hidden + update
        qb_beta = jnp.broadcast_to(jnp.mean(update.astype(jnp.float32)), (config.experts,))
        return block_hidden, qb_beta

    for layer in layers:
        if config.remat_mode == "none":
            activation, qb_beta = isolated_block(layer, activation)
        else:
            remat_policy = None
            if config.remat_mode == "save_moe":
                remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
            activation, qb_beta = eqx.filter_checkpoint(isolated_block, policy=remat_policy)(layer, activation)
        proxy_qb_betas.append(qb_beta)
    qb_beta_per_layer = jax.lax.stop_gradient(jnp.stack(proxy_qb_betas, axis=0))
    if len(original_shape) == 3:
        activation = jnp.reshape(
            activation,
            original_shape,
            out_sharding=NamedSharding(mesh, stage_partition_specs(config).sequence_activation),
        )
    return activation, qb_beta_per_layer


def _finalize_hidden(params: LastStageParameters, hidden: jax.Array) -> jax.Array:
    dtype = hidden.dtype
    hidden_f32 = hidden.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_f32), axis=-1, keepdims=True)
    hidden = (hidden_f32 * jax.lax.rsqrt(variance + _RMS_NORM_EPS) * params.final_norm_weight).astype(dtype)
    gate_hidden = jax.nn.silu(jnp.einsum("...d,dr->...r", hidden, params.final_gate_down))
    gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, params.final_gate_up))
    return hidden * gate.astype(dtype)


def _apply_full_stage_qb_betas(
    stage: TransformerPipelineStage,
    qb_betas: jax.Array,
) -> TransformerPipelineStage:
    if qb_betas.shape != (len(stage.blocks), stage.config.num_experts):
        raise ValueError(
            "full block boundary requires one QB-beta row per local block; "
            f"got {qb_betas.shape} for {len(stage.blocks)} blocks"
        )
    blocks = []
    for local_index, block in enumerate(stage.blocks):
        router_bias = -qb_betas[local_index]
        router_bias = (router_bias - jnp.mean(router_bias)).astype(block.mlp.router_bias.dtype)
        mlp = eqx.tree_at(lambda module: module.router_bias, block.mlp, router_bias)
        blocks.append(eqx.tree_at(lambda module: module.mlp, block, mlp))
    return eqx.tree_at(lambda module: module.blocks, stage, tuple(blocks))


def _full_stage_loss_and_gradients(
    params: TransformerPipelineStage,
    hidden: jax.Array,
    qb_betas: jax.Array,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    dependency: jax.Array,
) -> tuple[jax.Array, jax.Array, TransformerPipelineStage, jax.Array]:
    def loss_fn(stage_params: TransformerPipelineStage, stage_hidden: jax.Array):
        compute_stage = _apply_full_stage_qb_betas(stage_params, qb_betas)
        stage_hidden = stage_hidden + dependency.astype(stage_hidden.dtype)
        stage_hidden, router_metrics = compute_stage.block_range(stage_hidden, mask=AttentionMask.causal())
        stage_hidden = compute_stage.finalize_hidden(stage_hidden)
        loss, metrics = compute_stage.hidden_next_token_loss(
            stage_hidden,
            token_ids,
            loss_weight,
            router_metrics,
            reduction="mean",
            logsumexp_weight=None,
            loss_dtype=jnp.float32,
            return_router_metrics=True,
        )
        return loss, metrics["qb_beta_per_layer"]

    (loss, next_qb_betas), (grads, d_hidden) = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(params, hidden)
    return loss, next_qb_betas, grads, d_hidden


def loss_and_gradients(
    params: Parameters,
    x: jax.Array,
    dependency: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, Parameters, jax.Array]:
    """Return the scalar loss, parameter gradients, and activation gradient."""
    if isinstance(params, LastStageParameters):
        raise ValueError("MSE boundary requires expert-only parameters")

    def loss_fn(stage_params, activation):
        activation = activation + dependency.astype(activation.dtype)
        activation, _ = _expert_stack_forward(stage_params, activation, config, mesh)
        return jnp.mean(jnp.square(activation.astype(jnp.float32)))

    loss, (grads, dx) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params, x)
    return loss, grads, dx


def _last_stage_loss_and_gradients(
    params: Parameters,
    hidden: jax.Array,
    qb_betas: jax.Array | None,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    dependency: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, Parameters, jax.Array]:
    """Mirror the production last-stage loss-and-backward output tree."""
    if isinstance(params, TransformerPipelineStage):
        if qb_betas is None:
            raise ValueError("full block boundary requires learned QB betas")
        return _full_stage_loss_and_gradients(
            params,
            hidden,
            qb_betas,
            token_ids,
            loss_weight,
            dependency,
        )
    if not isinstance(params, LastStageParameters):
        raise ValueError("next-token boundary requires last-stage parameters")

    def loss_fn(stage_params: LastStageParameters, stage_hidden: jax.Array):
        stage_hidden = stage_hidden + dependency.astype(stage_hidden.dtype)
        router_z_loss = None
        if config.routing_mode == "learned_qb":
            stage_hidden, router_metrics = _learned_qb_stack_forward(
                stage_params.expert_layers,
                stage_hidden,
                qb_betas,
                config,
            )
            qb_beta_per_layer = router_metrics["qb_beta_per_layer"]
            router_z_loss = jnp.mean(router_metrics["router_z_loss_per_layer"])
        else:
            stage_hidden, qb_beta_per_layer = _expert_stack_forward(
                stage_params.expert_layers,
                stage_hidden,
                config,
                mesh,
            )
        stage_hidden = _finalize_hidden(stage_params, stage_hidden)
        labels = jnp.concatenate([token_ids[:, 1:], jnp.zeros_like(token_ids[:, :1])], axis=1).astype(jnp.int32)
        loss = fused_linear_softmax_cross_entropy_loss(
            stage_hidden,
            stage_params.lm_head,
            labels,
            weight=loss_weight.astype(jnp.float32),
            reduction="mean",
            logsumexp_weight=None,
            dtype=jnp.float32,
            implementation="xla",
        )
        if router_z_loss is not None:
            loss = loss + jnp.asarray(0.0, jnp.float32) * router_z_loss
        return loss, qb_beta_per_layer

    (loss, qb_beta_per_layer), (grads, d_hidden) = jax.value_and_grad(
        loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(params, hidden)
    return loss, qb_beta_per_layer, grads, d_hidden


def last_stage_loss_and_gradients(
    params: Parameters,
    hidden: jax.Array,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    dependency: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, Parameters, jax.Array]:
    """Run the fixed-routing last-stage boundary retained by completed gates."""
    return _last_stage_loss_and_gradients(
        params,
        hidden,
        None,
        token_ids,
        loss_weight,
        dependency,
        config,
        mesh,
    )


def routed_last_stage_loss_and_gradients(
    params: Parameters,
    hidden: jax.Array,
    qb_betas: jax.Array,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    dependency: jax.Array,
    config: Config,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, Parameters, jax.Array]:
    """Run the production learned-QB routing path inside the last-stage boundary."""
    return _last_stage_loss_and_gradients(
        params,
        hidden,
        qb_betas,
        token_ids,
        loss_weight,
        dependency,
        config,
        mesh,
    )


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


def _activation_shape(config: Config) -> tuple[int, ...]:
    if config.loss_boundary == "next_token":
        return (config.batch_size, config.sequence_length, config.hidden)
    return (config.tokens, config.hidden)


def _activation_sharding(config: Config, shardings: StageShardings) -> NamedSharding:
    if config.loss_boundary == "next_token":
        return shardings.sequence_activation
    return shardings.activation


def _materialize_microbatches(
    config: Config,
    shardings: StageShardings,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    activation_sharding = _activation_sharding(config, shardings)
    xs = tuple(
        _make_array(
            _activation_shape(config),
            jnp.bfloat16,
            0.02 + index * 0.001,
            activation_sharding,
        )
        for index in range(config.microbatches)
    )
    if config.loss_boundary == "mse":
        return xs, (), ()
    token_ids = tuple(
        _make_array(
            (config.batch_size, config.sequence_length),
            jnp.int32,
            1 + index,
            shardings.token,
        )
        for index in range(config.microbatches)
    )
    loss_weights = tuple(
        _make_array(
            (config.batch_size, config.sequence_length),
            jnp.float32,
            1.0,
            shardings.token,
        )
        for _ in range(config.microbatches)
    )
    return xs, token_ids, loss_weights


def _run_direct(config: Config, process_id: int, devices: list[jax.Device], event_prefix: str) -> None:
    mesh = _stage_mesh(config, devices)
    shardings = stage_shardings(config, mesh)
    params = materialize_parameters(config, shardings)
    xs, token_ids, loss_weights = _materialize_microbatches(config, shardings)
    dependency = _make_array((), jnp.float32, 0.0, shardings.state)
    qb_betas = None
    if config.routing_mode == "learned_qb":
        qb_betas = _make_array((config.layers, config.experts), jnp.float32, 0.0, shardings.qb_beta)
        backward = jax.jit(
            lambda p, x, qb, tokens, weights, d: routed_last_stage_loss_and_gradients(
                p,
                x,
                qb,
                tokens,
                weights,
                d,
                config,
                mesh,
            )
        )
        lower_args = (params, xs[0], qb_betas, token_ids[0], loss_weights[0], dependency)
    elif config.loss_boundary == "next_token":
        backward = jax.jit(
            lambda p, x, tokens, weights, d: last_stage_loss_and_gradients(
                p,
                x,
                tokens,
                weights,
                d,
                config,
                mesh,
            )
        )
        lower_args = (params, xs[0], token_ids[0], loss_weights[0], dependency)
    else:
        backward = jax.jit(lambda p, x, d: loss_and_gradients(p, x, d, config, mesh))
        lower_args = (params, xs[0], dependency)
    event(f"{event_prefix}_loss_backward_lower_entered", process_id=process_id)
    started = time.perf_counter()
    with jax.set_mesh(mesh):
        lowered = backward.lower(*lower_args)
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
        if config.routing_mode == "learned_qb":
            assert qb_betas is not None
            loss, qb_beta_per_layer, grads, dx = compiled_backward(
                params,
                x,
                qb_betas,
                token_ids[microbatch_index],
                loss_weights[microbatch_index],
                dependency,
            )
            jax.block_until_ready((loss, qb_beta_per_layer, grads, dx))
        elif config.loss_boundary == "next_token":
            loss, qb_beta_per_layer, grads, dx = compiled_backward(
                params,
                x,
                token_ids[microbatch_index],
                loss_weights[microbatch_index],
                dependency,
            )
            jax.block_until_ready((loss, qb_beta_per_layer, grads, dx))
        else:
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


def _initialize_distributed(
    config: Config,
    process_id: int,
    external_context: ExternalDistributedContext | None,
) -> None:
    if external_context is None:
        jax.distributed.initialize(
            coordinator_address=f"127.0.0.1:{config.coordinator_port}",
            num_processes=2,
            process_id=process_id,
            local_device_ids=_distributed_local_device_ids(config, process_id),
            cluster_detection_method="deactivate",
        )
        return

    if external_context.process_id != process_id:
        raise ValueError(
            f"external context process_id={external_context.process_id} does not match worker process_id={process_id}"
        )
    event(
        "external_worker_bootstrap",
        process_id=process_id,
        num_processes=external_context.num_processes,
        bootstrap=external_context.bootstrap,
        coordinator_address=external_context.coordinator_address,
    )
    if external_context.bootstrap == "iris_job_info":
        initialize_iris_jax(port=config.coordinator_port, endpoint_name="jaxpp_fp8_repro_coordinator")
        return
    assert external_context.coordinator_address is not None
    jax.distributed.initialize(
        coordinator_address=external_context.coordinator_address,
        num_processes=external_context.num_processes,
        process_id=process_id,
        cluster_detection_method="deactivate",
    )


def _validate_local_stage_devices(config: Config) -> None:
    actual_devices = len(jax.local_devices())
    if actual_devices != config.devices_per_stage:
        raise ValueError(
            f"{config.worker_mode} worker requires exactly {config.devices_per_stage} local devices, "
            f"got {actual_devices}"
        )


def run_distributed_direct_worker(
    config: Config,
    process_id: int,
    external_context: ExternalDistributedContext | None = None,
) -> None:
    _configure_worker(config, process_id)
    _initialize_distributed(config, process_id, external_context)
    try:
        _validate_local_stage_devices(config)
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


def run_jaxpp_worker(
    config: Config,
    process_id: int,
    external_context: ExternalDistributedContext | None = None,
) -> None:
    _configure_worker(config, process_id)
    actual_revision = jaxpp_revision()
    if actual_revision != JAXPP_REVISION:
        raise RuntimeError(f"expected JaxPP revision {JAXPP_REVISION}, got {actual_revision}")
    _initialize_distributed(config, process_id, external_context)
    try:
        _validate_local_stage_devices(config)
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
        activation_sharding = _activation_sharding(config, compute_shardings)
        x_shapes = tuple(
            jax.ShapeDtypeStruct(
                _activation_shape(config),
                jnp.bfloat16,
                sharding=activation_sharding,
            )
            for _ in range(config.microbatches)
        )
        x_shardings = tuple(activation_sharding for _ in x_shapes)
        scalar_shape = jax.ShapeDtypeStruct((), jnp.float32, sharding=source)

        if config.loss_boundary == "next_token":
            token_shapes = tuple(
                jax.ShapeDtypeStruct(
                    (config.batch_size, config.sequence_length),
                    jnp.int32,
                    sharding=compute_shardings.token,
                )
                for _ in range(config.microbatches)
            )
            weight_shapes = tuple(
                jax.ShapeDtypeStruct(
                    (config.batch_size, config.sequence_length),
                    jnp.float32,
                    sharding=compute_shardings.token,
                )
                for _ in range(config.microbatches)
            )
            token_shardings = tuple(compute_shardings.token for _ in token_shapes)

            def next_token_program_body(seed, params, xs, token_ids, loss_weights, qb_betas):
                seed = jaxpp_mpmd.task(lambda value: value + 1, name="repro_source", out_shardings=source)(seed)
                dependency = jaxpp_mpmd.transfer(seed, out_shardings=compute).done()
                accumulated = None
                loss_sum = None
                qb_beta_sum = None
                input_gradients = []
                for microbatch_index, (x, tokens, weights) in enumerate(zip(xs, token_ids, loss_weights, strict=True)):
                    task_shardings = (
                        compute,
                        compute_shardings.qb_beta,
                        param_shardings,
                        compute_shardings.sequence_activation,
                    )
                    if qb_betas is None:
                        loss, qb_beta_per_layer, grads, d_hidden = jaxpp_mpmd.task(
                            lambda p, value, token, weight, dep: last_stage_loss_and_gradients(
                                p,
                                value,
                                token,
                                weight,
                                dep,
                                config,
                                compute_mesh,
                            ),
                            name=f"repro_mb{microbatch_index}_loss_backward",
                            out_shardings=task_shardings,
                        )(params, x, tokens, weights, dependency)
                    else:
                        loss, qb_beta_per_layer, grads, d_hidden = jaxpp_mpmd.task(
                            lambda p, qb, value, token, weight, dep: routed_last_stage_loss_and_gradients(
                                p,
                                value,
                                qb,
                                token,
                                weight,
                                dep,
                                config,
                                compute_mesh,
                            ),
                            name=f"repro_mb{microbatch_index}_loss_backward",
                            out_shardings=task_shardings,
                        )(params, qb_betas, x, tokens, weights, dependency)
                    input_gradients.append(d_hidden)
                    if accumulated is None:
                        accumulated = grads
                        loss_sum = loss
                        qb_beta_sum = qb_beta_per_layer
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
                        qb_beta_sum = jaxpp_mpmd.task(
                            lambda left, right: left + right,
                            name=f"repro_mb{microbatch_index}_accumulate_qb",
                            out_shardings=compute_shardings.qb_beta,
                        )(qb_beta_sum, qb_beta_per_layer)
                assert accumulated is not None
                assert loss_sum is not None
                assert qb_beta_sum is not None
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
                mean_qb_beta = jaxpp_mpmd.task(
                    lambda value: value / config.microbatches,
                    name="repro_average_qb",
                    out_shardings=compute_shardings.qb_beta,
                )(qb_beta_sum)
                return mean_loss, averaged, mean_qb_beta, tuple(input_gradients)

            if config.routing_mode == "learned_qb":
                qb_beta_shape = jax.ShapeDtypeStruct(
                    (config.layers, config.experts),
                    jnp.float32,
                    sharding=compute_shardings.qb_beta,
                )

                @jaxpp_mpmd.mpmd(
                    mpmd_mesh,
                    in_shardings=(
                        source,
                        param_shardings,
                        x_shardings,
                        token_shardings,
                        token_shardings,
                        compute_shardings.qb_beta,
                    ),
                    infer_donation=False,
                )
                def routed_next_token_program(seed, params, xs, token_ids, loss_weights, qb_betas):
                    return next_token_program_body(seed, params, xs, token_ids, loss_weights, qb_betas)

                program = routed_next_token_program
                abstract_args = (
                    scalar_shape,
                    param_shapes,
                    x_shapes,
                    token_shapes,
                    weight_shapes,
                    qb_beta_shape,
                )
            else:

                @jaxpp_mpmd.mpmd(
                    mpmd_mesh,
                    in_shardings=(source, param_shardings, x_shardings, token_shardings, token_shardings),
                    infer_donation=False,
                )
                def fixed_next_token_program(seed, params, xs, token_ids, loss_weights):
                    return next_token_program_body(seed, params, xs, token_ids, loss_weights, None)

                program = fixed_next_token_program
                abstract_args = (scalar_shape, param_shapes, x_shapes, token_shapes, weight_shapes)
        else:

            @jaxpp_mpmd.mpmd(
                mpmd_mesh,
                in_shardings=(source, param_shardings, x_shardings),
                infer_donation=False,
            )
            def mse_program(seed, params, xs):
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

            program = mse_program
            abstract_args = (scalar_shape, param_shapes, x_shapes)

        event("environment", process_id=process_id, config=asdict(config), environment=_environment())
        event("jaxpp_lower_entered", process_id=process_id)
        started = time.perf_counter()
        lowered = program.lower(*abstract_args)
        event("jaxpp_lower_returned", process_id=process_id, elapsed=time.perf_counter() - started)
        if config.stop_after == "lower":
            multihost_utils.sync_global_devices("jaxpp_fp8_repro_lower_complete")
            return

        seed = _make_array((), jnp.float32, 0.0, source)
        params = materialize_parameters(config, compute_shardings)
        xs, token_ids, loss_weights = _materialize_microbatches(config, compute_shardings)
        if config.routing_mode == "learned_qb":
            qb_betas = _make_array(
                (config.layers, config.experts),
                jnp.float32,
                0.0,
                compute_shardings.qb_beta,
            )
            runtime_args = (seed, params, xs, token_ids, loss_weights, qb_betas)
        elif config.loss_boundary == "next_token":
            runtime_args = (seed, params, xs, token_ids, loss_weights)
        else:
            runtime_args = (seed, params, xs)
        flat_args, _ = jax.tree_util.tree_flatten(runtime_args)
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


def run_external_worker(config: Config) -> int:
    job_info = get_job_info()
    context = external_distributed_context(os.environ, job_info)
    if config.runtime == "distributed_direct":
        run_distributed_direct_worker(config, context.process_id, context)
    elif config.runtime == "jaxpp":
        run_jaxpp_worker(config, context.process_id, context)
    else:
        raise ValueError("external worker mode supports only distributed_direct or jaxpp runtime")
    event(
        "verdict",
        verdict="pass",
        process_id=context.process_id,
        worker_mode=config.worker_mode,
    )
    return 0


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
    parser.add_argument("--worker-mode", choices=("local", "external"), default="local")
    parser.add_argument("--kernel", choices=("bf16", "fp8", "bf16_ring", "fp8_ring"), default="fp8")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--loss-boundary", choices=("mse", "next_token"), default="mse")
    parser.add_argument("--remat-mode", choices=("none", "recompute_all", "save_moe"), default="none")
    parser.add_argument("--routing-mode", choices=("fixed", "learned_qb"), default="fixed")
    parser.add_argument("--block-boundary", choices=("moe_only", "full"), default="moe_only")
    parser.add_argument(
        "--attention-implementation",
        choices=("reference", "gpu_fa4_cute"),
        default="reference",
    )
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--total-layers", type=int, default=8)
    parser.add_argument("--sliding-window", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=128)
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
    if config.worker_mode == "external":
        raise SystemExit(run_external_worker(config))
    raise SystemExit(run_supervised(config))


if __name__ == "__main__":
    main()
