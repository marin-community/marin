# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import levanter.tracker
from levanter.callbacks import profile_ctx
from levanter.grug.grug_moe import DispatchImplementation, moe_mlp
from levanter.tracker import NoopTracker
from levanter.utils.activation import ActivationFunctionEnum


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _append_xla_flag(flag: str) -> None:
    current = os.environ.get("XLA_FLAGS", "")
    parts = current.split()
    if flag in parts:
        return
    os.environ["XLA_FLAGS"] = (current + " " + flag).strip()


def _configure_xla_dump_dir(xla_dump_dir: str | None) -> str | None:
    if xla_dump_dir is None:
        return None
    resolved = os.path.abspath(xla_dump_dir)
    os.makedirs(resolved, exist_ok=True)
    _append_xla_flag(f"--xla_dump_to={resolved}")
    _append_xla_flag("--xla_dump_hlo_as_text")
    return resolved


@contextmanager
def _tee_stdio(log_path: str | None):
    if log_path is None:
        yield
        return
    resolved = os.path.abspath(log_path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, handle)
        sys.stderr = _TeeStream(old_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass(frozen=True)
class BenchCfg:
    batch: int
    seq: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    warmup: int
    iters: int
    dtype: jnp.dtype
    implementations: tuple[DispatchImplementation, ...]
    profile_implementation: DispatchImplementation | None
    profile_dir: str | None
    expert_axis_size: int
    capacity_factor: float


def _make_mesh(devices: list[jax.Device], expert_axis_size: int) -> Mesh:
    if expert_axis_size <= 1:
        arr = np.array(devices).reshape(len(devices), 1)
        return Mesh(
            arr,
            axis_names=("data", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )

    if len(devices) % expert_axis_size != 0:
        raise ValueError(
            f"Need device count divisible by expert_axis_size, got devices={len(devices)}, "
            f"expert_axis_size={expert_axis_size}"
        )
    data = len(devices) // expert_axis_size
    arr = np.array(devices).reshape(data, expert_axis_size, 1)
    return Mesh(
        arr,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _parse_dtype(name: str) -> jnp.dtype:
    mapping = {
        "bf16": jnp.bfloat16,
        "f32": jnp.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}") from exc


def _parse_impls(values: list[str]) -> tuple[DispatchImplementation, ...]:
    valid = {"xla", "sparsecore", "sparsecore_pipeline", "sparsecore_expert_pipeline", "auto"}
    for value in values:
        if value not in valid:
            raise ValueError(f"Unsupported implementation {value!r}; expected one of {sorted(valid)}")
    return tuple(values)  # type: ignore[return-value]


def _resolve_arg(value, env_name: str, default, cast):
    if value is not None:
        return cast(value)
    env_value = os.environ.get(env_name)
    if env_value is not None:
        return cast(env_value)
    return default


def _profile_match_preset(devices: list[jax.Device]) -> dict[str, int | float]:
    target_local_tokens = 40960
    seq = 4096
    local_sequences = target_local_tokens // seq
    if local_sequences * seq != target_local_tokens:
        raise ValueError(f"Preset local token target {target_local_tokens} must be divisible by seq {seq}")
    return {
        "batch": local_sequences * len(devices),
        "seq": seq,
        "hidden": 2048,
        "intermediate": 1536,
        "experts": 128,
        "topk": 4,
        "expert_axis_size": 4,
        "capacity_factor": 1.0,
    }


def _local_geometry(cfg: BenchCfg, mesh: Mesh) -> tuple[int, int]:
    batch_shards = int(np.prod([int(mesh.shape[axis]) for axis in ("data", "expert") if axis in mesh.shape]))
    if cfg.batch % batch_shards != 0:
        raise ValueError(f"Global batch {cfg.batch} must be divisible by batch shard count {batch_shards}")
    local_batch = cfg.batch // batch_shards
    return local_batch, local_batch * cfg.seq


def _make_inputs(cfg: BenchCfg) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens = cfg.batch * cfg.seq
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(k_x, (tokens, cfg.hidden), dtype=cfg.dtype)
    selected_experts = jax.random.randint(k_sel, (tokens, cfg.topk), 0, cfg.experts, dtype=jnp.int32)
    combine_logits = jax.random.normal(k_logits, (tokens, cfg.topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(jnp.float32)
    w_up_gate = jax.random.normal(k_w13, (cfg.experts, cfg.hidden, 2 * cfg.intermediate), dtype=cfg.dtype)
    w_down = jax.random.normal(k_w2, (cfg.experts, cfg.intermediate, cfg.hidden), dtype=cfg.dtype)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _bench_one(
    cfg: BenchCfg,
    mesh: Mesh,
    implementation: DispatchImplementation,
) -> tuple[float, float, float]:
    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(cfg)
        batch_axis = ("data", "expert") if "expert" in mesh.shape and int(mesh.shape["expert"]) > 1 else ("data",)
        batch_sharding = NamedSharding(mesh, P(batch_axis, None))
        expert_sharding = (
            NamedSharding(mesh, P("expert", None, None))
            if "expert" in mesh.shape
            else NamedSharding(mesh, P(None, None, None))
        )
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        w_up_gate = jax.device_put(w_up_gate, expert_sharding)
        w_down = jax.device_put(w_down, expert_sharding)

        def loss_fn(x_in, up_in, down_in):
            out = moe_mlp(
                x_in,
                selected_experts,
                combine_weights,
                up_in,
                down_in,
                mesh=mesh,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
                dispatch_implementation=implementation,
            )
            return jnp.mean(jnp.square(out.astype(jnp.float32)))

        step = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2)))

        start = time.perf_counter()
        loss, grads = step(x, w_up_gate, w_down)
        jax.block_until_ready((loss, grads))
        compile_time = time.perf_counter() - start

        for _ in range(cfg.warmup):
            loss, grads = step(x, w_up_gate, w_down)
            jax.block_until_ready((loss, grads))

        prof_ctx = nullcontext()
        if cfg.profile_dir is not None and cfg.profile_implementation == implementation:
            prof_ctx = profile_ctx(cfg.profile_dir, create_perfetto_link=False)

        with levanter.tracker.current_tracker(NoopTracker()):
            with prof_ctx:
                start = time.perf_counter()
                for _ in range(cfg.iters):
                    loss, grads = step(x, w_up_gate, w_down)
                    jax.block_until_ready((loss, grads))
                steady_time = (time.perf_counter() - start) / cfg.iters

    tokens = cfg.batch * cfg.seq
    return compile_time, steady_time, tokens / steady_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("qwen3-32b-ep4-profile",),
        default=None,
        help="Apply a preset benchmark shape. qwen3-32b-ep4-profile matches the profiled EP4 run's local token load.",
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument(
        "--implementation",
        action="append",
        dest="implementations",
        help="Repeat to benchmark multiple implementations. Defaults to xla,sparsecore.",
    )
    parser.add_argument("--profile-implementation", type=str, default=None)
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--xla-dump-dir", type=str, default=None)
    parser.add_argument("--compiler-log-path", type=str, default=None)
    parser.add_argument("--expert-axis-size", type=int, default=None)
    parser.add_argument("--capacity-factor", type=float, default=None)
    args = parser.parse_args()

    impl_args = args.implementations or ["xla", "sparsecore"]
    implementations = _parse_impls(impl_args)
    profile_implementation = None
    profile_impl_arg = args.profile_implementation or os.environ.get("BENCH_PROFILE_IMPL")
    if profile_impl_arg is not None:
        profile_implementation = _parse_impls([profile_impl_arg])[0]

    preset = _profile_match_preset(jax.devices()) if args.preset == "qwen3-32b-ep4-profile" else {}
    batch = _resolve_arg(args.batch, "BENCH_BATCH", int(preset.get("batch", 192)), int)
    seq = _resolve_arg(args.seq, "BENCH_SEQ", int(preset.get("seq", 128)), int)
    hidden = _resolve_arg(args.hidden, "BENCH_HIDDEN", int(preset.get("hidden", 1024)), int)
    intermediate = _resolve_arg(args.intermediate, "BENCH_INTERMEDIATE", int(preset.get("intermediate", 3072)), int)
    experts = _resolve_arg(args.experts, "BENCH_EXPERTS", int(preset.get("experts", 8)), int)
    topk = _resolve_arg(args.topk, "BENCH_TOPK", int(preset.get("topk", 2)), int)
    warmup = _resolve_arg(args.warmup, "BENCH_WARMUP", 1, int)
    iters = _resolve_arg(args.iters, "BENCH_ITERS", 3, int)
    dtype_name = _resolve_arg(args.dtype, "BENCH_DTYPE", "bf16", str)
    expert_axis_size = _resolve_arg(
        args.expert_axis_size, "BENCH_EXPERT_AXIS_SIZE", int(preset.get("expert_axis_size", 1)), int
    )
    capacity_factor = _resolve_arg(
        args.capacity_factor, "BENCH_CAPACITY_FACTOR", float(preset.get("capacity_factor", 1.25)), float
    )
    profile_dir = args.profile_dir or os.environ.get("BENCH_PROFILE_DIR")
    xla_dump_dir = args.xla_dump_dir or os.environ.get("BENCH_XLA_DUMP_DIR")
    compiler_log_path = args.compiler_log_path or os.environ.get("BENCH_COMPILER_LOG_PATH")

    cfg = BenchCfg(
        batch=batch,
        seq=seq,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        warmup=warmup,
        iters=iters,
        dtype=_parse_dtype(dtype_name),
        implementations=implementations,
        profile_implementation=profile_implementation,
        profile_dir=profile_dir,
        expert_axis_size=expert_axis_size,
        capacity_factor=capacity_factor,
    )

    dump_dir = _configure_xla_dump_dir(xla_dump_dir)

    with _tee_stdio(compiler_log_path):
        print("devices", jax.devices())
        print("cfg", cfg)
        print("preset", args.preset)
        print("LIBTPU_INIT_ARGS", os.environ.get("LIBTPU_INIT_ARGS", ""))
        print("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
        print(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_STATIC_ROUTING",
            os.environ.get("GRUG_SPARSECORE_EXPERT_PIPELINE_STATIC_ROUTING", ""),
        )
        print(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS",
            os.environ.get("GRUG_SPARSECORE_EXPERT_PIPELINE_CHUNK_EXPERTS", ""),
        )
        print(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_SINGLE_EXPERT_DENSE",
            os.environ.get("GRUG_SPARSECORE_EXPERT_PIPELINE_SINGLE_EXPERT_DENSE", ""),
        )
        print(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_BARRIER",
            os.environ.get("GRUG_SPARSECORE_EXPERT_PIPELINE_BARRIER", ""),
        )
        print(
            "GRUG_SPARSECORE_EXPERT_PIPELINE_TOKEN_CHUNK_SIZE",
            os.environ.get("GRUG_SPARSECORE_EXPERT_PIPELINE_TOKEN_CHUNK_SIZE", ""),
        )
        if dump_dir is not None:
            print("xla_dump_dir", dump_dir)

        mesh = _make_mesh(jax.devices(), cfg.expert_axis_size)
        local_batch, local_tokens = _local_geometry(cfg, mesh)
        print("mesh", mesh)
        print("local_batch", local_batch)
        print("local_tokens", local_tokens)
        print("local_assignments", local_tokens * cfg.topk)
        print("columns: implementation,compile_s,steady_s,tokens_per_s")
        baseline_tps: float | None = None
        for implementation in cfg.implementations:
            compile_s, steady_s, tokens_per_s = _bench_one(cfg, mesh, implementation)
            print(
                f"{implementation},compile_s={compile_s:.6f},"
                f"steady_s={steady_s:.6f},tokens_per_s={tokens_per_s:.2f}"
            )
            if implementation == "xla":
                baseline_tps = tokens_per_s
            elif baseline_tps is not None:
                print(f"speedup_vs_xla({implementation})={tokens_per_s / baseline_tps:.4f}x")


if __name__ == "__main__":
    main()
