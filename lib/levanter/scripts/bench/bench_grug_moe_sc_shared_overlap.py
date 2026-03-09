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
from haliax.jax_utils import named_call
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
    shared_widths: tuple[int, ...]
    warmup: int
    iters: int
    dtype: jnp.dtype
    implementations: tuple[DispatchImplementation, ...]
    profile_implementation: DispatchImplementation | None
    profile_shared_width: int | None
    profile_dir: str | None
    capacity_factor: float
    train: bool


def _make_mesh(devices: list[jax.Device]) -> Mesh:
    arr = np.array(devices).reshape(len(devices), 1)
    return Mesh(
        arr,
        axis_names=("data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
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
    valid = {"xla", "sparsecore", "auto"}
    for value in values:
        if value not in valid:
            raise ValueError(f"Unsupported implementation {value!r}; expected one of {sorted(valid)}")
    return tuple(values)  # type: ignore[return-value]


def _shared_overlap_preset(devices: list[jax.Device]) -> dict[str, int | float]:
    target_local_tokens = 32768
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
        "capacity_factor": 1.25,
    }


@named_call
def _shared_dense_mlp(
    x: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> jax.Array:
    gate_up = jnp.matmul(x, w_up_gate)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return jnp.matmul(jax.nn.silu(gate) * up, w_down)


def _make_inputs(
    cfg: BenchCfg,
    shared_width: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    tokens = cfg.batch * cfg.seq
    k_x, k_sel, k_logits, k_w13, k_w2, k_shared13, k_shared2 = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(k_x, (tokens, cfg.hidden), dtype=cfg.dtype)
    selected_experts = jax.random.randint(k_sel, (tokens, cfg.topk), 0, cfg.experts, dtype=jnp.int32)
    combine_logits = jax.random.normal(k_logits, (tokens, cfg.topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(jnp.float32)
    moe_w13 = jax.random.normal(k_w13, (cfg.experts, cfg.hidden, 2 * cfg.intermediate), dtype=cfg.dtype)
    moe_w2 = jax.random.normal(k_w2, (cfg.experts, cfg.intermediate, cfg.hidden), dtype=cfg.dtype)
    shared_w13 = None
    shared_w2 = None
    if shared_width > 0:
        shared_w13 = jax.random.normal(k_shared13, (cfg.hidden, 2 * shared_width), dtype=cfg.dtype)
        shared_w2 = jax.random.normal(k_shared2, (shared_width, cfg.hidden), dtype=cfg.dtype)
    return x, selected_experts, combine_weights, moe_w13, moe_w2, shared_w13, shared_w2


def _bench_one(
    cfg: BenchCfg,
    mesh: Mesh,
    implementation: DispatchImplementation,
    shared_width: int,
) -> tuple[float, float, float]:
    with jax.set_mesh(mesh):
        x, selected_experts, combine_weights, moe_w13, moe_w2, shared_w13, shared_w2 = _make_inputs(cfg, shared_width)
        batch_sharding = NamedSharding(mesh, P("data", None))
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        moe_w13 = jax.device_put(moe_w13, NamedSharding(mesh, P(None, None, None)))
        moe_w2 = jax.device_put(moe_w2, NamedSharding(mesh, P(None, None, None)))
        if shared_w13 is not None:
            shared_w13 = jax.device_put(shared_w13, NamedSharding(mesh, P(None, None)))
            shared_w2 = jax.device_put(shared_w2, NamedSharding(mesh, P(None, None)))

        def forward_fn(
            x_in: jax.Array,
            moe_w13_in: jax.Array,
            moe_w2_in: jax.Array,
            shared_w13_in: jax.Array | None,
            shared_w2_in: jax.Array | None,
        ) -> jax.Array:
            with jax.named_scope("routed_moe"):
                routed = moe_mlp(
                    x_in,
                    selected_experts,
                    combine_weights,
                    moe_w13_in,
                    moe_w2_in,
                    mesh=mesh,
                    activation=ActivationFunctionEnum.silu,
                    capacity_factor=cfg.capacity_factor,
                    dispatch_implementation=implementation,
                )
            if shared_w13_in is None:
                shared = jnp.zeros_like(routed)
            else:
                with jax.named_scope("shared_dense_mlp"):
                    shared = _shared_dense_mlp(x_in, shared_w13_in, shared_w2_in)
            with jax.named_scope("join_shared_and_routed"):
                return routed + shared

        if cfg.train:

            def loss_fn(x_in, moe_w13_in, moe_w2_in, shared_w13_in, shared_w2_in):
                out = forward_fn(x_in, moe_w13_in, moe_w2_in, shared_w13_in, shared_w2_in)
                return jnp.mean(jnp.square(out.astype(jnp.float32)))

            step = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4), allow_int=False))
        else:
            step = jax.jit(forward_fn)

        args = (x, moe_w13, moe_w2, shared_w13, shared_w2)

        start = time.perf_counter()
        out = step(*args)
        jax.block_until_ready(out)
        compile_time = time.perf_counter() - start

        for _ in range(cfg.warmup):
            out = step(*args)
            jax.block_until_ready(out)

        prof_ctx = nullcontext()
        if (
            cfg.profile_dir is not None
            and cfg.profile_implementation == implementation
            and cfg.profile_shared_width == shared_width
        ):
            prof_ctx = profile_ctx(cfg.profile_dir, create_perfetto_link=False)

        with levanter.tracker.current_tracker(NoopTracker()):
            with prof_ctx:
                start = time.perf_counter()
                for _ in range(cfg.iters):
                    out = step(*args)
                    jax.block_until_ready(out)
                steady_time = (time.perf_counter() - start) / cfg.iters

    tokens = cfg.batch * cfg.seq
    return compile_time, steady_time, tokens / steady_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("qwen-shared-overlap",),
        default=None,
        help="Apply a preset benchmark shape. qwen-shared-overlap matches the target Grug shape with 32k local tokens/chip.",
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--shared-widths", type=int, nargs="+", default=(0, 2048, 4096))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--implementation", dest="implementations", nargs="+", default=("xla", "sparsecore"))
    parser.add_argument("--profile-implementation", default=None)
    parser.add_argument("--profile-shared-width", type=int, default=None)
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--xla-dump-dir", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--capacity-factor", type=float, default=None)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    with _tee_stdio(args.log_path):
        if args.xla_dump_dir is not None:
            resolved_dump = _configure_xla_dump_dir(args.xla_dump_dir)
            print(f"Configured XLA dump directory: {resolved_dump}")

        devices = list(jax.devices())
        if not devices:
            raise RuntimeError("No JAX devices available")
        mesh = _make_mesh(devices)

        preset = _shared_overlap_preset(devices) if args.preset == "qwen-shared-overlap" else {}
        batch = args.batch if args.batch is not None else int(preset.get("batch", 8))
        seq = args.seq if args.seq is not None else int(preset.get("seq", 4096))
        hidden = args.hidden if args.hidden is not None else int(preset.get("hidden", 2048))
        intermediate = args.intermediate if args.intermediate is not None else int(preset.get("intermediate", 1536))
        experts = args.experts if args.experts is not None else int(preset.get("experts", 128))
        topk = args.topk if args.topk is not None else int(preset.get("topk", 4))
        capacity_factor = (
            args.capacity_factor if args.capacity_factor is not None else float(preset.get("capacity_factor", 1.25))
        )
        implementations = _parse_impls(list(args.implementations))
        dtype = _parse_dtype(args.dtype)
        shared_widths = tuple(int(width) for width in args.shared_widths)
        profile_implementation = None
        if args.profile_implementation is not None:
            profile_implementation = _parse_impls([args.profile_implementation])[0]

        cfg = BenchCfg(
            batch=batch,
            seq=seq,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            shared_widths=shared_widths,
            warmup=args.warmup,
            iters=args.iters,
            dtype=dtype,
            implementations=implementations,
            profile_implementation=profile_implementation,
            profile_shared_width=args.profile_shared_width,
            profile_dir=args.profile_dir,
            capacity_factor=capacity_factor,
            train=not args.forward_only,
        )

        print(
            "Benchmark config:",
            {
                "batch": cfg.batch,
                "seq": cfg.seq,
                "hidden": cfg.hidden,
                "intermediate": cfg.intermediate,
                "experts": cfg.experts,
                "topk": cfg.topk,
                "shared_widths": cfg.shared_widths,
                "capacity_factor": cfg.capacity_factor,
                "dtype": str(cfg.dtype),
                "train": cfg.train,
                "mesh": dict(mesh.shape),
            },
        )

        for shared_width in cfg.shared_widths:
            print(f"\nshared_width={shared_width}")
            for implementation in cfg.implementations:
                compile_s, steady_s, tokens_per_s = _bench_one(cfg, mesh, implementation, shared_width)
                print(
                    {
                        "implementation": implementation,
                        "shared_width": shared_width,
                        "compile_s": round(compile_s, 6),
                        "steady_s": round(steady_s, 6),
                        "tokens_per_s": round(tokens_per_s, 2),
                    }
                )


if __name__ == "__main__":
    main()
