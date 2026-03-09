# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward-only EP MoE benchmark harness for fused-kernel experiments."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.callbacks import profile_ctx
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


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
    implementation: str
    profile_dir: str | None
    expert_axis_size: int
    capacity_factor: float
    vllm_bt: int | None
    vllm_bf: int | None
    vllm_bd1: int | None
    vllm_bd2: int | None
    vllm_btc: int | None
    vllm_bfc: int | None
    vllm_bd1c: int | None
    vllm_bd2c: int | None
    vllm_vmem_limit_kib: int | None


def _parse_dtype(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "f32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def _make_mesh(devices: list[jax.Device], expert_axis_size: int) -> Mesh:
    if expert_axis_size <= 1:
        arr = np.array(devices).reshape(len(devices), 1)
        return Mesh(arr, axis_names=("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))

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


def _make_vllm_reference_mesh(devices: list[jax.Device], expert_axis_size: int) -> Mesh:
    if expert_axis_size <= 1:
        arr = np.array(devices).reshape(len(devices), 1)
        return Mesh(arr, axis_names=("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))

    if len(devices) % expert_axis_size != 0:
        raise ValueError(
            f"Need device count divisible by expert_axis_size, got devices={len(devices)}, "
            f"expert_axis_size={expert_axis_size}"
        )
    data = len(devices) // expert_axis_size
    arr = np.array(devices).reshape(data, expert_axis_size)
    return Mesh(arr, axis_names=("data", "expert"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def _profile_match_preset(devices: list[jax.Device]) -> dict[str, int | float]:
    target_local_tokens = 40960
    seq = 4096
    local_sequences = target_local_tokens // seq
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


def _align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


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


def _forward_impl(
    implementation: str,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    gating_logits: jax.Array | None,
    topk: int,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    mesh: Mesh,
    capacity_factor: float,
    *,
    vllm_bt: int | None,
    vllm_bf: int | None,
    vllm_bd1: int | None,
    vllm_bd2: int | None,
    vllm_btc: int | None,
    vllm_bfc: int | None,
    vllm_bd1c: int | None,
    vllm_bd2c: int | None,
    vllm_vmem_limit_kib: int | None,
) -> jax.Array:
    if implementation == "xla":
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            activation=ActivationFunctionEnum.silu,
            capacity_factor=capacity_factor,
            dispatch_implementation="xla",
        )
    if implementation == "xla_router":
        topk_logits, topk_idx = jax.lax.top_k(gating_logits.astype(jnp.float32), topk)
        if topk > 1:
            topk_weights = jax.nn.softmax(topk_logits, axis=-1).astype(jnp.float32)
        else:
            topk_weights = jnp.ones_like(topk_logits, dtype=jnp.float32)
        return moe_mlp(
            x,
            topk_idx.astype(jnp.int32),
            topk_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            activation=ActivationFunctionEnum.silu,
            capacity_factor=capacity_factor,
            dispatch_implementation="xla",
        )
    if implementation == "vllm_reference":
        from levanter.grug.vendor.fused_moe_v1 import fused_ep_moe

        gate_w, up_w = jnp.split(w_up_gate, 2, axis=-1)
        w1 = jnp.stack((gate_w, up_w), axis=1)
        return fused_ep_moe(
            mesh,
            x,
            w1,
            w_down,
            gating_logits,
            topk,
            renormalize_topk_logits=False,
            act_fn="silu",
            scoring_fn="softmax",
            ep_axis_name="expert",
            bt=vllm_bt,
            bf=vllm_bf,
            bd1=vllm_bd1,
            bd2=vllm_bd2,
            btc=vllm_btc,
            bfc=vllm_bfc,
            bd1c=vllm_bd1c,
            bd2c=vllm_bd2c,
            vmem_limit_bytes=(100 * 1024 * 1024 if vllm_vmem_limit_kib is None else vllm_vmem_limit_kib * 1024),
        )
    if implementation == "vllm_precomputed":
        from levanter.grug.vendor.fused_moe_v1 import fused_ep_moe

        gate_w, up_w = jnp.split(w_up_gate, 2, axis=-1)
        w1 = jnp.stack((gate_w, up_w), axis=1)
        return fused_ep_moe(
            mesh,
            x,
            w1,
            w_down,
            gating_logits,
            topk,
            use_precomputed_routing=True,
            renormalize_topk_logits=False,
            act_fn="silu",
            scoring_fn="softmax",
            ep_axis_name="expert",
            bt=vllm_bt,
            bf=vllm_bf,
            bd1=vllm_bd1,
            bd2=vllm_bd2,
            btc=vllm_btc,
            bfc=vllm_bfc,
            bd1c=vllm_bd1c,
            bd2c=vllm_bd2c,
            vmem_limit_bytes=(100 * 1024 * 1024 if vllm_vmem_limit_kib is None else vllm_vmem_limit_kib * 1024),
        )
    raise ValueError(f"Unknown implementation {implementation!r}")


def _bench_one(cfg: BenchCfg, mesh: Mesh) -> tuple[float, float, float]:
    impl_mesh = mesh
    if cfg.implementation in {"vllm_reference", "vllm_precomputed"}:
        impl_mesh = _make_vllm_reference_mesh(list(jax.devices()), cfg.expert_axis_size)

    with jax.set_mesh(impl_mesh):
        x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(cfg)
        gating_logits = None
        if cfg.implementation in {"vllm_reference", "xla_router"}:
            tokens = x.shape[0]
            gating_logits = jnp.full((tokens, cfg.experts), -1e9, dtype=cfg.dtype)
            gating_logits = gating_logits.at[jnp.arange(tokens, dtype=jnp.int32)[:, None], selected_experts].set(
                jnp.log(jnp.clip(combine_weights.astype(jnp.float32), a_min=1e-20)).astype(cfg.dtype)
            )
        elif cfg.implementation == "vllm_precomputed":
            tokens = x.shape[0]
            metadata_width = _align_to(cfg.experts, 128)
            gating_logits = jnp.zeros((tokens, metadata_width), dtype=cfg.dtype)
            gating_logits = gating_logits.at[:, : cfg.topk].set(combine_weights.astype(cfg.dtype))
            gating_logits = gating_logits.at[:, cfg.topk : 2 * cfg.topk].set(selected_experts.astype(cfg.dtype))
        batch_axis = (
            ("data", "expert") if "expert" in impl_mesh.shape and int(impl_mesh.shape["expert"]) > 1 else ("data",)
        )
        batch_sharding = NamedSharding(impl_mesh, P(batch_axis, None))
        expert_sharding = (
            NamedSharding(impl_mesh, P("expert", None, None))
            if "expert" in impl_mesh.shape
            else NamedSharding(impl_mesh, P(None, None, None))
        )
        x = jax.device_put(x, batch_sharding)
        selected_experts = jax.device_put(selected_experts, batch_sharding)
        combine_weights = jax.device_put(combine_weights, batch_sharding)
        if gating_logits is not None:
            gating_logits = jax.device_put(gating_logits, batch_sharding)
        w_up_gate = jax.device_put(w_up_gate, expert_sharding)
        w_down = jax.device_put(w_down, expert_sharding)

        step = jax.jit(
            lambda x_in, up_in, down_in: _forward_impl(
                cfg.implementation,
                x_in,
                selected_experts,
                combine_weights,
                gating_logits,
                cfg.topk,
                up_in,
                down_in,
                impl_mesh,
                cfg.capacity_factor,
                vllm_bt=cfg.vllm_bt,
                vllm_bf=cfg.vllm_bf,
                vllm_bd1=cfg.vllm_bd1,
                vllm_bd2=cfg.vllm_bd2,
                vllm_btc=cfg.vllm_btc,
                vllm_bfc=cfg.vllm_bfc,
                vllm_bd1c=cfg.vllm_bd1c,
                vllm_bd2c=cfg.vllm_bd2c,
                vllm_vmem_limit_kib=cfg.vllm_vmem_limit_kib,
            )
        )

        start = time.perf_counter()
        out = step(x, w_up_gate, w_down)
        jax.block_until_ready(out)
        compile_time = time.perf_counter() - start

        for _ in range(cfg.warmup):
            out = step(x, w_up_gate, w_down)
            jax.block_until_ready(out)

        prof_ctx = nullcontext()
        if cfg.profile_dir is not None:
            prof_ctx = profile_ctx(cfg.profile_dir, create_perfetto_link=False)

        with prof_ctx:
            start = time.perf_counter()
            for _ in range(cfg.iters):
                out = step(x, w_up_gate, w_down)
                jax.block_until_ready(out)
            steady_time = (time.perf_counter() - start) / cfg.iters

    tokens = cfg.batch * cfg.seq
    return compile_time, steady_time, tokens / steady_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("qwen3-32b-ep4-profile",), default="qwen3-32b-ep4-profile")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--intermediate", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--implementation", type=str, default="xla")
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--expert-axis-size", type=int, default=None)
    parser.add_argument("--capacity-factor", type=float, default=None)
    parser.add_argument("--vllm-bt", type=int, default=None)
    parser.add_argument("--vllm-bf", type=int, default=None)
    parser.add_argument("--vllm-bd1", type=int, default=None)
    parser.add_argument("--vllm-bd2", type=int, default=None)
    parser.add_argument("--vllm-btc", type=int, default=None)
    parser.add_argument("--vllm-bfc", type=int, default=None)
    parser.add_argument("--vllm-bd1c", type=int, default=None)
    parser.add_argument("--vllm-bd2c", type=int, default=None)
    parser.add_argument("--vllm-vmem-limit-kib", type=int, default=None)
    args = parser.parse_args()

    preset = _profile_match_preset(jax.devices()) if args.preset == "qwen3-32b-ep4-profile" else {}
    cfg = BenchCfg(
        batch=int(args.batch or preset["batch"]),
        seq=int(args.seq or preset["seq"]),
        hidden=int(args.hidden or preset["hidden"]),
        intermediate=int(args.intermediate or preset["intermediate"]),
        experts=int(args.experts or preset["experts"]),
        topk=int(args.topk or preset["topk"]),
        warmup=args.warmup,
        iters=args.iters,
        dtype=_parse_dtype(args.dtype),
        implementation=args.implementation,
        profile_dir=args.profile_dir,
        expert_axis_size=int(args.expert_axis_size or preset["expert_axis_size"]),
        capacity_factor=float(args.capacity_factor or preset["capacity_factor"]),
        vllm_bt=args.vllm_bt,
        vllm_bf=args.vllm_bf,
        vllm_bd1=args.vllm_bd1,
        vllm_bd2=args.vllm_bd2,
        vllm_btc=args.vllm_btc,
        vllm_bfc=args.vllm_bfc,
        vllm_bd1c=args.vllm_bd1c,
        vllm_bd2c=args.vllm_bd2c,
        vllm_vmem_limit_kib=args.vllm_vmem_limit_kib,
    )

    devices = list(jax.devices())
    mesh = _make_mesh(devices, cfg.expert_axis_size)
    compile_s, steady_s, tokens_per_s = _bench_one(cfg, mesh)

    print("devices", devices)
    print("cfg", cfg)
    print("mesh", mesh)
    print("columns: implementation,compile_s,steady_s,tokens_per_s")
    print(f"{cfg.implementation},compile_s={compile_s:.6f},steady_s={steady_s:.6f},tokens_per_s={tokens_per_s:.2f}")


if __name__ == "__main__":
    main()
