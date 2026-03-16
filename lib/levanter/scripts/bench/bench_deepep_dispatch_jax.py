# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark pure-JAX DeepEP intranode dispatch/combine via XLA FFI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

Distribution = Literal["random", "runs", "deterministic"]
ExecutionModel = Literal["shard_map", "pmap"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_DEEPEP_ROOT = _repo_root() / "lib" / "levanter" / "src" / "levanter" / "kernels" / "deepep"
_LAYOUT_FFI = _load_module("levanter_deepep_layout_ffi", _DEEPEP_ROOT / "layout_ffi.py")
_TRANSPORT_FFI = _load_module("levanter_deepep_transport_ffi", _DEEPEP_ROOT / "transport_ffi.py")
deepep_get_dispatch_layout = _LAYOUT_FFI.deepep_get_dispatch_layout
deepep_dispatch_intranode = _TRANSPORT_FFI.deepep_dispatch_intranode
deepep_combine_intranode = _TRANSPORT_FFI.deepep_combine_intranode
probe_dispatch_kernel_attributes = _TRANSPORT_FFI.probe_dispatch_kernel_attributes
run_host_dispatch_round = _TRANSPORT_FFI.run_host_dispatch_round


def _print0(*args, **kwargs) -> None:
    if jax.process_index() == 0:
        print(*args, **kwargs)


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted, dtype=np.float32)
    denom = np.sum(exp_x, axis=-1, keepdims=True, dtype=np.float32)
    return exp_x / denom


def _sample_router_logits_numpy(
    *,
    seed: int,
    tokens: int,
    experts: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "random":
        return rng.normal(loc=0.0, scale=1.0, size=(tokens, experts)).astype(np.float32)

    if distribution == "deterministic":
        raise ValueError("deterministic routing does not use sampled router logits")

    if distribution != "runs":
        raise ValueError(f"Unknown distribution: {distribution}")

    mean_run = max(2.0, 1.0 / max(1e-6, 1.0 - run_alpha))
    p = min(0.9, max(0.01, 1.0 / mean_run))
    assigned = np.empty((tokens,), dtype=np.int32)
    loads = np.zeros((experts,), dtype=np.int32)
    prev_expert = -1
    pos = 0
    while pos < tokens:
        run_len = int(rng.geometric(p))
        run_len = min(run_len, tokens - pos)
        min_load = int(np.min(loads))
        candidates = np.flatnonzero(loads == min_load)
        if prev_expert in candidates and candidates.size > 1:
            candidates = candidates[candidates != prev_expert]
        expert = int(rng.choice(candidates))
        assigned[pos : pos + run_len] = expert
        loads[expert] += run_len
        prev_expert = expert
        pos += run_len

    logits = rng.normal(loc=0.0, scale=float(run_noise_scale), size=(tokens, experts)).astype(np.float32)
    logits[np.arange(tokens), assigned] += 6.0
    return logits


def _route_topk_numpy(router_logits: np.ndarray, *, topk: int) -> tuple[np.ndarray, np.ndarray]:
    topk_unsorted = np.argpartition(router_logits, kth=-topk, axis=-1)[:, -topk:]
    topk_logits = np.take_along_axis(router_logits, topk_unsorted, axis=-1)
    order = np.argsort(-topk_logits, axis=-1)
    topk_idx = np.take_along_axis(topk_unsorted, order, axis=-1).astype(np.int64)
    topk_logits = np.take_along_axis(topk_logits, order, axis=-1)
    topk_weights = _softmax_numpy(topk_logits).astype(np.float32)
    return topk_idx, topk_weights


def _global_arrays(
    *,
    seed: int,
    tokens: int,
    hidden: int,
    experts: int,
    topk: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distribution == "deterministic":
        x = np.arange(tokens * hidden, dtype=np.float32).reshape(tokens, hidden) / max(hidden, 1)
        topk_idx = np.empty((tokens, topk), dtype=np.int64)
        for token in range(tokens):
            for slot in range(topk):
                topk_idx[token, slot] = (token + slot) % experts
        raw_weights = np.arange(topk, 0, -1, dtype=np.float32)
        raw_weights /= np.sum(raw_weights, dtype=np.float32)
        topk_weights = np.broadcast_to(raw_weights, (tokens, topk)).copy()
        return x, topk_idx, topk_weights

    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(tokens, hidden)).astype(np.float32)
    router_logits = _sample_router_logits_numpy(
        seed=seed + 1,
        tokens=tokens,
        experts=experts,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
    )
    topk_idx, topk_weights = _route_topk_numpy(router_logits, topk=topk)
    return x, topk_idx, topk_weights


def _make_mesh(ep_size: int) -> Mesh:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(devices) != ep_size:
        raise ValueError(
            f"bench_deepep_dispatch_jax currently requires all visible local GPUs to participate; "
            f"got ep_size={ep_size} and visible_gpus={len(devices)}"
        )
    mesh_devices = np.array(devices).reshape(1, ep_size, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _shard_inputs(
    mesh: Mesh,
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    return (
        jax.device_put(x, batch_sharding),
        jax.device_put(topk_idx, batch_sharding),
        jax.device_put(topk_weights, batch_sharding),
    )


def _transport_local(
    x_local: jax.Array,
    topk_idx_local: jax.Array,
    topk_weights_local: jax.Array,
    *,
    num_experts: int,
    num_ranks: int,
    dispatch_config,
    combine_config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        topk_idx_local,
        num_ranks=num_ranks,
        num_experts=num_experts,
    )
    (
        recv_x,
        _,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        _channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        topk_idx_local,
        topk_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
    )
    combined_x, combined_topk_weights = deepep_combine_intranode(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
    )
    fanout = jnp.sum(is_token_in_rank.astype(jnp.int32), axis=1)
    return combined_x, combined_topk_weights, fanout


def _transport_step(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    mesh: Mesh,
    num_experts: int,
    dispatch_config,
    combine_config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    shard_fn = shard_map(
        partial(
            _transport_local,
            num_experts=num_experts,
            num_ranks=mesh.shape["expert"],
            dispatch_config=dispatch_config,
            combine_config=combine_config,
        ),
        mesh=mesh,
        in_specs=(P("expert", None), P("expert", None), P("expert", None)),
        out_specs=(P("expert", None), P("expert", None), P("expert")),
        check_vma=False,
    )
    return shard_fn(x, topk_idx, topk_weights)


def _transport_step_pmap(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    num_ranks: int,
    dispatch_config,
    combine_config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    mapped = jax.pmap(
        partial(
            _transport_local,
            num_experts=num_experts,
            num_ranks=num_ranks,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
        ),
        in_axes=0,
        out_axes=(0, 0, 0),
    )
    return mapped(x, topk_idx, topk_weights)


def _time_fn(fn, *args, warmup: int, iters: int, jit_compile: bool = True) -> float:
    compiled = jax.jit(fn) if jit_compile else fn
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / iters


def _array_payload(array: np.ndarray, *, dtype_label: str, max_elements: int) -> dict[str, object]:
    values_array = array
    if dtype_label in {"bfloat16", "float16"}:
        values_array = array.astype(np.float32)
    payload: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": dtype_label,
        "numel": int(array.size),
    }
    if array.ndim > 0:
        payload["strides"] = [int(stride // array.dtype.itemsize) for stride in array.strides]
    else:
        payload["strides"] = []
    if array.size <= max_elements:
        payload["values"] = values_array.tolist()
    else:
        payload["values_head"] = values_array.reshape(-1)[:max_elements].tolist()
    return payload


def _config_payload(config) -> dict[str, int]:
    return {
        "num_sms": int(config.num_sms),
        "num_max_send_tokens": int(config.num_max_send_tokens),
        "num_max_recv_tokens": int(config.num_max_recv_tokens),
    }


def _dispatch_config_from_args(args, ep_size: int):
    config = _TRANSPORT_FFI._default_dispatch_config(ep_size)
    if args.dispatch_num_sms is None:
        return config
    return _TRANSPORT_FFI.IntranodeConfig(
        num_sms=args.dispatch_num_sms,
        num_max_send_tokens=args.dispatch_num_max_send_tokens or config.num_max_send_tokens,
        num_max_recv_tokens=args.dispatch_num_max_recv_tokens or config.num_max_recv_tokens,
    )


def _combine_config_from_args(args, ep_size: int):
    config = _TRANSPORT_FFI._default_combine_config(ep_size)
    if args.combine_num_sms is None and args.dispatch_num_sms is not None:
        return _TRANSPORT_FFI.IntranodeConfig(
            num_sms=args.dispatch_num_sms,
            num_max_send_tokens=args.combine_num_max_send_tokens or config.num_max_send_tokens,
            num_max_recv_tokens=args.combine_num_max_recv_tokens or config.num_max_recv_tokens,
        )
    if args.combine_num_sms is None:
        return config
    return _TRANSPORT_FFI.IntranodeConfig(
        num_sms=args.combine_num_sms,
        num_max_send_tokens=args.combine_num_max_send_tokens or config.num_max_send_tokens,
        num_max_recv_tokens=args.combine_num_max_recv_tokens or config.num_max_recv_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pure-JAX DeepEP intranode dispatch/combine via XLA FFI.")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--distribution", choices=("random", "runs", "deterministic"), default="random")
    parser.add_argument("--run-alpha", type=float, default=0.97)
    parser.add_argument("--run-noise-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--host-kernel-probe-only", action="store_true")
    parser.add_argument("--host-dispatch-round-only", action="store_true")
    parser.add_argument("--execution-model", choices=("shard_map", "pmap"), default="shard_map")
    parser.add_argument("--probe-max-elements", type=int, default=256)
    parser.add_argument("--dispatch-num-sms", type=int)
    parser.add_argument("--dispatch-num-max-send-tokens", type=int)
    parser.add_argument("--dispatch-num-max-recv-tokens", type=int)
    parser.add_argument("--combine-num-sms", type=int)
    parser.add_argument("--combine-num-max-send-tokens", type=int)
    parser.add_argument("--combine-num-max-recv-tokens", type=int)
    args = parser.parse_args()

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("CUDA JAX devices are required for the DeepEP JAX transport benchmark.")
    ep_size = len(devices)
    if args.tokens % ep_size != 0:
        raise ValueError(f"tokens={args.tokens} must be divisible by visible_gpus={ep_size}")
    if args.experts % ep_size != 0:
        raise ValueError(f"experts={args.experts} must be divisible by visible_gpus={ep_size}")

    if args.host_kernel_probe_only:
        dispatch_config = _dispatch_config_from_args(args, ep_size)
        combine_config = _combine_config_from_args(args, ep_size)
        hidden_bytes = args.hidden * max(jnp.dtype(jnp.bfloat16).itemsize, 2)
        _TRANSPORT_FFI.ensure_intranode_runtime(
            num_ranks=ep_size,
            hidden_bytes=hidden_bytes,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
        )
        payload = {
            "framework": "jax",
            "mode": "host_kernel_probe",
            "world_size": ep_size,
            "num_experts": args.experts,
            "topk": args.topk,
            "hidden": args.hidden,
            "dispatch_config": _config_payload(dispatch_config),
            "result": probe_dispatch_kernel_attributes(),
        }
        _print0("HOST_PROBE_JSON " + json.dumps(payload, sort_keys=True))
        return

    if args.host_dispatch_round_only:
        dispatch_config = _dispatch_config_from_args(args, ep_size)
        combine_config = _combine_config_from_args(args, ep_size)
        hidden_bytes = args.hidden * max(jnp.dtype(jnp.bfloat16).itemsize, 2)
        _TRANSPORT_FFI.ensure_intranode_runtime(
            num_ranks=ep_size,
            hidden_bytes=hidden_bytes,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
        )
        payload = {
            "framework": "jax",
            "mode": "host_dispatch_round",
            "world_size": ep_size,
            "num_experts": args.experts,
            "topk": args.topk,
            "hidden": args.hidden,
            "tokens_per_rank": args.tokens // ep_size,
            "dispatch_config": _config_payload(dispatch_config),
            "combine_config": _config_payload(combine_config),
            "result": run_host_dispatch_round(
                num_tokens=args.tokens // ep_size,
                hidden=args.hidden,
                num_experts=args.experts,
                num_topk=args.topk,
            ),
        }
        _print0("HOST_DISPATCH_JSON " + json.dumps(payload, sort_keys=True))
        return

    x_np, topk_idx_np, topk_weights_np = _global_arrays(
        seed=args.seed,
        tokens=args.tokens,
        hidden=args.hidden,
        experts=args.experts,
        topk=args.topk,
        distribution=args.distribution,
        run_alpha=args.run_alpha,
        run_noise_scale=args.run_noise_scale,
    )
    if args.probe_only:
        tokens_per_rank = args.tokens // ep_size
        device0 = next(device for device in devices if device.id == 0)
        with jax.default_device(device0):
            x_local = jax.device_put(jnp.asarray(x_np[:tokens_per_rank], dtype=jnp.bfloat16), device0)
            topk_idx_local = jax.device_put(jnp.asarray(topk_idx_np[:tokens_per_rank], dtype=jnp.int64), device0)
            topk_weights_local = jax.device_put(
                jnp.asarray(topk_weights_np[:tokens_per_rank], dtype=jnp.float32), device0
            )
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
                topk_idx_local,
                num_ranks=ep_size,
                num_experts=args.experts,
            )
        dispatch_config = _TRANSPORT_FFI._default_dispatch_config(ep_size)
        combine_config = _TRANSPORT_FFI._default_combine_config(ep_size)
        payload = {
            "framework": "jax",
            "distribution": args.distribution,
            "rank": 0,
            "world_size": ep_size,
            "mode": "intranode",
            "num_experts": args.experts,
            "topk": args.topk,
            "runtime": {
                "device": str(device0),
                "dispatch_config": _config_payload(dispatch_config),
                "combine_config": _config_payload(combine_config),
            },
            "inputs": {
                "x": _array_payload(
                    np.asarray(jax.device_get(x_local)),
                    dtype_label="bfloat16",
                    max_elements=args.probe_max_elements,
                ),
                "topk_idx": _array_payload(
                    np.asarray(jax.device_get(topk_idx_local)),
                    dtype_label="int64",
                    max_elements=args.probe_max_elements,
                ),
                "topk_weights": _array_payload(
                    np.asarray(jax.device_get(topk_weights_local)),
                    dtype_label="float32",
                    max_elements=args.probe_max_elements,
                ),
            },
            "layout": {
                "num_tokens_per_rank": _array_payload(
                    np.asarray(jax.device_get(num_tokens_per_rank)),
                    dtype_label="int32",
                    max_elements=args.probe_max_elements,
                ),
                "num_tokens_per_expert": _array_payload(
                    np.asarray(jax.device_get(num_tokens_per_expert)),
                    dtype_label="int32",
                    max_elements=args.probe_max_elements,
                ),
                "is_token_in_rank": _array_payload(
                    np.asarray(jax.device_get(is_token_in_rank)),
                    dtype_label="bool",
                    max_elements=args.probe_max_elements,
                ),
            },
        }
        _print0("PROBE_JSON " + json.dumps(payload, sort_keys=True))
        return

    if args.execution_model == "pmap":
        dispatch_config = _dispatch_config_from_args(args, ep_size)
        combine_config = _combine_config_from_args(args, ep_size)
        tokens_per_rank = args.tokens // ep_size
        device_list = [device for device in devices if device.platform == "gpu"]
        x_inputs = jax.device_put_sharded(
            [x_np[i * tokens_per_rank : (i + 1) * tokens_per_rank].astype(np.float32) for i in range(ep_size)],
            device_list,
        ).astype(jnp.bfloat16)
        topk_idx_inputs = jax.device_put_sharded(
            [topk_idx_np[i * tokens_per_rank : (i + 1) * tokens_per_rank].astype(np.int64) for i in range(ep_size)],
            device_list,
        )
        topk_weights_inputs = jax.device_put_sharded(
            [
                topk_weights_np[i * tokens_per_rank : (i + 1) * tokens_per_rank].astype(np.float32)
                for i in range(ep_size)
            ],
            device_list,
        )
        step_fn = partial(
            _transport_step_pmap,
            num_experts=args.experts,
            num_ranks=ep_size,
            dispatch_config=dispatch_config,
            combine_config=combine_config,
        )
        combined_x, combined_topk_weights, fanout = step_fn(x_inputs, topk_idx_inputs, topk_weights_inputs)
        if args.check:
            fanout_f32 = jnp.maximum(fanout.astype(jnp.float32), 1.0)
            x_error = float(
                jnp.max(jnp.abs(combined_x.astype(jnp.float32) / fanout_f32[..., None] - x_inputs.astype(jnp.float32)))
            )
            topk_error = float(jnp.max(jnp.abs(combined_topk_weights - topk_weights_inputs.astype(jnp.float32))))
            _print0(f"CHECK x_max_abs={x_error:.6e} topk_max_abs={topk_error:.6e}")
        dt = _time_fn(
            step_fn,
            x_inputs,
            topk_idx_inputs,
            topk_weights_inputs,
            warmup=args.warmup,
            iters=args.iters,
            jit_compile=False,
        )
    else:
        dispatch_config = _dispatch_config_from_args(args, ep_size)
        combine_config = _combine_config_from_args(args, ep_size)
        x = jnp.asarray(x_np, dtype=jnp.bfloat16)
        topk_idx = jnp.asarray(topk_idx_np, dtype=jnp.int64)
        topk_weights = jnp.asarray(topk_weights_np, dtype=jnp.float32)

        mesh = _make_mesh(ep_size)
        x_sharded, topk_idx_sharded, topk_weights_sharded = _shard_inputs(mesh, x, topk_idx, topk_weights)

        with jax.set_mesh(mesh):
            step_fn = partial(
                _transport_step,
                mesh=mesh,
                num_experts=args.experts,
                dispatch_config=dispatch_config,
                combine_config=combine_config,
            )
            combined_x, combined_topk_weights, fanout = jax.jit(step_fn)(
                x_sharded, topk_idx_sharded, topk_weights_sharded
            )
            if args.check:
                combined_x_host = np.asarray(combined_x, dtype=np.float32)
                combined_topk_weights_host = np.asarray(combined_topk_weights, dtype=np.float32)
                fanout_host = np.maximum(np.asarray(fanout, dtype=np.float32), 1.0)
                x_host = np.asarray(x_sharded, dtype=np.float32)
                topk_weights_host = np.asarray(topk_weights_sharded, dtype=np.float32)
                x_error = float(np.max(np.abs(combined_x_host / fanout_host[:, None] - x_host)))
                topk_error = float(np.max(np.abs(combined_topk_weights_host - topk_weights_host)))
                _print0(f"CHECK x_max_abs={x_error:.6e} topk_max_abs={topk_error:.6e}")

            dt = _time_fn(
                step_fn,
                x_sharded,
                topk_idx_sharded,
                topk_weights_sharded,
                warmup=args.warmup,
                iters=args.iters,
            )

    tokens_per_second = args.tokens / dt
    _print0(f"devices={devices}")
    _print0(
        "shape "
        f"tokens={args.tokens} hidden={args.hidden} experts={args.experts} topk={args.topk} "
        f"distribution={args.distribution} dtype=bfloat16 mode=intranode execution_model={args.execution_model}"
    )
    _print0(f"RESULT step_s={dt:.6f} tokens_per_s={tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
