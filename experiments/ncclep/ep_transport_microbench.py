# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NCCL_EP dispatch/combine transport microbench (issue #7331, NCCLEP-004).

One process per GPU (TE requirement). Launch shape follows TE's multi-process
tests: ``python ep_transport_microbench.py <coord_addr> <proc_id> <num_procs>``
with flags after. Times the jitted dispatch -> weight-multiply -> combine round
trip (forward, and optionally forward+backward) at configurable shapes, and
reports per-call latency + effective wire GB/s.

Reference-config shapes (#7012/#7279, B200MFU-032): hidden 5120, top_k 4,
num_experts 64, 65,536 tokens/rank (16 seq x 4096).

XLA `ragged_all_to_all` baselines for the same legs (B200MFU-018/-025): one-shot
kernel ~297 ms/call at ~800 MB legs; NCCL fallback via
`--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`.
"""

import argparse
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("coord_addr", nargs="?", help="omit under the iris supervised launcher")
    p.add_argument("proc_id", nargs="?", type=int)
    p.add_argument("num_procs", nargs="?", type=int)
    p.add_argument("--ep", type=int, required=True, help="EP group size (dp = num_procs // ep)")
    p.add_argument("--tokens-per-rank", type=int, default=65536)
    p.add_argument("--hidden-dim", type=int, default=5120)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--capacity-factor", type=float, default=1.25,
                   help="recv capacity = tokens_per_rank * top_k * cf (uniform-routing expectation x margin)")
    p.add_argument("--max-num-sms", type=int, default=0, help="TE comm SM budget; 0 = auto")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--backward", action="store_true", help="also time fwd+bwd via jax.grad")
    p.add_argument("--routing", choices=["uniform", "skewed"], default="uniform")
    return p.parse_args(argv)


def main(argv):
    import os

    args = parse_args(argv)
    if os.environ.get("IRIS_MULTIGPU_PROCESS_COUNT"):
        # iris gang job with --processes-per-task: supervised jax_init joins
        # the mesh (one GPU per process) via the endpoint registry.
        from iris.runtime.jax_init import initialize_jax

        initialize_jax()
    else:
        assert args.coord_addr and args.num_procs is not None, "coord_addr/proc_id/num_procs required"
        jax.distributed.initialize(
            coordinator_address=args.coord_addr,
            num_processes=args.num_procs,
            process_id=args.proc_id,
            local_device_ids=[args.proc_id % 4],
        )
    rank, world = jax.process_index(), jax.process_count()
    ep, dp = args.ep, world // args.ep
    assert dp * ep == world, f"num_procs {world} must equal dp*ep"
    assert args.num_experts % ep == 0

    from transformer_engine.jax.ep import EpLayerConfig, ep_bootstrap, ep_combine, ep_dispatch
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard

    devs = np.asarray(jax.devices()).reshape(dp, ep)
    mesh = Mesh(devs, ("dp", "ep"))
    recv_capacity = int(args.tokens_per_rank * args.top_k * args.capacity_factor)

    T_global = args.tokens_per_rank * world
    H, K, E = args.hidden_dim, args.top_k, args.num_experts

    with mesh, global_shard_guard(MeshResource(dp_resource="dp", ep_resource="ep")):
        ep_bootstrap(
            world_size=world,
            rank=rank,
            num_experts=E,
            max_tokens_per_rank=args.tokens_per_rank,
            recv_capacity_per_rank=recv_capacity,
            hidden_dim=H,
            max_num_sms=args.max_num_sms,
        )
        cfg = EpLayerConfig(top_k=K, dispatch_output_per_expert_alignment=16)

        rng = np.random.default_rng(1234)
        if args.routing == "uniform":
            # Round-robin over experts: exactly uniform load.
            idx = (np.arange(T_global * K, dtype=np.int32).reshape(T_global, K) + rng.integers(0, E)) % E
        else:
            # Zipf-ish skew: stresses capacity headroom.
            probs = 1.0 / np.arange(1, E + 1)
            probs /= probs.sum()
            idx = rng.choice(E, size=(T_global, K), p=probs).astype(np.int32)
        tokens_np = rng.standard_normal((T_global, H), dtype=np.float32).astype(np.float16)

        sharding = NamedSharding(mesh, PartitionSpec(("dp", "ep")))

        def _shard(arr):
            return jax.make_array_from_callback(
                arr.shape, sharding, lambda i: arr[i]
            )

        topk_idx = _shard(idx)
        tokens = _shard(tokens_np.astype(jnp.bfloat16))
        topk_w = _shard(np.full((T_global, K), 1.0 / K, dtype=np.float32).astype(np.float16))
        num_local_tokens = args.tokens_per_rank * dp  # leading dim of the per-EP-group output? asserted below

        def round_trip(tokens, topk_idx, topk_w):
            recv_tokens, recv_w, handle_mem, token_counts = ep_dispatch(
                cfg, topk_idx, tokens, topk_w, recv_capacity
            )
            expert_out = recv_tokens * recv_w[..., None].astype(recv_tokens.dtype)
            return ep_combine(cfg, handle_mem, token_counts, expert_out, tuple(tokens.shape[:-1]))

        fwd = jax.jit(round_trip)

        def loss(tokens, topk_idx, topk_w):
            out = round_trip(tokens, topk_idx, topk_w)
            return jnp.sum(out.astype(jnp.float32))

        grad_fn = jax.jit(jax.grad(loss, argnums=(0, 2)))

        def bench(fn, label, *fn_args):
            for _ in range(args.warmup):
                jax.block_until_ready(fn(*fn_args))
            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*fn_args))
                times.append(time.perf_counter() - t0)
            med = statistics.median(times)
            if rank == 0:
                # Wire bytes per round trip (bf16): dispatch moves T*K*H per rank
                # (upper bound; same-rank routes stay local), combine moves it back.
                leg = args.tokens_per_rank * K * H * 2
                print(
                    f"[{label}] median {med * 1e3:.3f} ms  p10 {sorted(times)[len(times)//10]*1e3:.3f}"
                    f"  p90 {sorted(times)[-max(1, len(times)//10)]*1e3:.3f}"
                    f"  ~2x{leg / 1e9:.3f} GB/rank -> {2 * leg / med / 1e9:.1f} GB/s eff",
                    flush=True,
                )

        out = fwd(tokens, topk_idx, topk_w)
        jax.block_until_ready(out)
        if rank == 0:
            print(f"round-trip out shape {out.shape} dtype {out.dtype}", flush=True)

        bench(fwd, "dispatch+combine fwd", tokens, topk_idx, topk_w)
        if args.backward:
            bench(grad_fn, "dispatch+combine fwd+bwd", tokens, topk_idx, topk_w)

    if rank == 0:
        print("MICROBENCH DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
