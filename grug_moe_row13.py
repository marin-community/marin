# Copied from Will's harness (delphi-rl/grugmoe/marin/grug_moe_row13.py, 2026-07-07) for a comparable baseline.
#!/usr/bin/env python3
"""Replicate row 13 of issue #6979 (full model.py) on B200.

Row 13: d2560 / 26 layers / 64 experts (top-4) / MHA / sonic dispatch / EP1 (FSDP
over data) / MuonH / stacked-scan blocks / b128 / seq4k. On 8xH100 this measured
26.7% MFU. Here we run the identical model config on 8xB200 (single node) and
report MFU vs the B200 bf16 dense peak plus absolute throughput (comparable
across hardware). A single random batch is reused each step (throughput probe).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import _compute_flops, _make_train_step, initial_state

# B200 SXM bf16 dense (non-sparse) peak per GPU, FLOP/s. H100 SXM bf16 dense is ~9.89e14
# for reference (what row 13's 26.7% is measured against).
_B200_BF16_PEAK_FLOPS = 2.25e15
_H100_BF16_PEAK_FLOPS = 9.89e14

_BATCH_AXES = ("replica_dcn", "data", "expert")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=2560)
    p.add_argument("--num-layers", type=int, default=26)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--num-experts-per-token", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--moe-implementation", default="sonic")
    p.add_argument("--attention-implementation", default="gpu_fa4_cute")
    return p.parse_args()


def build_model(args) -> GrugModelConfig:
    num_heads = args.hidden_dim // args.head_dim
    intermediate = args.hidden_dim // 2
    return GrugModelConfig(
        vocab_size=128256,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,  # full multi-head attention (MHA)
        head_dim=args.head_dim,
        intermediate_dim=intermediate,
        shared_expert_intermediate_dim=intermediate,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        max_seq_len=args.seq_len,
        sliding_window=2048,
        initializer_std=0.5 / (args.hidden_dim**0.5),
        qk_mult=1.3,
        attention_implementation=args.attention_implementation,
        moe_implementation=args.moe_implementation,
        use_array_stacked_blocks=True,  # stacked-blocks lax.scan (needs disable_pko)
        disable_pko=True,
        remat_mode="recompute_all",
    )


def make_batch(batch_size: int, seq_len: int, vocab_size: int, seed: int, mesh) -> GrugLmExample:
    key = jax.random.PRNGKey(seed)
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)
    loss_weight = jnp.broadcast_to(
        GrugLmExample.causal_loss_mask(seq_len).astype(jnp.float32), (batch_size, seq_len)
    )
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    segment_ids = jax.device_put(jnp.zeros((batch_size, seq_len), dtype=jnp.int32), sharding)
    return GrugLmExample(
        tokens=jax.device_put(tokens, sharding),
        loss_weight=jax.device_put(loss_weight, sharding),
        attn_mask=AttentionMask.causal(sliding_window=seq_len).with_segment_ids(segment_ids),
    )


def main() -> None:
    jax.config.update("jax_threefry_partitionable", True)
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(args)
    # MuonH (May Recipe): LR values don't affect throughput; keep them small/stable.
    optimizer = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    opt = optimizer.build(args.steps)
    train_step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    peak = args.num_gpus * _B200_BF16_PEAK_FLOPS

    # EP1 + FSDP: experts are NOT expert-parallel; params/opt shard over the data axis.
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    metrics: list[dict] = []
    tokens_per_step = args.batch_size * args.seq_len

    with set_mesh(mesh):
        batch = make_batch(args.batch_size, args.seq_len, model.vocab_size, args.seed, mesh)

        @jax.jit
        def init(rng):
            return initial_state(model, optimizer=opt, mp=mp, key=rng, ema_beta=None)

        state = init(jax.random.PRNGKey(args.seed))

        loss = jnp.array(0.0)
        for _ in range(args.steps):
            t0 = time.perf_counter()
            state, step_metrics, _w = train_step(state, batch, compute_watch=False)
            loss = step_metrics["train/loss"]
            jax.block_until_ready(loss)
            dur = time.perf_counter() - t0
            step = int(state.step) - 1
            eps = args.batch_size / dur
            achieved = flops_per_example * eps
            m = {
                "step": step,
                "duration": dur,
                "tokens_per_second": tokens_per_step / dur,
                "achieved_flops_per_second": achieved,
                "mfu_b200": achieved / peak,
                "mfu_h100_equiv": achieved / (args.num_gpus * _H100_BF16_PEAK_FLOPS),
                "loss": float(loss),
            }
            metrics.append(m)
            print(json.dumps(m, sort_keys=True), flush=True)

        try:
            ms = jax.local_devices()[0].memory_stats() or {}
            peak_gib = ms.get("peak_bytes_in_use", 0) / 1024**3
        except Exception:
            peak_gib = None

    steady = [m for m in metrics if m["step"] >= args.warmup_steps]

    def med(xs):
        return None if not xs else sorted(xs)[len(xs) // 2]

    summary = {
        "args": vars(args),
        "config": {
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "num_kv_heads": model.num_kv_heads,
            "intermediate_dim": model.intermediate_dim,
            "shared_expert_intermediate_dim": model.shared_expert_intermediate_dim,
            "num_experts": model.num_experts,
            "num_experts_per_token": model.num_experts_per_token,
            "moe_implementation": model.moe_implementation,
            "attention_implementation": model.attention_implementation,
            "use_array_stacked_blocks": model.use_array_stacked_blocks,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "ep": 1,
            "tokens_per_step": tokens_per_step,
            **flops_summary,
            "peak_gib_per_gpu": peak_gib,
        },
        "metrics": metrics,
        "steady_median_mfu_b200": med([m["mfu_b200"] for m in steady]),
        "steady_median_mfu_h100_equiv": med([m["mfu_h100_equiv"] for m in steady]),
        "steady_median_tokens_per_second": med([m["tokens_per_second"] for m in steady]),
        "steady_median_achieved_tflops": None if not steady else med([m["achieved_flops_per_second"] for m in steady]) / 1e12,
        "steady_median_duration": med([m["duration"] for m in steady]),
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "metrics"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
