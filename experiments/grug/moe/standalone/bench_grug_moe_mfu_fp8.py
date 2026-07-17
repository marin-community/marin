# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row-13 MFU benchmark on H100 with optional FP8 (grouped + dense GEMMs + wire).

Port of experiments/grug/moe/standalone/grug_moe_mfu.py (branch
mcwitt/moe-standalone-ep) onto the fp8-moe-mlp-comms branch: instead of the
inlined bf16 model it imports the FP8-wired model/train-step from
experiments.grug.moe, keeping the standalone's methodology — deterministic
synthetic tokens, fixed step count, steady-state median MFU over the
post-warmup steps. Differences from the standalone flagged in the summary:
capacity factor is this branch's hardcoded 1.25 (standalone default 1.0), and
there is no use_array_stacked_blocks knob on this branch.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugFp8Config, GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import _compute_flops, _make_train_step, initial_state

_H100_BF16_PEAK_FLOPS = 9.89e14
_B200_BF16_PEAK_FLOPS = 2.25e15
_BATCH_AXES = ("replica_dcn", "data", "expert")


def synthetic_tokens(global_batch: int, seq_len: int, vocab_size: int, step: int) -> np.ndarray:
    """Deterministic synthetic token ids (no real data); same recipe as the standalone."""
    stride = 9973
    base = np.arange(seq_len, dtype=np.int64)
    idx = step * global_batch + np.arange(global_batch, dtype=np.int64)
    return ((base[None, :] + idx[:, None] * stride) % vocab_size).astype(np.int32)


def _global_array(host_value: np.ndarray, sharding: NamedSharding) -> jax.Array:
    return jax.make_array_from_callback(host_value.shape, sharding, lambda idx: host_value[idx])


def _make_batch(bs: int, seq_len: int, vocab: int, step: int, mesh) -> GrugLmExample:
    sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
    tokens = _global_array(synthetic_tokens(bs, seq_len, vocab, step), sharding)
    loss_weight = _global_array(np.ones((bs, seq_len), dtype=np.float32), sharding)
    segment_ids = _global_array(np.zeros((bs, seq_len), dtype=np.int32), sharding)
    return GrugLmExample(
        tokens=tokens,
        loss_weight=loss_weight,
        attn_mask=AttentionMask.causal(sliding_window=seq_len).with_segment_ids(segment_ids),
    )


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fp8", action="store_true", help="FP8 grouped + dense GEMMs + wire collectives")
    p.add_argument("--no-fp8-wire", action="store_true", help="with --fp8: keep EP collectives bf16")
    p.add_argument("--no-fp8-dense", action="store_true", help="with --fp8: keep dense GEMMs bf16")
    p.add_argument("--no-fp8-grouped", action="store_true", help="with --fp8: keep expert GEMMs bf16 (dense-only arm)")
    p.add_argument(
        "--fp8-recipe",
        default="per_tensor",
        choices=["per_tensor", "mxfp8"],
        help="grouped-GEMM recipe: per_tensor (Hopper Fp8RaggedDotOp) or mxfp8 (Blackwell fused kernels)",
    )
    p.add_argument("--mxfp8-producer", default="auto", choices=["auto", "cute", "xla"])
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
    p.add_argument("--moe-implementation", default="ring")
    p.add_argument("--expert-parallelism", type=int, default=8)
    p.add_argument("--attention-implementation", default="gpu_fa4_cute")
    p.add_argument("--remat-mode", default="recompute_all", choices=["recompute_all", "save_moe"])
    return p.parse_args()


def main():
    jax.config.update("jax_threefry_partitionable", True)
    a = _parse()
    if jax.device_count() != a.num_gpus:
        raise ValueError(f"--num-gpus={a.num_gpus} but jax sees {jax.device_count()} devices")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nh = a.hidden_dim // a.head_dim
    inter = a.hidden_dim // 2
    fp8 = None
    if a.fp8:
        fp8 = GrugFp8Config(
            wire=not a.no_fp8_wire,
            dense=not a.no_fp8_dense,
            grouped=not a.no_fp8_grouped,
            recipe=a.fp8_recipe,
            mxfp8_producer=a.mxfp8_producer,
        )
    model = GrugModelConfig(
        vocab_size=128256,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
        num_heads=nh,
        num_kv_heads=nh,
        head_dim=a.head_dim,
        intermediate_dim=inter,
        shared_expert_intermediate_dim=inter,
        num_experts=a.num_experts,
        num_experts_per_token=a.num_experts_per_token,
        max_seq_len=a.seq_len,
        sliding_window=2048,
        initializer_std=0.5 / (a.hidden_dim**0.5),
        qk_mult=1.3,
        attention_implementation=a.attention_implementation,
        moe_implementation=a.moe_implementation,
        fp8=fp8,
        remat_mode=a.remat_mode,
    )
    optimizer = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    opt = optimizer.build(a.steps)
    train_step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    peak = a.num_gpus * _H100_BF16_PEAK_FLOPS
    mesh = compact_grug_mesh(expert_axis_size=a.expert_parallelism, replica_axis_size=1)
    metrics = []
    tps = a.batch_size * a.seq_len
    with set_mesh(mesh):

        @jax.jit
        def init(rng):
            return initial_state(model, optimizer=opt, mp=mp, key=rng, ema_beta=None)

        state = init(jax.random.PRNGKey(0))
        for step in range(a.steps):
            batch = _make_batch(a.batch_size, a.seq_len, model.vocab_size, step, mesh)
            t0 = time.perf_counter()
            state, sm, _w = train_step(state, batch, compute_watch=False)
            loss = sm["train/loss"]
            jax.block_until_ready(loss)
            dur = time.perf_counter() - t0
            s = int(state.step) - 1
            eps = a.batch_size / dur
            achieved = flops_per_example * eps
            m = {
                "step": s,
                "duration": dur,
                "tokens_per_second": tps / dur,
                "achieved_flops_per_second": achieved,
                "mfu_h100": achieved / peak,
                "mfu_b200_conv": achieved / (a.num_gpus * _B200_BF16_PEAK_FLOPS),
                "loss": float(loss),
            }
            metrics.append(m)
            print(json.dumps(m, sort_keys=True), flush=True)
    steady = [m for m in metrics if m["step"] >= a.warmup_steps]

    def med(xs):
        return None if not xs else sorted(xs)[len(xs) // 2]

    summary = {
        "args": vars(a),
        "config": {
            **flops_summary,
            "hidden_dim": model.hidden_dim,
            "moe_implementation": model.moe_implementation,
            "fp8": (
                None
                if fp8 is None
                else {"wire": fp8.wire, "dense": fp8.dense, "grouped": fp8.grouped, "recipe": fp8.recipe}
            ),
            "capacity_factor_note": "branch-hardcoded 1.25 (standalone default was 1.0)",
            "num_gpus": a.num_gpus,
        },
        "steady_median_mfu_h100": med([m["mfu_h100"] for m in steady]),
        "steady_median_mfu_b200_conv": med([m["mfu_b200_conv"] for m in steady]),
        "steady_median_achieved_tflops": (
            None if not steady else med([m["achieved_flops_per_second"] for m in steady]) / 1e12
        ),
        "steady_median_tokens_per_second": med([m["tokens_per_second"] for m in steady]),
        "steady_median_duration": med([m["duration"] for m in steady]),
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    multihost_utils.sync_global_devices("bench_grug_moe_mfu_fp8_done")


if __name__ == "__main__":
    main()
