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
import traceback
from pathlib import Path

import jax
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugFp8Config, GrugModelConfig
from experiments.grug.moe.mxfp8_dense import mxfp8_dense_mesh_context
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import _compute_flops, _make_train_step, initial_state

_H100_BF16_PEAK_FLOPS = 9.89e14
_B200_BF16_PEAK_FLOPS = 2.25e15
# GB200 NVL72 bf16 dense per-GPU (360 PF sparse / 72 / 2) — matches fray device_flops
# and the #7201 scale-run convention.
_GB200_BF16_PEAK_FLOPS = 2.5e15
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
    p.add_argument(
        "--fp8-dense-recipe",
        default="per_tensor",
        choices=["per_tensor", "mxfp8"],
        help="dense-GEMM recipe: per_tensor delayed scaling or stateless Transformer Engine mxfp8",
    )
    p.add_argument("--no-fp8-grouped", action="store_true", help="with --fp8: keep expert GEMMs bf16 (dense-only arm)")
    p.add_argument(
        "--fp8-recipe",
        default="auto",
        choices=["auto", "per_tensor", "mxfp8"],
        help="grouped-GEMM recipe: auto (resolve from GPU arch), per_tensor (Hopper Fp8RaggedDotOp), "
        "or mxfp8 (Blackwell fused kernels)",
    )
    p.add_argument("--mxfp8-producer", default="auto", choices=["auto", "cute", "xla"])
    p.add_argument(
        "--mxfp8-save-qweights",
        action="store_true",
        help="with --fp8-recipe mxfp8: save fwd-orientation weight quantize across the remat recompute",
    )
    p.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="after the timed steps, trace this many extra steps with jax.profiler and print a phase breakdown",
    )
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=2560)
    p.add_argument("--num-layers", type=int, default=26)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--num-experts-per-token", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=0, help="0 = MHA (num_heads)")
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--moe-implementation", default="ring")
    p.add_argument("--expert-parallelism", type=int, default=8)
    p.add_argument("--replica-axis", type=int, default=1, help="replica_dcn axis size (DDP replicas)")
    p.add_argument("--attention-implementation", default="gpu_fa4_cute")
    p.add_argument("--remat-mode", default="recompute_all", choices=["recompute_all", "save_moe"])
    p.add_argument("--stacked-blocks", action="store_true", help="lax.scan over ArrayStacked blocks")
    return p.parse_args()


def main():
    jax.config.update("jax_threefry_partitionable", True)
    a = _parse()
    # Multi-node gangs: coordinator discovery via the Iris endpoint registry
    # (no-op for single-task jobs). --num-gpus is the GLOBAL device count.
    initialize_jax()
    if jax.device_count() != a.num_gpus:
        raise ValueError(f"--num-gpus={a.num_gpus} but jax sees {jax.device_count()} devices")
    is_proc0 = jax.process_index() == 0
    print(f"process {jax.process_index()}/{jax.process_count()}, local devices {jax.local_device_count()}")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nh = a.hidden_dim // a.head_dim
    nkv = a.num_kv_heads or nh
    inter = a.hidden_dim // 2
    fp8 = None
    if a.fp8:
        # wire=None resolves with the recipe at model init (per_tensor: fp8 wire, mxfp8: bf16).
        wire = False if a.no_fp8_wire else None
        fp8 = GrugFp8Config(
            wire=wire,
            dense=not a.no_fp8_dense,
            grouped=not a.no_fp8_grouped,
            recipe=a.fp8_recipe,
            dense_recipe=a.fp8_dense_recipe,
            mxfp8_producer=a.mxfp8_producer,
            mxfp8_save_qweights=a.mxfp8_save_qweights,
        )
    model = GrugModelConfig(
        vocab_size=128256,
        hidden_dim=a.hidden_dim,
        num_layers=a.num_layers,
        num_heads=nh,
        num_kv_heads=nkv,
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
        use_array_stacked_blocks=a.stacked_blocks,
    )
    optimizer = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    opt = optimizer.build(a.steps)
    train_step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
    flops_per_example, flops_summary = _compute_flops(model_config=model)
    peak = a.num_gpus * _H100_BF16_PEAK_FLOPS
    mesh = compact_grug_mesh(expert_axis_size=a.expert_parallelism, replica_axis_size=a.replica_axis)
    metrics = []
    tps = a.batch_size * a.seq_len
    use_mxfp8_dense = fp8 is not None and fp8.dense and fp8.dense_recipe == "mxfp8"
    with set_mesh(mesh), mxfp8_dense_mesh_context(enabled=use_mxfp8_dense):

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
                "mfu_gb200": achieved / (a.num_gpus * _GB200_BF16_PEAK_FLOPS),
                "loss": float(loss),
            }
            metrics.append(m)
            if is_proc0:
                print(json.dumps(m, sort_keys=True), flush=True)
        if a.profile_steps > 0:
            prof_dir = out / "profiler"
            with jax.profiler.trace(str(prof_dir)):
                for step in range(a.steps, a.steps + a.profile_steps):
                    batch = _make_batch(a.batch_size, a.seq_len, model.vocab_size, step, mesh)
                    state, sm, _w = train_step(state, batch, compute_watch=False)
                    jax.block_until_ready(sm["train/loss"])
            if is_proc0:
                try:
                    from trace_phases import analyze  # noqa: PLC0415 (sibling module, GPU-job only)

                    analyze(prof_dir, steps=a.profile_steps, num_gpus=jax.local_device_count())
                except Exception:  # trace parsing must not lose the MFU result
                    traceback.print_exc()
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
                else {
                    "wire": fp8.wire,
                    "dense": fp8.dense,
                    "grouped": fp8.grouped,
                    "recipe": fp8.recipe,
                    "dense_recipe": fp8.dense_recipe,
                }
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
    if is_proc0:
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    multihost_utils.sync_global_devices("bench_grug_moe_mfu_fp8_done")


if __name__ == "__main__":
    main()
