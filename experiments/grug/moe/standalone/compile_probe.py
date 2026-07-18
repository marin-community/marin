# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AOT compile-only memory probe for the row-13 FP8 MFU harness (MXFP8-006).

Compiles the exact `train_step` of bench_grug_moe_mfu_fp8.py without ever
executing it (abstract avals with real output shardings from an AOT-compiled
`init`), then prints XLA's memory analysis and the largest temp buffers.
Diagnoses step-0 OOMs (e.g. the 851.61 GiB allocation at d5120/L48/B512
32-way, job mxfp8-006-ladder) without holding a full gang: no parameters or
activations are ever allocated.

Run on CPU with fake devices for graph structure, or on a small GPU
reservation for the real backend:

  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32 \\
    python compile_probe.py --num-gpus 32 --batch-size 512 \\
    --attention-implementation reference
"""

import argparse
import json
import time

import jax
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from iris.runtime.jax_init import initialize_jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugFp8Config, GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHConfig
from experiments.grug.moe.train import _make_train_step, initial_state

_BATCH_AXES = ("replica_dcn", "data", "expert")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--fp8-recipe", default="auto", choices=["auto", "per_tensor", "mxfp8"])
    p.add_argument("--no-fp8-wire", action="store_true")
    p.add_argument("--no-fp8-dense", action="store_true")
    p.add_argument("--no-fp8-grouped", action="store_true")
    p.add_argument("--mxfp8-producer", default="auto", choices=["auto", "cute", "xla"])
    p.add_argument("--mxfp8-save-qweights", action="store_true")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=5120)
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--num-experts-per-token", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-kv-heads", type=int, default=0, help="0 = MHA (num_heads)")
    p.add_argument("--num-gpus", type=int, default=32)
    p.add_argument("--moe-implementation", default="ring")
    p.add_argument("--expert-parallelism", type=int, default=8)
    p.add_argument("--replica-axis", type=int, default=1, help="replica_dcn axis size (DDP replicas)")
    p.add_argument("--attention-implementation", default="gpu_fa4_cute")
    p.add_argument("--remat-mode", default="recompute_all", choices=["recompute_all", "save_moe"])
    p.add_argument("--stacked-blocks", action="store_true", help="lax.scan over ArrayStacked blocks")
    p.add_argument("--optimizer", default="muonh", choices=["muonh", "adamh"])
    p.add_argument("--top-buffers", type=int, default=25)
    return p.parse_args()


def main():
    jax.config.update("jax_threefry_partitionable", True)
    a = _parse()
    initialize_jax()
    # Code-version markers: confirm the bundle actually carries the FA4-bounds fix
    # (levanter attention side + bench model side).
    import experiments.grug.moe.model as _model_mod  # noqa: PLC0415

    print(
        f"PROBE_CODE levanter_fa4_bounds={hasattr(AttentionMask, 'with_fa4_bounds')} "
        f"model_fa4_bounds={hasattr(_model_mod, 'fa4_cute_segment_bounds')}",
        flush=True,
    )
    if jax.device_count() != a.num_gpus:
        raise ValueError(f"--num-gpus={a.num_gpus} but jax sees {jax.device_count()} devices")
    fp8 = None
    if a.fp8:
        # wire=None resolves with the recipe at model init (per_tensor: fp8 wire, mxfp8: bf16).
        wire = False if a.no_fp8_wire else None
        fp8 = GrugFp8Config(
            wire=wire,
            dense=not a.no_fp8_dense,
            grouped=not a.no_fp8_grouped,
            recipe=a.fp8_recipe,
            mxfp8_producer=a.mxfp8_producer,
            mxfp8_save_qweights=a.mxfp8_save_qweights,
        )
    nh = a.hidden_dim // a.head_dim
    nkv = a.num_kv_heads or nh
    inter = a.hidden_dim // 2
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
    if a.optimizer == "muonh":
        optimizer = GrugMoeMuonHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)
    else:
        optimizer = GrugMoeAdamHConfig(learning_rate=1e-3, adam_lr=1e-4, min_lr_ratio=0.0, warmup=0.1)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    opt = optimizer.build(20)
    train_step = _make_train_step(opt, mp, z_loss_weight=1e-4, ema_beta=None, watch_config=None)
    mesh = compact_grug_mesh(expert_axis_size=a.expert_parallelism, replica_axis_size=a.replica_axis)

    with set_mesh(mesh):
        t0 = time.perf_counter()
        init_compiled = (
            jax.jit(lambda rng: initial_state(model, optimizer=opt, mp=mp, key=rng, ema_beta=None))
            .lower(jax.random.PRNGKey(0))
            .compile()
        )
        print(f"init compiled in {time.perf_counter() - t0:.1f}s", flush=True)
        state_shardings = init_compiled.output_shardings
        state_avals = jax.eval_shape(
            lambda rng: initial_state(model, optimizer=opt, mp=mp, key=rng, ema_beta=None), jax.random.PRNGKey(0)
        )
        state_abs = jax.tree.map(
            lambda aval, sh: jax.ShapeDtypeStruct(aval.shape, aval.dtype, sharding=sh),
            state_avals,
            state_shardings,
        )
        bs = NamedSharding(mesh, P(_BATCH_AXES, None))
        tok = jax.ShapeDtypeStruct((a.batch_size, a.seq_len), np.int32, sharding=bs)
        lw = jax.ShapeDtypeStruct((a.batch_size, a.seq_len), np.float32, sharding=bs)
        seg = jax.ShapeDtypeStruct((a.batch_size, a.seq_len), np.int32, sharding=bs)
        batch_abs = GrugLmExample(
            tokens=tok,
            loss_weight=lw,
            attn_mask=AttentionMask.causal(sliding_window=a.seq_len).with_segment_ids(seg),
        )
        t0 = time.perf_counter()
        lowered = jax.jit(lambda s, b: train_step(s, b, compute_watch=False)).lower(state_abs, batch_abs)
        print(f"train_step lowered in {time.perf_counter() - t0:.1f}s", flush=True)
        t0 = time.perf_counter()
        compiled = lowered.compile()
        print(f"train_step compiled in {time.perf_counter() - t0:.1f}s", flush=True)

    ma = compiled.memory_analysis()
    if ma is not None:
        stats = {
            k: getattr(ma, k) for k in dir(ma) if not k.startswith("_") and isinstance(getattr(ma, k), (int, float))
        }
        print("MEMORY_ANALYSIS " + json.dumps({k: f"{v / 2**30:.2f}GiB" if v > 2**20 else v for k, v in stats.items()}))

    _report_dump_buffers(a.top_buffers)
    print("PROBE_DONE", flush=True)


def _report_dump_buffers(top_n: int) -> None:
    """Aggregate large values from an XLA buffer-assignment dump, if one was requested.

    Set XLA_FLAGS to include ``--xla_dump_to=<dir> --xla_dump_hlo_as_text`` and
    pass the same dir via PROBE_DUMP_DIR; pod-local dumps die with the pod, so
    the attribution table is parsed here and printed to stdout.
    """
    import collections  # noqa: PLC0415
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415
    import re  # noqa: PLC0415

    dump_dir = os.environ.get("PROBE_DUMP_DIR")
    if not dump_dir:
        return
    cands = sorted(glob.glob(f"{dump_dir}/*buffer-assignment*"), key=os.path.getsize, reverse=True)
    if not cands:
        print(f"PROBE_DUMP: no buffer-assignment files under {dump_dir}")
        return
    # Allocation-level composition (true coexisting buffers): headers like
    # "allocation N: size S, ..." from the buffer-assignment file, with the
    # first value line naming the defining op/shape.
    for path in cands:
        if not path.endswith("buffer-assignment.txt"):
            continue
        allocs = []
        cur = None
        with open(path) as f:
            for line in f:
                mh = re.match(r"allocation (\d+): size (\d+), (.*)", line)
                if mh:
                    cur = [int(mh.group(2)), mh.group(3).strip()[:60], ""]
                    allocs.append(cur)
                elif cur is not None and cur[2] == "" and "value:" in line:
                    cur[2] = line.strip()[:150]
        allocs.sort(reverse=True)
        temp_total = sum(a[0] for a in allocs if "parameter" not in a[1])
        print(f"PROBE_ALLOCS file={os.path.basename(path)} n={len(allocs)} nonparam_total={temp_total/2**30:.1f}GiB")
        for size, kind, val in allocs[:top_n]:
            print(f"  {size/2**30:9.2f} GiB  {kind[:45]:45s} {val[:130]}", flush=True)
        break
    # Value census keyed on the defining HLO instruction, parsed from the
    # ``value: <ID name @0> (size=N,offset=M): shape`` lines. The op name
    # (all-gather vs fusion vs copy) says whether a big buffer is an SPMD
    # gather, a remat product, or a layout copy.
    for path in cands:
        if not path.endswith("buffer-assignment.txt"):
            continue
        byshape: collections.Counter = collections.Counter()
        counts: collections.Counter = collections.Counter()
        ops: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        with open(path) as f:
            for line in f:
                m = re.search(r"value: <\d+ ([\w.\-{}]+) @\d+> \(size=(\d+),offset=\d+\): (.+)", line)
                if not m:
                    continue
                name, size_s, shape = m.groups()
                size = int(size_s)
                if size < 256 * 2**20:
                    continue
                op = re.sub(r"[.\d]+$", "", name.rstrip("{}"))
                shape = shape.strip()[:60]
                byshape[shape] += size
                counts[shape] += 1
                ops[shape][op] += 1
        print(f"PROBE_DUMP file={os.path.basename(path)} matched={sum(counts.values())}")
        for shape, s in byshape.most_common(top_n):
            top_ops = ",".join(f"{o}x{c}" for o, c in ops[shape].most_common(4))
            print(f"  {s / 2**30:9.2f} GiB  n={counts[shape]:5d}  {shape}  [{top_ops}]", flush=True)
        break


if __name__ == "__main__":
    main()
