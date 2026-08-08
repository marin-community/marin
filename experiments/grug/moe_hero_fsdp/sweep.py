# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch and score parallel single-rack hero configurations.

Measurement protocol
--------------------
Each arm runs ``STEPS`` steps of the 1-rack FSDP hero with checkpoints and watch dumps off.
Steps 0-1 are compile and the PGLE recompile; steps 2-4 absorb the one-time first-batch data-loader
stall. The score is the **median** ``throughput/duration`` over steps ``WARMUP..STEPS-1``, because
the steady-state distribution is right-skewed (run A: median 18.086 s, min 17.936, max 18.699,
MAD 0.077).

Every wave carries a ``control`` arm launched in the same window, so cluster drift cannot be read as
a win. An arm counts only if it beats the control by more than the control-to-control spread.

``launch`` gates on ``sweep_preflight.py``, which traces every arm on CPU first.

Usage
-----
    uv run python -m experiments.grug.moe_hero_fsdp.sweep launch <wave>
    uv run python -m experiments.grug.moe_hero_fsdp.sweep score <wave>
"""

import json
import os
import pathlib
import statistics
import subprocess
import sys

import wandb

CLUSTER = "cw-us-east-08a"
PROJECT = "marin-community/marin_moe"
STEPS = 20
WARMUP = 5  # first scored step
# 15 steps separates the multi-percent effects a screening wave looks for; the wrap-up waves that
# produce the reported number take 40.
WAVE_STEPS = {"final": 45, "final2": 45, "final3": 45, "final4": 45}
PREFIX = "hs"  # hero sweep
PREFLIGHT_MODULE = "experiments.grug.moe_hero_fsdp.sweep_preflight"

# The configuration adopted after wave 4, now the hero's own default. The pre-adoption hero is no
# longer expressible as an arm, so the early waves' `control` records what was measured rather than
# what re-running it today would produce.
COMBINED_ENV = {"NCCL_ALGO": "NVLS,Ring", "NCCL_NVLS_ENABLE": "1"}
COMBINED_ARGS = ["--small-param-sharding", "fsdp"]
# Wave 5 measured expert_chunks=2 at +1.63% on the unmodified hero. Wave 7 suggests it does not
# survive stacking with the adopted configuration, which wave 8 settles in-window.
CHUNKED_ARGS = [*COMBINED_ARGS, "--expert-chunks", "2"]
# `_apply_hero_fsdp_runtime_defaults` leaves XLA_FLAGS alone once it already names
# `--xla_gpu_enable_command_buffer`, so an arm can re-enable CUDA graphs without a code change.
COMMAND_BUFFER_FLAGS = "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL"
# Arms that set XLA_FLAGS for another reason must restate the hero's own disable, or they would
# silently enable CUDA graphs too and confound themselves with the `cudagraphs` arm.
_COMMAND_BUFFER_DISABLED = "--xla_gpu_enable_command_buffer="
# Already active on the EP hero (`experiments/grug/moe_hero_ep/train.py`), never on the FSDP one.
EP_SCHEDULER_FLAGS = (
    "--xla_gpu_enable_latency_hiding_scheduler=true " "--xla_gpu_experimental_parallel_collective_overlap_limit=4"
)
SOL_ESTIMATOR_FLAG = "--xla_gpu_enable_analytical_sol_latency_estimator=true"
# `xla_gpu_cudnn_gemm_fusion_level` is an int, not a bool, and the two other fusion switches this
# once carried no longer exist in XLA 0.11 (`enable_custom_fusions`, and
# `enable_address_computation_fusion`, whose successor `enable_dynamic_slice_fusion` is already on).
# Level 2 lets cuDNN take GEMM epilogues, which is the lever against the memory-bound elementwise
# kernels trailing the projections.
CUDNN_FUSION_FLAGS = "--xla_gpu_cudnn_gemm_fusion_level=2"
# The block-level (Triton) fusion emitter, off by default.
BLOCK_FUSION_FLAGS = "--xla_gpu_experimental_enable_fusion_block_level_rewriter=true"
# NCCL user buffers: registered buffers let NVLS skip a staging copy. The flag needs
# NCCL_NVLS_ENABLE=1, which the adopted configuration already sets, and a preallocated collective
# arena outside the client pool.
USER_BUFFER_FLAGS = "--xla_gpu_enable_nccl_user_buffers=true"
USER_BUFFER_POOL_MB = "2048"
# Exhaustive tiling search rather than the default heuristic shortlist, at the highest autotune
# level. Costs compile time, which a 20-step run pays before the scored window opens at step 5.
# `xla_gpu_experimental_autotune_cache_mode` takes no value XLA_FLAGS accepts, so it is untestable.
EXHAUSTIVE_AUTOTUNE_FLAGS = "--xla_gpu_exhaustive_tiling_search=true --xla_gpu_autotune_level=5"
# The components of `JAX_OPTIMIZATION_LEVEL=O1`, which measured -3.70% as a bundle. Its latency
# hiding scheduler is separately measured at -4.22%, so the bundle's loss is accounted for and its
# other three parts have never been read on their own.
DOUBLE_BUFFER_FLAGS = "--xla_gpu_enable_while_loop_double_buffering=true"
PIPELINED_COLLECTIVE_FLAGS = (
    "--xla_gpu_enable_pipelined_all_reduce=true "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true"
)
_COMBINE_COLLECTIVES = ("all_gather", "all_reduce", "reduce_scatter")


def _combine_threshold_flags(nbytes):
    """Raise the size at which XLA stops merging collectives, from the 31.5 MB default.

    One layer's FSDP shard is 231 MB, already cleared by the 512 MB setting that measured -3.37%.
    All 48 layers run inside a single `jax.scan`, so nothing merges across layers at any threshold.
    """
    return " ".join(f"--xla_gpu_{c}_combine_threshold_bytes={nbytes}" for c in _COMBINE_COLLECTIVES)


COMBINE_THRESHOLD_FLAGS = f"{_combine_threshold_flags(512 * 1024 * 1024)} {_COMMAND_BUFFER_DISABLED}"
BIG_COMBINE_THRESHOLD_FLAGS = _combine_threshold_flags(8 * 1024 * 1024 * 1024)
# `xla_gpu_enable_address_computation_fusion` was renamed; the successor defaults to false, so this
# is untested rather than already on.
DYNAMIC_SLICE_FUSION_FLAGS = "--xla_gpu_enable_dynamic_slice_fusion=true"
LOGDIR = pathlib.Path("scratch/hero_sweep")
PEAK_FLOPS_PER_DEVICE = 2.5e15
NUM_DEVICES = 64
HERO_BATCH_SIZE = 1024
HERO_SEQ_LEN = 4096


# Early waves named the in-window baseline `control`; from wave 6 on it is `base`, the adopted
# configuration. Scoring against whichever arm happens to be fastest would turn cluster drift into a
# result, so the reference is always one of these and scoring fails loudly if a wave has neither.
CONTROL_TAGS = frozenset({"control", "base"})


class Arm:
    """One configuration under test: coordinator env plus launcher flags."""

    def __init__(self, tag, *, env=None, args=(), note="", batch_size=HERO_BATCH_SIZE):
        self.tag = tag
        self.env = env or {}
        self.args = list(args)
        self.note = note
        self.batch_size = batch_size

    def run_id(self, wave):
        return f"{PREFIX}-{wave}-{self.tag}"


def wandb_run_exists(run_id):
    """W&B refuses to re-initialize a run id that a crashed attempt already claimed."""
    try:
        wandb.Api().run(f"{PROJECT}/{run_id}")
    except wandb.errors.CommError as exc:
        if "could not find run" in str(exc).lower():
            return False
        raise
    return True


WAVES = {
    # Wave 1: NCCL protocol and algorithm. No code change; the training gang inherits the
    # coordinator's environment. Nothing in the repo sets NCCL_PROTO or NCCL_ALGO.
    "w1": [
        Arm("control", note="unchanged baseline, same wall-clock window"),
        Arm("simple", env={"NCCL_PROTO": "Simple"}, note="drop LL's 8-byte-flag-per-8-byte overhead"),
        Arm("ll128", env={"NCCL_PROTO": "LL128"}, note="128-byte LL variant"),
        Arm("nvls", env={"NCCL_ALGO": "NVLS,Ring", "NCCL_NVLS_ENABLE": "1"}, note="NVLink SHARP"),
    ],
    # Wave 2: the two cheap code wins plus the activation-memory probe. `save_moe` already exists
    # as a remat mode; the hero does not use it.
    "w2": [
        Arm("control"),
        Arm("shardsmall", args=["--small-param-sharding", "fsdp"], note="kill the per-layer all-reduce"),
        Arm("savemoe", args=["--remat-mode", "save_moe"], note="probe the HBM ceiling; may OOM"),
    ],
    # Wave 3: confirm the wave-2 winners stack, and attack the two biggest remaining blocks of
    # compute-stream work that are reachable by a config knob. Two earlier attempts died in
    # startup and burned the `w3` run ids, so this wave carries a fresh one.
    "w3b": [
        Arm("control"),
        Arm(
            "combo",
            args=["--small-param-sharding", "fsdp"],
            note="both wave-2 winners together",
        ),
        Arm("cebig", args=["--ce-b-block-size", "8192"], note="8x fewer CE launches, larger GEMMs"),
        Arm("offloadmoe", args=["--remat-mode", "offload_moe"], note="park dispatch output on the host"),
    ],
    # Wave 4: the stacked winner measured end-to-end against its own control, plus the two
    # remaining single knobs.
    "w4": [
        Arm("control"),
        Arm(
            "combined",
            env=COMBINED_ENV,
            args=COMBINED_ARGS,
            note="every winner from waves 1-3, measured end to end",
        ),
        Arm("chunks8", args=["--expert-chunks", "8"], note="shorter gather prologue, smaller GEMMs"),
        Arm("chunks2", args=["--expert-chunks", "2"], note="fewer, larger expert GEMMs"),
    ],
    # Wave 5: bracket the expert-chunk count below the hero's 4, and replicate the headline
    # `combined` number in a fresh window against a fresh control.
    "w5": [
        Arm("control"),
        Arm(
            "combined",
            env=COMBINED_ENV,
            args=COMBINED_ARGS,
            note="replication of the wave-4 result",
        ),
        Arm("chunks2", args=["--expert-chunks", "2"], note="rerun; wave 4 lost a node mid-run"),
        Arm("chunks1", args=["--expert-chunks", "1"], note="no chunking, one gather of the full bank"),
    ],
    # Wave 6: stop capping HBM. Nothing sets XLA_PYTHON_CLIENT_MEM_FRACTION, so the allocator
    # offers 138.22 GiB of the ~184.3 GiB on the part, while XLA reports it cannot get peak below
    # 160.30 GiB without recomputing. Every arm carries the adopted `combined` configuration, so
    # `base` is this wave's control and the deltas are increments on top of it. `memory/limit_gib`
    # is logged per step, so an arm that silently ignores the knob is visible in the result.
    "w6": [
        Arm(
            "base",
            env=COMBINED_ENV,
            args=COMBINED_ARGS,
            note="adopted configuration; this wave's control",
        ),
        Arm(
            "memfrac88",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.88"},
            args=COMBINED_ARGS,
            note="162.2 GiB, just above the control's 160.30 GiB remat floor",
        ),
        Arm(
            "memfrac93",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93"},
            args=COMBINED_ARGS,
            note="171.4 GiB; may OOM, since NCCL and cuBLAS workspaces live outside the pool",
        ),
        Arm(
            "ceautotune",
            env={**COMBINED_ENV, "LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS": "1"},
            args=COMBINED_ARGS,
            note="tune cross-entropy block sizes for this shape instead of using the bucket default",
        ),
    ],
    # Wave 7: XLA and NCCL flags, on top of the adopted configuration plus `expert_chunks=2`.
    # 128 experts only divide into powers of two, so 2 is the single step below the hero's 4 and
    # the chunk count is settled once wave 5 confirms it.
    "w7": [
        Arm("base", env=COMBINED_ENV, args=CHUNKED_ARGS, note="this wave's control"),
        Arm(
            "cudagraphs",
            env={**COMBINED_ENV, "XLA_FLAGS": COMMAND_BUFFER_FLAGS},
            args=CHUNKED_ARGS,
            note="cross-entropy alone dispatches 43.7k kernels per step averaging 17.5 us",
        ),
        Arm(
            "combinethresh",
            env={**COMBINED_ENV, "XLA_FLAGS": COMBINE_THRESHOLD_FLAGS},
            args=CHUNKED_ARGS,
            note="the profile shows 1,009 uncombined collective launches at 52 GB/s",
        ),
        Arm(
            "epflags",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{EP_SCHEDULER_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=CHUNKED_ARGS,
            note="the two scheduler flags the EP hero already runs in production",
        ),
    ],
    # Wave 8: NVIDIA's JAX-Toolbox Blackwell guidance, none of which this repo sets today.
    # `O1` is the interesting one: it bundles the latency-hiding scheduler, pipelined collectives,
    # while-loop double buffering, and the analytical SOL latency estimator. That estimator is the
    # direct substitute for PGLE, which has never produced a non-empty trace on this cluster, so
    # the scheduler has been working from static cost estimates the whole time.
    "w8": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="adopted configuration; this wave's control"),
        Arm(
            "chunkstack",
            env=COMBINED_ENV,
            args=CHUNKED_ARGS,
            note="expert_chunks=2 on top of the adopted configuration, in-window",
        ),
        Arm(
            "o1",
            env={**COMBINED_ENV, "JAX_OPTIMIZATION_LEVEL": "O1"},
            args=COMBINED_ARGS,
            note="the O1 optimization bundle, including the analytical SOL latency estimator",
        ),
        Arm(
            "solonly",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{SOL_ESTIMATOR_FLAG} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="the SOL estimator alone, to separate it from the rest of O1",
        ),
    ],
    # Wave 9: zero-copy collectives, the B200 launch-mode workaround, and the fusion flags.
    "w9": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "userbuffers",
            env={
                **COMBINED_ENV,
                "XLA_FLAGS": f"{USER_BUFFER_FLAGS} {_COMMAND_BUFFER_DISABLED}",
                "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": USER_BUFFER_POOL_MB,
            },
            args=COMBINED_ARGS,
            note="zero-copy collectives with a preallocated user-buffer pool",
        ),
        Arm(
            "solonly",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{SOL_ESTIMATOR_FLAG} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="rerun; wave 8's attempt never got past a node fault",
        ),
        Arm(
            "fusions",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{CUDNN_FUSION_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="dead: shipped three flag names XLA 0.11 does not have; rerun as w10 cudnnfusion",
        ),
    ],
    # Wave 10: confirm the one candidate win, retry the arm iris#7650 killed, and try the fusion
    # flags under their real names. `solonly` measured +1.10% in wave 9, close enough to the
    # control-to-control drift that a second in-window reading decides it.
    "w10": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "solonly",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{SOL_ESTIMATOR_FLAG} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="confirmation of wave 9's +1.10%",
        ),
        Arm(
            "cudnnfusion",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{CUDNN_FUSION_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="cuDNN takes GEMM epilogues, against 5.00 s/step of memory-bound fusions",
        ),
        Arm(
            "userbuffers",
            env={
                **COMBINED_ENV,
                "XLA_FLAGS": f"{USER_BUFFER_FLAGS} {_COMMAND_BUFFER_DISABLED}",
                "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": USER_BUFFER_POOL_MB,
            },
            args=COMBINED_ARGS,
            note="rerun; zero-copy collectives with a preallocated user-buffer pool",
        ),
    ],
    # Wave 11: the cross-entropy kernel's own tiling, the block-level fusion emitter, and the SOL
    # estimator stacked on whichever of wave 10's arms won.
    "w11": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "ceautotune",
            env={**COMBINED_ENV, "LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS": "1"},
            args=COMBINED_ARGS,
            note="rerun; let the cross-entropy kernel autotune its tiling instead of taking the cached miss",
        ),
        Arm(
            "blockfusion",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{BLOCK_FUSION_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="the block-level Triton fusion emitter, off by default",
        ),
        Arm(
            "exhaustive",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{EXHAUSTIVE_AUTOTUNE_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="autotuning turned up as far as XLA goes; compile time falls outside the scored window",
        ),
    ],
    # Wave 12: decompose `JAX_OPTIMIZATION_LEVEL=O1`. Its latency hiding scheduler already measured
    # -4.22% on its own, which accounts for the bundle's -3.70%, leaving its other components
    # unread. `doublebuffer` matters most: the hero runs all 48 layers under one `jax.scan`, and
    # double-buffering unrolls that loop twice, which is the cheap partial form of breaking it.
    "w12": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "doublebuffer",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{DOUBLE_BUFFER_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="2x unroll of the 48-layer scan, against 5.09 s/step of recompute",
        ),
        Arm(
            "pipelined",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{PIPELINED_COLLECTIVE_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="collective pipelining, the untested flags for the MoE all-gather candidate",
        ),
        Arm(
            "o1nolhs",
            env={
                **COMBINED_ENV,
                "XLA_FLAGS": (
                    f"{DOUBLE_BUFFER_FLAGS} {PIPELINED_COLLECTIVE_FLAGS} "
                    f"{SOL_ESTIMATOR_FLAG} {_COMMAND_BUFFER_DISABLED}"
                ),
            },
            args=COMBINED_ARGS,
            note="O1 without its latency hiding scheduler: everything in the bundle we have not ruled out",
        ),
    ],
    # Wave 13: the remaining flags that XLA reports as default-off and that plausibly touch this
    # graph (`--xla_gpu_dump_defaults` lists them). `userbuffers` gets a fourth attempt after losing
    # three racks to iris#7650.
    "w13": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "dsfusion",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{DYNAMIC_SLICE_FUSION_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="the renamed address-computation fusion, default off",
        ),
        Arm(
            "combine8g",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{BIG_COMBINE_THRESHOLD_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="8 GB thresholds, 37x a layer shard, against the 512 MB arm's -3.37%",
        ),
        Arm(
            "userbuffers",
            env={
                **COMBINED_ENV,
                "XLA_FLAGS": f"{USER_BUFFER_FLAGS} {_COMMAND_BUFFER_DISABLED}",
                "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": USER_BUFFER_POOL_MB,
            },
            args=COMBINED_ARGS,
            note="fourth attempt; the first three never reached step 0",
        ),
    ],
    # Wave 14: every remaining latency-hiding and collective flag is dropped. Six independent
    # measurements agree that a 90.3% compute-busy step has nothing for that family to recover, so
    # `p2ppermute` and `unrollcombine` were retired unrun.
    #
    # `doublebuffer` is a rerun. Its wave-12 attempt compiled for 22 minutes against the control's
    # 3.5, then lost every rank's coordinator connection at 24 minutes. That launch predates the
    # #7994 cherry-pick, so it still bound the fixed port 8476.
    #
    # The batch arms spend the HBM headroom on tokens. Peak sits at 137.2 GiB against a 138.22 GiB
    # ceiling, and raising the allocator fraction lifts that to 162.18 GiB at no cost of its own.
    # Activations scale with the batch while weights and optimizer state do not. These arms do more
    # work per step, so step time is not comparable and scoring switches to tokens/s; both batches
    # divide the 64-device mesh evenly.
    "w14": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control, batch 1024"),
        Arm(
            "doublebuffer",
            env={**COMBINED_ENV, "XLA_FLAGS": f"{DOUBLE_BUFFER_FLAGS} {_COMMAND_BUFFER_DISABLED}"},
            args=COMBINED_ARGS,
            note="rerun; 2x unroll of the 48-layer scan, against 5.09 s/step of recompute",
        ),
        Arm(
            "batch1152",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.88"},
            args=COMBINED_ARGS,
            note="+12.5% sequences, 18 per device",
            batch_size=1152,
        ),
        Arm(
            "batch1280",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93"},
            args=COMBINED_ARGS,
            note="+25% sequences, 20 per device",
            batch_size=1280,
        ),
    ],
    # The wrap-up. Everything the sweep adopted, measured end to end against the hero as it was,
    # over 40 scored steps instead of 15, on the merged main.
    #
    # `control2` is a byte-identical second control. Every arm in a wave lands on a different rack,
    # so placement variance has been confounded with the effect under test throughout; four
    # cross-wave controls put it at 1.28% peak to peak, but that also carries time. Two controls in
    # one window measure placement alone, which is the number every delta here should be read
    # against.
    "final": [
        Arm("control", note="the hero as it was before this sweep"),
        Arm("control2", note="byte-identical to control; measures placement variance in-window"),
        Arm(
            "adopted",
            env=COMBINED_ENV,
            args=COMBINED_ARGS,
            note="FSDP-sharded small parameters, interleave before gather, NVLink SHARP",
        ),
        Arm(
            "adoptedbatch",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.88"},
            args=COMBINED_ARGS,
            note="batch 1152 under a 0.88 ceiling",
            batch_size=1152,
        ),
    ],
    # The wrap-up, second attempt. `control` and `adoptedbatch` both lost their windows above, and
    # `adoptedbatch` ran at 40 s/step against its siblings' 17: merged main raised the baseline peak
    # from 137.2 to 141.2 GiB, so batch 1152 sits close enough to the 0.88 limit for XLA's
    # rematerialization to start buying memory with recompute. Retried at 0.93 for the slack.
    "final2": [
        Arm("control", note="the hero as it was before this sweep"),
        Arm("control2", note="byte-identical to control; measures placement variance in-window"),
        Arm(
            "adopted",
            env=COMBINED_ENV,
            args=COMBINED_ARGS,
            note="replicate of the wrap-up result",
        ),
        Arm(
            "adoptedbatch",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93"},
            args=COMBINED_ARGS,
            note="rerun at 0.93; 0.88 left only 6 GiB of slack under the merged topology",
            batch_size=1152,
        ),
    ],
    # Batch 1152 at 0.93 dies at step 4 inside `jit_train_step`. Batch 1088 is the remaining step:
    # +6.25% tokens, well clear of the 0.78% placement variance.
    #
    # Only one rank raises. The other fifteen sit in the shutdown barrier for its full five minutes
    # and then report `INTERNAL` with no traceback.
    "final3": [
        Arm(
            "control",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93"},
            args=COMBINED_ARGS,
            note="the adopted config at batch 1024; isolates batch from everything else adopted changes",
        ),
        Arm(
            "adoptedbatch",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.93"},
            args=COMBINED_ARGS,
            note="1088 is the largest batch left untried after 1152 exhausted HBM",
            batch_size=1088,
        ),
    ],
    # Dead end. 0.88 and 0.93 both fail at batch 1024 with the same 122.10 GiB `jit_train_step`
    # request, so the allocation is a fixed requirement of the un-rematerialized program rather than
    # something sized to the ceiling. Peak against the allocator limit decides whether
    # `HloRematerialization` runs; post-merge the hero clears 138.22 GiB and survives only because
    # that pass shrinks it. Any ceiling high enough to switch it off exposes an allocation the pool
    # cannot serve. Batch 1152 runs at 0.88 only because its 153.14 GiB peak keeps remat engaged, at
    # 40 s/step.
    "final4": [
        Arm(
            "control",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.88"},
            args=COMBINED_ARGS,
            note="batch 1024 at the same ceiling as its arm",
        ),
        Arm(
            "adoptedbatch",
            env={**COMBINED_ENV, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.88"},
            args=COMBINED_ARGS,
            note="+6.25% tokens, against 1152's rematerialization thrash at this ceiling",
            batch_size=1088,
        ),
    ],
}


def launch(wave):
    arms = WAVES[wave]
    if subprocess.run([sys.executable, "-m", PREFLIGHT_MODULE, wave], check=False).returncode:
        raise SystemExit(f"{wave} failed preflight; not spending four racks on it")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    # An empty key satisfies a plain lookup and then fails 6 minutes in, on every rack at once,
    # inside `trainer.initialize()` with `UsageError: No API key configured`.
    wandb_key = os.environ["WANDB_API_KEY"]
    if not wandb_key.strip():
        raise SystemExit("WANDB_API_KEY is empty; the training gang would fail at wandb.init")
    taken = [arm.tag for arm in arms if wandb_run_exists(arm.run_id(wave))]
    if taken:
        raise SystemExit(f"{wave} run ids already exist on W&B ({', '.join(taken)}); relaunch under a new wave")
    for arm in arms:
        rid = arm.run_id(wave)
        env_flags = []
        for k, v in {"WANDB_API_KEY": wandb_key, **arm.env}.items():
            env_flags += ["-e", k, v]
        cmd = [
            "uv",
            "run",
            "iris",
            f"--cluster={CLUSTER}",
            "job",
            "run",
            "--no-wait",
            "--enable-extra-resources",
            "--cpu",
            "2",
            "--memory",
            "8GB",
            "--disk",
            "32GB",
            "--timeout",
            "3600",
            "--priority",
            "production",
            "--job-name",
            f"{rid}-coord",
            *env_flags,
            "--",
            "python",
            "-m",
            "experiments.grug.moe_hero_fsdp.launch",
            "--run-id",
            rid,
            "--dp-racks",
            "1",
            "--num-steps",
            str(WAVE_STEPS.get(wave, STEPS)),
            "--no-save-checkpoints",
            "--watch-interval",
            "0",
            "--version",
            "dev",
            "--batch-size",
            str(arm.batch_size),
            "--run",
            *arm.args,
        ]
        log = LOGDIR / f"{rid}.log"
        with log.open("w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=900)
        shown = " ".join(f"{k}={v}" for k, v in arm.env.items()) or "(none)"
        print(f"submitted {rid:28s} env: {shown}")


def scored_steps(run_id):
    """Return ``(step, duration, peak_gib)`` for every step in the scored window, ordered by step."""
    run = wandb.Api().run(f"{PROJECT}/{run_id}")
    hist = run.history(keys=["throughput/duration", "memory/peak_gib", "_step"], pandas=False, samples=1000)
    return sorted(
        (x["_step"], x["throughput/duration"], x.get("memory/peak_gib"))
        for x in hist
        if x.get("throughput/duration") is not None and x["_step"] >= WARMUP
    )


def score_run(run_id, batch_size=HERO_BATCH_SIZE):
    rows = scored_steps(run_id)
    if len(rows) < 3:
        return None
    d = [t for _, t, _ in rows]
    hbm = [m for _, _, m in rows if m is not None]
    med = statistics.median(d)
    return {
        "run_id": run_id,
        "n": len(d),
        "median": med,
        "mad": statistics.median([abs(x - med) for x in d]),
        "min": min(d),
        "max": max(d),
        "peak_gib": max(hbm) if hbm else None,
        "batch_size": batch_size,
        # At a fixed shape MFU is a pure restatement of step time. An arm that changes the batch
        # does more work per step, so its FLOPs scale with the batch and step time stops being
        # comparable; tokens/s is the metric that survives both cases.
        "mfu": 100 * COUNTED_FLOPS * batch_size / HERO_BATCH_SIZE / med / PEAK_FLOPS_PER_DEVICE / NUM_DEVICES,
        "tokens_per_s": batch_size * HERO_SEQ_LEN / med,
    }


# Model FLOPs per step at the hero shape, from the run's own throughput/mfu and duration
# (19.4145% at 18.0223 s over 64 devices).
COUNTED_FLOPS = 0.194145 * 18.0223 * PEAK_FLOPS_PER_DEVICE * NUM_DEVICES


def score(wave):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in WAVES[wave]:
        try:
            s = score_run(arm.run_id(wave), batch_size=arm.batch_size)
        except Exception as exc:
            print(f"{arm.tag:12s} unavailable: {type(exc).__name__}: {exc}")
            continue
        if s is None:
            print(f"{arm.tag:12s} too few scored steps yet")
            continue
        s["tag"] = arm.tag
        s["note"] = arm.note
        results.append(s)

    if not results:
        return
    control = next((r for r in results if r["tag"] in CONTROL_TAGS), None)
    if control is None:
        scored = ", ".join(r["tag"] for r in results) or "none"
        print(f"\nno control arm has scored steps yet (scored so far: {scored})")
        return
    # Step time is only comparable when every arm processes the same tokens. A batch-size arm is
    # scored on tokens/s instead, where a slower but larger step can still be a win.
    varies_batch = len({r["batch_size"] for r in results}) > 1
    metric = "tokens_per_s" if varies_batch else "median"
    reference = control[metric]
    if varies_batch:
        print(f"scoring on tokens/s: arms differ in batch size (control = {control['batch_size']})")
    print(
        f"\n{'arm':12s} {'n':>3} {'batch':>6} {'median':>9} {'MAD':>7} "
        f"{'Mtok/s':>7} {'MFU':>7} {'peak HBM':>9} {'vs control':>11}"
    )
    for r in sorted(results, key=lambda r: -r[metric] if varies_batch else r[metric]):
        delta = 100 * (r[metric] - reference) / reference
        if metric == "median":  # lower is better
            delta = -delta
        hbm = f"{r['peak_gib']:.1f} GiB" if r["peak_gib"] is not None else "-"
        print(
            f"{r['tag']:12s} {r['n']:3d} {r['batch_size']:6d} {r['median']:8.3f}s {r['mad']:6.3f}s "
            f"{r['tokens_per_s'] / 1e6:7.3f} {r['mfu']:6.2f}% {hbm:>9} {delta:+10.2f}%"
        )
    out = LOGDIR / f"{wave}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    action, wave = sys.argv[1], sys.argv[2]
    {"launch": launch, "score": score}[action](wave)
