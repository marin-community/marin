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

``launch`` gates on ``preflight.py``, which traces every arm on CPU first.

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
PREFIX = "hs"  # hero sweep
PREFLIGHT_MODULE = "experiments.grug.moe_hero_fsdp.sweep_preflight"

# The configuration adopted after wave 4: FSDP-sharded small parameters, the gate/up interleave
# moved ahead of its all-gather, and NVLink SHARP for collectives.
COMBINED_ENV = {"NCCL_ALGO": "NVLS,Ring", "NCCL_NVLS_ENABLE": "1"}
COMBINED_ARGS = ["--small-param-sharding", "fsdp", "--interleave-before-gather"]
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
NCCL_MEMORY_FLAGS = "--xla_gpu_enable_nccl_comm_splitting=true --xla_gpu_enable_nccl_per_stream_comms=false"
# `xla_gpu_cudnn_gemm_fusion_level` is an int, not a bool, and the two other fusion switches this
# once carried no longer exist in XLA 0.11 (`enable_custom_fusions`, and
# `enable_address_computation_fusion`, whose successor `enable_dynamic_slice_fusion` is already on).
# Level 2 lets cuDNN take GEMM epilogues, which is the lever against the memory-bound elementwise
# kernels trailing the projections.
CUDNN_FUSION_FLAGS = "--xla_gpu_cudnn_gemm_fusion_level=2"
# The block-level (Triton) fusion emitter, off by default.
BLOCK_FUSION_FLAGS = "--xla_gpu_experimental_enable_fusion_block_level_rewriter=true"
# Autotuning at its limit: exhaustive tiling search rather than the default heuristic shortlist, at
# the highest autotune level. This buys compile time, which a 20-step run pays in full before the
# scored window opens at step 5. `xla_gpu_experimental_autotune_cache_mode` is not settable here --
# it takes no value form XLA_FLAGS accepts.
EXHAUSTIVE_AUTOTUNE_FLAGS = "--xla_gpu_exhaustive_tiling_search=true --xla_gpu_autotune_level=5"
_COMBINE_BYTES = 512 * 1024 * 1024
COMBINE_THRESHOLD_FLAGS = (
    " ".join(
        f"--xla_gpu_{collective}_combine_threshold_bytes={_COMBINE_BYTES}"
        for collective in ("all_gather", "all_reduce", "reduce_scatter")
    )
    + f" {_COMMAND_BUFFER_DISABLED}"
)
LOGDIR = pathlib.Path("scratch/hero_sweep")
PEAK_FLOPS_PER_DEVICE = 2.5e15
NUM_DEVICES = 64


# Early waves named the in-window baseline `control`; from wave 6 on it is `base`, the adopted
# configuration. Scoring against whichever arm happens to be fastest would turn cluster drift into a
# result, so the reference is always one of these and scoring fails loudly if a wave has neither.
CONTROL_TAGS = frozenset({"control", "base"})


class Arm:
    """One configuration under test: coordinator env plus launcher flags."""

    def __init__(self, tag, *, env=None, args=(), note=""):
        self.tag = tag
        self.env = env or {}
        self.args = list(args)
        self.note = note

    def run_id(self, wave):
        return f"{PREFIX}-{wave}-{self.tag}"


def wandb_run_exists(run_id):
    """W&B refuses to re-initialize a run id that a crashed attempt already claimed."""
    try:
        wandb.Api().run(f"{PROJECT}/{run_id}")
    except Exception:
        return False
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
    # as a remat mode and the hero simply does not use it, so this arm costs no implementation.
    "w2": [
        Arm("control"),
        Arm("shardsmall", args=["--small-param-sharding", "fsdp"], note="kill the per-layer all-reduce"),
        Arm("interleave", args=["--interleave-before-gather"], note="interleave w13 on the local shard"),
        Arm("savemoe", args=["--remat-mode", "save_moe"], note="probe the HBM ceiling; may OOM"),
    ],
    # Wave 3: confirm the wave-2 winners stack, and attack the two biggest remaining blocks of
    # compute-stream work that are reachable by a config knob. Two earlier attempts died in
    # startup and burned the `w3` run ids, so this wave carries a fresh one.
    "w3b": [
        Arm("control"),
        Arm(
            "combo",
            args=["--small-param-sharding", "fsdp", "--interleave-before-gather"],
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
    # `nccl_user_buffers` needs NCCL_NVLS_ENABLE=1, which the adopted configuration already sets.
    "w9": [
        Arm("base", env=COMBINED_ENV, args=COMBINED_ARGS, note="this wave's control"),
        Arm(
            "userbuffers",
            env={
                **COMBINED_ENV,
                "XLA_FLAGS": f"--xla_gpu_enable_nccl_user_buffers=true {_COMMAND_BUFFER_DISABLED}",
                "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": "2048",
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
                "XLA_FLAGS": f"--xla_gpu_enable_nccl_user_buffers=true {_COMMAND_BUFFER_DISABLED}",
                "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": "2048",
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
            str(STEPS),
            "--no-save-checkpoints",
            "--watch-interval",
            "0",
            "--version",
            "dev",
            "--run",
            *arm.args,
        ]
        log = LOGDIR / f"{rid}.log"
        with log.open("w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=900)
        shown = " ".join(f"{k}={v}" for k, v in arm.env.items()) or "(none)"
        print(f"submitted {rid:28s} env: {shown}")


def step_times(run_id):
    run = wandb.Api().run(f"{PROJECT}/{run_id}")
    hist = run.history(keys=["throughput/duration", "memory/peak_gib", "_step"], pandas=False, samples=1000)
    return sorted(
        (x["_step"], x["throughput/duration"], x.get("memory/peak_gib"))
        for x in hist
        if x.get("throughput/duration") is not None and x["_step"] >= WARMUP
    )


def score_run(run_id):
    rows = step_times(run_id)
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
        # The MFU denominator is fixed by shape, so MFU is a pure restatement of step time.
        "mfu": 100 * COUNTED_FLOPS / med / PEAK_FLOPS_PER_DEVICE / NUM_DEVICES,
    }


# Model FLOPs per step at the hero shape, from the run's own throughput/mfu and duration
# (19.4145% at 18.0223 s over 64 devices).
COUNTED_FLOPS = 0.194145 * 18.0223 * PEAK_FLOPS_PER_DEVICE * NUM_DEVICES


def score(wave):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in WAVES[wave]:
        try:
            s = score_run(arm.run_id(wave))
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
    control = next(r for r in results if r["tag"] in CONTROL_TAGS)
    reference = control["median"]
    print(f"\n{'arm':12s} {'n':>3} {'median':>9} {'MAD':>7} {'min':>8} {'MFU':>7} {'peak HBM':>9} {'vs control':>11}")
    for r in sorted(results, key=lambda r: r["median"]):
        delta = 100 * (reference - r["median"]) / reference
        hbm = f"{r['peak_gib']:.1f} GiB" if r["peak_gib"] is not None else "-"
        print(
            f"{r['tag']:12s} {r['n']:3d} {r['median']:8.3f}s {r['mad']:6.3f}s {r['min']:7.3f}s "
            f"{r['mfu']:6.2f}% {hbm:>9} {delta:+10.2f}%"
        )
    out = LOGDIR / f"{wave}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    action, wave = sys.argv[1], sys.argv[2]
    {"launch": launch, "score": score}[action](wave)
