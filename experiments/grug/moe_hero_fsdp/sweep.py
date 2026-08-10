# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep single-rack hero configurations, one rack per arm.

Measurement protocol
--------------------
Each arm in ``ARMS`` is one step on its own rack, running in a fresh trainer subprocess so
process-start environment takes effect, writing the W&B run ``hs-<arm>``. The score is the
**median** ``throughput/duration`` over steps ``WARMUP..``, because the steady-state distribution is
right-skewed (run A: median 18.086 s, min 17.936, max 18.699, MAD 0.077). Steps 0-1 are compile and
the PGLE recompile; steps 2-4 absorb the one-time first-batch data-loader stall.

An arm scores against the ``baseline`` it names, and only against a baseline launched alongside it:
placement moved two byte-identical arms 0.78% apart, which is the resolution floor here. A single
paired reading resolves 1.57% at 95%.

Usage
-----
    uv run python -m experiments.grug.moe_hero_fsdp.sweep launch <arm>... --version dev --run
    uv run python -m experiments.grug.moe_hero_fsdp.sweep score <arm>...

An ``<arm>`` is an arm name or a name prefix, and every selection pulls in the baselines it needs.
Reruns resume: an arm already built under the given ``--version`` is skipped, so ``--version dev``
(mutable, rebuilds everything) is for iterating and a calendar version is for spending racks.
"""

import dataclasses
import json
import pathlib
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import click
import wandb
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from levanter.recovery.types import AblationSpec
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import HeroOverrides, HeroSweepArm, build_hero_sweep_run
from experiments.grug.moe_hero_fsdp.model import RematMode

PROJECT = "marin-community/marin_moe"
# 15 scored steps separates the multi-percent effects a screening arm looks for; the wrap-up arms
# that produce the reported number take 40.
STEPS = 20
WRAPUP_STEPS = 45
WARMUP = 5  # first scored step
PREFIX = "hs"  # hero sweep
LOGDIR = pathlib.Path("scratch/hero_sweep")
PEAK_FLOPS_PER_DEVICE = 2.5e15
DP_RACKS = 1
# HERO_NODES_PER_RACK * HERO_GPUS_PER_TASK in the launcher.
DEVICES_PER_RACK = 64
NUM_DEVICES = DP_RACKS * DEVICES_PER_RACK
HERO_BATCH_SIZE = 1024
HERO_SEQ_LEN = 4096

_COMMAND_BUFFER = "--xla_gpu_enable_command_buffer"
COMMAND_BUFFERS_OFF = f"{_COMMAND_BUFFER}="
COMMAND_BUFFERS_ON = f"{_COMMAND_BUFFER}=FUSION,CUSTOM_CALL"


@dataclass(frozen=True)
class Setting:
    """One named knob -- process env, XLA flags, model overrides -- composable with ``|``.

    Composition is left-to-right: a later setting's env keys and model overrides win, and its XLA
    flags append. ``ADOPTED | SOL_ESTIMATOR`` reads as the arm it names.
    """

    env: Mapping[str, str] = field(default_factory=dict)
    xla_flags: tuple[str, ...] = ()
    overrides: HeroOverrides = field(default_factory=HeroOverrides)

    def __or__(self, other: "Setting") -> "Setting":
        merged = {}
        for f in dataclasses.fields(HeroOverrides):
            value = getattr(other.overrides, f.name)
            merged[f.name] = value if value is not None else getattr(self.overrides, f.name)
        return Setting(
            env={**self.env, **other.env},
            xla_flags=self.xla_flags + other.xla_flags,
            overrides=HeroOverrides(**merged),
        )

    def process_env(self) -> dict[str, str]:
        """Process-start environment, with ``xla_flags`` materialized into ``XLA_FLAGS``.

        The hero disables CUDA graphs at startup only when ``XLA_FLAGS`` does not already name the
        flag, so anything that sets ``XLA_FLAGS`` for another reason has to carry the disable or
        silently turn CUDA graphs back on. Appending it here is what keeps that from being every
        arm's problem.
        """
        if not self.xla_flags:
            return dict(self.env)
        flags = self.xla_flags
        if not any(flag.startswith(_COMMAND_BUFFER) for flag in flags):
            flags = (*flags, COMMAND_BUFFERS_OFF)
        return {**self.env, "XLA_FLAGS": " ".join(flags)}


def expert_chunks(count: int) -> Setting:
    return Setting(overrides=HeroOverrides(expert_chunks=count))


def remat(mode: RematMode) -> Setting:
    return Setting(overrides=HeroOverrides(remat_mode=mode))


def ce_block(tokens: int) -> Setting:
    return Setting(overrides=HeroOverrides(ce_b_block_size=tokens))


def mem_fraction(fraction: str) -> Setting:
    """Raise the allocator ceiling from JAX's 0.75 default (138.22 GiB of the ~184.3 GiB part).

    On merged main any value that lifts peak clear of `HloRematerialization` exposes a fixed
    122.10 GiB `jit_train_step` allocation the pool cannot serve, so 0.88 and 0.93 both abort.
    """
    return Setting(env={"XLA_PYTHON_CLIENT_MEM_FRACTION": fraction})


_COMBINE_COLLECTIVES = ("all_gather", "all_reduce", "reduce_scatter")


def combine_threshold(nbytes: int) -> Setting:
    """Raise the size at which XLA stops merging collectives, from the 31.5 MB default.

    One layer's FSDP shard is 231 MB, already cleared by the 512 MB setting that measured -3.37%.
    All 48 layers run inside a single `jax.scan`, so nothing merges across layers at any threshold.
    """
    return Setting(xla_flags=tuple(f"--xla_gpu_{c}_combine_threshold_bytes={nbytes}" for c in _COMBINE_COLLECTIVES))


NVLS = Setting(env={"NCCL_ALGO": "NVLS,Ring", "NCCL_NVLS_ENABLE": "1"})
NCCL_SIMPLE = Setting(env={"NCCL_PROTO": "Simple"})
NCCL_LL128 = Setting(env={"NCCL_PROTO": "LL128"})
SHARD_SMALL = Setting(overrides=HeroOverrides(small_param_sharding="fsdp"))
# Adopted after wave 4 and now the hero's own default, so restating it is a no-op. The pre-adoption
# hero is no longer expressible as an arm, which is why the early waves' `control` records what was
# measured rather than what re-running it today would produce.
ADOPTED = NVLS | SHARD_SMALL
CHUNKED = ADOPTED | expert_chunks(2)
CE_AUTOTUNE = Setting(env={"LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS": "1"})
COMMAND_BUFFERS = Setting(xla_flags=(COMMAND_BUFFERS_ON,))
# Already active on the EP hero (`experiments/grug/moe_hero_ep/train.py`), never on the FSDP one.
EP_SCHEDULER = Setting(
    xla_flags=(
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_experimental_parallel_collective_overlap_limit=4",
    )
)
SOL_ESTIMATOR = Setting(xla_flags=("--xla_gpu_enable_analytical_sol_latency_estimator=true",))
# `xla_gpu_cudnn_gemm_fusion_level` is an int, not a bool, and the two other fusion switches this
# once carried no longer exist in XLA 0.11 (`enable_custom_fusions`, and
# `enable_address_computation_fusion`, whose successor `enable_dynamic_slice_fusion` is already on).
# Level 2 lets cuDNN take GEMM epilogues, against the memory-bound elementwise kernels trailing the
# projections.
CUDNN_FUSION = Setting(xla_flags=("--xla_gpu_cudnn_gemm_fusion_level=2",))
BLOCK_FUSION = Setting(xla_flags=("--xla_gpu_experimental_enable_fusion_block_level_rewriter=true",))
# `xla_gpu_enable_address_computation_fusion` was renamed; the successor defaults to false, so this
# is untested rather than already on.
DYNAMIC_SLICE_FUSION = Setting(xla_flags=("--xla_gpu_enable_dynamic_slice_fusion=true",))
# Registered buffers let NVLS skip a staging copy. Needs NCCL_NVLS_ENABLE=1, which ADOPTED sets, and
# a collective arena outside the client pool.
USER_BUFFERS = Setting(
    xla_flags=("--xla_gpu_enable_nccl_user_buffers=true",),
    env={"XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB": "2048"},
)
# Exhaustive tiling search rather than the default heuristic shortlist, at the highest autotune
# level. Costs compile time, which a run pays before the scored window opens at step 5.
# `xla_gpu_experimental_autotune_cache_mode` takes no value XLA_FLAGS accepts, so it is untestable.
EXHAUSTIVE_AUTOTUNE = Setting(xla_flags=("--xla_gpu_exhaustive_tiling_search=true", "--xla_gpu_autotune_level=5"))
# The components of `JAX_OPTIMIZATION_LEVEL=O1`, which measured -3.70% as a bundle. Its latency
# hiding scheduler measured -4.22% alone, which accounts for the bundle, leaving the rest unread.
O1 = Setting(env={"JAX_OPTIMIZATION_LEVEL": "O1"})
DOUBLE_BUFFER = Setting(xla_flags=("--xla_gpu_enable_while_loop_double_buffering=true",))
PIPELINED_COLLECTIVES = Setting(
    xla_flags=(
        "--xla_gpu_enable_pipelined_all_reduce=true",
        "--xla_gpu_enable_pipelined_all_gather=true",
        "--xla_gpu_enable_pipelined_reduce_scatter=true",
    )
)


@dataclass(frozen=True)
class Arm:
    """One configuration under test, on its own rack, scored against ``baseline``.

    ``name`` is globally unique and fixes the W&B run id, so a rerun of an arm needs a new name.
    ``baseline`` is ``None`` on the arms other arms are scored against.
    """

    name: str
    setting: Setting = Setting()
    baseline: str | None = None
    note: str = ""
    batch_size: int = HERO_BATCH_SIZE
    steps: int = STEPS

    @property
    def run_id(self) -> str:
        return f"{PREFIX}-{self.name}"

    def sweep_arm(self) -> HeroSweepArm:
        return HeroSweepArm(
            spec=AblationSpec(name=self.name, env=self.setting.process_env(), num_steps=self.steps, notes=self.note),
            run_id=self.run_id,
            overrides=self.setting.overrides,
            batch_size=self.batch_size,
        )


def _catalog(*arms: Arm) -> dict[str, Arm]:
    by_name: dict[str, Arm] = {}
    for arm in arms:
        if arm.name in by_name:
            raise ValueError(f"duplicate arm name {arm.name!r}; a rerun needs its own name")
        if arm.baseline is not None and arm.baseline not in by_name:
            raise ValueError(f"{arm.name!r} is scored against {arm.baseline!r}, which is not defined above it")
        by_name[arm.name] = arm
    return by_name


# Names group by the round that launched them. That prefix is a naming convention and a selector,
# not a structure: what an arm is scored against is `baseline`, and nothing else pairs arms.
ARMS = _catalog(
    # Round 1: NCCL protocol and algorithm. Nothing in the repo sets NCCL_PROTO or NCCL_ALGO.
    Arm("w1-control", note="unchanged baseline"),
    Arm("w1-simple", NCCL_SIMPLE, "w1-control", "drop LL's 8-byte-flag-per-8-byte overhead"),
    Arm("w1-ll128", NCCL_LL128, "w1-control", "128-byte LL variant"),
    Arm("w1-nvls", NVLS, "w1-control", "NVLink SHARP"),
    # Round 2: the two cheap code wins plus the activation-memory probe. `save_moe` already exists
    # as a remat mode; the hero does not use it.
    Arm("w2-control"),
    Arm("w2-shardsmall", SHARD_SMALL, "w2-control", "kill the per-layer all-reduce"),
    Arm("w2-savemoe", remat("save_moe"), "w2-control", "probe the HBM ceiling; may OOM"),
    # Round 3: confirm the round-2 winners stack, and attack the two biggest remaining blocks of
    # compute-stream work reachable by a config knob. Named `w3b` because two earlier attempts died
    # in startup and burned the `w3` run ids.
    Arm("w3b-control"),
    Arm("w3b-combo", SHARD_SMALL, "w3b-control", "both round-2 winners together"),
    Arm("w3b-cebig", ce_block(8192), "w3b-control", "8x fewer CE launches, larger GEMMs"),
    Arm("w3b-offloadmoe", remat("offload_moe"), "w3b-control", "park dispatch output on the host"),
    # Round 4: the stacked winner measured end to end against its own control, plus the two
    # remaining single knobs.
    Arm("w4-control"),
    Arm("w4-combined", ADOPTED, "w4-control", "every winner from rounds 1-3, measured end to end"),
    Arm("w4-chunks8", expert_chunks(8), "w4-control", "shorter gather prologue, smaller GEMMs"),
    Arm("w4-chunks2", expert_chunks(2), "w4-control", "fewer, larger expert GEMMs"),
    # Round 5: bracket the expert-chunk count below the hero's 4, and replicate the headline
    # `combined` number against a fresh control.
    Arm("w5-control"),
    Arm("w5-combined", ADOPTED, "w5-control", "replication of the round-4 result"),
    Arm("w5-chunks2", expert_chunks(2), "w5-control", "rerun; round 4 lost a node mid-run"),
    Arm("w5-chunks1", expert_chunks(1), "w5-control", "no chunking, one gather of the full bank"),
    # Round 6: stop capping HBM. XLA reports it cannot get peak below 160.30 GiB without recomputing.
    # `memory/limit_gib` is logged per step, so an arm that silently ignores the knob is visible.
    Arm("w6-base", ADOPTED, note="adopted configuration"),
    Arm("w6-memfrac88", ADOPTED | mem_fraction("0.88"), "w6-base", "162.2 GiB, just above the remat floor"),
    Arm("w6-memfrac93", ADOPTED | mem_fraction("0.93"), "w6-base", "171.4 GiB; NCCL and cuBLAS live outside the pool"),
    Arm("w6-ceautotune", ADOPTED | CE_AUTOTUNE, "w6-base", "tune CE block sizes for this shape"),
    # Round 7: XLA and NCCL flags on the adopted configuration plus `expert_chunks=2`. 128 experts
    # only divide into powers of two, so 2 is the single step below the hero's 4.
    Arm("w7-base", CHUNKED),
    Arm("w7-cudagraphs", CHUNKED | COMMAND_BUFFERS, "w7-base", "CE alone dispatches 43.7k kernels/step at 17.5 us"),
    Arm(
        "w7-combinethresh",
        CHUNKED | combine_threshold(512 * 1024 * 1024),
        "w7-base",
        "the profile shows 1,009 uncombined collective launches at 52 GB/s",
    ),
    Arm("w7-epflags", CHUNKED | EP_SCHEDULER, "w7-base", "the two scheduler flags the EP hero runs in production"),
    # Round 8: NVIDIA's JAX-Toolbox Blackwell guidance, none of which this repo sets. `O1` bundles the
    # latency-hiding scheduler, pipelined collectives, double buffering, and the SOL estimator. That
    # estimator substitutes for PGLE, which has never produced a non-empty trace on this cluster, so
    # the scheduler has been working from static cost estimates throughout.
    Arm("w8-base", ADOPTED, note="adopted configuration"),
    Arm("w8-chunkstack", CHUNKED, "w8-base", "expert_chunks=2 on top of the adopted configuration"),
    Arm("w8-o1", ADOPTED | O1, "w8-base", "the O1 bundle, including the analytical SOL latency estimator"),
    Arm("w8-solonly", ADOPTED | SOL_ESTIMATOR, "w8-base", "the SOL estimator alone"),
    # Round 9: zero-copy collectives, the B200 launch-mode workaround, and the fusion flags.
    Arm("w9-base", ADOPTED),
    Arm("w9-userbuffers", ADOPTED | USER_BUFFERS, "w9-base", "zero-copy collectives with a preallocated pool"),
    Arm("w9-solonly", ADOPTED | SOL_ESTIMATOR, "w9-base", "rerun; round 8's attempt hit a node fault"),
    # Round 10: confirm the one candidate win, retry the arm iris#7650 killed, and try the fusion
    # flags under their real names. `solonly` measured +1.10% in round 9, close enough to drift that
    # a second reading decides it.
    Arm("w10-base", ADOPTED),
    Arm("w10-solonly", ADOPTED | SOL_ESTIMATOR, "w10-base", "confirmation of round 9's +1.10%"),
    Arm("w10-cudnnfusion", ADOPTED | CUDNN_FUSION, "w10-base", "cuDNN epilogues, against 5.00 s/step of fusions"),
    Arm("w10-userbuffers", ADOPTED | USER_BUFFERS, "w10-base", "rerun; zero-copy collectives"),
    # Round 11: the cross-entropy kernel's own tiling, the block-level fusion emitter, and autotuning
    # at its limit.
    Arm("w11-base", ADOPTED),
    Arm("w11-ceautotune", ADOPTED | CE_AUTOTUNE, "w11-base", "rerun; autotune CE tiling instead of the cached miss"),
    Arm("w11-blockfusion", ADOPTED | BLOCK_FUSION, "w11-base", "the block-level Triton fusion emitter"),
    Arm("w11-exhaustive", ADOPTED | EXHAUSTIVE_AUTOTUNE, "w11-base", "compile time falls outside the scored window"),
    # Round 12: decompose O1. `doublebuffer` matters most: the hero runs all 48 layers under one
    # `jax.scan`, and double-buffering unrolls that loop twice, the cheap partial form of breaking it.
    Arm("w12-base", ADOPTED),
    Arm("w12-doublebuffer", ADOPTED | DOUBLE_BUFFER, "w12-base", "2x unroll, against 5.09 s/step of recompute"),
    Arm("w12-pipelined", ADOPTED | PIPELINED_COLLECTIVES, "w12-base", "the untested MoE all-gather flags"),
    Arm(
        "w12-o1nolhs",
        ADOPTED | DOUBLE_BUFFER | PIPELINED_COLLECTIVES | SOL_ESTIMATOR,
        "w12-base",
        "O1 without its latency hiding scheduler",
    ),
    # Round 13: the remaining flags XLA reports as default-off that plausibly touch this graph
    # (`--xla_gpu_dump_defaults` lists them).
    Arm("w13-base", ADOPTED),
    Arm("w13-dsfusion", ADOPTED | DYNAMIC_SLICE_FUSION, "w13-base", "the renamed address-computation fusion"),
    Arm(
        "w13-combine8g",
        ADOPTED | combine_threshold(8 * 1024 * 1024 * 1024),
        "w13-base",
        "8 GB thresholds, 37x a layer shard, against the 512 MB arm's -3.37%",
    ),
    Arm("w13-userbuffers", ADOPTED | USER_BUFFERS, "w13-base", "fourth attempt; the first three never reached step 0"),
    # Round 14: every remaining latency-hiding and collective flag is dropped. Six independent
    # measurements agree that a 90.3% compute-busy step has nothing for that family to recover.
    #
    # The batch arms spend HBM headroom on tokens: activations scale with the batch while weights and
    # optimizer state do not. They do more work per step, so scoring switches to tokens/s.
    Arm("w14-base", ADOPTED, note="batch 1024"),
    Arm("w14-doublebuffer", ADOPTED | DOUBLE_BUFFER, "w14-base", "rerun; 2x unroll of the 48-layer scan"),
    Arm("w14-batch1152", ADOPTED | mem_fraction("0.88"), "w14-base", "+12.5% sequences, 18 per device", 1152),
    Arm("w14-batch1280", ADOPTED | mem_fraction("0.93"), "w14-base", "+25% sequences, 20 per device", 1280),
    # The wrap-up: everything adopted, measured end to end over 40 scored steps on merged main.
    # `control2` is byte-identical to `control` and measures the spread between two arms that differ
    # in nothing.
    Arm("final-control", note="the hero as it was before this sweep", steps=WRAPUP_STEPS),
    Arm("final-control2", baseline="final-control", note="byte-identical to control", steps=WRAPUP_STEPS),
    Arm(
        "final-adopted",
        ADOPTED,
        "final-control",
        "FSDP-sharded small parameters, local-shard interleave, NVLink SHARP",
        steps=WRAPUP_STEPS,
    ),
    Arm(
        "final-adoptedbatch",
        ADOPTED | mem_fraction("0.88"),
        "final-control",
        "batch 1152 under a 0.88 ceiling",
        1152,
        WRAPUP_STEPS,
    ),
    # Second attempt: `control` and `adoptedbatch` both lost their windows above, and `adoptedbatch`
    # ran at 40 s/step against its siblings' 17. Merged main raised the baseline peak from 137.2 to
    # 141.2 GiB, so batch 1152 sits close enough to the 0.88 limit for rematerialization to engage.
    Arm("final2-control", note="the hero as it was before this sweep", steps=WRAPUP_STEPS),
    Arm("final2-control2", baseline="final2-control", note="byte-identical to control", steps=WRAPUP_STEPS),
    Arm("final2-adopted", ADOPTED, "final2-control", "replicate of the wrap-up result", steps=WRAPUP_STEPS),
    Arm(
        "final2-adoptedbatch",
        ADOPTED | mem_fraction("0.93"),
        "final2-control",
        "rerun at 0.93 for the slack",
        1152,
        WRAPUP_STEPS,
    ),
    # Batch 1152 at 0.93 dies at step 4 inside `jit_train_step`. Batch 1088 is the remaining step.
    Arm("final3-control", ADOPTED | mem_fraction("0.93"), note="the adopted config at batch 1024", steps=WRAPUP_STEPS),
    Arm(
        "final3-adoptedbatch",
        ADOPTED | mem_fraction("0.93"),
        "final3-control",
        "+6.25% tokens",
        1088,
        WRAPUP_STEPS,
    ),
    # Dead end. 0.88 and 0.93 both fail at batch 1024 with the same 122.10 GiB `jit_train_step`
    # request, so the allocation is a fixed requirement of the un-rematerialized program rather than
    # something sized to the ceiling. Peak against the allocator limit decides whether
    # `HloRematerialization` runs; post-merge the hero clears 138.22 GiB and survives only because
    # that pass shrinks it. Batch 1152 runs at 0.88 only because its 153.14 GiB peak keeps remat
    # engaged, at 40 s/step.
    Arm(
        "final4-control",
        ADOPTED | mem_fraction("0.88"),
        note="batch 1024 at the same ceiling as its arm",
        steps=WRAPUP_STEPS,
    ),
    Arm(
        "final4-adoptedbatch",
        ADOPTED | mem_fraction("0.88"),
        "final4-control",
        "+6.25% tokens",
        1088,
        WRAPUP_STEPS,
    ),
)


def select(tokens: Sequence[str]) -> dict[str, Arm]:
    """Arms named by ``tokens``: an exact name, or every arm under a ``<token>-`` prefix.

    Each selected arm pulls in its baseline, since an arm that runs without one cannot be scored.
    """
    chosen: dict[str, Arm] = {}
    for token in tokens:
        matched = [ARMS[token]] if token in ARMS else [a for n, a in ARMS.items() if n.startswith(f"{token}-")]
        if not matched:
            raise click.BadParameter(f"no arm named or prefixed {token!r}")
        for arm in matched:
            chosen[arm.name] = arm
            if arm.baseline is not None:
                chosen[arm.baseline] = ARMS[arm.baseline]
    return chosen


def scored_steps(run):
    """Return ``(step, duration, peak_gib)`` for every step in the scored window, ordered by step."""
    hist = run.history(keys=["throughput/duration", "memory/peak_gib", "_step"], pandas=False, samples=1000)
    return sorted(
        (x["_step"], x["throughput/duration"], x.get("memory/peak_gib"))
        for x in hist
        if x.get("throughput/duration") is not None and x["_step"] >= WARMUP
    )


# Model FLOPs per step at the hero shape, from the run's own throughput/mfu and duration
# (19.4145% at 18.0223 s over 64 devices).
COUNTED_FLOPS = 0.194145 * 18.0223 * PEAK_FLOPS_PER_DEVICE * NUM_DEVICES


def score_arm(arm: Arm):
    """One arm's steady-state summary, or ``None`` when its run has too few scored steps."""
    rows = scored_steps(wandb.Api().run(f"{PROJECT}/{arm.run_id}"))
    if len(rows) < 3:
        return None
    d = [t for _, t, _ in rows]
    hbm = [m for _, _, m in rows if m is not None]
    med = statistics.median(d)
    return {
        "name": arm.name,
        "baseline": arm.baseline,
        "note": arm.note,
        "n": len(d),
        "median": med,
        "mad": statistics.median([abs(x - med) for x in d]),
        "min": min(d),
        "max": max(d),
        "peak_gib": max(hbm) if hbm else None,
        "batch_size": arm.batch_size,
        # At a fixed shape MFU is a pure restatement of step time. An arm that changes the batch does
        # more work per step, so its FLOPs scale with the batch and step time stops being
        # comparable; tokens/s is the metric that survives both cases.
        "mfu": 100 * COUNTED_FLOPS * arm.batch_size / HERO_BATCH_SIZE / med / PEAK_FLOPS_PER_DEVICE / NUM_DEVICES,
        "tokens_per_s": arm.batch_size * HERO_SEQ_LEN / med,
    }


def _print_against(baseline_name, rows):
    """Print ``rows`` as deltas against the row for ``baseline_name``."""
    baseline = next((r for r in rows if r["name"] == baseline_name), None)
    if baseline is None:
        print(f"\n{baseline_name} has no scored steps; {', '.join(r['name'] for r in rows)} cannot be read")
        return
    varies_batch = len({r["batch_size"] for r in rows}) > 1
    if varies_batch:
        print(f"scoring on tokens/s: arms differ in batch size (baseline = {baseline['batch_size']})")
    print(
        f"\n{baseline_name:22s} {'n':>3s} {'batch':>6s} {'median':>9s} {'MAD':>7s} "
        f"{'Mtok/s':>7s} {'MFU':>7s} {'peak HBM':>9s} {'vs base':>9s}"
    )
    for r in sorted(rows, key=lambda r: -r["tokens_per_s"]):
        if varies_batch:
            r["delta_pct"] = 100 * (r["tokens_per_s"] / baseline["tokens_per_s"] - 1)
        else:
            r["delta_pct"] = 100 * (baseline["median"] - r["median"]) / baseline["median"]
        peak = f"{r['peak_gib']:.1f} GiB" if r["peak_gib"] else "--"
        print(
            f"{r['name']:22s} {r['n']:3d} {r['batch_size']:6d} {r['median']:8.3f}s {r['mad']:6.3f}s "
            f"{r['tokens_per_s'] / 1e6:7.3f} {r['mfu']:6.2f}% {peak:>9s} {r['delta_pct']:+8.2f}%"
        )


def score(tokens):
    arms = select(tokens)
    rows = []
    for name, arm in arms.items():
        try:
            row = score_arm(arm)
        except Exception as exc:
            print(f"{name:22s} unavailable: {type(exc).__name__}: {exc}")
            continue
        if row is None:
            print(f"{name:22s} too few scored steps yet")
            continue
        rows.append(row)
    if not rows:
        return
    for baseline_name in dict.fromkeys(r["baseline"] or r["name"] for r in rows):
        group = [r for r in rows if r["name"] == baseline_name or r["baseline"] == baseline_name]
        _print_against(baseline_name, group)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    out = LOGDIR / f"{'-'.join(tokens)}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


@click.group()
def cli():
    pass


@cli.command("launch")
@click.argument("arms", nargs=-1, required=True)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    default="production",
    show_default=True,
    help="Iris band for the training gangs. 'production' is admin-only and never preempted.",
)
@build_options
def launch_cmd(arms, priority):
    """Build the named ARMS, one rack each. Add --run to submit them.

    An ARM is an arm name or a prefix: `w7` takes every `w7-` arm. Each is its own step, so
    `--max-concurrent` bounds how many racks are live and a rerun under the same `--version` skips
    the arms that already finished.
    """
    band = priority_band_value(priority)
    return [
        build_hero_sweep_run(
            run_id=arm.run_id,
            dp_racks=DP_RACKS,
            steps_per_arm=arm.steps,
            arms=[arm.sweep_arm()],
            priority=band,
        )
        for arm in select(arms).values()
    ]


@cli.command("score")
@click.argument("arms", nargs=-1, required=True)
def score_cmd(arms):
    """Score the named ARMS against their baselines."""
    score(arms)


if __name__ == "__main__":
    cli()
