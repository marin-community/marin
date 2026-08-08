# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep single-rack hero configurations.

Measurement protocol
--------------------
``--racks N`` splits a wave's arms into N allocations, each running its arms back to back in a
fresh trainer subprocess so process-start environment takes effect, each writing its own W&B run
``hs-<wave>[-g<i>]-<tag>``. The score is the **median** ``throughput/duration`` over steps
``WARMUP..``, because the steady-state distribution is right-skewed (run A: median 18.086 s, min
17.936, max 18.699, MAD 0.077). Steps 0-1 are compile and the PGLE recompile; steps 2-4 absorb the
one-time first-batch data-loader stall.

Every group carries the wave's ``control`` or ``base`` arm and an arm scores only against the
control that ran beside it. Waves w1 through final4 ran one rack per arm, where two byte-identical
controls differed by 0.78% -- the resolution floor on every delta below. Sharing an allocation
trades that for drift over the hours a group takes, which nothing has measured yet.

Usage
-----
    uv run python -m experiments.grug.moe_hero_fsdp.sweep_preflight <wave>   # CPU trace, first
    uv run python -m experiments.grug.moe_hero_fsdp.sweep launch <wave> --version dev --racks 4 --run
    uv run python -m experiments.grug.moe_hero_fsdp.sweep score <wave>
"""

import collections
import dataclasses
import json
import pathlib
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field

import click
import wandb
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_value
from levanter.recovery.types import AblationSpec
from marin.experiment.cli import build_options

from experiments.grug.moe_hero_fsdp.launch import HeroOverrides, HeroSweepArm, build_hero_sweep_run
from experiments.grug.moe_hero_fsdp.model import RematMode

PROJECT = "marin-community/marin_moe"
STEPS = 20
WARMUP = 5  # first scored step
# 15 steps separates the multi-percent effects a screening wave looks for; the wrap-up waves that
# produce the reported number take 40.
WAVE_STEPS = {"final": 45, "final2": 45, "final3": 45, "final4": 45}
PREFIX = "hs"  # hero sweep
LOGDIR = pathlib.Path("scratch/hero_sweep")
PEAK_FLOPS_PER_DEVICE = 2.5e15
DP_RACKS = 1
# HERO_NODES_PER_RACK * HERO_GPUS_PER_TASK in the launcher.
DEVICES_PER_RACK = 64
NUM_DEVICES = DP_RACKS * DEVICES_PER_RACK
HERO_BATCH_SIZE = 1024
HERO_SEQ_LEN = 4096

# Early waves named the in-window baseline `control`; from wave 6 on it is `base`, the adopted
# configuration. Scoring against whichever arm happens to be fastest would turn drift into a result,
# so the reference is always one of these and scoring fails loudly if a wave has neither.
CONTROL_TAGS = frozenset({"control", "base"})

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
    """One configuration under test."""

    tag: str
    setting: Setting = Setting()
    note: str = ""
    batch_size: int = HERO_BATCH_SIZE

    def run_id(self, wave: str) -> str:
        return f"{PREFIX}-{wave}-{self.tag}"

    def sweep_arm(self, wave: str, steps: int) -> HeroSweepArm:
        return HeroSweepArm(
            spec=AblationSpec(name=self.tag, env=self.setting.process_env(), num_steps=steps, notes=self.note),
            overrides=self.setting.overrides,
            batch_size=self.batch_size,
        )


WAVES = {
    # Wave 1: NCCL protocol and algorithm. Nothing in the repo sets NCCL_PROTO or NCCL_ALGO.
    "w1": [
        Arm("control", note="unchanged baseline"),
        Arm("simple", NCCL_SIMPLE, note="drop LL's 8-byte-flag-per-8-byte overhead"),
        Arm("ll128", NCCL_LL128, note="128-byte LL variant"),
        Arm("nvls", NVLS, note="NVLink SHARP"),
    ],
    # Wave 2: the two cheap code wins plus the activation-memory probe. `save_moe` already exists
    # as a remat mode; the hero does not use it.
    "w2": [
        Arm("control"),
        Arm("shardsmall", SHARD_SMALL, note="kill the per-layer all-reduce"),
        Arm("savemoe", remat("save_moe"), note="probe the HBM ceiling; may OOM"),
    ],
    # Wave 3: confirm the wave-2 winners stack, and attack the two biggest remaining blocks of
    # compute-stream work reachable by a config knob. Keyed `w3b` because two earlier attempts died
    # in startup and burned the `w3` run ids.
    "w3b": [
        Arm("control"),
        Arm("combo", SHARD_SMALL, note="both wave-2 winners together"),
        Arm("cebig", ce_block(8192), note="8x fewer CE launches, larger GEMMs"),
        Arm("offloadmoe", remat("offload_moe"), note="park dispatch output on the host"),
    ],
    # Wave 4: the stacked winner measured end to end against its own control, plus the two
    # remaining single knobs.
    "w4": [
        Arm("control"),
        Arm("combined", ADOPTED, note="every winner from waves 1-3, measured end to end"),
        Arm("chunks8", expert_chunks(8), note="shorter gather prologue, smaller GEMMs"),
        Arm("chunks2", expert_chunks(2), note="fewer, larger expert GEMMs"),
    ],
    # Wave 5: bracket the expert-chunk count below the hero's 4, and replicate the headline
    # `combined` number against a fresh control.
    "w5": [
        Arm("control"),
        Arm("combined", ADOPTED, note="replication of the wave-4 result"),
        Arm("chunks2", expert_chunks(2), note="rerun; wave 4 lost a node mid-run"),
        Arm("chunks1", expert_chunks(1), note="no chunking, one gather of the full bank"),
    ],
    # Wave 6: stop capping HBM. XLA reports it cannot get peak below 160.30 GiB without recomputing.
    # `memory/limit_gib` is logged per step, so an arm that silently ignores the knob is visible.
    "w6": [
        Arm("base", ADOPTED, note="adopted configuration; this wave's control"),
        Arm("memfrac88", ADOPTED | mem_fraction("0.88"), note="162.2 GiB, just above the remat floor"),
        Arm("memfrac93", ADOPTED | mem_fraction("0.93"), note="171.4 GiB; NCCL and cuBLAS live outside the pool"),
        Arm("ceautotune", ADOPTED | CE_AUTOTUNE, note="tune CE block sizes for this shape"),
    ],
    # Wave 7: XLA and NCCL flags on the adopted configuration plus `expert_chunks=2`. 128 experts
    # only divide into powers of two, so 2 is the single step below the hero's 4.
    "w7": [
        Arm("base", CHUNKED, note="this wave's control"),
        Arm("cudagraphs", CHUNKED | COMMAND_BUFFERS, note="CE alone dispatches 43.7k kernels/step at 17.5 us"),
        Arm(
            "combinethresh",
            CHUNKED | combine_threshold(512 * 1024 * 1024),
            note="the profile shows 1,009 uncombined collective launches at 52 GB/s",
        ),
        Arm("epflags", CHUNKED | EP_SCHEDULER, note="the two scheduler flags the EP hero runs in production"),
    ],
    # Wave 8: NVIDIA's JAX-Toolbox Blackwell guidance, none of which this repo sets. `O1` bundles the
    # latency-hiding scheduler, pipelined collectives, double buffering, and the SOL estimator. That
    # estimator substitutes for PGLE, which has never produced a non-empty trace on this cluster, so
    # the scheduler has been working from static cost estimates throughout.
    "w8": [
        Arm("base", ADOPTED, note="adopted configuration; this wave's control"),
        Arm("chunkstack", CHUNKED, note="expert_chunks=2 on top of the adopted configuration"),
        Arm("o1", ADOPTED | O1, note="the O1 bundle, including the analytical SOL latency estimator"),
        Arm("solonly", ADOPTED | SOL_ESTIMATOR, note="the SOL estimator alone"),
    ],
    # Wave 9: zero-copy collectives, the B200 launch-mode workaround, and the fusion flags.
    "w9": [
        Arm("base", ADOPTED, note="this wave's control"),
        Arm("userbuffers", ADOPTED | USER_BUFFERS, note="zero-copy collectives with a preallocated pool"),
        Arm("solonly", ADOPTED | SOL_ESTIMATOR, note="rerun; wave 8's attempt hit a node fault"),
    ],
    # Wave 10: confirm the one candidate win, retry the arm iris#7650 killed, and try the fusion
    # flags under their real names. `solonly` measured +1.10% in wave 9, close enough to drift that
    # a second reading decides it.
    "w10": [
        Arm("base", ADOPTED, note="this wave's control"),
        Arm("solonly", ADOPTED | SOL_ESTIMATOR, note="confirmation of wave 9's +1.10%"),
        Arm("cudnnfusion", ADOPTED | CUDNN_FUSION, note="cuDNN epilogues, against 5.00 s/step of fusions"),
        Arm("userbuffers", ADOPTED | USER_BUFFERS, note="rerun; zero-copy collectives"),
    ],
    # Wave 11: the cross-entropy kernel's own tiling, the block-level fusion emitter, and autotuning
    # at its limit.
    "w11": [
        Arm("base", ADOPTED, note="this wave's control"),
        Arm("ceautotune", ADOPTED | CE_AUTOTUNE, note="rerun; autotune CE tiling instead of the cached miss"),
        Arm("blockfusion", ADOPTED | BLOCK_FUSION, note="the block-level Triton fusion emitter"),
        Arm("exhaustive", ADOPTED | EXHAUSTIVE_AUTOTUNE, note="compile time falls outside the scored window"),
    ],
    # Wave 12: decompose O1. `doublebuffer` matters most: the hero runs all 48 layers under one
    # `jax.scan`, and double-buffering unrolls that loop twice, the cheap partial form of breaking it.
    "w12": [
        Arm("base", ADOPTED, note="this wave's control"),
        Arm("doublebuffer", ADOPTED | DOUBLE_BUFFER, note="2x unroll, against 5.09 s/step of recompute"),
        Arm("pipelined", ADOPTED | PIPELINED_COLLECTIVES, note="the untested MoE all-gather flags"),
        Arm(
            "o1nolhs",
            ADOPTED | DOUBLE_BUFFER | PIPELINED_COLLECTIVES | SOL_ESTIMATOR,
            note="O1 without its latency hiding scheduler",
        ),
    ],
    # Wave 13: the remaining flags XLA reports as default-off that plausibly touch this graph
    # (`--xla_gpu_dump_defaults` lists them).
    "w13": [
        Arm("base", ADOPTED, note="this wave's control"),
        Arm("dsfusion", ADOPTED | DYNAMIC_SLICE_FUSION, note="the renamed address-computation fusion"),
        Arm(
            "combine8g",
            ADOPTED | combine_threshold(8 * 1024 * 1024 * 1024),
            note="8 GB thresholds, 37x a layer shard, against the 512 MB arm's -3.37%",
        ),
        Arm("userbuffers", ADOPTED | USER_BUFFERS, note="fourth attempt; the first three never reached step 0"),
    ],
    # Wave 14: every remaining latency-hiding and collective flag is dropped. Six independent
    # measurements agree that a 90.3% compute-busy step has nothing for that family to recover.
    #
    # The batch arms spend HBM headroom on tokens: activations scale with the batch while weights and
    # optimizer state do not. They do more work per step, so scoring switches to tokens/s.
    "w14": [
        Arm("base", ADOPTED, note="this wave's control, batch 1024"),
        Arm("doublebuffer", ADOPTED | DOUBLE_BUFFER, note="rerun; 2x unroll of the 48-layer scan"),
        Arm("batch1152", ADOPTED | mem_fraction("0.88"), note="+12.5% sequences, 18 per device", batch_size=1152),
        Arm("batch1280", ADOPTED | mem_fraction("0.93"), note="+25% sequences, 20 per device", batch_size=1280),
    ],
    # The wrap-up: everything adopted, measured end to end over 40 scored steps on merged main.
    # `control2` is byte-identical to `control` and measures the spread between two arms that differ
    # in nothing.
    "final": [
        Arm("control", note="the hero as it was before this sweep"),
        Arm("control2", note="byte-identical to control"),
        Arm("adopted", ADOPTED, note="FSDP-sharded small parameters, local-shard interleave, NVLink SHARP"),
        Arm("adoptedbatch", ADOPTED | mem_fraction("0.88"), note="batch 1152 under a 0.88 ceiling", batch_size=1152),
    ],
    # Second attempt: `control` and `adoptedbatch` both lost their windows above, and `adoptedbatch`
    # ran at 40 s/step against its siblings' 17. Merged main raised the baseline peak from 137.2 to
    # 141.2 GiB, so batch 1152 sits close enough to the 0.88 limit for rematerialization to engage.
    "final2": [
        Arm("control", note="the hero as it was before this sweep"),
        Arm("control2", note="byte-identical to control"),
        Arm("adopted", ADOPTED, note="replicate of the wrap-up result"),
        Arm("adoptedbatch", ADOPTED | mem_fraction("0.93"), note="rerun at 0.93 for the slack", batch_size=1152),
    ],
    # Batch 1152 at 0.93 dies at step 4 inside `jit_train_step`. Batch 1088 is the remaining step.
    "final3": [
        Arm("control", ADOPTED | mem_fraction("0.93"), note="the adopted config at batch 1024"),
        Arm("adoptedbatch", ADOPTED | mem_fraction("0.93"), note="+6.25% tokens", batch_size=1088),
    ],
    # Dead end. 0.88 and 0.93 both fail at batch 1024 with the same 122.10 GiB `jit_train_step`
    # request, so the allocation is a fixed requirement of the un-rematerialized program rather than
    # something sized to the ceiling. Peak against the allocator limit decides whether
    # `HloRematerialization` runs; post-merge the hero clears 138.22 GiB and survives only because
    # that pass shrinks it. Batch 1152 runs at 0.88 only because its 153.14 GiB peak keeps remat
    # engaged, at 40 s/step.
    "final4": [
        Arm("control", ADOPTED | mem_fraction("0.88"), note="batch 1024 at the same ceiling as its arm"),
        Arm("adoptedbatch", ADOPTED | mem_fraction("0.88"), note="+6.25% tokens", batch_size=1088),
    ],
}


def wave_steps(wave: str) -> int:
    return WAVE_STEPS.get(wave, STEPS)


def split_arms(arms: list[Arm], racks: int) -> list[list[Arm]]:
    """Split ``arms`` across ``racks`` allocations, replicating the control into each.

    An arm is only comparable against a control that ran beside it, so a group without one could
    not be scored. ``racks == len(arms) - 1`` reproduces the one-rack-per-arm fan-out.
    """
    if racks == 1:
        return [list(arms)]
    control = next(arm for arm in arms if arm.tag in CONTROL_TAGS)
    rest = [arm for arm in arms if arm is not control]
    if racks > len(rest):
        raise click.BadParameter(f"{racks} racks for {len(rest)} arms besides the control")
    groups: list[list[Arm]] = [[control] for _ in range(racks)]
    for i, arm in enumerate(rest):
        groups[i % racks].append(arm)
    return groups


def group_prefix(wave: str, index: int, racks: int) -> str:
    """W&B run-id prefix for one group. A wave on a single allocation keeps the bare wave id."""
    return f"{PREFIX}-{wave}" if racks == 1 else f"{PREFIX}-{wave}-g{index}"


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


def score_run(run, batch_size=HERO_BATCH_SIZE):
    rows = scored_steps(run)
    if len(rows) < 3:
        return None
    d = [t for _, t, _ in rows]
    hbm = [m for _, _, m in rows if m is not None]
    med = statistics.median(d)
    return {
        "run_id": run.id,
        "n": len(d),
        "median": med,
        "mad": statistics.median([abs(x - med) for x in d]),
        "min": min(d),
        "max": max(d),
        "peak_gib": max(hbm) if hbm else None,
        "batch_size": batch_size,
        # At a fixed shape MFU is a pure restatement of step time. An arm that changes the batch does
        # more work per step, so its FLOPs scale with the batch and step time stops being
        # comparable; tokens/s is the metric that survives both cases.
        "mfu": 100 * COUNTED_FLOPS * batch_size / HERO_BATCH_SIZE / med / PEAK_FLOPS_PER_DEVICE / NUM_DEVICES,
        "tokens_per_s": batch_size * HERO_SEQ_LEN / med,
    }


def discover_runs(wave):
    """Every W&B run for ``wave``, keyed by allocation prefix then arm tag.

    How a wave was split across allocations is a launch-time choice that nothing records, so the
    grouping is read back off the run ids rather than reconstructed.
    """
    groups = collections.defaultdict(dict)
    for run in wandb.Api().runs(PROJECT, filters={"name": {"$regex": f"^{PREFIX}-{wave}-"}}):
        prefix, _, tag = run.id.rpartition("-")
        groups[prefix][tag] = run
    return groups


def score_group(prefix, runs, by_tag):
    """Score one allocation's arms against the control that ran beside them."""
    results = []
    for tag, run in runs.items():
        arm = by_tag.get(tag)
        if arm is None:
            print(f"{tag:12s} not an arm of this wave; skipped")
            continue
        try:
            s = score_run(run, batch_size=arm.batch_size)
        except Exception as exc:
            print(f"{tag:12s} unavailable: {type(exc).__name__}: {exc}")
            continue
        if s is None:
            print(f"{tag:12s} too few scored steps yet")
            continue
        results.append({**s, "group": prefix, "tag": tag, "note": arm.note})

    if not results:
        return []
    control = next((r for r in results if r["tag"] in CONTROL_TAGS), None)
    if control is None:
        scored = ", ".join(r["tag"] for r in results)
        print(f"{prefix}: no control arm has scored steps yet (scored so far: {scored})")
        return results

    varies_batch = len({r["batch_size"] for r in results}) > 1
    if varies_batch:
        print(f"scoring on tokens/s: arms differ in batch size (control = {control['batch_size']})")
    print(
        f"\n{prefix:12s} {'n':>3s} {'batch':>6s} {'median':>9s} {'MAD':>7s} "
        f"{'Mtok/s':>7s} {'MFU':>7s} {'peak HBM':>9s} {'vs control':>11s}"
    )
    for r in sorted(results, key=lambda r: -r["tokens_per_s"]):
        if varies_batch:
            r["delta_pct"] = 100 * (r["tokens_per_s"] / control["tokens_per_s"] - 1)
        else:
            r["delta_pct"] = 100 * (control["median"] - r["median"]) / control["median"]
        peak = f"{r['peak_gib']:.1f} GiB" if r["peak_gib"] else "--"
        print(
            f"{r['tag']:12s} {r['n']:3d} {r['batch_size']:6d} {r['median']:8.3f}s {r['mad']:6.3f}s "
            f"{r['tokens_per_s'] / 1e6:7.3f} {r['mfu']:6.2f}% {peak:>9s} {r['delta_pct']:+10.2f}%"
        )
    return results


def score(wave):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    by_tag = {arm.tag: arm for arm in WAVES[wave]}
    groups = discover_runs(wave)
    if not groups:
        print(f"no W&B runs for {wave}")
        return
    results = [r for prefix in sorted(groups) for r in score_group(prefix, groups[prefix], by_tag)]
    if not results:
        return
    out = LOGDIR / f"{wave}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


@click.group()
def cli():
    pass


@cli.command("launch")
@click.argument("wave", type=click.Choice(sorted(WAVES)))
@click.option(
    "--racks", type=click.IntRange(min=1), default=1, show_default=True, help="Allocations to spread the wave over."
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES),
    default="production",
    show_default=True,
    help="Iris band for the training gangs. 'production' is admin-only and never preempted.",
)
@build_options
def launch_cmd(wave, racks, priority):
    """Build the wave. Add --run to submit it.

    Each group of arms is one allocation and one step, so ``--racks`` trades wall clock for racks
    and ``--max-concurrent`` bounds how many run at once. Every group carries the wave's control,
    since an arm is scored only against a control that ran beside it.
    """
    steps = wave_steps(wave)
    band = priority_band_value(priority)
    groups = split_arms(WAVES[wave], racks)
    return [
        build_hero_sweep_run(
            run_id=group_prefix(wave, i, racks),
            dp_racks=DP_RACKS,
            steps_per_arm=steps,
            arms=[arm.sweep_arm(wave, steps) for arm in group],
            priority=band,
        )
        for i, group in enumerate(groups)
    ]


@cli.command("score")
@click.argument("wave", type=click.Choice(sorted(WAVES)))
def score_cmd(wave):
    """Score a wave from its W&B runs."""
    score(wave)


if __name__ == "__main__":
    cli()
