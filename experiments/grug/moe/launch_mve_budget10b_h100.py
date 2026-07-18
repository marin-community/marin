# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE budget-transfer probe on cw-rno2a H100s (mixing-via-embeddings, #7067).

Twelve training runs that reproduce twelve of the 840-run swarm's mixtures at a
SHORT budget: ~10.02B tokens/run (4,776 steps x 512 x 4096) instead of the
swarm's 100.16B (47,759 steps). Everything else -- model, optimizer, batch,
seq, tokenizer, mixture-block size, simulated-epoching target_budget, seed --
is held at the verified swarm config (imported from ``launch_mve_seedpanel``).

Question: do mixture rankings at 100B tokens/run transfer to 10B tokens/run?
(Resolves the report's tier C: whether future sweeps can cost 1/10th.)

The 12 mixtures are selected from ``train_runs.parquet`` (never the quarantined
test labels), stratified to span realized ``zmacro_english_20`` -- 2 per quintile
(nearest each quintile's within-quintile p25/p75) + the best train run + the run
nearest the token-proportional anchor by per-phase Hellinger. Their 100B outcomes
are the reference ranks. Weights + reference outcomes live in the sibling
``budget10b_mixtures.json`` (bundled with the launcher; the pre-registration
record).

Deltas vs the swarm runs, all intentional and documented:

- BUDGET: 4,776 steps (~10.02B tok) vs 47,759 (~100.16B). The fractional LR
  schedule (warmup 0.1, no explicit decay, min_lr_ratio 0, linear) scales with
  ``num_train_steps`` automatically -- verified: warmup peak lands at 0.100*N in
  both the 47,759- and 4,776-step schedules.
- PHASE BOUNDARY: requested 0.8 fraction, block-quantized for the SHORT run by
  the launcher's own rule -> step 3,776 (fractions 0.7906/0.2094). At the swarm's
  47,759 steps the same rule gives 38,144 (0.7987/0.2013); the small drift is the
  honest consequence of quantizing 0.8*N to the 64-step mixture-block multiple at
  a smaller N -- this measures the practical 10B-sweep member, not a pure
  data-effect isolation.
- SIMULATED EPOCHING: ``target_budget`` UNCHANGED (10.372e12); ``experiment_budget``
  is the short 10.016e9, so each cache is sliced to ratio ~= 0.0966% (vs the
  swarm member's 0.966%). This preserves "what a 10B-budget sweep member would
  have seen" -- 10x fewer post-warmup tokens over 10x-thinner cache slices.
- HARDWARE/NUMERICS: cw-rno2a H100x8 (data-axis=8; 80GB HBM needs 8-way sharding
  of the ~521GiB step footprint), ``gpu_fa4_cute`` (SM90 path), ``cuda_async``
  allocator -- identical to the H100 seed panel.
- CW mirror data (rno2a pods carry ``MARIN_PREFIX=s3://marin-us-east-02a/marin``);
  no in-training validation (readout is post-hoc ``eval_logprob``).

CONFOUND (stated honestly): a 10B run has both ~10x fewer post-warmup tokens AND
a compressed LR schedule. This measures the practical question -- would a cheap
sweep have ranked these mixtures the same -- not a pure data-quantity effect.

Modes mirror ``launch_mve_seedpanel_b200``: ``--dry-run`` (parity + config, no
side effects); ``--direct --index N`` (train ONE run in-process, the shape used
inside a federated/direct ``iris job run --gpu H100x8`` job).
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Short-budget constants. Model/optimizer/batch/seq/block/target_budget are the
# verified swarm values, imported unchanged. Only the step count (and therefore
# the token budget, phase boundary, and epoching slice ratio) differs.
# ---------------------------------------------------------------------------

STEPS = 4_776  # ~10.02B tok/run at 512 x 4096
BATCH_SIZE = tpu_panel.BATCH_SIZE  # 512
SEQ_LEN = tpu_panel.SEQ_LEN  # 4096
MIXTURE_BLOCK_SIZE = tpu_panel.MIXTURE_BLOCK_SIZE  # 32_768
PHASE_1_START_FRACTION = tpu_panel.PHASE_1_START_FRACTION  # 0.8
TARGET_BUDGET_TOKENS = tpu_panel.TARGET_BUDGET_TOKENS  # 10_372_343_704_053 (UNCHANGED)
EXPERIMENT_BUDGET_TOKENS = STEPS * BATCH_SIZE * SEQ_LEN  # 10_015_997_952

NUM_RUNS = 12
SEED = 0  # all 12 differ in MIXTURE only; the swarm used seed 0 everywhere
RUN_ID_TEMPLATE = "rav_mve_budget10b_{index:02d}"
STEP_NAME_TEMPLATE = "users/rav/grug/rav_mve_budget10b_{index:02d}"
WANDB_PROJECT = tpu_panel.WANDB_PROJECT  # marin_moe
WANDB_GROUP = "rav_mve_budget10b"

MIXTURES_PATH = Path(__file__).with_name("budget10b_mixtures.json")
_MIXTURES = json.loads(MIXTURES_PATH.read_text())

DEFAULT_GPUS_PER_RUN = 8  # one gd-8xh100ib node; 512x4096 needs 8-way sharding on 80GB HBM
HOST_CPU = 32
HOST_RAM = "256g"
HOST_DISK = "256g"


def train_resources(gpus_per_run: int) -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=gpus_per_run,
        cpu=HOST_CPU,
        ram=HOST_RAM,
        disk=HOST_DISK,
        preemptible=False,
    )


def _phase_1_start_step() -> int:
    """0.8-fraction phase boundary, block-quantized for the SHORT run.

    Same rule as ``launch_mve_seedpanel._phase_1_start_step`` but at STEPS=4776:
    step_multiple = 32768 // gcd(32768, 512) = 64; requested = int(4776*0.8) = 3820;
    boundary = (3820 // 64) * 64 = 3776 -> fractions 0.7906 / 0.2094.
    """
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    requested = max(1, int(STEPS * PHASE_1_START_FRACTION))
    return max(step_multiple, (requested // step_multiple) * step_multiple)


PHASE_1_START_STEP = _phase_1_start_step()  # 3_776


def _mixture_phase_weights(index: int, phase: int) -> dict[str, float]:
    """The selected train run's exact stored phase weights (all 168 buckets).

    Weights already sum to 1 (verified in selection). Buckets with weight 0 in
    both phases are tolerated as zero-weight keys -- exactly the mechanism by
    which the seed panel passes zero-weight validation components (skipped from
    the train datasets, harmless in the schedule).
    """
    return dict(_MIXTURES[f"{index:02d}"][f"phase{phase}_weights"])


def _mixture_data_config(index: int) -> LmDataConfig:
    """Panel data config with this run's mixture, CW-absolute paths, short budget.

    Reuses the B200/H100 panel's CW-absolutized components (167 direct buckets +
    the ``tail`` concat). ``experiment_budget`` is the SHORT budget while
    ``target_budget`` stays the swarm's -> a ~0.097% cache slice per bucket.
    """
    p0 = _mixture_phase_weights(index, 0)
    p1 = _mixture_phase_weights(index, 1)
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=b200_panel._abs_datakit_components(),
        train_weights=[
            (0, p0),
            (PHASE_1_START_STEP, p1),
        ],
        auto_build_caches=False,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        target_budget=TARGET_BUDGET_TOKENS,
        experiment_budget=EXPERIMENT_BUDGET_TOKENS,
    )


def _budget10b_launch_config(
    *,
    index: int,
    output_path: str,
    total_steps: int,
    resources: ResourceConfig,
    run_suffix: str | None = None,
) -> GrugMoeLaunchConfig:
    run_id = RUN_ID_TEMPLATE.format(index=index)
    if run_suffix:
        run_id = f"{run_id}-{run_suffix}"
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # swarm shapes + gpu_fa4_cute (SM90 path)
        data=_mixture_data_config(index),
        output_path=output_path,
        run_id=run_id,
        resources=resources,
        steps=total_steps,
        batch_size=BATCH_SIZE,
        seed=SEED,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            tags=[
                "moe",
                "budget10b",
                "mve",
                "d512",
                "h100",
                f"exp{_MIXTURES[f'{index:02d}']['experiment_index']}",
                _MIXTURES[f"{index:02d}"]["role"],
            ],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,  # no in-training validation on CW (readout is post-hoc)
    )


def _verify_schedule_scales() -> dict:
    """Assert the fractional LR schedule scales: warmup peak at 0.1*N for short + swarm N."""
    out = {}
    for n in (tpu_panel.STEPS, STEPS):
        sched = tpu_panel.SWARM_OPTIMIZER.lr_scheduler(n)
        lrs = np.array([float(sched(i)) for i in range(n)])
        peak = int(np.argmax(lrs))
        frac = peak / n
        if abs(frac - 0.1) > 0.001:
            raise AssertionError(f"warmup peak fraction {frac} != 0.1 at N={n}")
        out[str(n)] = {"peak_step": peak, "peak_frac": frac, "peak_lr": float(lrs.max())}
    return out


def _verify_budget10b_parity(*, check_schedule: bool = False) -> dict:
    """Model/optimizer parity vs swarm + short-budget invariants + mixture sanity.

    ``check_schedule`` builds and *evaluates* the optax LR schedule, which
    initialises the XLA backend -- safe under ``--dry-run`` (single process) but
    NOT in a ``--direct`` GPU job, where it must not precede
    ``jax.distributed.initialize()``. So it is off by default.
    """
    tpu_panel._assert_matches(tpu_panel.SWARM_MODEL, tpu_panel._EXPECTED_MODEL, "model")
    tpu_panel._assert_matches(tpu_panel.SWARM_OPTIMIZER, tpu_panel._EXPECTED_OPTIMIZER, "optimizer")
    # B200_MODEL differs from SWARM_MODEL only by the GPU attention backend.
    if b200_panel.B200_MODEL.attention_implementation != "gpu_fa4_cute":
        raise AssertionError("B200_MODEL attention backend is not gpu_fa4_cute")

    if STEPS != 4_776:
        raise AssertionError(f"STEPS {STEPS} != 4776")
    if EXPERIMENT_BUDGET_TOKENS != 4_776 * 512 * 4096:
        raise AssertionError(f"experiment budget {EXPERIMENT_BUDGET_TOKENS} != 4776*512*4096")
    if EXPERIMENT_BUDGET_TOKENS != 10_015_997_952:
        raise AssertionError(f"experiment budget {EXPERIMENT_BUDGET_TOKENS} != 10_015_997_952")
    if PHASE_1_START_STEP != 3_776:
        raise AssertionError(f"phase boundary {PHASE_1_START_STEP} != 3776")
    if EXPERIMENT_BUDGET_TOKENS >= TARGET_BUDGET_TOKENS:
        raise AssertionError("experiment budget must be < target budget for epoching")
    if TARGET_BUDGET_TOKENS != datakit_mix._TARGET_BUDGET_TOKENS:
        raise AssertionError("target budget disagrees with launch_datakit_moe_mix")

    component_names = set(b200_panel._abs_datakit_components().keys())
    if len(component_names) != 168:
        raise AssertionError(f"expected 168 components, got {len(component_names)}")

    if len(_MIXTURES) != NUM_RUNS:
        raise AssertionError(f"expected {NUM_RUNS} mixtures, got {len(_MIXTURES)}")
    for i in range(NUM_RUNS):
        key = f"{i:02d}"
        if key not in _MIXTURES:
            raise AssertionError(f"missing mixture {key}")
        for phase in (0, 1):
            w = _mixture_phase_weights(i, phase)
            if not set(w).issubset(component_names):
                raise AssertionError(f"mixture {key} phase {phase} has non-component buckets")
            total = sum(w.values())
            if abs(total - 1.0) > 1e-6:
                raise AssertionError(f"mixture {key} phase {phase} sums to {total}")
            if any(v < 0 for v in w.values()):
                raise AssertionError(f"mixture {key} phase {phase} has negative weight")

    ratio = EXPERIMENT_BUDGET_TOKENS / TARGET_BUDGET_TOKENS
    schedule = _verify_schedule_scales() if check_schedule else None
    return {
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "tokens_per_run": EXPERIMENT_BUDGET_TOKENS,
        "swarm_tokens_per_run": tpu_panel.EXPERIMENT_BUDGET_TOKENS,
        "phase_1_start_step": PHASE_1_START_STEP,
        "phase_fractions": [PHASE_1_START_STEP / STEPS, 1 - PHASE_1_START_STEP / STEPS],
        "swarm_phase_1_start_step": tpu_panel.PHASE_1_START_STEP,
        "swarm_phase_fractions": [
            tpu_panel.PHASE_1_START_STEP / tpu_panel.STEPS,
            1 - tpu_panel.PHASE_1_START_STEP / tpu_panel.STEPS,
        ],
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "target_budget": TARGET_BUDGET_TOKENS,
        "experiment_budget": EXPERIMENT_BUDGET_TOKENS,
        "epoching_slice_ratio": ratio,
        "swarm_epoching_slice_ratio": tpu_panel.EXPERIMENT_BUDGET_TOKENS / TARGET_BUDGET_TOKENS,
        "warmup_fraction_check": schedule,
        "seed": SEED,
        "n_runs": NUM_RUNS,
        "mixtures_sha256_source": MIXTURES_PATH.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE run in-process (direct GPU job)")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="step override for --direct (smoke sizing)")
    parser.add_argument("--run-suffix", type=str, default=None, help="run-id/output suffix for --direct (e.g. 'smoke')")
    parser.add_argument("--gpus-per-run", type=int, default=DEFAULT_GPUS_PER_RUN)
    args = parser.parse_args()

    # The schedule-scaling check evaluates optax (touches the XLA backend), so it
    # runs ONLY for --dry-run -- never in a --direct GPU job before jax.distributed.
    summary = _verify_budget10b_parity(check_schedule=args.dry_run)
    summary["runs"] = [
        {
            "index": i,
            "run_id": RUN_ID_TEMPLATE.format(index=i),
            "step_name": STEP_NAME_TEMPLATE.format(index=i),
            "experiment_index": _MIXTURES[f"{i:02d}"]["experiment_index"],
            "role": _MIXTURES[f"{i:02d}"]["role"],
            "zmacro_english_20_100b": _MIXTURES[f"{i:02d}"]["zmacro_english_20_100b"],
            "macro_bpb_100b": _MIXTURES[f"{i:02d}"]["macro_bpb_100b"],
        }
        for i in range(NUM_RUNS)
    ]
    summary["gpus_per_run"] = args.gpus_per_run
    summary["cluster"] = "cw-rno2a"
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    summary["in_training_validation"] = False

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity checks passed; nothing launched.")
        return

    if not args.direct:
        raise SystemExit("budget10b jobs are submitted directly (one iris job per --index); use --direct.")

    total_steps = args.steps or STEPS
    run_id = RUN_ID_TEMPLATE.format(index=args.index)
    leaf = f"{run_id}-{args.run_suffix}" if args.run_suffix else run_id
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{leaf}/dev")
    logger.info("direct mode: run %s, %d steps, output %s", leaf, total_steps, output_path)
    config = _budget10b_launch_config(
        index=args.index,
        output_path=output_path,
        total_steps=total_steps,
        resources=train_resources(args.gpus_per_run),
        run_suffix=args.run_suffix,
    )
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
