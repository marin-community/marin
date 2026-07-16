# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE seed-repeat noise panel (mixing-via-embeddings, issue #7067).

Ten training runs identical to the 840-run ``swarm_fisher_dsp_d512`` swarm in
mixture, budget, model, optimizer, and data pipeline, differing ONLY in the
seed. ``GrugMoeLaunchConfig.seed`` feeds ``TrainerConfig.seed``, which
``run_grug`` splits into the data key and the model-init key; the
simulated-epoching subset is a slice of the *shuffled* dataset, so one seed
knob varies init + data order + epoching subset together (the variable-subset
flavor of the swarm-branch SNR methodology).

All training constants below were copied from the swarm runs' own
``.executor_info`` (verified identical across runs 000000/000108/000419/
000736/000839): ``gs://marin-us-central2/grug/swarm_fisher_dsp_d512_*``.
NOTE: they intentionally do NOT match ``launch_datakit_moe_mix``'s heuristic
build, which post-dates the swarm (May Recipe, 256 experts, seq 8192).

The anchor mixture is the ``mixture-3`` two-phase bucket schedule, reused
unchanged from ``launch_datakit_moe_mix._BUCKET_PHASE_WEIGHTS``.

Known deltas vs the swarm runs (code drift; affects all 10 runs equally):

- The current model treats the LAST block as a "long" (full-window) attention
  layer; swarm-era code (2026-06-02, pre-#6153) used only ``i % 4 == 3``.
- ``disable_long_rope=False`` restores swarm-era RoPE-on-all-layers;
  ``disable_pko=True`` (the default) matches swarm-era code, which had no PKO.
- levanter/fray data+infra code has moved since 2026-06; config semantics are
  identical but binaries are not.

Run ``--dry-run`` to print the assembled configs and assert parity against the
executor_info-verified constants without touching GCS or launching anything.
"""

import argparse
import dataclasses
import json
import logging
import math

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Swarm constants, verified against .executor_info of the 840-run swarm.
# ---------------------------------------------------------------------------

STEPS = 47_759
BATCH_SIZE = 512
SEQ_LEN = 4096
PHASE_1_START_STEP = 38_144  # int(0.8 * STEPS) quantized down to the 64-step mixture-block multiple
PHASE_1_START_FRACTION = 0.8
MIXTURE_BLOCK_SIZE = 32_768
TARGET_BUDGET_TOKENS = 10_372_343_704_053
EXPERIMENT_BUDGET_TOKENS = STEPS * BATCH_SIZE * SEQ_LEN  # 100_157_882_368

NUM_RUNS = 10
SEED_BASE = 1000
RUN_ID_TEMPLATE = "rav_mve_seedpanel_{index:02d}"
# Deterministic per-user namespace (equivalent to user_namespaced_name on rav's
# machine; hardcoded so the launcher behaves identically inside an Iris pod,
# where getpass.getuser() is not rav).
STEP_NAME_TEMPLATE = "users/rav/grug/rav_mve_seedpanel_{index:02d}"
WANDB_PROJECT = "marin_moe"  # same project as the swarm so dashboards overlay
WANDB_GROUP = "rav_mve_seedpanel"

SWARM_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=256,
    shared_expert_intermediate_dim=512,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=6,
    num_heads=4,
    num_kv_heads=1,
    head_dim=None,
    max_seq_len=SEQ_LEN,
    sliding_window=4096,
    layer_norm_eps=1e-5,
    initializer_std=0.022097086912079608,  # 0.5 / sqrt(512), as serialized by the swarm
    qk_mult=1.3,
    router_z_loss_coef=0.001,
    disable_pko=True,  # swarm-era code had no PKO on any layer
    disable_long_rope=False,  # swarm-era code applied RoPE on all layers
)

SWARM_OPTIMIZER = GrugMoeAdamHConfig(
    learning_rate=0.01296243821773254,
    adam_lr=0.002991331896399817,
    expert_lr=None,
    weight_decay=0.1,
    min_lr_ratio=0.0,
    warmup=0.1,
    lr_schedule="linear",
    beta1=0.9062,
    beta2=0.9841194418156399,
    epsilon=2.11457707125721e-15,
    max_grad_norm=1.0,
)

# Matches the swarm resources exactly (cpu 32 / ram 128g / disk 50g,
# non-preemptible v4-8 in us-central2-b, the datakit store's region).
TRAIN_RESOURCES = ResourceConfig.with_tpu(
    "v4-8",
    zone="us-central2-b",
    cpu=32,
    ram="128g",
    disk="50g",
    preemptible=False,
)


def _phase_1_start_step() -> int:
    """Phase boundary quantized to the mixture-block step multiple, as the swarm launcher did."""
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    requested = max(1, int(STEPS * PHASE_1_START_FRACTION))
    return max(step_multiple, (requested // step_multiple) * step_multiple)


def _panel_data_config(val_components: dict[str, DatasetComponent]) -> LmDataConfig:
    boundary = _phase_1_start_step()
    if boundary != PHASE_1_START_STEP:
        raise ValueError(f"phase boundary {boundary} != expected {PHASE_1_START_STEP}")
    if EXPERIMENT_BUDGET_TOKENS > TARGET_BUDGET_TOKENS:
        raise ValueError("experiment budget exceeds target budget")
    val_zero_weights = {name: 0.0 for name in val_components}
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components={**datakit_mix._datakit_components(), **val_components},
        train_weights=[
            (0, {**datakit_mix._phase_weights(0), **val_zero_weights}),
            (boundary, {**datakit_mix._phase_weights(1), **val_zero_weights}),
        ],
        auto_build_caches=False,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        target_budget=TARGET_BUDGET_TOKENS,
        experiment_budget=EXPERIMENT_BUDGET_TOKENS,
    )


def _panel_launch_config(ctx: StepContext, *, index: int) -> GrugMoeLaunchConfig:
    run_id = RUN_ID_TEMPLATE.format(index=index)
    seed = SEED_BASE + index
    if ctx.is_fingerprint:
        val_components = {v.name: datakit_mix._val_component(ctx.artifact_path(v)) for v in datakit_mix._VALIDATION}
    else:
        val_components = {v.name: ctx.resolved(v).as_component() for v in datakit_mix._VALIDATION}
    return GrugMoeLaunchConfig(
        model=SWARM_MODEL,
        data=_panel_data_config(val_components),
        output_path=ctx.output_path,
        run_id=run_id,
        resources=ctx.runtime_arg("train_resources"),
        steps=STEPS,
        batch_size=BATCH_SIZE,
        seed=seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            tags=["moe", "seedpanel", "mve", "d512", f"seed{seed}"],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch by run_grug_moe_trial
        ),
        optimizer=SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=None,
            eval_current=True,
            eval_ema=False,
        ),
    )


def build_panel_step(index: int, *, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """One seed-panel training run: swarm config + seed 1000+index."""
    return ArtifactStep(
        name=STEP_NAME_TEMPLATE.format(index=index),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=lambda ctx: _panel_launch_config(ctx, index=index),
        deps=tuple(datakit_mix._VALIDATION),
        runtime_args={"train_resources": TRAIN_RESOURCES},
    )


# Field-by-field expectations transcribed from
# gs://marin-us-central2/grug/swarm_fisher_dsp_d512_000000-75258a/.executor_info
# (spot-verified identical on runs 000108/000419/000736/000839).
_EXPECTED_MODEL = {
    "vocab_size": 128_256,
    "hidden_dim": 512,
    "intermediate_dim": 256,
    "shared_expert_intermediate_dim": 512,
    "num_experts": 64,
    "num_experts_per_token": 4,
    "num_layers": 6,
    "num_heads": 4,
    "num_kv_heads": 1,
    "head_dim": None,
    "max_seq_len": 4096,
    "sliding_window": 4096,
    "layer_norm_eps": 1e-05,
    "initializer_std": 0.022097086912079608,
    "qk_mult": 1.3,
    "router_z_loss_coef": 0.001,
}
_EXPECTED_OPTIMIZER = {
    "learning_rate": 0.01296243821773254,
    "weight_decay": 0.1,
    "min_lr_ratio": 0.0,
    "warmup": 0.1,
    "decay": None,
    "rewarmup": 0.0,
    "cooldown": None,
    "cycle_length": None,
    "cycles": None,
    "lr_schedule": "linear",
    "haps": None,
    "weight_decay_modules": None,
    "default_weight_decay_mask": None,
    "beta1": 0.9062,
    "beta2": 0.9841194418156399,
    "epsilon": 2.11457707125721e-15,
    "max_grad_norm": 1.0,
    "adam_lr": 0.002991331896399817,
    "expert_lr": None,
}


def _assert_matches(actual: object, expected: dict, label: str) -> None:
    actual_dict = dataclasses.asdict(actual)
    mismatches = {k: (actual_dict.get(k), v) for k, v in expected.items() if actual_dict.get(k) != v}
    if mismatches:
        raise AssertionError(f"{label} mismatches vs swarm executor_info: {mismatches}")


def verify_swarm_parity() -> dict:
    """Assert the assembled configs equal the executor_info-verified swarm constants."""
    _assert_matches(SWARM_MODEL, _EXPECTED_MODEL, "model")
    _assert_matches(SWARM_OPTIMIZER, _EXPECTED_OPTIMIZER, "optimizer")

    if _phase_1_start_step() != 38_144:
        raise AssertionError(f"phase boundary {_phase_1_start_step()} != 38144")
    if EXPERIMENT_BUDGET_TOKENS != 100_157_882_368:
        raise AssertionError(f"experiment budget {EXPERIMENT_BUDGET_TOKENS} != 100157882368")
    if TARGET_BUDGET_TOKENS != datakit_mix._TARGET_BUDGET_TOKENS:
        raise AssertionError("target budget disagrees with launch_datakit_moe_mix")

    weights = {p: datakit_mix._phase_weights(p) for p in (0, 1)}
    for p, w in weights.items():
        total = sum(w.values())
        if abs(total - 1.0) > 1e-9:
            raise AssertionError(f"phase {p} weights sum to {total}")

    components = datakit_mix._datakit_components()
    if len(components) != 168:
        raise AssertionError(f"expected 168 train components, got {len(components)}")

    return {
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "tokens_per_run": EXPERIMENT_BUDGET_TOKENS,
        "phase_1_start_step": _phase_1_start_step(),
        "phase_fractions": [
            _phase_1_start_step() / STEPS,
            1 - _phase_1_start_step() / STEPS,
        ],
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "target_budget": TARGET_BUDGET_TOKENS,
        "n_train_components": len(components),
        "n_validation_sets": len(datakit_mix._VALIDATION),
        "phase_weight_support": [len(weights[0]), len(weights[1])],
        "runs": [
            {
                "step_name": STEP_NAME_TEMPLATE.format(index=i),
                "run_id": RUN_ID_TEMPLATE.format(index=i),
                "seed": SEED_BASE + i,
            }
            for i in range(NUM_RUNS)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print configs + parity checks, launch nothing")
    parser.add_argument("--max-concurrent", type=int, default=14)
    args = parser.parse_args()

    summary = verify_swarm_parity()
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("DRY RUN: parity checks passed; nothing launched.")
        return

    steps = [build_panel_step(i).lower() for i in range(NUM_RUNS)]
    StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
