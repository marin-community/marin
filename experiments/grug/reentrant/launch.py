# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-entrant (self-looping) grug MoE experiment series.

A copy-first variant of `experiments/grug/moe` (per the change-grug skill) used to
prototype recurrent-depth / looped transformers. Model, train loop, and launch
wiring are all self-contained in `experiments/grug/reentrant` so each experiment
(E0 baseline, E1 re-entrant, ...) can be iterated independently. See
`.agents/projects/reentrant-model-testing.md` for the design and rollout.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import BlockShuffleConfig, LmDataConfig, TextLmDatasetFormat
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_data_config
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.reentrant.heuristic import build_from_heuristic
from experiments.grug.reentrant.model import GrugModelConfig
from experiments.grug.reentrant.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import nemotron_mix
from experiments.tokenization import default_tokenize


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def slimpajama_6b_data() -> LmDataConfig:
    """SlimPajama-6B, llama3-tokenized with block-shuffle, re-tokenized on first run.

    A small, R2-local corpus for GPU smoke/scale runs; returns a ready-to-train
    ``LmDataConfig``. A production pretraining mixture would instead need its
    tokenized cache already materialized to avoid a cross-region tokenize.
    """
    tokenize_step = default_tokenize(
        name="slimpajama-6b-cw",
        dataset="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )
    tokenize_step = dataclasses.replace(
        tokenize_step,
        config=dataclasses.replace(
            tokenize_step.config,
            # SlimPajama-6B tokenization OOMs at the default 10g worker_resources.
            worker_resources=ResourceConfig(ram="64g", disk="64g"),
        ),
    )
    return lm_data_config(
        training_set=tokenize_step,
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
    )


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=None,
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# Re-entrant experiment series. All runs share one compute budget / model size
# (the d512 ~130M-class MoE point) so re-entrant variants are directly comparable
# to the E0 baseline. Only the model architecture changes between experiments.
_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512
_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept because
# consumers (e.g. experiments/ferries/canary_ferry.py) import it by name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model

# v5p-8 in us-central1-a: region-local to the gs://marin-us-central1 data and
# checkpoint bucket and to the cluster controller. The marin v6e pool lives only
# in us-east/europe, which would force cross-region I/O. This matches the README
# d512 baseline hardware (the v5p pool's smallest slice is 8 chips).
_REENTRANT_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])


def reentrant_step(*, name: str, run_id: str, model: GrugModelConfig, tags: list[str]) -> ExecutorStep:
    """Build an ExecutorStep for one re-entrant experiment.

    Every experiment shares optimizer / batch / steps / data / trainer / eval with
    the E0 baseline; only ``model`` (and the run metadata) changes, so curves are
    directly comparable.
    """
    return ExecutorStep(
        name=name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(_REENTRANT_RESOURCES),
            steps=versioned(_baseline_steps),
            batch_size=versioned(_baseline_batch),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(project="marin_moe", tags=tags, group="reentrant", name=None),
            optimizer=versioned(_baseline_optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


# E0 — baseline: unchanged d512 Grug MoE. Reference curve for all re-entrant variants.
e0_baseline = reentrant_step(
    name="grug/reentrant_e0_d512",
    run_id=_resolve_run_id("reentrant_e0_d512"),
    model=_baseline_model,
    tags=["moe", "reentrant", "e0-baseline"],
)

# E1 — basic re-entrant: 1 prelude + 1 weight-tied core looped 4x + 1 coda.
# Effective depth 6 (compute-matched to E0) with 3 unique blocks (~half the block
# params). Tests Saunshi's "looped k-layer ~= kL-layer" at fixed compute.
_E1_MODEL = dataclasses.replace(
    _baseline_model,
    num_layers=3,
    num_prelude_layers=1,
    num_coda_layers=1,
    recurrence_steps=4,
)
e1_reentrant = reentrant_step(
    name="grug/reentrant_e1_loop4",
    run_id=_resolve_run_id("reentrant_e1_loop4"),
    model=_E1_MODEL,
    tags=["moe", "reentrant", "e1-loop4"],
)

# E2 — iteration-conditioned re-entrant: E1 + per-iteration FiLM (adaLN) on the
# shared core block. Tests whether telling the looped block which step it is on
# (coarse-to-fine) helps, at ~free parameter cost. Identity at init == E1.
_E2_MODEL = dataclasses.replace(_E1_MODEL, iteration_film=True)
e2_reentrant = reentrant_step(
    name="grug/reentrant_e2_filmloop4",
    run_id=_resolve_run_id("reentrant_e2_filmloop4"),
    model=_E2_MODEL,
    tags=["moe", "reentrant", "e2-film-loop4"],
)

# E3 — randomized-depth re-entrant: E1 with the core loop count sampled per step
# from {2,4,8} during training. Trains one weight-tied core to be correct at many
# depths so the SAME checkpoint can be evaluated at higher loop counts at test time
# (the depth-scaling experiment). recurrence_steps=4 stays the default/eval depth.
_E3_MODEL = dataclasses.replace(
    _E1_MODEL,
    randomize_recurrence=True,
    recurrence_choices=(2, 4, 8),
)
e3_reentrant = reentrant_step(
    name="grug/reentrant_e3_randdepth",
    run_id=_resolve_run_id("reentrant_e3_randdepth"),
    model=_E3_MODEL,
    tags=["moe", "reentrant", "e3-randdepth"],
)

# Experiment registry. Select with GRUG_EXPERIMENT (comma-separated names); default E0.
_STEPS = {"e0": e0_baseline, "e1": e1_reentrant, "e2": e2_reentrant, "e3": e3_reentrant}


if __name__ == "__main__":
    _selected = os.environ.get("GRUG_EXPERIMENT", "e0").split(",")
    executor_main(
        steps=[_STEPS[name.strip()] for name in _selected],
        description="Re-entrant grug MoE experiments (d512).",
    )
