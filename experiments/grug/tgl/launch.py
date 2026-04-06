# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TGL expert-count sweep: 3 trials (2, 4, 8 routed experts) for 100 steps each on v4-8.

Base config (from the paper):
    hidden_dim=384, intermediate_dim=384, shared_expert_intermediate_dim=384,
    num_experts_per_token=2, num_layers=8, num_heads=8, num_kv_heads=2,
    max_seq_len=4096, vocab_size=128256, initializer_std=0.006,
    lbl_coef=0.01, rzl_coef=0.001,
    AdamW (lr=1.52e-3, b1=0.9, b2=0.95, wd=0.1), linear LR with 10% decay,
    batch_size=96, TPU v4-8.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.tgl.model import GrugModelConfig
from experiments.grug.tgl.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix

logger = logging.getLogger(__name__)

EXPERT_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256]


@dataclass(frozen=True)
class GrugTglLaunchConfig:
    """Last-mile run config for the TGL MoE baseline replication."""

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
    description: str = ""
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


@dataclass(frozen=True)
class AggregateMetricsConfig:
    """Config for aggregating tracker_metrics.jsonl from multiple runs."""

    run_paths: list[str]
    run_labels: list[str]
    output_path: str


GRUG_TGL_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=384,
    intermediate_dim=384,
    shared_expert_intermediate_dim=384,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=4096,
    initializer_std=0.006,
    lbl_coef=0.01,
    rzl_coef=0.001,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str, output_path: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id, replicate_path=output_path)
    return tracker


def run_grug_tgl_trial(config: GrugTglLaunchConfig) -> None:
    """Run a single TGL training trial."""
    trainer = TrainerConfig(
        mesh=MeshConfig(axes={"data": -1, "expert": 1, "model": 1}),
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id, config.output_path),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
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


def run_aggregate_metrics(config: AggregateMetricsConfig) -> None:
    """Read tracker_metrics.jsonl from each run, tag with run_label, write combined file."""
    import fsspec

    records: list[dict] = []
    for path, label in zip(config.run_paths, config.run_labels):
        metrics_file = f"{path}/tracker_metrics.jsonl"
        logger.info("Reading %s", metrics_file)
        fs, _, _ = fsspec.get_fs_token_paths(metrics_file)
        try:
            with fs.open(metrics_file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    record["run_label"] = label
                    records.append(record)
        except FileNotFoundError:
            logger.warning("Missing metrics file: %s", metrics_file)

    out_file = f"{config.output_path}/aggregated_metrics.jsonl"
    logger.info("Writing %d records to %s", len(records), out_file)
    fs, _, _ = fsspec.get_fs_token_paths(out_file)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(out_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Build one training step per expert count
# ---------------------------------------------------------------------------

RESOLVED_RUN_ID = _resolve_run_id("grug-tgl-phase1-run8-sweep")

training_steps: list[ExecutorStep] = []
for _num_experts in EXPERT_COUNTS:
    _model = dataclasses.replace(GRUG_TGL_MODEL, num_experts=_num_experts)
    _run_id = f"{RESOLVED_RUN_ID}-experts{_num_experts}"

    _step = ExecutorStep(
        name=f"grug/tgl-phase1-run8-sweep-e{_num_experts}",
        fn=run_grug_tgl_trial,
        config=GrugTglLaunchConfig(
            model=versioned(_model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-8")),
            steps=versioned(9_600),
            batch_size=versioned(96),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="dial_moe",
                tags=["grug", "tgl", "moe", f"experts{_num_experts}"],
                group="tgl-expert-sweep",
                name=None,
            ),
            optimizer=versioned(
                AdamConfig(
                    learning_rate=1.52e-3,
                    weight_decay=0.1,
                    beta1=0.9,
                    beta2=0.95,
                    lr_schedule="linear",
                    decay=0.1,
                    min_lr_ratio=0.1,
                    warmup=0.01,
                )
            ),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0,
                    ema_beta=None,
                    log_every=1,
                )
            ),
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
    training_steps.append(_step)

# ---------------------------------------------------------------------------
# Aggregation step — depends on all training steps via as_input_name()
# ---------------------------------------------------------------------------

aggregate_step = ExecutorStep(
    name="grug/tgl-phase1-run8-sweep-aggregate",
    fn=run_aggregate_metrics,
    config=AggregateMetricsConfig(
        run_paths=[s.as_input_name() for s in training_steps],
        run_labels=[f"experts{n}" for n in EXPERT_COUNTS],
        output_path=this_output_path(),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=training_steps + [aggregate_step],
        description="TGL expert-count sweep (run8, run8 params, Switch-style LBL) + metric aggregation.",
    )
