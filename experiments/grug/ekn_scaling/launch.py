# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EKN scaling experiment: layer-count sweep (Small=12L, Medium=16L) on v4p-64.

Fixed K=4, E=32. Sweep lbl_coef in {0.0025, 0.01, 0.04}.
Steps truncated to 5k for quick iteration.

Small (12L): d_model=768, d_expert=384, lr=1.22e-3, bs=208.
Medium (16L): d_model=1024, d_expert=512, lr=9.57e-4, bs=376.
"""

import dataclasses
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
from experiments.grug.ekn_scaling.model import GrugModelConfig
from experiments.grug.ekn_scaling.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix

SWEEP_LBL = [0.0025, 0.01, 0.04]

# (num_layers, hidden_dim, num_heads, num_kv_heads, lr, batch_size)
SCALE_CONFIGS = {
    "small": (12, 768, 12, 4, 1.22e-3, 208),
    "medium": (16, 1024, 16, 4, 9.57e-4, 376),
}
FIXED_K = 4
FIXED_E = 32
FIXED_STEPS = 5_000


@dataclass(frozen=True)
class EknScalingLaunchConfig:
    """Launch config for the EKN scaling experiment."""

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


EKN_BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=768,
    intermediate_dim=384,
    shared_expert_intermediate_dim=384,
    num_experts=FIXED_E,
    num_experts_per_token=FIXED_K,
    num_layers=12,
    num_heads=12,
    num_kv_heads=4,
    max_seq_len=4096,
    initializer_std=0.006,
    load_balancing_loss_coef=0.01,
    router_z_loss_coef=0.001,
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


def run_ekn_scaling_trial(config: EknScalingLaunchConfig) -> None:
    """Run a single EKN scaling training trial."""
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
        description=config.description,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# ---------------------------------------------------------------------------
# Build one training step per (K, E, lbl) combination
# ---------------------------------------------------------------------------

RESOLVED_RUN_ID = _resolve_run_id("ekn-scaling-layer-run1-sweep")

training_steps: list[ExecutorStep] = []
for _scale_name, (_layers, _hidden, _heads, _kv_heads, _lr, _bs) in SCALE_CONFIGS.items():
    _intermediate = _hidden // 2
    for _lbl in SWEEP_LBL:
        _model = dataclasses.replace(
            EKN_BASE_MODEL,
            hidden_dim=_hidden,
            intermediate_dim=_intermediate,
            shared_expert_intermediate_dim=_intermediate,
            num_layers=_layers,
            num_heads=_heads,
            num_kv_heads=_kv_heads,
            load_balancing_loss_coef=_lbl,
        )
        _lbl_tag = f"lbl{_lbl}".replace(".", "p")
        _run_id = f"{RESOLVED_RUN_ID}-{_scale_name}-{_lbl_tag}"

        _step = ExecutorStep(
            name=f"grug/ekn-scaling-layer-run1-{_scale_name}-{_lbl_tag}",
            fn=run_ekn_scaling_trial,
            config=EknScalingLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v4p-64")),
                steps=versioned(FIXED_STEPS),
                batch_size=versioned(_bs),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                description=(
                    f"EKN layer scaling {_scale_name}: {_layers}L, d_model={_hidden},"
                    f" K={FIXED_K}, E={FIXED_E}, lbl={_lbl}, aux loss averaged over layers."
                ),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["grug", "ekn_scaling", _scale_name, "moe", f"k{FIXED_K}", f"e{FIXED_E}", _lbl_tag],
                    group="ekn-scaling-layer-sweep",
                    name=None,
                ),
                optimizer=versioned(
                    AdamConfig(
                        learning_rate=_lr,
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


if __name__ == "__main__":
    executor_main(
        steps=training_steps,
        description="EKN layer scaling: 12L/16L sweep with K=4, E=32, lbl_coef sweep.",
    )
