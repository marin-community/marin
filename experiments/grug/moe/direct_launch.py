# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct-submission launch for the May MoE Recipe.

Bypasses ``executor_main`` / ``ExecutorStep``. The script builds a
``GrugMoeDirectLaunchConfig``, submits it directly to Iris via the Fray
client, and exits. The iris coordinator job is independent of the local
process, so closing the laptop is safe (``max_retries_preemption``
defaults to 100 on the JobRequest).

The model reuses :mod:`experiments.grug.moe.model` and the trainer reuses
:mod:`experiments.grug.moe.train`; only the launch wiring changes.

All architecture/recipe knobs (PKO every-4th + last-layer, half-rope on
non-PKO layers, split ``w_gate``/``w_up``, ``routing_renorm_sum=2.5``,
router z-loss off) are baked into ``experiments.grug.moe.model``. Only
sizes and the 1pct-noclip optimizer recipe live here.

Usage:

    .venv/bin/python -m experiments.grug.moe.direct_launch

Submits one job to Iris (no zone pin — iris picks based on capacity /
reservation) and prints the job id immediately.
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
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import (
    compute_output_path,
    materialize,
    resolve_local_placeholders,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import _submit_train_job, default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe.train import (
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    _run_grug_local,
)
from experiments.pretraining_datasets import nemotron_mix


@dataclass(frozen=True)
class GrugMoeDirectLaunchConfig:
    """Last-mile run config for the direct-submission MoE template."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # Step interval for permanent checkpoints. ``None`` = only temporary
    # (every save_interval) + final permanent checkpoint at end of training.
    checkpoint_keep_every: int | None = 1000


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def _build_grug_run_config(launch: GrugMoeDirectLaunchConfig, *, output_path: str) -> GrugRunConfig:
    """Map launch knobs into the trainer's full ``GrugRunConfig``."""
    keep_every = launch.checkpoint_keep_every
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=launch.seed,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(launch.mp),
        tracker=_resolve_tracker(launch.tracker, launch.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[] if keep_every is None else [{"every": keep_every}],
        ),
    )

    grug_trainer = dataclasses.replace(launch.grug_trainer, trainer=trainer)
    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        optimizer=launch.optimizer,
        trainer=grug_trainer,
        eval=launch.eval,
    )


def resolve_grug_run_config(
    name: str,
    raw_launch: GrugMoeDirectLaunchConfig,
    override_output_path: str | None = None,
) -> GrugRunConfig:
    """Resolve a placeholder-bearing launch config into a runnable run config.

    Designed to be invoked on the Iris worker so paths reflect the worker's
    region.
    """
    output_path = compute_output_path(name, raw_launch, override_output_path=override_output_path)
    launch = resolve_local_placeholders(raw_launch, output_path)
    run_config = _build_grug_run_config(launch, output_path=output_path)
    return materialize(run_config)


def _run_grug_moe_on_worker(
    name: str,
    raw_launch: GrugMoeDirectLaunchConfig,
    override_output_path: str | None,
) -> None:
    """MoE training entrypoint: resolve under worker region, then run locally.

    Top-level so Fray can pickle it as a JobRequest entrypoint.
    """
    run_config = resolve_grug_run_config(name, raw_launch, override_output_path)
    _run_grug_local(run_config)


def train_grug_moe(
    name: str,
    launch: GrugMoeDirectLaunchConfig,
    *,
    override_output_path: str | None = None,
    env_vars: dict[str, str] | None = None,
    wait: bool = True,
) -> str:
    """Submit a MoE training job to Iris.

    Defaults to ``wait=True`` because the calling process is the Iris
    lifecycle anchor for the child training job — if the caller exits before
    the child finishes, Iris finalizes (kills) the child. To detach from
    your laptop, wrap the script with ``iris job run --no-wait``: the
    coordinator job stays alive inside Iris and waits on the training job
    while your laptop disconnects.

    Combined with ``max_retries_preemption=100`` (the default on
    ``JobRequest``), the training job auto-retries on TPU preemption.

    Returns:
        The Iris job id.
    """
    resources = unwrap_versioned_value(launch.resources)
    env = dict(env_vars or {})
    if "WANDB_API_KEY" not in env and "WANDB_API_KEY" in os.environ:
        env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    return _submit_train_job(
        name=name,
        entrypoint_callable=_run_grug_moe_on_worker,
        args=[name, launch, override_output_path],
        resources=resources,
        env_vars=env,
        wait=wait,
    )


# --- Default test config: d512 at its compute-optimal budget on v5p-8 ---

_TEST_HIDDEN_DIM: int = 512
_TEST_BUDGET: float = 2.19e17  # heuristic anchor for d512
_TEST_TARGET_STEPS: int = 2**14
_TEST_TPU: str = "v5p-8"
_TEST_RUN_SUFFIX: str = "v1"


def _build_test_launch() -> GrugMoeDirectLaunchConfig:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=_TEST_BUDGET,
        hidden_dim=_TEST_HIDDEN_DIM,
        target_steps=_TEST_TARGET_STEPS,
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=0.01,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )

    run_id = _resolve_run_id(
        f"grug-moe-direct-d{_TEST_HIDDEN_DIM}-{_TEST_BUDGET:.2e}-{_TEST_RUN_SUFFIX}".replace("+", "")
    )
    resources = ResourceConfig.with_tpu(_TEST_TPU)

    return GrugMoeDirectLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=resources,
        steps=versioned(num_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            tags=["moe", "moe_direct", "may_recipe", f"d{_TEST_HIDDEN_DIM}"],
            group="grug-moe-direct",
            name=None,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
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
        checkpoint_keep_every=1000,
    )


if __name__ == "__main__":
    launch = _build_test_launch()
    # wait=True so the iris coordinator blocks on the training job; iris kills
    # orphaned children when their parent exits. Detach from the laptop via
    # ``iris job run --no-wait`` on the outer wrapper.
    job_id = train_grug_moe(
        name=f"grug/moe-direct-d{_TEST_HIDDEN_DIM}-{_TEST_RUN_SUFFIX}",
        launch=launch,
        wait=True,
    )
    print(f"Training job finished: {job_id}")
