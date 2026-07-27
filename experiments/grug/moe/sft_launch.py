# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Grug-MoE backend for the shared chat-SFT launcher."""

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem import prefix_join

from experiments.grug.moe.launch import resolve_tracker
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    InitializationMode,
    run_grug,
)
from experiments.sft.launcher import SFTSpec


@dataclass(frozen=True)
class GrugMoeSFTConfig:
    """Launch config for fresh-optimizer SFT from a native Grug checkpoint."""

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
    init_from_path: str
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    expert_parallel: int = 1
    checkpoint_keep: list[dict] | None = None
    save_interval_minutes: int = 30
    per_device_parallelism: int = -1


def run_grug_moe_sft_trial(config: GrugMoeSFTConfig) -> None:
    """Dispatch completion-masked chat SFT with weights-only initialization."""
    if config.model.num_experts <= 1:
        raise ValueError(f"Grug SFT expects an MoE, got {config.model.num_experts} experts")

    initialize_from = latest_checkpoint_path(config.init_from_path)
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.steps,
        mp=jmp.get_policy(config.mp),
        tracker=resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=config.save_interval_minutes),
            keep=config.checkpoint_keep,
        ),
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=initialize_from,
    )
    grug_trainer = dataclasses.replace(
        config.grug_trainer,
        trainer=trainer,
        expert_axis_size=config.expert_parallel,
        initialization_mode=InitializationMode.WEIGHTS_ONLY,
    )
    run_grug(
        GrugRunConfig(
            model=config.model,
            data=config.data,
            resources=config.resources,
            optimizer=config.optimizer,
            trainer=grug_trainer,
            eval=None,
        )
    )


@dataclass(frozen=True)
class GrugModel:
    """Model source connecting a native Grug checkpoint to shared chat data."""

    model: GrugModelConfig
    tokenizer_path: str
    init_from_path: str
    expert_parallel: int
    replica_axis: int = 1
    per_device_parallelism: int = -1
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"
    z_loss_weight: float = 1e-4
    ema_beta: float | None = None
    log_every: int = 1
    seed: int = 0
    save_interval_minutes: int = 30
    checkpoint_keep: list[dict] | None = None
    wandb_tags: Sequence[str] = ()
    wandb_group: str | None = None

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path

    def resolve_tokenizer(self, _ctx: StepContext) -> str:
        return self.tokenizer_path

    @property
    def run(self) -> Callable[..., None]:
        return run_grug_moe_sft_trial

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> GrugMoeSFTConfig:
        run_id = spec.name.split("/")[-1]
        return GrugMoeSFTConfig(
            model=dataclasses.replace(self.model, max_seq_len=spec.seq_len),
            data=data_config,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=resources,
            steps=num_train_steps,
            batch_size=spec.batch_size,
            seed=self.seed,
            mp=self.mp,
            tracker=WandbConfig(
                project=spec.wandb_project,
                tags=list(self.wandb_tags),
                group=self.wandb_group,
                name=run_id,
            ),
            optimizer=spec.optimizer,
            init_from_path=self.init_from_path,
            expert_parallel=self.expert_parallel,
            per_device_parallelism=self.per_device_parallelism,
            save_interval_minutes=self.save_interval_minutes,
            checkpoint_keep=self.checkpoint_keep,
            grug_trainer=GrugTrainerConfig(
                z_loss_weight=self.z_loss_weight,
                ema_beta=self.ema_beta,
                log_every=self.log_every,
                replica_axis_size=self.replica_axis,
            ),
        )
