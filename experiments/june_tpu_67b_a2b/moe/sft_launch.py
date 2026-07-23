# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug MoE backend for the general chat-SFT launcher.

``GrugModel`` is the :class:`~experiments.sft.launcher.ModelSource` for the native ``GrugTrainState``
checkpoint: it plugs the vendored Grug training engine into ``experiments.sft.launcher.sft_step`` so
a Grug run shares that launcher's dataset transforms, chat template, spec, and CLI while keeping its
own model pytree, optimizer, mesh, and train loop. The dependency runs vendored experiment ->
launcher, never the reverse.

Two things make Grug a distinct backend rather than a config on the Levanter ``train_lm`` path:

- Weights-only native init (marin #650): ``init_from`` supplies only the model weights
  (``params`` + ``pending_qb_betas``) from a native Levanter Grug checkpoint; the optimizer state and
  step counter start fresh, so SFT runs a new LR schedule over the base weights. Wired through
  ``GrugTrainerConfig.sft_weights_only_init`` (see ``train.py``). Own-run checkpoints still take
  precedence, so iris preemption auto-resumes. ``initialize_from_hf`` (the Levanter path) does not
  apply — the base is a native Grug pytree, not an HF checkpoint.
- Model + train loop: the vendored ``Transformer`` is not a Levanter-registry ``LmHeadModel``, so
  it runs through ``run_grug`` with its own ring-EP mesh, not ``train_lm.main``.

The data side is shared: ``sft_step`` builds the chat ``LmDataConfig`` from the same
``transform_dataset_step`` outputs used for every other model, and hands it to
:meth:`GrugModel.build_train_config`, which the Grug trainer consumes unchanged.

The model architecture must match the checkpoint exactly (this is why the launcher lives in the
vendored tree, not ``experiments/grug/moe/``): pass the same ``GrugModelConfig`` the checkpoint was
trained with. ``build_train_config`` pins ``model.max_seq_len`` to ``spec.seq_len`` so the model and
the training sequence length cannot drift apart.
"""

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
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

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.sft.launcher import SFTSpec


@dataclass(frozen=True)
class GrugMoeSFTConfig:
    """Launch config for a Grug MoE SFT run (weights-only init + chat data)."""

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
    """Base checkpoint to initialise weights from (parent dir or a concrete ``step-N`` dir;
    the latest under it is loaded). Optimizer state and step are not taken from it -- SFT
    starts a fresh schedule. Loaded only on the first launch; iris restarts auto-resume from
    this run's own output checkpoints instead."""
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = None
    expert_parallel: int = 1
    checkpointer: CheckpointerConfig | None = None
    checkpoint_keep: list[dict] | None = None
    save_interval_minutes: int = 30
    per_device_parallelism: int = -1


def run_grug_moe_sft_trial(config: GrugMoeSFTConfig) -> None:
    """Map SFT launch knobs onto a Levanter trainer and dispatch the run.

    Resolves the latest checkpoint under ``init_from_path`` and hands it to the weights-only
    init path (``sft_weights_only_init=True``): only the base weights load, the optimizer state
    and step counter start fresh, so SFT runs a new LR schedule from step 0.
    """
    if config.model.num_experts <= 1:
        # marin #6252: the single-expert training path is buggy; SFT of an MoE must keep >1.
        raise ValueError(f"Grug SFT expects an MoE (num_experts > 1), got {config.model.num_experts}.")

    initialize_from = latest_checkpoint_path(config.init_from_path)

    # Trainer mesh bookkeeping. Grug builds its own compact (replica_dcn, data, expert, model) mesh for
    # the actual compute (train.py, via set_mesh + raw PartitionSpecs -- not the Trainer's logical axis
    # mapping), but the TrainerConfig still derives ``data_axis_size`` (and thus the batch-divisibility
    # check + per_device_parallelism) from this MeshConfig. With only ``expert`` declared, the
    # model-parallel slices get absorbed as ``replica_dcn`` and the default batch mapping
    # (replica_dcn, replica, data) then counts them -> data_axis_size = num_slices, which rejects bs=8 on
    # model_axis=5 ("train_batch_size (8) must be divisible by per_device_parallelism * data_axis_size
    # (1, 5)"). For model_axis>1, bs=8 forces data=1, so the true batch shards only over ``expert``: map
    # the batch axis to (data, expert) -> data_axis_size == expert_axis_size and per_device resolves to 1.
    # (``model`` cannot be declared in dcn_axes -- MeshConfig always seeds model into the ICI axes, which
    # would collide.) model_axis==1 keeps the original single-axis MeshConfig byte-identically.
    _model_axis = config.grug_trainer.model_axis_size
    if _model_axis > 1:
        mesh_config = MeshConfig(
            axes={"expert": config.expert_parallel},
            compute_mapping={"batch": ["data", "expert"]},
        )
    else:
        mesh_config = MeshConfig(axes={"expert": config.expert_parallel})

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=mesh_config,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=prefix_join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=config.save_interval_minutes),
            keep=config.checkpoint_keep,
        ),
        # First launch: output dir empty -> weights-only init from initialize_from. Once this
        # run saves its own checkpoints, every restart auto-resumes from those (full SFT state).
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=initialize_from,
    )

    grug_trainer = dataclasses.replace(
        config.grug_trainer,
        trainer=trainer,
        expert_axis_size=config.expert_parallel,
        sft_weights_only_init=True,
    )

    run_grug(
        GrugRunConfig(
            model=config.model,
            data=config.data,
            resources=config.resources,
            optimizer=config.optimizer,
            trainer=grug_trainer,
            eval=config.eval,
        )
    )


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


@dataclass(frozen=True)
class GrugModel:
    """``ModelSource`` for the native Grug MoE: weights-only init + the ring-EP ``run_grug`` backend.

    ``init_from`` is the base checkpoint: a static native-Levanter Grug checkpoint dir (the cooldown
    ``step-N``) or a dependency step whose ``checkpoints`` output is chained (a prior Grug ``sft_step``
    for two-stage SFT). Mesh geometry (``expert_parallel`` / ``model_axis`` / ``replica_axis``) and
    ``mp`` are validated for this model on its target cluster, so they live with the source rather
    than being free launch flags.
    """

    model: GrugModelConfig
    tokenizer_path: str
    init_from: str | ArtifactStep
    expert_parallel: int
    model_axis: int = 1
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

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_path

    @property
    def run(self) -> Callable[..., None]:
        return run_grug_moe_sft_trial

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return (self.init_from,) if isinstance(self.init_from, ArtifactStep) else ()

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config: LmDataConfig,
        resources: ResourceConfig,
        num_train_steps: int,
    ) -> GrugMoeSFTConfig:
        if isinstance(self.init_from, ArtifactStep):
            # A chained Grug stage: init from the prior stage's saved checkpoints.
            init_from_path = prefix_join(ctx.artifact_path(self.init_from), "checkpoints")
        else:
            init_from_path = self.init_from
        run_id = spec.name.split("/")[-1]
        tracker = WandbConfig(
            project=spec.wandb_project,
            tags=list(self.wandb_tags),
            group=self.wandb_group,
            name=run_id,
        )
        return GrugMoeSFTConfig(
            # Pin the model's max_seq_len to the run's seq_len so arch and training length can't drift.
            model=dataclasses.replace(self.model, max_seq_len=spec.seq_len),
            data=data_config,
            output_path=ctx.output_path,
            run_id=run_id,
            resources=resources,
            steps=num_train_steps,
            batch_size=spec.batch_size,
            seed=self.seed,
            mp=self.mp,
            tracker=tracker,
            optimizer=spec.optimizer,
            init_from_path=init_from_path,
            expert_parallel=self.expert_parallel,
            per_device_parallelism=self.per_device_parallelism,
            save_interval_minutes=self.save_interval_minutes,
            checkpoint_keep=self.checkpoint_keep,
            grug_trainer=GrugTrainerConfig(
                z_loss_weight=self.z_loss_weight,
                ema_beta=self.ema_beta,
                log_every=self.log_every,
                replica_axis_size=self.replica_axis,
                model_axis_size=self.model_axis,
            ),
            eval=None,
        )
