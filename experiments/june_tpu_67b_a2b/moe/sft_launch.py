# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT launcher for the vendored June TPU 67B-A2B Grug MoE.

A sibling to ``launch.py``'s ``run_grug_moe_trial`` and ``launch_2x_bs.py``'s
``run_grug_moe_trial_2x_bs``, specialised for supervised fine-tuning of the native
``GrugTrainState`` checkpoint:

- **Weights-only init (marin #650).** ``init_from_path`` supplies only the model weights
  (``params`` + ``pending_qb_betas``); the optimizer state and step counter start fresh, so
  SFT runs a new LR schedule over the base weights instead of resuming the base run's
  optimizer/step. This is wired through ``GrugTrainerConfig.sft_weights_only_init`` (see
  ``train.py``). Own-run checkpoints still take precedence, so iris preemption auto-resumes.
- **Chat data + completions masking + packing.** ``build_grug_chat_data_config`` assembles an
  ``LmDataConfig`` whose components are ``ChatLmDatasetFormat`` (pluggable chat template,
  ``mask_user_turns`` so only the assistant span is supervised, packing on) reading HF chat
  datasets directly — the Grug trainer already consumes ``LmDataConfig`` unchanged.

The model architecture must match the checkpoint exactly (this is why the launcher lives in the
vendored tree, not ``experiments/grug/moe/``): pass the same ``GrugModelConfig`` the checkpoint
was trained with.
"""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import DatasetComponent, HfDatasetSourceConfig, LmDataConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.optim.config import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug


@dataclass(frozen=True)
class ChatDatasetSpec:
    """One chat SFT source. ``messages_field`` names the transcript column (both target
    datasets are already OpenAI role/content lists, so no ShareGPT remap is needed:
    wildchat uses ``conversation``, nemotron uses ``messages``)."""

    slug: str
    hf_dataset_id: str
    revision: str
    messages_field: str
    weight: float = 1.0


def build_grug_chat_data_config(
    *,
    datasets: Sequence[ChatDatasetSpec],
    tokenizer: str,
    chat_template: str,
    pack: bool | None = None,
    mixture_block_size: int = 2048,
) -> LmDataConfig:
    """Assemble a chat-SFT ``LmDataConfig`` the Grug trainer can consume directly.

    Each component reads its HF dataset through a ``ChatLmDatasetFormat`` carrying the shared
    ``chat_template`` and ``mask_user_turns=True`` (assistant span only). ``pack=None`` uses
    the format default (packing on); ``auto_build_caches`` builds each cache on first use.
    """
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    for d in datasets:
        components[d.slug] = DatasetComponent(
            source=HfDatasetSourceConfig(id=d.hf_dataset_id, revision=d.revision),
            format=ChatLmDatasetFormat(
                messages_field=d.messages_field,
                chat_template=chat_template,
                mask_user_turns=True,
                pack=pack,
            ),
            split="train",
        )
        weights[d.slug] = d.weight
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        chat_template=chat_template,
        enforce_eos=True,
        auto_build_caches=True,
        components=components,
        train_weights=weights,
        mixture_block_size=mixture_block_size,
    )


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
    """Base checkpoint to initialise WEIGHTS from (parent dir or a concrete ``step-N`` dir;
    the latest under it is loaded). Optimizer state and step are NOT taken from it -- SFT
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
    init path (``sft_weights_only_init=True``); the optimizer/step reset happens inside
    ``_run_grug_local``.
    """
    if config.model.num_experts <= 1:
        # marin #6252: the single-expert training path is buggy; SFT of an MoE must keep >1.
        raise ValueError(f"Grug SFT expects an MoE (num_experts > 1), got {config.model.num_experts}.")

    initialize_from = latest_checkpoint_path(config.init_from_path)

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
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
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
