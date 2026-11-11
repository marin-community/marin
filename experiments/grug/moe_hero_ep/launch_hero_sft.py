# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch full-model hero SFT on the Grug 67B 10.05 recipe."""

import dataclasses
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import click
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import (
    ConcatDatasetComponent,
    DatasetComponent,
    DatasetComponentBase,
    LmDataConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import checkpoint_resume_paths, temporary_checkpoint_base_path
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.harrier_length_stats import harrier_length_statistics
from experiments.grug.moe_hero_ep.harrier_long_context_mix import harrier_long_context_data_config
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import HARRIER_MIX_2026_08_18_STORE, validation_components
from experiments.grug.moe_hero_ep.hero_recipe import (
    HERO_BALANCE_ATTENTION_ROWS,
    HERO_CONTEXT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_PROCESSES_PER_TASK,
    HERO_RACK_GPUS,
    HERO_SCHEDULE_STEPS,
    HERO_TENSORSTORE_CACHE_BYTES,
    HERO_WATCH_INTERVAL,
    HeroThroughputResult,
    hero_expert_axis_size,
    hero_grug_trainer_config,
    hero_rack_batch_size,
    hero_trainer_config,
    validate_model_split,
    validated_batch_axis_size,
    validation_datasets,
    with_transport_remat_mode,
)
from experiments.grug.moe_hero_ep.heuristic import build_hero_configs
from experiments.grug.moe_hero_ep.launch_scaling_ladder import (
    HERO_PROCESS_STALL_TIMEOUT,
    HERO_RESUME_CHECKPOINT_PREFIX,
    HERO_STARTUP_TIMEOUT,
    HERO_STEP_TIMEOUT,
    LADDER_RACKS,
    RESUME_CHECKPOINT_TTL_DAYS,
    RESUME_SAVE_INTERVAL,
)
from experiments.grug.moe_hero_ep.train import (
    GrugEvalConfig,
    GrugRunConfig,
    MixtureContinuation,
    RouterBiasMode,
    SpecialTokenInit,
    TrainingDataMode,
    WatchMode,
    run_grug,
)

logger = logging.getLogger(__name__)

SFT_WANDB_PROJECT = "marin_moe_sft"
# CoreWeave copies of gs://marin-us-central2/grug_sft/stores/full-ab-packed-2026.09.29.json, the 10.05 stores.
SFT_STORES_MANIFEST = Path(__file__).with_name("hero_sft_stores_2026_09_29.json")
SFT_TOKENIZER = "s3://marin-us-east-02a/marin/grug_sft/tokenizer/2026.09.18"
# The 10.05 run's 64,000-sequence mixture blocks, rounded up to whole training steps.
SFT_TARGET_BLOCK_SIZE = 64_000
SFT_COMPONENT_PREFIX = "sft/source/"
SFT_POOLED_COMPONENT = "sft/pooled"
REPLAY_COMPONENT_PREFIX = "replay/"
# 10.05: constant learning rate, then linear decay to 0.1x over the final tenth, with no warmup.
SFT_DECAY_FRACTION = 0.1
SFT_MIN_DECAY_STEPS = 2
SFT_MIN_LR_RATIO = 0.1
SPECIAL_TOKEN_LR_MULTIPLIER = math.sqrt(32)
# The highest routed capacity factor that fits at CP4 x EP16 with the default remat slop. At 1.6875 the
# first all-to-all runs out of HBM, and at 1.75 and 2.0 the rematerialization check fails during compile.
SFT_CAPACITY_FACTOR = 1.625
# Chat special tokens of the 09.18 tokenizer and the single ordinary token each LM-head column copies.
SPECIAL_TOKEN_ANCHORS = (
    (128006, " role"),
    (128007, " message"),
    (128002, " think"),
    (128003, " answer"),
    (128005, " tool"),
    (128011, " end"),
    (128009, "<|end_of_text|>"),
)
EXPECTED_ANCHOR_IDS = (3560, 1984, 1781, 4320, 5507, 842, 128001)


@dataclass(frozen=True)
class SftStore:
    cache_path: str
    tokenizer: str
    max_length: int
    packed_sequences: int


def load_sft_stores(manifest: Path = SFT_STORES_MANIFEST) -> dict[str, SftStore]:
    """Read the CoreWeave store manifest; every store must use the SFT tokenizer and the hero context."""
    raw = json.loads(manifest.read_text())
    stores = {name: SftStore(**record) for name, record in raw["stores"].items()}
    context = HERO_MODEL_CONFIG.max_seq_len
    for name, store in stores.items():
        if store.tokenizer != SFT_TOKENIZER:
            raise ValueError(f"{name} uses tokenizer {store.tokenizer}, expected {SFT_TOKENIZER}")
        if store.max_length != context:
            raise ValueError(f"{name} is packed at {store.max_length}, expected {context}")
        if store.packed_sequences <= 0:
            raise ValueError(f"{name} has no packed sequences")
    return stores


@dataclass(frozen=True)
class SftMixturePlan:
    """One epoch of the SFT stores, spread evenly across complete mixture blocks."""

    batch_size: int
    block_size: int
    num_blocks: int
    sft_sequences: int

    @property
    def run_sequences(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def steps(self) -> int:
        return self.run_sequences // self.batch_size

    @property
    def replay_fraction(self) -> float:
        return 1.0 - self.sft_sequences / self.run_sequences

    @property
    def decay_steps(self) -> int:
        return max(SFT_MIN_DECAY_STEPS, round(self.steps * SFT_DECAY_FRACTION))


def plan_sft_mixture(stores: Mapping[str, SftStore], *, batch_size: int, replay_fraction: float) -> SftMixturePlan:
    """Choose the block count whose replay share is closest to ``replay_fraction``."""
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(f"replay_fraction must be in (0, 1), got {replay_fraction}")
    block_size = batch_size * math.ceil(SFT_TARGET_BLOCK_SIZE / batch_size)
    sft_sequences = sum(store.packed_sequences for store in stores.values())
    num_blocks = max(1, round(sft_sequences / ((1.0 - replay_fraction) * block_size)))
    if sft_sequences >= num_blocks * block_size:
        raise ValueError("The SFT epoch leaves no room for replay; lower the replay fraction")
    return SftMixturePlan(
        batch_size=batch_size, block_size=block_size, num_blocks=num_blocks, sft_sequences=sft_sequences
    )


SFT_POSITION_METADATA_KEY = "sft_mixture_position"


@dataclass(frozen=True)
class SftMixturePosition:
    """Where a run reads its SFT epoch: ``step`` reads mixture sequence ``sequence``, ``batch_size`` per step.

    ``plan_dp_racks`` is the rack count that planned the epoch's mixture blocks. Each checkpoint records the
    position, so a later handoff resumes the same stream however many handoffs came before.
    """

    plan_dp_racks: int
    step: int
    sequence: int
    batch_size: int

    def sequence_at(self, step: int) -> int:
        if step < self.step:
            raise ValueError(f"step {step} precedes this run's first step {self.step}")
        return self.sequence + (step - self.step) * self.batch_size

    def checkpoint_metadata(self) -> dict[str, object]:
        return {SFT_POSITION_METADATA_KEY: dataclasses.asdict(self)}

    @classmethod
    def recorded_in(cls, metadata: Mapping[str, object]) -> "SftMixturePosition | None":
        raw = metadata.get(SFT_POSITION_METADATA_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"{SFT_POSITION_METADATA_KEY} must be a mapping, got {raw!r}")
        return cls(**raw)


@dataclass(frozen=True)
class SftHandoff:
    """A run of the same SFT epoch to continue from its permanent checkpoint at ``step``.

    ``step`` reads mixture sequence ``sequence`` of the epoch that ``plan_dp_racks`` racks planned.
    """

    checkpoint: str
    step: int
    plan_dp_racks: int
    sequence: int


@dataclass(frozen=True)
class SftSteps:
    """The steps a run trains: ``first_step`` reads mixture sequence ``first_sequence``."""

    first_step: int
    first_sequence: int
    num_train_steps: int
    decay_steps: int


def sft_steps(plan: SftMixturePlan, *, start_step: int, batch_size: int, handoff: SftHandoff | None = None) -> SftSteps:
    """Steps that read the rest of the plan's epoch once at ``batch_size``, from ``handoff`` if set.

    The decay covers the same final sequences as the plan's, and fewer than ``batch_size`` sequences at the
    end of the epoch stay unread.
    """
    if handoff is None:
        if batch_size != plan.batch_size:
            raise ValueError(f"batch {batch_size} does not match the plan's {plan.batch_size}")
        return SftSteps(start_step, 0, start_step + plan.steps, plan.decay_steps)
    if not 0 < handoff.sequence < plan.run_sequences:
        raise ValueError(f"handoff sequence {handoff.sequence} is outside the {plan.run_sequences}-sequence epoch")
    num_train_steps = handoff.step + (plan.run_sequences - handoff.sequence) // batch_size
    decay_steps = max(SFT_MIN_DECAY_STEPS, round(plan.decay_steps * plan.batch_size / batch_size))
    if num_train_steps - decay_steps < handoff.step:
        raise ValueError(f"handoff step {handoff.step} falls inside the learning-rate decay")
    return SftSteps(handoff.step, handoff.sequence, num_train_steps, decay_steps)


def _store_component(store: SftStore) -> DatasetComponent:
    # The stores hold only a train split; a flat cache gives evaluation no validation split to look for.
    return DatasetComponent(
        source=None,
        cache_dir=prefix_join(store.cache_path, "train"),
        format=TextLmDatasetFormat(),
        pack=store.max_length,
        flat_cache=True,
    )


def sft_components(
    stores: Mapping[str, SftStore], plan: SftMixturePlan
) -> tuple[dict[str, DatasetComponentBase], dict[str, int]]:
    """SFT components and the packed sequences each one contributes to the epoch.

    A store with fewer sequences than mixture blocks cannot appear in every block, so all such stores
    share one concatenated component.
    """
    components: dict[str, DatasetComponentBase] = {}
    counts: dict[str, int] = {}
    pooled: dict[str, DatasetComponent] = {}
    for name, store in sorted(stores.items()):
        if store.packed_sequences < plan.num_blocks:
            pooled[name] = _store_component(store)
            continue
        components[SFT_COMPONENT_PREFIX + name] = _store_component(store)
        counts[SFT_COMPONENT_PREFIX + name] = store.packed_sequences
    if pooled:
        components[SFT_POOLED_COMPONENT] = ConcatDatasetComponent(children=pooled)
        counts[SFT_POOLED_COMPONENT] = sum(stores[name].packed_sequences for name in pooled)
        if counts[SFT_POOLED_COMPONENT] < plan.num_blocks:
            raise ValueError("The pooled SFT component needs at least one sequence per mixture block")
    return components, counts


def one_epoch_weight_stages(
    sft_counts: Mapping[str, int], replay_weights: Mapping[str, float], plan: SftMixturePlan
) -> list[tuple[int, dict[str, float]]]:
    """Per-block mixture weights that read each SFT component exactly once over the run.

    Each block takes an even share of every SFT component's sequences and fills the rest with replay,
    split by ``replay_weights`` with at least one draw per replay component. Stage steps count from the
    mixture's start step (``LmDataConfig.mixture_start_step``).
    """
    if plan.block_size % plan.batch_size:
        raise ValueError("Mixture block boundaries must fall on training steps")
    if not replay_weights or any(weight <= 0 for weight in replay_weights.values()):
        raise ValueError("Replay needs at least one positive-weight component")
    replay_total = sum(replay_weights.values())
    stages = []
    for block in range(plan.num_blocks):
        counts = {
            name: (block + 1) * total // plan.num_blocks - block * total // plan.num_blocks
            for name, total in sft_counts.items()
        }
        replay_draws = plan.block_size - sum(counts.values())
        targets = {name: replay_draws * weight / replay_total for name, weight in replay_weights.items()}
        replay_counts = {name: max(1, int(target)) for name, target in targets.items()}
        remainder = replay_draws - sum(replay_counts.values())
        if remainder < 0:
            raise ValueError("Replay components do not fit in the mixture block")
        for name in sorted(targets, key=lambda name: targets[name] % 1, reverse=True)[:remainder]:
            replay_counts[name] += 1
        counts.update(replay_counts)
        assert sum(counts.values()) == plan.block_size
        stages.append(
            (block * plan.block_size // plan.batch_size, {name: n / plan.block_size for name, n in counts.items()})
        )
    return stages


def sft_data_config(
    stores: Mapping[str, SftStore],
    plan: SftMixturePlan,
    *,
    start_step: int,
    replay: LmDataConfig,
    validation: Mapping[str, DatasetComponentBase],
) -> LmDataConfig:
    """The SFT epoch from ``start_step``, replay from the first stage of ``replay``, and zero-weight ``validation``."""
    components, sft_counts = sft_components(stores, plan)
    replay_stage = replay.train_weights[0][1] if isinstance(replay.train_weights, list) else replay.train_weights
    assert replay_stage is not None
    replay_weights = {}
    for name, weight in replay_stage.items():
        if weight <= 0:
            continue
        components[REPLAY_COMPONENT_PREFIX + name] = replay.components[name]
        replay_weights[REPLAY_COMPONENT_PREFIX + name] = weight
    stages = one_epoch_weight_stages(sft_counts, replay_weights, plan)
    if collisions := components.keys() & validation.keys():
        raise ValueError(f"validation components collide with training components: {sorted(collisions)}")
    components.update(validation)
    stages = [(step, {**weights, **{name: 0.0 for name in validation}}) for step, weights in stages]
    return LmDataConfig(
        tokenizer=SFT_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights=stages,
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        mixture_block_size=plan.block_size,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
        mixture_start_step=start_step,
    )


def special_token_init(start_step: int, tokenizer_path: str = SFT_TOKENIZER) -> SpecialTokenInit:
    """The seven chat-token reseeds, checked against the SFT tokenizer."""
    tokenizer = load_tokenizer(tokenizer_path)
    anchors = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for _, text in SPECIAL_TOKEN_ANCHORS)
    if anchors != tuple((anchor,) for anchor in EXPECTED_ANCHOR_IDS):
        raise ValueError(f"SFT tokenizer anchors changed: {anchors}")
    return SpecialTokenInit(
        token_ids=tuple(token_id for token_id, _ in SPECIAL_TOKEN_ANCHORS),
        anchor_ids=EXPECTED_ANCHOR_IDS,
        start_step=start_step,
        lr_multiplier=SPECIAL_TOKEN_LR_MULTIPLIER,
    )


def permanent_checkpoint_metadata(checkpoint: str) -> dict[str, object]:
    """A complete permanent checkpoint's ``metadata.json``."""
    metadata = json.loads((StoragePath(checkpoint) / "metadata.json").read_text())
    if metadata.get("is_temporary"):
        raise ValueError(f"{checkpoint} is a temporary checkpoint")
    if StoragePath(checkpoint).name != f"step-{int(metadata['step'])}":
        raise ValueError(f"{checkpoint} records step {metadata['step']}")
    return metadata


def checkpoint_step(checkpoint: str) -> int:
    """The step recorded in a complete checkpoint's ``metadata.json``."""
    return int(permanent_checkpoint_metadata(checkpoint)["step"])


def sft_handoff(checkpoint: str, *, start_step: int, parent_dp_racks: int | None) -> SftHandoff:
    """The handoff from a permanent checkpoint of a run of the epoch that starts at ``start_step``.

    The checkpoint's recorded mixture position sets where the handoff resumes. A checkpoint that records none
    comes from a run that read the epoch from ``start_step`` on ``parent_dp_racks`` racks.
    """
    metadata = permanent_checkpoint_metadata(checkpoint)
    step = int(metadata["step"])
    rack_batch_size = hero_rack_batch_size(HERO_MODEL_CONFIG.max_seq_len)
    position = SftMixturePosition.recorded_in(metadata)
    if position is None:
        if parent_dp_racks is None:
            raise ValueError(f"{checkpoint} records no SFT mixture position; give the racks of the run that wrote it")
        position = SftMixturePosition(
            plan_dp_racks=parent_dp_racks, step=start_step, sequence=0, batch_size=rack_batch_size * parent_dp_racks
        )
    elif parent_dp_racks is not None and position.batch_size != rack_batch_size * parent_dp_racks:
        raise ValueError(f"{checkpoint} records batch {position.batch_size}, not {parent_dp_racks} racks")
    return SftHandoff(checkpoint, step, position.plan_dp_racks, position.sequence_at(step))


def build_sft_run(
    *,
    run_id: str,
    initialize_from: str,
    start_step: int,
    muon_lr: float,
    adam_lr: float,
    max_grad_norm: float | None,
    replay_fraction: float,
    dp_racks: int,
    router_bias_mode: RouterBiasMode,
    capacity_factor: float,
    stop_after_steps: int | None = None,
    eval_every: int = 0,
    checkpoint_root: str | None = None,
    expert_placement: ExpertPlacement | None = None,
    handoff: SftHandoff | None = None,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build the hero SFT run that continues ``initialize_from`` at ``start_step``.

    ``stop_after_steps`` bounds a smoke test to that many updates of the full one-epoch schedule.
    ``checkpoint_root`` puts every checkpoint, permanent ones included, below one lifecycle-managed prefix.
    ``eval_every`` counts absolute steps, so the first eval lands on the first multiple after ``start_step``;
    the trainer's first-step eval never fires on a restored run. ``capacity_factor`` replaces the hero's routed
    expert capacity; with a frozen bias, SFT data overflows the hero's 1.5. ``expert_placement`` renumbers the
    routed experts of ``initialize_from`` so each capacity block carries a similar load. ``handoff`` continues
    another run of this epoch from its checkpoint instead of starting at ``initialize_from``; ``stop_after_steps``
    then counts from the handoff.
    """
    if dp_racks <= 0:
        raise ValueError(f"dp_racks must be positive, got {dp_racks}")
    if muon_lr <= 0 or adam_lr <= 0:
        raise ValueError("learning rates must be positive")
    if expert_placement is not None and not expert_placement.permutes(initialize_from):
        raise ValueError(f"the expert placement permutes {expert_placement.source_checkpoint}, not {initialize_from}")
    if capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
    model = with_transport_remat_mode(dataclasses.replace(HERO_MODEL_CONFIG, capacity_factor=capacity_factor))
    context_axis_size = HERO_CONTEXT_AXIS_SIZE
    expert_axis_size = hero_expert_axis_size(context_axis_size)
    if expert_placement is not None:
        # Fail before scheduling racks; the trainer checks again against the mesh it builds.
        expert_placement.validate_for(
            num_layers=model.num_layers, num_experts=model.num_experts, expert_axis_size=expert_axis_size
        )
    rack_batch_size = hero_rack_batch_size(model.max_seq_len)
    batch_size = rack_batch_size * dp_racks
    stores = load_sft_stores()
    plan_racks = dp_racks if handoff is None else handoff.plan_dp_racks
    plan = plan_sft_mixture(stores, batch_size=rack_batch_size * plan_racks, replay_fraction=replay_fraction)
    steps = sft_steps(plan, start_step=start_step, batch_size=batch_size, handoff=handoff)
    position = SftMixturePosition(
        plan_dp_racks=plan_racks, step=steps.first_step, sequence=steps.first_sequence, batch_size=batch_size
    )
    num_train_steps = steps.num_train_steps
    stop_step = num_train_steps if stop_after_steps is None else steps.first_step + stop_after_steps
    if stop_step > num_train_steps:
        raise ValueError(f"stop_after_steps={stop_after_steps} is past the epoch's last step {num_train_steps}")

    # The restored optimizer state comes from the hero's pretraining recipe, which fixes beta2 and epsilon.
    _, pretrain_optimizer = build_hero_configs(
        num_train_steps=HERO_SCHEDULE_STEPS,
        batch_size=hero_rack_batch_size(model.max_seq_len) * LADDER_RACKS["d6144"],
    )
    optimizer = dataclasses.replace(
        pretrain_optimizer,
        learning_rate=muon_lr,
        adam_lr=adam_lr,
        warmup=0,
        decay=steps.decay_steps,
        min_lr_ratio=SFT_MIN_LR_RATIO,
        weight_decay=0.0,
        gate_router_weight_decay=0.0,
        schedule_extension=None,
    )
    batch_axes = validated_batch_axis_size(
        device_count=HERO_RACK_GPUS * dp_racks,
        dp_racks=dp_racks,
        batch_size=batch_size,
        context_axis_size=context_axis_size,
        expert_axis_size=expert_axis_size,
    )
    validate_model_split(model, context_axis_size=context_axis_size, expert_axis_size=expert_axis_size)
    grug_trainer = dataclasses.replace(
        hero_grug_trainer_config(
            replica_axis_size=dp_racks,
            training_data_mode=TrainingDataMode.MIXTURE,
            watch_mode=WatchMode.INLINE,
            save_checkpoints=True,
            max_grad_norm=max_grad_norm,
            context_axis_size=context_axis_size,
            expert_axis_size=expert_axis_size,
            balance_attention_rows=HERO_BALANCE_ATTENTION_ROWS,
            gc_interval=100,
        ),
        router_bias_mode=router_bias_mode,
        special_tokens=special_token_init(start_step),
        expert_placement=expert_placement,
        checkpoint_metadata=position.checkpoint_metadata(),
    )
    train_resources = ResourceConfig.with_gpu(
        "GB200",
        count=HERO_GPUS_PER_NODE,
        cpu=HERO_NODE_CPU,
        ram=HERO_NODE_RAM,
        disk=HERO_NODE_DISK,
        replicas=HERO_EP_NODES * dp_racks,
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    validation = validation_datasets() if eval_every > 0 else []
    length_statistics = harrier_length_statistics()
    logger.info(
        "Hero SFT %s: steps %d to %d of %d sequences from mixture sequence %d, %d blocks of %d, replay %.4f, "
        "decay over the last %d updates, router bias %s, capacity factor %g",
        run_id,
        steps.first_step,
        num_train_steps,
        batch_size,
        steps.first_sequence,
        plan.num_blocks,
        plan.block_size,
        plan.replay_fraction,
        steps.decay_steps,
        router_bias_mode.value,
        capacity_factor,
    )

    def build_config(ctx: StepContext) -> GrugRunConfig:
        # A placed run keeps its checkpoints under its placement's fingerprint, so a relaunch with another
        # placement, or none, starts again from the source checkpoint; the trainer would refuse to resume them.
        output_path = ctx.output_path if expert_placement is None else _placement_path(ctx.output_path, expert_placement)
        if checkpoint_root is None:
            permanent_path = prefix_join(output_path, "checkpoints")
            temporary_path = temporary_checkpoint_base_path(
                output_path, ttl_days=RESUME_CHECKPOINT_TTL_DAYS, temp_prefix=HERO_RESUME_CHECKPOINT_PREFIX
            )
            resume_paths = checkpoint_resume_paths(
                output_path, ttl_days=RESUME_CHECKPOINT_TTL_DAYS, temp_prefix=HERO_RESUME_CHECKPOINT_PREFIX
            )
        else:
            root = checkpoint_root if expert_placement is None else _placement_path(checkpoint_root, expert_placement)
            permanent_path = prefix_join(root, "checkpoints")
            temporary_path = prefix_join(root, "resume")
            resume_paths = [temporary_path, permanent_path]
        trainer = hero_trainer_config(
            run_id=run_id,
            seed=0,
            train_batch_size=batch_size,
            num_train_steps=num_train_steps,
            profiler=ProfilerConfig(enabled=False),
            tracker=WandbConfig(
                save_code=False,
                entity="marin-community",
                project=SFT_WANDB_PROJECT,
                tags=[
                    "grug",
                    "moe",
                    "hero",
                    "sft",
                    "full-ab-2026.09.29",
                    f"racks-{dp_racks}",
                    f"router-bias-{router_bias_mode.value}",
                    f"capacity-{capacity_factor:g}",
                    *([] if expert_placement is None else [_placement_label(expert_placement)]),
                    *([] if handoff is None else [f"handoff-step-{handoff.step}"]),
                    "gb200",
                ],
                group="hero-sft",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=HERO_WATCH_INTERVAL),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=HERO_STEP_TIMEOUT,
                process_timeout=HERO_PROCESS_STALL_TIMEOUT,
                startup_timeout=HERO_STARTUP_TIMEOUT,
            ),
            # The run's own checkpoints come first, so a retry resumes the SFT; the base or the handoff starts it.
            load_checkpoint_path=[*resume_paths, initialize_from if handoff is None else handoff.checkpoint],
            load_checkpoint=True,
            checkpointer=CheckpointerConfig(
                base_path=permanent_path,
                temporary_base_path=temporary_path,
                save_interval=RESUME_SAVE_INTERVAL,
                keep=None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
            context_axis_size=context_axis_size,
        )
        replay = harrier_long_context_data_config(
            ctx=ctx,
            start_step=0,
            max_seq_len=model.max_seq_len,
            validation=[],
            stats_path=ctx.artifact_path(length_statistics),
        )
        # A handoff reads the parent's mixture unchanged; `mixture_continuation` moves where this run reads it.
        data = sft_data_config(
            stores, plan, start_step=start_step, replay=replay, validation=validation_components(ctx, validation)
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=HERO_TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(grug_trainer, trainer=trainer),
            eval=(
                GrugEvalConfig(
                    steps_per_eval=eval_every,
                    eval_batch_size=batch_axes,
                    eval_current=False,
                    eval_ema=False,
                    compute_bpb=True,
                    dropless_eval=True,
                )
                if validation
                else None
            ),
            stop_after_steps=stop_step,
            processes_per_task=HERO_PROCESSES_PER_TASK,
            mixture_continuation=(
                None
                if handoff is None
                else MixtureContinuation(
                    batch_size=plan.batch_size, step=steps.first_step, sequence=steps.first_sequence
                )
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=HeroThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(HARRIER_MIX_2026_08_18_STORE, length_statistics, *validation),
        runtime_args={"train_resources": train_resources},
    )


def _placement_label(placement: ExpertPlacement) -> str:
    """Names a placed run's checkpoint directory and W&B tag."""
    return f"expert-placement-{placement.fingerprint}"


def _placement_path(base: str, placement: ExpertPlacement) -> str:
    return prefix_join(base, _placement_label(placement))


def _grad_norm(value: str) -> float | None:
    if value == "none":
        return None
    norm = float(value)
    if norm <= 0:
        raise click.BadParameter("must be positive or 'none'")
    return norm


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option("--initialize-from-checkpoint", "initialize_from", required=True, help="Complete pretraining checkpoint.")
@click.option("--muon-lr", type=click.FloatRange(min=0, min_open=True), required=True, help="Peak MuonH/AdamH rate.")
@click.option("--adam-lr", type=click.FloatRange(min=0, min_open=True), required=True, help="Peak Adam rate.")
@click.option("--max-grad-norm", "max_grad_norm", required=True, help="Global gradient-norm clip, or 'none'.")
@click.option(
    "--replay-fraction",
    type=click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    required=True,
    help="Target share of each block drawn from replay; the block count rounds it.",
)
@click.option("--dp-racks", type=click.IntRange(min=1), required=True, help="Data-parallel NVL72 racks.")
@click.option(
    "--stop-after-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Stop after this many updates of the one-epoch schedule (smoke tests). Unset runs the epoch.",
)
@click.option(
    "--router-bias",
    type=click.Choice([mode.value for mode in RouterBiasMode]),
    default=RouterBiasMode.FROZEN.value,
    show_default=True,
    help="'frozen' keeps the restored bias (the 10.05 recipe); 'update' keeps quantile balancing on.",
)
@click.option(
    "--capacity-factor",
    type=click.FloatRange(min=0, min_open=True),
    default=SFT_CAPACITY_FACTOR,
    show_default=True,
    help=f"Routed expert capacity per shard as a multiple of the even share; the hero trained at "
    f"{HERO_MODEL_CONFIG.capacity_factor:g}. The default is the highest that fits in HBM.",
)
@click.option(
    "--eval-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Dropless eval at every multiple of this absolute step; 0 disables evaluation.",
)
@click.option(
    "--checkpoint-root",
    default=None,
    help="Put all checkpoints below this prefix, e.g. a tmp/ttl=3d path for a smoke test.",
)
@click.option(
    "--expert-placement",
    "expert_placement_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Expert placement JSON from `expert_placement.py`; renumbers the start checkpoint's routed experts.",
)
@click.option(
    "--handoff-checkpoint",
    default=None,
    help="Permanent checkpoint of a run of this epoch to continue, at --dp-racks, from its step. Pass that "
    "run's other options unchanged.",
)
@click.option(
    "--handoff-dp-racks",
    type=click.IntRange(min=1),
    default=None,
    help="Data-parallel racks of the run that wrote --handoff-checkpoint. Required only when the checkpoint "
    "records no SFT mixture position.",
)
@build_options
def main(
    run_id: str,
    initialize_from: str,
    muon_lr: float,
    adam_lr: float,
    max_grad_norm: str,
    replay_fraction: float,
    dp_racks: int,
    stop_after_steps: int | None,
    router_bias: str,
    capacity_factor: float,
    eval_every: int,
    checkpoint_root: str | None,
    expert_placement_path: Path | None,
    handoff_checkpoint: str | None,
    handoff_dp_racks: int | None,
) -> ArtifactStep[HeroThroughputResult]:
    if handoff_checkpoint is None and handoff_dp_racks is not None:
        raise click.UsageError("--handoff-dp-racks needs --handoff-checkpoint")
    start_step = checkpoint_step(initialize_from)
    handoff = (
        None
        if handoff_checkpoint is None
        else sft_handoff(handoff_checkpoint, start_step=start_step, parent_dp_racks=handoff_dp_racks)
    )
    return build_sft_run(
        run_id=run_id,
        initialize_from=initialize_from,
        start_step=start_step,
        muon_lr=muon_lr,
        adam_lr=adam_lr,
        max_grad_norm=_grad_norm(max_grad_norm),
        replay_fraction=replay_fraction,
        dp_racks=dp_racks,
        router_bias_mode=RouterBiasMode(router_bias),
        capacity_factor=capacity_factor,
        stop_after_steps=stop_after_steps,
        eval_every=eval_every,
        checkpoint_root=checkpoint_root,
        expert_placement=(
            None if expert_placement_path is None else ExpertPlacement.from_json(expert_placement_path.read_text())
        ),
        handoff=handoff,
    )


if __name__ == "__main__":
    main()
