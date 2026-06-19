# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data import AsyncDataset
from levanter.data.text import (
    BlockShuffleConfig,
    DirectDatasetComponent,
    GrugLmExample,
    LmDataConfig,
    TextLmDatasetFormat,
)
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_data_config
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
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
    watch: WatchConfig = field(default_factory=WatchConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    trainer_mesh_expert_axis_size: int | None = None
    """Expert axis size for TrainerConfig's auxiliary mesh validation.

    Grug training builds its actual compact mesh from ``grug_trainer``. This
    override is only for the Levanter TrainerConfig mesh initialized before the
    raw Grug mesh exists.
    """
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    checkpointing_enabled: bool = True
    """False disables checkpoint restore, periodic saves, and final forced saves.
    Use this for disposable throughput probes where checkpoint I/O would dominate
    the measured run tail."""
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""
    log_jaxprs: bool = True
    log_xla_hlo: bool = True


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def env_bool(key: str, default: bool) -> bool:
    """Read a boolean from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def validate_local_expert_model_axes(
    *,
    expert_axis: int,
    model_axis: int,
    local_device_count: int,
    env_prefix: str,
    allow_cross_node_expert_axis: bool = False,
) -> None:
    """Validate that expert/model axes fit cleanly inside one worker."""
    if expert_axis <= 0:
        raise ValueError(f"{env_prefix}_EXPERT_AXIS must be positive, got {expert_axis}")
    if model_axis <= 0:
        raise ValueError(f"{env_prefix}_MODEL_AXIS must be positive, got {model_axis}")
    if local_device_count <= 0:
        raise ValueError(f"local_device_count must be positive, got {local_device_count}")

    local_axis_product = expert_axis * model_axis
    if allow_cross_node_expert_axis:
        return
    if local_device_count % local_axis_product != 0:
        raise ValueError(
            f"{env_prefix}_EXPERT_AXIS * {env_prefix}_MODEL_AXIS must divide the "
            f"{local_device_count} GPUs on each worker so expert/model groups stay local; "
            f"got {expert_axis} * {model_axis} = {local_axis_product}."
        )


def trainer_mesh_expert_axis_size(
    *,
    expert_axis: int,
    model_axis: int,
    local_device_count: int,
    allow_cross_node_expert_axis: bool = False,
) -> int:
    """Return the locally valid expert axis for TrainerConfig's auxiliary mesh."""
    validate_local_expert_model_axes(
        expert_axis=expert_axis,
        model_axis=model_axis,
        local_device_count=local_device_count,
        env_prefix="trainer_mesh",
        allow_cross_node_expert_axis=allow_cross_node_expert_axis,
    )
    if not allow_cross_node_expert_axis or expert_axis * model_axis <= local_device_count:
        return expert_axis

    if local_device_count % model_axis != 0:
        raise ValueError(
            f"model_axis ({model_axis}) must divide local_device_count ({local_device_count}) "
            "when expert_axis spans nodes."
        )
    local_expert_axis = local_device_count // model_axis
    if expert_axis % local_expert_axis != 0:
        raise ValueError(
            f"expert_axis ({expert_axis}) must be divisible by local expert axis "
            f"({local_expert_axis}) when expert_axis spans nodes."
        )
    return local_expert_axis


def validate_ring_expert_model_axes(
    *,
    expert_axis: int,
    model_axis: int,
    moe_implementation: str | None,
    env_prefix: str,
) -> None:
    """Reject ring EP/model-axis combinations that fail on CW H100s."""
    implementation = moe_implementation or "ring"
    if implementation == "ring" and expert_axis > 1 and model_axis > 1:
        raise ValueError(
            f"{env_prefix}_MOE_IMPLEMENTATION=ring currently requires either "
            f"{env_prefix}_EXPERT_AXIS=1 or {env_prefix}_MODEL_AXIS=1 on CoreWeave H100s; "
            f"got {env_prefix}_EXPERT_AXIS={expert_axis}, {env_prefix}_MODEL_AXIS={model_axis}. "
            "Use model_axis>1 only for attention/model-axis diagnostics with expert_axis=1, "
            "or keep model_axis=1 for ring expert-parallel runs."
        )


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


@dataclass(frozen=True)
class SyntheticGrugDataset(AsyncDataset[GrugLmExample]):
    """Deterministic in-memory token stream for distributed systems probes."""

    seq_len: int
    vocab_size: int
    num_examples: int
    eos_id: int | None = None
    eos_interval: int = 0
    block_cross_document_attention: bool = True

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1, got {self.vocab_size}")
        if self.num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {self.num_examples}")
        if self.eos_interval < 0:
            raise ValueError(f"eos_interval must be non-negative, got {self.eos_interval}")
        if self.eos_interval > 0 and self.eos_id is None:
            raise ValueError("eos_id must be set when eos_interval is positive")

        # Runtime caches are intentionally not dataclass fields. Marin executor
        # versions configs via dataclasses.replace, which cannot pass init=False fields.
        object.__setattr__(self, "_positions", np.arange(self.seq_len, dtype=np.int64))
        loss_weight = (np.arange(self.seq_len) < (self.seq_len - 1)).astype(np.float32)
        object.__setattr__(self, "_loss_weight", loss_weight)
        object.__setattr__(self, "_attn_mask", GrugAttentionMask.causal())

    async def async_len(self) -> int:
        return self.num_examples

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        if not indices:
            return []

        tokens = self._tokens_for_indices(indices)
        return [self._example_from_tokens(row) for row in tokens]

    def _tokens_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        positions = cast(np.ndarray, self.__dict__["_positions"])
        offsets = np.asarray(indices, dtype=np.int64)[:, None] * 9973
        tokens = (positions[None, :] + offsets) % self.vocab_size
        if self.eos_interval > 0:
            tokens[:, self.eos_interval - 1 :: self.eos_interval] = self.eos_id

        return tokens.astype(np.int32, copy=False)

    def _example_from_tokens(self, tokens: np.ndarray) -> GrugLmExample:
        loss_weight = cast(np.ndarray, self.__dict__["_loss_weight"])
        attn_mask = cast(GrugAttentionMask, self.__dict__["_attn_mask"])
        token_array = jnp.asarray(tokens, dtype=jnp.int32)
        loss_weight_array = jnp.asarray(loss_weight)
        if self.eos_interval > 0 and self.block_cross_document_attention:
            assert self.eos_id is not None
            eos_mask = np.roll(tokens, 1) == self.eos_id
            eos_mask[0] = False
            segment_ids = jnp.asarray(np.cumsum(eos_mask, dtype=np.int32))
            attn_mask = attn_mask.with_segment_ids(segment_ids)

        return GrugLmExample(tokens=token_array, loss_weight=loss_weight_array, attn_mask=attn_mask)


def synthetic_grug_data(
    *,
    seq_len: int,
    vocab_size: int,
    num_examples: int,
    eos_id: int | None = None,
    eos_interval: int = 0,
    block_cross_document_attention: bool = True,
) -> LmDataConfig:
    dataset = SyntheticGrugDataset(
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_examples=num_examples,
        eos_id=eos_id,
        eos_interval=eos_interval,
        block_cross_document_attention=block_cross_document_attention,
    )
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=False,
        block_cross_document_attention=block_cross_document_attention,
        components={"synthetic": DirectDatasetComponent(datasets={"train": dataset, "validation": dataset})},
        train_weights={"synthetic": 1.0},
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


class DisabledCheckpointerConfig(CheckpointerConfig):
    """TrainerConfig-compatible checkpoint config that never creates a checkpointer."""

    def create(self, run_id):
        del run_id
        return None


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    checkpointer = config.checkpointer or CheckpointerConfig(
        base_path=os.path.join(config.output_path, "checkpoints"),
        temporary_base_path=temporary_checkpoint_base_path(config.output_path),
        append_run_id_to_base_path=False,
        save_interval=timedelta(minutes=10),
        keep=None,
    )
    load_checkpoint = None
    if not config.checkpointing_enabled:
        checkpointer = DisabledCheckpointerConfig(base_path="/tmp/grug-disabled-checkpoints")
        load_checkpoint = False

    trainer_mesh_expert_axis = config.trainer_mesh_expert_axis_size or config.grug_trainer.expert_axis_size
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        mesh=MeshConfig(
            axes={
                "data": -1,
                "expert": trainer_mesh_expert_axis,
                "model": config.grug_trainer.model_axis_size,
            },
            dcn_axes={"replica_dcn": -1},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
        ),
        profiler=config.profiler,
        watch=config.watch,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        load_checkpoint=load_checkpoint,
        checkpointer=checkpointer,
        log_jaxprs=config.log_jaxprs,
        log_xla_hlo=config.log_xla_hlo,
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


RESOLVED_RUN_ID = _resolve_run_id("4_10_test_moe")


# Baseline: 1e18 compute budget, d1024. Model + optimizer + batch + steps are
# all derived from `MoeAdamHHeuristic`. To override any of these, swap in
# an explicit `GrugModelConfig` / `GrugMoeAdamHConfig` below.
_BASELINE_BUDGET: float = 1e18
_BASELINE_HIDDEN_DIM: int = 1024
_BASELINE_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept
# because consumers (e.g. experiments/ferries/canary_ferry.py) import it by
# name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model


baseline_moe = ExecutorStep(
    name="grug/4_10_baseline_moe",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(_baseline_model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/moe-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(_baseline_optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
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


if __name__ == "__main__":
    executor_main(
        steps=[baseline_moe],
        description="Baseline grug MoE (QB+GN+XSA+zloss) on Nemotron mix.",
    )
