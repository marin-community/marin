# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detached exact-state gradient probes and short rollouts for StarCoder WSD80."""

import argparse
import asyncio
import csv
import functools
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter.trainer
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import NamedLmDataset
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.optim.util import flatten_linear_layers
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer
from levanter.utils.thread_utils import blocking_wait
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import remote
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_probe_20260816 as freeze,
)

SCHEMA_VERSION = "2026-08-16-gradient-probe-v6"
ARTIFACT_VERSION = "2026.08.16.6"
RELEASE_PATH = freeze.OUTPUT_DIR / "release.json"
STARCODER_COMPONENT = "dolma/starcoder"
TARGET_DISTRIBUTIONS = frozenset(freeze.TARGET_COMPONENTS)
LEAF_DISTRIBUTIONS = {
    f"nemotron_{name}": f"nemotron_cc/{name}-llama3"
    for name in ("hq_actual", "hq_synth", "medium_high", "medium", "medium_low", "low_actual")
}
SOURCE_DISTRIBUTIONS = frozenset(
    {
        "starcoder_on_policy",
        "starcoder_support_reference",
        "starcoder_excluded_global",
        "nemotron_aggregate",
        *LEAF_DISTRIBUTIONS,
    }
)
PROBE_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 128
MIXTURE_BLOCK_SIZE = 2_048
FULL_SCOPE_MAX_CONCURRENT = 64
CANARY_SCOPE_MAX_CONCURRENT = 14
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_FROZEN_FULL_GRADIENT_PROBE"
FULL_LAUNCH_AUTHORIZATION_PATH = freeze.OUTPUT_DIR / "full_launch_authorization.json"
TPU_TYPE = "v5p-8"
TPU_REGION = "us-central1"
TPU_ZONE = "us-central1-a"
TPU_HOST_CPU = 16
TPU_HOST_RAM = "128g"
NUMERICAL_TOLERANCE = 1e-6
LAYER_PATTERN = re.compile(r"(?:layers|Layer)[.\[](?P<layer>\d+)")


@dataclass(frozen=True)
class ProbeGroupConfig:
    scope: str
    group_id: str
    checkpoint_uri: str
    checkpoint_step: int
    expected_restored_state_step: int
    rows: tuple[dict[str, Any], ...]
    pod_config: TrainLmOnPodConfig
    output_path: str
    cache_provenance_sha256: str
    release_sha256: str


@dataclass(frozen=True)
class OptimizerGroupConfig:
    scope: str
    group_id: str
    checkpoint_uri: str
    checkpoint_step: int
    expected_restored_state_step: int
    rows: tuple[dict[str, Any], ...]
    pod_config: TrainLmOnPodConfig
    output_path: str
    cache_provenance_sha256: str
    release_sha256: str
    target_block_counts: tuple[tuple[str, int], ...]
    target_sequence_set_ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RolloutGroupConfig:
    scope: str
    group_id: str
    checkpoint_uri: str
    checkpoint_step: int
    expected_restored_state_step: int
    rows: tuple[dict[str, Any], ...]
    pod_config: TrainLmOnPodConfig
    output_path: str
    cache_provenance_sha256: str
    release_sha256: str
    target_block_counts: tuple[tuple[str, int], ...]
    target_sequence_set_ids: tuple[tuple[str, str], ...]


class ShiftedRestartDataset(AsyncDataset[Any]):
    """A deterministic cyclic view beginning at an exact logical source offset."""

    def __init__(self, dataset: AsyncDataset[Any], *, start: int, length: int):
        if length <= 0:
            raise ValueError("ShiftedRestartDataset requires a positive source length")
        self.dataset = dataset
        self.start = start % length
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[Any]:
        return await self.dataset.get_batch([(self.start + int(index)) % self.length for index in indices])


def _canonical_json(value: Any) -> str:
    return freeze.canonical_json(value)


def _stable_seed_words(value: str) -> tuple[int, int]:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big")


def _stable_key(value: str) -> Any:
    first, second = _stable_seed_words(value)
    return jrandom.fold_in(jrandom.PRNGKey(first), second)


def _fold_in_stable(key: Any, value: str) -> Any:
    first, second = _stable_seed_words(value)
    return jrandom.fold_in(jrandom.fold_in(key, first), second)


def _path_join(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))


def _row_path(output_path: str, row_id: str) -> str:
    return _path_join(output_path, "rows", f"{row_id}.json")


def _write_create_only_json(path: str, payload: dict[str, Any], *, identity_sha256: str) -> str:
    document = {
        **payload,
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
    }
    encoded = (_canonical_json(document) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
        disposition = "created"
    except FileExistsError as error:
        with fs.open(plain_path, "rb") as handle:
            existing = json.load(handle)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Existing output has a different schema: {path}") from error
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(f"Existing output is claimed by another row identity: {path}") from error
        disposition = "skipped_existing"
    with fs.open(plain_path, "rb") as handle:
        persisted = handle.read()
    if disposition == "created" and persisted != encoded:
        raise RuntimeError(f"Create-only output did not persist exactly: {path}")
    return disposition


def _read_existing_identity(path: str) -> str | None:
    document = _read_output_document(path)
    if document is None:
        return None
    identity = document.get("identity_sha256")
    if not isinstance(identity, str):
        raise RuntimeError(f"Existing row has no identity: {path}")
    return identity


def _read_output_document(path: str) -> dict[str, Any] | None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    if not fs.exists(plain_path):
        return None
    with fs.open(plain_path, "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing row has an unexpected schema: {path}")
    return document


def _row_identity(row: Mapping[str, Any], release_sha256: str) -> str:
    return freeze.canonical_sha256(
        {
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "checkpoint_uri": row["checkpoint_uri"],
            "train_config_sha256": row["train_config_sha256"],
            "release_sha256": release_sha256,
            "release_version": freeze.PROBE_RELEASE_VERSION,
        }
    )


def _assert_existing_row_identity(output_path: str, row: Mapping[str, Any], release_sha256: str) -> bool:
    existing = _read_existing_identity(_row_path(output_path, str(row["row_id"])))
    if existing is None:
        return False
    expected = _row_identity(row, release_sha256)
    if existing != expected:
        raise RuntimeError(f"Completed row identity drifted: {row['row_id']}")
    return True


def _group_identity(group_id: str, rows: Sequence[Mapping[str, Any]], release_sha256: str) -> str:
    return freeze.canonical_sha256(
        {
            "group_id": group_id,
            "rows": sorted(row["row_id"] for row in rows),
            "release_sha256": release_sha256,
        }
    )


def _assert_existing_group_complete(config: ProbeGroupConfig | OptimizerGroupConfig | RolloutGroupConfig) -> bool:
    marker = _read_output_document(_path_join(config.output_path, "group_complete.json"))
    if marker is None:
        return False
    expected = _group_identity(config.group_id, config.rows, config.release_sha256)
    if marker.get("identity_sha256") != expected or marker.get("release_sha256") != config.release_sha256:
        raise RuntimeError(f"Completed group identity drifted: {config.group_id}")
    if marker.get("row_count") != len(config.rows):
        raise RuntimeError(f"Completed group row count drifted: {config.group_id}")
    for row in config.rows:
        if not _assert_existing_row_identity(config.output_path, row, config.release_sha256):
            raise RuntimeError(f"Completed group is missing row {row['row_id']}")
    return True


def _write_group_complete(
    config: ProbeGroupConfig | OptimizerGroupConfig | RolloutGroupConfig,
    *,
    kind: str,
    checkpoint_metadata: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
) -> None:
    _write_create_only_json(
        _path_join(config.output_path, "group_complete.json"),
        {
            "kind": f"{kind}_group",
            "scope": config.scope,
            "group_id": config.group_id,
            "row_count": len(config.rows),
            "checkpoint_metadata": checkpoint_metadata,
            "runtime_summary": runtime_summary,
            "cache_provenance_sha256": config.cache_provenance_sha256,
            "release_sha256": config.release_sha256,
            "endpoint_metrics_read": False,
        },
        identity_sha256=_group_identity(config.group_id, config.rows, config.release_sha256),
    )


def _verify_group_contract(config: ProbeGroupConfig | OptimizerGroupConfig | RolloutGroupConfig) -> None:
    if not config.output_path.startswith(f"{freeze.RESULT_ROOT}/{config.scope}/"):
        raise ValueError(f"Output path is outside the frozen central1 release root: {config.output_path}")
    config_sha256 = freeze._config_identity(config.pod_config)["full_train_config_sha256"]
    if not config.rows:
        raise ValueError(f"Execution group has no rows: {config.group_id}")
    trajectory_id = str(config.rows[0].get("trajectory_id", config.rows[0].get("parent_trajectory_id")))
    support = _starcoder_support_contract(cast(TrainLmConfig, config.pod_config.train_config))
    if f"_{support['support_id']}_" not in trajectory_id:
        raise ValueError(f"Trajectory {trajectory_id} does not match runtime support {support['support_id']}")
    for row in config.rows:
        if row["group_id"] != config.group_id:
            raise ValueError(f"Row {row['row_id']} belongs to a different group")
        if row["checkpoint_uri"] != config.checkpoint_uri:
            raise ValueError(f"Row {row['row_id']} points to a different checkpoint")
        row_step = int(row.get("checkpoint_step", row.get("parent_checkpoint_step")))
        if row_step != config.checkpoint_step:
            raise ValueError(f"Row {row['row_id']} points to checkpoint step {row_step}")
        if int(row["expected_restored_state_step"]) != config.expected_restored_state_step:
            raise ValueError(f"Row {row['row_id']} expects a different restored state step")
        if config.expected_restored_state_step != freeze.expected_restored_state_step(config.checkpoint_step):
            raise ValueError(f"Group {config.group_id} has inconsistent checkpoint-step semantics")
        if row["train_config_sha256"] != config_sha256:
            raise ValueError(f"Train configuration drifted for row {row['row_id']}")
        if isinstance(config, ProbeGroupConfig):
            if int(row["sequences_per_block"]) != PROBE_BATCH_SIZE:
                raise ValueError(f"Probe batch size drifted for row {row['row_id']}")
            if int(row["training_sequences_per_update"]) != TRAIN_BATCH_SIZE:
                raise ValueError(f"Optimizer batch size drifted for row {row['row_id']}")
        else:
            frozen_counts = tuple(sorted(json.loads(row["target_block_counts_json"]).items()))
            if frozen_counts != config.target_block_counts:
                raise ValueError(f"Target block counts drifted for row {row['row_id']}")
            frozen_sequence_sets = tuple(sorted(json.loads(row["target_sequence_set_ids_json"]).items()))
            if frozen_sequence_sets != config.target_sequence_set_ids:
                raise ValueError(f"Target sequence-set identities drifted for row {row['row_id']}")
        if isinstance(config, RolloutGroupConfig):
            readouts = [int(value) for value in str(row["readout_steps"]).split("|")]
            if not readouts or max(readouts) > int(row["updates"]):
                raise ValueError(f"Rollout readout exceeds the update horizon for row {row['row_id']}")


def _restored_optimizer_summary(
    state: Any,
    checkpoint_label_step: int,
    expected_restored_state_step: int,
    *,
    allow_partial_checkpoint: bool,
) -> dict[str, Any]:
    counters: dict[str, int] = {}
    leaves, _ = jax.tree_util.tree_flatten_with_path(state.opt_state)
    for path, value in leaves:
        array = _array(value)
        if not eqx.is_array(array) or array.ndim != 0 or not jnp.issubdtype(array.dtype, jnp.integer):
            continue
        name = jax.tree_util.keystr(path)
        if "count" in name.lower():
            counters[name] = int(jax.device_get(array))
    if not counters:
        raise RuntimeError("Restored optimizer state exposes no step counters")
    if expected_restored_state_step != freeze.expected_restored_state_step(checkpoint_label_step):
        raise RuntimeError("Frozen checkpoint label and restored-state expectation are inconsistent")
    state_step_matches_expected = int(state.step) == expected_restored_state_step
    optimizer_counter_matches_expected = expected_restored_state_step in counters.values()
    if not state_step_matches_expected:
        raise RuntimeError(
            f"Restored trainer state step is {int(state.step)}, expected next-step counter "
            f"{expected_restored_state_step} after checkpoint label {checkpoint_label_step}"
        )
    if not optimizer_counter_matches_expected:
        raise RuntimeError(
            f"Optimizer counters do not contain expected restored state step {expected_restored_state_step}: {counters}"
        )
    if allow_partial_checkpoint:
        raise RuntimeError("Gradient probes cannot validate a partial checkpoint restore")
    return {
        "checkpoint_label_step": checkpoint_label_step,
        "expected_restored_state_step": expected_restored_state_step,
        "trainer_state_step": int(state.step),
        "trainer_state_step_matches_expected": state_step_matches_expected,
        "optimizer_step_counters": counters,
        "optimizer_counter_matches_expected": optimizer_counter_matches_expected,
        "allow_partial_checkpoint": allow_partial_checkpoint,
    }


def _read_checkpoint_metadata(checkpoint_uri: str, expected_step: int) -> dict[str, Any]:
    if checkpoint_uri.rstrip("/").rsplit("/", 1)[-1] != f"step-{expected_step}":
        raise RuntimeError(f"Checkpoint URI label does not match expected step {expected_step}: {checkpoint_uri}")
    metadata_uri = _path_join(checkpoint_uri, "metadata.json")
    with fsspec.open(metadata_uri, "rb") as handle:
        metadata = json.load(handle)
    if int(metadata.get("step", -1)) != expected_step:
        raise RuntimeError(f"Checkpoint step mismatch at {checkpoint_uri}: {metadata}")
    if metadata.get("is_temporary") is not False:
        raise RuntimeError(f"Probe source is not a permanent checkpoint: {checkpoint_uri}")
    return metadata


def _prepare_train_config(pod_config: TrainLmOnPodConfig, checkpoint_uri: str, group_id: str) -> Any:
    train = cast(TrainLmConfig, pod_config.train_config)
    if train.trainer.allow_partial_checkpoint:
        raise ValueError("Gradient probes require an exact, non-partial trainer-state restore")
    if train.trainer.batch_schedule.batch_size_at_step(0) != TRAIN_BATCH_SIZE:
        raise ValueError(f"Expected an initial train batch size of {TRAIN_BATCH_SIZE}")
    trainer = replace(
        train.trainer,
        id=group_id,
        tracker=NoopConfig(),
        load_checkpoint=False,
        load_checkpoint_path=None,
        initialize_from=checkpoint_uri,
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    return replace(train, trainer=trainer)


def _effective_optimizer_schedule_num_train_steps(train_config: TrainLmConfig) -> int:
    configured = train_config.optimizer_schedule_num_train_steps
    effective = train_config.trainer.num_train_steps if configured is None else configured
    if effective < train_config.trainer.num_train_steps:
        raise ValueError(
            "optimizer_schedule_num_train_steps must be at least trainer.num_train_steps, got "
            f"{effective} < {train_config.trainer.num_train_steps}"
        )
    return effective


def _optimizer_schedule_summary(train_config: TrainLmConfig) -> dict[str, Any]:
    configured = train_config.optimizer_schedule_num_train_steps
    effective = _effective_optimizer_schedule_num_train_steps(train_config)
    return {
        "configured_num_train_steps": configured,
        "effective_num_train_steps": effective,
        "trainer_num_train_steps": train_config.trainer.num_train_steps,
        "matches_frozen_training_horizon": effective == train_config.trainer.num_train_steps,
    }


def _starcoder_support_contract(train_config: TrainLmConfig) -> dict[str, Any]:
    cap = train_config.data.max_train_batches
    start = train_config.data.max_train_batches_start
    seed = train_config.data.max_train_batches_subset_seed
    if cap is None:
        if start is not None or seed is not None:
            raise ValueError("Full StarCoder support cannot carry a finite-support offset or seed")
        return {
            "support_id": "full",
            "support_batches": None,
            "support_start_batches": None,
            "support_pool_seed": None,
            "permutation_type": train_config.data.permutation_type,
        }
    if set(cap) != {STARCODER_COMPONENT} or seed is None:
        raise ValueError(f"Invalid finite StarCoder support contract: cap={cap}, seed={seed}")
    if start is not None and set(start) != {STARCODER_COMPONENT}:
        raise ValueError(f"Invalid finite StarCoder support offset: {start}")
    if train_config.data.permutation_type != "feistel":
        raise ValueError(
            f"Finite StarCoder support requires feistel permutation, got {train_config.data.permutation_type}"
        )
    support_batches = int(cap[STARCODER_COMPONENT])
    support_start_batches = 0 if start is None else int(start[STARCODER_COMPONENT])
    if support_start_batches == 0:
        support_id = "m100a"
    elif support_start_batches == support_batches:
        support_id = "m100b"
    else:
        raise ValueError(f"Unknown finite StarCoder support offset: {support_start_batches}/{support_batches}")
    return {
        "support_id": support_id,
        "support_batches": support_batches,
        "support_start_batches": support_start_batches,
        "support_pool_seed": int(seed),
        "permutation_type": train_config.data.permutation_type,
    }


def _initialize_runtime(train_config: Any) -> tuple[Trainer, Any, Axis, Any, Any]:
    levanter.trainer.initialize(train_config)
    optimizer = train_config.optimizer.build(_effective_optimizer_schedule_num_train_steps(train_config))

    def loss_function(model: LmHeadModel, example: LmExample, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=train_config.z_loss_weight)

    trainer = Trainer(train_config.trainer, optimizer, loss_function, add_default_hooks=False)
    trainer.__enter__()
    seed = train_config.trainer.seed
    data_key, _, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    if train_config.data_seed is not None:
        data_key = jrandom.PRNGKey(train_config.data_seed)
    train_length = train_config.train_seq_len or train_config.model.max_seq_len
    Pos = train_config.model.max_Pos.resize(train_length)
    Vocab = round_axis_for_partitioning(
        Axis("vocab", len(train_config.data.the_tokenizer)), trainer.parameter_axis_mapping
    )
    state = trainer.initial_state(
        training_key,
        model_init=lambda: train_config.model.build(Vocab, key=model_key),
    )
    return trainer, state, Pos, data_key, train_config.optimizer.create_mask(state.model)


def _close_runtime(trainer: Trainer) -> None:
    trainer.__exit__(None, None, None)


def _logical_component_offset(mixture: MixtureDataset[Any], component: str, sequence_offset: int) -> int:
    try:
        dataset_id = list(mixture.dataset_index).index(component)
    except ValueError as error:
        raise ValueError(f"Mixture omits component {component}") from error
    complete_blocks, remainder = divmod(sequence_offset, mixture.block_size)
    count = 0
    for block_id in range(complete_blocks):
        stage = mixture._get_stage_for_block(block_id)
        count += int(mixture._counts_per_block_per_stage[stage][dataset_id])
    if remainder:
        block = mixture._get_block(complete_blocks)
        count += int(np.count_nonzero((block[:remainder] >> 16) == dataset_id))
    return count


def _source_views(
    train_config: Any, Pos: Axis, data_key: Any, restored_state_step: int
) -> tuple[dict[str, AsyncDataset[Any]], dict[str, Any]]:
    mix_key, shuffle_key = jrandom.split(data_key)
    train_sets = train_config.data.train_sets(
        Pos,
        key=shuffle_key,
        initial_batch_size=train_config.trainer.batch_schedule.batch_size_at_step(0),
    )
    weights = train_config.data.train_weights
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, train_config.trainer.batch_schedule)
    training_mixture = MixtureDataset(
        datasets=train_sets,
        weights=weights,
        stop_strategy=train_config.data.stop_strategy,
        key=mix_key,
        block_size=train_config.data.mixture_block_size,
    )
    sequence_offset = train_config.trainer.batch_schedule.global_data_offset_by_step(restored_state_step)
    offsets = {
        name: _logical_component_offset(training_mixture, name, sequence_offset) for name in freeze.TRAINING_COMPONENTS
    }
    source_sequence_counts = {name: len(dataset.as_sync_dataset()) for name, dataset in train_sets.items()}
    continued = {
        name: ShiftedRestartDataset(dataset, start=offsets[name], length=source_sequence_counts[name])
        for name, dataset in train_sets.items()
    }

    if not isinstance(train_config.data.train_weights, list):
        raise ValueError("Gradient probes require the frozen scheduled-mixture training configuration")
    first_weights = dict(train_config.data.train_weights[0][1])
    broad_weights = {name: weight for name, weight in first_weights.items() if name != STARCODER_COMPONENT}
    frozen_nemotron = MixtureDataset(
        datasets={name: train_sets[name] for name in broad_weights},
        weights=broad_weights,
        stop_strategy=train_config.data.stop_strategy,
        key=_stable_key("nemotron_aggregate"),
        block_size=MIXTURE_BLOCK_SIZE,
    )
    continued_nemotron = MixtureDataset(
        datasets={name: continued[name] for name in broad_weights},
        weights=broad_weights,
        stop_strategy=train_config.data.stop_strategy,
        key=_stable_key("nemotron_on_policy"),
        block_size=MIXTURE_BLOCK_SIZE,
    )
    views = {
        "starcoder_on_policy": continued[STARCODER_COMPONENT],
        "starcoder_support_reference": train_sets[STARCODER_COMPONENT],
        "nemotron_aggregate": frozen_nemotron,
        "nemotron_on_policy": continued_nemotron,
        **{distribution: train_sets[component] for distribution, component in LEAF_DISTRIBUTIONS.items()},
        **{f"continued:{name}": dataset for name, dataset in continued.items()},
    }
    return views, {
        "restored_state_step": restored_state_step,
        "global_sequence_offset": sequence_offset,
        "logical_component_offsets": offsets,
        "mixture_block_size": train_config.data.mixture_block_size,
        "step_schedule_rescaled_to_sequences": True,
        "on_policy_stream_rule": "continue_exact_per_source_logical_offset",
        "frozen_reference_stream_rule": "start_at_seeded_reference_origin",
        "source_sequence_counts": source_sequence_counts,
        "source_restart_semantics": "intentional_epoch_wrap_within_frozen_training_support",
        "starcoder_support": _starcoder_support_contract(train_config),
        "global_holdout": _global_holdout_contract(train_config),
    }


def _global_holdout_contract(train_config: Any) -> dict[str, Any]:
    if train_config.data.num_validation_sequences is not None:
        raise RuntimeError("Gradient probes require the original training configuration to have no validation split")
    holdout_sequences = train_config.data.train_holdout_sequences
    synthesized = holdout_sequences is None
    if holdout_sequences is None:
        if train_config.data.train_holdout_seed is not None or train_config.data.train_holdout_partition is not None:
            raise RuntimeError("Canary holdout provenance is only synthesized when all holdout fields are absent")
        holdout_sequences = {name: 4_096 for name in freeze.TRAINING_COMPONENTS}
        holdout_seed = freeze.CANARY_HOLDOUT_SEED
        holdout_partition = freeze.CANARY_HOLDOUT_PARTITION
    else:
        holdout_seed = train_config.data.train_holdout_seed
        holdout_partition = train_config.data.train_holdout_partition
        if holdout_seed != freeze.FULL_HOLDOUT_SEED:
            raise RuntimeError(f"Frozen full-panel holdout seed drifted: {holdout_seed}")
        if holdout_partition != freeze.FULL_HOLDOUT_PARTITION:
            raise RuntimeError(f"Frozen full-panel holdout partition drifted: {holdout_partition}")
    return {
        "synthesized": synthesized,
        "seed": int(holdout_seed),
        "partition": holdout_partition,
        "sequences_by_component": dict(sorted(holdout_sequences.items())),
        "scientific_inference_allowed": not synthesized,
    }


def _global_holdout_view(train_config: Any, Pos: Axis) -> AsyncDataset[Any]:
    contract = _global_holdout_contract(train_config)
    holdout_data = replace(
        train_config.data,
        train_holdout_sequences=contract["sequences_by_component"],
        train_holdout_seed=contract["seed"],
        train_holdout_partition=contract["partition"],
    )
    return holdout_data.holdout_sets(Pos)[STARCODER_COMPONENT]


def _distribution_dataset(
    distribution_id: str,
    *,
    sequence_set_id: str,
    train_config: Any,
    Pos: Axis,
    sources: Mapping[str, AsyncDataset[Any]],
) -> AsyncDataset[LmExample]:
    seed = _stable_key(sequence_set_id)
    if distribution_id in SOURCE_DISTRIBUTIONS:
        if distribution_id == "starcoder_excluded_global":
            dataset = _global_holdout_view(train_config, Pos)
        else:
            dataset = sources[distribution_id]
        if distribution_id in {"starcoder_on_policy", "nemotron_aggregate"}:
            return NamedLmDataset(dataset, Pos)
        dataset = dataset.shuffle(seed, perm_type="feistel")
        return NamedLmDataset(dataset, Pos)
    component = freeze.TARGET_COMPONENTS[distribution_id]
    return train_config.data.validation_sets(Pos)[component].shuffle(seed, perm_type="feistel")


def _weighted_training_dataset(
    *,
    starcoder_weight: float,
    sequence_set_id: str,
    train_config: Any,
    Pos: Axis,
    sources: Mapping[str, AsyncDataset[Any]],
) -> AsyncDataset[LmExample]:
    mixture = MixtureDataset(
        datasets={
            "starcoder": sources["starcoder_on_policy"],
            "nemotron": sources["nemotron_on_policy"],
        },
        weights={"starcoder": starcoder_weight, "nemotron": 1.0 - starcoder_weight},
        stop_strategy=train_config.data.stop_strategy,
        key=_stable_key(sequence_set_id),
        block_size=MIXTURE_BLOCK_SIZE,
    )
    return NamedLmDataset(mixture, Pos)


def _tree_scale(tree: Any, scale: float) -> Any:
    return jax.tree.map(lambda value: value * scale, tree)


def _tree_add(left: Any, right: Any) -> Any:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_subtract(left: Any, right: Any) -> Any:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _tree_zeros(tree: Any) -> Any:
    return jax.tree.map(jnp.zeros_like, tree)


def _array(value: Any) -> Any:
    return value.array if isinstance(value, hax.NamedArray) else value


def _flatten_named(tree: Any) -> dict[str, Any]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=lambda value: isinstance(value, hax.NamedArray) or value is None or isinstance(value, str),
    )
    return {
        jax.tree_util.keystr(path): value
        for path, value in leaves
        if isinstance(value, hax.NamedArray) or eqx.is_inexact_array(value) or isinstance(value, str)
    }


def _leaf_groups(path: str) -> tuple[str, ...]:
    groups = ["full"]
    lowered = path.lower()
    if "embedding" in lowered:
        groups.append("embedding")
    elif "lm_head" in lowered:
        groups.append("head")
    else:
        groups.append("trunk")
    match = LAYER_PATTERN.search(path)
    if match is not None:
        groups.append(f"layer_{int(match.group('layer')):02d}")
    return tuple(groups)


def _layer_axis_index(value: Any) -> int | None:
    if not isinstance(value, hax.NamedArray):
        return None
    return next((index for index, axis in enumerate(value.axes) if axis.name == "layer"), None)


def _muon_projection_coverage(model: Any, optimizer_mask: Any) -> dict[str, Any]:
    model_by_path = _flatten_named(flatten_linear_layers(model))
    mask_by_path = _flatten_named(optimizer_mask)
    muon_paths = sorted(
        path
        for path, label in mask_by_path.items()
        if label == "muonh" and path in model_by_path and not isinstance(model_by_path[path], str)
    )
    if not muon_paths:
        raise RuntimeError("MuonH tangent projection matched zero parameter leaves")
    layer_indices: set[int] = set()
    matrix_axis_counts: set[int] = set()
    for path in muon_paths:
        parameter = model_by_path[path]
        layer_axis = _layer_axis_index(parameter)
        matrix_axis_counts.add(len(_muon_matrix_axes(parameter)))
        if layer_axis is not None:
            layer_indices.update(range(parameter.axes[layer_axis].size))
    return {
        "muon_parameter_leaf_count": len(muon_paths),
        "muon_layer_count": len(layer_indices),
        "muon_matrix_axis_counts": sorted(matrix_axis_counts),
        "muon_projection_active": bool(muon_paths),
    }


def _runtime_muon_projection_coverage(model: Any, optimizer_mask: Any) -> dict[str, Any]:
    coverage = _muon_projection_coverage(model, optimizer_mask)
    if coverage["muon_layer_count"] <= 0:
        raise RuntimeError("MuonH projection found no named transformer layers")
    if coverage["muon_matrix_axis_counts"] != [2]:
        raise RuntimeError(f"MuonH projection has unexpected matrix geometry: {coverage}")
    return coverage


def _muon_matrix_axes(parameter: Any) -> tuple[int, int]:
    if isinstance(parameter, hax.NamedArray):
        layer_axis = _layer_axis_index(parameter)
        axes = tuple(index for index in range(parameter.ndim) if index != layer_axis)
    else:
        axes = tuple(range(parameter.ndim))
    if len(axes) != 2:
        raise RuntimeError(
            "MuonH tangent projection requires each flattened Linear weight to expose exactly two matrix axes; "
            f"got shape {parameter.shape}"
        )
    return cast(tuple[int, int], axes)


def _project_tangent(gradient: Any, parameter: Any, *, axes: tuple[int, int]) -> Any:
    numerator = jnp.sum(gradient * parameter, axis=axes, keepdims=True)
    denominator = jnp.maximum(jnp.sum(parameter * parameter, axis=axes, keepdims=True), 1e-30)
    return gradient - parameter * numerator / denominator


def _tree_pair_statistics(
    left: Any,
    right: Any,
    *,
    model: Any,
    optimizer_mask: Any,
    project_muon: bool,
) -> dict[str, dict[str, float | bool | None]]:
    if project_muon:
        left = flatten_linear_layers(left)
        right = flatten_linear_layers(right)
        model = flatten_linear_layers(model)
    left_by_path = _flatten_named(left)
    right_by_path = _flatten_named(right)
    model_by_path = _flatten_named(model)
    mask_by_path = _flatten_named(optimizer_mask)
    if set(left_by_path) != set(right_by_path):
        raise RuntimeError("Gradient/update tree structures differ")
    accum: dict[str, list[Any]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    expected_layer_groups: set[str] = set()
    for path, left_value in left_by_path.items():
        if isinstance(left_value, str):
            continue
        right_value = right_by_path[path]
        left_array = _array(left_value).astype(jnp.float32)
        right_array = _array(right_value).astype(jnp.float32)
        if project_muon and mask_by_path.get(path) == "muonh":
            parameter_value = model_by_path[path]
            parameter = _array(parameter_value).astype(jnp.float32)
            matrix_axes = _muon_matrix_axes(parameter_value)
            left_array = _project_tangent(left_array, parameter, axes=matrix_axes)
            right_array = _project_tangent(right_array, parameter, axes=matrix_axes)
        values = (
            jnp.sum(left_array * right_array),
            jnp.sum(left_array * left_array),
            jnp.sum(right_array * right_array),
        )
        leaf_groups = _leaf_groups(path)
        layer_axis = _layer_axis_index(left_value)
        if layer_axis is not None and any(group.startswith("layer_") for group in leaf_groups):
            raise RuntimeError(f"Parameter path and named layer axis both encode a layer identity: {path}")
        for group in leaf_groups:
            accum[group] = [accum[group][index] + value for index, value in enumerate(values)]
        if layer_axis is not None:
            reduction_axes = tuple(index for index in range(left_array.ndim) if index != layer_axis)
            per_layer_values = (
                jnp.sum(left_array * right_array, axis=reduction_axes),
                jnp.sum(left_array * left_array, axis=reduction_axes),
                jnp.sum(right_array * right_array, axis=reduction_axes),
            )
            for layer in range(left_value.axes[layer_axis].size):
                group = f"layer_{layer:02d}"
                expected_layer_groups.add(group)
                accum[group] = [
                    accum[group][index] + per_layer_values[index][layer] for index in range(len(per_layer_values))
                ]
    missing_layer_groups = expected_layer_groups - set(accum)
    if missing_layer_groups:
        raise RuntimeError(f"Gradient statistics omitted named transformer layers: {sorted(missing_layer_groups)}")
    result: dict[str, dict[str, float | bool | None]] = {}
    for group, (dot, left_sq, right_sq) in accum.items():
        dot_value, left_value, right_value = map(float, jax.device_get((dot, left_sq, right_sq)))
        denominator = math.sqrt(max(left_value * right_value, 0.0))
        result[group] = {
            "dot": dot_value,
            "left_norm": math.sqrt(max(left_value, 0.0)),
            "right_norm": math.sqrt(max(right_value, 0.0)),
            "cosine": dot_value / denominator if denominator > 0 else None,
            "cosine_defined": denominator > 0,
        }
    return result


def _raw_tree_dot(left: Any, right: Any) -> Any:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return sum(
        jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32)) for x, y in zip(left_leaves, right_leaves, strict=True)
    )


def _raw_tree_norm(tree: Any) -> Any:
    return jnp.sqrt(jnp.maximum(_raw_tree_dot(tree, tree), 0.0))


def _tree_max_abs_diff(left: Any, right: Any) -> float:
    values = [jnp.max(jnp.abs(x - y)) for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)]
    return float(jax.device_get(jnp.max(jnp.stack(values))))


def _tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if value is None or isinstance(value, str):
            continue
        array = np.asarray(jax.device_get(_array(value)))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_capacity(dataset: AsyncDataset[Any], blocks: int, *, label: str) -> dict[str, Any]:
    required = blocks * PROBE_BATCH_SIZE
    if not dataset.is_finite():
        return {
            "dataset_is_finite": False,
            "available_sequence_count": None,
            "required_sequence_count": required,
            "sequence_margin": None,
        }
    available = int(blocking_wait(dataset.async_len()))
    if available < required:
        raise RuntimeError(
            f"Finite dataset {label} supplies {available} sequences but {required} are required "
            f"for {blocks} complete probe blocks"
        )
    return {
        "dataset_is_finite": True,
        "available_sequence_count": available,
        "required_sequence_count": required,
        "sequence_margin": available - required,
    }


def _gradient_functions(trainer: Trainer):
    @named_jit(
        axis_resources=trainer.parameter_axis_mapping,
        out_axis_resources=trainer.parameter_axis_mapping,
    )
    def gradient(model: Any, example: LmExample, key: Any):
        loss, grads, _ = trainer._compute_gradients_microbatched(
            trainer.loss_fn,
            model,
            example,
            key=key,
        )
        return loss, grads

    @named_jit(
        axis_resources=trainer.parameter_axis_mapping,
        out_axis_resources=trainer.parameter_axis_mapping,
    )
    def update(state: Any, grads: Any, loss: Any, key: Any):
        _, updates = state.take_step(grads, obj_fun=None, loss=loss, key=key)
        return updates

    @named_jit(axis_resources=trainer.parameter_axis_mapping)
    def evaluation_loss(model: Any, example: LmExample):
        loss, _ = trainer.loss_fn(model, example, key=None)
        return loss

    train_step = named_jit(
        functools.partial(trainer._train_step, _no_hooks=True),
        axis_resources=trainer.parameter_axis_mapping,
        out_axis_resources=trainer.parameter_axis_mapping,
    )
    return gradient, update, evaluation_loss, train_step


def _loss_scalar(loss: Any) -> float:
    value = loss.scalar() if hasattr(loss, "scalar") else loss
    return float(jax.device_get(value))


def _mean_gradient_and_update(
    *,
    trainer: Trainer,
    state: Any,
    dataset: AsyncDataset[LmExample],
    blocks: int,
    update_draws: int,
    seed_id: str,
    gradient_fn: Any,
    update_fn: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    capacity = _dataset_capacity(dataset, blocks, label=f"gradient:{seed_id}")
    loader = trainer.data_loader(dataset, batch=PROBE_BATCH_SIZE)
    iterator = iter(loader)
    mean_gradient = None
    mean_update = None
    losses: list[float] = []
    prior_gradient = None
    prior_loss = None
    update_count = 0
    first_example = None
    first_gradient = None
    first_loss = None
    data_update_norms: list[float] = []
    no_data_update_norms: list[float] = []
    corrected_update_norms: list[float] = []
    base_key = _fold_in_stable(state.training_key, seed_id)
    for block in range(blocks):
        example = next(iterator)
        key = jrandom.fold_in(base_key, block)
        loss, gradient = gradient_fn(state.model, example, key)
        scalar_loss = _loss_scalar(loss)
        if not math.isfinite(scalar_loss):
            raise RuntimeError("Non-finite probe loss")
        losses.append(scalar_loss)
        mean_gradient = gradient if mean_gradient is None else _tree_add(mean_gradient, gradient)
        if block == 0:
            first_example, first_gradient, first_loss = example, gradient, scalar_loss
        if block % 2 == 0:
            prior_gradient, prior_loss = gradient, scalar_loss
            continue
        if update_count >= update_draws:
            continue
        assert prior_gradient is not None and prior_loss is not None
        training_gradient = _tree_scale(_tree_add(prior_gradient, gradient), 0.5)
        training_loss = 0.5 * (prior_loss + scalar_loss)
        update_key = jrandom.fold_in(base_key, 100_000 + update_count)
        data_update = update_fn(state, training_gradient, training_loss, update_key)
        no_data_update = update_fn(state, _tree_zeros(training_gradient), training_loss, update_key)
        corrected_update = _tree_subtract(data_update, no_data_update)
        data_update_norms.append(float(jax.device_get(_raw_tree_norm(data_update))))
        no_data_update_norms.append(float(jax.device_get(_raw_tree_norm(no_data_update))))
        corrected_update_norms.append(float(jax.device_get(_raw_tree_norm(corrected_update))))
        mean_update = corrected_update if mean_update is None else _tree_add(mean_update, corrected_update)
        update_count += 1
    if mean_gradient is None or first_example is None or first_gradient is None or first_loss is None:
        raise RuntimeError("Probe produced no gradients")
    mean_gradient = _tree_scale(mean_gradient, 1.0 / blocks)
    if update_count != update_draws:
        raise RuntimeError(f"Optimizer update draws incomplete: {update_count} != {update_draws}")
    if update_count:
        assert mean_update is not None
        mean_update = _tree_scale(mean_update, 1.0 / update_count)
    else:
        mean_update = _tree_zeros(mean_gradient)

    repeated_loss, repeated_gradient = gradient_fn(state.model, first_example, jrandom.fold_in(base_key, 0))
    repeated_loss_scalar = _loss_scalar(repeated_loss)
    repeat_gradient_difference = _tree_max_abs_diff(first_gradient, repeated_gradient)
    repeat_loss_difference = abs(first_loss - repeated_loss_scalar)
    if repeat_gradient_difference > NUMERICAL_TOLERANCE or repeat_loss_difference > NUMERICAL_TOLERANCE:
        raise RuntimeError(
            f"Gradient repeat is not deterministic: gradient={repeat_gradient_difference}, loss={repeat_loss_difference}"
        )
    return (
        mean_gradient,
        mean_update,
        {
            "replicate_block_count": blocks,
            "optimizer_update_draw_count": update_count,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "loss_min": min(losses),
            "loss_max": max(losses),
            "repeat_gradient_max_abs_difference": repeat_gradient_difference,
            "repeat_loss_absolute_difference": repeat_loss_difference,
            "first_batch_sha256": _tree_sha256(first_example),
            "data_supply": capacity,
            "data_update_norm_mean": float(np.mean(data_update_norms)) if data_update_norms else 0.0,
            "optimizer_memory_update_norm_mean": float(np.mean(no_data_update_norms)) if no_data_update_norms else 0.0,
            "corrected_update_norm_mean": float(np.mean(corrected_update_norms)) if corrected_update_norms else 0.0,
            "optimizer_memory_update_nonzero": bool(any(value > 0.0 for value in no_data_update_norms)),
        },
    )


def _per_block_reference_cosines(
    *,
    trainer: Trainer,
    state: Any,
    dataset: AsyncDataset[LmExample],
    blocks: int,
    seed_id: str,
    gradient_fn: Any,
    references: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    _dataset_capacity(dataset, blocks, label=f"per-block-reference:{seed_id}")
    loader = trainer.data_loader(dataset, batch=PROBE_BATCH_SIZE)
    iterator = iter(loader)
    base_key = _fold_in_stable(state.training_key, seed_id)
    values = {name: [] for name in references}
    for block in range(blocks):
        example = next(iterator)
        _, gradient = gradient_fn(state.model, example, jrandom.fold_in(base_key, block))
        gradient_norm = _raw_tree_norm(gradient)
        for name, reference in references.items():
            denominator = gradient_norm * _raw_tree_norm(reference)
            cosine = _raw_tree_dot(gradient, reference) / jnp.maximum(denominator, 1e-30)
            values[name].append(float(jax.device_get(cosine)))
    return {
        name: {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
            "count": len(samples),
        }
        for name, samples in values.items()
    }


def run_probe_group(config: ProbeGroupConfig) -> None:
    _verify_group_contract(config)
    if _assert_existing_group_complete(config):
        return
    metadata = _read_checkpoint_metadata(config.checkpoint_uri, config.checkpoint_step)
    train_config = _prepare_train_config(config.pod_config, config.checkpoint_uri, config.group_id)
    trainer, state, Pos, data_key, optimizer_mask = _initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError(
                f"Restored state step is {int(state.step)}, expected {config.expected_restored_state_step} "
                f"after checkpoint label {config.checkpoint_step}"
            )
        runtime_summary = {
            "restoration": _restored_optimizer_summary(
                state,
                config.checkpoint_step,
                config.expected_restored_state_step,
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "muon_projection": _runtime_muon_projection_coverage(state.model, optimizer_mask),
            "optimizer_schedule": _optimizer_schedule_summary(train_config),
            "optimizer_update_statistic": "finite_difference_Delta(g)_minus_Delta(0)_not_causal_attribution",
        }
        sources, stream_summary = _source_views(
            train_config,
            Pos,
            data_key,
            config.expected_restored_state_step,
        )
        runtime_summary["source_stream"] = stream_summary
        gradient_fn, update_fn, _, _ = _gradient_functions(trainer)
        means: dict[str, Any] = {}
        updates: dict[str, Any] = {}
        row_summaries: dict[str, dict[str, Any]] = {}
        datasets: dict[str, AsyncDataset[LmExample]] = {}
        for row in config.rows:
            distribution = str(row["distribution_id"])
            blocks = int(row["replicate_blocks"])
            update_draws = min(int(row["optimizer_update_draw_count"]), blocks // 2)
            dataset = _distribution_dataset(
                distribution,
                sequence_set_id=str(row["probe_sequence_set_id"]),
                train_config=train_config,
                Pos=Pos,
                sources=sources,
            )
            datasets[distribution] = dataset
            mean_gradient, mean_update, summary = _mean_gradient_and_update(
                trainer=trainer,
                state=state,
                dataset=dataset,
                blocks=blocks,
                update_draws=update_draws,
                seed_id=str(row["row_id"]),
                gradient_fn=gradient_fn,
                update_fn=update_fn,
            )
            means[distribution] = mean_gradient
            updates[distribution] = mean_update
            row_summaries[distribution] = summary

        for row in config.rows:
            if _assert_existing_row_identity(config.output_path, row, config.release_sha256):
                continue
            distribution = str(row["distribution_id"])
            counterparts = (
                {name: means[name] for name in TARGET_DISTRIBUTIONS if name in means}
                if distribution in SOURCE_DISTRIBUTIONS
                else {name: means[name] for name in SOURCE_DISTRIBUTIONS if name in means}
            )
            blocks = int(row["replicate_blocks"])
            per_block_reference_cosines = _per_block_reference_cosines(
                trainer=trainer,
                state=state,
                dataset=datasets[distribution],
                blocks=blocks,
                seed_id=str(row["row_id"]),
                gradient_fn=gradient_fn,
                references=counterparts,
            )
            pairwise = {
                other: {
                    "raw_gradient": _tree_pair_statistics(
                        means[distribution],
                        means[other],
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=False,
                    ),
                    "projected_gradient": _tree_pair_statistics(
                        means[distribution],
                        means[other],
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=True,
                    ),
                    "raw_optimizer_update": _tree_pair_statistics(
                        updates[distribution],
                        updates[other],
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=False,
                    ),
                    "projected_optimizer_update": _tree_pair_statistics(
                        updates[distribution],
                        updates[other],
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=True,
                    ),
                }
                for other in sorted(counterparts)
            }
            identity = _row_identity(row, config.release_sha256)
            _write_create_only_json(
                _row_path(config.output_path, str(row["row_id"])),
                {
                    "kind": "gradient_probe",
                    "scope": config.scope,
                    "group_id": config.group_id,
                    "row": row,
                    "checkpoint_metadata": metadata,
                    "restored_state_step": int(state.step),
                    "runtime_summary": runtime_summary,
                    "cache_provenance_sha256": config.cache_provenance_sha256,
                    "release_sha256": config.release_sha256,
                    "numerical_summary": row_summaries[distribution],
                    "pairwise_statistics": pairwise,
                    "per_block_reference_cosines_recomputed_same_blocks": per_block_reference_cosines,
                    "endpoint_metrics_read": False,
                },
                identity_sha256=identity,
            )
        _write_group_complete(
            config,
            kind="gradient_probe",
            checkpoint_metadata=metadata,
            runtime_summary=runtime_summary,
        )
    finally:
        _close_runtime(trainer)


def _target_mean_gradients(
    *,
    trainer: Trainer,
    state: Any,
    Pos: Axis,
    train_config: Any,
    sources: Mapping[str, AsyncDataset[Any]],
    gradient_fn: Any,
    update_fn: Any,
    block_counts: Mapping[str, int],
    sequence_set_ids: Mapping[str, str],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for distribution in sorted(TARGET_DISTRIBUTIONS):
        blocks = int(block_counts[distribution])
        dataset = _distribution_dataset(
            distribution,
            sequence_set_id=sequence_set_ids[distribution],
            train_config=train_config,
            Pos=Pos,
            sources=sources,
        )
        mean_gradient, _, _ = _mean_gradient_and_update(
            trainer=trainer,
            state=state,
            dataset=dataset,
            blocks=blocks,
            update_draws=0,
            seed_id=sequence_set_ids[distribution],
            gradient_fn=gradient_fn,
            update_fn=update_fn,
        )
        targets[distribution] = mean_gradient
    return targets


def run_optimizer_group(config: OptimizerGroupConfig) -> None:
    _verify_group_contract(config)
    if _assert_existing_group_complete(config):
        return
    metadata = _read_checkpoint_metadata(config.checkpoint_uri, config.checkpoint_step)
    train_config = _prepare_train_config(config.pod_config, config.checkpoint_uri, config.group_id)
    trainer, state, Pos, data_key, optimizer_mask = _initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError("Optimizer transform restored the wrong checkpoint step")
        runtime_summary = {
            "restoration": _restored_optimizer_summary(
                state,
                config.checkpoint_step,
                config.expected_restored_state_step,
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "muon_projection": _runtime_muon_projection_coverage(state.model, optimizer_mask),
            "optimizer_schedule": _optimizer_schedule_summary(train_config),
            "optimizer_update_statistic": "finite_difference_Delta(g)_minus_Delta(0)_not_causal_attribution",
        }
        sources, stream_summary = _source_views(
            train_config,
            Pos,
            data_key,
            config.expected_restored_state_step,
        )
        runtime_summary["source_stream"] = stream_summary
        gradient_fn, update_fn, _, _ = _gradient_functions(trainer)
        target_blocks = dict(config.target_block_counts)
        targets = _target_mean_gradients(
            trainer=trainer,
            state=state,
            Pos=Pos,
            train_config=train_config,
            sources=sources,
            gradient_fn=gradient_fn,
            update_fn=update_fn,
            block_counts=target_blocks,
            sequence_set_ids=dict(config.target_sequence_set_ids),
        )
        for row in config.rows:
            if _assert_existing_row_identity(config.output_path, row, config.release_sha256):
                continue
            draws = int(row["optimizer_update_draw_count"])
            dataset = _weighted_training_dataset(
                starcoder_weight=float(row["starcoder_weight"]),
                sequence_set_id=f"optimizer:{row['row_id']}",
                train_config=train_config,
                Pos=Pos,
                sources=sources,
            )
            mean_gradient, mean_update, summary = _mean_gradient_and_update(
                trainer=trainer,
                state=state,
                dataset=dataset,
                blocks=draws * 2,
                update_draws=draws,
                seed_id=str(row["row_id"]),
                gradient_fn=gradient_fn,
                update_fn=update_fn,
            )
            target_utility = {
                target: {
                    "raw_gradient": _tree_pair_statistics(
                        targets[target],
                        mean_gradient,
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=False,
                    ),
                    "projected_gradient": _tree_pair_statistics(
                        targets[target],
                        mean_gradient,
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=True,
                    ),
                    "raw_optimizer_update": _tree_pair_statistics(
                        _tree_scale(targets[target], -1.0),
                        mean_update,
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=False,
                    ),
                    "projected_optimizer_update": _tree_pair_statistics(
                        _tree_scale(targets[target], -1.0),
                        mean_update,
                        model=state.model,
                        optimizer_mask=optimizer_mask,
                        project_muon=True,
                    ),
                }
                for target in sorted(targets)
            }
            _write_create_only_json(
                _row_path(config.output_path, str(row["row_id"])),
                {
                    "kind": "optimizer_transform",
                    "scope": config.scope,
                    "group_id": config.group_id,
                    "row": row,
                    "checkpoint_metadata": metadata,
                    "restored_state_step": int(state.step),
                    "runtime_summary": runtime_summary,
                    "cache_provenance_sha256": config.cache_provenance_sha256,
                    "release_sha256": config.release_sha256,
                    "numerical_summary": summary,
                    "target_utility_statistics": target_utility,
                    "endpoint_metrics_read": False,
                },
                identity_sha256=_row_identity(row, config.release_sha256),
            )
        _write_group_complete(
            config,
            kind="optimizer_transform",
            checkpoint_metadata=metadata,
            runtime_summary=runtime_summary,
        )
    finally:
        _close_runtime(trainer)


def _evaluate_target(
    *,
    trainer: Trainer,
    model: Any,
    dataset: AsyncDataset[LmExample],
    evaluation_loss: Any,
    blocks: int,
) -> dict[str, Any]:
    capacity = _dataset_capacity(dataset, blocks, label="rollout-target")
    iterator = iter(trainer.data_loader(dataset, batch=PROBE_BATCH_SIZE))
    losses = [_loss_scalar(evaluation_loss(model, next(iterator))) for _ in range(blocks)]
    return {
        "loss_nats": float(np.mean(losses)),
        "bpb": float(np.mean(losses) / math.log(2.0)),
        "bpb_standard_error": (
            float(np.std(losses, ddof=1) / math.sqrt(len(losses)) / math.log(2.0)) if len(losses) > 1 else 0.0
        ),
        "block_count": blocks,
        "data_supply": capacity,
    }


def run_rollout_group(config: RolloutGroupConfig) -> None:
    _verify_group_contract(config)
    if _assert_existing_group_complete(config):
        return
    metadata = _read_checkpoint_metadata(config.checkpoint_uri, config.checkpoint_step)
    train_config = _prepare_train_config(config.pod_config, config.checkpoint_uri, config.group_id)
    trainer, state, Pos, data_key, optimizer_mask = _initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError("Rollout restored the wrong checkpoint step")
        runtime_summary = {
            "restoration": _restored_optimizer_summary(
                state,
                config.checkpoint_step,
                config.expected_restored_state_step,
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "muon_projection": _runtime_muon_projection_coverage(state.model, optimizer_mask),
            "optimizer_schedule": _optimizer_schedule_summary(train_config),
            "optimizer_update_statistic": "finite_difference_Delta(g)_minus_Delta(0)_not_causal_attribution",
        }
        sources, stream_summary = _source_views(
            train_config,
            Pos,
            data_key,
            config.expected_restored_state_step,
        )
        runtime_summary["source_stream"] = stream_summary
        _, _, evaluation_loss, train_step = _gradient_functions(trainer)
        target_blocks = dict(config.target_block_counts)["paloma_programming_languages"]
        target = _distribution_dataset(
            "paloma_programming_languages",
            sequence_set_id=dict(config.target_sequence_set_ids)["paloma_programming_languages"],
            train_config=train_config,
            Pos=Pos,
            sources=sources,
        )
        for row in config.rows:
            if _assert_existing_row_identity(config.output_path, row, config.release_sha256):
                continue
            updates = int(row["updates"])
            readouts = sorted({int(step) for step in str(row["readout_steps"]).split("|") if int(step) <= updates})
            dataset = _weighted_training_dataset(
                starcoder_weight=float(row["starcoder_weight"]),
                sequence_set_id=f"rollout:{config.group_id}:order{row['rollout_order_seed']}",
                train_config=train_config,
                Pos=Pos,
                sources=sources,
            )
            iterator = iter(trainer.data_loader(dataset))
            rollout_state = state
            measurements: list[dict[str, Any]] = []
            for update_index in range(1, updates + 1):
                example = next(iterator)
                result = train_step(rollout_state, (example,), {})
                rollout_state = result.new_state
                if update_index in readouts:
                    measurements.append(
                        {
                            "updates": update_index,
                            **_evaluate_target(
                                trainer=trainer,
                                model=rollout_state.model,
                                dataset=target,
                                evaluation_loss=evaluation_loss,
                                blocks=target_blocks,
                            ),
                        }
                    )
            if int(rollout_state.step) != config.expected_restored_state_step + updates:
                raise RuntimeError("Rollout state advanced by the wrong number of updates")
            if [measurement["updates"] for measurement in measurements] != readouts:
                raise RuntimeError("Rollout did not emit every frozen readout exactly once")
            _write_create_only_json(
                _row_path(config.output_path, str(row["row_id"])),
                {
                    "kind": "short_rollout",
                    "scope": config.scope,
                    "group_id": config.group_id,
                    "row": row,
                    "checkpoint_metadata": metadata,
                    "restored_state_step": int(state.step),
                    "final_state_step": int(rollout_state.step),
                    "runtime_summary": runtime_summary,
                    "cache_provenance_sha256": config.cache_provenance_sha256,
                    "release_sha256": config.release_sha256,
                    "readouts": measurements,
                    "endpoint_metrics_read": False,
                },
                identity_sha256=_row_identity(row, config.release_sha256),
            )
        _write_group_complete(
            config,
            kind="short_rollout",
            checkpoint_metadata=metadata,
            runtime_summary=runtime_summary,
        )
    finally:
        _close_runtime(trainer)


def _read_manifest(name: str) -> list[dict[str, Any]]:
    path = freeze.OUTPUT_DIR / name
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_release(expected_sha256: str) -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text())
    observed = release["release_sha256"]
    computed = freeze.canonical_sha256({**release, "release_sha256": ""})
    if observed != computed or observed != expected_sha256:
        raise ValueError(f"Probe release mismatch: observed={observed}, computed={computed}, expected={expected_sha256}")
    if release["endpoint_metrics_read"] is not False:
        raise ValueError("Probe release is contaminated by endpoint-result access")
    for relative_path, sha256 in release["implementation_files"].items():
        if freeze.file_sha256(freeze.REPO_ROOT / relative_path) != sha256:
            raise ValueError(f"Frozen implementation drifted: {relative_path}")
    for summary in release["manifests"].values():
        path = freeze.REPO_ROOT / summary["path"]
        if freeze.file_sha256(path) != summary["sha256"]:
            raise ValueError(f"Frozen manifest drifted: {summary['path']}")
    for name, sha256 in release["source_design_files"].items():
        if freeze.file_sha256(freeze.DESIGN_DIR / name) != sha256:
            raise ValueError(f"Frozen design input drifted: {name}")
    if (
        release["required_region"] != TPU_REGION
        or release["required_zone"] != TPU_ZONE
        or release["required_bucket_prefix"] != freeze.MARIN_PREFIX
        or not release["result_root"].startswith(release["required_bucket_prefix"])
    ):
        raise ValueError("Release locality contract does not match the worker implementation")
    return release


def _assert_full_launch_authorized(release: Mapping[str, Any], confirmation: str | None) -> None:
    if confirmation != FULL_LAUNCH_CONFIRMATION:
        raise ValueError("Full launch requires the explicit reviewed confirmation token")
    if not FULL_LAUNCH_AUTHORIZATION_PATH.exists():
        raise ValueError("Full launch remains blocked pending a user-authorized release sidecar")
    authorization = json.loads(FULL_LAUNCH_AUTHORIZATION_PATH.read_text())
    expected = {
        "full_launch_authorized": True,
        "release_sha256": release["release_sha256"],
        "confirmation": FULL_LAUNCH_CONFIRMATION,
    }
    if authorization != expected:
        raise ValueError("Full launch authorization sidecar does not match the frozen release")


def _group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return {group_id: tuple(group) for group_id, group in grouped.items()}


def _pod_configs(scope: str) -> dict[str, TrainLmOnPodConfig]:
    return freeze._canary_configs() if scope == "canary" else freeze._full_configs()


def _audit_frozen_provenance(
    scope: str,
    release: Mapping[str, Any],
    pod_configs: Mapping[str, TrainLmOnPodConfig],
) -> dict[str, Any]:
    cache_rows = _read_manifest("cache_provenance.csv")
    for row in cache_rows:
        for uri_key, hash_key in (
            ("shard_ledger_uri", "shard_ledger_sha256"),
            ("completion_uri", "completion_sha256"),
        ):
            observed = freeze._remote_file_sha256(str(row[uri_key]))
            if observed != row[hash_key]:
                raise RuntimeError(f"Frozen cache provenance drifted at {row[uri_key]}")

    config_rows = {row["trajectory_id"]: row for row in _read_manifest("config_provenance.csv") if row["scope"] == scope}
    if set(config_rows) != set(pod_configs):
        raise RuntimeError(f"Frozen {scope} configuration inventory drifted")
    for trajectory_id, pod_config in pod_configs.items():
        expected = config_rows[trajectory_id]
        observed = freeze._config_identity(pod_config)
        for key, value in observed.items():
            serialized = "" if value is None else str(value)
            if expected[key] != serialized:
                raise RuntimeError(f"Frozen configuration drifted for {trajectory_id}/{key}")

    representative = next(iter(pod_configs.values())).train_config
    sequence_length = representative.train_seq_len or representative.model.max_seq_len
    Pos = representative.model.max_Pos.resize(sequence_length)
    validation_sets = representative.data.validation_sets(Pos)
    target_lengths: dict[str, int] = {}
    for distribution, contract in release["target_sampling_contract"].items():
        if int(contract["materialized_sequence_length"]) != sequence_length:
            raise RuntimeError(f"Frozen target sequence length drifted for {distribution}")
        component = str(contract["component_name"])
        observed_length = int(blocking_wait(validation_sets[component].async_len()))
        if observed_length != int(contract["available_sequence_count"]):
            raise RuntimeError(f"Frozen target population drifted for {distribution}")
        target_lengths[distribution] = observed_length
    return {
        "scope": scope,
        "cache_objects_rehashed": len(cache_rows) * 2,
        "configurations_reconstructed": len(pod_configs),
        "target_materialized_sequence_counts": target_lengths,
        "target_materialized_sequence_length": sequence_length,
        "endpoint_metrics_read": False,
    }


def _target_block_counts(release: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    contract = release["target_sampling_contract"]
    counts = tuple(
        sorted(
            (
                distribution,
                min(freeze.PRIMARY_TARGET_BLOCKS, int(details["maximum_unique_full_blocks"])),
            )
            for distribution, details in contract.items()
        )
    )
    if {distribution for distribution, _ in counts} != set(TARGET_DISTRIBUTIONS):
        raise ValueError("Frozen target block contract is incomplete")
    return counts


def _target_sequence_set_ids(row: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    identities = tuple(
        sorted((str(key), str(value)) for key, value in json.loads(row["target_sequence_set_ids_json"]).items())
    )
    if {distribution for distribution, _ in identities} != set(TARGET_DISTRIBUTIONS):
        raise ValueError("Frozen target sequence-set contract is incomplete")
    return identities


def _resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(
        TPU_TYPE,
        cpu=TPU_HOST_CPU,
        ram=TPU_HOST_RAM,
        regions=(TPU_REGION,),
        zone=TPU_ZONE,
    )


def _artifact_name(scope: str, kind: str, group_id: str) -> str:
    prefix = f"{freeze.MARIN_PREFIX}/"
    if not freeze.RESULT_ROOT.startswith(prefix):
        raise ValueError(f"Result root is outside the frozen Marin prefix: {freeze.RESULT_ROOT}")
    return f"{freeze.RESULT_ROOT.removeprefix(prefix)}/{scope}/{kind}/{group_id}"


def _probe_steps(
    scope: str,
    *,
    release: dict[str, Any],
    pod_configs: Mapping[str, TrainLmOnPodConfig],
) -> list[ArtifactStep[Artifact]]:
    rows = _read_manifest(f"{scope}_probe_manifest.csv")
    groups = _group_rows(rows)
    cache_sha = release["manifests"]["cache_provenance"]["sha256"]
    resources = _resources()
    steps: list[ArtifactStep[Artifact]] = []
    for group_id, group in sorted(groups.items()):
        trajectory_id = group[0]["trajectory_id"]
        base = ProbeGroupConfig(
            scope=scope,
            group_id=group_id,
            checkpoint_uri=group[0]["checkpoint_uri"],
            checkpoint_step=int(group[0]["checkpoint_step"]),
            expected_restored_state_step=int(group[0]["expected_restored_state_step"]),
            rows=group,
            pod_config=pod_configs[trajectory_id],
            output_path="",
            cache_provenance_sha256=cache_sha,
            release_sha256=release["release_sha256"],
        )
        steps.append(
            ArtifactStep(
                name=_artifact_name(scope, "probe", group_id),
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_probe_group, resources=resources, name=group_id),
                build_config=lambda ctx, base=base: replace(base, output_path=ctx.output_path),
            )
        )
    return steps


def _optimizer_steps(
    scope: str,
    *,
    release: dict[str, Any],
    pod_configs: Mapping[str, TrainLmOnPodConfig],
) -> list[ArtifactStep[Artifact]]:
    rows = _read_manifest(f"{scope}_optimizer_manifest.csv")
    cache_sha = release["manifests"]["cache_provenance"]["sha256"]
    resources = _resources()
    steps: list[ArtifactStep[Artifact]] = []
    for group_id, group in sorted(_group_rows(rows).items()):
        trajectory_id = group[0]["parent_trajectory_id"]
        base = OptimizerGroupConfig(
            scope=scope,
            group_id=group_id,
            checkpoint_uri=group[0]["checkpoint_uri"],
            checkpoint_step=int(group[0]["parent_checkpoint_step"]),
            expected_restored_state_step=int(group[0]["expected_restored_state_step"]),
            rows=group,
            pod_config=pod_configs[trajectory_id],
            output_path="",
            cache_provenance_sha256=cache_sha,
            release_sha256=release["release_sha256"],
            target_block_counts=_target_block_counts(release),
            target_sequence_set_ids=_target_sequence_set_ids(group[0]),
        )
        steps.append(
            ArtifactStep(
                name=_artifact_name(scope, "optimizer", group_id),
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_optimizer_group, resources=resources, name=group_id),
                build_config=lambda ctx, base=base: replace(base, output_path=ctx.output_path),
            )
        )
    return steps


def _rollout_steps(
    scope: str,
    *,
    release: dict[str, Any],
    pod_configs: Mapping[str, TrainLmOnPodConfig],
) -> list[ArtifactStep[Artifact]]:
    rows = _read_manifest(f"{scope}_rollout_manifest.csv")
    cache_sha = release["manifests"]["cache_provenance"]["sha256"]
    resources = _resources()
    steps: list[ArtifactStep[Artifact]] = []
    for group_id, group in sorted(_group_rows(rows).items()):
        trajectory_id = group[0]["parent_trajectory_id"]
        base = RolloutGroupConfig(
            scope=scope,
            group_id=group_id,
            checkpoint_uri=group[0]["checkpoint_uri"],
            checkpoint_step=int(group[0]["parent_checkpoint_step"]),
            expected_restored_state_step=int(group[0]["expected_restored_state_step"]),
            rows=group,
            pod_config=pod_configs[trajectory_id],
            output_path="",
            cache_provenance_sha256=cache_sha,
            release_sha256=release["release_sha256"],
            target_block_counts=_target_block_counts(release),
            target_sequence_set_ids=_target_sequence_set_ids(group[0]),
        )
        steps.append(
            ArtifactStep(
                name=_artifact_name(scope, "rollout", group_id),
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_rollout_group, resources=resources, name=group_id),
                build_config=lambda ctx, base=base: replace(base, output_path=ctx.output_path),
            )
        )
    return steps


def _audit_checkpoint_readiness(scope: str) -> dict[str, Any]:
    manifests = [
        _read_manifest(f"{scope}_probe_manifest.csv"),
        _read_manifest(f"{scope}_optimizer_manifest.csv"),
        _read_manifest(f"{scope}_rollout_manifest.csv"),
    ]
    expected = {
        (row["checkpoint_uri"], int(row.get("checkpoint_step", row.get("parent_checkpoint_step"))))
        for rows in manifests
        for row in rows
    }

    async def audit_one(uri: str, step: int) -> dict[str, Any]:
        try:
            await asyncio.to_thread(_read_checkpoint_metadata, uri, step)
        except Exception as error:
            return {"checkpoint_uri": uri, "checkpoint_step": step, "ready": False, "error": repr(error)}
        return {"checkpoint_uri": uri, "checkpoint_step": step, "ready": True, "error": ""}

    async def audit_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(64)

        async def bounded(uri: str, step: int) -> dict[str, Any]:
            async with semaphore:
                return await audit_one(uri, step)

        return await asyncio.gather(*(bounded(uri, step) for uri, step in sorted(expected)))

    results = asyncio.run(audit_all())
    failures = [result for result in results if not result["ready"]]
    return {
        "expected": len(expected),
        "ready": len(results) - len(failures),
        "missing": len(failures),
        "failures": failures,
    }


def _contains_nonfinite_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_nonfinite_number(item) for item in value)
    return False


def _read_inventory_document(fs: Any, plain_path: str) -> dict[str, Any]:
    with fs.open(plain_path, "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing output has an unexpected schema: {plain_path}")
    return document


def audit_outputs(scope: str) -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text())
    kinds = ("probe", "optimizer", "rollout")
    expected_document_kind = {
        "probe": "gradient_probe",
        "optimizer": "optimizer_transform",
        "rollout": "short_rollout",
    }
    report: dict[str, Any] = {"scope": scope, "endpoint_metrics_read": False}
    for kind in kinds:
        rows = _read_manifest(f"{scope}_{kind}_manifest.csv")
        base_uri = _path_join(freeze.RESULT_ROOT, scope, kind)
        fs, base_path = fsspec.core.url_to_fs(base_uri)
        found_row_paths = set(fs.glob(f"{base_path}/*/{ARTIFACT_VERSION}/rows/*.json"))
        expected_row_paths = {
            f"{base_path}/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json": row for row in rows
        }
        present_row_paths = set(expected_row_paths) & found_row_paths
        duplicates = len(rows) - len({row["row_id"] for row in rows})
        identity_mismatches = 0
        invalid_documents = 0
        nonfinite_documents = 0
        for path in sorted(present_row_paths):
            row = expected_row_paths[path]
            document = _read_inventory_document(fs, path)
            identity_mismatches += document.get("identity_sha256") != _row_identity(row, release["release_sha256"])
            runtime_summary = document.get("runtime_summary", {})
            valid = (
                document.get("kind") == expected_document_kind[kind]
                and document.get("scope") == scope
                and document.get("group_id") == row["group_id"]
                and document.get("row", {}).get("row_id") == row["row_id"]
                and document.get("release_sha256") == release["release_sha256"]
                and document.get("endpoint_metrics_read") is False
                and runtime_summary.get("restoration", {}).get("trainer_state_step_matches_expected") is True
                and runtime_summary.get("restoration", {}).get("optimizer_counter_matches_expected") is True
                and runtime_summary.get("restoration", {}).get("allow_partial_checkpoint") is False
                and runtime_summary.get("muon_projection", {}).get("muon_projection_active") is True
                and runtime_summary.get("muon_projection", {}).get("muon_parameter_leaf_count", 0) > 0
                and runtime_summary.get("muon_projection", {}).get("muon_layer_count", 0) > 0
                and runtime_summary.get("muon_projection", {}).get("muon_matrix_axis_counts") == [2]
            )
            invalid_documents += not valid
            nonfinite_documents += _contains_nonfinite_number(document)
        groups = _group_rows(rows)
        found_marker_paths = set(fs.glob(f"{base_path}/*/{ARTIFACT_VERSION}/group_complete.json"))
        expected_marker_paths = {
            f"{base_path}/{group_id}/{ARTIFACT_VERSION}/group_complete.json": (group_id, group)
            for group_id, group in groups.items()
        }
        present_marker_paths = set(expected_marker_paths) & found_marker_paths
        invalid_group_markers = 0
        for path in sorted(present_marker_paths):
            group_id, group = expected_marker_paths[path]
            marker = _read_inventory_document(fs, path)
            marker_valid = (
                marker.get("identity_sha256") == _group_identity(group_id, group, release["release_sha256"])
                and marker.get("row_count") == len(group)
                and marker.get("release_sha256") == release["release_sha256"]
                and marker.get("endpoint_metrics_read") is False
                and marker.get("runtime_summary", {}).get("restoration", {}).get("trainer_state_step_matches_expected")
                is True
                and marker.get("runtime_summary", {}).get("restoration", {}).get("optimizer_counter_matches_expected")
                is True
                and marker.get("runtime_summary", {}).get("muon_projection", {}).get("muon_projection_active") is True
                and marker.get("runtime_summary", {}).get("muon_projection", {}).get("muon_layer_count", 0) > 0
                and marker.get("runtime_summary", {}).get("muon_projection", {}).get("muon_matrix_axis_counts") == [2]
            )
            invalid_group_markers += not marker_valid
        report[kind] = {
            "expected_rows": len(rows),
            "found_rows": len(present_row_paths),
            "missing_rows": len(rows) - len(present_row_paths),
            "unexpected_row_objects": len(found_row_paths - set(expected_row_paths)),
            "duplicate_manifest_rows": duplicates,
            "identity_mismatches": identity_mismatches,
            "invalid_documents": invalid_documents,
            "nonfinite_documents": nonfinite_documents,
            "expected_groups": len(groups),
            "complete_groups": len(present_marker_paths),
            "missing_group_markers": len(groups) - len(present_marker_paths),
            "unexpected_group_markers": len(found_marker_paths - set(expected_marker_paths)),
            "invalid_group_markers": invalid_group_markers,
        }
    return report


def launch(
    scope: str,
    *,
    release_sha256: str,
    max_concurrent: int,
    kinds: set[str],
    confirmation: str | None = None,
) -> None:
    release = _load_release(release_sha256)
    if scope == "full":
        _assert_full_launch_authorized(release, confirmation)
    configs = _pod_configs(scope)
    _audit_frozen_provenance(scope, release, configs)
    readiness = _audit_checkpoint_readiness(scope)
    if readiness["missing"]:
        raise RuntimeError(f"{scope} checkpoint precondition failed: {readiness}")
    steps: list[ArtifactStep[Artifact]] = []
    if "probe" in kinds:
        steps.extend(_probe_steps(scope, release=release, pod_configs=configs))
    if "optimizer" in kinds:
        steps.extend(_optimizer_steps(scope, release=release, pod_configs=configs))
    if "rollout" in kinds:
        steps.extend(_rollout_steps(scope, release=release, pod_configs=configs))
    run(*steps, max_concurrent=max_concurrent, force_run_failed=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--mode", choices=("readiness", "audit", "launch"), default="readiness")
    parser.add_argument("--kinds", default="probe,optimizer,rollout")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--confirm-full-launch")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    release = _load_release(args.release_sha256)
    if args.mode == "readiness":
        configs = _pod_configs(args.scope)
        print(
            json.dumps(
                {
                    "checkpoint_readiness": _audit_checkpoint_readiness(args.scope),
                    "frozen_provenance": _audit_frozen_provenance(args.scope, release, configs),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.mode == "audit":
        print(json.dumps(audit_outputs(args.scope), indent=2, sort_keys=True))
        return
    kinds = {kind.strip() for kind in args.kinds.split(",") if kind.strip()}
    unknown = kinds - {"probe", "optimizer", "rollout"}
    if unknown:
        raise ValueError(f"Unknown execution kinds: {sorted(unknown)}")
    default_concurrency = CANARY_SCOPE_MAX_CONCURRENT if args.scope == "canary" else FULL_SCOPE_MAX_CONCURRENT
    if args.scope == "full" and args.max_concurrent is None:
        raise ValueError("Full launch requires an explicit --max-concurrent")
    max_concurrent = args.max_concurrent or default_concurrency
    if max_concurrent < 1 or max_concurrent > default_concurrency:
        raise ValueError(f"max_concurrent must be in [1, {default_concurrency}] for {args.scope}")
    launch(
        args.scope,
        release_sha256=args.release_sha256,
        max_concurrent=max_concurrent,
        kinds=kinds,
        confirmation=args.confirm_full_launch,
    )


if __name__ == "__main__":
    main()
