# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind complete SFT store epochs to measured native packing capacities."""

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from draccus import ChoiceRegistry
from levanter.data import packing
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset, allocate_mixture_counts, rescale_mixture_schedule_for_batch_schedule
from levanter.data.mixture_epoch import epoch_weight_stages, plan_mixture_epoch
from levanter.schedule import BatchSchedule
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop.run_spec import FilePin, read_pin

SFT_PREFIX = "sft/"


class SftStopping(ChoiceRegistry):
    count: int

    @classmethod
    def default_choice_name(cls) -> str:
        return "mixture_epochs"


@SftStopping.register_subclass("mixture_epochs")
@dataclass(frozen=True)
class MixtureEpochs(SftStopping):
    count: int = 1


@SftStopping.register_subclass("steps")
@dataclass(frozen=True)
class SftSteps(SftStopping):
    count: int


def packing_code_sha256() -> str:
    return hashlib.sha256(Path(packing.__file__).read_bytes()).hexdigest()


def validate_capacity(recipe: dict, capacity: dict) -> None:
    """Reject stale, incomplete or differently packed source-capacity receipts."""
    data = recipe["data"]
    catalog = StoragePath(data["cache_catalog"]).read_bytes()
    catalog_sources = json.loads(catalog)["splits"]["train"]
    expected = {
        "catalog_uri": data["cache_catalog"],
        "catalog_sha256": hashlib.sha256(catalog).hexdigest(),
        "sequence_length": recipe["seq_len"],
        "packing_cap": recipe["packing_cap"],
        "lookahead": recipe["pack_lookahead_window"],
        "packing_sha256": packing_code_sha256(),
        "validation_reservations": data["num_validation_sequences"],
    }
    for field, value in expected.items():
        if capacity.get(field) != value:
            raise ValueError(f"SFT capacity receipt disagrees with runtime field {field}")
    if set(capacity["sources"]) != set(data["components"]):
        raise ValueError("SFT capacity receipt does not cover every included component")
    for name, row in capacity["sources"].items():
        ledger = catalog_sources[name]["ledger"]
        if (
            not ledger["is_finished"]
            or row["documents"] != sum(ledger["shard_rows"].values())
            or row["tokens"] != ledger["field_counts"]["input_ids"]
            or row["clipped_tokens"] != 0
            or row["training_sequences"] <= 0
            or row["reserved_validation_sequences"] != (data["num_validation_sequences"] or {}).get(name, 0)
            or row["training_sequences"] + row["reserved_validation_sequences"] != row["packed_sequences"]
        ):
            raise ValueError(f"SFT source {name} is empty, incomplete, clipped or not fully provisioned")
    if any(data.get(field) is not None for field in ("max_train_batches", "experiment_budget", "target_budget")):
        raise ValueError("Full-store SFT refuses max_train_batches and budget-based data slices")
    if data.get("prior_context_phases") or data.get("mixture_start_step") is not None:
        raise ValueError("Full-store SFT starts each included source at its first epoch")


def epoch_exposure(
    sources: dict,
    stages: list[tuple[int, dict[str, float]]],
    *,
    batch_size: int,
    block_size: int,
    count: int,
    sequence_count: int,
) -> dict:
    """Check the actual integer allocation covers every SFT store exactly ``count`` times."""
    samples = {name: 0 for name in sources}
    for step, weights in stages:
        slots = min(block_size, sequence_count - step * batch_size)
        for name, value in allocate_mixture_counts(weights, slots).items():
            samples[name] += value
    rows = {}
    for name, source in sources.items():
        required = source["training_sequences"] * count if name.startswith(SFT_PREFIX) else None
        if samples[name] == 0 or (required is not None and samples[name] != required):
            raise ValueError(f"SFT epoch coverage fails for {name}: {samples[name]} samples, required {required}")
        rows[name] = {
            **source,
            "kind": "sft" if required is not None else "replay",
            "samples": samples[name],
            "required_samples": required,
            "context_positions": samples[name] * source["sequence_length"],
            "coverage": samples[name] / source["training_sequences"],
        }
    return {
        "stopping": {"type": "mixture_epochs", "count": count},
        "components": rows,
        "updates": sum(samples.values()) // batch_size,
        "block_size": block_size,
        "total_samples": sum(samples.values()),
        "full_sft_coverage": True,
        "basis": "planned_native_sampler_allocation",
    }


def bind_sft_stopping(recipe: dict, stopping: SftStopping, capacity_pin: FilePin | None) -> int:
    """Resolve an explicit step limit or a full-store epoch into a checked recipe."""
    if stopping.count <= 0:
        raise ValueError("SFT stopping count must be positive")
    if capacity_pin is None:
        raise ValueError("SFT preflight requires a pinned native epoch_capacity receipt")
    capacity = read_pin(capacity_pin, Path.cwd())
    validate_capacity(recipe, capacity)
    recipe["sft_capacity"] = {"pin": dataclasses.asdict(capacity_pin), **capacity}
    if isinstance(stopping, SftSteps):
        recipe["sft_exposure"] = step_exposure(recipe, capacity["sources"], stopping.count)
        return stopping.count
    counts = {
        name: row["training_sequences"] * stopping.count
        for name, row in capacity["sources"].items()
        if name.startswith(SFT_PREFIX)
    }
    if not counts:
        raise ValueError("A mixture epoch requires at least one SFT store")
    stages = recipe["data"]["train_weights"]
    replay_stages = [
        {name: weight for name, weight in weights.items() if not name.startswith(SFT_PREFIX)} for _, weights in stages
    ]
    fractions = [
        sum(replay.values()) / sum(weights.values()) for replay, (_, weights) in zip(replay_stages, stages, strict=True)
    ]
    if any(abs(value - fractions[0]) > 1e-12 for value in fractions):
        raise ValueError("A full-store epoch requires the declared replay share to be constant across phases")
    replay = replay_stages[0]
    for other in replay_stages[1:]:
        if any(abs(other[name] / sum(other.values()) - replay[name] / sum(replay.values())) > 1e-12 for name in replay):
            raise ValueError("Replay composition changes across the declared epoch")
    plan = plan_mixture_epoch(counts, batch_size=recipe["batch_size"], replay_fraction=fractions[0])
    weights = epoch_weight_stages(counts, replay, plan)
    receipt = epoch_exposure(
        capacity["sources"],
        weights,
        batch_size=recipe["batch_size"],
        block_size=plan.block_size,
        count=stopping.count,
        sequence_count=plan.run_sequences,
    )
    receipt.update(declared_replay_fraction=fractions[0], realized_replay_fraction=plan.replay_fraction)
    recipe["data"]["train_weights"] = weights
    recipe["data"]["mixture_block_size"] = plan.block_size
    recipe["data"]["mixture_length"] = plan.run_sequences
    recipe["sft_exposure"] = receipt
    return plan.steps


def step_exposure(recipe: dict, sources: dict, updates: int, *, data_key: jax.Array | None = None) -> dict:
    """Reconstruct the seed-specific integer source slots for an explicit step limit."""
    data = recipe["data"]
    schedule = BatchSchedule(recipe["batch_size"])
    if data_key is None:
        data_key = jax.random.PRNGKey(recipe["data_seed"])
    mix_key, _ = jax.random.split(data_key)
    weights = data["train_weights"]
    if isinstance(weights, list):
        weights = rescale_mixture_schedule_for_batch_schedule(weights, schedule)
    mixture = MixtureDataset(
        {name: ListAsyncDataset([None]) for name in data["components"]},
        weights,
        data["mixture_block_size"],
        key=mix_key,
        sequence_count=data.get("mixture_length"),
    )
    total = updates * recipe["batch_size"]
    counts = np.zeros(len(mixture.dataset_index), dtype=np.int64)
    unique = [set() for _ in mixture.dataset_index]
    for block in range((total + mixture.block_size - 1) // mixture.block_size):
        slots = min(mixture.block_size, total - block * mixture.block_size)
        encoded = mixture._get_block(block)[:slots]
        ids = encoded >> 16
        before = mixture.sequence_counts_before_block(block)
        counts += np.bincount(ids, minlength=len(counts))
        for index, name in enumerate(mixture.dataset_index):
            ordinals = (encoded[ids == index] & 0xFFFF) + before[name]
            unique[index].update((ordinals % sources[name]["training_sequences"]).tolist())
    return {
        "stopping": {"type": "steps", "count": updates},
        "basis": "seed_specific_native_sampler_reconstruction",
        "components": {
            name: {
                **sources[name],
                "samples": int(counts[i]),
                "unique_sequences": len(unique[i]),
                "coverage": int(counts[i]) / sources[name]["training_sequences"],
            }
            for i, name in enumerate(mixture.dataset_index)
        },
        "full_sft_coverage": all(
            len(unique[i]) == sources[name]["training_sequences"]
            for i, name in enumerate(mixture.dataset_index)
            if name.startswith(SFT_PREFIX)
        ),
    }
