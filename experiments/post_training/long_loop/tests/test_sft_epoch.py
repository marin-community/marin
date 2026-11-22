# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.post_training.long_loop.round_runner import RoundSft, load_round
from experiments.post_training.long_loop.run_spec import FilePin
from experiments.post_training.long_loop.sft_epoch import (
    MixtureEpochs,
    bind_sft_stopping,
    epoch_exposure,
    packing_code_sha256,
)


@pytest.fixture
def epoch_inputs(tmp_path):
    counts = {"sft/large": 47_999, "sft/tiny": 1, "pretrain/large": 100_000, "pretrain/tiny": 1}
    sources = {
        name: {
            "sequence_length": 16,
            "documents": n,
            "tokens": n * 16,
            "packed_sequences": n,
            "training_sequences": n,
            "training_tokens": n * 16,
            "clipped_tokens": 0,
            "reserved_validation_sequences": 0,
        }
        for name, n in counts.items()
    }
    catalog = {
        "splits": {
            "train": {
                name: {
                    "ledger": {
                        "is_finished": True,
                        "shard_rows": {"shard": n},
                        "field_counts": {"input_ids": n * 16},
                    }
                }
                for name, n in counts.items()
            }
        }
    }
    raw_catalog = json.dumps(catalog).encode()
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_bytes(raw_catalog)
    capacity = {
        "catalog_uri": str(catalog_path),
        "catalog_sha256": hashlib.sha256(raw_catalog).hexdigest(),
        "sequence_length": 16,
        "packing_cap": 1,
        "lookahead": 1,
        "packing_sha256": packing_code_sha256(),
        "validation_reservations": {},
        "sources": sources,
    }
    capacity_path = tmp_path / "capacity.json"
    raw = json.dumps(capacity).encode()
    capacity_path.write_bytes(raw)
    weights = {"sft/large": 0.74999, "sft/tiny": 0.00001, "pretrain/large": 0.2499999, "pretrain/tiny": 0.0000001}
    recipe = {
        "data": {
            "cache_catalog": str(catalog_path),
            "components": dict.fromkeys(counts, {}),
            "num_validation_sequences": {},
            "train_weights": [(0, weights)],
            "mixture_block_size": 64_000,
        },
        "seq_len": 16,
        "packing_cap": 1,
        "pack_lookahead_window": 1,
        "batch_size": 256,
        "data_seed": 0,
    }
    return recipe, capacity, FilePin(str(capacity_path), hashlib.sha256(raw).hexdigest())


def test_default_epoch_covers_complete_sources_and_derives_steps_from_capacity(epoch_inputs):
    recipe, _, pin = epoch_inputs
    controls = RoundSft(
        batch_size=256, min_lr_ratio=0.1, learning_rate=1e-4, adam_learning_rate=1e-4, mesh={}, resources={}
    )
    updates = bind_sft_stopping(recipe, controls.stopping, pin)
    receipt = recipe["sft_exposure"]
    assert controls.training_budget is None and controls.stage_guard is None
    assert updates == 250
    assert receipt["components"]["sft/large"]["samples"] == 47_999
    assert receipt["components"]["sft/tiny"]["samples"] == 1
    assert receipt["components"]["pretrain/tiny"]["samples"] >= 1
    assert receipt["full_sft_coverage"]


def test_preflight_refuses_a_plan_that_omits_the_small_sft_store(epoch_inputs):
    recipe, capacity, pin = epoch_inputs
    bind_sft_stopping(recipe, MixtureEpochs(), pin)
    stages = copy.deepcopy(recipe["data"]["train_weights"])
    stage = stages[0][1]
    stage["pretrain/large"] += stage.pop("sft/tiny")
    with pytest.raises(ValueError, match="coverage fails for sft/tiny"):
        epoch_exposure(capacity["sources"], stages, batch_size=256, block_size=64_000, count=1, sequence_count=64_000)


def test_preflight_refuses_subsampled_training_data_even_with_a_valid_capacity_pin(epoch_inputs):
    recipe, _, pin = epoch_inputs
    recipe["data"]["max_train_batches"] = {"sft/large": 1}
    with pytest.raises(ValueError, match="refuses max_train_batches"):
        bind_sft_stopping(recipe, MixtureEpochs(), pin)


def test_epoch_keeps_explicit_iid_reservations_out_of_the_training_quota(epoch_inputs):
    recipe, capacity, pin = epoch_inputs
    recipe["data"]["num_validation_sequences"] = {"sft/large": 8}
    capacity["validation_reservations"] = {"sft/large": 8}
    row = capacity["sources"]["sft/large"]
    row.update(training_sequences=row["packed_sequences"] - 8, reserved_validation_sequences=8,
               training_tokens=row["tokens"] - 8 * 16)
    raw = json.dumps(capacity).encode()
    Path(pin.path).write_bytes(raw)
    bind_sft_stopping(recipe, MixtureEpochs(), FilePin(pin.path, hashlib.sha256(raw).hexdigest()))
    exposure = recipe["sft_exposure"]["components"]["sft/large"]
    assert exposure["samples"] == 47_991
    assert exposure["reserved_validation_sequences"] == 8
    assert exposure["required_samples"] == exposure["training_sequences"]


def test_round_yaml_omitting_stopping_selects_one_epoch_without_an_inherited_clock(tmp_path):
    source = Path(__file__).parents[1] / "rounds/exp5-setup.yaml"
    document = yaml.safe_load(source.read_text())
    for field in ("stopping", "training_budget", "stage_guard", "wall_clock_cooldown_fraction"):
        document["sft"].pop(field, None)
    path = tmp_path / "round.yaml"
    path.write_text(yaml.safe_dump(document))
    spec = load_round(path)
    assert spec.sft.stopping == MixtureEpochs(1)
    assert spec.sft.training_budget is None
