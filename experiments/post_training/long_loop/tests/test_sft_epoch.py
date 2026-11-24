# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from fray.types import GpuConfig, ResourceConfig
from haliax import Axis
from levanter import training_budget
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from levanter.training_budget import TimeLimitedIterator, TrainingBudgetConfig, TrainingStopReason
from levanter.utils.thread_utils import blocking_wait
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.expert_placement import ExpertPlacement
from experiments.grug.moe_hero_ep.launch_diagnostics import build_native_diagnostic_run
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import GrugRunConfig, GrugTrainerConfig, RouterBiasMode, SpecialTokenInit
from experiments.post_training.long_loop.execution_policy import (
    SFT_FINALIZATION_RESERVE,
    SFT_TRAINING_DURATION,
    SFT_UPDATE_RESERVE,
)
from experiments.post_training.long_loop.mochi_sft import (
    MochiSftControls,
    MochiSftInitializer,
    MochiSftRound,
    MochiSftSchedule,
    MochiWeightsInitializer,
    bind_mochi_sft_config,
)
from experiments.post_training.long_loop.round_runner import RoundSft, load_round
from experiments.post_training.long_loop.run_spec import FilePin, StorageConfig
from experiments.post_training.long_loop.sft_epoch import (
    MixtureEpochs,
    bind_sft_stopping,
    epoch_exposure,
    packing_code_sha256,
    step_exposure,
)
from experiments.post_training.long_loop.shared_sft import validate_epoch_completion


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
            "components": {name: {} for name in counts},
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
    assert controls.training_budget.duration == SFT_TRAINING_DURATION and controls.stage_guard is None
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
    row.update(
        training_sequences=row["packed_sequences"] - 8,
        reserved_validation_sequences=8,
        training_tokens=row["tokens"] - 8 * 16,
    )
    raw = json.dumps(capacity).encode()
    Path(pin.path).write_bytes(raw)
    bind_sft_stopping(recipe, MixtureEpochs(), FilePin(pin.path, hashlib.sha256(raw).hexdigest()))
    exposure = recipe["sft_exposure"]["components"]["sft/large"]
    assert exposure["samples"] == 47_991
    assert exposure["reserved_validation_sequences"] == 8
    assert exposure["required_samples"] == exposure["training_sequences"]


def test_round_yaml_omitting_stopping_selects_one_epoch_with_the_shared_time_cap(tmp_path):
    source = Path(__file__).parents[1] / "rounds/exp5-setup.yaml"
    document = yaml.safe_load(source.read_text())
    for field in ("stopping", "training_budget", "stage_guard", "wall_clock_cooldown_fraction"):
        document["sft"].pop(field, None)
    path = tmp_path / "round.yaml"
    path.write_text(yaml.safe_dump(document))
    spec = load_round(path)
    assert spec.sft.stopping == MixtureEpochs(1)
    assert spec.sft.training_budget.duration == SFT_TRAINING_DURATION
    assert spec.sft.training_budget.finalization_reserve == SFT_FINALIZATION_RESERVE
    assert spec.sft.training_budget.update_reserve == SFT_UPDATE_RESERVE
    assert spec.sft.task_timeout_seconds == SFT_TRAINING_DURATION.total_seconds()


@pytest.mark.parametrize("time_bound", [False, True])
def test_default_epoch_stops_at_epoch_or_clock_whichever_is_first(epoch_inputs, monkeypatch, time_bound):
    recipe, _, pin = epoch_inputs
    controls = RoundSft(
        batch_size=256, min_lr_ratio=0.1, learning_rate=1e-4, adam_learning_rate=1e-4, mesh={}, resources={}
    )
    updates = bind_sft_stopping(recipe, controls.stopping, pin)
    clock = [0.0]
    monkeypatch.setattr(training_budget, "monotonic", lambda: clock[0])
    monkeypatch.setattr(training_budget, "time", lambda: 1000 + clock[0])
    budget = replace(
        controls.training_budget, deadline_epoch_seconds=1000 + controls.training_budget.duration.total_seconds()
    )
    batches = TimeLimitedIterator(range(updates), budget)
    advance = budget.training_duration.total_seconds() if time_bound else 1
    consumed = []
    for _ in range(updates):
        try:
            consumed.append(next(batches))
        except StopIteration:
            break
        clock[0] += advance
    assert consumed == list(range(1 if time_bound else updates))
    assert batches.stop_reason == (TrainingStopReason.TIME_BOUND if time_bound else None)


def test_round_cannot_disable_the_shared_sft_time_cap(tmp_path):
    document = yaml.safe_load((Path(__file__).parents[1] / "rounds/exp5-setup.yaml").read_text())
    document["sft"]["training_budget"] = None
    source = tmp_path / "disabled-clock.yaml"
    source.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="cannot disable its training budget"):
        load_round(source)


@pytest.mark.parametrize(
    "complete,reason",
    [
        (True, TrainingStopReason.MAXIMUM_UPDATES),
        (False, TrainingStopReason.TIME_BOUND),
        (False, TrainingStopReason.DATA_EXHAUSTED),
        (False, TrainingStopReason.MAXIMUM_UPDATES),
    ],
)
def test_epoch_completion_accepts_only_coverage_or_certified_time_limit(epoch_inputs, complete, reason):
    recipe, _, pin = epoch_inputs
    updates = bind_sft_stopping(recipe, MixtureEpochs(), pin)
    actual = step_exposure(recipe, recipe["sft_capacity"]["sources"], updates if complete else 1)
    actual.update(epoch_target_complete=complete, stop_reason=reason)
    if not complete and reason != TrainingStopReason.TIME_BOUND:
        with pytest.raises(ValueError, match="without a native time-bound completion"):
            validate_epoch_completion(actual)
        return
    validate_epoch_completion(actual)
    assert actual["full_sft_coverage"] == complete


@pytest.mark.parametrize("target_updates,decay_steps", [(30, 3), (300, 30)])
def test_native_hero_restore_starts_the_new_epoch_and_scales_its_lr_tail(epoch_inputs, target_updates, decay_steps):
    recipe, _, pin = epoch_inputs
    recipe["batch_size"] = 176
    recipe["data"]["tokenizer"] = "gpt2"
    updates = bind_sft_stopping(recipe, MixtureEpochs(), pin)
    budget = TrainingBudgetConfig(SFT_TRAINING_DURATION, SFT_FINALIZATION_RESERVE, SFT_UPDATE_RESERVE)
    spec = MochiSftRound(
        1,
        "mochi_sft",
        "mochi/sft",
        "2026.11.23.127",
        "mochi-sft",
        "cw-us-east-08a",
        StorageConfig("s3://marin-us-east-08a/new", "s3://marin-us-east-08a/tmp"),
        pin,
        pin,
        pin,
        MochiSftInitializer("base", "s3://marin-us-east-08a/base", pin.path, "0" * 64),
        MochiSftControls(
            128, 176, 32, MochiSftSchedule(target_updates, 0, decay_steps), MixtureEpochs(), pin, budget, 72000
        ),
    )
    previous = GrugRunConfig(
        model=GrugModelConfig(vocab_size=64, max_seq_len=16, num_layers=1, num_experts=192),
        data=LmDataConfig(
            tokenizer="gpt2",
            components={"old": DatasetComponent(), "val": DatasetComponent(split="validation")},
            train_weights=[(0, {"old": 1.0})],
        ),
        optimizer=GrugMoeMuonHConfig(warmup=0, decay=692, lr_schedule="linear", min_lr_ratio=0.1),
        resources=ResourceConfig(device=GpuConfig("GB200", count=4), replicas=176),
        trainer=GrugTrainerConfig(
            trainer=TrainerConfig(train_batch_size=176, num_train_steps=6956),
            save_checkpoints=True,
            router_bias_mode=RouterBiasMode.FROZEN,
            expert_axis_size=16,
            context_axis_size=4,
            replica_axis_size=11,
            special_tokens=SpecialTokenInit((5,), (1,), 40, 2.0),
            expert_placement=ExpertPlacement("base", 16, 12, (tuple(range(192)),), "fixture"),
        ),
        stop_after_steps=6956,
    )
    reference = {"config": draccus.encode(previous)}
    reference["config"]["trainer"]["trainer"]["mp"] = {
        "param_dtype": {"__dtype__": "float32"},
        "compute_dtype": {"__dtype__": "bfloat16"},
        "output_dtype": {"__dtype__": "bfloat16"},
    }
    config = bind_mochi_sft_config(spec, reference, recipe["data"], updates=updates, output="s3://marin-us-east-08a/new")
    config.trainer.trainer.validate_batch_sizes(32)
    assert config.trainer.trainer.train_batch_size == 176
    assert config.trainer.trainer.per_device_parallelism == 6
    assert config.stop_after_steps == config.trainer.trainer.num_train_steps == 40 + target_updates
    assert config.optimizer == replace(previous.optimizer, decay=decay_steps)
    for peak in (previous.optimizer.learning_rate, previous.optimizer.adam_lr):
        lr = config.optimizer.lr_scheduler(config.trainer.trainer.num_train_steps, override_lr=peak)
        decay_start = 40 + target_updates - decay_steps
        assert float(lr(40)) == pytest.approx(peak)
        assert float(lr(decay_start)) == pytest.approx(peak)
        assert float(lr(decay_start + decay_steps / 2)) == pytest.approx(peak * 0.55)
        assert float(lr(40 + target_updates)) == pytest.approx(peak * 0.1)
    components = {
        name: DirectDatasetComponent(
            datasets={
                "train": ListAsyncDataset([GrugLmExample(tokens=jnp.full((16,), index), loss_weight=jnp.ones(16))])
            }
        )
        for index, name in enumerate(recipe["data"]["components"])
    }
    data = replace(config.data, components=components, cache_catalog=None, shuffle=False)
    relative = replace(data, mixture_start_step=None)
    schedule = BatchSchedule(176)
    native = data.train_mixture(Axis("position", 16), schedule, key=jax.random.PRNGKey(0))
    expected = relative.train_mixture(Axis("position", 16), schedule, key=jax.random.PRNGKey(0))
    start = schedule.global_data_offset_by_step(40)
    actual_batch = blocking_wait(native.get_batch(range(start, start + 176)))
    expected_batch = blocking_wait(expected.get_batch(range(176)))
    for actual, wanted in zip(actual_batch, expected_batch, strict=True):
        np.testing.assert_array_equal(actual.tokens, wanted.tokens)
    assert native.first_readable_index() == start
    diagnostic = build_native_diagnostic_run(config, run_id="memory-proof", num_steps=42, version="dev")
    probe = diagnostic.build_config(StepContext.for_fingerprint(runtime_arg_keys=(), deps=()))
    assert probe.model == config.model
    assert probe.data == config.data
    assert probe.optimizer == config.optimizer
    assert probe.resources == config.resources
    assert probe.trainer == replace(config.trainer, trainer=probe.trainer.trainer, save_checkpoints=False)
    assert probe.trainer.trainer.num_train_steps == config.trainer.trainer.num_train_steps
    assert probe.trainer.trainer.train_batch_size == 176
    assert probe.trainer.trainer.load_checkpoint_path == [spec.initializer.destination]
    assert probe.trainer.trainer.training_budget is None
    assert probe.trainer.trainer.record_device_memory
    assert probe.stop_step == 42
    assert probe.eval is None
    # A bounded probe stops early without re-scaling the production LR or resetting its data cursor.
    assert probe.data.mixture_start_step == 40
    assert config.trainer.save_checkpoints
    assert config.stop_step == 40 + target_updates
    actual_exposure = step_exposure(recipe, recipe["sft_capacity"]["sources"], updates)
    assert actual_exposure["full_sft_coverage"]
    assert set(config.evaluation_data.components) == {"val"}


def test_fresh_bankadam_round_preserves_batch_mesh_optimizer_and_weights_only_init(epoch_inputs):
    recipe, _, pin = epoch_inputs
    recipe["data"]["tokenizer"] = "gpt2"
    baseline = GrugRunConfig(
        model=GrugModelConfig(vocab_size=64, max_seq_len=16, num_layers=1, num_experts=192),
        data=LmDataConfig(tokenizer="gpt2"),
        optimizer=GrugMoeMuonHConfig(
            learning_rate=1.4030739143362721e-5,
            adam_lr=3.2378366391493846e-6,
            muonh_param_ndims=(2, 3),
            warmup=5,
            decay=3400,
            lr_schedule="linear",
            min_lr_ratio=0.1,
        ),
        resources=ResourceConfig(device=GpuConfig("GB200", count=4), replicas=32),
        trainer=GrugTrainerConfig(
            trainer=TrainerConfig(train_batch_size=32, num_train_steps=34000, initialize_from="base"),
            save_checkpoints=True,
            offload_opt_state=True,
            router_bias_mode=RouterBiasMode.FROZEN,
            expert_axis_size=16,
            context_axis_size=4,
            replica_axis_size=1,
            special_tokens=SpecialTokenInit((5,), (1,), 0, 5.656854249492381),
            expert_placement=ExpertPlacement("base", 16, 12, (tuple(range(192)),), "fixture"),
        ),
        eval=None,
        stop_after_steps=1000,
    )
    reference = {"config": draccus.encode(baseline)}
    reference["config"]["trainer"]["trainer"]["mp"] = {
        "param_dtype": {"__dtype__": "float32"},
        "compute_dtype": {"__dtype__": "bfloat16"},
        "output_dtype": {"__dtype__": "bfloat16"},
    }
    budget = TrainingBudgetConfig(SFT_TRAINING_DURATION, SFT_FINALIZATION_RESERVE, SFT_UPDATE_RESERVE)
    spec = MochiSftRound(
        1,
        "mochi_sft",
        "mochi/sft",
        "2026.11.24.128",
        "mochi-bankadam",
        "cw-us-east-08a",
        StorageConfig("s3://marin-us-east-08a/new", "s3://marin-us-east-08a/tmp"),
        pin,
        pin,
        pin,
        MochiWeightsInitializer("base", "0" * 64, 393491),
        MochiSftControls(128, 32, None, MochiSftSchedule(2100, 5, 210), MixtureEpochs(), pin, budget, 72000),
    )
    actual = bind_mochi_sft_config(spec, reference, recipe["data"], updates=20000, output="s3://marin-us-east-08a/new")
    assert actual.optimizer == replace(baseline.optimizer, decay=210)
    assert actual.trainer.replica_axis_size == 1
    assert actual.trainer.microbatch_size is None
    assert actual.trainer.data_seed is None
    assert actual.trainer.expert_placement == baseline.trainer.expert_placement
    assert actual.model == baseline.model
    assert actual.trainer.offload_opt_state
    assert actual.trainer.trainer.train_batch_size == 32
    assert actual.trainer.trainer.initialize_from == "base"
    assert actual.trainer.trainer.load_checkpoint_path is None
    assert actual.trainer.trainer.load_checkpoint is None
    assert not actual.trainer.trainer.allow_nondivisible_batch_size
    assert actual.trainer.trainer.num_train_steps == actual.stop_after_steps == 2100
    assert actual.eval is None
    assert actual.trainer.trainer.tracker == baseline.trainer.trainer.tracker
    lr = actual.optimizer.lr_scheduler(2100)
    assert float(lr(1890)) == pytest.approx(baseline.optimizer.learning_rate)
    assert float(lr(2100)) == pytest.approx(baseline.optimizer.learning_rate * 0.1)
