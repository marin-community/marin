# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.store.cache import TreeCache
from marin.execution.artifact import write_artifact
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.grug.fast_track.add_dataset import (
    AddDatasetPreparationConfig,
    add_dataset_cache_step,
    prepare_add_dataset_cache,
)
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
)
from experiments.grug.fast_track.data_pipeline import add_prepared_dataset_component
from experiments.grug.fast_track.launch import V16384_TOKENIZER, AddDatasetTrainingSource


class _SmallTokenizer:
    name_or_path = "small-test-tokenizer"
    vocab_size = 32
    bos_token_id = 1
    eos_token_id = 2
    eos_token = "EOS"

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool) -> list[list[int]]:
        del add_special_tokens
        return [[3 + len(piece) % 10 for piece in text.split()] for text in texts]

    def encode(self, text: str) -> list[int]:
        return self.encode_batch([text], add_special_tokens=False)[0]


def _prefix(*, token_cap: int = 5, max_rows: int = 10) -> DatasetPrefix:
    return DatasetPrefix(
        repo="org/dataset",
        revision="a" * 40,
        subset="subset-a",
        split="train",
        text_field="body",
        tokenizer="hero-bpe-v16384",
        sampling_policy=AddDatasetSamplingPolicy.PREFIX,
        requested_token_cap=token_cap,
        max_rows=max_rows,
        max_overshoot_tokens=2,
    )


def _config(tmp_path, *, token_cap: int = 5, max_rows: int = 10) -> AddDatasetPreparationConfig:
    return AddDatasetPreparationConfig(
        prefix=_prefix(token_cap=token_cap, max_rows=max_rows),
        output_path=str(tmp_path / "prepared"),
    )


def test_prepare_add_dataset_cache_writes_and_loads_the_measured_prefix(tmp_path):
    rows = [
        {"body": "a b"},
        {"body": "c"},
        {"body": "d e f"},
    ]

    config = _config(tmp_path, token_cap=7)
    prepared = prepare_add_dataset_cache(config, rows=rows, tokenizer=_SmallTokenizer())

    assert prepared.actual_num_rows == 2
    assert prepared.actual_num_tokens == 7
    cache = TreeCache.load(
        str(tmp_path / "prepared" / "train"),
        {"input_ids": np.zeros((0,), dtype=np.int32)},
    )
    assert len(cache) == 2
    assert sum(len(row["input_ids"]) for row in cache) == 7

    artifact_path = str(tmp_path / "artifact-record")
    write_artifact(prepared, artifact_path)
    cache_step = ArtifactStep(
        name="prepared/test",
        version="2026.10.03",
        artifact_type=PreparedAddDatasetCache,
        run=lambda _config: None,
        build_config=lambda _ctx: None,
        override_path=artifact_path,
    )
    source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            prefix=config.prefix,
            fraction=0.2,
            target_production_tokens=1_000,
            available_unique_tokens=500,
        ),
        baseline=FrozenBaselineManifest(
            tokenizer=V16384_TOKENIZER,
            components=(FrozenBaselineComponent(name="base", cache_dir="base", weight=1.0),),
        ),
    )
    training_data = source.data_config(
        ctx=StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(cache_step,)),
        validation=(),
        tokenizer=V16384_TOKENIZER,
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1),
    )
    assert training_data.train_weights == pytest.approx({"base": 0.8, "add-dataset": 0.2})
    assert training_data.max_train_batches == {"add-dataset": 5}
    assert training_data.components["add-dataset"].cache_dir == prepared.cache_dir


def test_prepare_add_dataset_cache_stops_when_the_bounded_prefix_runs_out(tmp_path):
    with pytest.raises(ValueError, match="requires 5 tokens"):
        prepare_add_dataset_cache(
            _config(tmp_path, token_cap=5, max_rows=1),
            rows=[{"body": "one"}, {"body": "unread"}],
            tokenizer=_SmallTokenizer(),
        )


def test_add_dataset_cache_identity_changes_with_revision_and_prefix_limit(tmp_path):
    config = _config(tmp_path)
    original = add_dataset_cache_step(config=config, version="2026.10.03")

    revision_change = add_dataset_cache_step(
        config=replace(config, prefix=config.prefix.model_copy(update={"revision": "b" * 40})),
        version="2026.10.03",
    )
    prefix_change = add_dataset_cache_step(
        config=replace(
            config,
            prefix=config.prefix.model_copy(update={"requested_token_cap": config.prefix.requested_token_cap + 1}),
        ),
        version="2026.10.03",
    )

    assert original.name != revision_change.name
    assert original.name != prefix_change.name


def test_add_dataset_weights_keep_the_baseline_ratio_and_clear_simulation_limits():
    baseline = LmDataConfig(
        tokenizer="hero-bpe-v16384",
        components={
            "a": DatasetComponent(cache_dir="a", flat_cache=True),
            "b": DatasetComponent(cache_dir="b", flat_cache=True),
        },
        train_weights={"a": 3.0, "b": 1.0},
        target_budget=100,
        experiment_budget=200,
    )

    data = add_prepared_dataset_component(
        baseline,
        name="add-dataset",
        component=DatasetComponent(cache_dir="new", flat_cache=True),
        fraction=0.2,
        max_train_batches=4,
    )

    assert data.train_weights == pytest.approx({"a": 0.6, "b": 0.2, "add-dataset": 0.2})
    assert data.max_train_batches == {"add-dataset": 4}
    assert data.target_budget is None
    assert data.experiment_budget is None
