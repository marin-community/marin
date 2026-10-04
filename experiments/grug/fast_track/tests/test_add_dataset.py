# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest
from click.testing import CliRunner
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.store.cache import TreeCache, write_levanter_cache
from marin.execution.artifact import write_artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import graph_handles
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.grug.fast_track.add_dataset import (
    AddDatasetPreparationConfig,
    AddDatasetTrainingSource,
    add_dataset_cache_step,
    add_prepared_dataset_component,
    prepare_add_dataset_cache,
)
from experiments.grug.fast_track.add_dataset_cli import main as add_dataset_cli
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import (
    V16384_TOKENIZER,
    FlatCacheTrainingSource,
    MatchMode,
    resolve_h100_ladder_budget,
)


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
        tokenizer_hash="sha256:test-tokenizer",
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


def test_prepare_add_dataset_cache_writes_and_loads_the_measured_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    rows = [
        {"body": "a b"},
        {"body": "c"},
        {"body": "d e f"},
    ]

    config = _config(tmp_path, token_cap=7)
    prepared = prepare_add_dataset_cache(
        config,
        rows=rows,
        tokenizer=_SmallTokenizer.name_or_path,
    )

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
        baseline=FlatCacheTrainingSource(
            manifest=FrozenBaselineManifest(
                tokenizer=V16384_TOKENIZER,
                components=(FrozenBaselineComponent(name="base", cache_dir="base", weight=1.0),),
            )
        ),
    )
    training_data = source.data_config(
        ctx=StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(cache_step,)),
        validation=(),
        tokenizer=V16384_TOKENIZER,
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1),
    )
    assert training_data.train_weights == pytest.approx({"base": 0.8, "add-dataset": 0.2})
    assert training_data.max_train_sequences == {"add-dataset": 5}
    assert training_data.components["add-dataset"].cache_dir == f"{prepared.cache_dir}/train"
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:changed-tokenizer"
    )
    with pytest.raises(ValueError, match="tokenizer content differs"):
        source.data_config(
            ctx=StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(cache_step,)),
            validation=(),
            tokenizer=V16384_TOKENIZER,
            budget=ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1),
        )


def test_prepared_prefix_reuses_one_cache_across_training_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    preparation = _config(tmp_path, token_cap=12)
    prepared = prepare_add_dataset_cache(
        preparation,
        rows=[{"body": "a b"} for _ in range(8)],
        tokenizer=_SmallTokenizer.name_or_path,
    )
    artifact_path = str(tmp_path / "prepared-artifact")
    write_artifact(prepared, artifact_path)
    cache_step = ArtifactStep(
        name="prepared/shared-budget-test",
        version="2026.10.03",
        artifact_type=PreparedAddDatasetCache,
        run=lambda _config: None,
        build_config=lambda _ctx: None,
        override_path=artifact_path,
    )
    source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            prefix=preparation.prefix,
            fraction=0.2,
            target_production_tokens=1_000,
            available_unique_tokens=500,
        ),
        baseline=FlatCacheTrainingSource(
            manifest=FrozenBaselineManifest(
                tokenizer=V16384_TOKENIZER,
                components=(FrozenBaselineComponent(name="base", cache_dir="base", weight=1.0),),
            )
        ),
    )
    context = StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(cache_step,))

    short_data = source.data_config(
        ctx=context,
        validation=(),
        tokenizer=V16384_TOKENIZER,
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1),
    )
    long_data = source.data_config(
        ctx=context,
        validation=(),
        tokenizer=V16384_TOKENIZER,
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=50, sequence_length=1),
    )

    assert short_data.components["add-dataset"].cache_dir == long_data.components["add-dataset"].cache_dir
    assert short_data.max_train_sequences == {"add-dataset": 5}
    assert long_data.max_train_sequences == {"add-dataset": 10}


def test_add_dataset_cli_prepare_only_writes_cache_loadable_by_python_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset_cli.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.load_dataset",
        lambda *args, **kwargs: [{"body": "a b"}, {"body": "c"}, {"body": "d e f"}],
    )
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    args = [
        "--repository",
        "org/dataset",
        "--revision",
        "a" * 40,
        "--subset",
        "subset-a",
        "--split",
        "train",
        "--text-field",
        "body",
        "--max-rows",
        "10",
        "--max-overshoot-tokens",
        "2",
        "--prepare-token-cap",
        "7",
        "--prepare-only",
        "--version",
        "2026.10.03",
        "--run",
    ]

    client = LocalClient()
    try:
        with set_current_client(client):
            result = CliRunner().invoke(add_dataset_cli, args)
    finally:
        client.shutdown()

    assert result.exit_code == 0, result.output
    step = add_dataset_cache_step(config=_config(tmp_path, token_cap=7), version="2026.10.03")
    prepared = PreparedAddDatasetCache.raw_load(step.path(str(tmp_path / "artifacts")))
    cache = TreeCache.load(
        f"{prepared.cache_dir}/train",
        {"input_ids": np.zeros((0,), dtype=np.int32)},
    )
    baseline_cache = str(tmp_path / "baseline-cache")
    write_levanter_cache(
        [{"input_ids": np.array([3, 4], dtype=np.int32)}],
        baseline_cache,
        metadata={"tokenizer": _SmallTokenizer.name_or_path, "format": "text", "text_field": "text"},
    )
    validation_root = tmp_path / "validation-artifact"
    write_levanter_cache(
        [{"input_ids": np.array([5, 6, 7], dtype=np.int32)}],
        str(validation_root / "validation"),
        metadata={"tokenizer": _SmallTokenizer.name_or_path, "format": "text", "text_field": "text"},
    )
    validation_step = ArtifactStep(
        name="baseline-validation",
        version="2026.10.03",
        artifact_type=TokenizedCache,
        run=lambda _config: None,
        build_config=lambda _ctx: None,
        override_path=str(validation_root),
    )
    training_source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=step,
            prefix=prepared.prefix,
            fraction=0.2,
            target_production_tokens=1_000,
            available_unique_tokens=500,
        ),
        baseline=FlatCacheTrainingSource(
            manifest=FrozenBaselineManifest(
                tokenizer=V16384_TOKENIZER,
                components=(FrozenBaselineComponent(name="base", cache_dir=baseline_cache, weight=1.0),),
            )
        ),
    )
    monkeypatch.setattr("levanter.data.text.datasets.load_marin_tokenizer", lambda _name: _SmallTokenizer())
    training_config = training_source.data_config(
        ctx=StepContext.for_run(
            output_path="unused",
            prefix=str(tmp_path / "artifacts"),
            deps=(step, validation_step),
        ),
        validation=(validation_step,),
        tokenizer=V16384_TOKENIZER,
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1),
    )
    train_caches = training_config.build_caches("train")
    validation_caches = training_config.build_caches("validation")
    assert prepared.actual_num_rows == 2
    assert prepared.actual_num_tokens == 7
    assert len(cache) == 2
    assert sum(len(row["input_ids"]) for row in cache) == 7
    assert set(train_caches) == {"base", "add-dataset"}
    assert train_caches["add-dataset"].flat_field_length("input_ids") == 7
    assert set(validation_caches) == {validation_step.name}
    assert training_config.train_weights[validation_step.name] == 0


def test_add_dataset_cli_reports_invalid_revision_as_usage_error(monkeypatch):
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset_cli.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    result = CliRunner().invoke(
        add_dataset_cli,
        [
            "--repository",
            "org/dataset",
            "--revision",
            "main",
            "--split",
            "train",
            "--text-field",
            "body",
            "--max-rows",
            "10",
            "--prepare-token-cap",
            "5",
            "--prepare-only",
            "--version",
            "2026.10.03",
        ],
    )

    assert result.exit_code == 2
    assert "revision must be" in result.output
    assert "immutable Hugging Face commit hash" in result.output
    assert "Traceback" not in result.output


def test_add_dataset_cli_accepts_prepared_artifacts_without_hugging_face_options(tmp_path):
    preparation = _config(tmp_path, token_cap=16_384)
    prepared_path = str(tmp_path / "prepared-artifact")
    write_artifact(
        PreparedAddDatasetCache(
            cache_dir=str(tmp_path / "prepared-cache"),
            prefix=preparation.prefix,
            actual_num_rows=3,
            actual_num_tokens=16_384,
            tokenization_policy="fast-track-long-string-v1",
        ),
        prepared_path,
    )
    args = [
        "--run-id",
        "prepared-cache-run",
        "--baseline-artifact",
        "s3://test/hero-sample",
        "--prepared-cache",
        prepared_path,
        "--fraction",
        "0.99",
        "--available-unique-tokens",
        "16384",
        "--target-production-tokens",
        "361758720",
        "--version",
        "2026.10.04",
    ]

    result = CliRunner().invoke(add_dataset_cli, args)

    assert result.exit_code == 0, result.output
    assert "prepared-cache" in result.output


def test_add_dataset_cli_prepares_maximum_prefix_once_for_smaller_rungs(monkeypatch):
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset_cli.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    handles = []
    monkeypatch.setattr("marin.experiment.cli._print_plan", lambda roots: handles.extend(roots))
    common_args = [
        "--run-id",
        "maximum-prefix",
        "--baseline-artifact",
        "s3://test/hero-sample",
        "--moe",
        "--repository",
        "org/dataset",
        "--revision",
        "a" * 40,
        "--split",
        "train",
        "--text-field",
        "body",
        "--max-rows",
        "1000000",
        "--fraction",
        "0.99",
        "--available-unique-tokens",
        "1000000000",
        "--version",
        "2026.10.04",
    ]

    plans = [CliRunner().invoke(add_dataset_cli, ["--size", size, *common_args]) for size in ("d512", "d1280")]

    assert all(plan.exit_code == 0 for plan in plans), [plan.output for plan in plans]
    caches = [
        next(step for step in graph_handles([root]) if step.artifact_type is PreparedAddDatasetCache) for root in handles
    ]
    configs = [cache.build_config(StepContext.for_fingerprint((), cache.deps)) for cache in caches]
    largest = resolve_h100_ladder_budget(
        size="d1280", dense=False, match=MatchMode.DATA, num_steps=None, batch_size=None
    )
    required_cap = unique_token_sample_cap(
        target_production_tokens=18_750_000_000_000,
        fast_track_budget=largest.token_count,
        available_unique_tokens=1_000_000_000,
        fraction=0.99,
        sequence_length=largest.sequence_length,
    )
    assert configs[0].prefix.requested_token_cap == configs[1].prefix.requested_token_cap == required_cap
    assert caches[0].name == caches[1].name


def test_add_dataset_cli_reports_minimum_unique_tokens_for_one_sequence():
    result = CliRunner().invoke(
        add_dataset_cli,
        [
            "--run-id",
            "small-unique-data",
            "--baseline-artifact",
            "s3://test/hero-sample",
            "--repository",
            "org/dataset",
            "--revision",
            "a" * 40,
            "--split",
            "train",
            "--text-field",
            "body",
            "--max-rows",
            "1000",
            "--fraction",
            "0.99",
            "--available-unique-tokens",
            "100000000",
            "--version",
            "2026.10.04",
        ],
    )

    assert result.exit_code == 2
    assert "--available-unique-tokens must be at least" in result.output
    assert "to supply one training sequence" in result.output


def test_prepare_add_dataset_cache_stops_when_the_bounded_prefix_runs_out(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    with pytest.raises(ValueError, match="requires 5 tokens"):
        prepare_add_dataset_cache(
            _config(tmp_path, token_cap=5, max_rows=1),
            rows=[{"body": "one"}, {"body": "unread"}],
            tokenizer=_SmallTokenizer.name_or_path,
        )


def test_long_document_prefix_preserves_tokens_and_defers_oversized_lookahead(tmp_path, monkeypatch):
    tokenizer = _SmallTokenizer()
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset.tokenizer_content_hash", lambda _name: "sha256:test-tokenizer"
    )
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.TOKENIZATION_MAX_DOCUMENT_BYTES", 200_000)
    texts = ["first", "hello 世界\n" * 10_000, "last"]
    expected = [[1, *tokenizer.encode(text + " EOS")] for text in texts]
    token_cap = sum(map(len, expected))
    config = _config(tmp_path, token_cap=token_cap)
    oversized = {"body": "x" * 200_001}
    prepared = prepare_add_dataset_cache(
        config,
        rows=[*({"body": text} for text in texts), oversized],
        tokenizer=tokenizer.name_or_path,
    )
    cache = TreeCache.load(prepared.cache_dir + "/train", {"input_ids": np.zeros((0,), dtype=np.int32)})
    assert [row["input_ids"].tolist() for row in cache] == expected
    with pytest.raises(ValueError, match="org/dataset row 3 has 200,001 UTF-8 bytes"):
        prepare_add_dataset_cache(
            replace(config, output_path=str(tmp_path / "too-long"), prefix=_prefix(token_cap=token_cap + 1)),
            rows=[*({"body": text} for text in texts), oversized],
            tokenizer=tokenizer.name_or_path,
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
    tokenizer_change = add_dataset_cache_step(
        config=replace(
            config,
            prefix=config.prefix.model_copy(update={"tokenizer_hash": "sha256:other-tokenizer"}),
        ),
        version="2026.10.03",
    )

    assert original.name != revision_change.name
    assert original.name != prefix_change.name
    assert original.name != tokenizer_change.name


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
        max_train_sequences=4,
    )

    assert data.train_weights == pytest.approx({"a": 0.6, "b": 0.2, "add-dataset": 0.2})
    assert data.max_train_sequences == {"add-dataset": 4}
    assert data.target_budget is None
    assert data.experiment_budget is None


def test_add_prepared_dataset_component_preserves_other_sequence_limits():
    baseline = LmDataConfig(
        tokenizer="hero-bpe-v16384",
        components={"a": DatasetComponent(cache_dir="a", flat_cache=True)},
        train_weights={"a": 1.0},
        max_train_sequences={"a": 2},
    )

    data = add_prepared_dataset_component(
        baseline,
        name="add-dataset",
        component=DatasetComponent(cache_dir="new", flat_cache=True),
        fraction=0.2,
        max_train_sequences=4,
    )

    assert data.max_train_sequences == {"a": 2, "add-dataset": 4}


def test_add_prepared_dataset_component_rejects_fraction_absent_from_mixture_blocks():
    baseline = LmDataConfig(
        tokenizer="hero-bpe-v16384",
        components={"a": DatasetComponent(cache_dir="a", flat_cache=True)},
        train_weights={"a": 1.0},
    )

    with pytest.raises(ValueError, match=r"at least 0\.000488281"):
        add_prepared_dataset_component(
            baseline,
            name="add-dataset",
            component=DatasetComponent(cache_dir="new", flat_cache=True),
            fraction=1e-6,
            max_train_sequences=1,
        )
