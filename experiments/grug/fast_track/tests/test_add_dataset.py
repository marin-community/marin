# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace

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
    TOKENIZATION_POLICY,
    AddDatasetConfig,
    DatasetPrefix,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.hero_sample import (
    HERO_LOADER_POLICY,
    HeroTrainingSource,
    PreparedHeroSample,
    _mixture_component_examples,
    hero_sample_manifest,
)
from experiments.grug.fast_track.launch import (
    V16384_TOKENIZER,
    FlatCacheTrainingSource,
    MatchMode,
    resolve_h100_ladder_budget,
)
from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE

TOKENIZER_HASH = "sha256:test-tokenizer"


@dataclass(frozen=True)
class _UnlimitedBaseline(FlatCacheTrainingSource):
    """A flat-cache baseline that has no measured capacity."""

    def mixture_config(self, **kwargs) -> LmDataConfig:
        return self.data_config(**kwargs)

    def require_capacity(self, data: LmDataConfig, *, ctx: StepContext, budget: ResolvedTrainingBudget) -> None:
        del data, ctx, budget


@pytest.fixture
def stable_tokenizer(monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.contracts.tokenizer_content_hash", lambda _name: TOKENIZER_HASH)
    monkeypatch.setattr(
        "experiments.grug.fast_track.add_dataset_cli.tokenizer_content_hash", lambda _name: TOKENIZER_HASH
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
        tokenizer_hash=TOKENIZER_HASH,
        requested_token_cap=token_cap,
        max_rows=max_rows,
    )


def _config(tmp_path, *, token_cap: int = 5, max_rows: int = 10) -> AddDatasetPreparationConfig:
    return AddDatasetPreparationConfig(
        prefix=_prefix(token_cap=token_cap, max_rows=max_rows),
        output_path=str(tmp_path / "prepared"),
    )


@dataclass(frozen=True)
class _PreparedSource:
    prepared: PreparedAddDatasetCache
    source: AddDatasetTrainingSource
    context: StepContext


@pytest.fixture
def prepared_source(tmp_path, monkeypatch, stable_tokenizer) -> _PreparedSource:
    """A 12-token prefix of three rows, at 20% of a baseline that cannot run out."""
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    prepared = prepare_add_dataset_cache(
        _config(tmp_path, token_cap=12),
        rows=[{"body": "a b"} for _ in range(8)],
        tokenizer=_SmallTokenizer.name_or_path,
    )
    artifact_path = str(tmp_path / "prepared-artifact")
    write_artifact(prepared, artifact_path)
    cache_step = ArtifactStep.adopt("prepared/test", "2026.10.03", artifact_path, kind=PreparedAddDatasetCache)
    source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            fraction=0.2,
            target_production_tokens=1_000,
            available_unique_tokens=500,
        ),
        baseline=_UnlimitedBaseline(
            manifest=FrozenBaselineManifest(
                tokenizer=V16384_TOKENIZER,
                components=(FrozenBaselineComponent(name="base", cache_dir="base", weight=1.0),),
            )
        ),
    )
    context = StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(cache_step,))
    return _PreparedSource(prepared, source, context)


def test_prepare_add_dataset_cache_writes_and_loads_the_measured_prefix(prepared_source, monkeypatch):
    prepared = prepared_source.prepared
    assert prepared.actual_num_rows == 3
    assert prepared.actual_num_tokens == 12
    cache = TreeCache.load(f"{prepared.cache_dir}/train", {"input_ids": np.zeros((0,), dtype=np.int32)})
    assert [row["input_ids"].tolist() for row in cache] == [[1, *_SmallTokenizer().encode("a b EOS")]] * 3

    budget = ResolvedTrainingBudget(batch_size=1, num_steps=25, sequence_length=1)
    training_data = prepared_source.source.data_config(
        ctx=prepared_source.context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget
    )
    assert training_data.train_weights == pytest.approx({"base": 0.8, "add-dataset": 0.2})
    assert training_data.max_train_sequences == {"add-dataset": 5}
    assert training_data.components["add-dataset"].cache_dir == f"{prepared.cache_dir}/train"

    monkeypatch.setattr("experiments.grug.fast_track.contracts.tokenizer_content_hash", lambda _name: "sha256:changed")
    with pytest.raises(ValueError, match="tokenizer 'hero-bpe-v16384' content is sha256:changed"):
        prepared_source.source.data_config(
            ctx=prepared_source.context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget
        )


def test_prepared_prefix_reuses_one_cache_across_training_budgets(prepared_source):
    short_data, long_data = (
        prepared_source.source.data_config(
            ctx=prepared_source.context,
            validation=(),
            tokenizer=V16384_TOKENIZER,
            budget=ResolvedTrainingBudget(batch_size=1, num_steps=num_steps, sequence_length=1),
        )
        for num_steps in (25, 50)
    )

    assert short_data.components["add-dataset"].cache_dir == long_data.components["add-dataset"].cache_dir
    assert short_data.max_train_sequences == {"add-dataset": 5}
    assert long_data.max_train_sequences == {"add-dataset": 10}


def test_hero_capacity_check_uses_the_final_add_dataset_weights(tmp_path, stable_tokenizer):
    """A small new share moves the block remainder to the largest Hero cell, which then needs more data."""
    budget = ResolvedTrainingBudget(batch_size=_MIXTURE_BLOCK_SIZE, num_steps=1, sequence_length=4096)
    manifest = hero_sample_manifest(str(tmp_path / "hero"))
    # The sample holds exactly what the Hero control reads in one mixture block.
    sequences = _mixture_component_examples(
        {component.name: component.weight for component in manifest.components}, budget.token_count
    )
    hero_path = str(tmp_path / "hero-artifact")
    write_artifact(
        PreparedHeroSample(
            baseline=manifest,
            source_store="s3://test/store",
            source_tokenizer="source-tokenizer",
            source_tokenizer_hash="sha256:source",
            target_tokenizer_hash=TOKENIZER_HASH,
            recipe_sha256="a" * 64,
            phase="main",
            loader_policy=HERO_LOADER_POLICY,
            sequence_length=4096,
            data_seed=0,
            requested_tokens=budget.token_count,
            actual_tokens=budget.token_count,
            component_tokens={name: count * 4096 for name, count in sequences.items()},
        ),
        hero_path,
    )
    prepared_path = str(tmp_path / "prepared-artifact")
    write_artifact(
        PreparedAddDatasetCache(
            cache_dir=str(tmp_path / "prepared-cache"),
            prefix=_prefix(token_cap=budget.token_count),
            actual_num_rows=1,
            actual_num_tokens=budget.token_count,
            tokenization_policy=TOKENIZATION_POLICY,
        ),
        prepared_path,
    )
    hero = HeroTrainingSource(ArtifactStep.adopt("hero/test", "2026.10.04", hero_path, kind=PreparedHeroSample))
    cache_step = ArtifactStep.adopt("prepared/test", "2026.10.04", prepared_path, kind=PreparedAddDatasetCache)
    context = StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=(hero.sample, cache_step))
    treatment = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            fraction=0.01,
            target_production_tokens=budget.token_count,
            available_unique_tokens=budget.token_count,
        ),
        baseline=hero,
    )

    hero.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)
    with pytest.raises(ValueError, match=r"hero component c\d\dq\d has [\d,]+ usable sequences"):
        treatment.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)


def test_add_dataset_cli_prepare_only_writes_cache_loadable_by_python_api(tmp_path, monkeypatch, stable_tokenizer):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
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
    step = add_dataset_cache_step(prefix=_prefix(token_cap=7), version="2026.10.03")
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
            fraction=0.2,
            target_production_tokens=1_000,
            available_unique_tokens=500,
        ),
        baseline=_UnlimitedBaseline(
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


def test_add_dataset_cli_reports_invalid_revision_as_usage_error(stable_tokenizer):
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


def test_add_dataset_cli_accepts_prepared_artifacts_without_hugging_face_options(tmp_path, monkeypatch):
    handles = []
    monkeypatch.setattr("marin.experiment.cli._print_plan", lambda roots: handles.extend(roots))
    prepared_path = str(tmp_path / "prepared-artifact")
    write_artifact(
        PreparedAddDatasetCache(
            cache_dir=str(tmp_path / "prepared-cache"),
            prefix=_prefix(token_cap=16_384),
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
        "--dense",
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
    caches = [step for step in graph_handles(handles) if step.artifact_type is PreparedAddDatasetCache]
    assert [cache.adopt_source for cache in caches] == [prepared_path]


def test_add_dataset_cli_prepares_maximum_prefix_once_for_smaller_rungs(monkeypatch, stable_tokenizer):
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
            "--dense",
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


def test_prepare_add_dataset_cache_stops_when_the_bounded_prefix_runs_out(tmp_path, monkeypatch, stable_tokenizer):
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: _SmallTokenizer())
    with pytest.raises(ValueError, match="requires 5 tokens"):
        prepare_add_dataset_cache(
            _config(tmp_path, token_cap=5, max_rows=1),
            rows=[{"body": "one"}, {"body": "unread"}],
            tokenizer=_SmallTokenizer.name_or_path,
        )


def test_long_document_prefix_preserves_tokens_and_defers_oversized_lookahead(tmp_path, monkeypatch, stable_tokenizer):
    tokenizer = _SmallTokenizer()
    monkeypatch.setattr("experiments.grug.fast_track.add_dataset.load_tokenizer", lambda _name: tokenizer)
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


def test_add_dataset_cache_identity_changes_with_revision_and_prefix_limit():
    prefix = _prefix()
    original = add_dataset_cache_step(prefix=prefix, version="2026.10.03")

    revision_change = add_dataset_cache_step(
        prefix=prefix.model_copy(update={"revision": "b" * 40}), version="2026.10.03"
    )
    prefix_change = add_dataset_cache_step(
        prefix=prefix.model_copy(update={"requested_token_cap": prefix.requested_token_cap + 1}),
        version="2026.10.03",
    )
    tokenizer_change = add_dataset_cache_step(
        prefix=prefix.model_copy(update={"tokenizer_hash": "sha256:other-tokenizer"}), version="2026.10.03"
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
