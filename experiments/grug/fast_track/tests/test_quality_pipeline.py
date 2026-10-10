# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import cast

import click
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from haliax import Axis
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheLedger, CacheMetadata, SerialCacheWriter, TreeCache
from marin.execution.lazy import ArtifactStep
from marin.execution.step_spec import StepSpec
from zephyr.context import ZephyrContext

from experiments.grug.fast_track import quality_pipeline
from experiments.grug.fast_track.contracts import ResolvedTrainingBudget
from experiments.grug.fast_track.corpus_sample import RawCorpusPool
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.quality_cli import main as quality_main
from experiments.grug.fast_track.quality_pipeline import (
    QualityConfig,
    QualityData,
    QualityTrainingSource,
    SelectionMethod,
    prepare_quality_data,
    score_raw_pool,
)
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal


def _write_selected_token_shard(path, token_rows, *, bad_checksum_row=None):
    rows = []
    for index, values in enumerate(token_rows):
        tokens = np.asarray(values, dtype="<i4")
        checksum = hashlib.sha256(tokens.tobytes()).hexdigest()
        if index == bad_checksum_row:
            checksum = "0" * 64
        rows.append(
            {
                "source": "source",
                "id": str(index),
                "input_ids": tokens.tolist(),
                "token_count": len(tokens),
                "token_sha256": checksum,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


class _Scorer:
    def scores(self, batch):
        return np.asarray([float(text.rsplit("-", 1)[1]) for text in batch.texts])

    def __call__(self):
        return _Scorer()


class _SmallTokenizer:
    name_or_path = "quality-test-tokenizer"
    vocab_size = 32
    bos_token_id = None
    eos_token_id = None
    eos_token = "EOS"

    def encode(self, text: str) -> list[int]:
        return [len(text)]


class _FingerprintContext:
    is_fingerprint = True

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def artifact_path(self, _selection):
        return self.cache_dir


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="quality-pipeline-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


def test_quality_pipeline_writes_exact_budget_tokens_from_selected_parquet(tmp_path, zephyr_context, monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.quality_pipeline.load_tokenizer", lambda _name: _SmallTokenizer())
    rows = []
    for index in range(12):
        token_values = [index + 1] * 5
        rows.append(
            {
                "source": "source",
                "id": str(index),
                "sample_rank": hashlib.sha256(f"seed:{index}".encode()).hexdigest(),
                "duplicate_group": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
                "text": f"text-{index}",
                "input_ids": token_values,
                "token_count": len(token_values),
            }
        )
    rows.sort(key=lambda row: (row["sample_rank"], row["source"], row["id"]))
    path_a, path_b = tmp_path / "raw-a.parquet", tmp_path / "raw-b.parquet"
    pq.write_table(pa.Table.from_pylist(rows[:6]), path_a)
    pq.write_table(pa.Table.from_pylist(rows[6:]), path_b)
    raw = RawCorpusPool(
        tokenizer="test-tokenizer",
        tokenizer_hash="a" * 64,
        sources=(),
        seed=17,
        requested_tokens=60,
        actual_tokens=60,
        documents=12,
        shards=(str(path_a), str(path_b)),
        range_totals=(
            RangeTokenTotal(0, str(path_a), 6, 30),
            RangeTokenTotal(1, str(path_b), 6, 30),
        ),
        source_tokens={"source": 60},
        manifest_path=str(tmp_path / "raw.json"),
    )
    with pytest.raises(ValueError, match="below the 60 scoring budget"):
        score_raw_pool(
            raw.model_copy(update={"requested_tokens": 55}),
            ctx=zephyr_context,
            output_path=str(tmp_path / "overshoot-only-capacity"),
            scorer_factory=_Scorer(),
            classifier_identity={"implementation": "text-test", "revision": "v1"},
            label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
            token_budget=60,
        )
    scored = score_raw_pool(
        raw,
        ctx=zephyr_context,
        output_path=str(tmp_path / "scored"),
        scorer_factory=_Scorer(),
        classifier_identity={"implementation": "text-test", "revision": "v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=40,
    )
    config = QualityConfig(
        4,
        SelectionMethod.CANDIDATE,
        7,
        scored,
        str(tmp_path / "selection"),
    )

    result = prepare_quality_data(config, ctx=zephyr_context)
    cache = TreeCache.load(result.cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)})
    selected = cache.get_batch_sync([0])
    expected_id = max((row["id"] for row in rows[:8]), key=int)

    assert result.tokenizer_hash == raw.tokenizer_hash
    assert result.requested_tokens == 4
    assert result.actual_tokens == 5
    assert len(selected) == 1
    np.testing.assert_array_equal(selected[0]["input_ids"], np.full(5, int(expected_id) + 1, dtype=np.int32))


def test_quality_cache_writer_preserves_rows_across_bounded_flushes(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_pipeline, "QUALITY_CACHE_MAX_PENDING_BYTES", 32)
    monkeypatch.setattr(quality_pipeline, "QUALITY_CACHE_MAX_PENDING_DOCUMENTS", 2)
    token_rows = [
        [1, 2],
        [10, 11, 12, 13, 14, 15],
        [21, 22, 23],
        [31, 32, 33, 34],
        [41],
        [50, 51, 52, 53, 54, 55, 56, 57, 58],
        [60],
    ]
    first_shard = tmp_path / "z-selected.parquet"
    second_shard = tmp_path / "a-selected.parquet"
    _write_selected_token_shard(first_shard, token_rows[:3])
    _write_selected_token_shard(second_shard, token_rows[3:])
    metadata = CacheMetadata({"preprocessor": "quality-cache-test"})
    cache_dir = str(tmp_path / "cache")

    # The supplied order is opposite to the shard names' lexical order.
    quality_pipeline._write_quality_token_cache([str(first_shard), str(second_shard)], cache_dir, metadata=metadata)

    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    cache = TreeCache.load(cache_dir, exemplar, metadata)
    ledger = CacheLedger.load(cache_dir)
    records = cache.get_batch_sync(list(range(len(token_rows))))
    assert len(cache) == len(token_rows)
    assert cache.flat_field_length("input_ids") == sum(map(len, token_rows))
    assert ledger.total_num_rows == len(token_rows)
    assert ledger.is_finished
    assert ledger.metadata == metadata
    for record, expected in zip(records, token_rows, strict=True):
        np.testing.assert_array_equal(record["input_ids"], np.asarray(expected, dtype="<i4"))
        assert record["input_ids"].dtype == np.dtype("<i4")


def test_quality_cache_writer_checksum_failure_does_not_commit_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(quality_pipeline, "QUALITY_CACHE_MAX_PENDING_BYTES", 32)
    shard_path = tmp_path / "bad-selected.parquet"
    _write_selected_token_shard(shard_path, [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10]], bad_checksum_row=1)
    cache_dir = str(tmp_path / "incomplete-cache")

    with pytest.raises(ValueError, match="checksum differs"):
        quality_pipeline._write_quality_token_cache(
            [str(shard_path)], cache_dir, metadata=CacheMetadata({"preprocessor": "quality-cache-test"})
        )

    with pytest.raises(FileNotFoundError):
        CacheLedger.load(cache_dir)


def test_quality_pipeline_fails_when_pool_capacity_does_not_meet_rung_budget(tmp_path, zephyr_context):
    rows = [
        {
            "source": "source",
            "id": str(index),
            "sample_rank": f"{index:064x}",
            "duplicate_group": str(index),
            "text": f"text-{index}",
            "input_ids": [index + 1] * 5,
            "token_count": 5,
        }
        for index in range(12)
    ]
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    raw = RawCorpusPool(
        tokenizer="test-tokenizer",
        tokenizer_hash="a" * 64,
        sources=(),
        seed=17,
        requested_tokens=60,
        actual_tokens=60,
        documents=12,
        shards=(str(path),),
        range_totals=(RangeTokenTotal(0, str(path), 12, 60),),
        source_tokens={"source": 60},
        manifest_path=str(tmp_path / "raw.json"),
    )
    with pytest.raises(ValueError, match="below the 70 scoring budget"):
        score_raw_pool(
            raw,
            ctx=zephyr_context,
            output_path=str(tmp_path / "undersized-score"),
            scorer_factory=_Scorer(),
            classifier_identity={"implementation": "text-test", "revision": "v1"},
            label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
            token_budget=70,
        )


def test_training_loader_reads_only_complete_mixture_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr("levanter.data.text.datasets.load_marin_tokenizer", lambda _name: _SmallTokenizer())
    artifact_dir = str(tmp_path / "artifact")
    cache_dir = f"{artifact_dir}/tokens"
    metadata = CacheMetadata(TextLmDatasetFormat().build_preprocessor(_SmallTokenizer()).metadata)
    with SerialCacheWriter(cache_dir, {"input_ids": np.zeros(0, dtype=np.int32)}, metadata=metadata) as writer:
        writer.write_batch([{"input_ids": np.asarray([value], dtype=np.int32)} for value in (31, 32, 33)])
    selection = cast(ArtifactStep[QualityData], object())
    source = QualityTrainingSource(selection, training_tokens=3)
    training_data = source.data_config(
        ctx=_FingerprintContext(artifact_dir),
        validation=(),
        tokenizer="small-test-tokenizer",
        budget=ResolvedTrainingBudget(batch_size=1, num_steps=3, sequence_length=1),
    )
    training_data = replace(training_data, shuffle=False)

    assert training_data.mixture_block_size == 1
    dataset = training_data.train_set(Axis("position", 1), BatchSchedule(1), key=jax.random.PRNGKey(5))
    examples = asyncio.run(dataset.get_batch([0, 1, 2]))
    token_values = [int(example.tokens.array[0]) for example in examples]

    assert sorted(token_values) == [31, 32, 33]


@pytest.mark.parametrize(
    ("stage", "terminal_artifact"),
    [
        ("features", "fast-track/quality-features"),
        ("score", "fast-track/scored-pool/"),
        ("select", "fast-track/quality/"),
        ("train", "grug/quality-cli-plan"),
    ],
)
def test_quality_cli_prints_selected_stage_graph(tmp_path, stage, terminal_artifact):
    args = ["--raw-pool", "s3://marin-test/raw-pool", "--stage", stage, "--version", "2026.10.04"]
    if stage != "features":
        exclusion_path = tmp_path / "labels.json"
        exclusion_path.write_text(
            LabelExclusion(label_revision="quality-cli-test-v1", duplicate_groups=frozenset()).model_dump_json()
        )
        args.extend(
            [
                "--scorer-factory",
                f"{__name__}:_Scorer",
                "--classifier-identity",
                '{"implementation":"test","revision":"v1"}',
                "--label-exclusion-manifest",
                str(exclusion_path),
            ]
        )
    if stage == "train":
        args.extend(["--run-id", "quality-cli-plan", "--size", "d512"])

    result = CliRunner().invoke(
        quality_main,
        args,
    )

    assert result.exit_code == 0, result.output
    assert "fast-track/raw-pool" in result.output
    assert "s3://marin-test/raw-pool" in result.output
    assert terminal_artifact in result.output


def test_quality_cli_reuses_scored_pool_for_moe_rung(monkeypatch):
    emitted = []
    monkeypatch.setattr(click, "echo", emitted.append)
    result = CliRunner().invoke(
        quality_main,
        [
            "--scored-pool",
            "s3://marin-test/scored-pool",
            "--stage",
            "train",
            "--run-id",
            "quality-cli-moe-plan",
            "--size",
            "d1280",
            "--moe",
            "--version",
            "2026.10.04",
        ],
    )

    assert result.exit_code == 0, result.exception
    training = next(item for item in emitted if isinstance(item, StepSpec))
    selection = next(dep for dep in training.deps if dep.name.startswith("fast-track/quality/"))
    adopted_pool = selection.deps[0]
    training_config = json.loads(training.fingerprint_payload)
    selection_config = json.loads(selection.fingerprint_payload)
    assert json.loads(adopted_pool.fingerprint_payload)["adopt_source"] == "s3://marin-test/scored-pool"
    assert training_config["model"]["dense_mlp"] is False
    assert training_config["trainer"]["trainer"]["num_train_steps"] == 16669
    assert selection_config["training_tokens"] == 17478713344


@pytest.mark.parametrize(
    "args",
    [
        ["--raw-pool", "s3://marin-test/raw", "--stage", stage, "--run-id", "ignored"]
        for stage in ("features", "score", "select")
    ]
    + [
        [
            "--scored-pool",
            "s3://marin-test/scored",
            "--stage",
            "train",
            "--run-id",
            "quality-cli-plan",
            "--training-tokens",
            "1000",
        ],
        [
            "--scored-pool",
            "s3://marin-test/scored",
            "--stage",
            "select",
            "--scorer-factory",
            f"{__name__}:_Scorer",
        ],
    ],
)
def test_quality_cli_rejects_options_that_would_be_ignored(args):
    # A discarded run ID, token budget, or scorer can misidentify an experiment.
    result = CliRunner().invoke(quality_main, [*args, "--version", "2026.10.04"])

    assert result.exit_code == 2
