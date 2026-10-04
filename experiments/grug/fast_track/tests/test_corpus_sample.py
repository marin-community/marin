# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from levanter.tokenizers import tokenizer_content_hash
from marin.datakit.normalize import NormalizedData, generate_id
from marin.execution.artifact import write_artifact
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from zephyr.context import ZephyrContext

from experiments.datakit.reference_pipeline import TokenizerSpec
from experiments.grug.fast_track.corpus_sample import (
    CorpusSampleConfig,
    CorpusSampleSpec,
    CorpusSource,
    RawCorpusPool,
    prepare_corpus_pool,
)
from experiments.grug.fast_track.label_exclusion import LabelExclusion


@pytest.fixture
def corpus_context(tmp_path):
    client = LocalClient()
    try:
        with ZephyrContext(
            client=client,
            max_workers=2,
            resources=ResourceConfig(cpu=2, ram="8g"),
            chunk_storage_prefix=str(tmp_path / "chunks"),
        ) as context:
            yield context
    finally:
        client.shutdown()


@pytest.fixture
def corpus_tokenizer(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(path)
    return TokenizerSpec(str(path), tokenizer_content_hash(str(path)))


def _source(tmp_path, name, count, *, partitions=1, estimate=10_000, text="hello world"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    for partition in range(partitions):
        pq.write_table(
            pa.Table.from_pylist([{"id": str(index), "text": text} for index in range(partition, count, partitions)]),
            directory / f"part-{partition:04d}.parquet",
        )
    artifact_path = str(tmp_path / f"{name}-artifact")
    write_artifact(
        NormalizedData(main_output_dir=str(directory), dup_output_dir=str(directory), counters={}), artifact_path
    )
    return CorpusSource(name, artifact_path, estimate)


def _records(pool):
    return [row for path in pool.shards for row in pq.read_table(path).to_pylist()]


def test_corpus_sample_keeps_raw_proportions_and_is_independent_of_input_partitions(
    tmp_path, corpus_context, corpus_tokenizer
):
    sources = (
        _source(tmp_path / "first", "large", 400, text="hello world " * 4),
        _source(tmp_path / "first", "small", 400),
    )
    spec = CorpusSampleSpec(sources, corpus_tokenizer, token_budget=1200, seed=11)
    first = prepare_corpus_pool(CorpusSampleConfig(spec, str(tmp_path / "pool1")), ctx=corpus_context)
    repartitioned = (
        _source(tmp_path / "second", "small", 400, partitions=3, estimate=1),
        _source(tmp_path / "second", "large", 400, partitions=7, estimate=1, text="hello world " * 4),
    )
    second = prepare_corpus_pool(
        CorpusSampleConfig(replace(spec, sources=repartitioned), str(tmp_path / "pool2")), ctx=corpus_context
    )
    first_rows, second_rows = _records(first), _records(second)
    locator_fields = {"normalized_shard", "normalized_row"}
    assert [{key: value for key, value in row.items() if key not in locator_fields} for row in first_rows] == [
        {key: value for key, value in row.items() if key not in locator_fields} for row in second_rows
    ]
    for row in second_rows:
        original = pq.read_table(row["normalized_shard"]).to_pylist()[row["normalized_row"]]
        assert (original["id"], original["text"]) == (row["id"], row["text"])
    assert first.actual_tokens >= 1200
    assert first.actual_tokens < 1200 + max(row["token_count"] for row in first_rows)
    assert first.source_tokens["large"] / first.actual_tokens == pytest.approx(0.75, abs=0.06)
    assert [row["sample_rank"] for row in first_rows] == sorted(row["sample_rank"] for row in first_rows)
    assert {row["text"] for row in first_rows} == {"hello world", "hello world " * 4}
    artifact_path = str(tmp_path / "published-pool")
    write_artifact(first, artifact_path)
    restored = RawCorpusPool.raw_load(artifact_path)
    assert restored.range_totals == first.range_totals
    assert sum(item.tokens for item in restored.range_totals) == restored.actual_tokens


def test_corpus_sample_reports_insufficient_real_capacity(tmp_path, corpus_context, corpus_tokenizer):
    source = _source(tmp_path, "tiny", 4, estimate=1)
    spec = CorpusSampleSpec((source,), corpus_tokenizer, token_budget=1000, seed=0)
    with pytest.raises(ValueError, match="corpus has"):
        prepare_corpus_pool(CorpusSampleConfig(spec, str(tmp_path / "pool")), ctx=corpus_context)


@pytest.mark.parametrize("use_normalized_ids", [False, True])
def test_corpus_excludes_label_duplicates_across_sources(tmp_path, corpus_context, corpus_tokenizer, use_normalized_ids):
    sources = (
        _source(tmp_path, "label-source", 30, estimate=1),
        _source(tmp_path, "copy-source", 30, estimate=1),
        _source(tmp_path, "unlabelled", 30, estimate=1, text="hello"),
    )
    if use_normalized_ids:
        for source in sources:
            directory = tmp_path / source.name
            for path in directory.glob("*.parquet"):
                rows = pq.read_table(path).to_pylist()
                for row in rows:
                    row["id"] = generate_id(row["text"])
                pq.write_table(pa.Table.from_pylist(rows), path)
    labels = LabelExclusion(
        label_revision="fixture-labels-v1",
        duplicate_groups=frozenset() if use_normalized_ids else frozenset({hashlib.sha256(b"hello world").hexdigest()}),
        normalized_document_ids=frozenset({generate_id("hello world")}) if use_normalized_ids else frozenset(),
    )
    spec = CorpusSampleSpec(sources, corpus_tokenizer, token_budget=30, seed=0, label_exclusion=labels)
    pool = prepare_corpus_pool(CorpusSampleConfig(spec, str(tmp_path / "pool")), ctx=corpus_context)
    assert set(pool.source_tokens) == {"unlabelled"}
    assert all(row["duplicate_group"] not in labels.duplicate_groups for row in _records(pool))
