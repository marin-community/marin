# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from marin.datakit.normalize import NormalizedData, generate_id
from marin.execution.artifact import write_artifact
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from zephyr.context import ZephyrContext

from experiments.datakit.reference_pipeline import TokenizerSpec
from experiments.grug.fast_track import corpus_sample
from experiments.grug.fast_track.corpus_sample import (
    CorpusSampleConfig,
    CorpusSampleSpec,
    CorpusSource,
    RawCorpusPool,
    _CorpusShard,
    _sample_records,
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


def _bpe_tokenizer(tmp_path):
    backend = Tokenizer(
        models.BPE(
            vocab={"a": 0, " ": 1, "a ": 2, "[UNK]": 3, "[EOS]": 4},
            merges=[("a", " ")],
            unk_token="[UNK]",
        )
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    path = tmp_path / "bpe-tokenizer"
    tokenizer.save_pretrained(path)
    return load_tokenizer(str(path))


def _text_shard(tmp_path, texts):
    path = tmp_path / "texts.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": str(index), "text": text} for index, text in enumerate(texts)]), path)
    return _CorpusShard("fixture", str(path))


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


def test_corpus_long_string_policy_is_stable_and_outer_batches_are_independent(tmp_path, monkeypatch):
    short_text = "a " * 250
    long_text = "a " * 12_000
    texts = [short_text, long_text, "a " * 17, "a " * 2_000]
    shard = _text_shard(tmp_path, texts)
    tokenizer = _bpe_tokenizer(tmp_path)

    monkeypatch.setattr(corpus_sample, "TOKENIZE_BATCH_MAX_BYTES", 1_000_000)
    grouped = list(
        _sample_records(
            shard,
            probability=1,
            seed=3,
            tokenizer=tokenizer,
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )
    monkeypatch.setattr(corpus_sample, "TOKENIZE_BATCH_MAX_BYTES", 1)
    single_document = list(
        _sample_records(
            shard,
            probability=1,
            seed=3,
            tokenizer=tokenizer,
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )

    assert grouped == single_document
    assert [row["id"] for row in grouped] == [str(index) for index in range(len(texts))]
    assert [row["normalized_row"] for row in grouped] == list(range(len(texts)))
    assert grouped[0]["input_ids"] == tokenizer.encode(short_text + " " + tokenizer.eos_token)
    assert grouped[2]["input_ids"] == tokenizer.encode(texts[2] + " " + tokenizer.eos_token)
    assert grouped[3]["input_ids"] == tokenizer.encode(texts[3] + " " + tokenizer.eos_token)

    long_policy_ids = [2] * 5000 + [0, 1] + [2] * 4999 + [0, 1] + [2] * 1999 + [1, 4]
    whole_text_ids = tokenizer.encode(long_text + " " + tokenizer.eos_token)
    assert whole_text_ids == [2] * 12_000 + [1, 4]
    assert grouped[1]["input_ids"] == long_policy_ids
    assert grouped[1]["input_ids"] != whole_text_ids
    assert grouped[1]["token_count"] == len(long_policy_ids)


def test_corpus_fails_on_oversize_utf8_document_with_source_and_id(tmp_path, monkeypatch, corpus_tokenizer):
    source = _source(tmp_path, "oversize", 1, text="é" * 7)
    shard = _CorpusShard(source.name, str(tmp_path / source.name / "part-0000.parquet"))
    monkeypatch.setattr(corpus_sample, "TOKENIZATION_MAX_DOCUMENT_BYTES", 12)

    with pytest.raises(ValueError, match=r"source='oversize'.*id='0'.*bytes=14.*limit=12"):
        list(
            _sample_records(
                shard,
                probability=1,
                seed=0,
                tokenizer=load_tokenizer(corpus_tokenizer.name),
                excluded_groups=frozenset(),
                excluded_document_ids=frozenset(),
            )
        )
