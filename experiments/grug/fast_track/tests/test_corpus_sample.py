# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

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
            pa.Table.from_pylist(
                [{"id": generate_id(f"{name}/{index}"), "text": text} for index in range(partition, count, partitions)]
            ),
            directory / f"part-{partition:04d}.parquet",
        )
    artifact_path = str(tmp_path / f"{name}-artifact")
    write_artifact(
        NormalizedData(main_output_dir=str(directory), dup_output_dir=str(directory), counters={}), artifact_path
    )
    return CorpusSource(name, artifact_path, estimate)


def _write_id_parquet(path, document_ids, *, text="hello world", row_group_size=None, write_statistics=True):
    pq.write_table(
        pa.Table.from_pylist([{"id": document_id, "text": text} for document_id in document_ids]),
        path,
        row_group_size=row_group_size,
        write_statistics=write_statistics,
    )


def _source_with_ids(tmp_path, name, document_ids, *, estimate=10_000, text="hello world"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    _write_id_parquet(directory / "part-0000.parquet", document_ids, text=text)
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
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": generate_id(f"fixture/{index}/{text}"), "text": text} for index, text in enumerate(texts)]
        ),
        path,
    )
    return _CorpusShard("fixture", str(path))


def _write_id_shard(tmp_path, name, document_ids, *, row_group_size=None, write_statistics=True):
    path = tmp_path / f"{name}.parquet"
    _write_id_parquet(
        path,
        document_ids,
        row_group_size=row_group_size,
        write_statistics=write_statistics,
    )
    return _CorpusShard(name, str(path))


def test_corpus_sample_keeps_raw_proportions_and_is_independent_of_input_partitions(
    tmp_path, corpus_context, corpus_tokenizer
):
    sources = (
        _source(tmp_path / "first", "large", 400, text="hello world " * 4),
        _source(tmp_path / "first", "small", 400),
    )
    spec = CorpusSampleSpec(sources, corpus_tokenizer, token_budget=1200, seed=11)
    first = prepare_corpus_pool(CorpusSampleConfig(spec, str(tmp_path / "pool1")), ctx=corpus_context, num_ranges=16)
    repartitioned = (
        _source(tmp_path / "second", "small", 400, partitions=3, estimate=1),
        _source(tmp_path / "second", "large", 400, partitions=7, estimate=1, text="hello world " * 4),
    )
    second = prepare_corpus_pool(
        CorpusSampleConfig(replace(spec, sources=repartitioned), str(tmp_path / "pool2")),
        ctx=corpus_context,
        num_ranges=16,
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
        prepare_corpus_pool(CorpusSampleConfig(spec, str(tmp_path / "pool")), ctx=corpus_context, num_ranges=8)


def test_corpus_sample_manifest_keeps_full_domain_cutoff_width(tmp_path, corpus_context, corpus_tokenizer):
    source = _source(tmp_path, "full-domain", 4, estimate=1)
    output_path = tmp_path / "pool"
    spec = CorpusSampleSpec((source,), corpus_tokenizer, token_budget=1, seed=0)

    prepare_corpus_pool(CorpusSampleConfig(spec, str(output_path)), ctx=corpus_context, num_ranges=8)

    manifest = json.loads((output_path / "corpus.json").read_text())
    assert manifest["sampling_cutoff_hex"] == f"{1 << 128:033x}"
    assert len(manifest["sampling_cutoff_hex"]) == 33


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
    output_path = tmp_path / "pool"
    pool = prepare_corpus_pool(CorpusSampleConfig(spec, str(output_path)), ctx=corpus_context, num_ranges=8)
    manifest = json.loads((output_path / "corpus.json").read_text())
    assert pool.source_tokens == {"label-source": 0, "copy-source": 0, "unlabelled": pool.actual_tokens}
    assert manifest["source_tokens"] == pool.source_tokens
    assert all(row["duplicate_group"] not in labels.duplicate_groups for row in _records(pool))


def test_corpus_sample_low_probability_preserves_global_rank_order_and_prefix(
    tmp_path, corpus_context, corpus_tokenizer
):
    id_space = 1 << 128
    source_name = "low-probability"
    token_budget = 32
    seed = 5
    estimate = 5120
    cutoff = corpus_sample._sampling_cutoff(token_budget, estimate)
    salt = corpus_sample._rank_salt(source_name, seed)
    ids = [f"{(index * cutoff // 64 - salt) % id_space:032x}" for index in range(64)]
    ids.extend(f"{(cutoff + index - salt) % id_space:032x}" for index in range(8064))
    source = _source_with_ids(tmp_path, source_name, ids, estimate=estimate)
    spec = CorpusSampleSpec((source,), corpus_tokenizer, token_budget=token_budget, seed=seed)
    output_path = tmp_path / "pool"
    pool = prepare_corpus_pool(
        CorpusSampleConfig(spec, str(output_path)),
        ctx=corpus_context,
        num_ranges=32,
    )

    sampled_rows = list(
        _sample_records(
            _CorpusShard(source.name, str(tmp_path / source.name / "part-0000.parquet")),
            cutoff=cutoff,
            seed=spec.seed,
            tokenizer=load_tokenizer(corpus_tokenizer.name),
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )
    sampled_by_id = {row["id"]: row for row in sampled_rows}
    expected_ranked_ids = sorted(
        (
            (int(document_id, 16) + salt) % id_space,
            document_id,
        )
        for document_id in ids
        if (int(document_id, 16) + salt) % id_space < cutoff
    )
    assert set(sampled_by_id) == {document_id for _, document_id in expected_ranked_ids}
    expected_prefix = []
    expected_tokens = 0
    for _, document_id in expected_ranked_ids:
        row = sampled_by_id[document_id]
        expected_prefix.append(document_id)
        expected_tokens += row["token_count"]
        if expected_tokens >= spec.token_budget:
            break

    attempt_ranges = list((output_path / "sample-000").glob("range-*.parquet"))
    persisted_rows = _records(pool)
    manifest = json.loads((output_path / "corpus.json").read_text())
    assert len(attempt_ranges) > 1
    assert [row["id"] for row in persisted_rows] == expected_prefix
    assert [row["sample_rank"] for row in persisted_rows] == sorted(row["sample_rank"] for row in persisted_rows)
    assert manifest["actual_tokens"] == sum(row["token_count"] for row in persisted_rows)
    assert manifest["requested_tokens"] == spec.token_budget


def test_corpus_sample_prunes_wrapped_id_ranges_and_keeps_boundary_rows(tmp_path, corpus_tokenizer):
    id_space = 1 << 128
    source_name = "wrapped"
    seed = 0
    salt = corpus_sample._rank_salt(source_name, seed)
    cutoff = salt + 20
    document_ids = [0, 19, 20, 21, id_space - salt - 1, id_space - salt, id_space - 2, id_space - 1]
    shard = _write_id_shard(tmp_path, source_name, [f"{value:032x}" for value in document_ids], row_group_size=2)
    intervals = corpus_sample._id_intervals(shard.source, seed, cutoff)
    parquet = pq.ParquetFile(shard.path)

    actual = list(
        _sample_records(
            shard,
            cutoff=cutoff,
            seed=seed,
            tokenizer=load_tokenizer(corpus_tokenizer.name),
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )
    all_rows = pq.read_table(shard.path).to_pylist()
    expected = []
    for row_offset, row in enumerate(all_rows):
        rank = (int(row["id"], 16) + salt) % id_space
        if rank < cutoff:
            expected.append((row["id"], row_offset, f"{rank:032x}"))

    assert intervals == ((0, 20), (id_space - salt, id_space))
    assert corpus_sample._row_groups_to_read(parquet, intervals) == {0, 2, 3}
    assert [(row["id"], row["normalized_row"], row["sample_rank"]) for row in actual] == expected
    assert [row["normalized_row"] for row in actual] == [0, 1, 5, 6, 7]


def test_corpus_sample_reads_row_groups_without_id_statistics(tmp_path, corpus_tokenizer):
    shard = _write_id_shard(
        tmp_path,
        "without-id-stats",
        [f"{index:032x}" for index in range(8)],
        row_group_size=2,
        write_statistics=False,
    )
    parquet = pq.ParquetFile(shard.path)

    assert corpus_sample._row_groups_to_read(parquet, corpus_sample._id_intervals(shard.source, 0, 1)) == {0, 1, 2, 3}
    rows = list(
        _sample_records(
            shard,
            cutoff=1 << 128,
            seed=0,
            tokenizer=load_tokenizer(corpus_tokenizer.name),
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )
    assert [(row["id"], row["normalized_shard"], row["normalized_row"]) for row in rows] == [
        (f"{index:032x}", shard.path, index) for index in range(8)
    ]


@pytest.mark.parametrize(
    "table, message",
    [
        (
            pa.table({"id": pa.array(["G" * 32]), "text": pa.array(["hello"])}),
            "32 lowercase hexadecimal characters",
        ),
        (
            pa.table({"id": pa.array([None], type=pa.string()), "text": pa.array(["hello"])}),
            "id and text must be strings at row 0",
        ),
        (
            pa.table({"id": pa.array([1], type=pa.int64()), "text": pa.array(["hello"])}),
            "normalized id field must use an Arrow string type",
        ),
    ],
)
def test_corpus_sample_fails_on_observed_invalid_id_or_schema(tmp_path, corpus_tokenizer, table, message):
    path = tmp_path / "invalid.parquet"
    pq.write_table(table, path, write_statistics=False)
    shard = _CorpusShard("invalid", str(path))

    with pytest.raises(ValueError, match=message):
        list(
            _sample_records(
                shard,
                cutoff=1,
                seed=0,
                tokenizer=load_tokenizer(corpus_tokenizer.name),
                excluded_groups=frozenset(),
                excluded_document_ids=frozenset(),
            )
        )


def test_corpus_sample_grows_integer_cutoff_until_pool_is_large_enough(tmp_path, corpus_context, corpus_tokenizer):
    id_space = 1 << 128
    source_name = "growing-cutoff"
    token_budget = 32
    seed = 7
    estimate = 40_960
    initial_cutoff = corpus_sample._sampling_cutoff(token_budget, estimate)
    salt = corpus_sample._rank_salt(source_name, seed)
    ids = [f"{(index * id_space // 64 - salt) % id_space:032x}" for index in range(64)]
    source = _source_with_ids(tmp_path, source_name, ids, estimate=estimate)
    output_path = tmp_path / "growing-pool"
    spec = CorpusSampleSpec((source,), corpus_tokenizer, token_budget=token_budget, seed=seed)

    pool = prepare_corpus_pool(CorpusSampleConfig(spec, str(output_path)), ctx=corpus_context, num_ranges=32)
    manifest = json.loads((output_path / "corpus.json").read_text())
    full_read_rows = list(
        _sample_records(
            _CorpusShard(source.name, str(tmp_path / source.name / "part-0000.parquet")),
            cutoff=id_space,
            seed=seed,
            tokenizer=load_tokenizer(corpus_tokenizer.name),
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )
    full_read_rows.sort(key=lambda row: (row["sample_rank"], row["source"], row["id"]))
    expected_ids = []
    expected_tokens = 0
    for row in full_read_rows:
        expected_ids.append(row["id"])
        expected_tokens += row["token_count"]
        if expected_tokens >= token_budget:
            break

    assert initial_cutoff == id_space // 1024
    assert int(manifest["sampling_cutoff_hex"], 16) > initial_cutoff
    assert len(manifest["sampling_cutoff_hex"]) == 33
    assert manifest["sampling_method"] == corpus_sample.SAMPLING_POLICY
    assert pool.actual_tokens >= token_budget
    assert [row["id"] for row in _records(pool)] == expected_ids
    assert pool.actual_tokens == expected_tokens


def test_corpus_sources_rejects_missing_normalized_pins(tmp_path, monkeypatch):
    manifest_path = tmp_path / "hero_data_paths.json"
    manifest_path.write_text(json.dumps({"normalized/present": "normalized/present-path"}))
    monkeypatch.setattr(corpus_sample.hero_data, "manifest_path", lambda: manifest_path)
    monkeypatch.setattr(
        corpus_sample,
        "all_sources",
        lambda: {
            "present": SimpleNamespace(rough_token_count_b=1.0),
            "missing": SimpleNamespace(rough_token_count_b=1.0),
        },
    )

    with pytest.raises(ValueError, match="missing normalized pins for registered sources"):
        corpus_sample.corpus_sources()


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
            cutoff=1 << 128,
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
            cutoff=1 << 128,
            seed=3,
            tokenizer=tokenizer,
            excluded_groups=frozenset(),
            excluded_document_ids=frozenset(),
        )
    )

    assert grouped == single_document
    assert [row["id"] for row in grouped] == [generate_id(f"fixture/{index}/{text}") for index, text in enumerate(texts)]
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

    with pytest.raises(ValueError, match=r"source='oversize'.*id='[0-9a-f]{32}'.*bytes=14.*limit=12"):
        list(
            _sample_records(
                shard,
                cutoff=1 << 128,
                seed=0,
                tokenizer=load_tokenizer(corpus_tokenizer.name),
                excluded_groups=frozenset(),
                excluded_document_ids=frozenset(),
            )
        )
