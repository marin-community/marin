# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the split tokenize pipeline (Stage A: attribute parquet, Stage B: store builder).

Unit tests cover the pure helpers (``attach_id``, ``IdPreservingPreprocessor``).
The integration test exercises the A→B pipeline end-to-end against the legacy
``tokenize()`` path on a tiny local parquet fixture.
"""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.store.cache import CacheLedger, TreeCache
from marin.datakit.normalize import NormalizedData, generate_id
from marin.processing.tokenize import _core as tokenize_core
from marin.processing.tokenize._core import IdPreservingPreprocessor, attach_id, split_oversized_token_record
from marin.processing.tokenize.attributes import (
    TokenizeAttributesConfig,
    TokenizedAttrData,
    tokenize_attributes,
    tokenize_attributes_step,
)
from marin.processing.tokenize.store_builder import (
    BuildLevanterStoreConfig,
    build_levanter_store,
    build_levanter_store_step,
)
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize


class _FakeProcessor:
    """1:1 processor stub: copies input shape into output."""

    def __init__(self, returns: object | None = None):
        self._returns = returns

    def __call__(self, batch):
        if self._returns is not None:
            return self._returns
        return [{"input_ids": [i, i + 1]} for i, _ in enumerate(batch)]


class _FakeWorkerContext:
    def get_shared(self, name: str) -> str:
        return {"tokenizer_name": "unused", "tokenizer_backend": "huggingface"}[name]


def test_attach_id_preserves_existing_id():
    record = {"id": "abc", "text": "hello"}
    assert attach_id(record) is record


def test_attach_id_treats_none_as_missing():
    record = {"id": None, "text": "hello"}
    out = attach_id(record)
    assert out["id"] == generate_id("hello")


def test_attach_id_uses_text_field():
    record = {"text": "hello world"}
    out = attach_id(record)
    assert out["id"] == generate_id("hello world")
    assert out["text"] == "hello world"


def test_attach_id_custom_text_field():
    record = {"body": "hello"}
    out = attach_id(record, text_field="body")
    assert out["id"] == generate_id("hello")


def test_attach_id_falls_back_to_record_serialization():
    """Records lacking the configured text field still get a deterministic id."""
    record = {"messages": [{"role": "user", "content": "hi"}]}
    a = attach_id(record, text_field="text")
    b = attach_id(record, text_field="text")
    assert a["id"] == b["id"]
    # Falls back to a JSON-of-record hash, which differs from hashing 'text'.
    assert a["id"] != generate_id("hi")


def test_attach_id_is_deterministic_across_dict_orders():
    record_a = {"messages": [{"role": "user", "content": "hi"}]}
    record_b = {"messages": [{"role": "user", "content": "hi"}]}
    assert attach_id(record_a, text_field="text")["id"] == attach_id(record_b, text_field="text")["id"]


def test_id_preserving_preprocessor_threads_id_through():
    inner = _FakeProcessor()
    wrapped = IdPreservingPreprocessor(inner)
    batch = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
    out = wrapped(batch)
    assert [r["id"] for r in out] == ["a", "b"]
    assert all("input_ids" in r for r in out)


def test_id_preserving_preprocessor_handles_struct_of_arrays():
    """Some processors return a Mapping of column-arrays instead of a list of dicts."""
    soa = {"input_ids": [[1, 2], [3, 4]]}
    inner = _FakeProcessor(returns=soa)
    wrapped = IdPreservingPreprocessor(inner)
    batch = [{"id": "a"}, {"id": "b"}]
    out = wrapped(batch)
    assert [r["id"] for r in out] == ["a", "b"]
    assert [r["input_ids"] for r in out] == [[1, 2], [3, 4]]


def test_id_preserving_preprocessor_raises_on_non_1_to_1():
    """A processor that drops or splits records must fail loudly, not silently misalign ids."""
    inner = _FakeProcessor(returns=[{"input_ids": [1]}])  # 1 output for 2 inputs
    wrapped = IdPreservingPreprocessor(inner)
    with pytest.raises(RuntimeError, match="1:1"):
        wrapped([{"id": "a"}, {"id": "b"}])


def test_tokenize_batches_uses_format_token_key(monkeypatch):
    monkeypatch.setattr(tokenize_core, "zephyr_worker_ctx", _FakeWorkerContext)
    monkeypatch.setattr(tokenize_core, "load_tokenizer", lambda *args, **kwargs: object())
    data_format = PrebuiltLmDatasetFormat(input_ids_key="tokens")
    batches = iter([[{"id": "empty", "tokens": []}, {"id": "kept", "tokens": [1, 2]}]])

    rows = list(tokenize_core.tokenize_batches_with_id(data_format=data_format, batches=batches))

    assert len(rows) == 1
    assert rows[0]["id"] == "kept"
    assert rows[0]["tokens"].tolist() == [1, 2]
    assert rows[0]["chunk_index"] == 0


def test_oversized_token_record_round_trips_as_ordered_chunks(tmp_path):
    """A split document keeps its id and orders its rows with chunk_index.

    The id stays the join key against the other datakit attribute datasets, so a
    consumer that joins on id still matches every row of the document, and gets
    the document back by a sort on (id, chunk_index).
    """
    record = {
        "id": "0123456789abcdef0123456789abcdef",
        "input_ids": list(range(11)),
        "assistant_masks": [i % 2 for i in range(11)],
    }

    chunks = list(split_oversized_token_record(record, max_tokens=4))
    output_path = tmp_path / "chunks.parquet"
    pq.write_table(pa.Table.from_pylist(chunks), output_path)
    rows = pq.read_table(output_path).to_pylist()

    assert [row["id"] for row in rows] == [record["id"]] * 3
    assert [row["chunk_index"] for row in rows] == [0, 1, 2]
    assert [token for row in rows for token in row["input_ids"]] == record["input_ids"]
    assert [mask for row in rows for mask in row["assistant_masks"]] == record["assistant_masks"]


def test_oversized_ndarray_token_fields_are_split():
    """Array token fields split too, not just lists.

    Levanter's chat and prebuilt-cache processors return ``np.ndarray`` for
    ``input_ids`` and ``assistant_masks``, and ndarray is not a ``Sequence``.
    Copying such a field whole into each chunk would leave the oversized row
    oversized and duplicate the document.
    """
    record = {
        "id": "chat-doc",
        "input_ids": np.arange(10, dtype=np.int32),
        "assistant_masks": np.ones(10, dtype=np.int32),
    }

    chunks = list(split_oversized_token_record(record, max_tokens=4))

    assert [len(c["input_ids"]) for c in chunks] == [4, 4, 2]
    assert [len(c["assistant_masks"]) for c in chunks] == [4, 4, 2]
    assert np.array_equal(np.concatenate([c["input_ids"] for c in chunks]), record["input_ids"])


def test_unsplit_token_record_is_chunk_zero():
    """A document that fits stays one row, so chunk_index is uniform across the dataset."""
    record = {"id": "abc", "input_ids": [1, 2, 3]}

    rows = list(split_oversized_token_record(record, max_tokens=16))

    assert rows == [{"id": "abc", "input_ids": [1, 2, 3], "chunk_index": 0}]


def _write_normalized_fixture(tmp_path, texts: list[str]) -> NormalizedData:
    """Write a small datakit-normalized parquet shard with {id, text} columns."""
    main_dir = tmp_path / "normalized" / "outputs" / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        ({"id": generate_id(t), "text": t} for t in texts),
        key=lambda r: r["id"],
    )
    table = pa.Table.from_pylist(rows, schema=pa.schema([("id", pa.string()), ("text", pa.string())]))
    pq.write_table(table, str(main_dir / "part-00000-of-00001.parquet"))
    return NormalizedData(
        main_output_dir=str(main_dir),
        dup_output_dir=str(tmp_path / "normalized" / "outputs" / "dups"),
        counters={},
    )


def test_split_pipeline_matches_legacy_tokenize(tmp_path, monkeypatch):
    """Stage A → Stage B should produce a Levanter cache with the same token count
    as the legacy raw-input ``tokenize()`` path on the same texts."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
        "Bright vixens jump; dozy fowl quack.",
    ]
    source = _write_normalized_fixture(tmp_path, texts)

    # --- Stage A: tokenize → attribute parquet ---
    attr_config = TokenizeAttributesConfig(
        train_source=source,
        output_path=str(tmp_path / "attrs"),
        tokenizer="gpt2",
        format=TextLmDatasetFormat(),
    )
    tokenized: TokenizedAttrData = tokenize_attributes(attr_config)

    assert tokenized.source_keys["train"] == "normalized/outputs/main"
    train_shards = tokenized.shard_paths("train")
    assert len(train_shards) == 1, f"expected 1 attribute shard, got {len(train_shards)}: {train_shards}"
    attr_table = pq.read_table(train_shards[0])
    assert set(attr_table.column_names) == {"id", "chunk_index", "input_ids"}
    # These texts are far below the token limit of one Parquet row, so each is a
    # single row, chunk 0. The store below thus sees one cache row per text.
    assert attr_table.num_rows == len(texts)
    assert attr_table["chunk_index"].to_pylist() == [0] * len(texts)
    # Datakit invariant: sorted by id within each partition.
    ids = attr_table["id"].to_pylist()
    assert ids == sorted(ids)

    # --- Stage B: attribute parquet → Levanter store ---
    store_config = BuildLevanterStoreConfig(
        sources=[tokenized],
        cache_path=str(tmp_path / "store"),
        max_workers=2,
    )
    build_levanter_store(store_config)

    split_ledger = CacheLedger.load(str(tmp_path / "store" / "train"))
    assert split_ledger.is_finished
    assert split_ledger.total_num_rows == len(texts)

    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    split_cache = TreeCache.load(str(tmp_path / "store" / "train"), exemplar=exemplar)
    split_total_tokens = sum(len(split_cache[i]["input_ids"]) for i in range(len(split_cache)))

    # --- Reference: legacy tokenize() on the same texts as raw jsonl ---
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "data.jsonl"
    with open(raw_path, "w") as f:

        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")

    legacy_config = TokenizeConfig(
        train_paths=[str(raw_path)],
        validation_paths=[],
        cache_path=str(tmp_path / "legacy_store"),
        tokenizer="gpt2",
        format=TextLmDatasetFormat(),
        # pytest tmp paths contain "test"; opt out of the train-path guard for this fixture.
        allow_test_in_train=True,
    )
    tokenize(legacy_config)

    legacy_ledger = CacheLedger.load(str(tmp_path / "legacy_store" / "train"))
    legacy_cache = TreeCache.load(str(tmp_path / "legacy_store" / "train"), exemplar=exemplar)
    legacy_total_tokens = sum(len(legacy_cache[i]["input_ids"]) for i in range(len(legacy_cache)))

    assert split_ledger.total_num_rows == legacy_ledger.total_num_rows
    assert split_total_tokens == legacy_total_tokens

    # Both stats files written.
    assert os.path.exists(tmp_path / "store" / "train" / ".stats.json")
    assert os.path.exists(tmp_path / "legacy_store" / "train" / ".stats.json")


def test_tokenize_attributes_step_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one"):
        tokenize_attributes_step(name="x", tokenizer="gpt2")


def test_build_levanter_store_step_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one"):
        build_levanter_store_step(name="store", tokenize_steps=[])
