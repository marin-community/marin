# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the split tokenize pipeline (Stage A: attribute parquet, Stage B: store builder).

Unit tests cover the pure helpers (``attach_id``, ``IdPreservingPreprocessor``).
The slow integration test exercises the A→B pipeline end-to-end against the
legacy ``tokenize()`` path on a tiny local parquet fixture.
"""
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, TreeCache
from marin.datakit.normalize import NormalizedData, generate_id
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize._core import IdPreservingPreprocessor, attach_id
from marin.processing.tokenize.attributes import (
    TokenizeAttributesConfig,
    TokenizedAttrData,
    tokenize_attributes,
    tokenize_attributes_step,
)
from marin.processing.tokenize.store_builder import (
    BuildLevanterStoreConfig,
    _structural_exemplar,
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


@pytest.mark.slow
def test_split_pipeline_matches_legacy_tokenize(tmp_path):
    """Stage A → Stage B should produce a Levanter cache with the same token count
    as the legacy raw-input ``tokenize()`` path on the same texts."""
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

    train_shards = tokenized.shard_paths("train")
    assert len(train_shards) == 1, f"expected 1 attribute shard, got {len(train_shards)}: {train_shards}"
    attr_table = pq.read_table(train_shards[0])
    assert set(attr_table.column_names) == {"id", "input_ids"}
    assert attr_table.num_rows == len(texts)
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


# ---------------------------------------------------------------------------
# StepSpec wrapper tests
# ---------------------------------------------------------------------------


def _stub_normalize_step(name: str = "normalize") -> StepSpec:
    """Return a StepSpec stub usable as an upstream `normalize` dep.

    The stub never gets executed; we only inspect identity/deps/hash_id.
    """
    return StepSpec(name=name, hash_attrs={"stub": name})


def test_tokenize_attributes_step_wires_deps_and_hash_attrs():
    train = _stub_normalize_step("normalize-train")
    val = _stub_normalize_step("normalize-validation")
    step = tokenize_attributes_step(
        name="fineweb/tokenize",
        train_normalize=train,
        validation_normalize=val,
        tokenizer="gpt2",
        sample_count=1000,
    )
    assert step.name == "fineweb/tokenize"
    assert step.deps == [train, val]
    assert step.hash_attrs["tokenizer"] == "gpt2"
    assert step.hash_attrs["sample_count"] == 1000
    assert "format" in step.hash_attrs


def test_tokenize_attributes_step_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one"):
        tokenize_attributes_step(name="x", tokenizer="gpt2")


def test_tokenize_attributes_step_hash_id_changes_with_tokenizer():
    train = _stub_normalize_step()
    a = tokenize_attributes_step(name="x", train_normalize=train, tokenizer="gpt2")
    b = tokenize_attributes_step(name="x", train_normalize=train, tokenizer="meta-llama/Llama-3.1-8B")
    assert a.hash_id != b.hash_id


def test_tokenize_attributes_step_hash_id_changes_with_sample_count():
    train = _stub_normalize_step()
    a = tokenize_attributes_step(name="x", train_normalize=train, tokenizer="gpt2")
    b = tokenize_attributes_step(name="x", train_normalize=train, tokenizer="gpt2", sample_count=100)
    assert a.hash_id != b.hash_id


def test_build_levanter_store_step_wires_deps():
    tok = StepSpec(name="upstream-tokens", hash_attrs={"x": 1})
    step = build_levanter_store_step(name="store", tokenize_steps=[tok])
    assert step.deps == [tok]


def test_build_levanter_store_step_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one"):
        build_levanter_store_step(name="store", tokenize_steps=[])


def test_structural_exemplar_slices_sequence_leaves():
    """Sequence leaves shrink to one element; scalars/strings pass through whole."""
    record = {
        "input_ids": np.arange(1000, dtype=np.int32),
        "segment_ids": [0, 0, 1, 1, 2],
        "weights": (0.5, 0.25, 0.25),
        "doc_id": "abc-123",
        "length": 1000,
    }
    out = _structural_exemplar(record)
    assert np.array_equal(out["input_ids"], np.array([0], dtype=np.int32))
    assert out["segment_ids"] == [0]
    assert out["weights"] == (0.5,)
    assert out["doc_id"] == "abc-123"
    assert out["length"] == 1000


def test_build_levanter_store_step_hash_id_changes_with_batch_size():
    tok = StepSpec(name="upstream-tokens", hash_attrs={"x": 1})
    a = build_levanter_store_step(name="store", tokenize_steps=[tok])
    b = build_levanter_store_step(name="store", tokenize_steps=[tok], levanter_batch_size=4096)
    assert a.hash_id != b.hash_id
