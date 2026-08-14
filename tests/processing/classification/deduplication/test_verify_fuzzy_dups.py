# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from random import Random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, MinHashParams
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    FuzzyVerificationStoreConfig,
    LocalRepresentativeParams,
    VerificationShard,
    _candidate_documents,
    _decompress_document_text,
    _parquet_rows,
    _system_arrow_memory_pool,
)
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    verify_fuzzy_dups as _verify_fuzzy_dups,
)
from zephyr.stage_io import ZephyrWorkerError
from zephyr.writers import write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0)
_COUNTER_PREFIX = "dedup/fuzzy/verification"
# The verifier anchors each cluster on its longest document. A test that
# exercises the local-representative path must therefore keep the canonical
# longest and unrelated to the members, so that no member matches the anchor
# and the local path is the only way to reach one.
LONG_UNRELATED_CANONICAL = " ".join(f"c{index}" for index in range(400))
TEST_LOCAL_PARAMS = LocalRepresentativeParams(
    maximum_comparisons_per_document=4,
    maximum_representatives_per_cluster=8,
    maximum_local_representative_chars=10_000,
    maximum_local_representative_chars_per_cluster=40_000,
    minimum_local_line_count_ratio=0.8,
)
TEST_STORE_CONFIG = FuzzyVerificationStoreConfig(
    recovery_timeout=30,
    ready_timeout=30,
    lookup_batch_size=2,
    shards_per_worker=1,
)
verify_fuzzy_dups = partial(_verify_fuzzy_dups, store_config=TEST_STORE_CONFIG)


def test_parquet_rows_converts_only_the_row_it_yields(tmp_path, monkeypatch):
    converted: list[tuple[int, int]] = []

    class Scalar:
        def __init__(self, column_index: int, row_index: int):
            self.column_index = column_index
            self.row_index = row_index

        def as_py(self):
            converted.append((self.column_index, self.row_index))
            return [["id-0", "id-1"], ["text-0", "text-1"]][self.column_index][self.row_index]

    class Column:
        def __init__(self, column_index: int):
            self.column_index = column_index

        def __getitem__(self, row_index: int):
            return Scalar(self.column_index, row_index)

    class Batch:
        num_columns = 2
        num_rows = 2

        def column(self, column_index: int):
            return Column(column_index)

        def to_pylist(self):
            return [
                {"id": Column(0)[row_index].as_py(), "text": Column(1)[row_index].as_py()}
                for row_index in range(self.num_rows)
            ]

    class Metadata:
        num_rows = 2

    class Schema:
        names = ("id", "text")

    class ParquetFile:
        metadata = Metadata()
        schema_arrow = Schema()

        def __init__(self, _stream):
            pass

        def iter_batches(self, **_kwargs):
            yield Batch()

    input_path = tmp_path / "input.parquet"
    input_path.touch()
    monkeypatch.setattr(pq, "ParquetFile", ParquetFile)

    rows = _parquet_rows(str(input_path), ["id", "text"])

    assert next(rows) == {"id": "id-0", "text": "text-0"}
    assert converted == [(0, 0), (1, 0)]


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    client = LocalClient()
    try:
        with set_current_client(client):
            yield
    finally:
        client.shutdown()


def _write_source(
    *,
    root: Path,
    name: str,
    shards: dict[str, list[dict]],
) -> tuple[str, NormalizedData]:
    main_dir = root / name / "outputs" / "main"
    dup_dir = root / name / "outputs" / "dups"
    main_dir.mkdir(parents=True)
    dup_dir.mkdir(parents=True)
    for basename, rows in shards.items():
        write_parquet_file(sorted(rows, key=lambda row: row["id"]), str(main_dir / basename))
    source = NormalizedData(main_output_dir=str(main_dir), dup_output_dir=str(dup_dir), counters={})
    return datakit_source_key(source.main_output_dir), source


def _write_candidates(
    *,
    root: Path,
    rows_by_source: dict[str, dict[str, list[dict]]],
) -> FuzzyDupsAttrData:
    sources: dict[str, FuzzyDupsPerSource] = {}
    for source_index, (source_key, shards) in enumerate(sorted(rows_by_source.items())):
        attr_dir = root / "candidates" / f"source_{source_index:03d}"
        attr_dir.mkdir(parents=True)
        for basename, rows in shards.items():
            write_parquet_file(sorted(rows, key=lambda row: row["id"]), str(attr_dir / basename))
        sources[source_key] = FuzzyDupsPerSource(attr_dir=str(attr_dir))
    return FuzzyDupsAttrData(params=TEST_MINHASH_PARAMS, sources=sources, counters={})


def _write_minhash(
    *,
    root: Path,
    name: str,
    source: NormalizedData,
    buckets_by_id: dict[str, list[str]] | None = None,
) -> MinHashAttrData:
    attr_dir = root / "minhash" / name
    attr_dir.mkdir(parents=True)
    for source_path in sorted(Path(source.main_output_dir).glob("*.parquet")):
        rows = [
            {
                "id": row["id"],
                "buckets": (buckets_by_id or {}).get(row["id"], [f"bucket-{row['id']}"]),
            }
            for row in pq.read_table(source_path, columns=["id"]).to_pylist()
        ]
        write_parquet_file(rows, str(attr_dir / source_path.name))
    return MinHashAttrData(
        params=TEST_MINHASH_PARAMS,
        source_key=datakit_source_key(source.main_output_dir),
        attr_dir=str(attr_dir),
        counters={},
    )


def _output_rows(verified, source_key: str) -> list[dict]:
    rows = []
    for path in sorted(Path(verified.sources[source_key].attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_verifier_accepts_only_direct_subset_and_filters_singletons(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    representative = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    accepted = "alpha beta gamma delta epsilon zeta eta theta"
    rejected = "alpha beta gamma delta epsilon zeta eta other theta"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "accepted", "text": accepted},
                {"id": "rejected", "text": rejected},
                {"id": "representative", "text": representative},
            ],
            "part-00001.parquet": [{"id": "singleton", "text": "a document outside all candidate clusters"}],
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "accepted", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "rejected", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "representative", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert verified.counters["dedup/fuzzy/verification/memory_store/workers"] == 2
    assert verified.counters["dedup/fuzzy/verification/memory_store/shards"] == 2
    assert verified.counters["dedup/fuzzy/verification/memory_store/items"] == 3
    assert _output_rows(verified, source_key) == [
        {
            "id": "accepted",
            "dup_doc": True,
            "dup_cluster_id": "cluster-a",
            "dup_representative_id": "representative",
            "dup_representative_source_key": source_key,
            "dup_representative_kind": "cluster_canonical",
            "dup_shared_lsh_buckets": 0,
            "dup_comparisons": 1,
            "dup_member_containment": 1.0,
            "dup_jaccard": 0.6,
            "dup_under_tokenized": False,
            "dup_char_jaccard": None,
            "dup_local_line_count_ratio": None,
        }
    ]
    empty_path = Path(verified.sources[source_key].attr_dir) / "part-00001.parquet"
    assert empty_path.exists()
    empty = pq.read_table(empty_path)
    assert empty.num_rows == 0
    assert empty.schema.names == [
        "id",
        "dup_doc",
        "dup_cluster_id",
        "dup_representative_id",
        "dup_representative_source_key",
        "dup_representative_kind",
        "dup_shared_lsh_buckets",
        "dup_comparisons",
        "dup_member_containment",
        "dup_jaccard",
        "dup_under_tokenized",
        "dup_char_jaccard",
        "dup_local_line_count_ratio",
    ]
    assert verified.counters["dedup/fuzzy/verification/candidate_members"] == 3
    assert verified.counters["dedup/fuzzy/verification/candidate_shards_missing"] == 1
    assert verified.counters["dedup/fuzzy/verification/clusters"] == 1
    assert verified.counters["dedup/fuzzy/verification/cluster_members"] == 3
    assert verified.counters["dedup/fuzzy/verification/decision/accepted"] == 1
    assert verified.counters["dedup/fuzzy/verification/decision/retained_no_match"] == 1
    assert verified.counters["dedup/fuzzy/verification/comparison/containment_below_threshold"] == 1
    assert verified.sources[source_key].source_tag == "source_000"


def test_verifier_removes_a_member_that_contains_the_canonical(tmp_path, monkeypatch):
    """A short canonical must not hide the duplicate that contains it.

    Connected components names each cluster after its minimum content ID, which
    says nothing about length. Anchoring on that canonical made the whole
    cluster unremovable whenever it was the shorter document: the canonical is
    never its own removal candidate, and every longer member failed the length
    test before containment was ever computed.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    canonical_text = "alpha beta gamma delta epsilon zeta"
    longer_text = f"{canonical_text} eta theta iota kappa lambda mu nu xi omicron"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "doc-a", "text": canonical_text},
                {"id": "doc-b", "text": longer_text},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "doc-a", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                    {"id": "doc-b", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["doc-a"]
    assert rows[0]["dup_representative_id"] == "doc-b"
    assert rows[0]["dup_member_containment"] == 1.0
    assert verified.counters[f"{_COUNTER_PREFIX}/representative_longer_than_first"] == 1
    assert verified.counters[f"{_COUNTER_PREFIX}/representative_not_canonical"] == 1


def test_representative_selection_is_stable_across_input_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    representative_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    member_text = "alpha beta gamma delta epsilon zeta eta theta"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "representative", "text": representative_text}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "member", "text": member_text}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [
                    {"id": "representative", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}
                ]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )
    minhash_a = _write_minhash(root=tmp_path, name="source-a", source=source_a)
    minhash_b = _write_minhash(root=tmp_path, name="source-b", source=source_b)

    first = verify_fuzzy_dups(
        normalized_sources={"z-source-a": source_a, "a-source-b": source_b},
        minhash_sources={
            "z-source-a": minhash_a,
            "a-source-b": minhash_b,
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified-first"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
        max_workers=1,
        pipeline_shards_per_worker=1,
    )
    second = verify_fuzzy_dups(
        normalized_sources={"a-source-b": source_b, "z-source-a": source_a},
        minhash_sources={
            "a-source-b": minhash_b,
            "z-source-a": minhash_a,
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified-second"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
        max_workers=1,
        pipeline_shards_per_worker=2,
    )

    assert first.counters["dedup/fuzzy/verification/pipeline/source_shards"] == 2
    assert first.counters["dedup/fuzzy/verification/pipeline/shards"] == 1
    assert second.counters["dedup/fuzzy/verification/pipeline/shards"] == 2
    assert _output_rows(first, source_a_key) == _output_rows(second, source_a_key) == []
    assert _output_rows(first, source_b_key) == _output_rows(second, source_b_key)
    assert _output_rows(first, source_b_key)[0]["dup_representative_source_key"] == source_a_key


def test_verifier_defers_exact_copies_to_global_exact_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    text = "alpha beta gamma delta epsilon zeta eta theta"
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "same", "text": text}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same", "text": text}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [{"id": "same", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "same", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"z-source-a": source_a, "a-source-b": source_b},
        minhash_sources={
            "z-source-a": _write_minhash(root=tmp_path, name="source-a", source=source_a),
            "a-source-b": _write_minhash(root=tmp_path, name="source-b", source=source_b),
        },
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_a_key) == []
    assert _output_rows(verified, source_b_key) == []
    assert verified.counters["dedup/fuzzy/verification/decision/delegated_global_exact"] == 1


def test_local_verifier_rejects_different_token_sequences(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    local_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    duplicate_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": local_text},
                {"id": "b-duplicate", "text": duplicate_text},
                {"id": "canonical", "text": LONG_UNRELATED_CANONICAL},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-duplicate", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-local": ["local-a", "local-b"],
            "b-duplicate": ["local-a", "local-b"],
            "canonical": ["canonical"],
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/comparison/local_token_sequence_differs"] == 1


def test_local_verifier_accepts_equal_token_sequences_with_different_whitespace(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokens = [f"t{index}" for index in range(40)]
    representative_text = "   ".join(tokens)
    member_text = "\t".join(tokens)
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": representative_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": LONG_UNRELATED_CANONICAL},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={"a-local": ["local"], "b-member": ["local"], "canonical": ["canonical"]},
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    row = _output_rows(verified, source_key)[0]
    assert row["id"] == "b-member"
    assert row["dup_representative_kind"] == "local_representative"
    assert row["dup_local_line_count_ratio"] == 1.0


def test_local_verifier_rejects_collapsed_line_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokens = [f"t{index}" for index in range(40)]
    representative_text = "\n".join(tokens)
    member_text = " ".join(tokens)
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": representative_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": LONG_UNRELATED_CANONICAL},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={"a-local": ["local"], "b-member": ["local"], "canonical": ["canonical"]},
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/comparison/local_line_count_ratio_below_threshold"] == 1


def test_verifier_rejects_token_ngram_saturation(tmp_path, monkeypatch):
    """Two documents over a tiny vocabulary contain each other by exhaustion.

    20,000 and 40,000 random binary tokens share every distinct token 3-gram
    and character 13-gram. The verifier must retain the independent member.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    def digit_text(seed: int, size: int) -> str:
        random = Random(seed)
        return " ".join(str(random.randrange(2)) for _ in range(size))

    representative_text = digit_text(1, 40_000)
    member_text = digit_text(2, 20_000)
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-local", "text": representative_text},
                {"id": "b-member", "text": member_text},
                {"id": "canonical", "text": LONG_UNRELATED_CANONICAL},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "a-local", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "b-member", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "canonical", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={"a-local": ["local"], "b-member": ["local"], "canonical": ["canonical"]},
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=8,
        maximum_local_representative_chars=100_000,
        maximum_local_representative_chars_per_cluster=200_000,
        minimum_local_line_count_ratio=0.8,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/comparison/saturated_token_sequence_not_contained"] == 1


def test_local_representative_selection_is_stable_across_input_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    local_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    duplicate_text = "\t".join(local_text.split())
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "canonical", "text": LONG_UNRELATED_CANONICAL}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "a-local", "text": local_text}]},
    )
    source_c_key, source_c = _write_source(
        root=tmp_path,
        name="source-c",
        shards={"part-00000.parquet": [{"id": "z-duplicate", "text": duplicate_text}]},
    )
    sources = {
        "source-a": (source_a_key, source_a),
        "source-b": (source_b_key, source_b),
        "source-c": (source_c_key, source_c),
    }
    ids = {"source-a": "canonical", "source-b": "a-local", "source-c": "z-duplicate"}
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": ids[source_name],
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": source_name == "source-a",
                    }
                ]
            }
            for source_name, (source_key, _source) in sources.items()
        },
    )
    minhash_sources = {
        source_name: _write_minhash(
            root=tmp_path,
            name=source_name,
            source=source,
            buckets_by_id={
                "canonical": ["canonical"],
                "a-local": ["local"],
                "z-duplicate": ["local"],
            },
        )
        for source_name, (_source_key, source) in sources.items()
    }

    first = verify_fuzzy_dups(
        normalized_sources={source_name: source for source_name, (_source_key, source) in sources.items()},
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=str(tmp_path / "verified-first"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )
    second = verify_fuzzy_dups(
        normalized_sources={
            source_name: source for source_name, (_source_key, source) in reversed(list(sources.items()))
        },
        minhash_sources={source_name: minhash_sources[source_name] for source_name in reversed(minhash_sources)},
        candidates=candidates,
        output_path=str(tmp_path / "verified-second"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(first, source_c_key) == _output_rows(second, source_c_key)
    assert _output_rows(first, source_c_key)[0]["dup_representative_source_key"] == source_b_key
    assert _output_rows(first, source_c_key)[0]["dup_representative_kind"] == "local_representative"


def test_verifier_ranks_local_nominees_and_bounds_comparisons(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    b_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    target_text = "\t".join(b_text.split())
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "a-first", "text": "red orange yellow green blue indigo violet black white gray"},
                {"id": "b-best", "text": b_text},
                {"id": "canonical", "text": LONG_UNRELATED_CANONICAL},
                {"id": "z-target", "text": target_text},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": candidate_id, "dup_cluster_id": "cluster-a", "is_cluster_canonical": is_canonical}
                    for candidate_id, is_canonical in (
                        ("a-first", False),
                        ("b-best", False),
                        ("canonical", True),
                        ("z-target", False),
                    )
                ]
            }
        },
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-first": ["shared-a"],
            "b-best": ["shared-b1", "shared-b2"],
            "canonical": ["canonical"],
            "z-target": ["shared-a", "shared-b1", "shared-b2"],
        },
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=8,
        maximum_local_representative_chars=10_000,
        maximum_local_representative_chars_per_cluster=40_000,
        minimum_local_line_count_ratio=0.8,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["z-target"]
    assert rows[0]["dup_representative_id"] == "b-best"
    assert rows[0]["dup_comparisons"] == 2
    assert verified.counters["dedup/fuzzy/verification/comparison_limit_reached"] == 1
    assert (
        max(
            int(key.rsplit("/", maxsplit=1)[1])
            for key in verified.counters
            if key.startswith("dedup/fuzzy/verification/comparisons_per_document/")
        )
        == 2
    )


def test_verifier_bounds_representative_count_and_text(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    long_text = " ".join(f"token-{index}" for index in range(30))
    short_duplicate = " ".join(f"token-{index}" for index in range(29))
    rows = [
        {"id": "a-count", "text": "red orange yellow green blue indigo violet"},
        {"id": "b-count", "text": long_text},
        {"id": "canonical-count", "text": LONG_UNRELATED_CANONICAL},
        {"id": "z-count", "text": short_duplicate},
        {"id": "a-long", "text": long_text},
        {"id": "canonical-long", "text": LONG_UNRELATED_CANONICAL},
        {"id": "z-long", "text": short_duplicate},
    ]
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={"part-00000.parquet": rows},
    )
    candidate_rows = [
        {
            "id": row["id"],
            "dup_cluster_id": "cluster-count" if row["id"].endswith("count") else "cluster-long",
            "is_cluster_canonical": row["id"].startswith("canonical"),
        }
        for row in rows
    ]
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={source_key: {"part-00000.parquet": candidate_rows}},
    )
    minhash = _write_minhash(
        root=tmp_path,
        name="source",
        source=source,
        buckets_by_id={
            "a-count": ["count-a"],
            "b-count": ["count-b"],
            "canonical-count": ["canonical-count"],
            "z-count": ["count-b"],
            "a-long": ["long"],
            "canonical-long": ["canonical-long"],
            "z-long": ["long"],
        },
    )
    local_params = LocalRepresentativeParams(
        maximum_comparisons_per_document=2,
        maximum_representatives_per_cluster=2,
        maximum_local_representative_chars=100,
        maximum_local_representative_chars_per_cluster=1_000,
        minimum_local_line_count_ratio=0.8,
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=local_params,
    )

    assert _output_rows(verified, source_key) == []
    assert verified.counters["dedup/fuzzy/verification/representative_skipped/cluster_limit"] >= 1
    assert verified.counters["dedup/fuzzy/verification/representative_skipped/document_chars"] >= 1


def test_repeated_noncanonical_ids_stay_available_as_local_representatives(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    exact_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
    near_text = "\t".join(exact_text.split())
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "canonical", "text": LONG_UNRELATED_CANONICAL}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same", "text": exact_text}]},
    )
    source_c_key, source_c = _write_source(
        root=tmp_path,
        name="source-c",
        shards={"part-00000.parquet": [{"id": "same", "text": exact_text}]},
    )
    source_d_key, source_d = _write_source(
        root=tmp_path,
        name="source-d",
        shards={"part-00000.parquet": [{"id": "z-near", "text": near_text}]},
    )
    sources = {
        "source-a": (source_a_key, source_a),
        "source-b": (source_b_key, source_b),
        "source-c": (source_c_key, source_c),
        "source-d": (source_d_key, source_d),
    }
    id_by_source = {"source-a": "canonical", "source-b": "same", "source-c": "same", "source-d": "z-near"}
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": id_by_source[source_name],
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": source_name == "source-a",
                    }
                ]
            }
            for source_name, (source_key, source) in sources.items()
        },
    )
    minhash_sources = {
        source_name: _write_minhash(
            root=tmp_path,
            name=source_name,
            source=source,
            buckets_by_id={
                "canonical": ["canonical"],
                "same": ["local"],
                "z-near": ["local"],
            },
        )
        for source_name, (_source_key, source) in sources.items()
    }

    verified = verify_fuzzy_dups(
        normalized_sources={source_name: source for source_name, (_source_key, source) in sources.items()},
        minhash_sources=minhash_sources,
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    assert _output_rows(verified, source_b_key) == []
    assert _output_rows(verified, source_c_key) == []
    near_rows = _output_rows(verified, source_d_key)
    assert [row["id"] for row in near_rows] == ["z-near"]
    assert near_rows[0]["dup_representative_id"] == "same"
    assert near_rows[0]["dup_representative_source_key"] == source_b_key
    assert verified.counters["dedup/fuzzy/verification/decision/retained_no_match"] == 1
    assert verified.counters["dedup/fuzzy/verification/decision/delegated_global_exact"] == 1


def test_verifier_rejects_different_text_for_canonical_content_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_a_key, source_a = _write_source(
        root=tmp_path,
        name="source-a",
        shards={"part-00000.parquet": [{"id": "same-id", "text": "canonical text"}]},
    )
    source_b_key, source_b = _write_source(
        root=tmp_path,
        name="source-b",
        shards={"part-00000.parquet": [{"id": "same-id", "text": "different text"}]},
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_a_key: {
                "part-00000.parquet": [{"id": "same-id", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True}]
            },
            source_b_key: {
                "part-00000.parquet": [{"id": "same-id", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False}]
            },
        },
    )

    with pytest.raises(ZephyrWorkerError, match="different text for content ID"):
        verify_fuzzy_dups(
            normalized_sources={"source-a": source_a, "source-b": source_b},
            minhash_sources={
                "source-a": _write_minhash(root=tmp_path, name="source-a", source=source_a),
                "source-b": _write_minhash(root=tmp_path, name="source-b", source=source_b),
            },
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )


def test_verifier_rejects_more_than_one_candidate_canonical(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "first", "text": "alpha beta gamma delta"},
                {"id": "second", "text": "alpha beta gamma delta epsilon"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {
                        "id": candidate_id,
                        "dup_cluster_id": "cluster-a",
                        "is_cluster_canonical": is_canonical,
                    }
                    for candidate_id, is_canonical in (("first", True), ("second", True))
                ]
            }
        },
    )

    with pytest.raises(ZephyrWorkerError, match="has more than one canonical member"):
        verify_fuzzy_dups(
            normalized_sources={"source": source},
            minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )


def test_verifier_rejects_mismatched_source_sets(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={"part-00000.parquet": [{"id": "doc", "text": "some text"}]},
    )
    candidates = _write_candidates(root=tmp_path, rows_by_source={source_key: {}})
    candidates.sources["extra/source"] = FuzzyDupsPerSource(attr_dir=str(tmp_path / "extra"))

    with pytest.raises(ValueError, match="source sets differ"):
        verify_fuzzy_dups(
            normalized_sources={"source": source},
            minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
            candidates=candidates,
            output_path=str(tmp_path / "verified"),
            verification_params=FuzzyVerificationParams(),
            local_representative_params=TEST_LOCAL_PARAMS,
        )


def test_verifier_falls_back_when_a_cluster_lost_its_canonical(tmp_path, monkeypatch):
    """Connected components can leave a cluster whose label owner moved away.

    Those members are still candidates for one another, so verification anchors
    on the longest of them and reports ``cluster_longest`` instead of failing
    the job.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "aaa", "text": "alpha beta gamma delta epsilon"},
                {"id": "bbb", "text": "alpha beta gamma delta"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "aaa", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                    {"id": "bbb", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["bbb"]
    assert rows[0]["dup_doc"] is True
    assert rows[0]["dup_representative_id"] == "aaa"
    assert rows[0]["dup_representative_kind"] == "cluster_longest"
    assert verified.counters["dedup/fuzzy/verification/representative_not_canonical"] == 1


def test_verifier_accepts_repeated_source_ids(tmp_path, monkeypatch):
    """Sources that normalize with DedupMode.NONE keep byte-identical rows.

    Their IDs come from an upstream content hash, so a repeat means the same
    text twice and the join can bind either row.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "aaa", "text": "alpha beta gamma delta"},
                {"id": "aaa", "text": "alpha beta gamma delta"},
                {"id": "bbb", "text": "alpha beta gamma"},
            ]
        },
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "aaa", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                    {"id": "bbb", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": _write_minhash(root=tmp_path, name="source", source=source)},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["bbb"]
    assert rows[0]["dup_doc"] is True
    assert rows[0]["dup_representative_id"] == "aaa"
    assert verified.counters["dedup/fuzzy/verification/repeated_source_ids"] >= 1


def test_verifier_reads_candidate_text_from_unordered_source_shard(tmp_path, monkeypatch):
    """Candidate text lookup must not depend on normalized row order."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_key, source = _write_source(
        root=tmp_path,
        name="source",
        shards={
            "part-00000.parquet": [
                {"id": "aaa", "text": "alpha beta gamma delta epsilon zeta"},
                {"id": "bbb", "text": "alpha beta gamma delta"},
            ]
        },
    )
    minhash = _write_minhash(root=tmp_path, name="source", source=source)
    write_parquet_file(
        [
            {"id": "bbb", "text": "alpha beta gamma delta"},
            {"id": "aaa", "text": "alpha beta gamma delta epsilon zeta"},
        ],
        str(Path(source.main_output_dir) / "part-00000.parquet"),
    )
    candidates = _write_candidates(
        root=tmp_path,
        rows_by_source={
            source_key: {
                "part-00000.parquet": [
                    {"id": "aaa", "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
                    {"id": "bbb", "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
                ]
            }
        },
    )

    verified = verify_fuzzy_dups(
        normalized_sources={"source": source},
        minhash_sources={"source": minhash},
        candidates=candidates,
        output_path=str(tmp_path / "verified"),
        verification_params=FuzzyVerificationParams(),
        local_representative_params=TEST_LOCAL_PARAMS,
    )

    rows = _output_rows(verified, source_key)
    assert [row["id"] for row in rows] == ["bbb"]
    assert rows[0]["dup_representative_id"] == "aaa"


def test_candidate_text_load_streams_unordered_source_shard(tmp_path, monkeypatch):
    """The first match is available before the normalized scan completes."""
    candidate_path = tmp_path / "candidates.parquet"
    candidate_path.touch()
    shard = VerificationShard(
        file_idx=7,
        normalized_path=str(tmp_path / "normalized.parquet"),
        candidate_path=str(candidate_path),
        minhash_path=str(tmp_path / "minhash.parquet"),
        output_path=str(tmp_path / "output.parquet"),
        source_key="source",
        source_tag="source_000",
    )

    def candidate_rows(path, columns, *, repeated_ids=False):
        del path, columns, repeated_ids
        yield {"id": "aaa"}
        yield {"id": "bbb"}

    def normalized_rows(path, columns):
        del path, columns
        yield {"id": "bbb", "text": "beta"}
        raise AssertionError("candidate loading read past the first match")

    monkeypatch.setattr(
        "marin.processing.classification.deduplication.verify_fuzzy_dups._rows",
        candidate_rows,
    )
    monkeypatch.setattr(
        "marin.processing.classification.deduplication.verify_fuzzy_dups._parquet_rows",
        normalized_rows,
    )

    key, text = next(_candidate_documents([shard]))
    assert key == (7, "bbb")
    assert _decompress_document_text(text) == "beta"


def test_candidate_text_load_releases_arrow_pages_after_each_shard(tmp_path, monkeypatch):
    candidate_path = tmp_path / "candidates.parquet"
    candidate_path.touch()
    shard = VerificationShard(
        file_idx=7,
        normalized_path=str(tmp_path / "normalized.parquet"),
        candidate_path=str(candidate_path),
        minhash_path=str(tmp_path / "minhash.parquet"),
        output_path=str(tmp_path / "output.parquet"),
        source_key="source",
        source_tag="source_000",
    )

    def candidate_rows(path, columns, *, repeated_ids=False):
        del path, columns, repeated_ids
        yield {"id": "aaa"}

    def normalized_rows(path, columns):
        del path, columns
        yield {"id": "aaa", "text": "alpha"}

    class MemoryPool:
        def __init__(self):
            self.release_count = 0

        def release_unused(self):
            self.release_count += 1

    previous_pool = MemoryPool()
    memory_pool = MemoryPool()
    monkeypatch.setattr(
        "marin.processing.classification.deduplication.verify_fuzzy_dups._rows",
        candidate_rows,
    )
    monkeypatch.setattr(
        "marin.processing.classification.deduplication.verify_fuzzy_dups._parquet_rows",
        normalized_rows,
    )
    selected_pools = []
    monkeypatch.setattr(pa, "default_memory_pool", lambda: previous_pool)
    monkeypatch.setattr(pa, "system_memory_pool", lambda: memory_pool)
    monkeypatch.setattr(pa, "set_memory_pool", selected_pools.append)

    assert len(list(_candidate_documents([shard]))) == 1
    assert memory_pool.release_count == 2
    assert selected_pools == [memory_pool, previous_pool]


def test_candidate_text_load_shares_arrow_pool_across_threads(monkeypatch):
    class MemoryPool:
        def __init__(self):
            self.release_count = 0

        def release_unused(self):
            self.release_count += 1

    previous_pool = MemoryPool()
    memory_pool = MemoryPool()
    selected_pools = []
    monkeypatch.setattr(pa, "default_memory_pool", lambda: previous_pool)
    monkeypatch.setattr(pa, "system_memory_pool", lambda: memory_pool)
    monkeypatch.setattr(pa, "set_memory_pool", selected_pools.append)
    load_barrier = threading.Barrier(2)

    def use_memory_pool():
        with _system_arrow_memory_pool() as selected_pool:
            load_barrier.wait(timeout=5)
            return selected_pool

    with ThreadPoolExecutor(max_workers=2) as executor:
        pools = list(executor.map(lambda _: use_memory_pool(), range(2)))

    assert pools == [memory_pool, memory_pool]
    assert memory_pool.release_count == 1
    assert selected_pools == [memory_pool, previous_pool]
