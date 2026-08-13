# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral smoke for the Datakit clustered store.

Builds tiny synthetic co-partitioned inputs (tokenize / decon / cluster /
quality / exact dedup / fuzzy dedup) on the local filesystem, runs ``build_clustered_store``
through a local Zephyr context, and checks the externally observable contract:
the right docs survive filtering, land in the right ``(cluster, quality)``
bucket, round-trip out of the materialized caches, and task batching changes
the leaf-cache count without losing or duplicating data.
"""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.store.cache import TreeCache
from marin.datakit.decon import DeconAttributes
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    VerifiedFuzzyDupsAttrData,
    VerifiedFuzzyDupsPerSource,
)
from marin.processing.tokenize.attributes import TokenizedAttrData

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData
from experiments.datakit.store.bucket_writer import BucketSpillRun, write_bucket_cache, write_bucket_cache_from_spills
from experiments.datakit.store.datakit_store import (
    ClusteredStoreData,
    _iter_tokenized_documents,
    build_clustered_store,
)
from experiments.datakit.store.verify_store import discover_cache_paths, verify_store

CLUSTER_VIEW = 2  # cluster column "cluster_2", clusters {0, 1}
SPLIT = "train"
EXEMPLAR = {"input_ids": np.array([0], dtype=np.int32)}

# One doc = (id, cluster, quality_bucket, contaminated, verified duplicate, token value, length).
# ``quality_bucket`` is the precomputed calibrated bucket the scorer writes as a
# column (consumed as-is by the store -- no score->bucket mapping).
# ``verified duplicate``: True -> dropped; False or None -> absent from the sparse attrs.
# Each surviving doc's tokens are a run of its unique ``token`` value so we can
# recover identity from the cache and confirm dropped docs never appear.
_Doc = tuple[str, int, int, bool, bool | None, int, int]

SHARD0: list[_Doc] = [
    ("d0", 0, 0, False, None, 100, 3),  # -> (0, 0)
    ("d1", 0, 4, False, None, 101, 5),  # -> (0, 4)
    ("d2", 1, 2, True, None, 902, 9),  # contaminated -> dropped
    ("d3", 1, 2, False, False, 103, 2),  # -> (1, 2)
    ("d6", 1, 4, False, False, 906, 6),  # exact canonical -> (1, 4)
    ("d8", 0, 0, False, None, 108, 3),  # exact canonical -> (0, 0)
]
SHARD1: list[_Doc] = [
    ("d4", 0, 0, False, None, 104, 4),  # -> (0, 0)
    ("d5", 1, 2, False, True, 905, 8),  # verified fuzzy duplicate -> dropped
    ("d6", 1, 4, False, False, 106, 6),  # exact duplicate -> dropped
    ("d7", 0, 4, False, None, 107, 7),  # -> (0, 4)
    ("d8", 0, 0, False, None, 908, 3),  # exact duplicate without fuzzy attrs -> dropped
]

# Expected surviving buckets: {(cluster, quality): {token_value: length}}.
EXPECTED: dict[tuple[int, int], dict[int, int]] = {
    (0, 0): {100: 3, 104: 4, 108: 3},
    (0, 4): {101: 5, 107: 7},
    (1, 2): {103: 2},
    (1, 4): {906: 6},
}
EXACT_DUPLICATES = {("part-00001-of-00002.parquet", "d6"), ("part-00001-of-00002.parquet", "d8")}
DROPPED_TOKEN_VALUES = {902, 905, 106, 908}

# Documents the tokenizer split across several Parquet rows, and how many rows
# each one occupies. The tokenize dataset is the only input with such rows: the
# other inputs stay one row per document. The expectations above do not change,
# because the store must join these rows back into one document -- one that
# survives (d1) and one that a filter drops (d2, contaminated).
CHUNKED_DOCS = {"d1": 2, "d2": 3}


def _write_parquet(path: str, table: pa.Table) -> None:
    pq.write_table(table, path)


def _tokenize_rows(docs: list[_Doc]) -> tuple[list[str], list[int], list[list[int]]]:
    """Build the tokenize shard's rows, splitting the documents in ``CHUNKED_DOCS``."""
    ids: list[str] = []
    chunk_indices: list[int] = []
    token_rows: list[list[int]] = []
    for doc in docs:
        tokens = [doc[5]] * doc[6]
        num_chunks = CHUNKED_DOCS.get(doc[0], 1)
        size = -(-len(tokens) // num_chunks)  # ceiling, so the last chunk is the short one
        for chunk_index, start in enumerate(range(0, len(tokens), size)):
            ids.append(doc[0])
            chunk_indices.append(chunk_index)
            token_rows.append(tokens[start : start + size])
    return ids, chunk_indices, token_rows


def _write_shard(dirs: dict[str, str], basename: str, docs: list[_Doc]) -> None:
    """Write one shard's co-partitioned Parquet inputs in the same document order."""
    ids = [d[0] for d in docs]
    tok_ids, tok_chunk_indices, tok_token_rows = _tokenize_rows(docs)
    _write_parquet(
        f"{dirs['tokenize']}/{basename}",
        pa.table(
            {
                "id": tok_ids,
                "chunk_index": pa.array(tok_chunk_indices, type=pa.int32()),
                "input_ids": pa.array(tok_token_rows),
            }
        ),
    )
    _write_parquet(
        f"{dirs['decon']}/{basename}",
        pa.table({"id": ids, "contaminated": pa.array([d[3] for d in docs])}),
    )
    _write_parquet(
        f"{dirs['cluster']}/{basename}",
        pa.table({"id": ids, f"cluster_{CLUSTER_VIEW}": pa.array([d[1] for d in docs], type=pa.int32())}),
    )
    # Quality parquet: flat {id, score, quality_bucket}; the store reads the
    # precomputed quality_bucket column (score is carried for schema fidelity).
    _write_parquet(
        f"{dirs['quality']}/{basename}",
        pa.table(
            {
                "id": ids,
                "score": pa.array([float(d[2]) for d in docs], type=pa.float64()),
                "quality_bucket": pa.array([d[2] for d in docs], type=pa.int32()),
            }
        ),
    )
    exact_duplicates = [doc_id for doc_id in ids if (basename, doc_id) in EXACT_DUPLICATES]
    if exact_duplicates:
        _write_parquet(
            f"{dirs['exact_dedup']}/{basename}",
            pa.Table.from_pylist(
                [{"id": doc_id, "dup_doc": True} for doc_id in exact_duplicates],
                schema=pa.schema(
                    [
                        pa.field("id", pa.string(), nullable=False),
                        pa.field("dup_doc", pa.bool_(), nullable=False),
                    ]
                ),
            ),
        )
    # Verified dedup is sparse: only documents that must be removed appear.
    marked = [d[0] for d in docs if d[4]]
    if marked:
        _write_parquet(
            f"{dirs['dedup']}/{basename}",
            pa.table(
                {
                    "id": marked,
                    "dup_doc": pa.array([True] * len(marked)),
                }
            ),
        )


def _build_inputs(tmp_path):
    """Materialize the synthetic inputs and return the typed artifacts."""
    main_dir = str(tmp_path / "src")
    source_key = datakit_source_key(main_dir)
    dirs = {k: str(tmp_path / k) for k in ("tokenize", "decon", "cluster", "quality", "exact_dedup", "dedup")}
    tok_train = f"{dirs['tokenize']}/{SPLIT}"
    for d in (*dirs.values(), tok_train):
        os.makedirs(d, exist_ok=True)

    shard_dirs = {**dirs, "tokenize": tok_train}
    _write_shard(shard_dirs, "part-00000-of-00002.parquet", SHARD0)
    _write_shard(shard_dirs, "part-00001-of-00002.parquet", SHARD1)

    tokenize = {
        "src": TokenizedAttrData(
            output_dirs={SPLIT: tok_train},
            source_keys={SPLIT: source_key},
            tokenizer="dummy",
            tokenizer_backend="huggingface",
            counters={},
        )
    }
    decontam = {
        "src": DeconAttributes(
            main_output_dir=dirs["decon"],
            flagged_output_dir="",
            num_partitions=2,
            eval_hash_index_path="",
            counters={},
        )
    }
    cluster_assign = {
        "src": AssignmentAttrData(
            output_dir=dirs["cluster"],
            source_key=source_key,
            embedding_output_dir="",
            k_train=CLUSTER_VIEW,
            k_views=[],
        )
    }
    quality = {
        "src": QualityScores(
            main_output_dir=dirs["quality"],
            samples_output_dir="",
            model_dir="model",
            calib_file="calib.json",
            bucket_edges=[0.2, 0.4, 0.6, 0.8],
            counters={},
        )
    }
    dedup = VerifiedFuzzyDupsAttrData(
        verification=FuzzyVerificationParams(),
        local_representatives=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
        sources={source_key: VerifiedFuzzyDupsPerSource(attr_dir=dirs["dedup"], source_tag="source_000")},
        counters={},
    )
    exact_dedup = GlobalExactDedupData(
        sources={
            source_key: ExactDupsPerSource(
                attr_dir=dirs["exact_dedup"],
            )
        },
        counters={},
    )
    return tokenize, decontam, cluster_assign, quality, exact_dedup, dedup


def _read_bucket_tokens(bucket_root: str) -> list[np.ndarray]:
    cache = TreeCache.load(bucket_root, EXEMPLAR)
    return [np.asarray(doc["input_ids"]) for doc in cache]


def test_store_filters_routes_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokenize, decontam, cluster_assign, quality, exact_dedup, dedup = _build_inputs(tmp_path)
    output_path = str(tmp_path / "store")

    artifact = build_clustered_store(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        exact_dedup=exact_dedup,
        dedup=dedup,
        output_path=output_path,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
    )

    by_key = {(b.cluster_id, b.quality_bucket): b for b in artifact.buckets}
    assert set(by_key) == set(EXPECTED), "exactly the surviving buckets, nothing for the dropped docs"

    all_tokens: list[int] = []
    for key, expected_docs in EXPECTED.items():
        bucket = by_key[key]
        assert bucket.total_elements == len(expected_docs)
        assert bucket.total_tokens == sum(expected_docs.values())

        # Round-trip: the bucket's caches hold exactly the expected docs.
        recovered = _read_bucket_tokens(bucket.path)
        assert len(recovered) == len(expected_docs)
        recovered_by_value = {int(arr[0]): len(arr) for arr in recovered}
        assert recovered_by_value == expected_docs
        for arr in recovered:
            assert len(set(arr.tolist())) == 1  # each doc is a run of its unique token value
            all_tokens.extend(arr.tolist())

    assert DROPPED_TOKEN_VALUES.isdisjoint(all_tokens), "filtered docs must be absent"
    assert artifact.counters["datakit_store/records_in"] == len(SHARD0) + len(SHARD1)
    assert artifact.counters["datakit_store/contaminated_dropped"] == 1
    assert artifact.counters["datakit_store/exact_duplicate_dropped"] == 2
    assert artifact.counters["datakit_store/fuzzy_duplicate_dropped"] == 1
    assert artifact.counters["datakit_store/records_out"] == sum(len(docs) for docs in EXPECTED.values())
    assert artifact.counters["datakit_store/tokens_out"] == sum(
        length for docs in EXPECTED.values() for length in docs.values()
    )
    record = json.loads((tmp_path / "store" / ".artifact.json").read_text())
    assert record["result"]["cache_path"] == "store"
    assert all(bucket["path"].startswith("store/") for bucket in record["result"]["buckets"])
    loaded = read_artifact(output_path, ClusteredStoreData)
    assert loaded.cache_path == output_path
    assert [bucket.path for bucket in loaded.buckets] == [bucket.path for bucket in artifact.buckets]


def test_store_batches_input_shards_into_one_leaf_per_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokenize, decontam, cluster_assign, quality, exact_dedup, dedup = _build_inputs(tmp_path)
    output_path = str(tmp_path / "store_batched")

    artifact = build_clustered_store(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        exact_dedup=exact_dedup,
        dedup=dedup,
        output_path=output_path,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        task_count=1,
        partition_processes=2,
    )

    by_key = {(b.cluster_id, b.quality_bucket): b for b in artifact.buckets}
    assert all(bucket.n_shards == 1 for bucket in artifact.buckets)
    for key, expected_docs in EXPECTED.items():
        recovered = {int(arr[0]): len(arr) for arr in _read_bucket_tokens(by_key[key].path)}
        assert recovered == expected_docs


def test_bucket_writer_roundtrips_aligned_chunks(tmp_path):
    cache_path = str(tmp_path / "leaf")
    documents = [np.arange(3, dtype=np.int32), np.arange(3, 9, dtype=np.int32), np.array([9], dtype=np.int32)]

    ledger = write_bucket_cache(
        cache_path,
        documents,
        [len(document) for document in documents],
        write_chunk_elements=4,
        max_pending_commits=2,
    )

    assert ledger.total_num_rows == 3
    assert ledger.field_counts == {"input_ids": 10}
    recovered = [np.asarray(row["input_ids"]) for row in TreeCache.load(cache_path, EXEMPLAR)]
    assert [row.tolist() for row in recovered] == [row.tolist() for row in documents]


def test_bucket_writer_roundtrips_local_spill_runs(tmp_path):
    documents = [np.arange(3, dtype=np.int32), np.arange(3, 9, dtype=np.int32), np.array([9], dtype=np.int32)]
    runs = []
    for index, group in enumerate((documents[:2], documents[2:])):
        data_path = tmp_path / f"run-{index}.tokens"
        lengths_path = tmp_path / f"run-{index}.lengths"
        np.concatenate(group).tofile(data_path)
        np.asarray([len(document) for document in group], dtype=np.int64).tofile(lengths_path)
        runs.append(
            BucketSpillRun(
                data_path=str(data_path),
                lengths_path=str(lengths_path),
                rows=len(group),
                tokens=sum(len(document) for document in group),
            )
        )

    cache_path = str(tmp_path / "leaf")
    ledger = write_bucket_cache_from_spills(cache_path, runs, write_chunk_elements=4, max_pending_commits=2)

    assert ledger.total_num_rows == 3
    assert ledger.field_counts == {"input_ids": 10}
    recovered = [np.asarray(row["input_ids"]) for row in TreeCache.load(cache_path, EXEMPLAR)]
    assert [row.tolist() for row in recovered] == [row.tolist() for row in documents]


def test_store_verifier_matches_reordered_documents(tmp_path):
    documents = [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32), np.array([1, 2], dtype=np.int32)]
    input_path = str(tmp_path / "input.parquet")
    _write_parquet(
        input_path,
        pa.table(
            {
                "id": ["first", "first", "second", "duplicate"],
                "chunk_index": pa.array([0, 1, 0, 0], type=pa.int32()),
                "input_ids": pa.array([[1], [2], documents[1], documents[2]], type=pa.list_(pa.int32())),
            }
        ),
    )
    output_root = tmp_path / "output"
    write_bucket_cache(str(output_root / "bucket-0"), [documents[2]], [len(documents[2])], write_chunk_elements=4)
    write_bucket_cache(
        str(output_root / "bucket-1"),
        [documents[1], documents[0]],
        [len(documents[1]), len(documents[0])],
        write_chunk_elements=4,
    )

    result = verify_store([input_path], discover_cache_paths(str(output_root)))

    assert result.matches
    assert result.input_documents == result.output_documents == 3
    assert result.input_tokens == result.output_tokens == 5
    assert result.mismatches == []


def test_store_verifier_reports_same_length_token_corruption(tmp_path):
    documents = [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    input_path = str(tmp_path / "input.parquet")
    _write_parquet(
        input_path,
        pa.table(
            {
                "id": ["intact", "corrupted"],
                "input_ids": pa.array(documents, type=pa.list_(pa.int32())),
            }
        ),
    )
    output_root = tmp_path / "output"
    write_bucket_cache(
        str(output_root / "bucket-0"),
        [documents[0], np.array([3, 5], dtype=np.int32)],
        [2, 2],
        write_chunk_elements=4,
    )

    result = verify_store([input_path], discover_cache_paths(str(output_root)))

    assert not result.matches
    assert result.input_documents == result.output_documents == 2
    assert result.input_tokens == result.output_tokens == 4
    assert result.missing_documents == 1
    assert result.extra_documents == 1
    assert len(result.mismatches) == 2
    assert {mismatch.sample_id for mismatch in result.mismatches} == {"corrupted", None}


def test_store_ignores_orphans_and_resumes_completed_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokenize, decontam, cluster_assign, quality, exact_dedup, dedup = _build_inputs(tmp_path)
    output_path = str(tmp_path / "store_resumed")
    orphan_path = f"{output_path}/cluster=0/quality=0/part-orphan"
    write_bucket_cache(orphan_path, [np.array([999], dtype=np.int32)], [1], write_chunk_elements=4)

    arguments = dict(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        exact_dedup=exact_dedup,
        dedup=dedup,
        output_path=output_path,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        task_count=1,
    )
    first = build_clustered_store(**arguments)
    second = build_clustered_store(**arguments)

    bucket = next(bucket for bucket in second.buckets if (bucket.cluster_id, bucket.quality_bucket) == (0, 0))
    recovered = {int(tokens[0]) for tokens in _read_bucket_tokens(bucket.path)}
    assert recovered == set(EXPECTED[(0, 0)])
    assert first.counters["datakit_store/tokens_out"] == second.counters["datakit_store/tokens_out"]
    assert second.counters["datakit_store/tasks_resumed"] == 1


@pytest.mark.parametrize("label", ["exact_dedup", "dedup"])
def test_store_rejects_dedup_source_set_mismatch(tmp_path, monkeypatch, label):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokenize, decontam, cluster_assign, quality, exact_dedup, dedup = _build_inputs(tmp_path)
    if label == "exact_dedup":
        exact_dedup.sources["extra/source"] = ExactDupsPerSource(attr_dir=str(tmp_path / "extra"))
    else:
        dedup.sources["extra/source"] = VerifiedFuzzyDupsPerSource(
            attr_dir=str(tmp_path / "extra"),
            source_tag="source_001",
        )

    with pytest.raises(ValueError, match=rf"{label} source set must equal tokenize source keys"):
        build_clustered_store(
            tokenize=tokenize,
            decontam=decontam,
            cluster_assign=cluster_assign,
            quality=quality,
            exact_dedup=exact_dedup,
            dedup=dedup,
            output_path=str(tmp_path / "store"),
            cluster_view=CLUSTER_VIEW,
            split=SPLIT,
        )


def test_store_rejects_reordered_tokenize_documents(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tokenize, decontam, cluster_assign, quality, exact_dedup, dedup = _build_inputs(tmp_path)
    tokenized_path = tmp_path / "tokenize" / SPLIT / "part-00000-of-00002.parquet"
    table = pq.read_table(tokenized_path)
    ids = table.column("id").to_pylist()
    ids[0] = "wrong-id"
    _write_parquet(str(tokenized_path), table.set_column(0, "id", pa.array(ids)))

    with pytest.raises(RuntimeError, match="tokenize/decon id mismatch"):
        build_clustered_store(
            tokenize=tokenize,
            decontam=decontam,
            cluster_assign=cluster_assign,
            quality=quality,
            exact_dedup=exact_dedup,
            dedup=dedup,
            output_path=str(tmp_path / "store"),
            cluster_view=CLUSTER_VIEW,
            split=SPLIT,
        )


def test_tokenized_documents_regroup_chunks_and_reject_an_orphan(tmp_path):
    """Chunked rows rejoin into one document, and a chunk without its chunk 0 is an error.

    Two adjacent documents that share an id stay two documents: the boundary comes
    from ``chunk_index == 0``, not from a change of id.
    """
    path = str(tmp_path / "tokenize.parquet")
    _write_parquet(
        path,
        pa.table(
            {
                "id": ["a", "a", "b", "b"],
                "chunk_index": pa.array([0, 1, 0, 0], type=pa.int32()),
                "input_ids": pa.array([[1, 2], [3], [4], [5, 6]]),
            }
        ),
    )

    documents = [(doc_id, list(tokens)) for doc_id, tokens in _iter_tokenized_documents(path)]
    assert documents == [("a", [1, 2, 3]), ("b", [4]), ("b", [5, 6])]

    orphan_path = str(tmp_path / "orphan.parquet")
    _write_parquet(
        orphan_path,
        pa.table(
            {
                "id": ["a", "b"],
                "chunk_index": pa.array([0, 1], type=pa.int32()),
                "input_ids": pa.array([[1], [2]]),
            }
        ),
    )

    with pytest.raises(RuntimeError, match="chunk 1 of b, but chunk 1 of a must come next"):
        list(_iter_tokenized_documents(orphan_path))

    # Out of order within one id: concatenating in file order would silently
    # reverse a document's tokens.
    shuffled_path = str(tmp_path / "shuffled.parquet")
    _write_parquet(
        shuffled_path,
        pa.table(
            {
                "id": ["a", "a", "a"],
                "chunk_index": pa.array([0, 2, 1], type=pa.int32()),
                "input_ids": pa.array([[1], [3], [2]]),
            }
        ),
    )

    with pytest.raises(RuntimeError, match="chunk 2 of a, but chunk 1 of a must come next"):
        list(_iter_tokenized_documents(shuffled_path))


def test_tokenized_shard_without_chunk_index_is_rejected(tmp_path):
    """A shard written before the column existed fails legibly, not with a KeyError."""
    path = str(tmp_path / "old.parquet")
    _write_parquet(path, pa.table({"id": ["a"], "input_ids": pa.array([[1, 2]])}))

    with pytest.raises(RuntimeError, match="no chunk_index column"):
        list(_iter_tokenized_documents(path))
