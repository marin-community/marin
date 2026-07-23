# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral smoke for the shuffle-based datakit clustered store build.

Builds tiny synthetic co-partitioned inputs (tokenize / decon / cluster /
quality / dedup) on the local filesystem, runs ``build_clustered_store_shuffle``
through a local Zephyr context, and checks the externally observable contract:
the right docs survive filtering, land in the right ``(cluster, quality)``
bucket, round-trip out of the materialized caches, and hot-bucket subsharding
splits a bucket without losing or duplicating data.
"""

import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.store.cache import TreeCache
from marin.datakit.decon import DeconAttributes
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams
from marin.processing.tokenize.attributes import TokenizedAttrData
from zephyr.shard_keys import deterministic_hash

from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.store.datakit_store_shuffle import build_clustered_store_shuffle

CLUSTER_VIEW = 2  # cluster column "cluster_2", clusters {0, 1}
SPLIT = "train"
EXEMPLAR = {"input_ids": np.array([0], dtype=np.int32)}

# One doc = (id, cluster, quality_bucket, contaminated, dedup-canonical, token value, length).
# ``quality_bucket`` is the precomputed calibrated bucket the scorer writes as a
# column (consumed as-is by the store -- no score->bucket mapping).
# ``canonical``: None -> singleton (absent from dedup, kept); True -> kept; False -> dropped.
# Each surviving doc's tokens are a run of its unique ``token`` value so we can
# recover identity from the cache and confirm dropped docs never appear.
_Doc = tuple[str, int, int, bool, bool | None, int, int]

SHARD0: list[_Doc] = [
    ("d0", 0, 0, False, None, 100, 3),  # -> (0, 0)
    ("d1", 0, 4, False, None, 101, 5),  # -> (0, 4)
    ("d2", 1, 2, True, None, 902, 9),  # contaminated -> dropped
    ("d3", 1, 2, False, True, 103, 2),  # canonical -> (1, 2)
]
SHARD1: list[_Doc] = [
    ("d4", 0, 0, False, None, 104, 4),  # -> (0, 0)
    ("d5", 1, 2, False, False, 905, 8),  # non-canonical -> dropped
    ("d6", 1, 4, False, None, 106, 6),  # -> (1, 4)
    ("d7", 0, 4, False, None, 107, 7),  # -> (0, 4)
]

# Expected surviving buckets: {(cluster, quality): {token_value: length}}.
EXPECTED: dict[tuple[int, int], dict[int, int]] = {
    (0, 0): {100: 3, 104: 4},
    (0, 4): {101: 5, 107: 7},
    (1, 2): {103: 2},
    (1, 4): {106: 6},
}
DROPPED_TOKEN_VALUES = {902, 905}


def _write_parquet(path: str, table: pa.Table) -> None:
    pq.write_table(table, path)


def _write_shard(dirs: dict[str, str], basename: str, docs: list[_Doc]) -> None:
    """Write one shard's five co-partitioned parquets from a doc list (same row order)."""
    ids = [d[0] for d in docs]
    _write_parquet(
        f"{dirs['tokenize']}/{basename}",
        pa.table({"id": ids, "input_ids": pa.array([[d[5]] * d[6] for d in docs])}),
    )
    _write_parquet(
        f"{dirs['decon']}/{basename}",
        pa.table({"id": ids, "attributes": pa.array([{"contaminated": d[3]} for d in docs])}),
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
    # Dedup is sparse: only non-singleton docs (canonical True/False) appear.
    marked = [(d[0], d[4]) for d in docs if d[4] is not None]
    if marked:
        _write_parquet(
            f"{dirs['dedup']}/{basename}",
            pa.table(
                {
                    "id": [m[0] for m in marked],
                    "attributes": pa.array([{"is_cluster_canonical": m[1]} for m in marked]),
                }
            ),
        )


def _build_inputs(tmp_path):
    """Materialize the synthetic inputs and return the five typed artifacts."""
    main_dir = str(tmp_path / "src")
    dirs = {k: str(tmp_path / k) for k in ("tokenize", "decon", "cluster", "quality", "dedup")}
    tok_train = f"{dirs['tokenize']}/{SPLIT}"
    for d in (*dirs.values(), tok_train):
        os.makedirs(d, exist_ok=True)

    shard_dirs = {**dirs, "tokenize": tok_train}
    _write_shard(shard_dirs, "part-00000-of-00002.parquet", SHARD0)
    _write_shard(shard_dirs, "part-00001-of-00002.parquet", SHARD1)

    tokenize = {
        "src": TokenizedAttrData(
            output_dirs={SPLIT: tok_train},
            source_main_dirs={SPLIT: main_dir},
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
            source_main_dir=main_dir,
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
    dedup = FuzzyDupsAttrData(
        params=MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0),
        sources={main_dir: FuzzyDupsPerSource(attr_dir=dirs["dedup"])},
        counters={},
    )
    return tokenize, decontam, cluster_assign, quality, dedup


def _read_bucket_tokens(bucket_root: str) -> list[np.ndarray]:
    """Read every doc's input_ids across all of a bucket's ``sub=*`` materialized caches."""
    docs: list[np.ndarray] = []
    for sub_dir in sorted(glob.glob(f"{bucket_root}/sub=*")):
        cache = TreeCache.load(sub_dir, EXEMPLAR)
        docs.extend(np.asarray(doc["input_ids"]) for doc in cache)
    return docs


def test_shuffle_store_filters_routes_and_roundtrips(tmp_path):
    tokenize, decontam, cluster_assign, quality, dedup = _build_inputs(tmp_path)
    output_path = str(tmp_path / "store")

    artifact = build_clustered_store_shuffle(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        dedup=dedup,
        output_path=output_path,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        reduce_shards=4,
    )

    assert isinstance(artifact, ClusteredStoreData)
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

    assert DROPPED_TOKEN_VALUES.isdisjoint(all_tokens), "contaminated + non-canonical docs must be absent"


def test_shuffle_store_subshards_hot_bucket_without_data_loss(tmp_path):
    tokenize, decontam, cluster_assign, quality, dedup = _build_inputs(tmp_path)
    output_path = str(tmp_path / "store_split")

    # Force bucket (0, 4) to split 4 ways; every other bucket stays at 1 sub.
    split_key = (0, 4)
    subshards = 4
    hint = {split_key: subshards * 1_000_000}
    expected_subs = {deterministic_hash(doc_id) % subshards for doc_id in ("d1", "d7")}
    assert len(expected_subs) >= 2, "pick doc ids that hash to different subshards so the split is exercised"

    artifact = build_clustered_store_shuffle(
        tokenize=tokenize,
        decontam=decontam,
        cluster_assign=cluster_assign,
        quality=quality,
        dedup=dedup,
        output_path=output_path,
        cluster_view=CLUSTER_VIEW,
        split=SPLIT,
        reduce_shards=4,
        bucket_token_hint=hint,
        target_tokens_per_subshard=1_000_000,
        max_subshards=subshards,
    )

    by_key = {(b.cluster_id, b.quality_bucket): b for b in artifact.buckets}
    split_bucket = by_key[split_key]
    # The bucket fans out to one cache per distinct hashed subshard...
    assert split_bucket.n_shards == len(expected_subs)
    # ...while still carrying exactly its docs (no loss / no duplication across subs).
    assert split_bucket.total_elements == len(EXPECTED[split_key])
    assert split_bucket.total_tokens == sum(EXPECTED[split_key].values())
    recovered = {int(arr[0]): len(arr) for arr in _read_bucket_tokens(split_bucket.path)}
    assert recovered == EXPECTED[split_key]

    # A non-split bucket is unaffected.
    assert by_key[(0, 0)].n_shards == 1
