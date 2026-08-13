# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shard join, over an embed side that spans two leaves.

The corpus embeddings live in two stage roots -- one run embedded what the global
fuzzy dedup kept, a second embedded what it dropped -- so a shard's embed side is
the union of two files. These tests pin the property that makes that safe: every
embedded document is scored regardless of which leaf carries it, and regardless of
the order the union happens to land in. A join that quietly matched only the first
leaf would look exactly like a smaller corpus.
"""

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    EMBED_DIM,
    ShardTask,
    join_shard,
    read_embed_side,
)

VOCAB = 4096
MAX_TOKENS = 8


@pytest.fixture
def fs():
    return fsspec.filesystem("file")


def write_embed(path, ids, markers):
    """An embed shard whose row `i` points along axis `markers[i]`.

    A distinct *direction* per row, not a distinct magnitude: the join L2-normalizes
    the embeddings, so anything encoded in the length is gone by the time a test can
    read it, and constant vectors all normalize to the same point.
    """
    rows = np.zeros((len(ids), EMBED_DIM), dtype=np.int8)
    rows[np.arange(len(ids)), np.asarray(markers)] = 100
    table = pa.table(
        {"id": pa.array(ids, pa.string()), "embedding": pa.FixedSizeListArray.from_arrays(rows.reshape(-1), EMBED_DIM)}
    )
    pq.write_table(table, path)


def write_tokens(path, records):
    """A token shard: `records` is a list of (id, chunk_index)."""
    table = pa.table(
        {
            "id": pa.array([r[0] for r in records], pa.string()),
            "chunk_index": pa.array([r[1] for r in records], pa.int32()),
            "input_ids": pa.array([[1, 2, 3] for _ in records], pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, path)


def test_read_embed_side_concatenates_leaves(tmp_path, fs):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_embed(a, ["id2", "id0"], [10, 20])
    write_embed(b, ["id1"], [30])

    ids, embeddings = read_embed_side((str(a), str(b)), fs)

    assert list(ids) == ["id2", "id0", "id1"]
    assert [int(row.argmax()) for row in embeddings] == [10, 20, 30]


def test_read_embed_side_skips_a_leaf_that_does_not_exist(tmp_path, fs):
    # The second run is per-source: a source can have one leaf while another has
    # two, and that means "no rows here", not a corrupt shard.
    a = tmp_path / "a.parquet"
    write_embed(a, ["id0"], [10])

    ids, embeddings = read_embed_side((str(a), str(tmp_path / "absent.parquet")), fs)

    assert list(ids) == ["id0"]
    assert embeddings.shape == (1, EMBED_DIM)


def test_read_embed_side_with_no_surviving_leaf_is_empty(tmp_path, fs):
    ids, embeddings = read_embed_side((str(tmp_path / "absent.parquet"),), fs)
    assert len(ids) == 0
    assert embeddings.shape == (0, EMBED_DIM)


def join(task, fs, block_docs=1024):
    blocks, stats = [], None
    for item in join_shard(task, MAX_TOKENS, VOCAB, block_docs, fs):
        if hasattr(item, "doc_ids"):
            blocks.append(item)
        else:
            stats = item
    return blocks, stats


def test_join_scores_documents_from_both_leaves(tmp_path, fs):
    """The regression this file exists for.

    The dedup-surviving and fuzzy-duplicate leaves are disjoint, so the union is
    unordered even when each leaf is sorted. Ids here are chosen so the second
    leaf's ids sort *before* the first's: a join that trusted the concatenated
    order would drop them, which is the shape of the 36.3M-row loss an earlier
    sorted-merge join produced.
    """
    kept, dropped = tmp_path / "kept.parquet", tmp_path / "dropped.parquet"
    write_embed(kept, ["d_kept", "e_kept"], [1, 2])
    write_embed(dropped, ["a_drop", "b_drop"], [3, 4])
    tokens = tmp_path / "tok.parquet"
    write_tokens(
        tokens,
        [
            ("e_kept", 0),
            ("a_drop", 0),
            ("e_kept", 1),  # a later chunk of an already-matched document
            ("b_drop", 0),
            ("z_absent", 0),  # a token-side document with no embedding
            ("d_kept", 0),
        ],
    )
    task = ShardTask(
        source="s",
        shard_index=0,
        tokens_path=str(tokens),
        embed_paths=(str(kept), str(dropped)),
        output_path="",
        total_bytes=0,
    )

    blocks, stats = join(task, fs)

    assert stats.embed_rows == 4
    assert stats.matched == 4, "every embedded document must be scored, from either leaf"
    assert stats.unmatched_embed == 0
    assert sorted(np.concatenate([b.doc_ids for b in blocks])) == ["a_drop", "b_drop", "d_kept", "e_kept"]


def test_join_pairs_each_document_with_its_own_embedding(tmp_path, fs):
    """Matching on the key must carry the *row* through, not the rank.

    The union's sort order is not its storage order, so the gather back into the
    embedding array runs through an index map. Getting that wrong scores documents
    against other documents' embeddings with no visible error.
    """
    kept, dropped = tmp_path / "kept.parquet", tmp_path / "dropped.parquet"
    write_embed(kept, ["d_kept", "b_kept"], [11, 22])
    write_embed(dropped, ["a_drop", "c_drop"], [33, 44])
    tokens = tmp_path / "tok.parquet"
    write_tokens(tokens, [("a_drop", 0), ("b_kept", 0), ("c_drop", 0), ("d_kept", 0)])
    task = ShardTask(
        source="s",
        shard_index=0,
        tokens_path=str(tokens),
        embed_paths=(str(kept), str(dropped)),
        output_path="",
        total_bytes=0,
    )

    blocks, _ = join(task, fs)
    doc_ids = np.concatenate([b.doc_ids for b in blocks])
    embeddings = np.concatenate([b.embedding for b in blocks])
    # Each row points along one axis, and normalization scales but does not rotate,
    # so the surviving axis names the embedding the document was paired with.
    got = dict(zip(doc_ids, embeddings.argmax(axis=1), strict=True))

    assert got == {"d_kept": 11, "b_kept": 22, "a_drop": 33, "c_drop": 44}


def test_join_reports_embed_rows_the_token_side_did_not_carry(tmp_path, fs):
    # `unmatched_embed` is the containment check: every embedded document should
    # have a chunk-0 token row, so a nonzero value means co-partitioning broke.
    kept = tmp_path / "kept.parquet"
    write_embed(kept, ["a", "b"], [1, 2])
    tokens = tmp_path / "tok.parquet"
    write_tokens(tokens, [("a", 0)])
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_paths=(str(kept),), output_path="", total_bytes=0
    )

    _, stats = join(task, fs)

    assert stats.embed_rows == 2
    assert stats.matched == 1
    assert stats.unmatched_embed == 1
