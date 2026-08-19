# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shard join, the pairing guards, and the checks that gate a write.

Each source has one complete harrier leaf co-partitioned with its token side. What
still has to be pinned is that the match is on the *key* rather than on stored
order -- the row order inside a shard is not guaranteed, and on at least one source
it is not sorted -- and that every embedded document is scored, since a join that
quietly matched a subset looks exactly like a smaller corpus. The same theme covers
the two identity checks: which shards pair with which, and which model's bytes are
allowed to write under a path that names a scorer.
"""

import shutil
from dataclasses import replace

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit import hero_data
from experiments.datakit.cluster.quality.fast_transformer import score_corpus
from experiments.datakit.cluster.quality.fast_transformer.score_corpus import (
    EMBED_DIM,
    HOLD_SOURCES,
    Block,
    ShardTask,
    join_shard,
    model_dir_sha256,
    pair_shards,
    read_embed_side,
    require_containment,
    require_paired_shards,
    require_pinned_model,
    with_retry,
)

VOCAB = 4096
MAX_TOKENS = 8


@pytest.fixture
def fs():
    return fsspec.filesystem("file")


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # The retry path is under test, not the wall clock it would otherwise burn.
    # Small but positive: the backoff schedule rejects a non-positive initial.
    monkeypatch.setattr(score_corpus, "RETRY_BASE_DELAY", 0.001)
    monkeypatch.setattr(score_corpus, "RETRY_MAX_DELAY", 0.002)


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


def join(task, fs, block_docs=1024):
    blocks, stats = [], None
    for item in join_shard(task, MAX_TOKENS, VOCAB, block_docs, fs):
        if isinstance(item, Block):
            blocks.append(item)
        else:
            stats = item
    return blocks, stats


def test_every_held_source_is_registered():
    # A hold naming a source that no longer exists is a hold that silently stopped
    # holding anything. Empty is fine; stale is not.
    assert not set(HOLD_SOURCES) - set(hero_data.source_names())


def test_read_embed_side_reads_a_shard(tmp_path, fs):
    path = tmp_path / "e.parquet"
    write_embed(path, ["id2", "id0", "id1"], [10, 20, 30])

    ids, embeddings = read_embed_side(str(path), fs)

    assert list(ids) == ["id2", "id0", "id1"]
    assert [int(row.argmax()) for row in embeddings] == [10, 20, 30]


def test_with_retry_recovers_from_a_transient_fault():
    """CoreWeave severs connections mid-body; every observed one recovered at once.

    Unretried across tens of thousands of shards this surfaces as a task failure
    indistinguishable from a real data fault, which is the actual cost.
    """
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("Response payload is not completed")
        return "ok"

    assert with_retry(flaky, "flaky read") == "ok"
    assert len(calls) == 3


def test_with_retry_gives_up_and_reraises():
    # A genuine fault must still fail the task rather than spin forever.
    def broken():
        raise OSError("nope")

    with pytest.raises(OSError, match="nope"):
        with_retry(broken, "broken read", attempts=2)


def test_join_matches_on_key_not_stored_order(tmp_path, fs):
    """The regression this file exists for.

    Shard writers do not guarantee ordering and `common-crawl-focus-2026-22` did
    not deliver it. A match that read the stored order as sorted degenerated
    silently there, emitting 3 rows for a 6,048-document shard rather than failing,
    and lost 36.3M documents across the leaf. The embed ids here are stored in an
    order that is not sorted, and the token side is in a third order again.
    """
    embed = tmp_path / "e.parquet"
    write_embed(embed, ["d_doc", "a_doc", "c_doc", "b_doc"], [11, 22, 33, 44])
    tokens = tmp_path / "tok.parquet"
    write_tokens(
        tokens,
        [
            ("c_doc", 0),
            ("a_doc", 0),
            ("c_doc", 1),  # a later chunk of an already-matched document
            ("d_doc", 0),
            ("z_absent", 0),  # a token-side document with no embedding
            ("b_doc", 0),
        ],
    )
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    blocks, stats = join(task, fs)

    assert stats.embed_rows == 4
    assert stats.matched == 4, "every embedded document must be scored"
    assert stats.unmatched_embed == 0
    assert sorted(np.concatenate([b.doc_ids for b in blocks])) == ["a_doc", "b_doc", "c_doc", "d_doc"]


def test_join_pairs_each_document_with_its_own_embedding(tmp_path, fs):
    """Matching on the key must carry the *row* through, not the rank.

    The sort order is not the storage order, so the gather back into the embedding
    array runs through an index map. Getting that wrong scores documents against
    other documents' embeddings with no visible error.
    """
    embed = tmp_path / "e.parquet"
    write_embed(embed, ["d_doc", "b_doc", "a_doc", "c_doc"], [11, 22, 33, 44])
    tokens = tmp_path / "tok.parquet"
    write_tokens(tokens, [("a_doc", 0), ("b_doc", 0), ("c_doc", 0), ("d_doc", 0)])
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    blocks, _ = join(task, fs)
    doc_ids = np.concatenate([b.doc_ids for b in blocks])
    embeddings = np.concatenate([b.embedding for b in blocks])
    # Each row points along one axis, and normalization scales but does not rotate,
    # so the surviving axis names the embedding the document was paired with.
    got = dict(zip(doc_ids, embeddings.argmax(axis=1), strict=True))

    assert got == {"d_doc": 11, "b_doc": 22, "a_doc": 33, "c_doc": 44}


def test_join_emits_one_row_per_embed_row_when_ids_repeat_on_the_embed_side(tmp_path, fs):
    """A duplicate id earns its own score row; it is byte-identical text.

    Driven from the token side, a single token row would match one embed row and
    the other would be silently dropped.
    """
    embed = tmp_path / "e.parquet"
    write_embed(embed, ["dup", "solo", "dup"], [11, 22, 33])
    tokens = tmp_path / "tok.parquet"
    write_tokens(tokens, [("dup", 0), ("solo", 0)])
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    blocks, stats = join(task, fs)
    doc_ids = list(np.concatenate([b.doc_ids for b in blocks]))
    embeddings = np.concatenate([b.embedding for b in blocks])

    assert stats.embed_rows == 3
    assert stats.matched == 3, "both copies of the duplicate id must be scored"
    assert stats.unmatched_embed == 0
    assert stats.duplicate_embed_ids == 1
    assert sorted(doc_ids) == ["dup", "dup", "solo"]
    # Each output row carries its own embed row, not the same one twice.
    assert sorted(embeddings.argmax(axis=1)) == [11, 22, 33]


def test_join_does_not_cross_product_when_ids_repeat_on_both_sides(tmp_path, fs):
    """The regression the cardinality assert exists for.

    With an id twice on each side, an id-keyed match takes their cross product:
    2 embed rows become 4 outputs. A probe hit exactly this on the focus crawl,
    turning 6,095 embed rows into 6,331.
    """
    embed = tmp_path / "e.parquet"
    write_embed(embed, ["dup", "dup", "solo"], [11, 22, 33])
    tokens = tmp_path / "tok.parquet"
    # The same id on two chunk-0 token rows, as a duplicated document produces.
    write_tokens(tokens, [("dup", 0), ("solo", 0), ("dup", 0)])
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    blocks, stats = join(task, fs)
    doc_ids = list(np.concatenate([b.doc_ids for b in blocks]))

    assert stats.matched == 3, f"expected one row per embed row, got {stats.matched}"
    assert stats.embed_rows == 3
    assert sorted(doc_ids) == ["dup", "dup", "solo"]


def test_join_holds_cardinality_across_batch_boundaries(tmp_path, fs):
    """An id split across token batches must still claim its embed rows once.

    The claim is tracked per embed row rather than per batch, so a duplicate id
    landing in a later batch cannot re-claim a run an earlier batch already took.
    """
    embed = tmp_path / "e.parquet"
    ids = ["dup"] * 3 + [f"id{i:04d}" for i in range(200)]
    write_embed(embed, ids, list(range(len(ids))))
    tokens = tmp_path / "tok.parquet"
    # `dup` appears at both ends, so the two occurrences fall in different batches.
    records = [("dup", 0)] + [(f"id{i:04d}", 0) for i in range(200)] + [("dup", 0)]
    write_tokens(tokens, records)
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    # A block size below the row count forces several batches through the join.
    blocks, stats = join(task, fs, block_docs=16)
    doc_ids = list(np.concatenate([b.doc_ids for b in blocks]))

    assert stats.matched == len(ids) == 203
    assert stats.unmatched_embed == 0
    assert doc_ids.count("dup") == 3
    assert stats.duplicate_embed_ids == 2


def test_a_shard_the_token_side_does_not_contain_fails_instead_of_publishing(tmp_path, fs):
    """`unmatched_embed` is the containment check, and it has to be fatal.

    Every embedded document should have a chunk-0 token row, so a nonzero value
    means co-partitioning broke. Writing the matched subset anyway published a
    short shard from a worker that exited zero, and the shortfall showed up only
    if someone later ran `verify` on that source.
    """
    embed = tmp_path / "e.parquet"
    write_embed(embed, ["a", "b"], [1, 2])
    tokens = tmp_path / "tok.parquet"
    write_tokens(tokens, [("a", 0)])
    task = ShardTask(
        source="s", shard_index=0, tokens_path=str(tokens), embed_path=str(embed), output_path="", total_bytes=0
    )

    _, stats = join(task, fs)

    assert stats.embed_rows == 2
    assert stats.matched == 1
    assert stats.unmatched_embed == 1
    with pytest.raises(ValueError, match="absent from the token side"):
        require_containment(task, stats)


# ---------------------------------------------------------------------------
# pairing the two leaves
# ---------------------------------------------------------------------------


def listing(directory, names):
    """An `fs.ls(detail=True)` listing, as `manifest` reads one."""
    return [{"name": f"bucket/{directory}/{name}", "size": 1024} for name in names]


def paired(token_names, embed_names, source="src"):
    return pair_shards(source, "s3://bucket/out", listing("tok", token_names), listing("emb", embed_names))


def test_matching_basename_sets_pair_and_build():
    names = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]

    leaf = paired(names, names)

    assert [row["embed_leaves"] for row in leaf.rows] == [1, 1]
    assert [row["output_path"] for row in leaf.rows] == [f"s3://bucket/out/{name}" for name in names]
    assert leaf.embed_only == []
    require_paired_shards([leaf])


def test_an_embed_shard_the_token_side_does_not_name_refuses_the_manifest():
    """The regression: a *partial* basename overlap used to build a manifest.

    Rows are enumerated from the token side, so an embed-only shard never becomes
    one. Its documents miss the manifest, the footer totals, scoring and
    verification together, and every count that remains is self-consistent -- so
    only the complete basename sets show it. The old guard fired on a wholly
    disjoint pair alone, and this pair overlaps in half its shards.
    """
    leaf = paired(
        ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"],
        ["part-00000-of-00002.parquet", "part-00002-of-00003.parquet"],
    )

    assert leaf.embed_only == ["part-00002-of-00003.parquet"]
    assert [row["embed_leaves"] for row in leaf.rows] == [1, 0], "the old guard counted a match and passed"
    with pytest.raises(ValueError, match="token side does not have"):
        require_paired_shards([leaf])


def test_two_sides_sharing_no_basename_refuse_the_manifest():
    # A normalize rerun on one side: every document would resolve to no embedding.
    leaf = paired(["part-00000-of-00001.parquet"], ["part-00007-of-00009.parquet"])

    with pytest.raises(ValueError, match="no shard basename in common"):
        require_paired_shards([leaf])


# ---------------------------------------------------------------------------
# binding the loaded bytes to the output path
# ---------------------------------------------------------------------------


def write_model_dir(root, contents):
    for relative, body in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return root


def test_the_model_digest_covers_contents_layout_and_nothing_else(tmp_path):
    """What the digest has to distinguish, since the output path is built from it.

    Different weights under an unchanged filename, and an added artifact, are both
    different scorers. Where the directory happens to sit is not, or a checkpoint
    could not be staged anywhere but the path it was digested at.
    """
    root = write_model_dir(tmp_path / "model", {"a.eqx": b"weights", "nested/meta.json": b'{"k": 1}'})
    base = model_dir_sha256(str(root))

    (root / "nested" / "meta.json").write_bytes(b'{"k": 2}')
    assert model_dir_sha256(str(root)) != base, "changed bytes are a different model"

    (root / "nested" / "meta.json").write_bytes(b'{"k": 1}')
    (root / "extra.json").write_bytes(b"")
    assert model_dir_sha256(str(root)) != base, "an added artifact is a different model"

    (root / "extra.json").unlink()
    shutil.copytree(root, tmp_path / "elsewhere" / "model")
    assert model_dir_sha256(str(tmp_path / "elsewhere" / "model")) == base


def test_scoring_refuses_a_checkpoint_that_is_not_the_pinned_one(tmp_path):
    """The write side of the collision the output path exists to close.

    `hero_data.quality` folds `model_sha256` into the path, so scores written
    there claim to be that model's -- but `--model-dir` is a free argument and
    nothing checked the directory the worker actually loaded.
    """
    root = write_model_dir(tmp_path / "model", {"impostor.eqx": b"some other checkpoint"})
    its_own_pin = replace(hero_data.NEMOTRON_88K, model_sha256=model_dir_sha256(str(root)))

    assert require_pinned_model(its_own_pin, str(root)) == its_own_pin.model_sha256
    with pytest.raises(ValueError, match=hero_data.NEMOTRON_88K.model_sha256):
        require_pinned_model(hero_data.NEMOTRON_88K, str(root))


def test_an_absent_model_dir_does_not_digest_to_the_empty_hash(tmp_path):
    # A mistyped path must name itself rather than resolve to a stable digest over
    # no files, which reads downstream as a plain pin mismatch.
    with pytest.raises(ValueError, match="cannot digest to nothing"):
        model_dir_sha256(str(tmp_path / "absent"))
