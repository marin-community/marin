# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fusion scoring step: its pure pieces (shard pairing, aligned reads, id packing,
the pin checks) and the step end to end on a local pool over hand-written tokenize
and embedding leaves, with a tiny fusion model."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import equinox as eqx
import jax.random as jr
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.tokenize.attributes import TokenizedAttrData

from experiments.datakit.cluster.quality.fast_transformer.artifact import MODEL_CALIB
from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID, remap_table
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    QualityPin,
    calibration_sha256,
    model_sha256,
    require_pinned_calibration,
    require_pinned_model,
)
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import (
    BATCH_DOCS,
    AlignedColumn,
    AlignedDocuments,
    normalize_embeddings,
    pad_ids,
    paired_basenames,
    rebatch,
    score_fusion,
    validated_tokenize_dir,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer

VOCAB = 10
TABLE = remap_table({t: t + 2 for t in range(VOCAB)})


def test_pad_ids_truncates_remaps_and_marks_unknown_ids():
    rows = [np.array(r, dtype=np.int32) for r in ([0, 1, 2, 3], [7], [], [12, 0])]
    ids = pad_ids(rows, max_tokens=3, remap_table=TABLE)

    assert ids.dtype == np.int32
    # Raw ids shift past the two reserved slots; a raw id the table does not know is UNK.
    assert ids.tolist() == [[2, 3, 4], [9, PAD_ID, PAD_ID], [PAD_ID] * 3, [UNK_ID, 2, PAD_ID]]


def test_rebatch_yields_full_batches_and_one_tail():
    batches = [
        pa.RecordBatch.from_pydict({"id": [str(i) for i in range(start, start + n)]})
        for start, n in ((0, 3), (3, 5), (8, 1))
    ]

    out = list(rebatch(iter(batches), 4))

    assert [b.num_rows for b in out] == [4, 4, 1]
    assert [i for b in out for i in b.column("id").to_pylist()] == [str(i) for i in range(9)]


def aligned(ids: list[str]) -> AlignedColumn:
    return AlignedColumn(np.array(ids, dtype=object), np.arange(len(ids), dtype=np.float32), "shard")


def test_aligned_column_takes_rows_in_order_across_batches():
    # A duplicate id is two documents at two positions, not one row to share.
    side = aligned(["b", "a", "dup", "dup"])

    first = side.take(np.array(["b", "a"], dtype=object))
    second = side.take(np.array(["dup", "dup"], dtype=object))

    assert first.tolist() == [0.0, 1.0] and second.tolist() == [2.0, 3.0]
    side.require_consumed()


def test_aligned_column_refuses_ids_out_of_step():
    side = aligned(["a", "b"])
    with pytest.raises(ValueError, match=r"rows 0\.\.2 do not carry"):
        side.take(np.array(["b", "a"], dtype=object))


def test_aligned_column_refuses_more_documents_than_rows():
    side = aligned(["a"])
    with pytest.raises(ValueError, match="do not carry"):
        side.take(np.array(["a", "b"], dtype=object))


def test_aligned_column_refuses_rows_left_over():
    side = aligned(["a", "b"])
    side.take(np.array(["a"], dtype=object))
    with pytest.raises(ValueError, match="2 rows against 1 documents"):
        side.require_consumed()


def documents(*docs: tuple[str, list[int]], max_tokens: int = 3) -> AlignedDocuments:
    return AlignedDocuments(iter((i, np.array(t, dtype=np.int32)) for i, t in docs), max_tokens, "shard")


def test_aligned_documents_take_the_first_max_tokens_in_order_across_batches():
    side = documents(("b", [1, 2, 3, 4]), ("a", [5]), ("dup", [6]), ("dup", [7]))

    first = side.take(np.array(["b", "a"], dtype=object))
    second = side.take(np.array(["dup", "dup"], dtype=object))

    assert [r.tolist() for r in first] == [[1, 2, 3], [5]]
    assert [r.tolist() for r in second] == [[6], [7]]
    side.require_consumed()


def test_aligned_documents_refuse_a_document_out_of_step():
    side = documents(("a", [1]), ("b", [2]))
    with pytest.raises(ValueError, match="row 0: normalized id 'b' but tokenize document 'a'"):
        side.take(np.array(["b", "a"], dtype=object))


def test_aligned_documents_refuse_a_missing_document():
    # The tokenize stage drops zero-token documents; such a row must not shift the rest.
    side = documents(("a", [1]))
    with pytest.raises(ValueError, match="row 1: normalized id 'b' but tokenize document None"):
        side.take(np.array(["a", "b"], dtype=object))


def test_aligned_documents_refuse_documents_left_over():
    side = documents(("a", [1]), ("b", [2]))
    side.take(np.array(["a"], dtype=object))
    with pytest.raises(ValueError, match="tokenize document 'b' left over after 1 normalized rows"):
        side.require_consumed()


def write_parquet(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        pq.write_table(pa.table({"id": pa.array([], pa.string())}), directory / name)


def test_paired_basenames_refuses_an_asymmetric_leaf(tmp_path):
    """A basename one side lacks is a document set that would leave no trace in the output."""
    names = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
    write_parquet(tmp_path / "text", names)
    write_parquet(tmp_path / "embed", names)
    write_parquet(tmp_path / "other", [names[0], "part-00002-of-00003.parquet"])

    assert paired_basenames(str(tmp_path / "text"), str(tmp_path / "embed")) == names
    with pytest.raises(ValueError, match="not co-partitioned"):
        paired_basenames(str(tmp_path / "text"), str(tmp_path / "other"))
    with pytest.raises(FileNotFoundError):
        paired_basenames(str(tmp_path / "empty"), str(tmp_path / "embed"))


def write_model_dir(root, calibration=b"knots"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "m.eqx").write_bytes(b"weights")
    (root / "m_remap.json").write_bytes(b"{}")
    (root / "m_meta.json").write_bytes(b'{"k": 1}')
    (root / MODEL_CALIB).write_bytes(calibration)
    return root


def test_model_digest_covers_the_scorer_artifacts_and_not_the_calibration(tmp_path):
    """The two digests move independently: a refit calibration must not rescore the corpus."""
    root = write_model_dir(tmp_path / "model")
    model = model_sha256(str(root))
    calibration = calibration_sha256(str(root))

    (root / MODEL_CALIB).write_bytes(b"other knots")
    assert model_sha256(str(root)) == model
    assert calibration_sha256(str(root)) != calibration

    (root / "m_meta.json").write_bytes(b'{"k": 2}')
    assert model_sha256(str(root)) != model, "changed bytes are a different model"

    assert calibration == hashlib.sha256(MODEL_CALIB.encode() + b"\0" + hashlib.sha256(b"knots").digest()).hexdigest()


def test_digests_survive_copying_the_directory(tmp_path):
    root = write_model_dir(tmp_path / "model")
    copy = write_model_dir(tmp_path / "elsewhere" / "model")
    assert model_sha256(str(root)) == model_sha256(str(copy))
    assert calibration_sha256(str(root)) == calibration_sha256(str(copy))


def test_scoring_refuses_bytes_that_are_not_the_pinned_ones(tmp_path):
    """The write side of the collision the output path exists to close."""
    root = write_model_dir(tmp_path / "model")
    pin = QualityPin(
        name="pin",
        model_path="model",
        model_sha256=model_sha256(str(root)),
        calibration_sha256=calibration_sha256(str(root)),
        tokenizer="tok@0123abcd",
    )

    assert require_pinned_model(pin, str(root)) == pin.model_sha256
    assert require_pinned_calibration(pin, str(root)) == pin.calibration_sha256
    with pytest.raises(ValueError, match="claims"):
        require_pinned_model(replace(pin, model_sha256="0" * 64), str(root))
    with pytest.raises(ValueError, match="different calibration"):
        require_pinned_calibration(replace(pin, calibration_sha256="0" * 64), str(root))


def test_a_model_dir_without_an_artifact_does_not_digest(tmp_path):
    with pytest.raises(ValueError, match=r"no \.eqx artifact"):
        model_sha256(str(tmp_path))


def test_a_tokenizer_without_a_revision_is_refused():
    # A bare repo name is mutable: the same pin could tokenize to different ids.
    with pytest.raises(ValueError, match="<repo>@<commit>"):
        QualityPin(name="pin", model_path="m", model_sha256="0", calibration_sha256="0", tokenizer="tok")


# ---------- the step end to end on a local pool ----------

SPLIT = "train"
BASENAMES = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
MAX_TOKENS = 8
EMBED_DIM = 4
LOCAL = ResourceConfig(cpu=1, ram="1g")
# Documents per shard, in the normalized shard's (not id-sorted) row order; ``a`` is
# written as two chunk rows, and ``b`` is longer than the scorer's window.
SHARDS = {
    BASENAMES[0]: [("b", list(range(1, 13))), ("a", [3, 4, 5, 6, 7]), ("c", [8, 9, VOCAB + 5])],
    BASENAMES[1]: [("z", [2])],
}


def write_shard(directory: Path, name: str, table: pa.Table) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / name)


def make_source(root: Path, shards: dict[str, list[tuple[str, list[int]]]]) -> NormalizedData:
    main = root / "normalized" / "outputs" / "main"
    for name, docs in shards.items():
        ids = [i for i, _ in docs]
        write_shard(main, name, pa.table({"id": ids, "text": [f"text of {i}" for i in ids]}))
    return NormalizedData(
        main_output_dir=str(main), dup_output_dir=str(root / "normalized" / "outputs" / "dups"), counters={}
    )


def tokenize_rows(docs: list[tuple[str, list[int]]]) -> pa.Table:
    """Chunk rows as the tokenize stage writes them; a document longer than four ids spans two rows."""
    ids, chunk_index, input_ids = [], [], []
    for doc_id, tokens in docs:
        pieces = [tokens[:4], tokens[4:]] if len(tokens) > 4 else [tokens]
        for index, piece in enumerate(pieces):
            ids.append(doc_id)
            chunk_index.append(index)
            input_ids.append(piece)
    return pa.table(
        {
            "id": ids,
            "chunk_index": pa.array(chunk_index, type=pa.int32()),
            "input_ids": pa.array(input_ids, type=pa.list_(pa.int32())),
        }
    )


def make_tokenized(root: Path, normalized: NormalizedData, shards: dict, tokenizer: str = "tok") -> TokenizedAttrData:
    directory = root / "tokenize" / SPLIT
    for name, docs in shards.items():
        write_shard(directory, name, tokenize_rows(docs))
    return TokenizedAttrData(
        output_dirs={SPLIT: str(directory)},
        source_keys={SPLIT: datakit_source_key(normalized.main_output_dir)},
        tokenizer=tokenizer,
        tokenizer_backend="hf",
        counters={},
    )


def make_embeddings(root: Path, shards: dict) -> dict[str, np.ndarray]:
    """One int8[EMBED_DIM] row per document, written as the Harrier leaf is; returned by id."""
    rng = np.random.default_rng(0)
    by_id: dict[str, np.ndarray] = {}
    for name, docs in shards.items():
        rows = rng.integers(-128, 128, size=(len(docs), EMBED_DIM), dtype=np.int8)
        for (doc_id, _), row in zip(docs, rows, strict=True):
            by_id[doc_id] = row
        table = pa.table(
            {
                "id": [i for i, _ in docs],
                "embedding": pa.array(rows.tolist(), type=pa.list_(pa.int8(), EMBED_DIM)),
            }
        )
        write_shard(root / "embed", name, table)
    return by_id


def make_fusion_pin(root: Path) -> QualityPin:
    """A tiny fusion scorer in the layout ``load_pooled_scorer`` reads, pinned by its digests."""
    config = FastTransformerConfig(
        vocab_size=VOCAB + 2,
        max_tokens=MAX_TOKENS,
        pool_window=4,
        embed_dim=4,
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        doc_embed_dim=EMBED_DIM,
        doc_embed_super_token=True,
    )
    model_dir = root / "models" / "fusion"
    model_dir.mkdir(parents=True)
    eqx.tree_serialise_leaves(str(model_dir / "tiny.eqx"), FastTransformer(config, key=jr.PRNGKey(0)))
    (model_dir / "tiny_remap.json").write_text(json.dumps({t: t + 2 for t in range(VOCAB)}))
    (model_dir / "tiny_meta.json").write_text(
        json.dumps({"tokenizer": "tok", "max_tokens": MAX_TOKENS, "config": asdict(config)})
    )
    (model_dir / MODEL_CALIB).write_text(json.dumps({"xk": [0.0, 1.0], "yk": [0.0, 1.0]}))
    return QualityPin(
        name="tiny",
        model_path="models/fusion",
        model_sha256=model_sha256(str(model_dir)),
        calibration_sha256=calibration_sha256(str(model_dir)),
        tokenizer="tok@0123abcd",
    )


def run_score(tmp_path: Path, normalized, tokenized, pin: QualityPin, max_workers: int = 2):
    return score_fusion(
        str(tmp_path / "out"),
        normalized=normalized,
        tokenized=tokenized,
        embedding_dir=str(tmp_path / "embed"),
        quality_model=pin,
        worker_resources=LOCAL,
        task_resources=LOCAL,
        max_workers=max_workers,
    )


def test_score_fusion_writes_the_normalized_order_from_stored_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = make_source(tmp_path, SHARDS)
    tokenized = make_tokenized(tmp_path, normalized, SHARDS)
    embeddings = make_embeddings(tmp_path, SHARDS)
    pin = make_fusion_pin(tmp_path)

    artifact = run_score(tmp_path, normalized, tokenized, pin)

    scorer = load_pooled_scorer(str(tmp_path / "models" / "fusion"))
    for name, docs in SHARDS.items():
        out = pq.read_table(tmp_path / "out" / name).to_pydict()
        assert out["id"] == [i for i, _ in docs]
        # The score of a document is the forward over its first MAX_TOKENS stored ids
        # (chunk rows regrouped, the window truncated) and its normalized embedding.
        tokens = pad_ids([np.array(t, dtype=np.int32) for _, t in docs], MAX_TOKENS, scorer.remap_table)
        embedding = normalize_embeddings(np.stack([embeddings[i] for i, _ in docs]))
        expected = predict(scorer.model, tokens, batch_size=BATCH_DOCS, doc_embed=embedding)
        np.testing.assert_allclose(out["score"], expected, atol=1e-6, rtol=0)

    assert artifact.output_dir == str(tmp_path / "out")
    assert artifact.tokenize_dir == tokenized.output_dirs[SPLIT]
    assert artifact.embedding_dir == str(tmp_path / "embed")
    assert artifact.source_key == datakit_source_key(normalized.main_output_dir)
    assert artifact.tokenizer == pin.tokenizer
    assert artifact.counters["fusion/docs_scored"] == 4
    assert artifact.counters["fusion/shards"] == 2


def test_a_tokenize_shard_missing_a_document_fails_the_source(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = make_source(tmp_path, SHARDS)
    short = {BASENAMES[0]: SHARDS[BASENAMES[0]][:2], BASENAMES[1]: SHARDS[BASENAMES[1]]}
    tokenized = make_tokenized(tmp_path, normalized, short)
    make_embeddings(tmp_path, SHARDS)
    pin = make_fusion_pin(tmp_path)

    with pytest.raises(Exception, match="row 2: normalized id 'c' but tokenize document None"):
        run_score(tmp_path, normalized, tokenized, pin, max_workers=1)


def test_a_tokenize_artifact_of_another_source_or_tokenizer_is_refused_before_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = make_source(tmp_path, SHARDS)
    tokenized = make_tokenized(tmp_path, normalized, SHARDS)
    pin = make_fusion_pin(tmp_path)

    assert validated_tokenize_dir(pin, normalized, tokenized, SPLIT) == tokenized.output_dirs[SPLIT]
    with pytest.raises(ValueError, match="reads 'tok' ids but the tokenize stage wrote 'other'"):
        validated_tokenize_dir(pin, normalized, replace_tokenizer(tokenized, "other"), SPLIT)
    with pytest.raises(ValueError, match="not from this source"):
        validated_tokenize_dir(pin, normalized, tokenized.model_copy(update={"source_keys": {SPLIT: "x"}}), SPLIT)
    with pytest.raises(ValueError, match="not from this source"):
        validated_tokenize_dir(pin, normalized, tokenized, "validation")
    assert not (tmp_path / "out").exists()


def replace_tokenizer(tokenized: TokenizedAttrData, tokenizer: str) -> TokenizedAttrData:
    return tokenized.model_copy(update={"tokenizer": tokenizer})
