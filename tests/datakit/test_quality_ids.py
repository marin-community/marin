# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract of the quality stage on tokenize-stage ids.

A tiny scorer is trained on the offline GPT-2 fixture tokenizer, a normalize
shard is tokenized by the real tokenize stage, and ``score_normalized`` runs
over both. The assertions are what downstream consumers rely on: output
co-partitioned with normalize and tokenize, the same score the trainer-side
``score_bme`` gives for the document text (train == serve), chunked documents
scored once, a foreign tokenize artifact refused before anything is listed, and
a misaligned shard or one that fails mid-stream leaving no output that a re-run
would mistake for a finished one.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData, generate_id
from marin.datakit.source_key import datakit_source_key
from marin.processing.tokenize.attributes import (
    TokenizeAttributesConfig,
    TokenizedAttrData,
    iter_tokenized_documents,
    tokenize_attributes,
)

from experiments.datakit.cluster.quality.fast_transformer.artifact import MODEL_CALIB
from experiments.datakit.cluster.quality.fast_transformer.calibrate import calibrate_model
from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts
from experiments.datakit.cluster.quality.fast_transformer.score import SAMPLE_TEXT_CHARS, score_normalized
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme
from experiments.datakit.cluster.quality.fast_transformer.train import TrainHParams, train_from_labels

SHARD = "part-00000-of-00001.parquet"
SOURCE = "fixture"
SPLIT = "train"

# Built from the label vocabulary (``w0``..``w22``) so the fixture model's scores vary
# across documents instead of saturating on unknown tokens.
_PROSE = " ".join(f"w{(k * 5) % 23}" for k in range(40)) + " "
# The fixture tokenizer emits roughly one id per character, so lengths in chars are
# lengths in tokens: the first text needs three 512-token windows, the second takes
# the tokenizer's >10k-char long-string path.
TEXTS = [
    _PROSE * 20,
    _PROSE * 70,
    "hello world  ",
    "line\n\n",
    "<|endoftext|>",
    "abc",
    "A plain sentence about nothing in particular.",
]


def _label_rows() -> list[dict]:
    rows = []
    for i in range(25):
        quality = i % 5 + 1
        words = " ".join(f"w{(i * 7 + k) % 23}" for k in range(20 + 3 * quality))
        text = words * 400 if i == 3 else words
        rows.append({"text": text, "quality": quality, "score_normalized": (quality - 1) / 4})
    return rows


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory, gpt2_tokenizer_path) -> str:
    root = tmp_path_factory.mktemp("quality_model")
    labels_path = str(root / "labels.parquet")
    pq.write_table(pa.Table.from_pylist(_label_rows()), labels_path)
    out_dir = str(root / "model")
    train_from_labels(
        labels_path=labels_path,
        out_dir=out_dir,
        tokenizer=gpt2_tokenizer_path,
        hp=TrainHParams(epochs=1, batch_size=8),
    )
    calibrate_model(labels_path, out_dir, f"{out_dir}/{MODEL_CALIB}")
    return out_dir


def _write_normalized(tmp_path: Path, rows: list[dict]) -> NormalizedData:
    main_dir = tmp_path / "normalized" / "outputs" / "main"
    main_dir.mkdir(parents=True)
    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(main_dir / SHARD))
    return NormalizedData(
        main_output_dir=str(main_dir),
        dup_output_dir=str(tmp_path / "normalized" / "outputs" / "dups"),
        counters={},
    )


def _normalized_fixture(tmp_path: Path) -> NormalizedData:
    rows = sorted(({"id": generate_id(t), "text": t} for t in TEXTS), key=lambda r: r["id"])
    return _write_normalized(tmp_path, rows)


def _tokenize(tmp_path: Path, normalized: NormalizedData, tokenizer: str) -> TokenizedAttrData:
    config = TokenizeAttributesConfig(
        train_source=normalized, output_path=str(tmp_path / "tokenize"), tokenizer=tokenizer
    )
    return tokenize_attributes(config)


def _hand_tokenized(tmp_path: Path, tokenizer: str, source_key: str) -> TokenizedAttrData:
    return TokenizedAttrData(
        output_dirs={SPLIT: str(tmp_path / "tokenize" / SPLIT)},
        source_keys={SPLIT: source_key},
        tokenizer=tokenizer,
        tokenizer_backend="hf",
        counters={},
    )


def _score(tmp_path: Path, normalized: NormalizedData, tokenized: TokenizedAttrData, model_dir: str):
    return score_normalized(
        output_path=str(tmp_path / "quality"),
        normalized=normalized,
        tokenized=tokenized,
        source=SOURCE,
        model_dir=model_dir,
        sample_pct=1.0,
        max_workers=1,
    )


def _write_longest_document_as_chunks(table: pa.Table, shard: str, chunk_order: list[int]) -> None:
    """Write ``table`` to ``shard`` with its longest document as three chunk rows, in ``chunk_order``."""
    rows = table.to_pylist()
    pos = max(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    doc = rows[pos]
    third = len(doc["input_ids"]) // 3
    pieces = [doc["input_ids"][:third], doc["input_ids"][third : 2 * third], doc["input_ids"][2 * third :]]
    chunks = [{**doc, "chunk_index": c, "input_ids": pieces[c]} for c in chunk_order]
    pq.write_table(pa.Table.from_pylist(rows[:pos] + chunks + rows[pos + 1 :], schema=table.schema), shard)


def _calibrated_scores(model_dir: str, docs: list[np.ndarray]) -> np.ndarray:
    calib = json.loads(Path(model_dir, MODEL_CALIB).read_text())
    return np.asarray(np.interp(score_bme(load_pooled_scorer(model_dir), docs), calib["xk"], calib["yk"]))


def test_quality_output_is_co_partitioned_with_tokenize_and_normalize(
    tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _tokenize(tmp_path, normalized, gpt2_tokenizer_path)

    result = _score(tmp_path, normalized, tokenized, trained_model_dir)

    assert os.listdir(result.main_output_dir) == [SHARD]
    norm = pq.read_table(f"{normalized.main_output_dir}/{SHARD}")
    norm_ids, norm_texts = norm.column("id").to_pylist(), norm.column("text").to_pylist()
    main = pq.read_table(f"{result.main_output_dir}/{SHARD}")
    assert set(main.column_names) == {"source", "id", "score", "quality_bucket"}
    assert main.column("id").to_pylist() == norm_ids
    scores = np.asarray(main.column("score").to_pylist())
    assert np.all(np.isfinite(scores)) and np.all((scores >= 0.0) & (scores <= 1.0))
    # The model must separate these documents, or the train == serve check below is vacuous.
    assert len(set(np.round(scores, 6))) > 1
    assert set(main.column("quality_bucket").to_pylist()) <= set(range(5))
    assert result.counters["ft_quality/scored"] == len(TEXTS)

    samples = pq.read_table(f"{result.samples_output_dir}/{SHARD}")
    sampled = dict(zip(samples.column("id").to_pylist(), samples.column("text").to_pylist(), strict=True))
    assert sampled == {i: t[:SAMPLE_TEXT_CHARS] for i, t in zip(norm_ids, norm_texts, strict=True)}

    # train == serve: the stage's score for a document is the score the trainer-side
    # path gives for its text, so calibration fitted on label text transfers.
    docs = [np.asarray(ids, dtype=np.int32) for ids in encode_texts(gpt2_tokenizer_path, norm_texts)]
    expected = _calibrated_scores(trained_model_dir, docs)
    np.testing.assert_allclose(scores, expected, atol=1e-6, rtol=0)


def test_label_encoding_matches_tokenize_stage(tmp_path, monkeypatch, gpt2_tokenizer_path):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _tokenize(tmp_path, normalized, gpt2_tokenizer_path)
    [shard] = tokenized.shard_paths(SPLIT)

    norm = pq.read_table(f"{normalized.main_output_dir}/{SHARD}")
    stage_ids = dict(iter_tokenized_documents(shard))
    label_ids = encode_texts(gpt2_tokenizer_path, norm.column("text").to_pylist())

    assert len(stage_ids) == len(TEXTS)
    for doc_id, ids in zip(norm.column("id").to_pylist(), label_ids, strict=True):
        assert np.array_equal(stage_ids[doc_id], ids)


def test_chunked_document_scores_as_one_row(tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _write_normalized(tmp_path, [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}])
    a_text = " ".join(f"w{k % 23}" for k in range(250))
    a_ids, b_ids = (np.asarray(ids, dtype=np.int32) for ids in encode_texts(gpt2_tokenizer_path, [a_text, "w1 w2"]))
    split = len(a_ids) * 3 // 4
    tok_dir = tmp_path / "tokenize" / SPLIT
    tok_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["a", "a", "b"],
                "chunk_index": pa.array([0, 1, 0], type=pa.int32()),
                "input_ids": pa.array([a_ids[:split], a_ids[split:], b_ids], type=pa.list_(pa.int32())),
            }
        ),
        str(tok_dir / SHARD),
    )
    source_key = datakit_source_key(normalized.main_output_dir)
    tokenized = _hand_tokenized(tmp_path, gpt2_tokenizer_path, source_key)

    result = _score(tmp_path, normalized, tokenized, trained_model_dir)

    main = pq.read_table(f"{result.main_output_dir}/{SHARD}")
    assert main.column("id").to_pylist() == ["a", "b"]
    expected = _calibrated_scores(trained_model_dir, [a_ids, b_ids])
    np.testing.assert_allclose(main.column("score").to_pylist(), expected, atol=1e-6, rtol=0)


def test_missing_tokenize_document_leaves_no_output(tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _tokenize(tmp_path, normalized, gpt2_tokenizer_path)
    [shard] = tokenized.shard_paths(SPLIT)
    dropped = pq.read_table(f"{normalized.main_output_dir}/{SHARD}").column("id")[2]
    table = pq.read_table(shard)
    pq.write_table(table.filter(pc.not_equal(table.column("id"), dropped)), shard)

    with pytest.raises(RuntimeError, match=r"row 2.*co-partitioning broken"):
        _score(tmp_path, normalized, tokenized, trained_model_dir)

    assert list((tmp_path / "quality").rglob("*.parquet")) == []


def test_out_of_order_chunk_rows_leave_no_output(tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _tokenize(tmp_path, normalized, gpt2_tokenizer_path)
    [shard] = tokenized.shard_paths(SPLIT)
    table = pq.read_table(shard)
    _write_longest_document_as_chunks(table, shard, [0, 2, 1])

    with pytest.raises(RuntimeError, match=r"row \d+ is chunk 2 of .*, but chunk 1 of .* must come next"):
        _score(tmp_path, normalized, tokenized, trained_model_dir)
    assert list((tmp_path / "quality").rglob("*.parquet")) == []

    _write_longest_document_as_chunks(table, shard, [0, 1, 2])
    result = _score(tmp_path, normalized, tokenized, trained_model_dir)

    main = pq.read_table(f"{result.main_output_dir}/{SHARD}")
    norm_ids = pq.read_table(f"{normalized.main_output_dir}/{SHARD}").column("id").to_pylist()
    assert main.column("id").to_pylist() == norm_ids


def test_shard_failing_mid_stream_leaves_no_output_and_is_rescored(
    tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _tokenize(tmp_path, normalized, gpt2_tokenizer_path)
    # A file where the samples dir belongs makes the samples writer fail after the
    # shard's main rows were already handed to the main writer.
    samples_dir = tmp_path / "quality" / "outputs" / "samples"
    samples_dir.parent.mkdir(parents=True)
    samples_dir.write_text("")

    with pytest.raises(RuntimeError):
        _score(tmp_path, normalized, tokenized, trained_model_dir)
    assert list((tmp_path / "quality").rglob("*.parquet")) == []

    samples_dir.unlink()
    result = _score(tmp_path, normalized, tokenized, trained_model_dir)

    main = pq.read_table(f"{result.main_output_dir}/{SHARD}")
    norm_ids = pq.read_table(f"{normalized.main_output_dir}/{SHARD}").column("id").to_pylist()
    assert main.column("id").to_pylist() == norm_ids
    assert result.counters["ft_quality/scored"] == len(TEXTS)


def test_scorer_refuses_tokenizer_mismatch(tmp_path, monkeypatch, trained_model_dir):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    source_key = datakit_source_key(normalized.main_output_dir)
    tokenized = _hand_tokenized(tmp_path, "other/tokenizer", source_key)

    with pytest.raises(ValueError, match="trained for tokenizer"):
        _score(tmp_path, normalized, tokenized, trained_model_dir)

    assert not (tmp_path / "quality").exists()


def test_tokenize_from_other_source_is_refused(tmp_path, monkeypatch, gpt2_tokenizer_path, trained_model_dir):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    normalized = _normalized_fixture(tmp_path)
    tokenized = _hand_tokenized(tmp_path, gpt2_tokenizer_path, "elsewhere/outputs/main")

    with pytest.raises(ValueError, match="not from this source"):
        _score(tmp_path, normalized, tokenized, trained_model_dir)

    assert not (tmp_path / "quality").exists()
