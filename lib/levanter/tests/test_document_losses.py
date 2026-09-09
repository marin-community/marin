# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from levanter.analysis.document_losses import Document, DocumentSourceConfig, iter_documents


@pytest.mark.parametrize("source_kind", ["prefix", "glob", "braces"])
def test_document_sources_stream_sorted_jsonl_and_parquet(tmp_path, source_kind):
    (tmp_path / "b.jsonl").write_text(json.dumps({"id": 42, "corpus_id": "web", "text": "second"}) + "\n")
    pq.write_table(pa.Table.from_pylist([{"id": "p", "corpus_id": "books", "text": "first"}]), tmp_path / "a.parquet")
    (tmp_path / "README.md").write_text("Not a document shard")
    source = {
        "prefix": str(tmp_path),
        "glob": str(tmp_path / "*"),
        "braces": str(tmp_path / "{b.jsonl,a.parquet}"),
    }[source_kind]
    assert list(iter_documents(DocumentSourceConfig(input_path=source))) == [
        Document(doc_id="p", corpus_id="books", text="first"),
        Document(doc_id="42", corpus_id="web", text="second"),
    ]


def test_document_source_missing_ids_and_limit_preserve_row_identity(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps({"body": text}) for text in ["one", "", "three"]) + "\n")
    config = DocumentSourceConfig(input_path=str(path), text_field="body")
    documents = list(iter_documents(config))
    assert documents == [
        Document(doc_id=f"{path}#documents_jsonl:{index}", corpus_id=str(path), text=text)
        for index, text in enumerate(["one", "", "three"])
    ]
    assert (
        list(iter_documents(DocumentSourceConfig(input_path=str(path), text_field="body", max_documents=2)))
        == documents[:2]
    )


def test_document_source_limit_does_not_read_next_row_and_override_wins(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps({"key": "original", "corpus": "row-corpus", "text": "kept"}) + "\ninvalid json\n")
    config = DocumentSourceConfig(
        input_path=str(path), doc_id_field="key", corpus_id_field="corpus", corpus_id="override", max_documents=1
    )
    assert list(iter_documents(config)) == [Document(doc_id="original", corpus_id="override", text="kept")]
    assert list(iter_documents(DocumentSourceConfig(input_path=str(path), max_documents=0))) == []


def test_document_source_hf_preserves_configured_fields(monkeypatch):
    def load_dataset(path, *, split, name, revision, streaming):
        # The HF API is the external I/O boundary. Select different fixture content
        # for the requested source so omitted source parameters change the result.
        if (path, split, name, revision, streaming) == ("org/corpus", "validation", "subset", "pinned", True):
            return datasets.Dataset.from_list([{"key": "hf-id", "body": "HF text", "corpus": "hf-corpus"}])
        return datasets.Dataset.from_list([{"key": "wrong", "body": "wrong source", "corpus": "wrong"}])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    config = DocumentSourceConfig(
        hf_dataset="org/corpus",
        hf_name="subset",
        hf_split="validation",
        hf_revision="pinned",
        text_field="body",
        doc_id_field="key",
        corpus_id_field="corpus",
    )
    assert list(iter_documents(config)) == [Document(doc_id="hf-id", corpus_id="hf-corpus", text="HF text")]


@pytest.mark.parametrize(
    "row, error",
    [
        ({"text": None}, "text must be a string"),
        ({"body": "missing text"}, "text must be a string"),
        ({"text": "valid", "id": [1]}, "id must be a nonempty string or integer"),
        ({"text": "valid", "corpus_id": None}, "corpus_id must be a nonempty string or integer"),
    ],
)
def test_document_source_rejects_invalid_rows_with_location(tmp_path, row, error):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=error) as exc:
        list(iter_documents(DocumentSourceConfig(input_path=str(path))))
    assert f"{path}#bad_jsonl:0" in str(exc.value)


def test_document_source_unmatched_glob_fails(tmp_path):
    with pytest.raises(ValueError, match="No JSONL or Parquet shards"):
        list(iter_documents(DocumentSourceConfig(input_path=str(tmp_path / "*.jsonl"))))
