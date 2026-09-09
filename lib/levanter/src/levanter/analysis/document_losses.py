# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream raw documents and describe document-level evaluation output."""

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator

from rigging.filesystem.storage_path import StoragePath

from levanter.data.sharded_datasource import ShardedDataSource, UrlDataSource, WrappedHFDataSource


@dataclass(frozen=True)
class DocumentSourceConfig:
    input_path: str | None = None
    hf_dataset: str | None = None
    hf_name: str | None = None
    hf_split: str = "train"
    hf_revision: str | None = None
    text_field: str = "text"
    doc_id_field: str = "id"
    corpus_id: str | None = None
    corpus_id_field: str = "corpus_id"
    max_documents: int | None = None

    def __post_init__(self):
        if (self.input_path is None) == (self.hf_dataset is None):
            raise ValueError("Specify exactly one of input_path and hf_dataset")
        if self.input_path == "" or self.hf_dataset == "":
            raise ValueError("Document source must not be empty")
        if self.max_documents is not None and self.max_documents < 0:
            raise ValueError("max_documents must be nonnegative")
        if not self.text_field or not self.doc_id_field or not self.corpus_id_field:
            raise ValueError("Document field names must not be empty")
        if self.corpus_id == "":
            raise ValueError("corpus_id must not be empty")


@dataclass(frozen=True)
class Document:
    doc_id: str
    corpus_id: str
    text: str


@dataclass(frozen=True)
class DocumentLoss:
    doc_id: str
    corpus_id: str
    loss: float | None
    bits_per_byte: float | None


def _document_urls(input_path: str) -> list[str]:
    path = StoragePath(input_path)
    pattern = path / "**" if path.isdir() else path
    urls = []
    for match in pattern.expand_glob():
        suffixes = PurePosixPath(str(match)).suffixes
        if ".jsonl" in suffixes[-2:] or ".parquet" in suffixes[-2:]:
            urls.append(str(match))
    if not urls:
        raise ValueError(f"No JSONL or Parquet shards found at {input_path}")
    return sorted(set(urls))


def _document_id(value: object, field: str, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError(f"{location}: {field} must be a nonempty string or integer")
    return str(value)


def iter_documents(config: DocumentSourceConfig) -> Iterator[Document]:
    """Yield whole documents in sorted shard order, preserving source row order.

    Missing IDs use the source identity, shard name, and zero-based row index.
    Corpus IDs prefer the configured override, then the row field, then the source
    identity. Present ID fields must contain nonempty strings or integers.
    """
    if config.max_documents == 0:
        return
    source: ShardedDataSource[dict]
    if config.input_path is not None:
        source = UrlDataSource(_document_urls(config.input_path))
        identity = str(StoragePath(config.input_path))
    else:
        assert config.hf_dataset is not None
        source = WrappedHFDataSource(
            config.hf_dataset,
            split=config.hf_split,
            name=config.hf_name,
            revision=config.hf_revision,
            streaming=True,
        )
        identity = f"hf://{config.hf_dataset}/{config.hf_name or 'default'}/{config.hf_split}"
        if config.hf_revision is not None:
            identity += f"@{config.hf_revision}"

    count = 0
    for shard in sorted(source.shard_names):
        for row_index, row in enumerate(source.open_shard_at_row(shard, 0)):
            location = f"{identity}#{shard}:{row_index}"
            text = row.get(config.text_field)
            if not isinstance(text, str):
                raise ValueError(f"{location}: {config.text_field} must be a string")
            doc_id = (
                _document_id(row[config.doc_id_field], config.doc_id_field, location)
                if config.doc_id_field in row
                else location
            )
            corpus_id = config.corpus_id
            if corpus_id is None:
                corpus_id = (
                    _document_id(row[config.corpus_id_field], config.corpus_id_field, location)
                    if config.corpus_id_field in row
                    else identity
                )
            yield Document(doc_id=doc_id, corpus_id=corpus_id, text=text)
            count += 1
            if config.max_documents is not None and count >= config.max_documents:
                return
