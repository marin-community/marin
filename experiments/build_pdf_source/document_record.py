# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The record both PDF extraction routes produce.

A PDF leaves this pipeline through one of two extractors -- docling for documents with usable
embedded text, a vision model for the rest -- and a consumer should not have to know which one
produced a document. These fields are the part of the output that is the same either way, so the
two routes concatenate.

``id`` and ``text`` come first because that is the contract every datakit consumer reads; the rest
is the provenance a PDF carries, and the extraction outcome. Each route appends its own columns
after these.
"""

import pyarrow as pa

# Both routes write ``extraction_status``, but the vocabularies are their own: docling reports its
# ``ConversionStatus``, OCR reports :class:`~experiments.build_pdf_source.extract_ocr.OcrStatus`.
# The column is shared so a consumer can filter on it uniformly, not so the values can be compared.
PDF_DOCUMENT_FIELDS: tuple[pa.Field, ...] = (
    pa.field("id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    # Unique where content_digest is not: the crawl holds ~9.8% exact-duplicate PDFs.
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("warc_filename", pa.string(), nullable=False),
    pa.field("warc_record_offset", pa.int64(), nullable=False),
    pa.field("content_digest", pa.string(), nullable=False),
    pa.field("url", pa.string(), nullable=False),
    pa.field("num_pages", pa.int32(), nullable=False),
    # Cumulative character counts, so a span of ``text`` can be traced to the page it came from.
    # Recomputed after boilerplate removal, so they index the text actually stored here.
    pa.field("page_offsets", pa.list_(pa.int64()), nullable=False),
    pa.field("extraction_status", pa.string(), nullable=False),
    pa.field("extraction_error", pa.string(), nullable=True),
    pa.field("boilerplate_lines_removed", pa.int32(), nullable=False),
)


def source_id(warc_filename: str, warc_record_offset: int) -> str:
    """The document's identity in the crawl, which is its WARC record.

    Distinct from ``id``, which is derived from the extracted text and is therefore shared by
    duplicate PDFs, and from ``content_digest``, which the crawl assigns to identical bytes.
    """
    return f"{warc_filename}:{warc_record_offset}"
