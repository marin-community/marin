# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The text route's shared extraction contract: columns, schema, and post-extraction options.

The conversion itself runs on the converter fleet (:mod:`~experiments.datakit.build_pdf_source
.extract_fleet`); this module holds what any docling transport must agree on so the output stays
byte-identical whichever transport produced it:

* the columns read from the fetch artifact (:data:`SOURCE_COLUMNS`),
* the stored record's schema (:data:`OUTPUT_SCHEMA` -- the shared document record, nothing
  route-specific; see :mod:`~experiments.datakit.build_pdf_source.document_record`),
* the boilerplate pass applied before the text is hashed into ``id``
  (:data:`BOILERPLATE_OPTIONS`), and the picture-alpha filter (:data:`PICTURE_ALPHA_RATIO`),
* the normalize reducer that deliberately keeps duplicates (:func:`keep_all`).

An earlier in-task extraction step (docling converter built inside each Zephyr map task, PyMuPDF
tables, 600s document budget) lived here; the fleet superseded it with the measured TableFormer
and 45-minute-budget decisions recorded in :mod:`~experiments.datakit.build_pdf_source.extract_fleet`.
"""

from collections.abc import Iterator

import pyarrow as pa
from marin.datakit.normalize import MainOutput

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS

PICTURE_ALPHA_RATIO = 0.4
# Running headers and footers are stripped before the text is stored, so the id is computed over
# the text a consumer actually reads. See :mod:`experiments.datakit.build_pdf_source.boilerplate`.
BOILERPLATE_OPTIONS = BoilerplateOptions()

SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

# This route adds nothing of its own beyond the fleet's provenance column: everything docling
# reports about a document is already part of the record both routes share.
OUTPUT_SCHEMA = pa.schema(PDF_DOCUMENT_FIELDS)


def keep_all(_key: str, records: Iterator[dict]) -> Iterator[MainOutput]:
    """Emit every record to the main output.

    Extraction deliberately does not deduplicate. The crawl holds roughly 9.8% exact-duplicate
    PDFs and extraction turns those into byte-identical text, so it is tempting to collapse them
    here -- but deduplication and decontamination are #7620, which has to make that decision across
    every source and against the eval sets, not just within this one. Extraction's job is to
    produce documents with a content-derived ``id``; #7620 is what uses it.

    The grouping this reducer runs under is still worth its cost: it sorts records by ``id``, which
    is part of the datakit normalized format and is what makes a later dedup pass a linear scan.
    """
    yield from (MainOutput(data=record) for record in records)
