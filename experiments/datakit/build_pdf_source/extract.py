# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What both extraction routes agree on: the columns they read, the post-processing, the reducer.

A PDF leaves this pipeline through pdf-inspector on CPU
(:mod:`~experiments.datakit.build_pdf_source.extract_inspector`) or through a vision model on GPU
(:mod:`~experiments.datakit.build_pdf_source.extract_ocr`). The two share an input -- the fetch
step's Parquet shards -- and a post-processing pass, and this module holds both so the parts of a
stored record that ought to be route-independent cannot drift apart:

* the columns read from the fetch artifact (:data:`SOURCE_COLUMNS`),
* the boilerplate pass applied before the text is hashed into ``id``
  (:data:`BOILERPLATE_OPTIONS`),
* the normalize reducer that deliberately keeps duplicates (:func:`keep_all`).

The record's *schema* lives in :mod:`~experiments.datakit.build_pdf_source.document_record`; each
route appends its own diagnostic columns after those fields and
:mod:`~experiments.datakit.build_pdf_source.combine_routes` carries both sets through.

Two earlier occupants of this module are gone with the route they served: a Docling converter built
inside each Zephyr map task, and the fleet transport that superseded it. Docling cost 278 CPU
core-hours per million pages against pdf-inspector's 2.1 -- 132x -- for corpus-wide quality parity,
and this cluster is CPU-constrained (``pdf-inspector-evaluation.md``, ``pdf-router-v2.md``).
"""

from collections.abc import Iterator

from marin.datakit.normalize import MainOutput

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions

# Running headers and footers are stripped before the text is stored, so the id is computed over
# the text a consumer actually reads. See :mod:`experiments.datakit.build_pdf_source.boilerplate`.
BOILERPLATE_OPTIONS = BoilerplateOptions()

SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]


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
