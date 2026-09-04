# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What both extraction routes agree on: the columns they read and the post-processing they apply.

A PDF leaves this pipeline through pdf-inspector on CPU
(:mod:`~experiments.datakit.build_pdf_source.extract_inspector`) or through a vision model on GPU
(:mod:`~experiments.datakit.build_pdf_source.extract_ocr`). The two share an input -- the fetch
step's Parquet shards -- and a post-processing pass, and this module holds both so the parts of a
stored record that ought to be route-independent cannot drift apart:

* the columns read from the fetch artifact (:data:`SOURCE_COLUMNS`),
* the boilerplate pass applied before the text is hashed into ``id``
  (:data:`BOILERPLATE_OPTIONS`),
* the render budget the routing table's decisions are expressed against (:data:`RENDER_OPTIONS`).

Neither route sorts or deduplicates. Each writes one shard per fetched shard, named after it, and
the global sort by content hash and the exact dedup are the normalize step over the union of both
routes (:mod:`~experiments.datakit.build_pdf_source.dedup`), where they are paid for once and
where byte-identical text recovered on either side of the router collapses to one document.

The record's *schema* lives in :mod:`~experiments.datakit.build_pdf_source.document_record`; each
route appends its own diagnostic columns after those fields and
:mod:`~experiments.datakit.build_pdf_source.combine_routes` carries both sets through.
"""

from experiments.datakit.build_pdf_source.boilerplate import BoilerplateOptions
from experiments.datakit.build_pdf_source.ocr_extract.render import RenderOptions

# Running headers and footers are stripped before the text is stored, so the id is computed over
# the text a consumer actually reads. See :mod:`experiments.datakit.build_pdf_source.boilerplate`.
BOILERPLATE_OPTIONS = BoilerplateOptions()
# The default render budget: the geometry pass measures against it and the OCR route renders at it.
RENDER_OPTIONS = RenderOptions()

SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]
