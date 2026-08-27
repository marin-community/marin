# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry each cluster's span geometry from the assembled page onto the assembled document.

:mod:`.assemble` measures a median glyph advance and a last-line box per layout cluster, and the
postprocessors in :mod:`.postprocess` need both -- they judge whether two blocks belong to one
paragraph in characters, not points. Between the two sits docling's reading-order stage, which
turns page elements into document items and builds a fresh ``ProvenanceItem`` from nothing but the
cluster's bounding box, dropping everything else the element carried.

Every construction site is one of three small methods, so they are overridden here to copy the two
values across after the fact rather than reimplementing the stage. Each provenance entry
corresponds to exactly one source element, including the merged case, where the entry appended for
the merged element takes that element's own measurements.
"""

import logging

from docling.models.stages.reading_order.readingorder_model import ReadingOrderModel

from experiments.datakit.build_pdf_source.docling_extract.fields import patch_docling_models

logger = logging.getLogger(__name__)


def _copy_span_geometry(provenance, element) -> None:
    """Copy an assembled element's span measurements onto the provenance entry made from it."""
    provenance.media_char_width = getattr(element, "media_char_width", None)
    provenance.last_line_bbox = getattr(element, "last_line_bbox", None)


class SpanAwareReadingOrderModel(ReadingOrderModel):
    """:class:`ReadingOrderModel` that preserves the measurements :mod:`.assemble` recorded."""

    def __init__(self, options):
        patch_docling_models()
        super().__init__(options)

    def _handle_text_element(self, element, out_doc, current_list, page_height):
        new_item, current_list = super()._handle_text_element(element, out_doc, current_list, page_height)
        _copy_span_geometry(new_item.prov[-1], element)
        return new_item, current_list

    def _add_caption_or_footnote(self, elem, out_doc, parent, page_height):
        new_item = super()._add_caption_or_footnote(elem, out_doc, parent, page_height)
        _copy_span_geometry(new_item.prov[-1], elem)
        return new_item

    def _merge_elements(self, element, merged_elem, new_item, page_height):
        super()._merge_elements(element, merged_elem, new_item, page_height)
        _copy_span_geometry(new_item.prov[-1], merged_elem)
