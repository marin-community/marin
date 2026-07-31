# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The four model fields this extractor adds to docling, and the reason they are added this way.

FinePDFs carries the same four fields by vendoring forks of ``docling-core`` and ``docling`` pinned
at 2.38.x. We track upstream instead, so the fields are attached to the released models at import
time. Every one is optional with a default, so a document that never passes through our code
validates and serialises exactly as upstream does.

What they carry:

``TextCell.info``
    The per-span measurements PyMuPDF reports and docling's ``TextCell`` has no place for --
    ``line_bbox``, ``line_angle``, and the font ``flags`` word. :mod:`.assemble` needs the line
    geometry to decide whether two spans are on one line, and bit 0 of ``flags`` to know a span is
    superscript.

``TextElement.media_char_width`` / ``TextElement.last_line_bbox``
    Written by :mod:`.assemble` from the cells of a layout cluster: the median glyph advance over
    the cluster, and the bounding box of its final line.

``ProvenanceItem.media_char_width`` / ``ProvenanceItem.last_line_bbox``
    The same two values, carried onto the assembled document by :mod:`.reading_order` so the
    postprocessors in :mod:`.postprocess` can still see them. Distances there are measured in
    characters rather than points, because a 5-point gap means something different in a footnote
    than in a title.

Upstream FinePDFs sets the two ``TextElement`` values but never declares the fields, and
``BasePageElement`` is a plain :class:`pydantic.BaseModel`, whose default ``extra`` policy is
``ignore``. Both values are dropped at construction, nothing ever writes them onto a
``ProvenanceItem``, and so ``ParagraphMerger.should_merge_blocks`` and ``CellsMerger.is_on_same_line``
return ``False`` on their first guard for every input. The paragraph and span merging in the
released FinePDFs pipeline does not run. :func:`patch_docling_models` is what makes it run here;
``tests/datakit/test_docling_fields.py`` pins the behaviour so a docling upgrade that renames or
reuses one of these fields fails loudly instead of silently reverting to that state.
"""

import logging
from functools import cache

from docling.datamodel.base_models import TextElement
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.page import TextCell
from pydantic import BaseModel
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

# (model, field name, annotation, default factory or None for a plain ``None`` default).
_ADDED_FIELDS: tuple[tuple[type[BaseModel], str, object, object], ...] = (
    (TextCell, "info", dict, dict),
    (TextElement, "media_char_width", float | None, None),
    (TextElement, "last_line_bbox", BoundingBox | None, None),
    (ProvenanceItem, "media_char_width", float | None, None),
    (ProvenanceItem, "last_line_bbox", BoundingBox | None, None),
)


@cache
def patch_docling_models() -> None:
    """Attach this extractor's fields to the released docling models. Idempotent.

    Raises:
        RuntimeError: if docling already declares one of these names, which would mean upstream has
            given it a meaning of its own and the two uses have to be reconciled by hand.
    """
    for model, name, annotation, default_factory in _ADDED_FIELDS:
        existing = model.model_fields.get(name)
        if existing is not None:
            # Same annotation means this is our own field from an earlier call, or an upstream
            # addition that happens to mean the same thing; either way there is nothing to do. A
            # different annotation is a genuine collision and has to be resolved by a person.
            if existing.annotation == annotation:
                continue
            raise RuntimeError(
                f"{model.__name__}.{name} is already declared by docling as "
                f"{existing.annotation}, which collides with the {annotation} this extractor "
                "attaches; reconcile the two in "
                "experiments.build_pdf_source.docling_extract.fields before extracting."
            )
        model.model_fields[name] = (
            FieldInfo(annotation=annotation, default_factory=default_factory)
            if default_factory is not None
            else FieldInfo(annotation=annotation, default=None)
        )
        model.model_rebuild(force=True)

    logger.debug("Attached %d fields to docling models", len(_ADDED_FIELDS))
