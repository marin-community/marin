# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model fields this extractor attaches to docling.

These are the upgrade guard. The extractor tracks upstream docling instead of vendoring a fork of
it, and pays for that by adding four fields to released pydantic models at import time. If a future
docling declares one of those names itself, the two meanings have to be reconciled by hand -- so
that case must fail loudly rather than silently win or lose. If upstream instead moves or renames
the models, these tests stop importing, which is also the signal.
"""

import pytest

pytest.importorskip("docling")

from docling.datamodel.base_models import TextElement
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import ProvenanceItem
from docling_core.types.doc.page import TextCell

from experiments.build_pdf_source.docling_extract.fields import (
    _ADDED_FIELDS,
    patch_docling_models,
)


@pytest.fixture(autouse=True)
def _patched():
    patch_docling_models()


def _bbox() -> BoundingBox:
    return BoundingBox(l=0.0, t=10.0, r=100.0, b=0.0, coord_origin=CoordOrigin.BOTTOMLEFT)


def test_provenance_carries_the_span_geometry_the_postprocessors_measure():
    provenance = ProvenanceItem(page_no=1, charspan=(0, 5), bbox=_bbox(), media_char_width=4.25)

    assert provenance.media_char_width == 4.25
    assert provenance.last_line_bbox is None


def test_span_geometry_survives_a_json_round_trip():
    """The assembled document is serialised as part of the step's output, so these must persist."""
    provenance = ProvenanceItem(page_no=1, charspan=(0, 5), bbox=_bbox(), media_char_width=4.25, last_line_bbox=_bbox())

    restored = ProvenanceItem.model_validate_json(provenance.model_dump_json())

    assert restored.media_char_width == 4.25
    assert restored.last_line_bbox is not None
    assert restored.last_line_bbox.t == 10.0


def test_span_geometry_can_be_assigned_after_construction():
    """The reading-order stage builds provenance first and stamps the geometry on afterwards."""
    provenance = ProvenanceItem(page_no=1, charspan=(0, 5), bbox=_bbox())

    provenance.media_char_width = 3.0
    provenance.last_line_bbox = _bbox()

    assert provenance.media_char_width == 3.0


def test_text_cells_carry_a_span_record_and_default_to_an_empty_one():
    """Cells from another backend -- OCR, say -- have no span record, and must still construct."""
    from docling_core.types.doc.page import BoundingRectangle  # noqa: PLC0415

    rect = BoundingRectangle.from_bounding_box(_bbox())
    with_info = TextCell(index=0, text="x", orig="x", from_ocr=False, rect=rect, info={"flags": 1})
    without_info = TextCell(index=1, text="y", orig="y", from_ocr=True, rect=rect)

    assert with_info.info["flags"] == 1
    assert without_info.info == {}


def test_text_elements_carry_the_measurements_the_assembler_records():
    """Upstream FinePDFs sets these without declaring them, so pydantic drops both on construction.

    That is what makes their paragraph and span mergers no-ops; this asserts ours are not.
    """
    from docling.datamodel.base_models import Cluster  # noqa: PLC0415
    from docling_core.types.doc import DocItemLabel  # noqa: PLC0415

    cluster = Cluster(id=0, label=DocItemLabel.TEXT, bbox=_bbox(), cells=[])
    element = TextElement(
        label=DocItemLabel.TEXT,
        id=0,
        text="text",
        page_no=1,
        cluster=cluster,
        media_char_width=5.5,
        last_line_bbox=_bbox(),
    )

    assert element.media_char_width == 5.5
    assert element.last_line_bbox is not None


def test_patching_twice_is_harmless():
    """Every module that needs the fields calls this, so it must be safe to call repeatedly."""
    patch_docling_models()
    patch_docling_models()

    assert "media_char_width" in ProvenanceItem.model_fields


def test_a_name_docling_gives_a_different_meaning_is_refused():
    """A future docling declaring one of these names for something else must stop the run."""
    from pydantic.fields import FieldInfo  # noqa: PLC0415

    model, name, _, _ = _ADDED_FIELDS[0]
    original = model.model_fields[name]
    model.model_fields[name] = FieldInfo(annotation=str, default="")
    patch_docling_models.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=f"{model.__name__}.{name} is already declared"):
            patch_docling_models()
    finally:
        model.model_fields[name] = original
        model.model_rebuild(force=True)
        patch_docling_models.cache_clear()
        patch_docling_models()


def test_a_name_docling_gives_the_same_meaning_is_accepted():
    """If upstream adds an identical field, adopting it is correct -- no reconciliation needed."""
    patch_docling_models.cache_clear()

    patch_docling_models()

    assert ProvenanceItem.model_fields["media_char_width"].annotation is not None
