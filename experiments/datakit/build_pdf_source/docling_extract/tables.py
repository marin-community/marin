# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Table cell recovery, from PyMuPDF's ruling-line detector rather than from TableFormer.

Docling detects *where* the tables are with its layout model and *what is in them* with TableFormer,
a second neural model. FinePDFs keeps the first and replaces the second: for each table the layout
model found, PyMuPDF re-reads that rectangle and recovers the grid from the drawn ruling lines.
That drops a model from the per-page cost, which is why they did it, and it is exact on tables that
are actually ruled and blind to tables that are not -- the opposite trade to TableFormer, which
infers a grid from text positions and is never exact but never blind.

Which is better for this corpus is an open question, so both are reachable. :class:`TableReader` is
the seam: a reader fills a detected table's cells after assembly, and selecting TableFormer instead
means passing no reader and letting docling's own table stage run. See
:mod:`.converter` for the switch and :data:`~.converter.TableBackend` for the two settings.

``strategy="lines"`` is deliberate. PyMuPDF also offers ``"text"``, which infers columns from
whitespace and would make this a worse-calibrated TableFormer rather than a different instrument.
"""

import logging
from typing import Protocol

import pymupdf
from docling.datamodel.base_models import Page, Table
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import TableCell, TableData

logger = logging.getLogger(__name__)


class TableReader(Protocol):
    """Fills in the cells of a table the layout model has already located."""

    def fill(self, *, table: Table, cluster, page: Page) -> None:
        """Populate ``table``'s cells in place, leaving it untouched if nothing could be read."""


def extract_table_data(page: pymupdf.Page, bbox: BoundingBox) -> TableData | None:
    """Recover a ruled table's grid from ``bbox``, or ``None`` if there is not exactly one there.

    Zero tables means the region has no ruling lines. More than one means the layout model's
    rectangle spans several tables, and picking any of them would silently drop the others -- so
    both cases decline, and the table keeps whatever cells it already had.
    """
    found = page.find_tables(clip=pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b), strategy="lines")
    if len(found.tables) != 1:
        logger.debug("Found %d ruled tables in %s; leaving the table as detected", len(found.tables), bbox)
        return None

    rows = found.tables[0].extract()
    if not rows or not rows[0]:
        return None

    num_columns = len(rows[0])
    cells = [
        TableCell(
            # PyMuPDF returns None for a cell that is present but empty, and may return text with
            # lone surrogates from a broken font; both have to become storable strings here.
            text=str(text or "").encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
            row_span=1,
            col_span=1,
            start_row_offset_idx=row_index,
            end_row_offset_idx=row_index + 1,
            start_col_offset_idx=column_index,
            end_col_offset_idx=column_index + 1,
            # PyMuPDF's grid has no header concept, so the first row is assumed to be one. That is
            # right far more often than not for ruled tables and costs one row when it is wrong.
            column_header=row_index == 0,
            row_header=False,
        )
        for row_index, row in enumerate(rows)
        for column_index, text in enumerate(row)
    ]
    return TableData(num_rows=len(rows), num_cols=num_columns, table_cells=cells)


class PyMuPdfTableReader:
    """Reads table grids from ruling lines, using the same PyMuPDF page the text came from."""

    def fill(self, *, table: Table, cluster, page: Page) -> None:
        backend = page._backend
        if backend is None or not backend.is_valid():
            return
        bbox = cluster.bbox.to_top_left_origin(backend.get_size().height)
        try:
            data = extract_table_data(backend.get_pymupdf_page(), bbox)
        except Exception:
            # A damaged content stream can fail inside the ruling-line detector. A table we cannot
            # read is data, not a pipeline failure: the page keeps its other content.
            logger.warning("Could not read table %s on page %d", bbox, page.page_no, exc_info=True)
            return
        if data is None:
            return
        table.num_rows = data.num_rows
        table.num_cols = data.num_cols
        table.table_cells = data.table_cells
