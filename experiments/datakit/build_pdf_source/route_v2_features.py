# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The router v2 feature contract: what the router reads, where each signal comes from, and its price.

:data:`ROUTER_FEATURES` is the columns the booster in
:mod:`~experiments.datakit.build_pdf_source.classify` was trained on, **in the order it was trained
on them**. Every group but ``inspector_detect`` is free: pdf-inspector's extraction runs on every
document regardless, and these are the signals it reports. Costs are denominated in CPU core-hours
per million crawl pages (see ``pdf-router-v2.md`` on the ``mark/pdf_pipeline`` campaign branch).

Nothing here opens a PDF. A group is a declaration: column names, a source, and a price.
:func:`with_derived` turns the columns :mod:`~experiments.datakit.build_pdf_source.extract_inspector`
stores into the columns the model was trained on, and that is the only computation in the module.
"""

import math
from dataclasses import dataclass

import polars as pl

from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_LEGIBILITY_FLOOR_DPI,
    DEFAULT_MAX_RENDER_DPI,
    DEFAULT_MAX_VISUAL_TOKENS,
)

# ---------------------------------------------------------------------------
# What each pass costs, per million crawl pages
# ---------------------------------------------------------------------------

# pdf-inspector's own extraction, paid on every document.
INSPECTOR_CORE_HOURS = 2.1
# The deleted PyMuPDF router pass, priced so a proposal to reintroduce a paid pass has a bar to clear.
ROUTE_FEATURES_CORE_HOURS = 1.86
INCUMBENT_FEATURES_CORE_HOURS = 1.54
# pdf-inspector's `detect_pdf_bytes`: a second library call reporting pdf_type, confidence and
# per-page OCR reasons.
INSPECTOR_DETECT_CORE_HOURS = 0.12
# Feeding the VLM: rasterise, PNG encode and base64 for every escalated page. An upper bound,
# measured before the feed switched renderer and compression level.
VLM_FEED_CORE_HOURS = 17.8
# Serving those pages, from the full-node brokered measurement in `ocr-budget-sweep.md` on the
# `mark/pdf_pipeline` campaign branch.
VLM_GPU_HOURS = 15.6

# The corpus this is all scaled to.
CRAWL_PAGES = 56_000_000


@dataclass(frozen=True)
class FeatureGroup:
    """One priced block of signals: where they come from, what they cost, and what they are called.

    ``core_hours`` is *incremental*: zero for the groups pdf-inspector's extraction already produces.
    """

    name: str
    source: str
    core_hours: float
    columns: tuple[str, ...]
    rationale: str

    @property
    def free(self) -> bool:
        return self.core_hours == 0.0


PDF_TYPES = ("text_based", "scanned", "image_based", "mixed")
OCR_REASONS = ("no_text", "scanned", "suspected_garbled_text", "vector_text")

# Signals the extraction returns alongside the text.
INSPECTOR_EXTRACT_COLUMNS = (
    "inspector_extract_is_complex_layout",
    "inspector_extract_pages_needing_ocr",
    "inspector_extract_pages_with_tables",
    "inspector_extract_pages_with_columns",
    "inspector_extract_table_page_fraction",
    "inspector_extract_column_page_fraction",
    "inspector_extract_ocr_page_fraction",
    "inspector_extracted_pages",
    "inspector_extract_page_deficit",
    "inspector_markdown_chars",
    "inspector_markdown_chars_per_page",
    "inspector_page_count",
)

# Measured on the markdown pdf-inspector actually produced.
INSPECTOR_OUTPUT_COLUMNS = (
    "inspector_output_replacement_ratio",
    "inspector_output_alpha_ratio",
    "inspector_output_digit_ratio",
    "inspector_output_space_ratio",
    "inspector_output_newline_ratio",
    "inspector_output_single_char_token_ratio",
    "inspector_output_mean_token_length",
    "inspector_output_long_token_ratio",
    "inspector_output_repeat_line_ratio",
    "inspector_output_max_line_repeats",
    "inspector_output_empty_page_fraction",
    "inspector_output_chars_per_source_page",
    "inspector_output_pipe_row_ratio",
    "inspector_output_heading_ratio",
)

INSPECTOR_DETECT_COLUMNS = (
    "inspector_confidence",
    "inspector_has_title",
    "inspector_detect_pages_needing_ocr",
    "inspector_detect_ocr_page_fraction",
    *(f"inspector_type_{name}" for name in PDF_TYPES),
    *(f"inspector_reason_{name}" for name in OCR_REASONS),
)

# The document's shape: page count, byte density, and render geometry.
DOCUMENT_SHAPE_COLUMNS = (
    "num_pages",
    "pdf_bytes",
    "bytes_per_page",
    "mean_render_dpi",
    "legible_page_fraction",
)

GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup(
        "inspector_extract",
        source="pdf-inspector extract_pages_markdown_bytes",
        core_hours=0.0,
        columns=INSPECTOR_EXTRACT_COLUMNS,
        rationale="Layout structure the extraction reports for free: tables, columns, page yield.",
    ),
    FeatureGroup(
        "inspector_output",
        source="pdf-inspector markdown, measured",
        core_hours=0.0,
        columns=INSPECTOR_OUTPUT_COLUMNS,
        rationale="Garbling, repetition and token shape measured on real output rather than inferred.",
    ),
    FeatureGroup(
        "document_shape",
        source="PDF header and render geometry",
        core_hours=0.0,
        columns=DOCUMENT_SHAPE_COLUMNS,
        rationale="Page count, byte density and what the render budget will actually resolve to.",
    ),
    FeatureGroup(
        "inspector_detect",
        source="pdf-inspector detect_pdf_bytes",
        core_hours=INSPECTOR_DETECT_CORE_HOURS,
        columns=INSPECTOR_DETECT_COLUMNS,
        rationale="A second library call at 0.441 ms/page: pdf_type, confidence, per-page OCR reasons.",
    ),
)

GROUPS_BY_NAME = {group.name: group for group in GROUPS}
FREE_GROUPS = tuple(group.name for group in GROUPS if group.free)
PAID_GROUPS = tuple(group.name for group in GROUPS if not group.free)
ALL_GROUPS = tuple(group.name for group in GROUPS)


def columns_for(names: tuple[str, ...]) -> list[str]:
    """Every column the named groups declare, in group order and without duplicates."""
    seen: dict[str, None] = {}
    for name in names:
        for column in GROUPS_BY_NAME[name].columns:
            seen.setdefault(column, None)
    return list(seen)


def cost_of(names: tuple[str, ...]) -> float:
    """Incremental CPU core-hours per million crawl pages of running the named groups."""
    return sum(GROUPS_BY_NAME[name].core_hours for name in names)


# Every column this contract declares, in group order. The order is load-bearing: XGBoost scores a
# bare float matrix by position, so a booster whose `feature_names` no longer match this list would
# score confident nonsense rather than fail. `classify.load_router` checks it.
ROUTER_FEATURES: tuple[str, ...] = tuple(columns_for(ALL_GROUPS))
ROUTER_CORE_HOURS = cost_of(ALL_GROUPS)


# ---------------------------------------------------------------------------
# The legibility floor, which is arithmetic rather than a learned decision
# ---------------------------------------------------------------------------


def dpi_at_budget(dpi_at_default: float, max_visual_tokens: int) -> float:
    """The DPI a page rendered at :data:`DEFAULT_MAX_VISUAL_TOKENS` would get at another budget.

    A page is scaled to fill the visual-token budget, so its DPI goes with the square root of the
    budget until the upscale cap binds.
    """
    scaled = dpi_at_default * math.sqrt(max_visual_tokens / DEFAULT_MAX_VISUAL_TOKENS)
    return min(scaled, DEFAULT_MAX_RENDER_DPI)


def legible_at_budget(max_visual_tokens: int) -> pl.Expr:
    """Whether the VLM could read this document's pages at a given budget.

    Arithmetic, computable before routing, and it decides **how to render** rather than whether to
    route. Read off the document's mean render DPI, which is a good proxy for its pages: a document
    below the floor is almost always below it on every page.
    """
    scale = math.sqrt(max_visual_tokens / DEFAULT_MAX_VISUAL_TOKENS)
    rendered = pl.min_horizontal(pl.col("mean_render_dpi") * scale, pl.lit(DEFAULT_MAX_RENDER_DPI))
    return rendered >= DEFAULT_LEGIBILITY_FLOOR_DPI


# ---------------------------------------------------------------------------
# Assembling the model frame
# ---------------------------------------------------------------------------


def with_derived(frame: pl.DataFrame, reasons: tuple[str, ...] = OCR_REASONS) -> pl.DataFrame:
    """Expand the extraction's categorical, JSON and raw-count columns into model input.

    Both the training frame and the pipeline's scoring batch go through this, so a document is
    scored on arithmetic identical to the arithmetic it was fit on.
    """
    pages = pl.col("inspector_page_count")
    return frame.with_columns(
        **{f"inspector_type_{name}": (pl.col("inspector_pdf_type") == name).cast(pl.Float64) for name in PDF_TYPES},
        **{
            f"inspector_reason_{name}": (
                pl.col("inspector_ocr_reasons").str.json_path_match(f"$.{name}").cast(pl.Float64).fill_null(0.0)
            )
            for name in reasons
        },
        **{
            name: pl.col(name).cast(pl.Float64)
            for name in ("inspector_has_title", "inspector_extract_is_complex_layout")
        },
        inspector_detect_ocr_page_fraction=pl.col("inspector_detect_pages_needing_ocr") / pages,
        inspector_extract_ocr_page_fraction=pl.col("inspector_extract_pages_needing_ocr") / pages,
        inspector_extract_table_page_fraction=pl.col("inspector_extract_pages_with_tables") / pages,
        inspector_extract_column_page_fraction=pl.col("inspector_extract_pages_with_columns") / pages,
        inspector_extract_page_deficit=pages - pl.col("inspector_extracted_pages"),
        inspector_markdown_chars_per_page=pl.col("inspector_markdown_chars") / pages,
        bytes_per_page=pl.col("pdf_bytes") / pl.col("num_pages"),
        legible_page_fraction=(1.0 - pl.col("pages_below_legibility_floor") / pl.col("num_pages").cast(pl.Float64)).clip(
            0.0, 1.0
        ),
    )
