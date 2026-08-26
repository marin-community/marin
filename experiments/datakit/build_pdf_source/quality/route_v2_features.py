# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The router v2 feature contract: what it may read, where each signal comes from, and what it costs.

Router v1 read ~70 decode-free PyMuPDF signals
(:mod:`~experiments.datakit.build_pdf_source.quality.route_features`) and nothing else, because at the
time the cheap route was Docling and Docling's output was not available before the routing decision.
That constraint is gone. pdf-inspector *is* the cheap route now, it runs on every document whether
or not the document is escalated, and its extraction reports garbling, table and column structure,
page yield and text volume **measured on real output** rather than inferred from the page's fonts
and geometry. Those signals cost the router nothing: the pass that produces them is the pass whose
result the router is deciding whether to replace.

So the feature set splits into groups that are priced differently, and the split is the point of the
module. :data:`GROUPS` names each group's source, its incremental cost per million crawl pages, and
its columns; :mod:`~experiments.datakit.build_pdf_source.quality.analyze_route_v2` trains one arm per
subset of them. The question that answers is whether the 3.4 core-h/M pages the PyMuPDF pass costs
buys anything over signals that are already free -- which is the same question as whether router v2
needs a router pass at all.

**Cost is denominated in CPU core-hours.** This cluster is CPU-constrained and GPU-rich, so a
frontier drawn against GPU time optimizes the resource that is spare. Per million crawl pages:
pdf-inspector :data:`INSPECTOR_CORE_HOURS`, the PyMuPDF router pass :data:`ROUTE_FEATURES_CORE_HOURS`,
and the VLM's feed path -- PyMuPDF render, PNG encode, base64 -- :data:`VLM_FEED_CORE_HOURS` on top
of :data:`VLM_GPU_HOURS` of GPU. The GPU number is carried so a spend can be quoted in both, not
because it is the axis being optimized.

**Nothing here imports PyMuPDF, and nothing here computes a feature.** A group is a declaration:
column names, a source, and a price. The signals themselves come from study tables that some other
pass already wrote, so replacing the PyMuPDF pass with a cheaper producer is an edit to
:data:`GROUPS` and to the table that feeds it, not to the router. That matters right now: whether
``pdf_oxide`` can supply the ``page_signals`` group more cheaply, or remove it, is under separate
evaluation, and the router must not have to be rewritten either way.
"""

import math
from dataclasses import dataclass

import polars as pl

from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_LEGIBILITY_FLOOR_DPI,
    DEFAULT_MAX_RENDER_DPI,
    DEFAULT_MAX_VISUAL_TOKENS,
)
from experiments.datakit.build_pdf_source.quality.route_feature_names import FEATURE_NAMES

# ---------------------------------------------------------------------------
# What each pass costs, per million crawl pages
# ---------------------------------------------------------------------------

# pdf-inspector's own extraction: 4.66 ms/page measured in the Stage 1 study. Paid on every
# document, because it is both the cheap route and the source of the free feature groups.
INSPECTOR_CORE_HOURS = 2.1
# The PyMuPDF router pass, split into the two feature sets it actually runs.
#
# `route_features`' own 36 page signals cost 1.86 core-h/M on x86 and 1.84 on aarch64, measured over
# 1,000 documents in `pdf-oxide-evaluation.md`. The 3.4 core-h/M the pipeline has been budgeting is
# that plus the 124-feature FinePDFs incumbent extraction, whose only surviving output is `ocr_prob`.
# Splitting them is the point: Stage 2 found `ocr_prob` ranked 7th by gain when the model could use
# everything and moved the frontier by 0.0019, so the incumbent half may be dead weight, and dropping
# it would halve the paid pass rather than trimming it.
ROUTE_FEATURES_CORE_HOURS = 1.86
INCUMBENT_FEATURES_CORE_HOURS = 1.54
# pdf-inspector's `detect_pdf_bytes`, at 0.441 ms/page. A second library call, so it is not free
# even though it is nearly so; it reports pdf_type, confidence and per-page OCR reasons that the
# extraction does not.
INSPECTOR_DETECT_CORE_HOURS = 0.12
# Feeding the VLM: PyMuPDF render, PNG encode and base64 for every escalated page.
VLM_FEED_CORE_HOURS = 17.8
# Serving those pages, from the full-node brokered measurement in `ocr-budget-sweep.md`
# (~71 pages/s per 4 GB200s at the 2048-token budget).
VLM_GPU_HOURS = 15.6

# The corpus this is all scaled to.
CRAWL_PAGES = 56_000_000


@dataclass(frozen=True)
class FeatureGroup:
    """One priced block of signals: where they come from, what they cost, and what they are called.

    ``core_hours`` is *incremental* -- what a router adds to the pipeline by insisting on this
    group. For the groups pdf-inspector's extraction already produces that is zero: the pipeline
    runs the extraction to get the text, and reading the signals it returned costs nothing further.
    Pricing them at their share of the extraction's 2.1 core-h would charge the router for a pass it
    did not cause.
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

# Signals the extraction returns alongside the text. Free: the pipeline runs this call regardless.
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

# Measured on the text pdf-inspector actually produced, which is the signal router v1 structurally
# could not have: v1 inferred garbling from font tables before any extraction existed to inspect.
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

# The document's shape, known from the PDF header before any extraction: page count and page size.
# Free in the same sense the extraction's signals are -- pdf-inspector already reports the page
# count, and the render geometry is what the feed path computes anyway.
DOCUMENT_SHAPE_COLUMNS = (
    "num_pages",
    "pdf_bytes",
    "bytes_per_page",
    "mean_render_dpi",
    "legible_page_fraction",
)

# The FinePDFs incumbent's contribution, as the study table carries it: the router's shipped rule is
# its probability with a garbled-text override, and those two columns are what a document gets for
# the price of its 124-feature PyMuPDF extraction. The raw features are not stored -- only what the
# incumbent concluded from them -- so this group prices the pass and tests the conclusion.
INCUMBENT_COLUMNS = ("ocr_prob", "garbled_text_ratio")

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
    FeatureGroup(
        "page_signals",
        source="PyMuPDF, 8 sampled pages (route_features)",
        core_hours=ROUTE_FEATURES_CORE_HOURS,
        columns=FEATURE_NAMES,
        rationale="Router v1's ~70 decode-free encoding, layer, math, structure, order and script signals.",
    ),
    FeatureGroup(
        "incumbent",
        source="PyMuPDF, 8 sampled pages (FinePDFs ocr_features) plus its booster",
        core_hours=INCUMBENT_FEATURES_CORE_HOURS,
        columns=INCUMBENT_COLUMNS,
        rationale="The shipped FinePDFs rule's own probability and its garbled-text override.",
    ),
)

GROUPS_BY_NAME = {group.name: group for group in GROUPS}
FREE_GROUPS = tuple(group.name for group in GROUPS if group.free)
PAID_GROUPS = tuple(group.name for group in GROUPS if not group.free)


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


# ---------------------------------------------------------------------------
# The legibility floor, which is arithmetic rather than a learned decision
# ---------------------------------------------------------------------------


def dpi_at_budget(dpi_at_default: float, max_visual_tokens: int) -> float:
    """The DPI a page rendered at :data:`DEFAULT_MAX_VISUAL_TOKENS` would get at another budget.

    A page is scaled to *fill* the visual-token budget, so its pixel count is proportional to the
    budget and its DPI to the square root of it -- until the 300-DPI upscale cap binds, past which
    more budget buys nothing because there is no more glyph detail in the source to recover.
    """
    scaled = dpi_at_default * math.sqrt(max_visual_tokens / DEFAULT_MAX_VISUAL_TOKENS)
    return min(scaled, DEFAULT_MAX_RENDER_DPI)


def legible_at_budget(max_visual_tokens: int) -> pl.Expr:
    """Whether the VLM could read this document's pages at a given budget.

    The gate this expresses is exactly computable before routing and is therefore not the model's
    job: a page rendered below :data:`DEFAULT_LEGIBILITY_FLOOR_DPI` is one the VLM cannot read at
    all, so escalating it spends a render and a GPU pass on a transcription of a blur.

    Read off the document's mean render DPI, which on this corpus is a good proxy for its pages:
    2,199 of the 2,368 documents with any page below the floor have *every* page below it, because
    the cause is a large paper size shared by the whole document rather than one odd page.
    """
    scale = math.sqrt(max_visual_tokens / DEFAULT_MAX_VISUAL_TOKENS)
    rendered = pl.min_horizontal(pl.col("mean_render_dpi") * scale, pl.lit(DEFAULT_MAX_RENDER_DPI))
    return rendered >= DEFAULT_LEGIBILITY_FLOOR_DPI


# ---------------------------------------------------------------------------
# Assembling the model frame
# ---------------------------------------------------------------------------


def with_derived(frame: pl.DataFrame, reasons: tuple[str, ...] = OCR_REASONS) -> pl.DataFrame:
    """Expand the study tables' categorical, JSON and raw-count columns into model input.

    Everything here is a ratio or an indicator rather than a raw count wherever a count would make
    the feature a proxy for document length. ``inspector_markdown_chars`` is kept as well as its
    per-page form because expected output length is what predicts VLM truncation, and truncation is
    one of the failure modes the score has to learn rather than be gated on.
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
