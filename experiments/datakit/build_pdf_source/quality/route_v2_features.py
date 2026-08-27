# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The router v2 feature contract: what the router reads, where each signal comes from, and its price.

Router v1 read ~70 decode-free PyMuPDF signals and nothing else, because at the time the cheap route
was Docling and Docling's output was not available before the routing decision. That constraint is
gone. pdf-inspector *is* the cheap route now
(:mod:`~experiments.datakit.build_pdf_source.extract_inspector`), it runs on every document whether
or not the document is escalated, and its extraction reports garbling, table and column structure,
page yield and text volume **measured on real output** rather than inferred from the page's fonts
and geometry. Those signals cost the router nothing: the pass that produces them is the pass whose
result the router is deciding whether to replace.

The groups declared here are what survived that comparison, and the survivors are the free ones.
Paired within each of five domain-disjoint splits, adding the PyMuPDF pass to the free groups made
page-weighted loss *worse* on all five (+0.0127 mean), and the 124-feature FinePDFs incumbent behind
``ocr_prob`` landed inside the 0.0608 split-draw noise floor for another 1.54 core-h/M
(``experiments/datakit/build_pdf_source/pdf-router-v2.md``, "Free features against paid ones"). Both
halves of that pass are deleted, so this module no longer declares them as groups. Their measured
prices survive as :data:`ROUTE_FEATURES_CORE_HOURS` and :data:`INCUMBENT_FEATURES_CORE_HOURS`,
because 3.4 core-h/M over 56M crawl pages is the 190 crawl core-hours the deletion saved and is the
whole reason it was worth making.

What is left is exactly the shipped arm. :data:`ROUTER_FEATURES` is the 43 columns the booster in
:mod:`~experiments.datakit.build_pdf_source.classify` was trained on, **in the order it was trained
on them**, and :data:`GROUPS` prices the lot at 0.12 CPU core-hours per million crawl pages.

**Cost is denominated in CPU core-hours.** This cluster is CPU-constrained and GPU-rich, so a
frontier drawn against GPU time optimizes the resource that is spare. Per million crawl pages:
pdf-inspector :data:`INSPECTOR_CORE_HOURS`, and the VLM's feed path -- render, PNG encode, base64 --
:data:`VLM_FEED_CORE_HOURS` on top of :data:`VLM_GPU_HOURS` of GPU. The GPU number is carried so a
spend can be quoted in both, not because it is the axis being optimized.

**Nothing here imports PyMuPDF, and nothing here computes a feature from a page.** A group is a
declaration: column names, a source, and a price. :func:`with_derived` turns the columns
:mod:`~experiments.datakit.build_pdf_source.extract_inspector` stores into the columns the model was
trained on, and that is the only computation in the module -- so replacing the producer of a group
is an edit to :data:`GROUPS` and to the step that fills it, never to the router.
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

# pdf-inspector's own extraction: 4.66 ms/page measured in the Stage 1 study. Paid on every
# document, because it is both the cheap route and the source of the free feature groups.
INSPECTOR_CORE_HOURS = 2.1
# The PyMuPDF router pass, deleted. Both halves are priced here rather than forgotten, because the
# saving is the largest single number in the router report and a future proposal to reintroduce a
# paid pass has to clear it: 3.40 core-h/M over CRAWL_PAGES is 190 crawl core-hours, bought for a
# page-weighted loss that measured *worse* on all five domain splits.
#
# `route_features`' own 36 page signals cost 1.86 core-h/M on x86 and 1.84 on aarch64, measured over
# 1,000 documents in `pdf-oxide-evaluation.md`; the 124-feature FinePDFs incumbent extraction behind
# `ocr_prob` is the other 1.54.
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


# The shipped arm -- `free + detect` -- is every group that is left, so the router's feature vector
# is simply every column this contract declares, in group order. The order is load-bearing: XGBoost
# scores a bare float matrix by position, so a booster whose `feature_names` no longer match this
# list would score confident nonsense rather than fail. `classify.load_router` checks it.
ROUTER_FEATURES: tuple[str, ...] = tuple(columns_for(ALL_GROUPS))
ROUTER_CORE_HOURS = cost_of(ALL_GROUPS)


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

    This is arithmetic, exactly computable before routing, and it decides **how to render** rather
    than whether to route. Router v1 skipped below-floor documents on the premise that escalating
    one buys a transcription of a blur; the preference label refutes that premise on this corpus,
    where judges escalated 79.0% of them (n=558). They are large-format scans -- posters, maps and
    plans, a median implied 787 square inches against US Letter's 93.5 -- for which pdf-inspector
    produces nothing usable, and the VLM reading a 50-DPI render still recovers more of the page
    than that. So the score decides them like anything else and this expression only chooses the
    budget they are rendered at: 16,384 visual tokens rescues 1,890 of 2,234 for +0.29% GPU
    crawl-wide, against +24.6% to raise the budget for every page in the corpus
    (``pdf-router-v2.md``, "The legibility floor: render it bigger").

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
    """Expand the extraction's categorical, JSON and raw-count columns into model input.

    This is the one place the stored columns and the trained columns are reconciled, and both the
    training frame and the pipeline's scoring batch go through it, so a document is scored on
    arithmetic identical to the arithmetic it was fit on.

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
