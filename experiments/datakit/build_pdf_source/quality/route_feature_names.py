# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The routing booster's feature contract, as pure data.

Separated from :mod:`~experiments.datakit.build_pdf_source.quality.route_features` because that module
imports PyMuPDF, which lives in marin-core's ``datakit`` extra. The pipeline driver builds its steps
without extras, so :mod:`experiments.datakit.build_pdf_source.classify` needs the feature count and order at
step-definition time on a process that cannot import PyMuPDF at all. This mirrors
:mod:`experiments.datakit.build_pdf_source.ocr_feature_names`, which exists for the same reason.

``route_features`` asserts its dataclass against :data:`PAGE_SIGNAL_NAMES` at import, so a signal
added there without being added here fails loudly rather than silently reordering the model input.
"""

# One entry per field of :class:`~experiments.datakit.build_pdf_source.quality.route_features.PageSignals`,
# in declaration order.
PAGE_SIGNAL_NAMES: tuple[str, ...] = (
    "char_count",
    "fonts_total",
    "fonts_not_embedded",
    "fonts_without_tounicode",
    "fonts_unmappable",
    "fonts_type3",
    "fonts_glyphless",
    "replacement_ratio",
    "pua_ratio",
    "control_ratio",
    "ligature_ratio",
    "alphanum_ratio",
    "space_ratio",
    "newline_ratio",
    "single_char_token_ratio",
    "mean_token_length",
    "invisible_char_ratio",
    "invisible_over_image_ratio",
    "text_over_image_ratio",
    "overlapping_line_ratio",
    "out_of_page_line_ratio",
    "duplicate_span_ratio",
    "rotated_char_ratio",
    "math_font_ratio",
    "math_unicode_ratio",
    "ruling_line_count",
    "rule_grid_cells",
    "ruled_area_ratio",
    "left_edge_concentration",
    "text_block_count",
    "column_count",
    "stream_order_inversion_ratio",
    "interleaved_column_ratio",
    "cjk_ratio",
    "rtl_ratio",
    "latin_ratio",
)

# Two document-level scalars, then every page signal aggregated over the sampled pages by mean and
# by max. A document routes on its worst page as much as on its average one: a 40-page report with
# one page of equations is not one-fortieth of a problem, because the extraction that page needs is
# not the one the other 39 need.
FEATURE_NAMES: tuple[str, ...] = (
    "pages_sampled",
    "page_count",
    *(f"{aggregate}_{name}" for name in PAGE_SIGNAL_NAMES for aggregate in ("mean", "max")),
)
