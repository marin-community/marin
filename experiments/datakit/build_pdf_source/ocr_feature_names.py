# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The FinePDFs OCR router's input contract: what its 124 features are, and in what order.

Split out from :mod:`experiments.datakit.build_pdf_source.ocr_features` because the contract is pure data
while extracting it needs PyMuPDF, which only ships in marin-core's ``datakit`` extra. Step
definitions run in the entrypoint job, which is synced without extras, so anything they touch has to
import without PyMuPDF present.
"""

# The booster has one slot per page for eight pages, so a document is described by a sample of at
# most eight pages; shorter documents repeat sampled pages to fill the remaining slots.
FEATURE_PAGES = 8

# Document-level features, in booster order. The upstream flattening also listed
# ``num_unique_image_xrefs``, ``num_junk_image_xrefs`` and ``class``, but the extractor never
# populated them (those assignments are commented out upstream), so they never reached the model and
# are absent from the booster's own feature names.
DOC_FEATURE_NAMES = (
    "num_pages_successfully_sampled",
    "garbled_text_ratio",
    "is_form",
    "creator_or_producer_is_known_scanner",
)

# Page-level features, in booster order, paired with the ``PageFeatures`` field each reads. The
# booster names are plural because upstream they named per-document lists; here each is one page's
# scalar. ``avg_text_box_lengths`` is a misnomer inherited from the original -- it is the mean *area*
# of a page's text boxes, not a length.
PAGE_FEATURE_FIELDS = (
    ("page_level_unique_font_counts", "unique_font_count"),
    ("page_level_char_counts", "char_count"),
    ("page_level_text_box_counts", "text_box_count"),
    ("page_level_avg_text_box_lengths", "mean_text_box_area"),
    ("page_level_text_area_ratios", "text_area_ratio"),
    ("page_level_hidden_char_counts", "hidden_char_count"),
    ("page_level_hidden_text_box_counts", "hidden_text_box_count"),
    ("page_level_hidden_avg_text_box_lengths", "mean_hidden_text_box_area"),
    ("page_level_hidden_text_area_ratios", "hidden_text_area_ratio"),
    ("page_level_image_counts", "image_count"),
    ("page_level_non_junk_image_counts", "non_junk_image_count"),
    ("page_level_bitmap_proportions", "bitmap_proportion"),
    ("page_level_max_merged_strip_areas", "max_merged_strip_area"),
    ("page_level_drawing_strokes_count", "drawing_stroke_count"),
    ("page_level_vector_graphics_obj_count", "vector_graphics_object_count"),
)

FEATURE_NAMES = DOC_FEATURE_NAMES + tuple(
    f"{name}_page{slot}" for name, _ in PAGE_FEATURE_FIELDS for slot in range(1, FEATURE_PAGES + 1)
)
